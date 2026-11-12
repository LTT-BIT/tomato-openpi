"""Compare causal crop streams, validate bounded motion and decode real videos."""
import argparse
import collections
import gzip
import json
import time
from pathlib import Path

import numpy as np

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.tomato_interaction_preview import native_crop


def find_episodes(root):
    result = {}
    for path in sorted(root.glob('**/episode_*/receipt.json')):
        row = json.loads(path.read_text())
        ep = row['episode_index']
        if ep in result:
            raise ValueError('Duplicate episode')
        result[ep] = path.parent
    return result


def entries(folder):
    with gzip.open(folder/'index.jsonl.gz','rt') as stream:
        return [json.loads(line) for line in stream]


def statistics(rows, display=False):
    validity = roles = burst = 0
    jumps = []
    scale = []
    previous = None
    previous_key = None
    run = 0
    for row in rows:
        key = (row['valid'],tuple(row['roles']) if row['valid'] else ())
        if key==previous_key:
            run += 1
        else:
            if previous_key is not None and run<=3:
                burst += 1
            run = 1
            previous_key = key
        if previous is not None:
            validity += row['valid']!=previous['valid']
            roles += row['roles']!=previous['roles']
            a = previous.get('display_crop_box') if display else previous['crop_box']
            b = row.get('display_crop_box') if display else row['crop_box']
            if a is not None and b is not None:
                a,b = np.asarray(a,float),np.asarray(b,float)
                old_side,new_side = a[2]-a[0],b[2]-b[0]
                jumps.append(float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/old_side))
                scale.append(float(max(new_side/old_side,old_side/new_side)))
        previous = row
    return dict(frames=len(rows),valid_frames=sum(r['valid'] for r in rows),
                validity_transitions=validity,role_changes=roles,short_validity_or_role_runs=burst,
                center_jumps_over25pct=sum(v>.25 for v in jumps),side_changes_over25pct=sum(v>1.25 for v in scale),
                max_normalized_center_step=max(jumps,default=0),max_side_ratio=max(scale,default=1),
                reasons=dict(collections.Counter(r['reason'] for r in rows if not r['valid'])))


def check_index(rows,length):
    if len(rows)!=length:
        raise ValueError('Missing frames')
    for frame,row in enumerate(rows):
        if row['source_frame']!=frame or abs(row['timestamp']-frame/30)>1e-6 or row['available_at']>row['timestamp']+1e-9:
            raise ValueError('Frame/time causality failure')
        b = np.asarray(row['display_crop_box'],float)
        w,h = row['source_size']
        if b.shape!=(4,) or not np.isfinite(b).all() or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h) or b[2]-b[0]!=b[3]-b[1]:
            raise ValueError('Invalid display square')
        if row['valid']:
            if ('left_grasp_state_confirmed' in row and row['roles']==[2]
                and (row['left_grasp_state_confirmed'] is not True or row['carry_evidence'] is not True)):
                raise ValueError('Left transport lacks current grasp and visual evidence')
            t = np.asarray(row['interaction_box'],float)
            if row['crop_box']!=row['display_crop_box'] or t.shape!=(4,) or not np.isfinite(t).all() or not np.all(t[:2]>=b[:2]) or not np.all(t[2:]<=b[2:]):
                raise ValueError('Valid crop misses observed interaction')
        elif row['crop_box'] is not None or row['interaction_box'] is not None:
            raise ValueError('Invalid ROI must have null training coordinates')
    metrics = statistics(rows,True)
    if metrics['max_normalized_center_step']>.04 or metrics['max_side_ratio']>1.025:
        raise ValueError('Stream motion exceeds stabilization limits')


def verify_pixels(row,folder,index):
    import av
    import cv2
    prefix = f"chunk-{row['source_episode_index']//1000:03d}"
    relative = f"videos/{prefix}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
    source = Path(row['dataset'])/relative
    if digest(source)!=row['source_sha256'][relative]:
        raise ValueError('Original video changed')
    # Include invalid frames too: their display must be CURRENT pixels, not black/stale.
    sampled = set(np.linspace(0,len(index)-1,24,dtype=int).tolist())
    sampled.update(i for i,r in enumerate(index) if not r['valid'] and i%97==0)
    expected = {}
    with av.open(str(source)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            if i in sampled:
                image = frame.to_ndarray(format='rgb24')
                expected[i] = cv2.resize(native_crop(image,index[i]['display_crop_box']),(224,224),interpolation=cv2.INTER_AREA)
    mae = []
    black = 0
    count = 0
    with av.open(str(folder/'interaction224.mp4')) as container:
        if float(container.streams.video[0].average_rate)!=30:
            raise ValueError('Wrong output FPS')
        for count,frame in enumerate(container.decode(video=0),1):
            i = count-1
            if abs(float(frame.pts*frame.time_base)-i/30)>1e-5:
                raise ValueError('Dropped/duplicated video timestamp')
            image = frame.to_ndarray(format='rgb24')
            black += bool(image.max()<5)
            if i in expected:
                mae.append(float(np.abs(image.astype(float)-expected[i]).mean()))
    if count!=len(index) or black or max(mae,default=100)>8:
        raise ValueError('Video frame count/black flash/current pixel check failed')
    if digest(source)!=row['source_sha256'][relative]:
        raise ValueError('Original video changed during check')
    return dict(decoded_frames=count,black_frames=black,sampled_current_pixel_checks=len(mae),
                maximum_encoded_pixel_mae=max(mae),original_source_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--wait-all',action='store_true',help='Validate each atomic episode as workers finish, publishing only when all100 pass')
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    baseline = find_episodes(args.baseline)
    results = []
    done = set()
    started = time.monotonic()
    while True:
        candidates = find_episodes(args.candidates)
        pending = sorted(set(candidates)-done)
        if not pending:
            if not args.wait_all or len(done)==100:
                break
            if time.monotonic()-started>3600:
                raise TimeoutError('Timed out waiting for all100 completed episodes')
            time.sleep(1)
            continue
        ep = pending[0]
        folder = candidates[ep]
        row = selection['episodes'][ep]
        current,old = entries(folder),entries(baseline[ep])
        receipt = json.loads((folder/'receipt.json').read_text())
        if digest(folder/'index.jsonl.gz')!=receipt['index_sha256'] or digest(folder/'interaction224.mp4')!=receipt['video_sha256']:
            raise ValueError('Generated artifact changed')
        check_index(current,row['length'])
        results.append(dict(episode_index=ep,before=statistics(old),after=statistics(current,True),
                            video=verify_pixels(row,folder,current)))
        done.add(ep)
        print(f'Checked ep{ep}: before valid {sum(r["valid"] for r in old)}, after {sum(r["valid"] for r in current)}',flush=True)
    result = dict(selection_sha256=digest(args.selection),episodes=len(results),
                  frames=sum(r['after']['frames'] for r in results),episodes_checked=results,
                  motion_and_current_pixel_checks_passed=True,semantic_quality_accepted=False,
                  training_ready=False,training_started=False,source_writes=False,
                  scope='Temporal and current-image integrity, not independent semantic accuracy.')
    out = derived_output_path(selection,args.out)
    with new_artifact_directory(out) as staging:
        (staging/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
