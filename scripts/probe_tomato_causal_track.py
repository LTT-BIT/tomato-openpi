"""Benchmark manual-seeded causal tracks on one recording; offline boxes are evaluation only."""
import argparse
import json
import time
from pathlib import Path

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.export_tomato_manual_crops import distribution
from scripts.prepare_tomato100_roi import worklist
from scripts.tomato_causal_roi import CausalBoxTrack
from scripts.tomato_interaction_preview import native_crop, square_box


def run(selection_path,annotations_path,out):
    import av
    import cv2
    import numpy as np
    selection=json.loads(selection_path.read_text());spec=json.loads(annotations_path.read_text())
    row=next(r for r in worklist(selection,spec,preview_sha256=digest(annotations_path))['episodes'] if r['roi_clips'])
    out=derived_output_path(selection,out)
    relative=f"videos/chunk-{row['source_episode_index']//1000:03d}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
    source=Path(row['dataset'])/relative;expected=row['source_sha256'][relative]
    if digest(source)!=expected:raise ValueError('Source changed')
    seed=spec['clips'][0]['keyframes'][0]
    if seed['frame']!=0:raise ValueError('Frame zero seed required')
    cv2.setNumThreads(1)
    tracks=None;rows=[];track_times=[];crop_times=[];valid_times=[]
    with new_artifact_directory(out) as staging:
        with av.open(str(source)) as container:
            for i,frame in enumerate(container.decode(video=0)):
                rgb=frame.to_ndarray(format='rgb24')
                start=time.perf_counter();gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                if i==0:
                    tracks={role:CausalBoxTrack(gray,box,0) for role,box in seed['boxes'].items()}
                    results={role:dict(valid=t.lost_reason is None,box=t.box.tolist() if t.lost_reason is None else None,
                                      reason=t.lost_reason) for role,t in tracks.items()}
                else:results={role:t.update(gray,i) for role,t in tracks.items()}
                duration=time.perf_counter()-start;track_times.append(duration)
                valid=all(r['valid'] for r in results.values())
                crop_box=None
                if valid:
                    valid_times.append(duration)
                    boxes=np.array([r['box'] for r in results.values()])
                    tight=[*boxes[:,:2].min(0),*boxes[:,2:].max(0)]
                    start=time.perf_counter();crop_box=square_box(tight,rgb.shape[1],rgb.shape[0])
                    cv2.resize(native_crop(rgb,crop_box),(224,224),interpolation=cv2.INTER_AREA)
                    crop_times.append(time.perf_counter()-start)
                rows.append(dict(frame=i,available_at=i/row['fps'],valid=valid,crop_box=crop_box,tracks=results))
        if len(rows)!=row['length'] or digest(source)!=expected:raise ValueError('Source/frame mismatch')
        result=dict(episode_index=row['episode_index'],frames=len(rows),seed_frame=0,seed_roles=list(tracks),
            seed_annotation_sha256=digest(annotations_path),source_sha256=expected,source_hash_unchanged=True,
            initial_seed='manual frame0 boxes only',future_annotations_read_by_tracker=False,
            automatic_target_acquisition=False,automatic_role_switching=False,automatic_reacquisition=False,
            invalid_is_sticky=True,valid_frames=sum(r['valid'] for r in rows),
            first_invalid_frame=next((r['frame'] for r in rows if not r['valid']),None),
            track_latency_active=distribution(valid_times) if valid_times else None,
            track_latency_all_including_invalid_fast_path=distribution(track_times),
            crop_latency_active=distribution(crop_times) if crop_times else None,
            validity='Heuristic feature/motion checks only; not semantic identity verification.',
            full_roi_coverage=False,training_ready=False,realtime_acceptance=False,source_writes=False)
        (staging/'tracks.json').write_text(json.dumps(rows,indent=2)+'\n')
        (staging/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    print(json.dumps(run(a.selection,a.annotations,a.out),indent=2))


if __name__=='__main__':main()
