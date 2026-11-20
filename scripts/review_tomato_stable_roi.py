"""Sample decoded crops across every recording and left-transport runs for visual QA."""
import argparse
import json
from pathlib import Path

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.audit_tomato_stable_roi import entries, find_episodes


def sample_frames(video,frames):
    import av
    images = {}
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for target in sorted(set(frames)):
            container.seek(int(target/30/stream.time_base),stream=stream,backward=True,any_frame=False)
            for frame in container.decode(video=0):
                index = round(frame.pts*frame.time_base*30)
                if index==target:
                    images[target] = frame.to_image()
                    break
                if index>target:
                    raise ValueError('Seek skipped requested source frame')
    if set(images)!=set(frames):
        raise ValueError('Incomplete visual sample')
    return images


def left_run_midpoint(rows,fallback):
    runs = []
    start = None
    for i,row in enumerate(rows+[dict(valid=False,roles=[])]):
        active = row['valid'] and row['roles']==[2]
        if active and start is None:
            start = i
        if not active and start is not None:
            runs.append((start,i))
            start = None
    if not runs:
        return fallback,'second_harvest_fallback'
    a,b = max(runs,key=lambda interval:interval[1]-interval[0])
    return (a+b-1)//2,'longest_valid_left_transport_candidate'


def main():
    from PIL import Image,ImageDraw
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    folders = find_episodes(args.candidates)
    if set(folders)!=set(range(100)):
        raise ValueError('All100 completed recordings required for final review')
    tiles = []
    samples = []
    for ep,folder in sorted(folders.items()):
        row = selection['episodes'][ep]
        index = entries(folder)
        chosen = [((row['training_segments'][i]['start_frame']+row['training_segments'][i]['end_frame']-1)//2,f'phase{i+1}') for i in [0,1,6]]
        segment = row['training_segments'][5]
        chosen.append(left_run_midpoint(index,(segment['start_frame']+segment['end_frame']-1)//2))
        receipt = json.loads((folder/'receipt.json').read_text())
        if digest(folder/'interaction224.mp4')!=receipt['video_sha256']:
            raise ValueError('Review video changed')
        frames = sample_frames(folder/'interaction224.mp4',[f for f,_ in chosen])
        for f,reason in chosen:
            tile = Image.new('RGB',(224,254),'#151515')
            tile.paste(frames[f],(0,30))
            draw = ImageDraw.Draw(tile)
            draw.text((3,2),f'ep{ep} f{f} valid={index[f]["valid"]}',fill='white')
            draw.text((3,16),f'{index[f]["roles"]} {reason[:20]}',fill='#00dfd0')
            tiles.append(tile)
            samples.append(dict(episode_index=ep,source_frame=f,reason=reason,valid=index[f]['valid'],roles=index[f]['roles']))
    out = derived_output_path(selection,args.out)
    with new_artifact_directory(out) as staging:
        for start in range(0,len(tiles),40):
            page = Image.new('RGB',(896,2540))
            for j,tile in enumerate(tiles[start:start+40]):
                page.paste(tile,((j%4)*224,(j//4)*254))
            page.save(staging/f'gallery_{start//40:02d}.jpg',quality=95)
        (staging/'samples.json').write_text(json.dumps(dict(selection_sha256=digest(args.selection),samples=samples,
            sample_count=len(samples),phase_labels_used='QA sampling only',semantic_accuracy_not_measured=True,
            source_writes=False,training_started=False),indent=2)+'\n')
    print('Generated',len(samples),'real decoded visual samples')


if __name__=='__main__':
    main()
