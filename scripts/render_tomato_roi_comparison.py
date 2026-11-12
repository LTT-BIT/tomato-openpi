"""Export actual native/old/new streams side by side at native 30Hz."""
import argparse
import json
from pathlib import Path

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.audit_tomato_stable_roi import entries, find_episodes


def main():
    import av
    import cv2
    import numpy as np
    from PIL import Image,ImageDraw
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--clips',required=True,help='episode:start:end,..., half-open frame ranges')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    before,after = find_episodes(args.before),find_episodes(args.after)
    clips = [tuple(map(int,c.split(':'))) for c in args.clips.split(',')]
    if not clips or any(len(c)!=3 or not 0<=c[0]<100 or not 0<=c[1]<c[2]<=selection['episodes'][c[0]]['length'] for c in clips):
        parser.error('Invalid clip identity/range')
    out = derived_output_path(selection,args.out)
    sources = []
    tiles = []
    count = 0
    with new_artifact_directory(out) as staging:
        with av.open(str(staging/'comparison30hz.mp4'),'w') as output:
            stream = output.add_stream('libx264',rate=30)
            stream.width,stream.height,stream.pix_fmt = 768,256,'yuv420p'
            stream.options = {'crf':'18','preset':'fast'}
            stream.thread_count = 2
            for ep,start,end in clips:
                row = selection['episodes'][ep]
                rel = f"videos/chunk-{row['source_episode_index']//1000:03d}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
                source = Path(row['dataset'])/rel
                if digest(source)!=row['source_sha256'][rel]:
                    raise ValueError('Original source changed')
                new_index = entries(after[ep])
                old_index = entries(before[ep])
                sources.append(dict(episode_index=ep,start=start,end=end,
                                    source_sha256=digest(source),old_video_sha256=digest(before[ep]/'interaction224.mp4'),
                                    new_video_sha256=digest(after[ep]/'interaction224.mp4')))
                with av.open(str(source)) as native,av.open(str(before[ep]/'interaction224.mp4')) as old,av.open(str(after[ep]/'interaction224.mp4')) as new:
                    decoded = zip(native.decode(video=0),old.decode(video=0),new.decode(video=0),strict=True)
                    for i,(a,b,c) in enumerate(decoded):
                        if i<start:
                            continue
                        if i>=end:
                            break
                        tile = Image.new('RGB',(768,256),'#151515')
                        image = a.to_ndarray(format='rgb24')
                        draw_source = Image.fromarray(image)
                        draw = ImageDraw.Draw(draw_source)
                        draw.rectangle(new_index[i]['display_crop_box'],outline='#00dfd0',width=2)
                        tile.paste(draw_source.resize((320,200)),(0,32))
                        tile.paste(Image.fromarray(b.to_ndarray(format='rgb24')),(320,32))
                        tile.paste(Image.fromarray(c.to_ndarray(format='rgb24')),(544,32))
                        draw = ImageDraw.Draw(tile)
                        draw.text((4,4),f'ep{ep} frame{i} native',fill='white')
                        draw.text((324,4),f'OLD valid={old_index[i]["valid"]}',fill='white')
                        draw.text((548,4),f'NEW valid={new_index[i]["valid"]}',fill='#00dfd0' if new_index[i]['valid'] else '#ffbd55')
                        draw.text((548,18),str(new_index[i]['roles'])+' '+str(new_index[i]['reason'] or ''),fill='white')
                        if i==(start+end)//2:
                            tiles.append(tile.copy())
                        for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(tile),format='rgb24')):
                            output.mux(packet)
                        count += 1
                if digest(source)!=row['source_sha256'][rel]:
                    raise ValueError('Original source changed during export')
            for packet in stream.encode():
                output.mux(packet)
        with av.open(str(staging/'comparison30hz.mp4')) as check:
            if sum(1 for _ in check.decode(video=0))!=count:
                raise ValueError('Comparison video frame count mismatch')
        sheet = Image.new('RGB',(768,256*len(tiles)))
        for i,tile in enumerate(tiles):
            sheet.paste(tile,(0,256*i))
        sheet.save(staging/'comparison_samples.jpg',quality=95)
        result = dict(selection_sha256=digest(args.selection),clips=sources,frames=count,fps=30,
                      video_sha256=digest(staging/'comparison30hz.mp4'),
                      clip_boundary_cuts=True,per_episode_outputs_have_no_editorial_cuts=True,
                      source_writes=False,training_started=False)
        (staging/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        print('Exported',count,'frames at30Hz:',out)


if __name__=='__main__':
    main()
