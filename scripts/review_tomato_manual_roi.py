"""Render a complete manually annotated recording, keeping raw sources read only.

This exports review artifacts, not a full100 training bundle or online locator.
"""
import argparse
import json
from pathlib import Path

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.prepare_tomato100_roi import worklist
from scripts.tomato_interaction_preview import annotation_available_frame, interaction_at, native_crop, square_box


def run(selection_path, annotations_path, out):
    import av
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    selection = json.loads(selection_path.read_text())
    annotations = json.loads(annotations_path.read_text())
    listing = worklist(selection, annotations, preview_sha256=digest(annotations_path))
    row = next(r for r in listing['episodes'] if r['roi_clips'])
    if row['remaining_frames']:
        raise ValueError('Complete recording annotation required for this review export')
    out = derived_output_path(selection, out)
    relative = (f"videos/chunk-{row['source_episode_index']//1000:03d}/"
                f"observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4")
    source = Path(row['dataset']) / relative
    expected = row['source_sha256'][relative]
    if digest(source) != expected:
        raise ValueError('Source video changed')
    wanted = {i: c for c in annotations['clips'] for i in range(c['start'], c['end'])}
    qa = {k['frame'] for c in annotations['clips'] for k in c['keyframes']}
    qa.update((a['frame']+b['frame'])//2 for c in annotations['clips']
              for a,b in zip(c['keyframes'],c['keyframes'][1:]))
    rows = []
    tiles = []
    page = 0
    with new_artifact_directory(out) as staging:
        with av.open(str(source)) as src, av.open(str(staging/'review.mp4'),'w') as video, \
                av.open(str(staging/'interaction224.mp4'),'w') as clean:
            if float(src.streams.video[0].average_rate) != row['fps']:
                raise ValueError('FPS mismatch')
            stream = video.add_stream('libx264', rate=row['fps'])
            stream.width, stream.height, stream.pix_fmt = 864, 432, 'yuv420p'
            stream.options = {'crf':'20','preset':'fast'}
            local = clean.add_stream('libx264', rate=row['fps'])
            local.width, local.height, local.pix_fmt = 224,224,'yuv420p'
            local.options = {'crf':'16','preset':'fast'}
            for i, frame in enumerate(src.decode(video=0)):
                if i not in wanted:
                    continue
                image = frame.to_ndarray(format='rgb24')
                h,w = image.shape[:2]
                if [w,h] != annotations['source_size']:
                    raise ValueError('Native source size changed')
                clip = wanted[i]
                tight = interaction_at(clip,i,w,h)
                box = square_box(tight,w,h)
                crop = cv2.resize(native_crop(image,box),(224,224),interpolation=cv2.INTER_AREA)
                overlay = Image.fromarray(image)
                draw = ImageDraw.Draw(overlay)
                draw.rectangle(tuple(tight),outline='#ffb000',width=2)
                draw.rectangle(box,outline='#00e0d0',width=2)
                panel = Image.new('RGB',(864,432),'#151515')
                panel.paste(overlay,(0,32));panel.paste(Image.fromarray(crop),(640,100))
                ImageDraw.Draw(panel).text((5,8),
                    f"{clip['name']} | f{i} | MANUAL OFFLINE | QA candidate",fill='white')
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.array(panel),format='rgb24')):
                    video.mux(packet)
                for packet in local.encode(av.VideoFrame.from_ndarray(crop,format='rgb24')):
                    clean.mux(packet)
                future_key = annotation_available_frame(clip,i)
                rows.append(dict(source_frame=i, timestamp=i/row['fps'], available_at=future_key/row['fps'],
                    interaction_box=tight,crop_box=box,source_camera='cam0',clip=clip['name'],
                    generation_mode='manual_offline_interpolated',review_status='temporal_qa_candidate'))
                if i in qa:
                    tile = Image.new('RGB',(224,248),'#151515')
                    tile.paste(Image.fromarray(crop),(0,24))
                    ImageDraw.Draw(tile).text((4,4),f'f{i} {clip["name"][:20]}',fill='white')
                    tiles.append(tile)
                    panel.save(staging/f'qa_f{i:06d}.jpg',quality=90)
                    if len(tiles)==20:
                        sheet=Image.new('RGB',(1120,992))
                        for j,t in enumerate(tiles):sheet.paste(t,((j%5)*224,(j//5)*248))
                        sheet.save(staging/f'crops_qa_{page:02d}.jpg',quality=95)
                        page+=1;tiles=[]
            for packet in stream.encode():video.mux(packet)
            for packet in local.encode():clean.mux(packet)
        if tiles:
            sheet=Image.new('RGB',(1120,248*((len(tiles)+4)//5)))
            for j,t in enumerate(tiles):sheet.paste(t,((j%5)*224,(j//5)*248))
            sheet.save(staging/f'crops_qa_{page:02d}.jpg',quality=95)
        if len(rows)!=row['length'] or digest(source)!=expected:
            raise ValueError('Frame count/source verification failed')
        for name in ['review.mp4','interaction224.mp4']:
            with av.open(str(staging/name)) as container:
                if sum(1 for _ in container.decode(video=0))!=len(rows):
                    raise ValueError('Encoded video frame count mismatch')
        (staging/'index.json').write_text(json.dumps(rows,indent=2)+'\n')
        receipt=dict(episode_index=row['episode_index'],source_recording=row['source_recording'],
            source_video=str(source),source_sha256=expected,source_hash_unchanged=True,
            selection_sha256=digest(selection_path),annotation_sha256=digest(annotations_path),
            frames=len(rows),qa_frames=len(qa),keyframes=sum(len(c['keyframes']) for c in annotations['clips']),
            full_recording_geometry_coverage=True,full_roi_coverage=False,independent_human_review=False,
            training_ready=False,source_writes=False,training_started=False,
            videos={name:digest(staging/name) for name in ['review.mp4','interaction224.mp4']})
        (staging/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.selection,a.annotations,a.out),indent=2))


if __name__=='__main__':
    main()
