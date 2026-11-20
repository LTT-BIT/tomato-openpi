"""Export only clean manual RGB crops and coordinates, omitting preview images/video."""
import argparse
import json
import time
from pathlib import Path
from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.prepare_tomato100_roi import worklist
from scripts.tomato_interaction_preview import annotation_available_frame, interaction_at, native_crop, square_box


def distribution(seconds):
    import numpy as np
    values=np.array(seconds)*1000
    return dict(samples=len(values),p50_ms=float(np.percentile(values,50)),
                p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)),
                max_ms=float(values.max()))


def lean_export(selection,annotations,out,*,selection_sha256,annotation_sha256,retain_samples=False):
    import av
    import cv2
    import numpy as np
    listing=worklist(selection,annotations,preview_sha256='bound_by_benchmark')
    row=next(r for r in listing['episodes'] if r['roi_clips'])
    if row['remaining_frames']:raise ValueError('Full recording required')
    out=derived_output_path(selection,out)
    rel=f"videos/chunk-{row['source_episode_index']//1000:03d}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
    source=Path(row['dataset'])/rel;expected=row['source_sha256'][rel]
    if digest(source)!=expected:raise ValueError('Source changed')
    wanted={i:c for c in annotations['clips'] for i in range(c['start'],c['end'])}
    timings={k:[] for k in ['decode_rgb','offline_geometry','crop_resize','encode_mux']}
    rows=[];samples=[]
    with new_artifact_directory(out) as staging:
        with av.open(str(source)) as src, av.open(str(staging/'interaction224.mp4'),'w') as dst:
            stream=dst.add_stream('libx264',rate=row['fps'])
            stream.width,stream.height,stream.pix_fmt=224,224,'yuv420p'
            stream.options={'crf':'16','preset':'fast'}
            iterator=iter(src.decode(video=0))
            for i in range(row['length']):
                start=time.perf_counter();image=next(iterator).to_ndarray(format='rgb24')
                timings['decode_rgb'].append(time.perf_counter()-start)
                h,w=image.shape[:2];clip=wanted[i]
                start=time.perf_counter();tight=interaction_at(clip,i,w,h);box=square_box(tight,w,h)
                timings['offline_geometry'].append(time.perf_counter()-start)
                start=time.perf_counter();crop=cv2.resize(native_crop(image,box),(224,224),interpolation=cv2.INTER_AREA)
                timings['crop_resize'].append(time.perf_counter()-start)
                start=time.perf_counter()
                for packet in stream.encode(av.VideoFrame.from_ndarray(crop,format='rgb24')):dst.mux(packet)
                timings['encode_mux'].append(time.perf_counter()-start)
                rows.append(dict(source_frame=i,crop_box=box,interaction_box=tight,
                    available_at=annotation_available_frame(clip,i)/row['fps'],source_camera='cam0',
                    generation_mode='manual_offline_interpolated'))
                if retain_samples and i%47==0:samples.append((i,image.copy(),clip))
            start=time.perf_counter()
            for packet in stream.encode():dst.mux(packet)
            flush=time.perf_counter()-start
        with av.open(str(staging/'interaction224.mp4')) as check:
            if sum(1 for _ in check.decode(video=0))!=row['length']:raise ValueError('Encoded frame count mismatch')
        (staging/'index.json').write_text(json.dumps(rows,indent=2)+'\n')
        if digest(source)!=expected:raise ValueError('Source changed')
        result=dict(frames=len(rows),source_sha256=expected,source_hash_unchanged=True,
            selection_sha256=selection_sha256,annotation_sha256=annotation_sha256,
            video_sha256=digest(staging/'interaction224.mp4'),
            stage_wall_seconds={k:sum(v) for k,v in timings.items()},
            stage_latency={k:distribution(v) for k,v in timings.items()},encoder_flush_seconds=flush,
            full_roi_coverage=False,training_ready=False,source_writes=False,
            note='Encode calls may use worker threads; per-stage wall times are not CPU utilization.')
        (staging/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result,samples



def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result,_=lean_export(json.loads(a.selection.read_text()),json.loads(a.annotations.read_text()),a.out,
        selection_sha256=digest(a.selection),annotation_sha256=digest(a.annotations))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
