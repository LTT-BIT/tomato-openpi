"""Time existing offline ROI geometry and exports on real data; never time a fake policy."""
import argparse
import json
import platform
import time
from pathlib import Path

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.prepare_tomato100_roi import export_preview, worklist
from scripts.review_tomato_manual_roi import run as review
from scripts.export_tomato_manual_crops import distribution, lean_export
from scripts.tomato_interaction_preview import annotation_available_frame, interaction_at, native_crop, square_box


def main():
    import av
    import cv2
    import numpy as np
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    selection=json.loads(a.selection.read_text());annotations=json.loads(a.annotations.read_text())
    out=derived_output_path(selection,a.out)
    cv2.setNumThreads(1)
    started=time.perf_counter()
    with new_artifact_directory(out) as staging:
        start=time.perf_counter();lean,samples=lean_export(selection,annotations,staging/'clean_only',selection_sha256=digest(a.selection),annotation_sha256=digest(a.annotations),retain_samples=True)
        lean['total_wall_seconds']=time.perf_counter()-start
        print('clean_only',lean['total_wall_seconds'],flush=True)
        geometry=[];crops=[]
        for _ in range(5):
            for frame,image,clip in samples:
                h,w=image.shape[:2]
                t=time.perf_counter();tight=interaction_at(clip,frame,w,h);box=square_box(tight,w,h)
                geometry.append(time.perf_counter()-t)
                t=time.perf_counter();cv2.resize(native_crop(image,box),(224,224),interpolation=cv2.INTER_AREA)
                crops.append(time.perf_counter()-t)
        start=time.perf_counter();review(a.selection,a.annotations,staging/'full_review')
        full_seconds=time.perf_counter()-start
        print('full_review',full_seconds,flush=True)
        start=time.perf_counter();export_preview(selection,annotations,staging/'compressed_npz')
        npz_seconds=time.perf_counter()-start
        print('compressed_npz',npz_seconds,flush=True)
        # Exactly the same clean-video bytes, not a speedup obtained by changing pixels/encoding quality.
        same_video=digest(staging/'clean_only/interaction224.mp4')==digest(staging/'full_review/interaction224.mp4')
        if not same_video:raise ValueError('Clean video differs between export modes')
        with np.load(staging/'compressed_npz/crops224.npz') as arrays:
            rgb=arrays['rgb']
            for frame,image,clip in samples:
                h,w=image.shape[:2];expected=cv2.resize(native_crop(image,square_box(interaction_at(clip,frame,w,h),w,h)),(224,224),interpolation=cv2.INTER_AREA)
                if not np.array_equal(rgb[frame],expected):raise ValueError('NPZ pixels differ')
        result=dict(selection_sha256=digest(a.selection),annotation_sha256=digest(a.annotations),
            machine=dict(platform=platform.platform(),python=platform.python_version(),opencv=cv2.__version__,
                         pyav=av.__version__,opencv_threads=cv2.getNumThreads()),
            frames=lean['frames'],source_duration_seconds=lean['frames']/annotations['fps'],
            single_frame=dict(offline_interpolation_and_square=distribution(geometry),crop_resize224=distribution(crops),
                warning='Offline interpolation reads future keyframes. This is NOT automatic/causal ROI selection latency.'),
            exports=dict(clean_only=lean,full_review_total_seconds=full_seconds,compressed_npz_total_seconds=npz_seconds,
                         clean_video_identical=True,npz_sample_pixels_verified=len(samples)),
            annotation_authoring=dict(measured=False,reason='Previous visual selection/rule editing was not instrumented; 15-30 min was an estimate, not a measured benchmark.'),
            four_view_policy_inference=dict(measured=False,p50_ms=None,p95_ms=None,
                reason='No complete runnable four-view checkpoint/model environment in this local benchmark; local pi05_base cache is a partial download. Adapter timing is not model inference.'),
            realtime_acceptance=False,training_started=False,source_writes=False,
            total_wall_seconds=time.perf_counter()-started)
        (staging/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
