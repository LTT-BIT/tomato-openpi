"""Measure warmed B1 RGB localization, causal selection and crop; no robot or policy calls."""
import argparse
import json
import time
from pathlib import Path

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.tomato_interaction_preview import native_crop
from scripts.tomato_roi_locator import CausalRoleSelector
from scripts.tomato_repair_locator import RgbLocator


def main():
    import av
    import cv2
    import numpy as np
    import pyarrow.parquet as pq
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--locator', type=Path, required=True)
    parser.add_argument('--norm', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stable', action='store_true', help='Measure the bounded-motion stable selector')
    parser.add_argument('--state-guarded', action='store_true', help='Measure stable selection with measured grasp and past RGB background')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    selection = json.loads(args.selection.read_text())
    row = selection['episodes'][86]
    prefix = f"chunk-{row['source_episode_index']//1000:03d}"
    video_rel = f"videos/{prefix}/observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4"
    data_rel = f"data/{prefix}/episode_{row['source_episode_index']:06d}.parquet"
    video, data = [Path(row['dataset']) / rel for rel in (video_rel, data_rel)]
    for path, rel in [(video, video_rel), (data, data_rel)]:
        if digest(path) != row['source_sha256'][rel]:
            raise ValueError('Original source changed')
    state = np.asarray(pq.read_table(data, columns=['observation.state'])['observation.state'].to_pylist())
    # Decode once outside the timer. Replay EVERY state and RGB in chronological order.
    with av.open(str(video)) as container:
        images = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    if len(images) != row['length']:
        raise ValueError('Frame mismatch')
    norm = json.loads((args.norm / 'norm_stats.json').read_text())['norm_stats']['state']
    if args.state_guarded:
        from scripts.tomato_transport_roi import StableRoi
        selector = StableRoi()
    elif args.stable:
        from scripts.tomato_stable_roi import StableRoi
        selector = StableRoi()
    else:
        selector = CausalRoleSelector(np.asarray(norm['q99']) - np.asarray(norm['q01']))
    locator = RgbLocator(args.locator)
    for _ in range(5):
        locator.predict([images[0]])
    torch.cuda.synchronize()
    timing = {'locator_ms': [], 'selector_ms': [], 'crop_ms': [], 'total_ms': []}
    valid_count = 0
    # First 300 consecutive frames, no gaps that would distort measured motion.
    for frame, image in enumerate(images[:300]):
        torch.cuda.synchronize()
        start = time.perf_counter()
        detections = locator.predict([image])[0]
        torch.cuda.synchronize()
        located = time.perf_counter()
        roi = selector.select(detections, state[frame], image.shape[1], image.shape[0], rgb=image)
        selected = time.perf_counter()
        crop = np.zeros((224, 224, 3), np.uint8)
        if roi['valid']:
            valid_count += 1
            crop = cv2.resize(native_crop(image, roi['crop_box']), (224, 224), interpolation=cv2.INTER_AREA)
        elif args.stable or args.state_guarded:
            crop = cv2.resize(native_crop(image, roi['display_crop_box']), (224, 224), interpolation=cv2.INTER_AREA)
        if crop.shape != (224, 224, 3):
            raise ValueError('Crop shape changed')
        finished = time.perf_counter()
        for key, delta in [('locator_ms', located-start), ('selector_ms', selected-located),
                           ('crop_ms', finished-selected), ('total_ms', finished-start)]:
            timing[key].append(delta*1000)
    for path, rel in [(video, video_rel), (data, data_rel)]:
        if digest(path) != row['source_sha256'][rel]:
            raise ValueError('Original source changed during benchmark')
    result = dict(device=torch.cuda.get_device_name(), batch_size=1, samples=300,
                  selector_mode='state_guarded' if args.state_guarded else ('stable' if args.stable else 'baseline'),
                  warmup_calls=5, episode_index=86, consecutive_source_frames=[0,300],
                  selection_sha256=digest(args.selection), locator_sha256=digest(args.locator/'model.pt'),
                  selector_sha256=digest(Path(selector.select.__code__.co_filename)),
                  normalization_sha256=digest(args.norm/'norm_stats.json'),
                  measurements={key: dict(p50=float(np.percentile(values,50)),
                                         p95=float(np.percentile(values,95)), maximum=max(values))
                                for key,values in timing.items()},
                  algorithm_valid_frames=valid_count, includes_decode=False, includes_encode=False,
                  includes_policy=False, quality_accepted=False, deployment_cadence_accepted=False,
                  source_writes=False, training_started=False)
    out = derived_output_path(selection,args.out)
    with new_artifact_directory(out) as staging:
        (staging/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['measurements'],indent=2))


if __name__ == '__main__':
    main()
