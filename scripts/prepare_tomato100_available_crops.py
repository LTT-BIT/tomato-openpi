"""Render every current frame: verified local geometry or explicit full-view fallback.

Preserves the parent's local-valid evidence; never promotes a fallback to a
successful localization. No future observations, robot I/O, or source writes.
"""
import argparse
import collections
import concurrent.futures
import gzip
import json
from pathlib import Path

from openpi.shared.local_storage import atomic_write_json, new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest
from scripts.materialize_tomato100_four_view import collect_candidates, read_candidate_index

STREAM = 'current_rgb_local_or_full_fallback_v5'


def image_plan(row):
    """Availability is distinct from localization reliability."""
    w, h = row['source_size']
    return dict(image_available=True, local_roi_valid=row['valid'],
                input_mode='local_crop' if row['valid'] else 'full_view_fallback',
                input_box=row['crop_box'] if row['valid'] else [0, 0, w, h],
                fallback_reason=None if row['valid'] else row['reason'])


def render(image, row):
    import cv2
    import numpy as np
    from scripts.tomato_interaction_preview import native_crop
    if row['input_mode'] == 'local_crop':
        return cv2.resize(native_crop(image, row['input_box']), (224, 224), interpolation=cv2.INTER_AREA)
    # Match full-view aspect-preserving letterboxing, using CURRENT pixels.
    h, w = image.shape[:2]
    ratio = max(w / 224, h / 224)
    rh, rw = int(h / ratio), int(w / ratio)
    result = np.zeros((224, 224, 3), np.uint8)
    top, left = (224-rh)//2, (224-rw)//2
    result[top:top+rh, left:left+rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA)
    return result


def prepare_episode(args):
    import av
    import cv2
    import numpy as np
    selection_row, parent_folder, parent_receipt, target, contract = args
    cv2.setNumThreads(1)
    ep = parent_receipt['episode_index']
    video_rel = f"videos/chunk-{selection_row['source_episode_index']//1000:03d}/observation.images.cam0/episode_{selection_row['source_episode_index']:06d}.mp4"
    source = Path(selection_row['dataset']) / video_rel
    index_path = parent_folder / 'index.jsonl.gz'
    if digest(index_path) != parent_receipt['index_sha256'] or digest(source) != selection_row['source_sha256'][video_rel]:
        raise ValueError('Parent index or original video changed')
    rows = read_candidate_index(index_path, selection_row['length'])
    counts = collections.Counter()
    checks = {}
    with new_artifact_directory(target) as staging:
        with av.open(str(source)) as src, av.open(str(staging/'interaction224.mp4'), 'w') as dst, gzip.open(staging/'index.jsonl.gz', 'wt') as output:
            stream = dst.add_stream('libx264', rate=30)
            stream.width, stream.height, stream.pix_fmt = 224, 224, 'yuv420p'
            stream.options = {'crf':'16', 'preset':'fast'}
            stream.thread_count = 2
            n = 0
            for n, frame in enumerate(src.decode(video=0), 1):
                row = rows[n-1]
                if frame.pts is None or abs(float(frame.pts*frame.time_base)-(n-1)/30) > 1e-4:
                    raise ValueError('Original video timestamp mismatch')
                row = {**row, **image_plan(row)}
                pixels = frame.to_ndarray(format='rgb24')
                if list(pixels.shape[1::-1]) != row['source_size']:
                    raise ValueError('Native dimensions changed')
                crop = render(pixels, row)
                counts[row['input_mode']] += 1
                # Validate every fallback against current-source rendering after encoding.
                if not row['local_roi_valid'] or (n-1)%200 == 0:
                    checks[n-1] = crop
                output.write(json.dumps(row)+'\n')
                for packet in stream.encode(av.VideoFrame.from_ndarray(crop, format='rgb24')):
                    dst.mux(packet)
            for packet in stream.encode():dst.mux(packet)
        if n != len(rows):raise ValueError('Original frame count mismatch')
        black = 0
        max_mae = 0.
        decoded = 0
        with av.open(str(staging/'interaction224.mp4')) as verify:
            for decoded, frame in enumerate(verify.decode(video=0), 1):
                pixels = frame.to_ndarray(format='rgb24')
                if pixels.shape != (224,224,3) or abs(float(frame.pts*frame.time_base)-(decoded-1)/30)>1e-4:
                    raise ValueError('Output shape/timestamp mismatch')
                black += int(not pixels.any())
                if decoded-1 in checks:
                    mae = float(np.abs(pixels.astype(np.float32)-checks[decoded-1]).mean())
                    max_mae = max(max_mae, mae)
                    if mae > 12:raise ValueError('Encoded pixels differ from current source')
        if decoded != n or black or digest(source) != selection_row['source_sha256'][video_rel]:
            raise ValueError('Incomplete/black output or modified source')
        receipt = dict(contract=contract, episode_index=ep, frames=n,
                       algorithm_valid_frames=counts['local_crop'], algorithm_invalid_frames=counts['full_view_fallback'],
                       image_available_frames=n, input_mode_counts=dict(counts),
                       video_sha256=digest(staging/'interaction224.mp4'), index_sha256=digest(staging/'index.jsonl.gz'),
                       parent_receipt_sha256=digest(parent_folder/'receipt.json'), source_video_sha256=digest(source),
                       source_hashes_unchanged=True, all_frames_decoded=True, black_frames=black,
                       current_pixel_checks=len(checks), current_pixel_max_mae=max_mae,
                       training_ready=False, source_writes=False)
        atomic_write_json(staging/'receipt.json', receipt)
    return receipt


def run(selection_path, parent, out, workers):
    selection = json.loads(selection_path.read_text())
    parent_contract, episodes = collect_candidates(parent)
    if parent_contract['selection_sha256'] != digest(selection_path):raise ValueError('Selection changed')
    out = derived_output_path(selection, out)
    out.mkdir(exist_ok=False, parents=True)
    contract = dict(stream_contract=STREAM, selection_sha256=digest(selection_path),
                    generator_sha256=digest(Path(__file__)), parent_contract=parent_contract,
                    fallback_policy='current_full_view_with_aspect_preserving_letterbox',
                    uses_future_frames_for_roi=False, preserves_parent_local_validity=True)
    atomic_write_json(out/'contract.json', contract)
    jobs = [(selection['episodes'][ep], folder, receipt, out/f'shard{ep%4}'/f'episode_{ep:06d}', contract)
            for ep, (folder,receipt) in sorted(episodes.items())]
    completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for receipt in pool.map(prepare_episode, jobs):
            completed.append(receipt)
            print(f"ep{receipt['episode_index']}: {receipt['frames']} current images, {receipt['algorithm_invalid_frames']} explicit full-view fallbacks", flush=True)
    for shard in range(4):
        receipts = [r for r in completed if r['episode_index']%4 == shard]
        atomic_write_json(out/f'shard{shard}'/'candidate_manifest.json', dict(contract=contract, episodes=receipts))
    result = dict(contract=contract, episodes=100, frames=sum(r['frames'] for r in completed),
                  image_available_frames=sum(r['image_available_frames'] for r in completed),
                  local_roi_valid_frames=sum(r['algorithm_valid_frames'] for r in completed),
                  full_view_fallback_frames=sum(r['algorithm_invalid_frames'] for r in completed),
                  full_image_coverage=True, full_roi_coverage=False, training_ready=False,
                  black_frames=sum(r['black_frames'] for r in completed), all_frames_decoded=True,
                  current_pixel_checks=sum(r['current_pixel_checks'] for r in completed),
                  source_writes=False, training_started=False,
                  limitation='Fallback changes field of view; smooth transitions and independent semantic accuracy are not certified.')
    if result['frames'] != 425556 or result['image_available_frames'] != result['frames']:
        raise ValueError('100% image coverage required')
    atomic_write_json(out/'availability_manifest.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if not 1 <= a.workers <= 16:p.error('workers must be 1..16')
    run(a.selection,a.parent,a.out,a.workers)
