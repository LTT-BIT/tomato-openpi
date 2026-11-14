"""Offline right-truss ROI packages for full-block-more, without raw writes.

Coordinates may use future detections. Pixels always come from the matching
current cam0 frame. Geometry checks do not certify semantic annotation quality.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from scripts.tomato_offline_roi import plan_windows, union


CAMERA = 'observation.images.interaction'
VERSION = 'full-block-right-truss-180-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def inventory(base):
    base = Path(base)
    result = []
    for category in ('normal', 'speedup'):
        for info in sorted((base / category / 'full-block-more').glob('[AB]/*/meta/info.json')):
            folder = info.parent.parent
            meta = json.loads(info.read_text())
            rows = [json.loads(line) for line in (info.parent / 'episodes.jsonl').read_text().splitlines()]
            if len(rows) != meta['total_episodes'] or sum(r['length'] for r in rows) != meta['total_frames']:
                raise ValueError(f'Metadata counts disagree: {folder}')
            for row in rows:
                ep = row['episode_index']
                rel = folder.relative_to(base)
                video = Path(meta['video_path'].format(episode_chunk=ep // meta['chunks_size'], episode_index=ep,
                                                     video_key='observation.images.cam0'))
                result.append(dict(category=category, group=folder.parent.name, dataset=folder.name,
                                   relative_dataset=str(rel), episode=ep, frames=row['length'], fps=meta['fps'],
                                   video=str(rel / video)))
    if not result:
        raise ValueError('No raw datasets found')
    return result


def select_datasets(rows, names):
    if not names:
        return rows
    wanted = set(names)
    found = {r['dataset'] for r in rows} & wanted
    if found != wanted:
        raise ValueError(f'Requested datasets missing: {sorted(wanted - found)}')
    return [r for r in rows if r['dataset'] in wanted]


def freeze(base, output, datasets=None):
    """Run on source host before processing; only read raw [AB]/dataset trees."""
    base = Path(base)
    rows = select_datasets(inventory(base), datasets)
    roots = sorted({r['relative_dataset'] for r in rows})
    paths = [p for rel in roots for p in sorted((base / rel).rglob('*')) if p.is_file()]
    def record(p):
        if p.is_symlink():
            raise ValueError(f'Raw symlink needs explicit review: {p}')
        return dict(path=str(p.relative_to(base)), bytes=p.stat().st_size, sha256=sha(p))
    with ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(record, paths))
    save(output, dict(version=VERSION, raw_base=str(base), episodes=rows, files=files,
                      raw_write_operations=0, generated_at_unix=time.time()))
    print(json.dumps(dict(frozen_files=len(files), episodes=len(rows), frames=sum(r['frames'] for r in rows))), flush=True)


def gap(a, b):
    return float(np.linalg.norm(np.maximum(0, np.maximum(a[:2], b[:2]) - np.minimum(a[2:], b[2:]))))


def interaction(rgb, detections):
    """Anchor the right hand/truss; do not follow left transport to its basket."""
    import cv2
    h, w = rgb.shape[:2]
    best = {}
    for d in detections:
        b = np.asarray(d['box'], float)
        label = d['label']
        if label not in (1, 2, 3) or d['score'] < (.75 if label != 3 else .85):
            continue
        if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= b[:2]):
            continue
        if np.prod(b[2:] - b[:2]) > .13 * w * h or min(b[2:] - b[:2]) < 10:
            continue
        if label not in best or d['score'] > best[label]['score']:
            best[label] = d
    if 1 not in best:
        return None
    hand = np.asarray(best[1]['box'], float)
    boxes = [hand]
    # Red connected components provide visible fruit extent; not outcome labels.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (((hsv[:, :, 0] < 13) | (hsv[:, :, 0] > 169)) &
            (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 60)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    candidates = [np.array([x, y, x + cw, y + ch], float)
                  for x, y, cw, ch, area in stats[1:] if 8 <= area <= 3000 and min(cw, ch) >= 3]
    active = hand.copy()
    for _ in range(4):
        retained = []
        changed = False
        for b in candidates:
            merged = union([active, b])
            if gap(active, b) <= (35 if len(boxes) == 1 else 15) and max(merged[2:] - merged[:2]) <= 240:
                boxes.append(b)
                active = merged
                changed = True
            else:
                retained.append(b)
        candidates = retained
        if not changed:
            break
    if 3 in best:
        obj = np.array(best[3]['box'])
        merged = union([active, obj])
        if gap(active, obj) < 30 and max(merged[2:] - merged[:2]) <= 240:
            boxes.append(obj)
    tight = union(boxes)
    # Context around the grasp includes the stem and approaching left fingertips.
    tight += np.array([-12, -18, 12, 18])
    tight[:2] = np.maximum(tight[:2], 0)
    tight[2:] = np.minimum(tight[2:], [w, h])
    return dict(box=tight.tolist(), right_score=best[1]['score'], evidence='right_hand_and_nearby_rgb_fruit')


def detect(row, source, cache, model, stride=3):
    import av
    path = source / row['video']
    identity = sha(path)
    if cache.exists():
        old = json.loads(cache.read_text())
        if old['source_sha256'] != identity or old['version'] != VERSION or old['stride'] != stride:
            raise ValueError(f'Stale detection cache: {cache}')
        return old
    samples, images, indices = [], [], []
    def consume():
        predictions = model.predict(images)
        for idx, rgb, prediction in zip(indices, images, predictions, strict=True):
            samples.append(dict(frame=idx, detections=prediction, interaction=interaction(rgb, prediction)))
        images.clear()
        indices.clear()
    with av.open(str(path)) as handle:
        count = 0
        for i, frame in enumerate(handle.decode(video=0)):
            if i % stride == 0 or i == row['frames'] - 1:
                images.append(frame.to_ndarray(format='rgb24'))
                indices.append(i)
                if len(images) >= 16:
                    consume()
            count += 1
        if images:
            consume()
    if count != row['frames']:
        raise ValueError(f'Raw cam0 frame count differs: {path}: {count}')
    value = dict(version=VERSION, source_sha256=identity, stride=stride, frames=count, samples=samples)
    save(cache, value)
    return value


def coordinates(detection, width=640, height=400):
    samples = [s for s in detection['samples'] if s['interaction'] is not None]
    if len(samples) < 2:
        raise ValueError('Insufficient right-truss anchors; needs review')
    positions = np.array([s['frame'] for s in samples])
    anchors = np.array([s['interaction']['box'] for s in samples])
    frames = np.arange(detection['frames'])
    tight = np.stack([np.interp(frames, positions, anchors[:, j]) for j in range(4)], axis=1)
    boxes, checks = plan_windows(tight, width, height, minimum=180., maximum=300., context_ratio=1.12)
    indices = np.searchsorted(positions, frames)
    before = positions[np.maximum(0, indices - 1)]
    after = positions[np.minimum(len(positions) - 1, indices)]
    nearest = np.minimum(np.abs(frames - before), np.abs(frames - after))
    checks.update(dict(detection_anchors=len(samples), sampled_frames=len(detection['samples']),
                       max_anchor_gap_frames=int(np.diff(positions).max(initial=0)),
                       max_nearest_anchor_distance_frames=int(nearest.max()),
                       frames_over_15_from_anchor=int((nearest > 15).sum()),
                       minimum_side=int((boxes[:, 2] - boxes[:, 0]).min()),
                       median_side=float(np.median(boxes[:, 2] - boxes[:, 0])),
                       maximum_side=int((boxes[:, 2] - boxes[:, 0]).max()),
                       future_coordinates_used=True, same_current_frame_pixels=True,
                       independent_semantic_quality_accepted=False))
    return boxes, tight, nearest, checks


def export(row, source, output, detection):
    import av
    import cv2
    from PIL import Image, ImageDraw
    relative = Path(row['category']) / 'full-block-more' / 'subtask-roi' / row['group'] / row['dataset']
    folder = output / relative
    ep = row['episode']
    receipt = folder / f'roi/chunk-{ep // 1000:03d}/episode_{ep:06d}.receipt.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous['source_sha256'] == detection['source_sha256'] and previous['version'] == VERSION:
            return previous
        raise ValueError(f'Output collision: {receipt}')
    boxes, tight, nearest, checks = coordinates(detection)
    video = folder / f'videos/chunk-{ep // 1000:03d}/{CAMERA}/episode_{ep:06d}.mp4'
    sidecar = receipt.with_name(f'episode_{ep:06d}.jsonl.gz')
    video.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = video.with_name(video.stem + '.partial.mp4')
    source_path = source / row['video']
    gallery_frames = set(np.linspace(0, row['frames'] - 1, 12).round().astype(int).tolist())
    # Additional middle interval frames support picking/reorientation review.
    gallery_frames.update(range(0, row['frames'], 180))
    gallery = []
    with av.open(str(source_path)) as incoming, av.open(str(temporary), 'w') as outgoing, gzip.open(sidecar, 'wt') as index:
        stream = outgoing.add_stream('libx264', rate=row['fps'])
        stream.width = stream.height = 224
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast', 'threads': '2'}
        count = 0
        timestamp_error = 0.
        for i, frame in enumerate(incoming.decode(video=0)):
            if i >= len(boxes):
                raise ValueError('Extra source frame')
            timestamp_error = max(timestamp_error, abs(float(frame.pts * frame.time_base) - i / row['fps']))
            rgb = frame.to_ndarray(format='rgb24')
            x1, y1, x2, y2 = boxes[i]
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError('Empty crop')
            resized = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)
            result = av.VideoFrame.from_ndarray(resized, format='rgb24')
            result.pts, result.time_base = i, Fraction(1, row['fps'])
            for packet in stream.encode(result):
                outgoing.mux(packet)
            index.write(json.dumps(dict(frame_index=i, source_frame_index=i, timestamp=i / row['fps'],
                                        crop_box=boxes[i].tolist(), interaction_box=tight[i].tolist(),
                                        nearest_detection_frames=int(nearest[i]),
                                        provenance='offline_neighbor_coordinate_interpolation_current_rgb',
                                        image_available=True, semantic_quality_verified=False)) + '\n')
            if i in gallery_frames:
                im = Image.fromarray(rgb)
                ImageDraw.Draw(im).rectangle((int(x1), int(y1), int(x2), int(y2)), outline='yellow', width=2)
                cell = Image.new('RGB', (544, 224), (12, 12, 12))
                cell.paste(im.resize((320, 200)), (0, 24))
                cell.paste(Image.fromarray(resized), (320, 0))
                ImageDraw.Draw(cell).text((4, 4), f'{i / row["fps"]:.1f}s | {x2-x1}px', fill='white')
                gallery.append(cell)
            count += 1
        for packet in stream.encode():
            outgoing.mux(packet)
    if count != row['frames'] or timestamp_error > .0034:
        raise ValueError(f'Source temporal alignment failure: {count}, {timestamp_error}')
    decoded = 0
    max_pts_error = 0.
    black_frames = 0
    with av.open(str(temporary)) as verified:
        for i, frame in enumerate(verified.decode(video=0)):
            if (frame.width, frame.height) != (224, 224):
                raise ValueError('Incorrect output dimensions')
            max_pts_error = max(max_pts_error, abs(float(frame.pts * frame.time_base) - i / row['fps']))
            if not np.any(frame.to_ndarray(format='rgb24')):
                black_frames += 1
            decoded += 1
    if decoded != count or black_frames or max_pts_error > .0034:
        raise ValueError('Export decode check failed')
    temporary.replace(video)
    for page in range(0, len(gallery), 12):
        cells = gallery[page:page + 12]
        sheet = Image.new('RGB', (1088, ((len(cells) + 1) // 2) * 224), (12, 12, 12))
        for j, cell in enumerate(cells):
            sheet.paste(cell, ((j % 2) * 544, (j // 2) * 224))
        sheet.save(receipt.with_name(f'episode_{ep:06d}.qa_{page // 12:02d}.jpg'), quality=92)
    result = dict(version=VERSION, source=row, source_sha256=detection['source_sha256'],
                  output_video=str(video.relative_to(output)), output_sha256=sha(video),
                  coordinates_sha256=sha(sidecar), frames=count, decoded_frames=decoded,
                  black_frames=black_frames, max_timestamp_error=max_pts_error,
                  source_max_timestamp_error=timestamp_error, checks=checks,
                  training_ready=False, subtask_labels_generated=False,
                  status='exported_pending_semantic_review')
    save(receipt, result)
    return result


def process(args):
    import torch
    import cv2
    from scripts.tomato_repair_locator import RgbLocator
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    rows = select_datasets(inventory(args.source), getattr(args, 'datasets', None))
    if args.stride < 1:
        raise ValueError('Detection stride must be positive')
    if args.category:
        rows = [r for r in rows if r['category'] == args.category]
    if args.select:
        choices = set(args.select.split(','))
        rows = [r for r in rows if f'{r["dataset"]}:{r["episode"]}' in choices]
        if len(rows) != len(choices):
            raise ValueError('Requested episodes not found')
    model = RgbLocator(args.model, device=args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / f'run_config_{args.category or "all"}.json', dict(version=VERSION, model=str(args.model), model_sha256=sha(Path(args.model) / 'model.pt'),
                                         stride=args.stride, source=str(args.source), future_coordinate_use=True,
                                         raw_write_operations=0, subtask_labels_generated=False))
    finished = 0
    failed = []
    def failure(row, exc):
        failed.append(row)
        save(output / 'failures' / row['category'] / row['dataset'] / f'episode_{row["episode"]:06d}.json',
             dict(source=row, error=f'{type(exc).__name__}: {exc}'))
        print(json.dumps(dict(failed=row, error=str(exc))), flush=True)
        if args.select:
            raise exc
    def collect(pending, ready):
        nonlocal finished
        for future in ready:
            row, start = pending.pop(future)
            try:
                receipt = future.result()
                error_record = output / 'failures' / row['category'] / row['dataset'] / f'episode_{row["episode"]:06d}.json'
                error_record.unlink(missing_ok=True)
                finished += 1
                print(json.dumps(dict(done=finished, total=len(rows), dataset=row['dataset'], episode=row['episode'],
                                      seconds=round(time.monotonic() - start, 2), checks=receipt['checks'])), flush=True)
            except Exception as exc:
                failure(row, exc)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {}
        for row in rows:
            start = time.monotonic()
            cache = output / 'detections' / row['category'] / row['group'] / row['dataset'] / f'episode_{row["episode"]:06d}.json'
            try:
                detection = detect(row, Path(args.source), cache, model, args.stride)
                pending[pool.submit(export, row, Path(args.source), output, detection)] = (row, start)
            except Exception as exc:
                failure(row, exc)
            ready = {f for f in pending if f.done()}
            if not ready and len(pending) >= 4:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            collect(pending, ready)
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            collect(pending, ready)
    if failed:
        raise RuntimeError(f'{len(failed)} episodes failed; see failure records')


def publish(args):
    """Materialize independent raw copies plus ROI under subtask-roi only."""
    base = Path(args.source).resolve()
    if Path(args.output).resolve() != base:
        raise ValueError('publish output must be the source base; writes are restricted to subtask-roi')
    frozen = json.loads(Path(args.freeze).read_text())
    rows = select_datasets(inventory(base), getattr(args, 'datasets', None))
    indexed = {f['path']: f for f in frozen['files']}
    current_files = {str(p.relative_to(base)) for rel in {r['relative_dataset'] for r in rows}
                     for p in (base / rel).rglob('*') if p.is_file()}
    if current_files != set(indexed):
        raise ValueError('Raw file membership changed after freeze')
    for item in frozen['files']:
        path = base / item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError(f'Raw file changed: {path}')
    packages = []
    for rel in sorted({r['relative_dataset'] for r in rows}):
        original = base / rel
        category, _, group, name = Path(rel).parts
        destination = base / category / 'full-block-more/subtask-roi' / group / name
        entries = [r for r in rows if r['relative_dataset'] == rel]
        review_path = destination / 'visual_review.json'
        review = json.loads(review_path.read_text())
        reviewed = {r['episode']: r for r in review['episodes']}
        if set(reviewed) != {r['episode'] for r in entries}:
            raise ValueError('Visual review membership differs from source episodes')
        receipts = []
        for row in entries:
            ep = row['episode']
            receipt_path = destination / f'roi/chunk-{ep // 1000:03d}/episode_{ep:06d}.receipt.json'
            receipt = json.loads(receipt_path.read_text())
            video = destination / f'videos/chunk-{ep // 1000:03d}/{CAMERA}/episode_{ep:06d}.mp4'
            coords = receipt_path.with_name(f'episode_{ep:06d}.jsonl.gz')
            if (receipt['source'] != row or receipt['source_sha256'] != indexed[row['video']]['sha256'] or
                    sha(video) != receipt['output_sha256'] or sha(coords) != receipt['coordinates_sha256'] or
                    receipt['frames'] != row['frames'] or receipt['decoded_frames'] != row['frames'] or
                    receipt['black_frames'] or not receipt['checks']['all_boxes_contained'] or
                    receipt['checks']['center_steps_over_3pct'] or receipt['checks']['scale_steps_over_2pct']):
                raise ValueError(f'ROI package verification failure: {receipt_path}')
            if reviewed[ep]['output_sha256'] != receipt['output_sha256']:
                raise ValueError('Visual review is bound to a different crop video')
            receipts.append(receipt)
        copied = []
        for item in frozen['files']:
            p = Path(item['path'])
            if not p.is_relative_to(Path(rel)):
                continue
            local_rel = p.relative_to(rel)
            if str(local_rel) == 'meta/info.json':
                continue
            target = destination / local_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if sha(target) != item['sha256']:
                    raise ValueError(f'Existing package file differs: {target}')
            else:
                temporary = target.with_suffix(target.suffix + '.partial')
                shutil.copyfile(base / p, temporary)
                if sha(temporary) != item['sha256']:
                    raise ValueError('Copy hash mismatch')
                temporary.replace(target)
            if target.stat().st_ino == (base / p).stat().st_ino:
                raise ValueError('Derived raw copies must not be hardlinks')
            copied.append(dict(path=str(local_rel), sha256=item['sha256'], bytes=item['bytes']))
        meta = json.loads((original / 'meta/info.json').read_text())
        feature = json.loads(json.dumps(meta['features']['observation.images.cam0']))
        feature['shape'] = [224, 224, 3]
        feature['info']['video.width'] = feature['info']['video.height'] = 224
        meta['features'][CAMERA] = feature
        meta['total_videos'] = 4 * meta['total_episodes']
        info_target = destination / 'meta/info.json'
        if info_target.exists() and json.loads(info_target.read_text()) != meta:
            raise ValueError(f'Existing derived metadata differs: {info_target}')
        save(info_target, meta)
        manifest = dict(version=VERSION, source_dataset=str(original), original_info_sha256=sha(original / 'meta/info.json'),
                        derived_info_sha256=sha(destination / 'meta/info.json'), raw_copies=copied,
                        episodes=len(entries), frames=sum(r['frames'] for r in entries),
                        image_keys=[f'observation.images.cam{i}' for i in range(3)] + [CAMERA],
                        roi_receipts=[dict(episode=r['source']['episode'], frames=r['frames'],
                                           output_sha256=r['output_sha256'], checks=r['checks']) for r in receipts],
                        state_action_unchanged=True, raw_source_unchanged=True, hardlinks_used=False,
                        future_coordinates_used=True, current_rgb_pixels_only=True,
                        subtask_labels_generated=False, training_ready=False,
                        visual_review_sha256=sha(review_path),
                        semantic_review_status='sampled_coding_agent_review_complete_not_independent_per_frame_accuracy')
        save(destination / 'roi_manifest.json', manifest)
        packages.append(dict(path=str(destination), category=category, group=group,
                             episodes=len(entries), frames=manifest['frames'],
                             manifest_sha256=sha(destination / 'roi_manifest.json')))
        print(json.dumps(packages[-1]), flush=True)
    # Recheck raw bytes after all derived writes, including the original metadata.
    for item in frozen['files']:
        if sha(base / item['path']) != item['sha256']:
            raise ValueError(f'Raw changed during publication: {item["path"]}')
    for category in ('normal', 'speedup'):
        selected = [p for p in packages if p['category'] == category]
        if not selected:
            continue
        manifest_path = base / category / 'full-block-more/subtask-roi/manifest.json'
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if previous:
            if previous['version'] != VERSION:
                raise ValueError('Different existing ROI version requires a separate destination')
            for entry in previous['packages']:
                if sha(Path(entry['path']) / 'roi_manifest.json') != entry['manifest_sha256']:
                    raise ValueError('Previously published package manifest changed')
            merged = {p['path']: p for p in previous['packages']}
            merged.update({p['path']: p for p in selected})
            selected = sorted(merged.values(), key=lambda p: p['path'])
        save(manifest_path,
             dict(version=VERSION, packages=selected, episodes=sum(p['episodes'] for p in selected),
                  frames=sum(p['frames'] for p in selected), raw_sha256_verified_files=len(indexed),
                  raw_verification_scope='selected_source_datasets_only',
                  raw_verification_datasets=sorted({r['relative_dataset'] for r in rows}),
                  raw_unchanged=True, training_ready=False, subtask_labels_generated=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'process', 'publish'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('/home/user/datasets/derived/tomato_roi_locator_hard_case_repair_r2'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stride', type=int, default=3)
    parser.add_argument('--select')
    parser.add_argument('--category', choices=('normal', 'speedup'))
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--datasets', nargs='+', help='Exact source dataset names; preserve other published packages')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.source, args.output, args.datasets)
    elif args.mode == 'process':
        process(args)
    else:
        if args.freeze is None:
            parser.error('publish requires --freeze')
        publish(args)


if __name__ == '__main__':
    main()
