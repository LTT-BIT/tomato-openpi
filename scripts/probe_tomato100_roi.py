"""Read-only, sparse detector probe; output is never an approved training ROI.

Saved stage labels select QA frames only. They are NOT passed to the detector.
Model files must already exist locally; no source images leave this machine.
"""
import argparse
import json
from pathlib import Path

from openpi.shared.local_storage import new_artifact_directory
from openpi.subtasks.rgb100 import derived_output_path, digest

MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
MODEL_REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'
MODEL_WEIGHT_SHA256 = '1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3'
DEFAULT_QUERY = 'robotic fingers. gripper. red tomatoes.'


def sample_frames(row):
    """One midpoint per saved segment, retaining identity (no full-coverage claim)."""
    segments = row['training_segments']
    if len(segments) != 7:
        raise ValueError('Expected seven saved segments')
    result = []
    previous = 0
    for segment in segments:
        start, end = segment['start_frame'], segment['end_frame']
        if start != previous or not 0 <= start < end <= row['length']:
            raise ValueError('Segments must partition source frames')
        result.append(dict(frame=(start + end - 1) // 2,
                           segment_index=segment['segment_index'], label=segment['label']))
        previous = end
    if previous != row['length']:
        raise ValueError('Segments must cover the full recording')
    return result


def checked_detections(labels, scores, boxes, width, height):
    import math
    if not len(labels) == len(scores) == len(boxes):
        raise ValueError('Detector output length mismatch')
    result = []
    for label, score, box in zip(labels, scores, boxes, strict=True):
        score = float(score)
        box = [float(v) for v in box]
        if len(box) != 4 or not all(math.isfinite(v) for v in [score, *box]):
            raise ValueError('Nonfinite detector output')
        if not 0 <= score <= 1:
            raise ValueError('Invalid detector score')
        x1, y1, x2, y2 = box
        box = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        result.append(dict(label=str(label), score=score, box_xyxy=box,
                           review_status='unreviewed_detector_candidate',
                           approved_for_training=False))
    return result


def run(args):
    import av
    import torch
    import transformers
    from PIL import Image, ImageDraw
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    selection = json.loads(args.selection.read_text())
    if selection.get('phase_label_resolution_required') != 0 or len(selection['episodes']) != 100:
        raise ValueError('Resolved frozen100 selection required')
    out = derived_output_path(selection, args.out)
    if out.exists():
        raise FileExistsError(out)
    model_dir = args.model_dir.expanduser().resolve()
    # Bind the actual locally loaded files, not merely an asserted revision string.
    model_hashes = {str(p.relative_to(model_dir)): digest(p)
                    for p in sorted(model_dir.iterdir()) if p.suffix in {'.json', '.txt', '.safetensors'}}
    if model_hashes.get('model.safetensors') != MODEL_WEIGHT_SHA256:
        raise ValueError('Local weights do not match the pinned model revision')
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir, local_files_only=True)
    model = model.to(args.device).eval()
    rows = selection['episodes'][:args.limit]
    results = []
    with new_artifact_directory(out) as staging:
        for row in rows:
            samples = sample_frames(row)
            wanted = {s['frame']: s for s in samples}
            relative = (f"videos/chunk-{row['source_episode_index']//1000:03d}/"
                        f"observation.images.cam0/episode_{row['source_episode_index']:06d}.mp4")
            source = Path(row['dataset']) / relative
            expected = row['source_sha256'][relative]
            if digest(source) != expected:
                raise ValueError(f'Source video changed: {source}')
            tiles = []
            episode_results = []
            with av.open(str(source)) as container:
                for frame_index, frame in enumerate(container.decode(video=0)):
                    if frame_index not in wanted:
                        continue
                    original = frame.to_image()
                    inputs = processor(images=original, text=args.query, return_tensors='pt').to(args.device)
                    with torch.inference_mode():
                        outputs = model(**inputs)
                    detection = processor.post_process_grounded_object_detection(
                        outputs, inputs.input_ids, threshold=args.threshold,
                        text_threshold=args.text_threshold,
                        target_sizes=[(original.height, original.width)])[0]
                    boxes = checked_detections(detection['text_labels'], detection['scores'].tolist(),
                                               detection['boxes'].tolist(), original.width, original.height)
                    record = dict(**wanted[frame_index], timestamp=frame_index/row['fps'],
                                  available_at=frame_index/row['fps'], source_size=list(original.size),
                                  source_camera='cam0', detections=boxes,
                                  interaction_box=None, interaction_valid=False,
                                  generation_mode='detector_probe', approved_for_training=False)
                    episode_results.append(record)
                    image = original.copy()
                    draw = ImageDraw.Draw(image)
                    for item in boxes:
                        draw.rectangle(item['box_xyxy'], outline='yellow', width=2)
                        draw.text(tuple(item['box_xyxy'][:2]),
                                  f"{item['label']} {item['score']:.2f}", fill='yellow')
                    tile = Image.new('RGB', (640, 430))
                    image.thumbnail((640, 400))
                    tile.paste(image, (0, 30))
                    ImageDraw.Draw(tile).text((5, 5),
                        f"ep{row['episode_index']} f{frame_index} {record['label']} | UNREVIEWED", fill='white')
                    tiles.append(tile)
                    if frame_index == max(wanted):
                        break
            if len(episode_results) != len(samples) or digest(source) != expected:
                raise ValueError('Source/frame verification failed')
            sheet = Image.new('RGB', (1280, 430*((len(tiles)+1)//2)))
            for i, tile in enumerate(tiles):
                sheet.paste(tile, ((i%2)*640, (i//2)*430))
            sheet_name = f"episode_{row['episode_index']:06d}.jpg"
            sheet.save(staging/sheet_name)
            results.append(dict(episode_index=row['episode_index'], dataset=row['dataset'],
                source_episode_index=row['source_episode_index'], source_video=str(source),
                source_sha256=expected, source_hash_unchanged=True,
                contact_sheet=sheet_name, samples=episode_results))
            print(f"Probed {len(results)}/{len(rows)} episodes; source unchanged", flush=True)
        receipt = dict(version='tomato100-sparse-detector-probe-v1',
            selection_sha256=digest(args.selection), model_id=MODEL_ID,
            download_revision=MODEL_REVISION, model_file_sha256=model_hashes,
            transformers_version=transformers.__version__, torch_version=torch.__version__,
            device=args.device, query=args.query, threshold=args.threshold,
            text_threshold=args.text_threshold, phase_labels_used='QA frame sampling only',
            episodes=results, sampled_frames=sum(len(r['samples']) for r in results),
            approved_roi_frames=0, full_roi_coverage=False, training_ready=False,
            source_writes=False, source_hashes_unchanged=True,
            note='Object detections are not interaction ROIs. No crops or training validity are fabricated.')
        (staging/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--query', default=DEFAULT_QUERY)
    p.add_argument('--threshold', type=float, default=.20)
    p.add_argument('--text-threshold', type=float, default=.20)
    args = p.parse_args()
    if not 1 <= args.limit <= 100 or not 0 < args.threshold < 1 or not 0 < args.text_threshold < 1:
        p.error('limit must be 1..100 and thresholds strictly between zero and one')
    run(args)


if __name__ == '__main__':
    main()
