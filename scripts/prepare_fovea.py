"""Create a separate label bundle without writing existing datasets or videos."""

import argparse
import contextlib
import gzip
import json
import os
from pathlib import Path
import shutil

import numpy as np

from openpi.subtasks import fovea
from openpi.subtasks import tomato_in_hand
from openpi.training.fovea_dataset import digest


def make_labels(rows, detections, segments, length):
    if len(rows) != length or detections["frames"] != length:
        raise ValueError("ROI/detection frame length mismatch")
    boxes = np.array([r["interaction_box"] for r in rows], dtype=float) / [640, 400, 640, 400]
    valid = np.array([r["nearest_detection_frames"] <= 15 for r in rows], dtype=bool)
    anchors = []
    for sample in detections["samples"]:
        candidates = [d for d in sample["detections"] if d["label"] == 2 and d["score"] >= 0.75]
        if candidates:
            best = max(candidates, key=lambda d: d["score"])
            b = np.clip(np.asarray(best["box"]) / [640, 400, 640, 400], 0, 1)
            with contextlib.suppress(ValueError):
                anchors.append((sample["frame"], fovea.validate_box(b)))
    anchor_frames = np.array([a[0] for a in anchors], dtype=int)
    left_used = 0
    for i, row in enumerate(rows):
        if row["frame_index"] != i or row["source_frame_index"] != i or abs(row["timestamp"] - i / 30) > 1e-6:
            raise ValueError("ROI timeline changed")
        fovea.validate_box(boxes[i])
    for segment in segments:
        label = segment["label"]
        for i in range(segment["start_frame"], segment["end_frame"]):
            components = [boxes[i]]
            if label == "harvest":
                components.append(fovea.LEFT_BASKET)
                if anchors:
                    pos = int(np.searchsorted(anchor_frames, i))
                    nearest = min(
                        (max(0, pos - 1), min(pos, len(anchors) - 1)), key=lambda j: abs(anchor_frames[j] - i)
                    )
                    if abs(anchor_frames[nearest] - i) <= 6:
                        components.append(anchors[nearest][1])
                        left_used += 1
                    else:
                        # Conservative left-hand workspace prior; explicit weak supervision.
                        components.append((0.18, 0.28, 0.60, 0.80))
                else:
                    components.append((0.18, 0.28, 0.60, 0.80))
            elif label in ("pickup", "return"):
                components.append(fovea.RIGHT_PLATE)
            boxes[i] = fovea.union_boxes(components)
    return boxes.astype(np.float32), valid, left_used


def prepare(args):
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use an empty, independent output directory")
    root.mkdir(parents=True, exist_ok=True)
    bundle = args.bundle.resolve()
    selection = json.loads((bundle / "selection.json").read_text())
    records = selection["episodes"]
    if len(records) != 119 or sum(r["length"] for r in records) != 357970:
        raise ValueError("Frozen119 selection required")
    source = bundle / "b_coarse"
    # Independent metadata, read-only references to existing physical data/video files.
    (root / "dataset").mkdir()
    shutil.copytree(source / "meta", root / "dataset/meta")
    for folder in ("data", "videos"):
        os.symlink(source / folder, root / "dataset" / folder, target_is_directory=True)
    metadata = {
        f"meta/{name}": digest(source / "meta" / name) for name in ("info.json", "episodes.jsonl", "tasks.jsonl")
    }
    (root / "labels").mkdir()
    result = []
    total_pairs = 0
    left_used_total = 0
    for r in records:
        p = Path(r["source_dataset"])
        ep = r["source_episode_index"]
        n = r["length"]
        sidecar = p / f"roi/chunk-{ep // 1000:03d}/episode_{ep:06d}.jsonl.gz"
        with gzip.open(sidecar, "rt") as f:
            rows = [json.loads(line) for line in f]
        annotation = Path(r["annotation"])
        if digest(annotation) != r["annotation_sha256"]:
            raise ValueError("Stage annotation changed")
        segments = tomato_in_hand.reviewed_segments(json.loads(annotation.read_text()), n)
        detection = args.detections / p.parent.name / p.name / f"episode_{ep:06d}.json"
        det = json.loads(detection.read_text())
        cam0 = f"videos/chunk-{ep // 1000:03d}/observation.images.cam0/episode_{ep:06d}.mp4"
        if det["source_sha256"] != r["source_sha256"][cam0]:
            raise ValueError("Detection source video identity differs")
        boxes, valid, left_used = make_labels(rows, det, segments, n)
        pairs = int((valid[: -fovea.EXECUTE_STEPS] & valid[fovea.EXECUTE_STEPS :]).sum())
        target = root / "labels" / f"episode_{r['episode_index']:06d}.npz"
        np.savez_compressed(target, boxes=boxes, valid=valid)
        result.append(
            {
                "episode_index": r["episode_index"],
                "frames": n,
                "labels": str(target.relative_to(root)),
                "sha256": digest(target),
                "source_dataset": str(p),
                "source_episode_index": ep,
                "source_roi_sha256": digest(sidecar),
                "source_annotation_sha256": digest(annotation),
                "source_detection_sha256": digest(detection),
                "training_pairs": pairs,
                "detected_left_hand_frames": left_used,
            }
        )
        total_pairs += pairs
        left_used_total += left_used
        print(f"Labels {len(result)}/119 pairs={pairs}", flush=True)
    contract = {
        "version": fovea.VERSION,
        "image_keys": list(fovea.IMAGE_KEYS),
        "source_size": [640, 400],
        "action_dim": 58,
        "action_horizon": 50,
        "action_hz": 30,
        "execute_steps": fovea.EXECUTE_STEPS,
        "box_loss_weight": 0.1,
        "coordinate_order": "ymin_xmin_ymax_xmax",
        "location_start": 256000,
        "bins": 1024,
        "initial_box": list(fovea.INITIAL_BOX),
        "crop_expansion": 1.5,
        "memory_enabled": False,
        "source_coordinates": "normalized_xyxy",
        "token_coordinates": "padded_full224",
        "target_priors": {"left_basket": list(fovea.LEFT_BASKET), "right_plate": list(fovea.RIGHT_PLATE)},
    }
    manifest = {
        "version": fovea.VERSION,
        "complete": True,
        "source_bundle": str(bundle),
        "source_selection_sha256": digest(bundle / "selection.json"),
        "dataset_metadata": metadata,
        "episodes": result,
        "frames": 357970,
        "training_pairs": total_pairs,
        "execute_steps": fovea.EXECUTE_STEPS,
        "source_size": [640, 400],
        "contract": contract,
        "detected_left_hand_frames": left_used_total,
        "weak_supervision": True,
        "independent_semantic_accuracy_measured": False,
        "original_write_operations": 0,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("frames", "training_pairs", "detected_left_hand_frames")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
