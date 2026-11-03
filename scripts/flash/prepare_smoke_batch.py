"""Prepare real numeric batch plus three current-frame RGB images, read-only SSH.

Only one normal dataset version; separate dataset roots for mechanical train/eval.
Not a benchmark split. No raw/normal/speedup cross-version mixing.
"""

import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
from PIL import Image

from openpi.training.flash_data import fit_action_statistics
from openpi.training.flash_data import load_episode
from openpi.training.flash_data import make_batch
from openpi.training.flash_data import read_manifest
from openpi.training.flash_data import validate_split

REMOTE_FRAME = """import av,io,json,sys
from PIL import Image
path=sys.argv[1]; target=int(sys.argv[2])
with av.open(path) as container:
 stream=container.streams.video[0]
 for index,frame in enumerate(container.decode(stream)):
  if index==target:
   payload=io.BytesIO(); frame.to_image().save(payload,format="PNG")
   info={"frame_index":index,"pts":frame.pts,"time_base":str(frame.time_base),"time_seconds":float(frame.time)}
   sys.stdout.buffer.write((json.dumps(info)+"\\n").encode()+payload.getvalue()); break
 else: raise RuntimeError("Video shorter than requested frame")
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path(".cache/flash_samples"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=302)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = read_manifest(args.samples)
    train_records = [r for r in records if r["kind"] == "normal" and "/A/1022_54_120_" in r["root"]]
    eval_records = [r for r in records if r["kind"] == "normal" and "/B/1022_56_106_" in r["root"]]
    train = [load_episode(r, args.samples) for r in train_records]
    evaluation = [load_episode(r, args.samples) for r in eval_records]
    validate_split(train, evaluation)
    stats = fit_action_statistics(train)
    record = train_records[0]
    start = args.frame
    batch = make_batch(train, [(record["id"], start)])
    np.savez(args.output / "batch.npz", **{k: v for k, v in batch.items() if isinstance(v, np.ndarray)})
    image_provenance = []
    info = record["info"]
    episode = batch["provenance"][0]["episode_index"]
    for role, key in [
        ("base_0_rgb", "observation.images.cam0"),
        ("left_wrist_0_rgb", "observation.images.cam2"),
        ("right_wrist_0_rgb", "observation.images.cam1"),
    ]:
        remote_path = (
            record["root"]
            + "/"
            + info["video_path"].format(
                episode_chunk=episode // info["chunks_size"], episode_index=episode, video_key=key
            )
        )
        command = (
            "/mnt/data/train_platform/environments/pi05/20261014/bin/python -c "
            + shlex.quote(REMOTE_FRAME)
            + " "
            + shlex.quote(remote_path)
            + " "
            + str(start)
        )
        raw = subprocess.check_output(["ssh", "-o", "BatchMode=yes", record["host"], command])
        metadata, png = raw.split(b"\n", 1)
        decoded = json.loads(metadata)
        if abs(decoded["time_seconds"] - start / info["fps"]) > 1 / info["fps"]:
            raise ValueError("Video frame timestamp differs from numeric current frame")
        Image.open(io.BytesIO(png)).verify()
        (args.output / (role + ".png")).write_bytes(png)
        image_provenance.append({"role": role, "host": record["host"], "path": remote_path, **decoded})
    task_text = subprocess.check_output(
        ["ssh", record["host"], "cat " + shlex.quote(record["root"] + "/meta/tasks.jsonl")], text=True
    )
    tasks = [json.loads(line) for line in task_text.splitlines() if line]
    # Dataset has one task; reject ambiguity instead of selecting a future task label.
    if len(tasks) != 1:
        raise ValueError("Specify per-frame task lookup for multi-task datasets")
    prompt = tasks[0]["task"]
    metadata = {
        "experiment": "flash_p1_standard_k6_offline_smoke",
        "normalization": stats.to_dict(),
        "batch_provenance": batch["provenance"],
        "images": image_provenance,
        "prompt": prompt,
        "train_ids": [r["id"] for r in train_records],
        "eval_ids": [r["id"] for r in eval_records],
        "split_scope": "mechanical smoke: single version and disjoint roots, not benchmark validation",
        "image_inputs": "real current-frame RGB, three mapped cameras; no future video frame",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps({"output": str(args.output), "sample": record["id"], "frame": start, "prompt": prompt}))


if __name__ == "__main__":
    main()
