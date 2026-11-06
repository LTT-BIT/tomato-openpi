"""Prepare paired A/B whole-episode LeRobot copies after coarse review is complete.

Input plan: {review_complete: true, episodes: [{dataset, episode_index,
annotation, recording_group}]}. recording_group identifies the ORIGINAL recording,
shared by all normal/speedup variants; never derive it from the output episode ID.
No source writes, video cuts, fabricated holds, state/action changes or training.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from openpi.subtasks.tomato import GOAL
from openpi.subtasks.tomato import PHASES
from openpi.subtasks.tomato import PROCEDURE
from openpi.subtasks.tomato import PROMPTS
from openpi.subtasks.tomato import SOURCE_ACTION_NAMES
from openpi.subtasks.tomato import SOURCE_STATE_NAMES
from openpi.subtasks.tomato import audit_token_budget
from openpi.subtasks.tomato import reviewed_segments


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def grouped_split(records, seed=42):
    """Adapted from lemon grouped_split: same original recording, same split."""
    groups = sorted({r["recording_group"] for r in records})
    if len(groups) < 10:
        raise ValueError("Need at least ten independent reviewed recordings for 80/10/10 evaluation")
    random.Random(seed).shuffle(groups)
    held = max(1, round(len(groups) * 0.1))
    assignment = {group: "test" if i < held else "val" if i < 2 * held else "train" for i, group in enumerate(groups)}
    return {
        name: [i for i, r in enumerate(records) if assignment[r["recording_group"]] == name]
        for name in ("train", "val", "test")
    }


def safe_file(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Source paths must stay inside the selected dataset")
    return path


def inspect_plan(plan):
    if plan.get("review_complete") is not True or not plan.get("episodes"):
        raise ValueError("Finish review of all intended data before preparing the unified training bundle")
    records, seen, source_groups = [], set(), {}
    for item in plan["episodes"]:
        root = Path(item["dataset"]).expanduser().resolve(strict=True)
        episode = item["episode_index"]
        group = item["recording_group"]
        if type(episode) is not int or episode < 0 or not isinstance(group, str) or not group.strip():
            raise ValueError("Expected an episode index and explicit original recording_group")
        if (str(root), episode) in seen:
            raise ValueError("Duplicate source episode")
        seen.add((str(root), episode))
        info = json.loads((root / "meta/info.json").read_text())
        if info.get("codebase_version") != "v2.1" or info.get("fps") != 30:
            raise ValueError("Expected 30 Hz LeRobot v2.1")
        if info["features"]["observation.state"]["shape"] != [116] or info["features"]["action"]["shape"] != [58]:
            raise ValueError("Expected converter Sharpa 116D state / 58D action")
        if info["features"]["observation.state"].get("names") != list(SOURCE_STATE_NAMES) or info["features"][
            "action"
        ].get("names") != list(SOURCE_ACTION_NAMES):
            raise ValueError("Sharpa source joint names/order mismatch")
        entries = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
        entry = next(row for row in entries if row["episode_index"] == episode)
        annotation_path = Path(item["annotation"]).expanduser().resolve(strict=True)
        doc = json.loads(annotation_path.read_text())
        rows = reviewed_segments(doc, entry["length"])
        identity = doc["source"]
        if (
            identity.get("episode_index") != episode
            or identity.get("length") != entry["length"]
            or identity.get("fps") != info["fps"]
            or identity.get("source_recording_id") != entry.get("source_recording_id")
        ):
            raise ValueError("Annotation/source episode identity mismatch")
        if entry.get("source_recording_id") is None:
            raise ValueError("Missing source_recording_id; restore original-recording provenance before splitting")
        # A namespace must be a stable acquisition identity, never normal/speedup.
        source = (str(item.get("recording_namespace", "sharpa")), str(entry["source_recording_id"]))
        if source in source_groups and source_groups[source] != group:
            raise ValueError("Variants of one recording assigned to different groups")
        source_groups[source] = group
        hashes = identity.get("sha256", {})
        if not hashes:
            raise ValueError("Reviewed annotation must carry source SHA256 provenance")
        for relative, expected in hashes.items():
            if digest(safe_file(root, relative)) != expected:
                raise ValueError(f"Source changed since review: {relative}")
        path = safe_file(root, info["data_path"].format(episode_chunk=episode // 1000, episode_index=episode))
        table = pq.read_table(path)
        if (
            len(table) != entry["length"]
            or table["frame_index"].to_pylist() != list(range(len(table)))
            or set(table["episode_index"].to_pylist()) != {episode}
        ):
            raise ValueError("Invalid source episode/frame indexing")
        if info["data_path"].format(episode_chunk=episode // 1000, episode_index=episode) not in hashes:
            raise ValueError("Reviewed provenance must include source Parquet")
        records.append(
            {
                "root": root,
                "episode": episode,
                "recording_group": group,
                "info": info,
                "entry": entry,
                "table": table,
                "segments": rows,
                "annotation": doc,
                "annotation_sha256": digest(annotation_path),
                "source_sha256": hashes,
            }
        )
    reference = records[0]["info"]
    if any(r["info"]["features"] != reference["features"] for r in records):
        raise ValueError("Cannot combine incompatible feature schemas")
    return records


def stats(arrays):
    values = np.concatenate(arrays)
    return {
        **{
            key: fn(values, axis=0).tolist()
            for key, fn in (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std))
        },
        "count": [len(values)],
    }


def prepare(plan, output, tokenizer):
    """Publish both variants atomically; each retains full original episodes."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Choose a new bundle directory; never overwrite existing data")
    budget = audit_token_budget(tokenizer)
    records = inspect_plan(plan)
    splits = grouped_split(records)
    cameras = [key for key, value in records[0]["info"]["features"].items() if value.get("dtype") == "video"]
    if set(cameras) != {f"observation.images.cam{i}" for i in range(3)}:
        raise ValueError("Expected exactly three Sharpa RGB cameras")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tomato-subtasks-", dir=output.parent))
    try:
        mapping = []
        for mode in ("a_goal", "b_coarse"):
            dest = temporary / mode
            texts = [GOAL] if mode == "a_goal" else list(dict.fromkeys(PROMPTS))
            tasks = {text: i for i, text in enumerate(texts)}
            episodes, episode_stats, offset = [], [], 0
            normalization = {key: [] for key in ("observation.state", "action")}
            for new_id, record in enumerate(records):
                source, table = record["root"], record["table"]
                phase_ids = np.empty(len(table), dtype=np.int64)
                for phase, segment in enumerate(record["segments"]):
                    phase_ids[segment["start_frame"] : segment["end_frame"]] = phase
                prompt_ids = [0] * len(table) if mode == "a_goal" else [tasks[PROMPTS[i]] for i in phase_ids]
                updates = {
                    "episode_index": [new_id] * len(table),
                    "index": list(range(offset, offset + len(table))),
                    "task_index": prompt_ids,
                    "coarse_phase_index": phase_ids,
                }
                for key, value in updates.items():
                    column = pa.array(
                        value, type=table.schema.field(key).type if key in table.column_names else pa.int64()
                    )
                    table = (
                        table.set_column(table.schema.get_field_index(key), key, column)
                        if key in table.column_names
                        else table.append_column(key, column)
                    )
                chunk = new_id // 1000
                relative = f"data/chunk-{chunk:03d}/episode_{new_id:06d}.parquet"
                (dest / relative).parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, dest / relative)
                for camera in cameras:
                    old = record["info"]["video_path"].format(
                        episode_chunk=record["episode"] // 1000, episode_index=record["episode"], video_key=camera
                    )
                    if old not in record["source_sha256"]:
                        raise ValueError("Reviewed provenance must cover every RGB video")
                    video = safe_file(source, old)
                    target = dest / f"videos/chunk-{chunk:03d}/{camera}/episode_{new_id:06d}.mp4"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(video, target)
                    if digest(target) != record["source_sha256"][old]:
                        raise ValueError("Video changed during copy")
                per_episode = {}
                for key, samples in normalization.items():
                    array = np.asarray(table[key].to_pylist(), dtype=np.float64)
                    if not np.isfinite(array).all():
                        raise ValueError("Nonfinite source state/action")
                    samples.append(array)
                    per_episode[key] = stats([array])
                episode_stats.append({"episode_index": new_id, "stats": per_episode})
                entry = {
                    **record["entry"],
                    "episode_index": new_id,
                    "tasks": list(dict.fromkeys(texts[i] for i in prompt_ids)),
                }
                entry.pop("task_index", None)
                episodes.append(entry)
                offset += len(table)
                write_json(dest / f"annotations/episode_{new_id:06d}.json", record["annotation"])
                if mode == "a_goal":
                    mapping.append(
                        {
                            "episode_index": new_id,
                            "source_dataset": str(source),
                            "source_episode_index": record["episode"],
                            "recording_group": record["recording_group"],
                            "annotation_sha256": record["annotation_sha256"],
                            "source_sha256": record["source_sha256"],
                        }
                    )
            info = {
                **records[0]["info"],
                "total_episodes": len(records),
                "total_frames": offset,
                "total_tasks": len(texts),
                "total_videos": len(records) * 3,
                "total_chunks": (len(records) + 999) // 1000,
                "splits": {"train": f"0:{len(records)}"},
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {
                    **records[0]["info"]["features"],
                    "coarse_phase_index": {"dtype": "int64", "shape": [1], "names": None},
                },
            }
            write_json(dest / "meta/info.json", info)
            write_json(dest / "meta/stats.json", {k: stats(v) for k, v in normalization.items()})
            write_jsonl(dest / "meta/episodes.jsonl", episodes)
            write_jsonl(dest / "meta/episodes_stats.jsonl", episode_stats)
            write_jsonl(dest / "meta/tasks.jsonl", [{"task_index": i, "task": text} for i, text in enumerate(texts)])
        # Verify all source hashes again before publishing. Source files remain untouched.
        for record in records:
            for relative, expected in record["source_sha256"].items():
                if digest(safe_file(record["root"], relative)) != expected:
                    raise ValueError("Source changed during preparation")
        write_json(temporary / "splits.json", {"splits": splits, "seed": 42, "grouping": "original recording"})
        write_json(
            temporary / "bundle.json",
            {
                "schema_version": 1,
                "review_complete": True,
                "phases": PHASES,
                "goal": GOAL,
                "procedure": PROCEDURE,
                "prompts": PROMPTS,
                "token_budget": budget,
                "mapping": mapping,
                "variants": ["a_goal", "b_coarse"],
                "episodes_unchanged_in_length": True,
            },
        )
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    args = parser.parse_args()
    import sentencepiece

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    print(prepare(json.loads(args.plan.read_text()), args.output, tokenizer))


if __name__ == "__main__":
    main()
