"""Build a standalone subtask-only dataset and horizon-aware normalization."""

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from create_fr3_wuji_episode_subset import create_subset
from create_fr3_wuji_episode_subset import read_jsonl
from create_fr3_wuji_episode_subset import write_jsonl
import numpy as np
from prepare_fr3_wuji_20hz import compute_stats
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/descfly/data-clean/1001"))
    parser.add_argument("--output", type=Path, default=Path("/home/descfly/data-clean/1001_subtasks"))
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    selected = [e for e in episodes if e["task_index"] in (1, 2, 3, 4) and e["length"] >= 50]
    if not selected or any(not e["segment_id"].startswith("subtask_") for e in selected):
        raise ValueError("Unexpected subtask selection")
    tasks = {t["task_index"]: t["task"] for t in read_jsonl(source / "meta/tasks.jsonl")}
    # Contiguous task IDs in the new dataset; preserve original IDs in provenance.
    task_map = {old: new for new, old in enumerate(sorted({e["task_index"] for e in selected}))}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".1001-subtasks-", dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        create_subset(source, root, [e["episode_index"] for e in selected])
        info = json.loads((root / "meta/info.json").read_text())
        stats = read_jsonl(root / "meta/episodes_stats.jsonl")
        new_episodes = read_jsonl(root / "meta/episodes.jsonl")
        for e, original, stat in zip(new_episodes, selected, stats, strict=True):
            e["source_episode_index"] = original["episode_index"]
            e["source_task_index"] = original["task_index"]
            e["task_index"] = task_map[original["task_index"]]
            path = root / info["data_path"].format(
                episode_chunk=e["episode_index"] // info["chunks_size"], episode_index=e["episode_index"]
            )
            table = pq.read_table(path)
            pos = table.schema.get_field_index("task_index")
            table = table.set_column(
                pos, "task_index", pa.array([e["task_index"]] * len(table), type=table.schema.field("task_index").type)
            )
            pq.write_table(table, path)
            # The generic subset copier retains scalar stats from the original IDs.
            for key in ("episode_index", "index", "task_index"):
                values = np.asarray(table[key], dtype=np.float64)
                stat["stats"][key] = {
                    "min": [float(values.min())],
                    "max": [float(values.max())],
                    "mean": [float(values.mean())],
                    "std": [float(values.std())],
                    "count": [len(values)],
                }
        info["total_tasks"] = len(task_map)
        (root / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        write_jsonl(
            root / "meta/tasks.jsonl", [{"task_index": new, "task": tasks[old]} for old, new in task_map.items()]
        )
        write_jsonl(root / "meta/episodes.jsonl", new_episodes)
        write_jsonl(root / "meta/episodes_stats.jsonl", stats)
        compute_stats((root,), root, 256, action_hz=30.0, rewrite_prompt=False)
        provenance = json.loads((root / "provenance.json").read_text())
        provenance["datasets"] = [str(output)]
        provenance["selection"] = {
            "source": str(source),
            "original_task_indices": [1, 2, 3, 4],
            "min_episode_frames": 50,
            "original_to_new_task_index": task_map,
            "excluded_full_episodes": sum(e["task_index"] == 0 for e in episodes),
            "excluded_short_subtask_episodes": sum(e["task_index"] in task_map and e["length"] < 50 for e in episodes),
            "episodes_per_original_task": dict(Counter(e["task_index"] for e in selected)),
        }
        (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        root.rename(output)
    print(f"Prepared {output}: {len(selected)} episodes, {sum(e['length'] for e in selected)} frames")


if __name__ == "__main__":
    main()
