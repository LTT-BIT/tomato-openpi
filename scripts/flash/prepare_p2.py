"""Freeze a complete normal-only LeRobot inventory, source-group split and numeric cache.

Remote access is read-only. All published episodes are included; the P0 samples
are not used as a surrogate complete dataset. No images or robot actions sent.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from collect_samples import RemoteFile
from collect_samples import ssh_json
import numpy as np
import pyarrow.parquet as pq

from openpi.training.flash_data import fit_action_statistics
from openpi.training.flash_data import load_episode


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_split(records, seed, forced_validation, forced_train):
    groups = sorted({r["source_recording_id"] for r in records})
    if not all(groups) or set(forced_validation) & set(forced_train):
        raise ValueError("Invalid source groups")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(groups))
    validation = set(shuffled[: max(1, round(len(groups) * 0.2))]) | set(forced_validation)
    validation -= set(forced_train)
    train = set(groups) - validation
    if not train or not validation:
        raise ValueError("Empty split")
    assignments = {r["id"]: ("validation" if r["source_recording_id"] in validation else "train") for r in records}
    return {
        "seed": seed,
        "method": "source_recording_id group shuffle 20%, predeclared challenge validation and fixed-batch train",
        "forced_validation_sources": forced_validation,
        "forced_train_sources": forced_train,
        "train_sources": sorted(train),
        "validation_sources": sorted(validation),
        "assignments": assignments,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Completed dataset snapshot exists; choose new output")
    inventory_path = args.output / "remote_inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
    else:
        code = """import os,json
from pathlib import Path
base=Path("/mnt/data/train_platform/datasets/tomato_sharpa/normal")
roots=[]
for folder,dirs,files in os.walk(base):
 dirs[:]=sorted(x for x in dirs if not x.startswith('.') and x not in ('videos','ingest','production','conversion'))
 if Path(folder).name=='meta' and 'info.json' in files:
  root=Path(folder).parent; info=json.loads((root/'meta/info.json').read_text())
  tasks=[json.loads(s) for s in (root/'meta/tasks.jsonl').read_text().splitlines() if s]
  episodes=[json.loads(s) for s in (root/'meta/episodes.jsonl').read_text().splitlines() if s]
  entries=[]
  for ep in episodes:
   i=ep['episode_index']; path=root/info['data_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i)
   if not path.is_file(): raise RuntimeError(str(path))
   entries.append(dict(meta=ep,path=str(path),size=path.stat().st_size))
  roots.append(dict(root=str(root),info=info,tasks=tasks,episodes=entries))
print(json.dumps(roots))"""
        inventory = ssh_json("28", code)
        inventory_path.write_text(json.dumps(inventory, indent=2))
    records = []
    for ds in inventory:
        if ds["info"]["fps"] != 30:
            raise ValueError("P2 requires 30Hz normal")
        for entry in ds["episodes"]:
            ep = entry["meta"]
            if not ep.get("source_recording_id"):
                raise ValueError("Missing original source ID")
            path = entry["path"]
            ident = hashlib.sha256(("28" + path).encode()).hexdigest()[:16]
            records.append(
                {
                    "id": ident,
                    "host": "28",
                    "kind": "normal",
                    "root": ds["root"],
                    "source_path": path,
                    "source_size": entry["size"],
                    "local_file": ident + ".parquet",
                    "rows": ep["length"],
                    "episode_index": ep["episode_index"],
                    "source_recording_id": ep["source_recording_id"],
                    "source_start_ns": ep.get("source_start_ns"),
                    "source_end_ns": ep.get("source_end_ns"),
                    "info": ds["info"],
                    "tasks": ds["tasks"],
                }
            )
    print(
        "complete inventory",
        len(inventory),
        "roots",
        len(records),
        "episodes",
        sum(r["rows"] for r in records),
        "frames",
        flush=True,
    )

    def fetch(record):
        target = args.output / record["local_file"]
        if not target.exists():
            remote = RemoteFile(record["host"], record["source_path"], record["source_size"])
            try:
                table = pq.read_table(
                    remote,
                    columns=[
                        "action",
                        "observation.state",
                        "timestamp",
                        "frame_index",
                        "episode_index",
                        "task_index",
                        "source_timestamp_ns",
                    ],
                )
            finally:
                remote.close()
            if len(table) != record["rows"]:
                raise ValueError("Published length changed")
            pq.write_table(table, target)
        return {**record, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, records))
    # Declare challenge sources before training, using exact source identities.
    fixed_id = "8fc6041fb6830400"
    challenge_ids = ["d8f720863e569786", "606140b9fb30c1e5"]
    baseline_validation = next(r["id"] for r in records if "/B/1022_56_106_" in r["root"] and r["episode_index"] == 0)
    by_id = {r["id"]: r for r in records}
    for ident in [fixed_id, *challenge_ids, baseline_validation]:
        if ident not in by_id:
            raise ValueError("Predeclared episode absent from frozen snapshot")
    split = source_split(
        records,
        args.seed,
        [by_id[i]["source_recording_id"] for i in [*challenge_ids, baseline_validation]],
        [by_id[fixed_id]["source_recording_id"]],
    )
    split["fixed_batch"] = {"id": fixed_id, "start": 302}
    split["challenge_windows"] = [{"id": challenge_ids[0], "start": 5952}, {"id": challenge_ids[1], "start": 352}]
    split["complete_validation_episode"] = baseline_validation
    split["sha256"] = digest(split)
    train_episodes = [load_episode(r, args.output) for r in records if split["assignments"][r["id"]] == "train"]
    stats = fit_action_statistics(train_episodes).to_dict()
    del train_episodes
    manifest = {
        "roots": [{"root": d["root"], "info": d["info"], "tasks": d["tasks"]} for d in inventory],
        "records": [{k: v for k, v in r.items() if k not in ("info", "tasks")} for r in records],
        "source": "ssh://28/mnt/data/train_platform/datasets/tomato_sharpa/normal",
        "stride": 1,
        "K": 6,
        "scope": "complete published normal inventory snapshot; bounded 100-step experiment, not full training",
    }
    manifest["sha256"] = digest(manifest)
    (args.output / "split.json").write_text(json.dumps(split, indent=2, ensure_ascii=False))
    (args.output / "normalization.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "episodes": len(records),
                "frames": sum(r["rows"] for r in records),
                "train_sources": len(split["train_sources"]),
                "validation_sources": len(split["validation_sources"]),
                "train_rows_for_norm": stats["train_rows"],
                "split_sha256": split["sha256"],
                "normalization_sha256": stats["statistics_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
