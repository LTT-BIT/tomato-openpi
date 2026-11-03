"""Read-only extended validation: all frozen held-out sources, paired replans."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.training.flash_dataset import CAMERAS
from openpi.training.flash_dataset import FlashLeRobotDataset


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def anchors_for_episode(states):
    """Select GT-stratified diagnostic anchors, before observing any predictions."""
    if len(states) < 160:
        raise ValueError("Need >=160 frames for two diagnostic anchors and +50 replans")
    hands = [i for i, name in enumerate(ORDER) if "fr3" not in name]
    starts = np.arange(7, len(states) - 101, 5)
    speed = abs(np.diff(states[:, hands], axis=0)) * 30
    scores = np.array([np.max(speed[t : t + 49]) for t in starts])
    transient = int(starts[np.argmax(scores)])
    eligible = abs(starts - transient) >= 50
    smooth = int(starts[np.flatnonzero(eligible)[np.argmin(scores[eligible])]])
    return [("smooth", smooth), ("transient", transient)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefetch", action="store_true")
    args = p.parse_args()
    d = FlashLeRobotDataset(args.data, args.images)
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        assert digest({k: v for k, v in plan.items() if k != "sha256"}) == plan["sha256"]
        assert plan["split_sha256"] == d.split["sha256"]
    else:
        windows, anchors, index = [], [], {}

        def add(ident, start, label):
            key = (ident, start)
            if key not in index:
                d.window({"id": ident, "start": start})
                r = d.records[ident]
                index[key] = len(windows)
                windows.append(
                    {
                        "id": ident,
                        "start": start,
                        "source_recording_id": r["source_recording_id"],
                        "root": r["root"],
                        "episode_index": r["episode_index"],
                        "strata": [],
                    }
                )
            i = index[key]
            if label not in windows[i]["strata"]:
                windows[i]["strata"].append(label)
            return i

        for ident, r in sorted(d.records.items()):
            if d.split["assignments"][ident] != "validation":
                continue
            candidates = anchors_for_episode(d.episode(ident)["states"])
            candidates += [
                ("p2_known_transient", item["start"]) for item in d.split["challenge_windows"] if item["id"] == ident
            ]
            for label, start in candidates:
                if start + 101 >= r["rows"]:
                    raise ValueError("Replan pair goes beyond episode")
                ids = {str(delta): add(ident, start + delta, label) for delta in (0, 10, 15, 50)}
                anchors.append({"id": ident, "start": start, "stratum": label, "window_indices": ids})
        plan = {
            "windows": windows,
            "anchors": anchors,
            "seed": 42,
            "method": "all held-out sources; measured-hand peak-speed min/max anchors, grid5, paired +10/+15/+50; known P2 transients added",
            "dataset_sha256": d.manifest["sha256"],
            "split_sha256": d.split["sha256"],
            "normalization_sha256": d.normalization["statistics_sha256"],
            "normalization_snapshot_sha256": d.normalization_snapshot_sha256,
            "source_count": len({w["source_recording_id"] for w in windows}),
            "K": 6,
            "stride": 1,
            "training_updates": 0,
        }
        plan["sha256"] = digest(plan)
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "sources": plan["source_count"],
                "windows": len(plan["windows"]),
                "anchors": len(plan["anchors"]),
                "sha256": plan["sha256"],
            }
        ),
        flush=True,
    )
    if args.prefetch:
        jobs = sorted({(w["id"], w["start"], role) for w in plan["windows"] for role in CAMERAS})
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=6) as pool:
            for count, _ in enumerate(pool.map(lambda job: d.image({"id": job[0], "start": job[1]}, job[2]), jobs), 1):
                if count % 100 == 0:
                    print(f"RGB cached {count}/{len(jobs)}, elapsed {time.perf_counter() - start:.1f}s", flush=True)
        print("prefetch complete", time.perf_counter() - start, flush=True)


if __name__ == "__main__":
    main()
