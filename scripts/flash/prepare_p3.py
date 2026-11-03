"""Freeze P3's exact 150 updates and a disjoint source-balanced uniform evaluation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from plan_p3 import extend_plan
from prepare_p25 import digest

from openpi.training.flash_dataset import CAMERAS
from openpi.training.flash_dataset import FlashLeRobotDataset


def uniform_plan(dataset, fixed, seed=20261104, per_source=5):
    rng = np.random.default_rng(seed)
    excluded = {(w["id"], w["start"]) for w in fixed["windows"]}
    windows = []
    for ident, r in sorted(dataset.records.items()):
        if dataset.split["assignments"][ident] != "validation":
            continue
        legal = np.array([t for t in range(7, r["rows"] - 51) if (ident, t) not in excluded])
        windows.extend(
            {
                "id": ident,
                "start": int(start),
                "source_recording_id": r["source_recording_id"],
                "root": r["root"],
                "episode_index": r["episode_index"],
                "strata": ["uniform_within_source"],
            }
            for start in sorted(rng.choice(legal, per_source, replace=False))
        )
    plan = {
        **{
            k: fixed[k]
            for k in (
                "dataset_sha256",
                "split_sha256",
                "normalization_sha256",
                "normalization_snapshot_sha256",
                "K",
                "stride",
            )
        },
        "windows": windows,
        "anchors": [],
        "seed": seed,
        "source_count": len({w["source_recording_id"] for w in windows}),
        "method": "5 uniformly random legal current frames per validation source, without replacement, excluding frozen P2.5 windows; same sources, not an independent test split",
        "training_updates_at_freeze": 0,
    }
    plan["sha256"] = digest(plan)
    return plan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefetch", action="store_true")
    args = p.parse_args()
    d = FlashLeRobotDataset(args.data, args.images)
    candidate = json.loads(Path("docs/flash/results/p25/p3_candidate_plan.json").read_text())
    assert candidate["sha256"] == digest({k: v for k, v in candidate.items() if k != "sha256"})
    original = json.loads((args.data / "window_plan.json").read_text())
    assert extend_plan(d, original) == {k: v for k, v in candidate.items() if k != "sha256"}
    fixed = json.loads(Path("docs/flash/results/p25/evaluation_plan.json").read_text())
    uniform = uniform_plan(d, fixed)
    train = {
        "windows": candidate["windows"][100:250],
        "global_steps": [101, 250],
        "effective_samples": 150,
        "candidate_sha256": candidate["sha256"],
        "split_sha256": d.split["sha256"],
        "normalization_sha256": d.normalization["statistics_sha256"],
        "flow_seed": candidate["flow_seed"],
        "next_flow_key_data": candidate["next_flow_key_data"],
        "frozen_fixed_evaluation_sha256": fixed["sha256"],
        "uniform_evaluation_sha256": uniform["sha256"],
    }
    train["sha256"] = digest(train)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, value in [("train_plan", train), ("uniform_plan", uniform)]:
        path = args.output / (name + ".json")
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            path.write_text(json.dumps(value, indent=2))
    print(
        json.dumps(
            {
                "updates": 150,
                "uniform_windows": len(uniform["windows"]),
                "uniform_sources": uniform["source_count"],
                "train_sha256": train["sha256"],
            }
        ),
        flush=True,
    )
    if args.prefetch:
        jobs = sorted({(w["id"], w["start"], role) for w in train["windows"] + uniform["windows"] for role in CAMERAS})
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i, _ in enumerate(pool.map(lambda j: d.image({"id": j[0], "start": j[1]}, j[2]), jobs), 1):
                if i % 100 == 0:
                    print("cached", i, "/", len(jobs), "elapsed", round(time.perf_counter() - start, 1), flush=True)
        print("complete", flush=True)


if __name__ == "__main__":
    main()
