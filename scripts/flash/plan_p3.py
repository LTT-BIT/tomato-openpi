"""Create a candidate Step100->1000 resume sidecar; NEVER updates model state."""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from prepare_p25 import digest

from openpi.training.flash_dataset import FlashLeRobotDataset


def extend_plan(dataset, original, target=1000):
    records = [r for r in dataset.records.values() if dataset.split["assignments"][r["id"]] == "train"]
    counts = np.array([max(0, r["rows"] - 58) for r in records])
    ends = np.cumsum(counts)
    rng = np.random.default_rng(original["seed"])
    used = rng.choice(int(ends[-1]), len(original["train"]), replace=False)

    def decode(value):
        episode = int(np.searchsorted(ends, value, side="right"))
        offset = int(value - (ends[episode - 1] if episode else 0))
        return {"id": records[episode]["id"], "start": offset + 7}

    assert [decode(v) for v in used] == original["train"]
    state_after_p2 = rng.bit_generator.state
    remaining = np.setdiff1d(np.arange(ends[-1]), used)
    new = rng.choice(remaining, target - len(used), replace=False)
    windows = original["train"] + [decode(v) for v in new]
    return {
        "windows": windows,
        "sampler_state_after_p2": state_after_p2,
        "sampler_policy": "retain P2 choice(N,100) exactly; extend by choice of unused flattened indices using next numpy RNG state",
        "sampler_seed": original["seed"],
        "flow_seed": 42002,
        "next_global_step": 101,
        "next_flow_key_data": np.asarray(jax.random.key_data(jax.random.fold_in(jax.random.key(42002), 101))).tolist(),
        "model_updates_executed": 0,
        "candidate_only": True,
        "milestones": [100, 250, 500, 1000],
        "learning_rate": 2.5e-5,
        "microbatch": 1,
        "effective_batch": 1,
        "K": 6,
        "stride": 1,
        "normalization_sha256": dataset.normalization["statistics_sha256"],
        "split_sha256": dataset.split["sha256"],
        "original_plan_sha256": original["sha256"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite candidate resume plan")
    d = FlashLeRobotDataset(args.data, ".cache/flash_p2_images")
    original = json.loads((args.data / "window_plan.json").read_text())
    candidate = extend_plan(d, original)
    candidate["sha256"] = digest(candidate)
    args.output.write_text(json.dumps(candidate, indent=2))
    print(candidate["sha256"])


if __name__ == "__main__":
    main()
