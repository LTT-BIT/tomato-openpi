"""Freeze the <=100-step sampler plan and cache only current-frame RGB images."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from openpi.training.flash_dataset import FlashLeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42001)
    args = parser.parse_args()
    dataset = FlashLeRobotDataset(args.data, args.images)
    plan = {
        "train": dataset.training_plan(args.steps, args.seed),
        "validation": dataset.validation_plan(),
        "fixed": dataset.split["fixed_batch"],
        "seed": args.seed,
        "microbatch_size": 1,
        "effective_batch_size": 1,
        "split_sha256": dataset.split["sha256"],
        "normalization_sha256": dataset.normalization["statistics_sha256"],
    }
    plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    path = args.data / "window_plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Do not replace a frozen sampling plan")
    path.write_text(json.dumps(plan, indent=2))
    start = time.perf_counter()
    dataset.prefetch_images([plan["fixed"], *plan["train"], *plan["validation"]])
    print(
        json.dumps(
            {
                "train_windows": len(plan["train"]),
                "validation_windows": len(plan["validation"]),
                "seconds": time.perf_counter() - start,
                "plan_sha256": plan["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
