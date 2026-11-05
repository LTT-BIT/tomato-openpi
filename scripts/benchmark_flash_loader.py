"""Read-only CPU benchmark: identical windows, native versus prefetched FLASH IO."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import jax
import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_loader import PrefetchedBatches
from openpi.training.flash_native import NativeDataset
from scripts.run_training import load_experiment


def fingerprint(batch):
    return [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in jax.tree.leaves(batch)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--start-step", type=int, default=2800)
    args = parser.parse_args()
    if args.batches < 2:
        parser.error("At least two batches for startup/steady-state comparison")
    config = replace(
        load_experiment("training_configs.experiments.tomato_sharpa.flash.tomato_fr3_sharpa_flash_fullft_v1"),
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    dataset = NativeDataset(config, tokenizer=PaligemmaTokenizer(max_len=config.model.max_token_len))
    report = {
        "status": "running",
        "batch_size": config.batch_size,
        "workers": config.num_workers,
        "start_step": args.start_step,
        "batches": args.batches,
        "formal_updates": 0,
        "native_seconds": [],
        "prefetched_seconds": [],
        "exact_batch_hashes": [],
        "scope": "CPU IO benchmark concurrent with existing training; no GPU or checkpoint changes",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    originals = []
    try:
        for step in range(args.start_step, args.start_step + args.batches):
            indices = [dataset.index(step * config.batch_size + i) for i in range(config.batch_size)]
            before = time.perf_counter()
            batch = dataset.batch(indices)
            report["native_seconds"].append(time.perf_counter() - before)
            originals.append(fingerprint(batch))
            del batch
            save()
        before = time.perf_counter()
        with PrefetchedBatches(dataset, args.start_step, args.start_step + args.batches) as batches:
            for i, batch in enumerate(batches):
                report["prefetched_seconds"].append(time.perf_counter() - before)
                actual = fingerprint(batch)
                assert actual == originals[i], f"Batch {i} bytes changed"
                report["exact_batch_hashes"].append(actual)
                del batch
                save()
                before = time.perf_counter()
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
