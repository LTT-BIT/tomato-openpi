"""CPU-only production data pipeline benchmark; no VLM, weights or GPU allocation.

Run the same module/options twice, with and without --uncached. Requires matching
normalization assets, uses the real transforms, sampler and persistent workers.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

# Must precede imports that initialize JAX, including in spawned workers.
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch

from openpi.training import data_loader
from openpi.training.mamba_training import model_data_config
from scripts.run_training import resolve_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--uncached", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args, overrides = parser.parse_known_args()
    if args.batches < 2 or args.warmup < 1:
        parser.error("--batches must be >=2 and --warmup >=1")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    config = resolve_experiment(args.experiment, overrides)
    effective = model_data_config(config)
    if args.uncached:
        effective = replace(effective, video_cache=None)
    source = data_loader.create_torch_dataset(
        effective, config.model.action_horizon, config.model, sampling_mode=config.sampling_mode
    )
    dataset = data_loader.transform_dataset(source, effective)
    count = args.warmup + args.batches
    loader = data_loader.TorchDataLoader(
        dataset,
        config.batch_size,
        num_workers=config.num_workers,
        num_batches=count,
        num_train_steps=max(config.num_train_steps, count),
        sampling_mode=config.sampling_mode,
        seed=config.data_seed if config.data_seed is not None else config.seed,
        framework="pytorch",
    )
    waits, first_digest = [], None
    iterator = iter(loader)
    for index in range(count):
        if index == args.warmup:
            warm_start = time.perf_counter()
        start = time.perf_counter()
        batch = next(iterator)
        waits.append(time.perf_counter() - start)
        if index == 0:
            # Compare deterministic first-batch inputs across cached/uncached runs.
            digest = hashlib.sha256()
            for key in sorted(batch):
                if isinstance(batch[key], dict):
                    for subkey in sorted(batch[key]):
                        digest.update(batch[key][subkey].numpy().tobytes())
                else:
                    digest.update(batch[key].numpy().tobytes())
            first_digest = digest.hexdigest()
    elapsed = time.perf_counter() - warm_start
    asset_dir = Path(config.data.assets.assets_dir or config.assets_dirs) / effective.asset_id
    norm_digests = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("norm_stats.json", "norm_metadata.json", "delta_paper_stats.npz", "delta_paper_manifest.json")
        if (path := asset_dir / name).is_file()
    }
    report = {
        "scope": "CPU data pipeline only; not GPU training throughput or capacity",
        "experiment_module": args.experiment,
        "config_name": config.name,
        "experiment_name": config.exp_name,
        "sources": [vars(spec) for spec in effective.lerobot_datasets],
        "prompt": config.policy_metadata.get("prompt_example"),
        "asset_id": effective.asset_id,
        "normalization_sha256": norm_digests,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "cache": vars(effective.video_cache) if effective.video_cache is not None else None,
        "persistent_workers": loader.torch_loader.persistent_workers,
        "prefetch_factor": loader.torch_loader.prefetch_factor,
        "warmup_batches": args.warmup,
        "measured_batches": args.batches,
        "cold_first_batch_s": waits[0],
        "warm_batch_p50_s": float(np.median(waits[args.warmup :])),
        "warm_batch_p95_s": float(np.percentile(waits[args.warmup :], 95)),
        "warm_samples_per_s": config.batch_size * args.batches / elapsed,
        "first_batch_sha256": first_digest,
        "waits_s": waits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
