"""Opt-in, fresh pi05_base FullFT smoke run (1..20 updates), never P3 resume.

Uses the frozen normal-data manifest/split/normalization and causal dataset API.
This is a capacity/gradient validation entry, not the multi-GPU production loop.
"""

import argparse
import dataclasses
import json
from pathlib import Path
import time

from flax import nnx
import jax
import jax.numpy as jnp
from train_p2 import load_model
from train_p2 import optimizer_only
import wandb

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_checkpoint import save_offline_checkpoint
from openpi.training.flash_dataset import FlashLeRobotDataset
from openpi.training.flash_train import create_optimizer
from openpi.training.flash_train import make_train_step
from openpi.training.optimizer import AdamW
from openpi.training.optimizer import CosineDecaySchedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True, help="Original pi05_base params, not P2/P3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional NEW full-model + Adam checkpoint; large")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20:
        raise ValueError("FullFT smoke is limited to 1..20 updates")
    if args.output.exists() or (args.checkpoint is not None and args.checkpoint.exists()):
        raise FileExistsError("Use new output/checkpoint directories")
    previous = args.pretrained.parent / "metadata.json"
    if previous.exists():
        prior_config = json.loads(previous.read_text()).get("config", {})
        if "poly_degree" in prior_config or "freeze_vlm" in prior_config:
            raise ValueError("Fresh FullFT requires pi05_base, not a FLASH diagnostic checkpoint")
    dataset = FlashLeRobotDataset(args.data, args.images, PaligemmaTokenizer(max_len=256))
    plan = dataset.training_plan(args.steps, 42001)
    args.output.mkdir(parents=True)
    schedule = CosineDecaySchedule()
    metadata = {
        "config_name": "pi05_flash_k6_stride1_fullft_smoke",
        "dataset_sha256": dataset.manifest["sha256"],
        "split_sha256": dataset.split["sha256"],
        "normalization": dataset.normalization,
        "normalization_snapshot_sha256": dataset.normalization_snapshot_sha256,
        "window_plan": plan,
        "model_seed": 42,
        "sampler_seed": 42001,
        "flow_and_augmentation_seed": 42002,
        "rng_policy": "fold_in(key(42002), zero_based_update); split inside loss_components",
        "optimizer": dataclasses.asdict(AdamW()),
        "schedule": dataclasses.asdict(schedule),
        "microbatch_size": 1,
        "effective_batch_size": 1,
        "gradient_accumulation": 1,
        "checkpoint_kind": "flash_full_model",
        "step": 0,
        "steps": [],
        "prompts": [],
        "pretrained": str(args.pretrained.resolve()),
        "status": "initializing",
    }
    (args.output / "result.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    run = wandb.init(
        project="openpi-flash",
        name=metadata["config_name"],
        mode="offline",
        dir=str(args.output.resolve()),
        config={k: v for k, v in metadata.items() if k not in {"steps", "prompts", "normalization"}},
    )
    try:
        model, config, audit = load_model(dataset.normalization, args.pretrained, freeze_vlm=False)
        metadata["config"] = dataclasses.asdict(config)
        metadata["weight_loading"] = audit
        metadata["wandb_directory"] = run.dir
        optimizer = create_optimizer(model, config, schedule.create())
        update = make_train_step(config)
        for i, index in enumerate(plan):
            observation, batch, prompt = dataset.observation(index)
            start = time.perf_counter()
            metrics = update(
                optimizer,
                observation,
                history=jnp.asarray(batch["history"]),
                target_ext=jnp.asarray(batch["target_ext"]),
                rng=jax.random.fold_in(jax.random.key(42002), i),
            )
            jax.block_until_ready(nnx.state(optimizer))
            row = {key: float(value) for key, value in metrics.items()}
            row.update(
                step=int(optimizer.step.value),
                seconds=time.perf_counter() - start,
                learning_rate=float(schedule.create()(i)),
            )
            memory = jax.devices()[0].memory_stats() or {}
            row["peak_bytes_in_use"] = memory.get("peak_bytes_in_use")
            metadata["steps"].append(row)
            metadata["step"] = row["step"]
            metadata["prompts"].append(prompt)
            (args.output / "result.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(row), flush=True)
            run.log(row, step=row["step"])
        if args.checkpoint is not None:
            save_offline_checkpoint(
                args.checkpoint,
                jax.device_get(nnx.state(model).to_pure_dict()),
                metadata=metadata,
                optimizer_state=jax.device_get(optimizer_only(optimizer).to_pure_dict()),
            )
        metadata["status"] = "completed"
    except Exception as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (args.output / "result.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        run.finish()


if __name__ == "__main__":
    main()
