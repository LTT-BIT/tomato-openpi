"""Explicit bounded GPU acceptance: four update computations, one full checkpoint.

Never starts the formal budget. Defaults to a new checkout-local verification
directory. The large temporary checkpoint is deleted only after restore passes.
"""

import argparse
from dataclasses import replace
import functools
import json
from pathlib import Path
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import checkpoints
from openpi.training import sharding
from openpi.training.flash_native import NativeDataset
from openpi.training.flash_native import preflight
from scripts.run_training import load_experiment
from scripts.train import init_train_state
from scripts.train_flash import evaluate_batch
from scripts.train_flash import step

MODULE = "training_configs.experiments.tomato_sharpa.flash.tomato_fr3_sharpa_flash_fullft_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-verification", action="store_true")
    parser.add_argument(
        "--capacity-only", action="store_true", help="One real update at the formal batch, no checkpoint"
    )
    args = parser.parse_args()
    from training_configs import common

    common.configure_environment()
    args.output = args.output.resolve()
    if not args.resume_verification:
        args.output.mkdir(parents=True, exist_ok=False)
    if args.capacity_only and args.resume_verification:
        parser.error("capacity-only cannot resume checkpoint verification")
    config = replace(load_experiment(MODULE), num_workers=8)
    if not args.capacity_only:
        config = replace(config, batch_size=8)
    report = {
        "purpose": "bounded_full_size_gpu_acceptance",
        "status": "running",
        "formal_updates": 0,
        "update_computations": 0,
        "data": preflight(config),
        "updates": [],
        "batch_size": config.batch_size,
        "capacity_only": args.capacity_only,
    }
    target = args.output / "result.json"
    if args.resume_verification:
        report = json.loads(target.read_text())
        assert report["status"] == "failed"
        assert report["data"] == preflight(config)
        report.setdefault("failure_history", []).append(report.pop("error", "unknown"))
        report["status"] = "running"
        report["resumed_from_completed_step"] = 2

    def flush():
        target.write_text(json.dumps(report, indent=2))

    flush()
    run = wandb.init(
        project="openpi-flash",
        name="flash_formal_backend_acceptance",
        mode="offline",
        dir=str(args.output.resolve()),
        config={"purpose": report["purpose"], "data": report["data"]},
    )
    manager = None
    try:
        assert jax.default_backend() == "gpu", "Real GPU backend required"
        assert jax.device_count() == 8
        assert config.batch_size % 8 == 0
        assert config.fsdp_devices == 8
        dataset = NativeDataset(config, tokenizer=PaligemmaTokenizer(max_len=config.model.max_token_len))
        mesh = sharding.make_mesh(8)
        data_layout = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        indices = [dataset.index(i) for i in range(config.batch_size)]
        batch = jax.device_put(dataset.batch(indices), data_layout)
        report["training_windows"] = indices
        report["batch_shapes"] = {"history": list(batch[1].shape), "target": list(batch[2].shape)}
        print("Real causal batch loaded", report["batch_shapes"], flush=True)
        state, layout = init_train_state(
            config, jax.random.key(config.model_seed), mesh, resume=args.resume_verification
        )
        if args.resume_verification:
            manager, _ = checkpoints.initialize_checkpoint_dir(
                args.output / "temporary_checkpoint", keep_period=None, overwrite=False, resume=True
            )
            state = checkpoints.restore_state(manager, state, dataset)
        jax.block_until_ready(state)
        report["parameter_count"] = sum(v.size for v in jax.tree.leaves(state.params))
        report["vlm_frozen"] = config.model.freeze_vlm
        update = jax.jit(
            functools.partial(step, config),
            in_shardings=(replicated, layout, data_layout),
            out_shardings=(layout, replicated),
        )

        def advance(current, function=update):
            begin = time.perf_counter()
            with sharding.set_mesh(mesh):
                new, metrics = function(jax.random.key(config.seed), current, batch)
            metrics = jax.device_get(metrics)
            assert bool(metrics["finite"])
            values = {k: float(v) for k, v in metrics.items() if k != "finite"}
            values.update(completed_step=int(new.step), seconds=time.perf_counter() - begin)
            report["update_computations"] += 1
            report["updates"].append(values)
            run.log(values, step=report["update_computations"])
            print(json.dumps(values), flush=True)
            flush()
            return new

        if args.capacity_only:
            state = advance(state)
            for name in ("vision_gradient_norm", "language_gradient_norm", "expert_gradient_norm"):
                assert report["updates"][-1][name] > 0, name
            report["device_memory"] = [d.memory_stats() for d in jax.devices()]
            report["status"] = "passed"
            return
        if not args.resume_verification:
            for _ in range(2):
                state = advance(state)
        for name in ("vision_gradient_norm", "language_gradient_norm", "expert_gradient_norm"):
            assert report["updates"][-1][name] > 0, name
        evaluator = jax.jit(
            functools.partial(evaluate_batch, state.model_def), in_shardings=(layout.params, data_layout, replicated)
        )
        val_indices = dataset.validation[: config.batch_size]
        val_batch = jax.device_put(dataset.batch(val_indices), data_layout)
        result = jax.device_get(evaluator(state.params, val_batch, jax.random.key(143)))
        assert result["prediction"].shape == (8, 50, 58)
        assert np.isfinite(result["prediction"]).all()
        report["sampling_shape"] = list(result["prediction"].shape)
        report["validation_prediction_mae"] = float(np.abs(result["prediction"] - result["truth"]).mean())
        report["validation_oracle_mae"] = float(np.abs(result["oracle"] - result["truth"]).mean())
        if not args.resume_verification:
            manager, _ = checkpoints.initialize_checkpoint_dir(
                args.output / "temporary_checkpoint", keep_period=None, overwrite=False, resume=False
            )
            begin = time.perf_counter()
            checkpoints.save_state(manager, state, dataset, 2)
            manager.wait_until_finished()
            report["checkpoint_seconds"] = time.perf_counter() - begin
            print("Full parameter/Adam checkpoint saved", flush=True)
        template, restored_layout = init_train_state(config, jax.random.key(config.model_seed), mesh, resume=True)
        restored = checkpoints.restore_state(manager, template, dataset)
        assert int(restored.step) == 2
        # Compare every parameter and optimizer leaf on devices, without giant host copies.
        maximum = 0.0
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
            maximum = max(maximum, float(jax.device_get(jnp.max(jnp.abs(a - b)))))
        report["restore_max_abs_difference"] = maximum
        assert maximum == 0
        expected = advance(state)
        restored_update = jax.jit(
            functools.partial(step, config),
            in_shardings=(replicated, restored_layout, data_layout),
            out_shardings=(restored_layout, replicated),
        )
        actual = advance(restored, restored_update)
        maximum = 0.0
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            maximum = max(maximum, float(jax.device_get(jnp.max(jnp.abs(a - b)))))
        report["next_update_max_abs_difference"] = maximum
        other = NativeDataset(config)
        report["sampler_restore_matches"] = all(
            dataset.index(2 * config.batch_size + i) == other.index(2 * config.batch_size + i)
            for i in range(config.batch_size)
        )
        assert maximum <= 1e-6
        assert report["sampler_restore_matches"]
        report["device_memory"] = [d.memory_stats() for d in jax.devices()]
        report["status"] = "passed"
        manager.close()
        manager = None
        shutil.rmtree(args.output / "temporary_checkpoint")
        report["temporary_checkpoint_removed"] = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if manager is not None:
            manager.close()
        flush()
        run.finish()


if __name__ == "__main__":
    main()
