"""Bounded normal-only FLASH experiment: <=20 fixed-batch + <=100 fresh updates.

P1 mathematics is reused unchanged (K6, stride1). No robot/RTC/Delta interfaces.
Only one final full checkpoint is saved. Stop on failed fixed-batch gate/nonfinite.
"""

import argparse
import dataclasses
from functools import partial
import gc
import hashlib
import json
from pathlib import Path
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import wandb

from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.training.flash_checkpoint import FlashCheckpointWeightLoader
from openpi.training.flash_checkpoint import is_flash_trainable_parameter
from openpi.training.flash_checkpoint import is_vlm_parameter
from openpi.training.flash_checkpoint import parameter_digest
from openpi.training.flash_checkpoint import restore_offline_checkpoint
from openpi.training.flash_checkpoint import save_offline_checkpoint
from openpi.training.flash_dataset import FlashLeRobotDataset
from openpi.training.optimizer import AdamW


def sync(value):
    return jax.block_until_ready(value)


def summary(error):
    error = np.asarray(error)
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "p95": float(np.quantile(absolute, 0.95)),
        "p99": float(np.quantile(absolute, 0.99)),
        "max": float(absolute.max()),
    }


def content_digest(tree):
    h = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        array = np.asarray(leaf)
        h.update(str(path).encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        h.update(memoryview(np.ascontiguousarray(array).view(np.uint8)))
    return h.hexdigest()


def optimizer_only(optimizer):
    return nnx.state(optimizer, lambda path, value: path[0] != "model")


def load_model(normalization, pretrained, *, freeze_vlm=True):
    config = Pi0FlashConfig(
        freeze_vlm=freeze_vlm, action_q01=tuple(normalization["q01"]), action_q99=tuple(normalization["q99"])
    )
    abstract = nnx.eval_shape(config.create, jax.random.key(42))
    graph, state = nnx.split(abstract)
    reference = state.to_pure_dict()
    rngs = nnx.Rngs(42)
    for name, inputs, outputs in [("action_in_proj", 58, 1024), ("action_out_proj", 1024, 58)]:
        reference[name] = jax.device_get(nnx.state(nnx.Linear(inputs, outputs, rngs=rngs)).to_pure_dict())
    loader = FlashCheckpointWeightLoader(str(pretrained))
    params = loader.load(reference)
    state.replace_by_pure_dict(jax.device_put(params))
    model = nnx.merge(graph, state)
    del params, reference, abstract, state
    gc.collect()
    return model, config, loader.report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=Path("/home/user/checkpoints/pi05_base/params"))
    parser.add_argument("--replay-result", type=Path, default=None)
    parser.add_argument("--fixed-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.fixed_steps <= 20 or not 1 <= args.steps <= 100:
        raise ValueError("P2 hard update caps: fixed<=20, fresh<=100")
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Use new output/checkpoint directories")
    args.output.mkdir(parents=True)
    dataset = FlashLeRobotDataset(args.data, args.images, PaligemmaTokenizer(max_len=256))
    plan = json.loads((args.data / "window_plan.json").read_text())
    if len(plan["train"]) != args.steps or plan["split_sha256"] != dataset.split["sha256"]:
        raise ValueError("Sampling plan mismatch")
    config_name = "pi05_flash_k6_stride1_normal_p2_frozen_vlm"
    optimizer_config = AdamW()
    learning_rate = 2.5e-5
    tx = optimizer_config.create(learning_rate)
    run = wandb.init(
        project="openpi-flash",
        name=config_name,
        mode="offline",
        dir=str(args.output.resolve()),
        config={
            "optimizer": dataclasses.asdict(optimizer_config),
            "learning_rate": learning_rate,
            "schedule": "constant",
            "microbatch": 1,
            "effective_batch": 1,
            "gradient_accumulation": 1,
            "split_sha256": dataset.split["sha256"],
            "normalization_sha256": dataset.normalization["statistics_sha256"],
            "seed": 42,
            "sampler_seed": plan["seed"],
            "fixed_steps": args.fixed_steps,
            "fresh_train_steps": args.steps,
            "K": 6,
            "stride": 1,
        },
    )
    run.define_metric("train/*", step_metric="train/step")
    run.define_metric("fixed/*", step_metric="fixed/step")
    run.define_metric("validation/*", step_metric="validation/optimizer_step")
    report = {
        "config_name": config_name,
        "optimizer": dataclasses.asdict(optimizer_config),
        "learning_rate": learning_rate,
        "schedule_reason": "main peak LR 2.5e-5 and main AdamW settings; no 1000-step warmup in a capped 100-step diagnostic",
        "microbatch_size": 1,
        "effective_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "seed": 42,
        "sampler_seed": plan["seed"],
        "model_seed": 42,
        "flow_seed": 42002,
        "split_sha256": dataset.split["sha256"],
        "dataset_sha256": dataset.manifest["sha256"],
        "normalization_sha256": dataset.normalization["statistics_sha256"],
        "wandb": {"mode": "offline", "id": run.id, "directory": run.dir},
        "fixed": [],
        "train": [],
        "validation_curve": [],
    }

    def flush():
        (args.output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def log(entry):
        with (args.output / "steps.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    model, config, weight_report = load_model(dataset.normalization, args.pretrained)
    report["configuration"] = dataclasses.asdict(config)
    report["weight_loading"] = weight_report
    report["initial_expert_sha256"] = parameter_digest(nnx.state(model, is_flash_trainable_parameter).to_pure_dict())
    vlm_before = parameter_digest(nnx.state(model, nnx.All(nnx.Param, is_vlm_parameter)).to_pure_dict(), vlm_only=True)
    initial_expert = jax.device_get(nnx.state(model, is_flash_trainable_parameter).to_pure_dict())
    initial_expert_digest = parameter_digest(initial_expert)
    optimizer = nnx.Optimizer(model, tx, wrt=is_flash_trainable_parameter)
    encode = nnx.jit(lambda m, o: m.encode_condition(o), donate_argnums=(0,))
    diff = nnx.DiffState(0, is_flash_trainable_parameter)

    @partial(nnx.jit, donate_argnums=(0,))
    def update(opt, mask, cache, history, target, rng):
        def objective(m):
            values = m.loss_from_cache(rng, mask, cache, history=history, target_ext=target, train=True)
            return jnp.mean(values["loss"]), {k: jnp.mean(v) for k, v in values.items()}

        (_, metrics), grads = nnx.value_and_grad(objective, argnums=diff, has_aux=True)(opt.model)
        norm = jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in jax.tree.leaves(grads)))
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]))
        opt.update(grads)
        return metrics, norm, finite

    @partial(nnx.jit, donate_argnums=(0,), static_argnames=("train",))
    def loss(m, mask, cache, history, target, rng, *, train=False):
        return {
            k: jnp.mean(v)
            for k, v in m.loss_from_cache(rng, mask, cache, history=history, target_ext=target, train=train).items()
        }

    @partial(nnx.jit, donate_argnums=(0,))
    def predict(m, mask, cache, history):
        source = m.polynomial.fit_history(m.normalize(history))
        coeff = source + m.coefficient_velocity(source, jnp.zeros((history.shape[0],)), mask, cache)
        return (
            m.denormalize(m.polynomial.reconstruct(coeff, extended=True)),
            m.polynomial.velocity(coeff, extended=True)
            * (jnp.asarray(m.action_q99) - jnp.asarray(m.action_q01) + 1e-6)
            / 2,
        )

    groups = {
        "all": list(range(58)),
        "arms": [i for i, n in enumerate(ORDER) if "fr3" in n],
        "hands": [i for i, n in enumerate(ORDER) if "fr3" not in n],
    }
    print("encode held-out fixed observations; cache valid because VLM frozen and augmentation disabled", flush=True)
    validation = []
    for index in plan["validation"]:
        obs, batch, prompt = dataset.observation(index)
        mask, cache = sync(encode(optimizer.model, obs))
        normalized = optimizer.model.normalize(jnp.asarray(batch["target_ext"]))
        oracle = optimizer.model.denormalize(
            optimizer.model.polynomial.reconstruct(optimizer.model.polynomial.fit_target(normalized))
        )
        validation.append((index, batch, mask, cache, np.asarray(oracle)))

    def evaluate(step, *, save_predictions=False):
        errors = []
        oracle_errors = []
        values = []
        extended = []
        velocities = []
        detail = []
        for i, (index, batch, mask, cache, oracle) in enumerate(validation):
            metrics = sync(
                loss(
                    optimizer.model,
                    mask,
                    cache,
                    jnp.asarray(batch["history"]),
                    jnp.asarray(batch["target_ext"]),
                    jax.random.key(10000 + i),
                )
            )
            q, v = sync(predict(optimizer.model, mask, cache, jnp.asarray(batch["history"])))
            q = np.asarray(q)
            v = np.asarray(v)
            if not np.isfinite(q).all() or not np.isfinite(v).all():
                raise FloatingPointError("Nonfinite held-out prediction")
            error = q[:, 2:52] - batch["actions"]
            errors.append(error)
            oracle_errors.append(oracle - batch["actions"])
            values.append({k: float(value) for k, value in metrics.items()})
            extended.append(q)
            velocities.append(v)
            detail.append(
                {
                    **index,
                    "source_recording_id": dataset.records[index["id"]]["source_recording_id"],
                    "groups": {g: summary(error[..., ids]) for g, ids in groups.items()},
                }
            )
        all_errors = np.concatenate(errors)
        oracle_error = np.concatenate(oracle_errors)
        gaps = []
        velocity_gaps = []
        for i, (left, _, _, _, _) in enumerate(validation):
            for j, (right, _, _, _, _) in enumerate(validation):
                if left["id"] == right["id"] and right["start"] == left["start"] + 50:
                    gaps.append(extended[i][:, 52] - extended[j][:, 2])
                    velocity_gaps.append(velocities[i][:, 52] - velocities[j][:, 2])
        result = {
            "step": step,
            "windows": len(validation),
            "metrics": {k: float(np.mean([v[k] for v in values])) for k in values[0]},
            "physical": {g: summary(all_errors[..., ids]) for g, ids in groups.items()},
            "oracle_physical": {g: summary(oracle_error[..., ids]) for g, ids in groups.items()},
            "per_joint": {name: summary(all_errors[..., i]) for i, name in enumerate(ORDER)},
            "windows_detail": detail,
            "adjacent_pairs": len(gaps),
            "position_gap": {g: summary(np.concatenate(gaps)[..., ids]) for g, ids in groups.items()},
            "velocity_gap": {g: summary(np.concatenate(velocity_gaps)[..., ids]) for g, ids in groups.items()},
        }
        report["validation_curve"].append(result)
        if save_predictions:
            np.savez(
                args.output / f"validation_step_{step:03d}.npz",
                prediction=np.concatenate(extended),
                velocity=np.concatenate(velocities),
                truth=np.concatenate([b["actions"] for _, b, _, _, _ in validation]),
                oracle=np.concatenate([o for _, _, _, _, o in validation]),
            )
        run.log(
            {
                "validation/optimizer_step": step,
                "validation/total_loss": result["metrics"]["loss"],
                "validation/action_mae_rad": result["physical"]["all"]["mae"],
            },
        )
        flush()
        print("validation", step, result["metrics"], result["physical"]["all"], flush=True)
        return result

    evaluate(0, save_predictions=True)
    obs, fixed, prompt = dataset.observation(plan["fixed"])
    fixed_mask, fixed_cache = sync(encode(optimizer.model, obs))
    report["prompt"] = prompt
    report["augmentation"] = "disabled for frozen VLM and reproducible causal observation caching"
    fixed_key = jax.random.key(42002)
    initial_fixed = {
        k: float(v)
        for k, v in sync(
            loss(
                optimizer.model,
                fixed_mask,
                fixed_cache,
                jnp.asarray(fixed["history"]),
                jnp.asarray(fixed["target_ext"]),
                fixed_key,
                train=True,
            )
        ).items()
    }
    report["fixed_initial"] = initial_fixed
    previous = json.loads(args.replay_result.read_text()) if args.replay_result else None
    if previous is not None:
        if not previous.get("fixed_gate_passed") or len(previous["train"]) != args.steps:
            raise ValueError("Replay requires completed original bounded training and passed fixed gate")
        if (
            previous["split_sha256"] != report["split_sha256"]
            or previous["normalization_sha256"] != report["normalization_sha256"]
        ):
            raise ValueError("Replay data identity mismatch")
        report["fixed"] = previous["fixed"]
        report["fixed_initial"] = previous["fixed_initial"]
        report["replay"] = {
            "reason": "interruption during final checkpoint write",
            "original_result": str(args.replay_result),
            "additional_unique_steps": 0,
            "comparison": "same data/seed/config; numerical identity not guaranteed; use this run own before/after metrics",
            "matched_steps": 0,
            "maximum_loss_difference": 0.0,
        }
    print("fixed batch gate, same batch AND flow random draw across <=20 updates", flush=True)
    for step in range(1, (args.fixed_steps if previous is None else 0) + 1):
        start = time.perf_counter()
        metrics, norm, finite = sync(
            update(
                optimizer,
                fixed_mask,
                fixed_cache,
                jnp.asarray(fixed["history"]),
                jnp.asarray(fixed["target_ext"]),
                fixed_key,
            )
        )
        entry = {
            "stage": "fixed",
            "step": step,
            **{k: float(v) for k, v in metrics.items()},
            "gradient_norm": float(norm),
            "compute_seconds": time.perf_counter() - start,
            "peak_gpu_bytes": int(jax.devices()[0].memory_stats()["peak_bytes_in_use"]),
        }
        if not bool(finite) or not np.isfinite(list(entry.values())[2:]).all():
            raise FloatingPointError("Nonfinite fixed batch update")
        report["fixed"].append(entry)
        log(entry)
        run.log({"fixed/" + k: v for k, v in entry.items() if isinstance(v, int | float)})
        flush()
    if previous is not None:
        report["fixed_final"] = previous["fixed_final"]
        report["fixed_gate_passed"] = True
        report["fresh_stage_reset"] = {
            "expert_equal_initial": True,
            "optimizer_step_reset": int(optimizer.step.value),
            "continues_fixed_training": False,
        }
        del initial_expert
        gc.collect()
    else:
        report["fixed_final"] = {
            k: float(v)
            for k, v in sync(
                loss(
                    optimizer.model,
                    fixed_mask,
                    fixed_cache,
                    jnp.asarray(fixed["history"]),
                    jnp.asarray(fixed["target_ext"]),
                    fixed_key,
                    train=True,
                )
            ).items()
        }
        report["fixed_gate_passed"] = report["fixed_final"]["loss"] < report["fixed_initial"]["loss"]
        flush()
        if not report["fixed_gate_passed"]:
            run.finish()
            raise RuntimeError("Fixed batch loss did not decrease; stopping without 100-step stage")
        # Reset only Expert and Adam moments. Do not duplicate the ~12GB VLM checkpoint.
        for array in jax.tree.leaves(optimizer_only(optimizer)):
            array.delete()
        old_expert = nnx.state(optimizer.model, is_flash_trainable_parameter)
        fresh_state = nnx.state(optimizer.model, is_flash_trainable_parameter)
        fresh_state.replace_by_pure_dict(jax.device_put(initial_expert))
        nnx.update(optimizer.model, fresh_state)
        for array in jax.tree.leaves(old_expert):
            array.delete()
        model = optimizer.model
        optimizer = nnx.Optimizer(model, tx, wrt=is_flash_trainable_parameter)
        assert parameter_digest(nnx.state(model, is_flash_trainable_parameter).to_pure_dict()) == initial_expert_digest
        del initial_expert, old_expert, fresh_state
        gc.collect()
        report["fresh_stage_reset"] = {
            "expert_equal_initial": True,
            "optimizer_step_reset": int(optimizer.step.value),
            "continues_fixed_training": False,
        }
    print("fixed gate passed; fresh original Expert + new Adam state for <=100 steps", flush=True)
    for step, index in enumerate(plan["train"], 1):
        start = time.perf_counter()
        obs, batch, prompt = dataset.observation(index)
        mask, cache = sync(encode(optimizer.model, obs))
        encode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        metrics, norm, finite = sync(
            update(
                optimizer,
                mask,
                cache,
                jnp.asarray(batch["history"]),
                jnp.asarray(batch["target_ext"]),
                jax.random.fold_in(jax.random.key(42002), step),
            )
        )
        entry = {
            "stage": "fresh",
            "step": step,
            **{k: float(v) for k, v in metrics.items()},
            "gradient_norm": float(norm),
            "compute_seconds": time.perf_counter() - start,
            "data_and_prefix_seconds": encode_seconds,
            "peak_gpu_bytes": int(jax.devices()[0].memory_stats()["peak_bytes_in_use"]),
            **index,
        }
        if not bool(finite) or not np.isfinite([entry[k] for k in ("loss", "gradient_norm")]).all():
            raise FloatingPointError("Nonfinite fresh update")
        if previous is not None:
            reference = previous["train"][step - 1]
            if reference["id"] != index["id"] or reference["start"] != index["start"]:
                raise ValueError("Replay index changed")
            differences = [
                abs(entry[k] - reference[k]) for k in ("loss", "flow_matching_loss", "polynomial_consistency_loss")
            ]
            # Cross-process mixed-precision GPU execution is not assumed bitwise
            # deterministic. Data/config identity is enforced; numerical drift is
            # reported, never hidden or used to choose a better checkpoint.
            report["replay"]["matched_steps"] = step
            report["replay"]["maximum_loss_difference"] = max(report["replay"]["maximum_loss_difference"], *differences)
        report["train"].append(entry)
        log(entry)
        run.log({"train/" + k: v for k, v in entry.items() if isinstance(v, int | float)})
        flush()
        if step % 20 == 0 or step == args.steps:
            evaluate(step, save_predictions=step == args.steps)
    vlm_after = parameter_digest(
        nnx.state(optimizer.model, nnx.All(nnx.Param, is_vlm_parameter)).to_pure_dict(), vlm_only=True
    )
    assert vlm_before == vlm_after
    report["freeze_audit"] = {"before": vlm_before, "after": vlm_after, "unchanged": True}
    report["gpu_stats"] = {k: int(v) for k, v in jax.devices()[0].memory_stats().items() if isinstance(v, int)}
    # Warm synchronized timing, excludes file decoding/tokenization and uses full RGB prefix.
    index = plan["validation"][0]
    obs, batch, _ = dataset.observation(index)
    timings = []
    for repeat in range(6):
        start = time.perf_counter()
        mask, cache = sync(encode(optimizer.model, obs))
        sync(predict(optimizer.model, mask, cache, jnp.asarray(batch["history"])))
        if repeat:
            timings.append((time.perf_counter() - start) * 1000)
    report["inference_latency_ms"] = {
        "median": float(np.median(timings)),
        "min": min(timings),
        "max": max(timings),
        "repeats": 5,
        "scope": "3RGB prefix + NFE1 + reconstruction, not disk or tokenizer",
    }
    print("save ONE final full checkpoint plus Adam moments and split", flush=True)
    params_hash = parameter_digest(nnx.state(optimizer.model).to_pure_dict())
    optimizer_hash = content_digest(optimizer_only(optimizer).to_pure_dict())
    report["checkpoint_pre_save_hashes"] = {"parameters": params_hash, "optimizer": optimizer_hash}
    flush()
    save_offline_checkpoint(
        args.checkpoint,
        nnx.state(optimizer.model).to_pure_dict(),
        optimizer_state=optimizer_only(optimizer).to_pure_dict(),
        metadata={
            "config": dataclasses.asdict(config),
            "normalization": dataset.normalization,
            "step": args.steps,
            "optimizer": {"name": "AdamW", "lr": learning_rate, **dataclasses.asdict(optimizer_config)},
            "split": dataset.split,
            "dataset_manifest_sha256": dataset.manifest["sha256"],
            "window_plan": plan,
            "prompt": prompt,
        },
    )
    # Release old GPU state before restoring full parameters and moments.
    graph, template = nnx.split(optimizer)
    template = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), template)
    arrays = jax.tree.leaves(nnx.state(optimizer))
    optimizer = None
    model = None
    for array in arrays:
        array.delete()
    del arrays
    gc.collect()
    restored, meta, opt_state = restore_offline_checkpoint(args.checkpoint)
    assert parameter_digest(restored) == params_hash
    template.replace_by_pure_dict({"model": jax.device_put(restored), **jax.device_put(opt_state)})
    optimizer = nnx.merge(graph, template)
    # Orbax serializes integer dict keys as strings; hash after NNX restores its typed paths.
    assert content_digest(optimizer_only(optimizer).to_pure_dict()) == optimizer_hash
    assert int(optimizer.step.value) == args.steps
    obs, batch, _ = dataset.observation(plan["validation"][0])
    mask, cache = sync(encode(optimizer.model, obs))
    restored_prediction = np.asarray(sync(predict(optimizer.model, mask, cache, jnp.asarray(batch["history"])))[0])
    expected = np.load(args.output / f"validation_step_{args.steps:03d}.npz")["prediction"][0:1]
    if not np.isfinite(restored_prediction).all():
        raise FloatingPointError("Restored prediction is not finite")
    report["checkpoint"] = {
        "path": str(args.checkpoint.resolve()),
        "parameter_sha256": params_hash,
        "optimizer_sha256": optimizer_hash,
        "step": int(optimizer.step.value),
        "parameter_roundtrip": True,
        "adam_state_roundtrip": True,
        "restored_prediction_max_error": float(np.max(abs(restored_prediction - expected))),
        "restored_prediction_mae_difference": float(np.mean(abs(restored_prediction - expected))),
        "sampling_shape": list(restored_prediction[:, 2:52].shape),
        "sampling_finite": True,
        "cross_process_note": "exact state hashes checked; report forward arithmetic drift separately",
        "full_checkpoints_created": 1,
    }
    report["completed"] = True
    flush()
    run.summary["completed"] = True
    run.finish()
    print(
        "P2 complete",
        json.dumps(
            {
                "fixed_initial": report["fixed_initial"],
                "fixed_final": report["fixed_final"],
                "validation_initial": report["validation_curve"][0]["physical"]["all"],
                "validation_final": report["validation_curve"][-1]["physical"]["all"],
                "checkpoint": report["checkpoint"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
