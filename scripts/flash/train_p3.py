"""Resume true Step100 Adam state and execute ONLY 150 updates, through Step250."""

import argparse
import dataclasses
from functools import partial
import gc
import json
import os
from pathlib import Path
import time

from evaluate_p25 import file_inventory
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from prepare_p25 import digest
from train_p2 import content_digest
from train_p2 import optimizer_only
import wandb

from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_checkpoint import is_flash_trainable_parameter
from openpi.training.flash_checkpoint import is_vlm_parameter
from openpi.training.flash_checkpoint import parameter_digest
from openpi.training.flash_checkpoint import restore_offline_checkpoint
from openpi.training.flash_checkpoint import save_offline_checkpoint
from openpi.training.flash_dataset import FlashLeRobotDataset
from openpi.training.optimizer import AdamW


def flow_key(step):
    return jax.random.fold_in(jax.random.key(42002, impl="threefry2x32"), step)


def random_audit(step):
    key = flow_key(step)
    # Exact path used by Pi0Flash.loss_from_cache: split into noise/time keys.
    noise_key, time_key = jax.random.split(key)
    noise = np.asarray(0.5 * jax.random.normal(noise_key, (1, 7, 58)))
    time_value = np.asarray(jax.random.uniform(time_key, (1,)))
    restored = jax.random.wrap_key_data(jax.random.key_data(key), impl="threefry2x32")
    n2, t2 = jax.random.split(restored)
    np.testing.assert_array_equal(noise, np.asarray(0.5 * jax.random.normal(n2, (1, 7, 58))))
    np.testing.assert_array_equal(time_value, np.asarray(jax.random.uniform(t2, (1,))))
    return {
        "step": step,
        "key_data": np.asarray(jax.random.key_data(key)).tolist(),
        "noise_sha256": content_digest(noise),
        "time": time_value.tolist(),
        "shape": [1, 7, 58],
        "source_noise_std": 0.5,
        "roundtrip": True,
    }


def restore_optimizer(checkpoint, reference):
    params, meta, moments = restore_offline_checkpoint(checkpoint)
    expected = reference["checkpoint_pre_save_hashes"]
    assert parameter_digest(params) == expected["parameters"]
    assert meta["step"] == 100
    assert meta["optimizer"]["lr"] == 2.5e-5
    assert {k: meta["optimizer"][k] for k in dataclasses.asdict(AdamW())} == dataclasses.asdict(AdamW())
    cfg = Pi0FlashConfig(
        **{
            **meta["config"],
            "action_q01": tuple(meta["config"]["action_q01"]),
            "action_q99": tuple(meta["config"]["action_q99"]),
        }
    )
    tx = AdamW().create(2.5e-5)
    abstract = nnx.eval_shape(
        lambda key: nnx.Optimizer(cfg.create(key), tx, wrt=is_flash_trainable_parameter), jax.random.key(42)
    )
    graph, state = nnx.split(abstract)
    state.replace_by_pure_dict({"model": params, **moments})
    opt = nnx.merge(graph, state)
    assert content_digest(optimizer_only(opt).to_pure_dict()) == expected["optimizer"]
    assert int(opt.step.value) == 100
    scalar_counts = {
        str(path): int(x)
        for path, x in jax.tree_util.tree_flatten_with_path(optimizer_only(opt).to_pure_dict())[0]
        if np.asarray(x).shape == () and np.issubdtype(np.asarray(x).dtype, np.integer)
    }
    assert scalar_counts
    assert all(x == 100 for x in scalar_counts.values())
    state.replace_by_pure_dict(jax.device_put(state.to_pure_dict()))
    opt = nnx.merge(graph, state)
    del params, moments, state, abstract
    gc.collect()
    return opt, meta, cfg, scalar_counts


def make_update():
    diff = nnx.DiffState(0, is_flash_trainable_parameter)

    @partial(nnx.jit, donate_argnums=(0,))
    def update(opt, mask, cache, history, target, key):
        def objective(m):
            values = m.loss_from_cache(key, mask, cache, history=history, target_ext=target, train=True)
            return jnp.mean(values["loss"]), {k: jnp.mean(v) for k, v in values.items()}

        (_, values), grads = nnx.value_and_grad(objective, argnums=diff, has_aux=True)(opt.model)
        norm = jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in jax.tree.leaves(grads)))
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]))
        opt.update(grads)
        return values, norm, finite

    return update


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path(".cache/flash_p2_checkpoint_100_recovered"))
    args = p.parse_args()
    if args.output.exists() or args.checkpoint.exists():
        raise FileExistsError("Never repeat updates or overwrite checkpoints; use saved state and inspect journal")
    args.output.mkdir(parents=True)
    reference = json.loads(Path("docs/flash/results/p2/training_result.json").read_text())
    candidate = json.loads(Path("docs/flash/results/p25/p3_candidate_plan.json").read_text())
    assert candidate["sha256"] == digest({k: v for k, v in candidate.items() if k != "sha256"})
    plan = json.loads(args.plan.read_text())
    assert digest({k: v for k, v in plan.items() if k != "sha256"}) == plan["sha256"]
    assert plan["windows"] == candidate["windows"][100:250]
    assert plan["global_steps"] == [101, 250]
    assert len(plan["windows"]) == 150
    d = FlashLeRobotDataset(args.data, args.images, PaligemmaTokenizer(max_len=256))
    assert plan["candidate_sha256"] == candidate["sha256"]
    assert d.manifest["sha256"] == reference["dataset_sha256"]
    assert plan["split_sha256"] == d.split["sha256"]
    assert plan["normalization_sha256"] == d.normalization["statistics_sha256"]
    source_inventory = file_inventory(args.source)
    opt, meta, config, counts = restore_optimizer(args.source, reference)
    assert meta["split"] == d.split
    assert meta["normalization"] == d.normalization
    assert meta["window_plan"]["train"] == candidate["windows"][:100]
    vlm_before = parameter_digest(
        nnx.state(opt.model, nnx.All(nnx.Param, is_vlm_parameter)).to_pure_dict(), vlm_only=True
    )
    assert vlm_before == reference["freeze_audit"]["after"]
    prng = [random_audit(s) for s in (100, 101, 250, 251)]
    assert prng[1]["key_data"] == candidate["next_flow_key_data"]
    run = wandb.init(
        project="openpi-flash",
        name="pi05_flash_k6_stride1_p3_step100_to250",
        mode="offline",
        dir=str(args.output.resolve()),
        config={
            "start_step": 100,
            "end_step": 250,
            "new_updates": 150,
            "lr": 2.5e-5,
            "batch": 1,
            "K": 6,
            "stride": 1,
            "frozen_vlm": True,
            "split_sha256": d.split["sha256"],
            "normalization_sha256": d.normalization["statistics_sha256"],
            "plan_sha256": plan["sha256"],
        },
    )
    run.define_metric("train/*", step_metric="train/global_step")
    run.define_metric("probe/*", step_metric="probe/global_step")
    report = {
        "config_name": "pi05_flash_k6_stride1_p3_step100_to250",
        "start_step": 100,
        "end_step": 250,
        "effective_samples": 0,
        "new_optimizer_updates": 0,
        "learning_rate": 2.5e-5,
        "optimizer": meta["optimizer"],
        "microbatch_size": 1,
        "effective_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "configuration": dataclasses.asdict(config),
        "source_checkpoint": str(args.source.resolve()),
        "source_checkpoint_hashes": reference["checkpoint_pre_save_hashes"],
        "source_inventory_sha256": digest(source_inventory),
        "source_scalar_counts": counts,
        "dataset_sha256": d.manifest["sha256"],
        "split_sha256": d.split["sha256"],
        "normalization_sha256": d.normalization["statistics_sha256"],
        "normalization_snapshot_sha256": d.normalization_snapshot_sha256,
        "train_plan_sha256": plan["sha256"],
        "p3_candidate_plan_sha256": candidate["sha256"],
        "flow_random_audit": prng,
        "train": [],
        "probes": [],
        "wandb": {"id": run.id, "mode": "offline", "directory": run.dir},
        "stop_policy": "nonfinite loss/gradient stops immediately; preclip norm>1000 for3 consecutive updates stops; no automatic replay",
    }

    def flush():
        path = args.output / "result.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(path)

    encode = nnx.jit(lambda m, o: m.encode_condition(o), donate_argnums=(0,))

    @partial(nnx.jit, donate_argnums=(0,))
    def loss(m, mask, cache, history, target, key):
        return {
            k: jnp.mean(v)
            for k, v in m.loss_from_cache(key, mask, cache, history=history, target_ext=target, train=False).items()
        }

    fixed_plan = json.loads(Path("docs/flash/results/p25/evaluation_plan.json").read_text())
    probes = []
    for label, indices in [("train_fixed16", plan["windows"][:16]), ("validation_fixed16", fixed_plan["windows"][:16])]:
        for i, index in enumerate(indices):
            obs, batch, prompt = d.observation(index)
            mask, cache = jax.block_until_ready(encode(opt.model, obs))
            probes.append((label, i, index, batch, mask, cache))

    def probe(step):
        values = {}
        for label, i, _, batch, mask, cache in probes:
            v = jax.block_until_ready(
                loss(
                    opt.model,
                    mask,
                    cache,
                    jnp.asarray(batch["history"]),
                    jnp.asarray(batch["target_ext"]),
                    jax.random.key(80000 + i),
                )
            )
            values.setdefault(label, []).append({k: float(x) for k, x in v.items()})
        entry = {
            "global_step": step,
            **{label: {k: float(np.mean([row[k] for row in rows])) for k in rows[0]} for label, rows in values.items()},
        }
        if not np.isfinite([v for row in entry.values() if isinstance(row, dict) for v in row.values()]).all():
            raise FloatingPointError("Nonfinite probe")
        report["probes"].append(entry)
        flush()
        run.log(
            {
                "probe/global_step": step,
                **{f"probe/{label}/{k}": v for label, rows in values.items() for k, v in entry[label].items()},
            }
        )
        print("probe", entry, flush=True)

    report["probe_indices"] = [{"label": label, "i": i, **index} for label, i, index, _, _, _ in probes]
    flush()
    probe(100)
    update = make_update()
    abnormal = 0
    for step, index in enumerate(plan["windows"], 101):
        assert int(opt.step.value) == step - 1
        assert d.split["assignments"][index["id"]] == "train"
        obs, batch, prompt = d.observation(index)
        before = time.perf_counter()
        mask, cache = jax.block_until_ready(encode(opt.model, obs))
        prefix_seconds = time.perf_counter() - before
        # Durable intent prevents silently replaying a possibly completed GPU update after interruption.
        with (args.output / "update_journal.jsonl").open("a") as f:
            f.write(json.dumps({"event": "intent", "step": step, **index}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        before = time.perf_counter()
        metrics, norm, finite = jax.block_until_ready(
            update(opt, mask, cache, jnp.asarray(batch["history"]), jnp.asarray(batch["target_ext"]), flow_key(step))
        )
        entry = {
            "global_step": step,
            **index,
            **{k: float(v) for k, v in metrics.items()},
            "gradient_norm": float(norm),
            "compute_seconds": time.perf_counter() - before,
            "prefix_seconds": prefix_seconds,
            "peak_gpu_bytes": int(jax.devices()[0].memory_stats()["peak_bytes_in_use"]),
            "effective_samples": step - 100,
        }
        report["train"].append(entry)
        report["new_optimizer_updates"] = step - 100
        report["effective_samples"] = step - 100
        flush()
        with (args.output / "update_journal.jsonl").open("a") as f:
            f.write(json.dumps({"event": "completed", **entry}) + "\n")
            f.flush()
            os.fsync(f.fileno())
        if (
            not bool(finite)
            or not np.isfinite(
                [entry[k] for k in ("loss", "flow_matching_loss", "polynomial_consistency_loss", "gradient_norm")]
            ).all()
        ):
            raise FloatingPointError("Nonfinite training state; do not repeat updates")
        abnormal = abnormal + 1 if float(norm) > 1000 else 0
        if abnormal >= 3:
            raise RuntimeError("Sustained abnormal preclip gradients; stopped")
        assert int(opt.step.value) == step
        run.log({"train/" + k: v for k, v in entry.items() if isinstance(v, int | float)})
        print(json.dumps(entry), flush=True)
        if step % 25 == 0:
            probe(step)
    assert int(opt.step.value) == 250
    assert len(report["train"]) == 150
    vlm_after = parameter_digest(
        nnx.state(opt.model, nnx.All(nnx.Param, is_vlm_parameter)).to_pure_dict(), vlm_only=True
    )
    assert vlm_after == vlm_before
    assert file_inventory(args.source) == source_inventory
    report["freeze_audit"] = {"before": vlm_before, "after": vlm_after, "unchanged": True}
    report["source_checkpoint_unchanged"] = True
    report["checkpoint_pre_save_hashes"] = {
        "parameters": parameter_digest(nnx.state(opt.model).to_pure_dict()),
        "optimizer": content_digest(optimizer_only(opt).to_pure_dict()),
    }
    report["prompt"] = prompt
    flush()
    save_offline_checkpoint(
        args.checkpoint,
        nnx.state(opt.model, is_flash_trainable_parameter).to_pure_dict(),
        optimizer_state=optimizer_only(opt).to_pure_dict(),
        metadata={
            **meta,
            "step": 250,
            "checkpoint_kind": "flash_expert_increment",
            "vlm_reference_checkpoint": str(args.source.resolve()),
            "vlm_sha256": vlm_after,
            "full_parameter_sha256": report["checkpoint_pre_save_hashes"]["parameters"],
            "parent_checkpoint": str(args.source.resolve()),
            "parent_hashes": reference["checkpoint_pre_save_hashes"],
            "flow_seed": 42002,
            "prng_implementation": "threefry2x32",
            "next_global_step": 251,
            "next_flow_key_data": random_audit(251)["key_data"],
            "training_candidate_plan": candidate,
            "new_updates": 150,
            "normalization_snapshot_sha256": d.normalization_snapshot_sha256,
            "p3_plan": plan,
            "p25_evaluation_sha256": fixed_plan["sha256"],
            "uniform_evaluation_sha256": plan["uniform_evaluation_sha256"],
            "prompt": prompt,
        },
    )
    report["checkpoint"] = {"path": str(args.checkpoint.resolve()), "step": 250, "saved": True, "restore_pending": True}
    report["training_completed"] = True
    flush()
    run.summary["training_completed"] = True
    run.finish()
    print("Step250 saved; stop, no update251", flush=True)


if __name__ == "__main__":
    main()
