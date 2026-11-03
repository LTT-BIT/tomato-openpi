"""Zero-update FLASH P2.5 evaluator. Writes new predictions, never checkpoints."""

import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from prepare_p25 import digest
from train_p2 import content_digest
from train_p2 import load_model
from train_p2 import optimizer_only

from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_checkpoint import is_flash_trainable_parameter
from openpi.training.flash_checkpoint import is_vlm_parameter
from openpi.training.flash_checkpoint import parameter_digest
from openpi.training.flash_checkpoint import restore_offline_checkpoint
from openpi.training.flash_dataset import FlashLeRobotDataset
from openpi.training.optimizer import AdamW


def file_inventory(directory):
    return {
        str(p.relative_to(directory)): [p.stat().st_size, p.stat().st_mtime_ns]
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def restore_model(checkpoint, reference):
    inventory = file_inventory(checkpoint)
    params, metadata, moments = restore_offline_checkpoint(checkpoint)
    expected = reference["checkpoint_pre_save_hashes"]
    assert parameter_digest(params) == expected["parameters"]
    cfg = Pi0FlashConfig(
        **{
            **metadata["config"],
            "action_q01": tuple(metadata["config"]["action_q01"]),
            "action_q99": tuple(metadata["config"]["action_q99"]),
        }
    )
    tx = AdamW(**{k: metadata["optimizer"][k] for k in dataclasses.asdict(AdamW())}).create(metadata["optimizer"]["lr"])
    abstract = nnx.eval_shape(
        lambda key: nnx.Optimizer(cfg.create(key), tx, wrt=is_flash_trainable_parameter), jax.random.key(42)
    )
    graph, state = nnx.split(abstract)
    state.replace_by_pure_dict({"model": params, **moments})
    host_optimizer = nnx.merge(graph, state)
    assert content_digest(optimizer_only(host_optimizer).to_pure_dict()) == expected["optimizer"]
    assert int(host_optimizer.step.value) == metadata["step"] == 100
    model_graph, model_state = nnx.split(host_optimizer.model)
    model_state.replace_by_pure_dict(jax.device_put(params))
    model = nnx.merge(model_graph, model_state)
    audit = {
        "parameter_sha256": expected["parameters"],
        "optimizer_sha256": expected["optimizer"],
        "optimizer_step": 100,
        "parameter_roundtrip": True,
        "adam_state_roundtrip": True,
        "checkpoint_inventory_sha256": digest(inventory),
        "optimizer_updates": 0,
    }
    del params, metadata, moments, host_optimizer, abstract, state, model_state
    gc.collect()
    return model, audit, inventory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["step100", "initial"], required=True)
    p.add_argument("--checkpoint", type=Path, default=Path(".cache/flash_p2_checkpoint_100_recovered"))
    p.add_argument("--reference", type=Path, default=Path(".cache/flash_p2_recovery_02/result.json"))
    p.add_argument("--pretrained", type=Path, default=Path("/home/user/checkpoints/pi05_base/params"))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite previous evaluation")
    args.output.mkdir(parents=True)
    d = FlashLeRobotDataset(args.data, args.images, PaligemmaTokenizer(max_len=256))
    plan = json.loads(args.plan.read_text())
    assert digest({k: v for k, v in plan.items() if k != "sha256"}) == plan["sha256"]
    assert plan["split_sha256"] == d.split["sha256"]
    assert plan["dataset_sha256"] == d.manifest["sha256"]
    assert plan["normalization_snapshot_sha256"] == d.normalization_snapshot_sha256
    reference = json.loads(args.reference.read_text())
    if args.method == "step100":
        model, audit, inventory = restore_model(args.checkpoint, reference)
        assert tuple(model.action_q01) == tuple(d.normalization["q01"])
        assert tuple(model.action_q99) == tuple(d.normalization["q99"])
    else:
        model, _, weight_report = load_model(d.normalization, args.pretrained)
        expert_hash = parameter_digest(nnx.state(model, is_flash_trainable_parameter).to_pure_dict())
        assert expert_hash == reference["initial_expert_sha256"]
        audit = {
            "initial_expert_sha256": expert_hash,
            "exact_initial_parameter_reconstruction": True,
            "weight_loading": weight_report,
            "optimizer_updates": 0,
        }
    vlm_hash = parameter_digest(nnx.state(model, nnx.All(nnx.Param, is_vlm_parameter)).to_pure_dict(), vlm_only=True)
    assert vlm_hash == reference["freeze_audit"]["before"]
    audit["vlm_sha256"] = vlm_hash
    encode = nnx.jit(lambda m, o: m.encode_condition(o), donate_argnums=(0,))

    @nnx.jit
    def predict(m, mask, cache, history):
        source = m.polynomial.fit_history(m.normalize(history))
        coeff = source + m.coefficient_velocity(source, jnp.zeros((1,)), mask, cache)
        q = m.denormalize(m.polynomial.reconstruct(coeff, extended=True))
        v = (
            m.polynomial.velocity(coeff, extended=True)
            * (jnp.asarray(m.action_q99) - jnp.asarray(m.action_q01) + 1e-6)
            / 2
        )
        return q, v

    @nnx.jit
    def oracle(m, target):
        coeff = m.polynomial.fit_target(m.normalize(target))
        return (
            m.denormalize(m.polynomial.reconstruct(coeff, extended=True)),
            m.polynomial.velocity(coeff, extended=True)
            * (jnp.asarray(m.action_q99) - jnp.asarray(m.action_q01) + 1e-6)
            / 2,
        )

    windows = plan["windows"][: args.limit or None]
    # Counterfactuals use older measured history only, never future target commands.
    partners = {}
    for anchor in plan["anchors"]:
        base = anchor["window_indices"]["0"]
        for delta in (10, 15, 50):
            other = anchor["window_indices"][str(delta)]
            if other < len(windows):
                partners.setdefault(other, []).append((base, delta))
    predictions, velocities, truth, oracles, oracle_v, histories = [], [], [], [], [], []
    counterfactuals, counter_indices, timings, repeats = [], [], [], []
    prompts = set()
    observation_hashes = []
    started = time.perf_counter()
    for i, index in enumerate(windows):
        assert d.split["assignments"][index["id"]] == "validation"
        obs, batch, prompt = d.observation(index)
        prompts.add(prompt)
        observation_hashes.append(
            content_digest(
                {"observation": dataclasses.asdict(obs), "history": batch["history"], "target": batch["target_ext"]}
            )
        )
        tick = time.perf_counter()
        mask, cache = jax.block_until_ready(encode(model, obs))
        q, v = jax.block_until_ready(predict(model, mask, cache, jnp.asarray(batch["history"])))
        timings.append(time.perf_counter() - tick)
        oq, ov = jax.block_until_ready(oracle(model, jnp.asarray(batch["target_ext"])))
        q, v, oq, ov = map(np.asarray, (q, v, oq, ov))
        assert q.shape == (1, 54, 58)
        if not all(np.isfinite(x).all() for x in (q, v, oq, ov)):
            raise FloatingPointError("Nonfinite evaluation result")
        predictions.append(q[0])
        velocities.append(v[0])
        oracles.append(oq[0])
        oracle_v.append(ov[0])
        truth.append(batch["target_ext"][0])
        histories.append(batch["history"][0])
        for base, delta in partners.get(i, []):
            old_history = d.window(windows[base])["history"]
            cq, cv = jax.block_until_ready(predict(model, mask, cache, jnp.asarray(old_history)))
            counterfactuals.append(np.concatenate([np.asarray(cq), np.asarray(cv)], axis=0))
            counter_indices.append([base, i, delta])
        if i < 8:
            repeat_q = np.asarray(jax.block_until_ready(predict(model, mask, cache, jnp.asarray(batch["history"])))[0])
            fresh_mask, fresh_cache = jax.block_until_ready(encode(model, obs))
            fresh_q = np.asarray(
                jax.block_until_ready(predict(model, fresh_mask, fresh_cache, jnp.asarray(batch["history"])))[0]
            )
            repeats.append(
                {
                    "index": i,
                    "same_cache_max": float(abs(repeat_q - q).max()),
                    "fresh_prefix_max": float(abs(fresh_q - q).max()),
                }
            )
        if (i + 1) % 25 == 0:
            print(args.method, i + 1, "/", len(windows), "elapsed", round(time.perf_counter() - started, 1), flush=True)
    np.savez(
        args.output / "predictions.npz",
        prediction=np.stack(predictions),
        velocity=np.stack(velocities),
        truth=np.stack(truth),
        oracle=np.stack(oracles),
        oracle_velocity=np.stack(oracle_v),
        history=np.stack(histories),
        counterfactual=np.stack(counterfactuals) if counterfactuals else np.zeros((0, 2, 54, 58)),
        counter_indices=np.asarray(counter_indices, dtype=int).reshape(-1, 3),
    )
    audit.update(
        environment={
            "jax": jax.__version__,
            "numpy": np.__version__,
            "device": str(jax.devices()[0]),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            "transformer_dtype": "bfloat16",
            "parameter_dtype": "float32",
        },
        method=args.method,
        windows=len(windows),
        plan_sha256=plan["sha256"],
        input_sequence_sha256=digest(observation_hashes),
        input_hashes=observation_hashes,
        prompts=sorted(prompts),
        repeats=repeats,
        median_forward_seconds=float(np.median(timings[1:])),
        total_evaluation_seconds=time.perf_counter() - started,
        peak_bytes_in_use=int(jax.devices()[0].memory_stats()["peak_bytes_in_use"]),
        prediction_file_sha256=hashlib.sha256((args.output / "predictions.npz").read_bytes()).hexdigest(),
    )
    if args.method == "step100":
        assert file_inventory(args.checkpoint) == inventory
        audit["checkpoint_files_unchanged"] = True
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2))
    print("complete", args.method, len(windows), flush=True)


if __name__ == "__main__":
    main()
