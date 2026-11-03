"""Offline standard FLASH P1 acceptance on pretrained full pi05.

One synthetic gradient check and ONE real SGD update; no training loop, no robot.
Requires prepare_smoke_batch.py. Checkpoints stay in an explicit new local path.
"""

import argparse
import dataclasses
from functools import partial
import gc
import json
from pathlib import Path
import subprocess
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import wandb

from openpi.models import model as model_lib
from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_checkpoint import FlashCheckpointWeightLoader
from openpi.training.flash_checkpoint import is_flash_trainable_parameter
from openpi.training.flash_checkpoint import is_vlm_parameter
from openpi.training.flash_checkpoint import parameter_digest
from openpi.training.flash_checkpoint import restore_offline_checkpoint
from openpi.training.flash_checkpoint import save_offline_checkpoint


def sync(value):
    return jax.block_until_ready(value)


def finite_tree(tree):
    return bool(sync(jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))))


def milliseconds(fn, repeats=5):
    sync(fn())
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        sync(fn())
        values.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(values)), "min_ms": min(values), "max_ms": max(values), "repeats": repeats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=Path("/home/user/checkpoints/pi05_base/params"))
    args = parser.parse_args()
    if args.checkpoint.exists():
        raise FileExistsError("Choose a new checkpoint path")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.inputs / "metadata.json").read_text())
    batch = dict(np.load(args.inputs / "batch.npz"))
    for array in batch.values():
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite physical input batch")
    norm = metadata["normalization"]
    run = wandb.init(
        project="openpi-flash",
        name="flash_p1_standard_k6_offline_smoke",
        mode="offline",
        dir=str(args.output.resolve()),
        config={
            "normalization_sha256": norm["statistics_sha256"],
            "pretrained": str(args.pretrained),
            "prompt": metadata["prompt"],
            "sgd_steps": 1,
        },
    )
    config = Pi0FlashConfig(action_q01=tuple(norm["q01"]), action_q99=tuple(norm["q99"]))
    print("abstract model and strict pretrained loading", flush=True)
    abstract = nnx.eval_shape(config.create, jax.random.key(42))
    graph, params = nnx.split(abstract)
    reference = params.to_pure_dict()
    # Only dimension-changing projections are initialized; never allocate random VLM.
    rngs = nnx.Rngs(42)
    for name, inputs, outputs in [("action_in_proj", 58, 1024), ("action_out_proj", 1024, 58)]:
        reference[name] = jax.device_get(nnx.state(nnx.Linear(inputs, outputs, rngs=rngs)).to_pure_dict())
    loader = FlashCheckpointWeightLoader(str(args.pretrained))
    merged = loader.load(reference)
    print("weights verified", loader.report, flush=True)
    params.replace_by_pure_dict(jax.device_put(merged))
    model = nnx.merge(graph, params)
    del merged, reference, abstract, params
    gc.collect()
    vlm_before = parameter_digest(nnx.state(model, nnx.All(nnx.Param, is_vlm_parameter)).to_pure_dict(), vlm_only=True)
    assert vlm_before == loader.report["source_vlm_sha256"]
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)
    state = batch["state"][0]
    sq01 = np.array(norm["state_q01"])
    sq99 = np.array(norm["state_q99"])
    normalized_state = (state - sq01) / (sq99 - sq01 + 1e-6) * 2 - 1
    tokens, token_mask = tokenizer.tokenize(metadata["prompt"], normalized_state)
    images = {
        role: jnp.asarray(np.asarray(Image.open(args.inputs / (role + ".png")).convert("RGB")), dtype=jnp.float32)[None]
        / 127.5
        - 1
        for role in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    real_obs = model_lib.Observation(
        images=images,
        image_masks={k: jnp.ones((1,), bool) for k in images},
        state=jnp.asarray(normalized_state[None], jnp.float32),
        tokenized_prompt=jnp.asarray(tokens[None], jnp.int32),
        tokenized_prompt_mask=jnp.asarray(token_mask[None]),
    )
    real_obs = model_lib.preprocess_observation(None, real_obs, train=False)
    synthetic_obs = config.fake_obs(batch_size=1)
    real_history = jnp.asarray(batch["history"])
    real_target = jnp.asarray(batch["target_ext"])
    # Synthetic physical sequence is smooth and completely separate from real labels.
    lower = jnp.asarray(norm["q01"])
    span = jnp.asarray(norm["q99"]) - lower + 1e-6
    synthetic_history = lower[None, None, :] + span[None, None, :] * (
        0.5 + 0.1 * jnp.sin(jnp.arange(-7, 1)[None, :, None] / 9)
    )
    synthetic_target = lower[None, None, :] + span[None, None, :] * (
        0.5 + 0.1 * jnp.sin(jnp.arange(-2, 52)[None, :, None] / 9)
    )
    encode = nnx.jit(lambda m, o: m.encode_condition(o), donate_argnums=(0,))
    gradient_filter = nnx.DiffState(0, is_flash_trainable_parameter)

    @partial(nnx.jit, donate_argnums=(0,))
    def gradient(m, mask, cache, history, target, rng):
        def objective(module):
            components = module.loss_from_cache(rng, mask, cache, history=history, target_ext=target, train=True)
            return jnp.mean(components["loss"]), {k: jnp.mean(v) for k, v in components.items()}

        (_, metrics), grads = nnx.value_and_grad(objective, argnums=gradient_filter, has_aux=True)(m)
        return metrics, grads

    @partial(nnx.jit, donate_argnums=(0,))
    def apply_sgd(m, grads):
        trainable = nnx.state(m, is_flash_trainable_parameter)
        # Clipping is part of the mechanical optimizer step, not a new policy algorithm.
        norm = jnp.sqrt(sum(jnp.sum(x.astype(jnp.float32) ** 2) for x in jax.tree.leaves(grads)))
        factor = jnp.minimum(1.0, 1.0 / (norm + 1e-12))
        nnx.update(m, jax.tree.map(lambda p, g: p - 1e-5 * factor * g, trainable, grads))
        return norm

    sample = nnx.jit(lambda m, mask, cache, h: m.sample_from_cache(mask, cache, history=h), donate_argnums=(0,))
    end_to_end = nnx.jit(lambda m, o, h: m.sample_actions(jax.random.key(0), o, history=h), donate_argnums=(0,))
    result = {
        "configuration": dataclasses.asdict(config),
        "metadata": metadata,
        "pretrained": str(args.pretrained),
        "weight_loading": loader.report,
        "device": str(jax.devices()[0]),
        "dtype": "float32 parameter storage, bfloat16 transformer arithmetic",
        "tests": {},
        "optimizer": {"name": "clipped SGD", "learning_rate": 1e-5, "clip_gradient_norm": 1.0, "steps": 1},
    }
    trainable = nnx.state(model, is_flash_trainable_parameter)
    result["trainable_parameter_count"] = sum(x.size for x in jax.tree.leaves(trainable))
    result["frozen_vlm_parameter_count"] = sum(x.size for x in jax.tree.leaves(nnx.state(model, is_vlm_parameter)))
    print("synthetic prefix/gradient compile", flush=True)
    start = time.perf_counter()
    syn_mask, syn_cache = sync(encode(model, synthetic_obs))
    syn_metrics, syn_grad = sync(
        gradient(model, syn_mask, syn_cache, synthetic_history, synthetic_target, jax.random.key(101))
    )
    assert finite_tree(syn_metrics)
    assert finite_tree(syn_grad)
    result["tests"]["synthetic"] = {
        "metrics": {k: float(v) for k, v in syn_metrics.items()},
        "gradients_finite": True,
        "compile_and_run_seconds": time.perf_counter() - start,
    }
    del syn_grad, syn_cache
    print("real RGB/LeRobot prefix and gradient", flush=True)
    mask, cache = sync(encode(model, real_obs))
    metrics, grads = sync(gradient(model, mask, cache, real_history, real_target, jax.random.key(102)))
    assert finite_tree(metrics)
    assert finite_tree(grads)
    result["tests"]["real"] = {
        "metrics": {k: float(v) for k, v in metrics.items()},
        "gradients_finite": True,
        "real_images": True,
    }
    before = sync(sample(model, mask, cache, real_history))
    assert before.shape == (1, 50, 58)
    assert finite_tree(before)
    trainable_before = parameter_digest(jax.device_get(nnx.state(model, is_flash_trainable_parameter).to_pure_dict()))
    grad_norm = float(sync(apply_sgd(model, grads)))
    del grads
    gc.collect()
    after = sync(sample(model, mask, cache, real_history))
    vlm_after = parameter_digest(nnx.state(model, nnx.All(nnx.Param, is_vlm_parameter)).to_pure_dict(), vlm_only=True)
    trainable_after = parameter_digest(jax.device_get(nnx.state(model, is_flash_trainable_parameter).to_pure_dict()))
    assert vlm_after == vlm_before
    assert trainable_before != trainable_after
    result["freeze_audit"] = {
        "before_sha256": vlm_before,
        "after_sha256": vlm_after,
        "unchanged": True,
        "expert_changed": True,
    }
    result["gradient_norm"] = grad_norm
    result["sampling"] = {
        "shape": list(after.shape),
        "finite": finite_tree(after),
        "max_abs_rad": float(jnp.max(jnp.abs(after))),
        "status": "untrained coefficient policy: shape/numerics only, not action quality",
    }
    np.savez(args.output / "sampling_local.npz", before=np.asarray(before), after=np.asarray(after))
    result["latency"] = {
        "cached_expert_nfe1": milliseconds(lambda: sample(model, mask, cache, real_history)),
        "end_to_end_3rgb_prefix_nfe1": milliseconds(lambda: end_to_end(model, real_obs, real_history)),
    }
    # Cached and full-prefix inference must agree. Future-label perturbation is tested separately in test_flash_math.py.
    repeated = sync(end_to_end(model, real_obs, real_history))
    np.testing.assert_allclose(after, repeated, atol=2e-5, rtol=2e-5)
    result["tests"]["cached_vs_end_to_end_max_error"] = float(jnp.max(jnp.abs(after - repeated)))
    result["gpu_allocator_stats"] = {
        k: int(v) for k, v in (jax.devices()[0].memory_stats() or {}).items() if isinstance(v, int | np.integer)
    }
    result["gpu_nvidia_smi"] = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"], text=True
    )
    (args.output / "pre_checkpoint_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    run.log(
        {
            "synthetic_loss": result["tests"]["synthetic"]["metrics"]["loss"],
            "real_loss": result["tests"]["real"]["metrics"]["loss"],
            "gradient_norm": grad_norm,
        }
    )
    print("save new checkpoint, full weights plus normalization/config/optimizer metadata", flush=True)
    checkpoint_metadata = {
        "config": dataclasses.asdict(config),
        "normalization": norm,
        "step": 1,
        "optimizer": result["optimizer"],
        "pretrained": str(args.pretrained),
        "prompt": metadata["prompt"],
        "batch_provenance": metadata["batch_provenance"],
    }
    save_offline_checkpoint(args.checkpoint, nnx.state(model).to_pure_dict(), metadata=checkpoint_metadata)
    expected_full_hash = parameter_digest(nnx.state(model).to_pure_dict())
    # Release old device weights BEFORE restoring, avoiding duplicate full VLM residency.
    graph, state = nnx.split(model)
    template = jax.tree.map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype), state)
    old_arrays = jax.tree.leaves(state)
    model = None
    cache = None
    mask = None
    trainable = None
    state = None
    for array in old_arrays:
        array.delete()
    del old_arrays
    gc.collect()
    print("restore checkpoint after releasing original GPU parameters", flush=True)
    restored, restored_metadata, _ = restore_offline_checkpoint(args.checkpoint)
    restored_hash = parameter_digest(restored)
    assert expected_full_hash == restored_hash
    assert restored_metadata["step"] == 1
    template.replace_by_pure_dict(jax.device_put(restored))
    restored_model = nnx.merge(graph, template)
    del restored
    gc.collect()
    restored_sample = sync(end_to_end(restored_model, real_obs, real_history))
    np.testing.assert_allclose(repeated, restored_sample, atol=2e-5, rtol=2e-5)
    result["checkpoint"] = {
        "path": str(args.checkpoint.resolve()),
        "full_parameter_sha256": restored_hash,
        "roundtrip_equal": True,
        "restored_sampling_max_error": float(jnp.max(jnp.abs(restored_sample - repeated))),
    }
    result["all_tests_passed"] = True
    run.summary["all_tests_passed"] = True
    run.finish()
    (args.output / "smoke_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "loss": result["tests"]["real"],
                "sampling": result["sampling"],
                "latency": result["latency"],
                "checkpoint": result["checkpoint"],
                "all_tests_passed": True,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
