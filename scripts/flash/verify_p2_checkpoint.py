"""Verify already-saved full P2 weights and Adam moments WITHOUT any new update."""

import argparse
import dataclasses
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from train_p2 import content_digest
from train_p2 import optimizer_only

from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_checkpoint import is_flash_trainable_parameter
from openpi.training.flash_checkpoint import parameter_digest
from openpi.training.flash_checkpoint import restore_offline_checkpoint
from openpi.training.flash_dataset import FlashLeRobotDataset
from openpi.training.optimizer import AdamW


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "result.json").read_text())
    params, metadata, moments = restore_offline_checkpoint(args.checkpoint)
    expected = report["checkpoint_pre_save_hashes"]
    actual_params = parameter_digest(params)
    assert actual_params == expected["parameters"]
    config = Pi0FlashConfig(
        **{
            **metadata["config"],
            "action_q01": tuple(metadata["config"]["action_q01"]),
            "action_q99": tuple(metadata["config"]["action_q99"]),
        }
    )
    options = {k: metadata["optimizer"][k] for k in dataclasses.asdict(AdamW())}
    tx = AdamW(**options).create(metadata["optimizer"]["lr"])
    abstract = nnx.eval_shape(
        lambda key: nnx.Optimizer(config.create(key), tx, wrt=is_flash_trainable_parameter), jax.random.key(42)
    )
    graph, state = nnx.split(abstract)
    # NNX's expected paths recover integer keys from Orbax's JSON string keys.
    state.replace_by_pure_dict({"model": params, **moments})
    optimizer = nnx.merge(graph, state)
    actual_optimizer = content_digest(optimizer_only(optimizer).to_pure_dict())
    assert actual_optimizer == expected["optimizer"]
    assert int(optimizer.step.value) == 100
    state.replace_by_pure_dict(jax.device_put(state.to_pure_dict()))
    optimizer = nnx.merge(graph, state)
    del params, moments, state, abstract
    dataset = FlashLeRobotDataset(args.data, args.images, PaligemmaTokenizer(max_len=256))
    plan = json.loads((args.data / "window_plan.json").read_text())
    obs, batch, _ = dataset.observation(plan["validation"][0])
    sample = nnx.jit(lambda m, o, h: m.sample_actions(jax.random.key(0), o, history=h), donate_argnums=(0,))
    prediction = np.asarray(jax.block_until_ready(sample(optimizer.model, obs, jnp.asarray(batch["history"]))))
    assert prediction.shape == (1, 50, 58)
    assert np.isfinite(prediction).all()
    reference = np.load(args.run / "validation_step_100.npz")["prediction"][0:1, 2:52]
    report["checkpoint"] = {
        "path": str(args.checkpoint.resolve()),
        "parameter_sha256": actual_params,
        "optimizer_sha256": actual_optimizer,
        "step": int(optimizer.step.value),
        "parameter_roundtrip": True,
        "adam_state_roundtrip": True,
        "restored_prediction_max_error": float(np.max(abs(prediction - reference))),
        "restored_prediction_mae_difference": float(np.mean(abs(prediction - reference))),
        "sampling_shape": list(prediction.shape),
        "sampling_finite": True,
        "verification_optimizer_updates": 0,
        "full_checkpoints_created": 1,
        "integer_key_restore": "hash typed NNX state after deserialization, not raw JSON-key tree",
        "cross_process_note": "parameter/state hashes are exact; forward arithmetic is not assumed bitwise reproducible across GPU processes",
    }
    report["completed"] = True
    (args.run / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["checkpoint"], indent=2), flush=True)


if __name__ == "__main__":
    main()
