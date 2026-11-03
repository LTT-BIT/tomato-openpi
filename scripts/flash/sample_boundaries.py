"""Measure causal prediction gaps and actual target projection at a real shared tick.

The two observations/history streams contain only data <= their own current tick.
Ground truth is used after inference for metrics, never as sampling conditioning.
"""

import argparse
import json
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from openpi.models import model as model_lib
from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.training.flash_checkpoint import restore_offline_checkpoint


def observation(directory, normalization, tokenizer):
    metadata = json.loads((directory / "metadata.json").read_text())
    batch = dict(np.load(directory / "batch.npz"))
    state = (batch["state"] - np.array(normalization["state_q01"])) / (
        np.array(normalization["state_q99"]) - np.array(normalization["state_q01"]) + 1e-6
    ) * 2 - 1
    tokens, mask = tokenizer.tokenize(metadata["prompt"], state[0])
    images = {
        role: jnp.asarray(np.asarray(Image.open(directory / (role + ".png")).convert("RGB")), jnp.float32)[None] / 127.5
        - 1
        for role in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    obs = model_lib.Observation(
        images=images,
        image_masks={k: jnp.ones((1,), bool) for k in images},
        state=jnp.asarray(state, jnp.float32),
        tokenized_prompt=jnp.asarray(tokens[None], jnp.int32),
        tokenized_prompt_mask=jnp.asarray(mask[None]),
    )
    return model_lib.preprocess_observation(None, obs, train=False), batch, metadata


def summary(error):
    error = np.asarray(error)
    return {
        "mae": float(np.mean(abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max": float(np.max(abs(error))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--next", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    params, checkpoint_metadata, _ = restore_offline_checkpoint(args.checkpoint)
    config = Pi0FlashConfig(
        **{
            **checkpoint_metadata["config"],
            "action_q01": tuple(checkpoint_metadata["config"]["action_q01"]),
            "action_q99": tuple(checkpoint_metadata["config"]["action_q99"]),
        }
    )
    abstract = nnx.eval_shape(config.create, jax.random.key(0))
    graph, state = nnx.split(abstract)
    state.replace_by_pure_dict(jax.device_put(params))
    model = nnx.merge(graph, state)
    del params, abstract, state
    tokenizer = PaligemmaTokenizer(max_len=config.max_token_len)
    norm = checkpoint_metadata["normalization"]
    obs_a, a, meta_a = observation(args.first, norm, tokenizer)
    obs_b, b, meta_b = observation(args.next, norm, tokenizer)
    assert meta_b["batch_provenance"][0]["current_frame"] - meta_a["batch_provenance"][0]["current_frame"] == 50
    assert meta_a["batch_provenance"][0]["sample_id"] == meta_b["batch_provenance"][0]["sample_id"]
    np.testing.assert_array_equal(a["target_ext"][:, 52], b["target_ext"][:, 2])

    @nnx.jit(donate_argnums=(0,))
    def predict(m, obs, history):
        mask, cache = m.encode_condition(obs)
        source = m.polynomial.fit_history(m.normalize(history))
        coeff = source + m.coefficient_velocity(source, jnp.zeros((history.shape[0],)), mask, cache)
        positions = m.denormalize(m.polynomial.reconstruct(coeff, extended=True))
        velocities = (
            m.polynomial.velocity(coeff, extended=True)
            * (jnp.asarray(m.action_q99) - jnp.asarray(m.action_q01) + 1e-6)
            / 2
        )
        return positions, velocities

    qa, va = jax.block_until_ready(predict(model, obs_a, jnp.asarray(a["history"])))
    qb, vb = jax.block_until_ready(predict(model, obs_b, jnp.asarray(b["history"])))
    for value in (qa, va, qb, vb):
        assert np.isfinite(value).all()
    real_q = b["target_ext"][:, 2]
    real_v = (b["target_ext"][:, 3] - b["target_ext"][:, 1]) * 15
    metrics = {}
    groups = {
        "arms": [i for i, name in enumerate(ORDER) if "fr3" in name],
        "hands": [i for i, name in enumerate(ORDER) if "fr3" not in name],
    }
    for group, ids in groups.items():
        metrics[group] = {
            "predicted_position_gap_rad": summary(np.asarray(qa[:, 52] - qb[:, 2])[:, ids]),
            "predicted_velocity_gap_rad_per_second": summary(np.asarray(va[:, 52] - vb[:, 2])[:, ids]),
            "first_boundary_position_error_rad": summary(np.asarray(qa[:, 52] - real_q)[:, ids]),
            "next_boundary_position_error_rad": summary(np.asarray(qb[:, 2] - real_q)[:, ids]),
        }
    target_effect = {}
    for constraint in (False, True):
        entries = []
        for batch in (a, b):
            coeff = model.polynomial.fit_target(
                model.normalize(jnp.asarray(batch["target_ext"])), boundary_constraint=constraint
            )
            q = model.denormalize(model.polynomial.reconstruct(coeff, extended=True))
            v = (
                model.polynomial.velocity(coeff, extended=True)
                * (jnp.asarray(model.action_q99) - jnp.asarray(model.action_q01) + 1e-6)
                / 2
            )
            entries.append((q, v))
        target_effect[str(constraint)] = {}
        for group, ids in groups.items():
            target_effect[str(constraint)][group] = {
                "position_error_at_shared_tick_rad": summary(np.asarray(entries[0][0][:, 52] - real_q)[:, ids]),
                "velocity_error_at_shared_tick_rad_per_second": summary(
                    np.asarray(entries[0][1][:, 52] - real_v)[:, ids]
                ),
                "position_gap_rad": summary(np.asarray(entries[0][0][:, 52] - entries[1][0][:, 2])[:, ids]),
                "velocity_gap_rad_per_second": summary(np.asarray(entries[0][1][:, 52] - entries[1][1][:, 2])[:, ids]),
            }
    result = {
        "checkpoint": str(args.checkpoint),
        "first_provenance": meta_a["batch_provenance"],
        "next_provenance": meta_b["batch_provenance"],
        "prediction_metrics": metrics,
        "target_constraint_effect": target_effect,
        "interpretation": "one SGD step, real causal RGB and measured history at each chunk; target constraints do not guarantee predicted C1",
        "finite": True,
        "leakage_guard": "predict function takes only obs/history; reference target used only after both predictions",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
