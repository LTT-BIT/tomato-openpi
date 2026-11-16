"""Behavior checks: no teacher-label leakage, real decoder gradients, and geometry."""

from dataclasses import replace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import pi0_fovea
from openpi.models import siglip
from openpi.subtasks import fovea


@pytest.mark.parametrize("size", [(640, 400), (640, 480), (1280, 720), (720, 1280)])
def test_padded_location_round_trip(size):
    for box in [(0.1, 0.2, 0.8, 0.9), (0, 0, 1, 1), fovea.INITIAL_BOX]:
        encoded = fovea.encode_box(box, *size)
        recovered = fovea.decode_box(encoded, *size)
        np.testing.assert_allclose(recovered, box, atol=0.002)


def test_branch_mask_and_padding():
    mask = np.asarray(pi0_fovea.branch_mask(jnp.array([[True, False, True]]), 4, 5))[0]
    assert not mask[:3, 3:].any()
    assert not mask[3:7, 7:].any()
    assert not mask[7:, 3:7].any()
    assert not mask[:, 1].any()
    np.testing.assert_array_equal(mask[3:7, 3:7], np.tril(np.ones((4, 4), bool)))
    assert mask[7:, 7:].all()


def test_crop_boundary_and_input_validation():
    raw = np.full((400, 640, 3), 150, np.uint8)
    result = fovea.crop_rgb(raw, (0, 0, 1, 1))
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 150
    with pytest.raises(ValueError, match="nonempty"):
        fovea.decode_box(np.array([256000] * 4), 640, 400)
    with pytest.raises(ValueError, match="nonempty"):
        fovea.validate_box([0.4, 0.3, 0.2, 0.8])


def test_joint_network_gradients_and_teacher_isolation(monkeypatch):
    # Only the huge image encoder is replaced. Gemma, attention, location output,
    # action expert, flow integration and optimizer all execute real code.
    original = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda **kw: original(**{**kw, "variant": "mu/224"}))
    cfg = pi0_fovea.FoveaConfig(
        dtype="float32",
        action_dim=2,
        action_horizon=4,
        execute_steps=2,
        max_token_len=3,
        image_keys=("base_0_rgb",),
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = cfg.create(jax.random.key(1))
    obs = replace(cfg.fake_obs(1), foveation_tokens=jnp.array([[256010, 256020, 256600, 256800]], jnp.int32))
    actions = jnp.ones((1, 4, 2), jnp.float32) * 0.1
    other = replace(obs, foveation_tokens=jnp.array([[256100, 256200, 256700, 256900]], jnp.int32))
    loss_fn = nnx.jit(lambda m, o: m.compute_loss_components(jax.random.key(3), o, actions))
    left = loss_fn(model, obs)
    right = loss_fn(model, other)
    np.testing.assert_array_equal(left["action_loss"], right["action_loss"])
    assert np.isfinite(left["box_loss"]).all()
    np.testing.assert_allclose(left["loss"], left["action_loss"] + 0.1 * left["box_loss"])

    @nnx.jit
    def gradients(m):
        return nnx.value_and_grad(lambda x: x.compute_loss(jax.random.key(3), obs, actions).mean())(m)

    loss, grads = gradients(model)
    assert np.isfinite(loss)
    assert np.isfinite(optax.global_norm(grads))
    assert optax.global_norm(grads) > 0
    flat = grads.to_pure_dict()
    assert np.linalg.norm(flat["PaliGemma"]["llm"]["embedder"]["input_embedding"][256000:257024]) > 0
    sampler = nnx.jit(lambda m, o: m.sample_joint(jax.random.key(4), o, num_steps=2))
    result = sampler(model, obs)
    second = sampler(model, other)
    assert result["actions"].shape == (1, 4, 2)
    assert result["foveation_tokens"].shape == (1, 4)
    assert np.isfinite(result["actions"]).all()
    assert np.asarray(result["foveation_tokens"]).min() >= 256000
    assert np.asarray(result["foveation_tokens"]).max() <= 257023
    np.testing.assert_array_equal(result["actions"], second["actions"])
    np.testing.assert_array_equal(result["foveation_tokens"], second["foveation_tokens"])
    before = np.asarray(model.action_out_proj.kernel.value).copy()
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3))
    optimizer.update(grads)
    assert not np.array_equal(before, model.action_out_proj.kernel.value)


@pytest.mark.parametrize(("offset", "expected_pairs"), [(8, 62), (15, 48)])
def test_episode_pair_filter_and_future_target(tmp_path, offset, expected_pairs):
    import json

    from openpi.training.fovea_dataset import FoveaDataset
    from openpi.training.fovea_dataset import digest

    class Table:
        def select(self, indices):
            return indices

    class Source:
        root = tmp_path
        hf_dataset = Table()

        def __len__(self):
            return 80

        def __getitem__(self, index):
            return {
                "episode_index": index // 40,
                "frame_index": index % 40,
                "observation.images.cam0": np.full((400, 640, 3), 80, np.uint8),
            }

    records = []
    for ep in range(2):
        boxes = np.tile([0.2, 0.2, 0.7, 0.7], (40, 1)).astype(np.float32)
        boxes[:, 0] += np.arange(40) * 0.001
        valid = np.ones(40, bool)
        if ep == 0:
            valid[20] = False
        path = tmp_path / f"{ep}.npz"
        np.savez(path, boxes=boxes, valid=valid)
        records.append({"episode_index": ep, "frames": 40, "labels": path.name, "sha256": digest(path)})
    manifest = {
        "version": fovea.VERSION,
        "complete": True,
        "execute_steps": offset,
        "dataset_metadata": {},
        "episodes": records,
        "training_pairs": expected_pairs,
        "source_size": [640, 400],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    dataset = FoveaDataset(Source(), tmp_path, offset)
    assert len(dataset) == expected_pairs
    assert 20 - offset not in dataset.indices
    assert 20 not in dataset.indices
    assert not set(range(40 - offset, 40)).intersection(dataset.indices)
    assert not set(range(80 - offset, 80)).intersection(dataset.indices)
    row = dataset[0]
    np.testing.assert_array_equal(row["foveation_tokens"], fovea.encode_box(dataset.labels[0][0][offset], 640, 400))
    assert row["observation.images.interaction"].shape == (3, 224, 224)
