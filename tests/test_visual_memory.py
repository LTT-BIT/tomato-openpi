"""Causal history and offline/runtime equivalence; no robot or dataset downloads."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.model import Observation
from openpi.models.model import preprocess_observation
from openpi.models.pi0_visual_memory import Pi0VisualMemory
from openpi.models.pi0_visual_memory import VisualMemoryConfig
from openpi.models.pi0_visual_memory import pool_tokens
from openpi.subtasks.visual_memory import HISTORY
from openpi.subtasks.visual_memory import IMAGE_KEYS
from openpi.subtasks.visual_memory import RAW_KEYS
from openpi.subtasks.visual_memory import VisualHistory
from openpi.subtasks.visual_memory import VisualMemoryPolicy
from openpi.subtasks.visual_memory import VisualMemorySpec
from openpi.training.visual_memory import LeRobotVisualMemoryDataConfig
from openpi.training.visual_memory import VisualMemoryDataset


def sample(index, *, valid=True):
    return {
        **{key: np.full((16, 16, 3), index, np.uint8) for key in RAW_KEYS},
        "observation.interaction_available": np.bool_(valid),
        "observation.state": np.full(116, index, np.float32),
        "action": np.full((3, 58), index + 10, np.float32),
        "prompt": "pick up",
    }


class Table:
    def __init__(self, columns):
        self.columns = columns

    def select_columns(self, keys):
        return self

    def with_format(self, name):
        return self

    def __getitem__(self, key):
        return self.columns


class Source:
    def __init__(self):
        self.meta = SimpleNamespace(fps=30)
        self.root = "synthetic"
        self.hf_dataset = Table({"episode_index": [0] * 7 + [1] * 7, "frame_index": list(range(7)) * 2})
        self.samples = [sample(i, valid=i != 2) for i in range(14)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def test_dataset_matches_online_across_episode_and_prompt_boundaries():
    source = Source()
    spec = VisualMemorySpec(frames=3, stride=2)
    dataset, online = VisualMemoryDataset(source, spec), VisualHistory(spec)
    for index in range(len(source)):
        current = source[index]
        actual = online.observe(current, episode_id=str(index // 7), frame_index=index % 7, timestamp=index / 30)
        expected = dataset[index]
        assert len(online.frames) <= 5
        assert set(actual[HISTORY]) == {"image", "image_mask"}
        for key in spec.history_keys:
            np.testing.assert_array_equal(actual[HISTORY]["image"][key], expected[HISTORY]["image"][key])
            assert actual[HISTORY]["image_mask"][key] == expected[HISTORY]["image_mask"][key]
        np.testing.assert_array_equal(expected["action"], current["action"])
        np.testing.assert_array_equal(expected["observation.state"], current["observation.state"])
    assert not dataset[7][HISTORY]["image_mask"][spec.key(IMAGE_KEYS[0], 2)]
    assert not dataset[4][HISTORY]["image_mask"][spec.key(IMAGE_KEYS[-1], 2)]
    # A caller mutating returned arrays must not alter retained history.
    actual[HISTORY]["image"][spec.key(IMAGE_KEYS[0], 2)][:] = 255
    assert int(online.frames[-3][1][0][IMAGE_KEYS[0]][0, 0, 0]) != 255


def test_missing_observations_do_not_become_fabricated_history():
    spec = VisualMemorySpec(frames=3, stride=2)
    online = VisualHistory(spec)
    online.observe(sample(0), episode_id="a", frame_index=0, timestamp=0.0)
    value = online.observe(sample(3), episode_id="a", frame_index=3, timestamp=0.1)
    assert not any(value[HISTORY]["image_mask"].values())
    with pytest.raises(ValueError, match="increasing"):
        online.observe(sample(3), episode_id="a", frame_index=3, timestamp=0.1)
    with pytest.raises(ValueError, match="FPS"):
        online.observe(sample(4), episode_id="a", frame_index=4, timestamp=2.0)
    online.reset()
    assert not online.frames


@pytest.mark.parametrize("kwargs", [{"frames": 0}, {"stride": True}, {"pool_grid": 3}, {"fps": -1}])
def test_invalid_spec(kwargs):
    with pytest.raises(ValueError, match="positive|divide"):
        VisualMemorySpec(**kwargs)


def test_discontinuous_dataset_rejected():
    source = Source()
    source.hf_dataset.columns["frame_index"][3] = 8
    with pytest.raises(ValueError, match="contiguous"):
        VisualMemoryDataset(source, VisualMemorySpec())


def test_transform_history_ignores_past_state_and_preserves_current_delta():
    spec = VisualMemorySpec(frames=2, stride=2)
    model = VisualMemoryConfig(memory=spec, action_dim=58)
    factory = LeRobotVisualMemoryDataConfig(
        extra_delta_transform=True, interaction_mask_key="observation.interaction_available"
    )
    repack, vectors = factory.create_vector_transforms(model)
    assert not repack.inputs
    source = Source()
    payload = VisualMemoryDataset(source, spec)[4]
    result = vectors.inputs[0](payload)
    assert set(result["image"]) == set(spec.all_image_keys)
    np.testing.assert_array_equal(result["actions"][:, :7], 10)
    np.testing.assert_array_equal(result["actions"][:, 7:29], 14)
    np.testing.assert_array_equal(result["state"], 4)
    # Source action target must not be modified in place.
    np.testing.assert_array_equal(source[4]["action"], 14)
    assert HISTORY not in result
    # Deployment-style offline calls need no action target, and history stores no vectors.
    inference_payload = {k: v for k, v in payload.items() if k != "action"}
    inferred = vectors.inputs[0](inference_payload)
    assert "actions" not in inferred
    np.testing.assert_array_equal(inferred["state"], result["state"])
    source.samples[2]["observation.state"][:] = 999
    source.samples[2]["action"][:] = 999
    changed = vectors.inputs[0](VisualMemoryDataset(source, spec)[4])
    np.testing.assert_array_equal(changed["state"], result["state"])
    for key in result["image"]:
        np.testing.assert_array_equal(changed["image"][key], result["image"][key])
    with pytest.raises(ValueError, match="Explicit visual history"):
        vectors.inputs[0](sample(4))


def test_pool_keeps_spatial_structure_and_gradient():
    data = jnp.arange(256 * 8, dtype=jnp.float32).reshape(1, 256, 8)
    pooled = pool_tokens(data, 4)
    assert pooled.shape == (1, 16, 8)
    np.testing.assert_allclose(pooled[0, 0], data.reshape(1, 16, 16, 8)[0, :4, :4].mean((0, 1)))
    gradient = jax.grad(lambda x: pool_tokens(x, 4).sum())(data)
    np.testing.assert_allclose(gradient, 1 / 16)


class TinyImageEncoder:
    def __call__(self, image, *, train):
        pooled = image.mean((1, 2, 3))
        return jnp.broadcast_to(pooled[:, None, None], (len(image), 256, 8)), {}


class TinyLanguageEncoder:
    def __call__(self, prompt, *, method):
        return jnp.broadcast_to(prompt[..., None], (*prompt.shape, 8)).astype(jnp.float32)


def prefix_model(spec):
    # Real prefix method with a deterministic test encoder, avoiding multi-GB weights.
    return SimpleNamespace(memory=spec, PaliGemma=SimpleNamespace(img=TinyImageEncoder(), llm=TinyLanguageEncoder()))


def observation(spec):
    return Observation(
        images={k: jnp.zeros((1, 224, 224, 3)) for k in spec.all_image_keys},
        image_masks={k: jnp.ones((1,), dtype=bool) for k in spec.all_image_keys},
        state=jnp.zeros((1, 58)),
        tokenized_prompt=jnp.ones((1, 3), dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, 3), dtype=bool),
    )


def test_real_prefix_reads_history_and_masks_missing_slots():
    spec = VisualMemorySpec(frames=2)
    model, obs = prefix_model(spec), observation(spec)
    prefix, mask, _ = Pi0VisualMemory.embed_prefix(model, obs)
    assert prefix.shape == (1, 64 + 1024 + 3, 8)
    key = spec.history_keys[0]
    changed = replace(obs, images={**obs.images, key: jnp.ones_like(obs.images[key])})
    updated, _, _ = Pi0VisualMemory.embed_prefix(model, changed)
    assert not np.array_equal(prefix[:, :16], updated[:, :16])
    np.testing.assert_array_equal(prefix[:, 64:], updated[:, 64:])
    missing = replace(obs, image_masks={**obs.image_masks, key: jnp.array([False])})
    _, missing_mask, _ = Pi0VisualMemory.embed_prefix(model, missing)
    assert not missing_mask[:, :16].any()
    assert mask.all()


def test_temporal_augmentation_is_consistent_for_same_camera():
    spec = VisualMemorySpec(frames=2)
    obs = observation(spec)
    pattern = jax.random.uniform(jax.random.key(1), (1, 224, 224, 3), minval=-1, maxval=1)
    obs = replace(obs, images=dict.fromkeys(spec.all_image_keys, pattern))
    augmented = preprocess_observation(jax.random.key(4), obs, train=True, image_keys=spec.all_image_keys)
    for camera in spec.image_keys:
        np.testing.assert_array_equal(augmented.images[camera], augmented.images[spec.key(camera, spec.stride)])


def test_policy_contract_and_reset():
    spec = VisualMemorySpec(frames=2)
    policy = SimpleNamespace(infer=lambda obs: obs)
    with pytest.raises(ValueError, match="checkpoint"):
        VisualMemoryPolicy(policy, spec, {})
    wrapper = VisualMemoryPolicy(policy, spec, {**spec.contract(), "training_checkpoint": True})
    actual = wrapper.infer(sample(0), episode_id="a", frame_index=0, timestamp=0.0)
    assert not any(actual[HISTORY]["image_mask"].values())
    wrapper.reset()
    assert wrapper.history.episode_id is None
