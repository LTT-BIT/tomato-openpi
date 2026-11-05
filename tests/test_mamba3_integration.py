"""Diagnostic-only Pi05 integration; no synthetic test claims pretrained FullFT."""
# ruff: noqa: PLC0415, SLF001

from dataclasses import replace

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models.mamba3.context_compressor import ContextCompressor
from openpi.models.mamba3.context_compressor import segment_mean_pool
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_mamba import MambaPi0
from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.shared.normalize import NormStats
from openpi.training.mamba_training import MambaTrainConfig
from openpi.training.mamba_training import validate_stats
from openpi.training.mamba_weight_loader import merge_vlm_weights
from scripts.mamba3_diagnostics import diagnostic_config
from scripts.mamba3_diagnostics import diagnostic_network
from scripts.mamba3_diagnostics import record_evidence


@pytest.fixture
def tiny():
    with diagnostic_network():
        config = diagnostic_config()
        yield config, config.create(jax.random.key(42))


def random_observation(cfg, batch=1):
    obs = cfg.fake_obs(batch)
    return replace(
        obs,
        images={
            key: jax.random.uniform(jax.random.key(i + 11), value.shape, minval=-1, maxval=1)
            for i, (key, value) in enumerate(obs.images.items())
        },
        tokenized_prompt=jnp.tile(jnp.arange(cfg.max_token_len)[None] + 3, (batch, 1)),
    )


def test_prefix_only_matches_original_with_identical_weights(tiny):
    cfg, model = tiny
    base_config = Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=58,
        action_horizon=50,
        max_token_len=8,
        dtype="float32",
    )
    original = base_config.create(jax.random.key(10))
    merged, report = merge_vlm_weights(nnx.state(model).to_pure_dict(), nnx.state(original).to_pure_dict())
    assert report["unexpected_missing"] == report["mismatched"] == 0
    assert report["ignored_old_expert_leaves"] > 0
    graph, state = nnx.split(model)
    state.replace_by_pure_dict(merged)
    model = nnx.merge(graph, state)
    obs = random_observation(cfg)
    obs = replace(obs, tokenized_prompt_mask=obs.tokenized_prompt_mask.at[:, -2:].set(False))
    tokens, mask, ar = original.embed_prefix(obs)
    (reference, _), _ = original.PaliGemma.llm(
        [tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, axis=1) - 1
    )
    actual, actual_mask, _ = model.prefix_hidden(obs)
    np.testing.assert_array_equal(actual_mask, mask)
    np.testing.assert_allclose(
        np.asarray(actual)[np.asarray(mask)], np.asarray(reference)[np.asarray(mask)], rtol=1e-4, atol=1e-4
    )
    paths = traverse_util.flatten_dict(nnx.state(model).to_pure_dict())
    assert all(not any(str(v).endswith("_1") for v in p) for p in paths if p[:2] == ("PaliGemma", "llm"))
    model2 = cfg.load(nnx.state(model).to_pure_dict(), remove_extra_params=False)
    np.testing.assert_array_equal(model2.prefix_hidden(obs)[0], actual)


def test_plain_flow_random_streams_loss_and_augmentation(monkeypatch, tiny):
    cfg, model = tiny
    observation = cfg.fake_obs(3)
    actions = jax.random.normal(jax.random.key(8), (3, 50, 58))
    observed = {}

    def preprocess(rng, obs, *, train):
        observed["rng"] = jax.random.key_data(rng)
        return obs

    def velocity(self, x, time, condition):
        observed["x"], observed["time"] = x, time
        return jnp.ones_like(x) * 0.3

    monkeypatch.setattr(model_lib, "preprocess_observation", preprocess)
    monkeypatch.setattr(MambaPi0, "encode_condition", lambda self, obs: None)
    monkeypatch.setattr(MambaPi0, "velocity", velocity)
    rng = jax.random.key(7)
    keys = jax.random.split(rng, 3)
    epsilon = jax.random.normal(keys[1], actions.shape)
    time = jax.random.beta(keys[2], 1.5, 1.0, (3,)) * 0.999 + 0.001
    for aug in (None, jax.random.key(42002)):
        loss = model.compute_loss(rng, observation, actions, train=True, augmentation_rng=aug)
        np.testing.assert_array_equal(observed["rng"], jax.random.key_data(keys[0] if aug is None else aug))
        np.testing.assert_array_equal(observed["time"], time)
        np.testing.assert_array_equal(
            observed["x"], time[:, None, None] * epsilon + (1 - time[:, None, None]) * actions
        )
        np.testing.assert_allclose(loss, ((0.3 - (epsilon - actions)) ** 2).mean(-1), rtol=1e-6)
        assert loss.shape == (3, 50)


def test_masked_pool_and_empty_rows():
    compressor = ContextCompressor(
        16, 16, 32, 2, vision_tokens=5, language_tokens=3, global_tokens=2, dtype="float32", rngs=nnx.Rngs(3)
    )
    x = jax.random.normal(jax.random.key(9), (3, 20, 16))
    mask = jnp.ones((3, 20), bool).at[0, 4:8].set(False).at[1, 15:].set(False).at[2].set(False)
    a = compressor(x, mask, 12)
    b = compressor(jnp.where(mask[..., None], x, 100), mask, 12)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()
    np.testing.assert_array_equal(a[2], 0)
    pooled, valid = segment_mean_pool(x, mask, 7)
    for i in range(7):
        start, end = i * 20 // 7, (i + 1) * 20 // 7
        m = np.asarray(mask[:, start:end])
        expected = (np.asarray(x[:, start:end]) * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1)
        np.testing.assert_allclose(pooled[:, i], expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(valid[:, i], m.any(1))


def test_query_cache_is_local_and_modalities_connect_after_gate_probe(tiny):
    cfg, model = tiny
    obs = random_observation(cfg)
    # Production zero initialization faithfully blocks z-gradient at step zero.
    for block in model.action_expert.blocks.values():
        np.testing.assert_array_equal(block.film_out.kernel.value, 0)
        np.testing.assert_array_equal(block.cond_proj.kernel.value, 0)
        np.testing.assert_allclose(jax.nn.sigmoid(block.film_out.bias.value[-16:]), 0.2, rtol=1e-6)
        # Test-only nonzero FiLM/time projection makes the graph observable.
        block.film_out.kernel.value = jax.random.normal(jax.random.key(31), block.film_out.kernel.shape) * 0.03
        block.cond_proj.kernel.value = jnp.eye(16) * 0.02
    condition = model.encode_condition(obs)
    changed_image = replace(obs, images={**obs.images, "base_0_rgb": obs.images["base_0_rgb"] * 0.2})
    changed_text = replace(obs, tokenized_prompt=obs.tokenized_prompt + 2)
    changed_camera = replace(obs, image_masks={**obs.image_masks, "left_wrist_0_rgb": jnp.array([False])})
    for changed in (changed_image, changed_text, changed_camera):
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(jax.tree.leaves(condition), jax.tree.leaves(model.encode_condition(changed)), strict=True)
        )
    masked = replace(
        changed_camera,
        images={**changed_camera.images, "left_wrist_0_rgb": jnp.zeros_like(obs.images["left_wrist_0_rgb"])},
    )
    for a, b in zip(
        jax.tree.leaves(model.encode_condition(changed_camera)),
        jax.tree.leaves(model.encode_condition(masked)),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    x = jax.random.normal(jax.random.key(1), (1, 50, 58))
    t = jnp.ones(1) * 0.5
    np.testing.assert_array_equal(model.velocity(x, t, condition), model.velocity(x, t, model.encode_condition(obs)))
    a = model.sample_actions(jax.random.key(1), obs, noise=x, num_steps=10)
    model.sample_actions(jax.random.key(4), changed_image, noise=x, num_steps=1)
    np.testing.assert_array_equal(a, model.sample_actions(jax.random.key(1), obs, noise=x, num_steps=10))
    assert np.isfinite(a).all()
    assert a.shape == (1, 50, 58)
    _, grads = nnx.value_and_grad(lambda m: m.compute_loss(jax.random.key(9), obs, jnp.ones((1, 50, 58)) * 0.1).mean())(
        model
    )
    for group in (grads.PaliGemma.img, grads.PaliGemma.llm, grads.context_compressor, grads.action_expert):
        assert sum(float(jnp.sum(v**2)) for v in jax.tree.leaves(group)) > 0


def test_strict_loader_and_invalid_configurations(tiny):
    cfg, model = tiny
    source = nnx.state(model).to_pure_dict()
    ref = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), source)
    merged, report = merge_vlm_weights(ref, source)
    assert report["reused_vlm_leaves"] > 0
    assert isinstance(merged["action_expert"]["action_in_proj"]["kernel"], jax.ShapeDtypeStruct)
    broken = {**source, "PaliGemma": {**source["PaliGemma"], "img": {}}}
    with pytest.raises(ValueError, match="Missing required"):
        merge_vlm_weights(ref, broken)
    for changes in (
        {"paligemma_variant": "gemma_2b_lora"},
        {"action_representation": "delta"},
        {"normalization": "paper"},
    ):
        with pytest.raises(ValueError, match="Phase A"):
            replace(MambaPi0Config(), **changes)
    with pytest.raises(ValueError, match="Flow time"):
        model.velocity(jnp.ones((1, 50, 58)), jnp.ones((1, 49)), model.encode_condition(cfg.fake_obs()))
    with pytest.raises(ValueError, match="prefix_length"):
        model.sample_actions(jax.random.key(0), cfg.fake_obs(), prefix_length=1)
    stats = NormStats(mean=np.zeros(58), std=np.ones(58), q01=-np.ones(58), q99=np.ones(58))
    validate_stats({"state": stats, "actions": stats}, 58)
    with pytest.raises(ValueError, match="statistics"):
        validate_stats({"state": stats, "actions": replace(stats, mean=np.zeros((50, 58)))}, 58)
    from training_configs.experiments.tomato_sharpa.mamba.phase_a import CONFIG

    with pytest.raises(ValueError, match="freeze"):
        replace(CONFIG, freeze_filter=nnx.Param)
    assert isinstance(CONFIG, MambaTrainConfig)
    assert CONFIG.ema_decay is None
    assert CONFIG.batch_size == 512
    assert CONFIG.num_workers == CONFIG.fsdp_devices == 8
    assert CONFIG.num_train_steps == 8000
    assert len(CONFIG.data.lerobot_datasets) == 6


def test_absolute_transform_roundtrip_and_no_delta():
    from openpi.policies.fr3_sharpa_multimodal_policy import Fr3SharpaMultimodalInputs

    state = np.arange(116, dtype=np.float32) * 0.001
    action = np.arange(50 * 58, dtype=np.float32).reshape(50, 58) * 0.0001
    raw = {
        "observation/state": state,
        "actions": action,
        "observation/image": np.zeros((224, 224, 3), np.uint8),
        "observation/left_wrist_image": np.zeros((224, 224, 3), np.uint8),
        "observation/right_wrist_image": np.zeros((224, 224, 3), np.uint8),
    }
    canonical = Fr3SharpaMultimodalInputs(model_lib.ModelType.PI05)(raw)
    expected = np.concatenate([action[:, :7], action[:, 14:36], action[:, 7:14], action[:, 36:]], -1)
    np.testing.assert_array_equal(canonical["actions"], expected)
    stat = NormStats(mean=np.arange(58, dtype=np.float32) * 0.001, std=np.linspace(0.2, 2, 58, dtype=np.float32))
    normalized = transforms.Normalize({"state": stat, "actions": stat})(canonical)
    restored = transforms.Unnormalize({"actions": stat})({"actions": normalized["actions"]})
    np.testing.assert_allclose(restored["actions"], expected, atol=1e-5)


def test_fixed_flow_fixture_learns_with_registered_parameters():
    """S01: diagnostic expert only, fixed x/t/target, not evidence of task convergence."""
    import optax

    from openpi.models.mamba3.action_expert import MambaActionExpert

    cfg = diagnostic_config()
    expert = MambaActionExpert(cfg.mamba, 58, 2, 32, 32, rngs=nnx.Rngs(12))
    graph, params = nnx.split(expert)
    x = jax.random.normal(jax.random.key(18), (1, 50, 58)) * 0.2
    z = jax.random.normal(jax.random.key(19), (1, 16))
    target = jnp.ones_like(x) * 0.15

    def objective(ps):
        m = nnx.merge(graph, ps)
        return ((m(x, jnp.array([0.4]), m.prepare_condition(z)) - target) ** 2).mean()

    tx = optax.adam(0.003)
    opt = tx.init(params)

    @jax.jit
    def update(ps, os):
        loss, grad = jax.value_and_grad(objective)(ps)
        delta, os = tx.update(grad, os, ps)
        return optax.apply_updates(ps, delta), os, loss

    losses = []
    for _ in range(30):
        params, opt, loss = update(params, opt)
        losses.append(float(loss))
    record_evidence("fixed_flow_fixture", {"diagnostic_only": True, "steps": 30, "lr": 0.003, "losses": losses})
    assert np.isfinite(losses).all()
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) * 0.5


def test_offline_policy_uses_absolute_action_stats_once(tiny):
    from openpi.policies.mamba_policy import MambaOfflinePolicy
    from openpi.training import config as train_cfg

    cfg, model = tiny
    state_stat = NormStats(
        mean=np.full(58, 10, np.float32), std=np.full(58, 2, np.float32), q01=np.zeros(58), q99=np.ones(58)
    )
    action_stat = NormStats(
        mean=np.full(58, 0.2, np.float32), std=np.full(58, 0.5, np.float32), q01=np.zeros(58), q99=np.ones(58)
    )
    data = train_cfg.DataConfig(norm_stats={"state": state_stat, "actions": action_stat})
    raw = {
        "state": np.ones(58, np.float32),
        "image": {key: np.zeros((224, 224, 3), np.float32) for key in model_lib.IMAGE_KEYS},
        "image_mask": dict.fromkeys(model_lib.IMAGE_KEYS, np.True_),
        "tokenized_prompt": np.arange(8, dtype=np.int32),
        "tokenized_prompt_mask": np.ones(8, bool),
    }
    noise = jax.random.normal(jax.random.key(4), (1, 50, 58))
    actual = MambaOfflinePolicy(model, data).infer(raw, rng=jax.random.key(4), noise=noise, num_steps=1)["actions"]
    normalized = transforms.Normalize(data.norm_stats)(raw)
    observation = model_lib.Observation.from_dict(jax.tree.map(lambda v: jnp.asarray(v)[None], normalized))
    # Unnormalize mirrors Normalize's epsilon; use its exact inverse convention.
    expected = transforms.Unnormalize({"actions": action_stat})(
        {"actions": model.sample_actions(jax.random.key(4), observation, noise=noise, num_steps=1)}
    )["actions"]
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (1, 50, 58)
