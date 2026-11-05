"""RTC=10 math, token conditioning and physical-prefix contracts."""
# ruff: noqa: PLC0415

from dataclasses import replace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.models.mamba3.action_expert import MambaActionExpert
from openpi.models.pi0_mamba import MambaPi0
from scripts.mamba3_diagnostics import diagnostic_config
from scripts.mamba3_diagnostics import diagnostic_network


class FlowProbe:
    action_horizon = 12
    action_dim = 2
    rtc_max_delay = 10
    noise = MambaPi0.noise

    def __init__(self, prediction=None):
        self.prediction = prediction

    def encode_condition(self, obs):
        return None

    def velocity(self, actions, time, condition):
        self.inputs, self.time = actions, time
        return jnp.ones_like(actions) if self.prediction is None else self.prediction


def test_rtc10_clean_prefix_mask_rng_and_suffix_gradients(monkeypatch):
    config = replace(diagnostic_config(), action_horizon=12)
    actions = jnp.arange(72, dtype=jnp.float32).reshape(3, 12, 2) / 100
    obs = config.fake_obs(3)
    rng = jax.random.key(42)
    keys = jax.random.split(rng, 4)
    noise = jax.random.normal(keys[1], actions.shape)
    time = jax.random.beta(keys[2], 1.5, 1.0, (3,)) * 0.999 + 0.001
    delays = jnp.array([0, 1, 10])
    seen = {}

    def randint(key, shape, minval, maxval):
        assert shape == (3,)
        assert minval == 0
        assert maxval == 11
        np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(keys[3]))
        return delays

    def preprocess(key, observation, **kwargs):
        seen["augmentation"] = jax.random.key_data(key)
        return observation

    monkeypatch.setattr(jax.random, "randint", randint)
    monkeypatch.setattr(model_lib, "preprocess_observation", preprocess)
    probe = FlowProbe()
    augmentation = jax.random.key(42002)
    loss = MambaPi0.compute_loss(probe, rng, obs, actions, train=True, augmentation_rng=augmentation)
    supervised = jnp.arange(12)[None] >= delays[:, None]
    times = jnp.where(supervised, time[:, None], 0)
    np.testing.assert_array_equal(probe.time, times)
    np.testing.assert_array_equal(probe.inputs, times[..., None] * noise + (1 - times[..., None]) * actions)
    np.testing.assert_array_equal(probe.inputs[~supervised], actions[~supervised])
    np.testing.assert_array_equal(seen["augmentation"], jax.random.key_data(augmentation))
    raw = jnp.mean((1 - noise + actions) ** 2, axis=-1)
    np.testing.assert_allclose(loss, jnp.where(supervised, raw, 0) * (36 / supervised.sum()), rtol=1e-6)
    gradient = jax.grad(lambda p: MambaPi0.compute_loss(FlowProbe(p), rng, obs, actions).mean())(jnp.ones_like(actions))
    expected = jnp.where(supervised[..., None], 2 * (1 - noise + actions) / (supervised.sum() * 2), 0)
    np.testing.assert_allclose(gradient, expected, atol=1e-7)


@pytest.mark.parametrize("length", [0, 1, 10])
def test_jitted_sampler_keeps_prefix_at_every_ode_step(monkeypatch, length):
    obs = replace(diagnostic_config(), action_horizon=12).fake_obs(2)
    prefix = jnp.arange(48, dtype=jnp.float32).reshape(2, 12, 2) / 10
    noise = jnp.full(prefix.shape, 5.0)
    monkeypatch.setattr(model_lib, "preprocess_observation", lambda key, obs, **kw: obs)
    probe = FlowProbe()

    def velocity(actions, time, condition):
        if length:
            assert time.shape == (2, 12)
            invalid = jnp.any(time[:, :length] != 0) | jnp.any(actions[:, :length] != prefix[:, :length])
        else:
            assert time.shape == (2,)
            invalid = False
        # An invalid intermediate prefix changes the suffix, so final clamping
        # alone cannot make this check pass.
        return jnp.ones_like(actions) * jnp.where(invalid, 100.0, 1.0)

    probe.velocity = velocity
    result = jax.jit(
        lambda p: MambaPi0.sample_actions(
            probe, jax.random.key(0), obs, noise=noise, num_steps=4, prefix_actions=p, prefix_length=length
        )
    )(prefix)
    np.testing.assert_array_equal(result[:, :length], prefix[:, :length])
    np.testing.assert_array_equal(result[:, length:], noise[:, length:] - 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefix_length": -1},
        {"prefix_length": 11},
        {"prefix_length": True},
        {"prefix_length": 10},
        {"prefix_actions": jnp.zeros((1, 2, 2))},
    ],
)
def test_sampler_rejects_invalid_or_missing_prefix(kwargs):
    with pytest.raises(ValueError, match="prefix"):
        MambaPi0.sample_actions(FlowProbe(), jax.random.key(0), diagnostic_config().fake_obs(), **kwargs)


def test_tokenwise_time_reuses_weights_and_prefix_influences_suffix():
    config = diagnostic_config()
    expert = MambaActionExpert(config.mamba, 58, 2, 32, 32, rngs=nnx.Rngs(3))
    for block in expert.blocks.values():
        # Zero conditioning projections would hide a token broadcasting error.
        block.cond_proj.kernel.value = jax.random.normal(jax.random.key(5), (16, 16)) * 0.1
    condition = expert.prepare_condition(jax.random.normal(jax.random.key(6), (2, 16)))
    actions = jax.random.normal(jax.random.key(7), (2, 12, 58))
    time = jnp.array([0.3, 0.8])
    scalar = expert(actions, time, condition)
    repeated = expert(actions, jnp.repeat(time[:, None], 12, axis=1), condition)
    np.testing.assert_allclose(repeated, scalar, atol=2e-6, rtol=2e-6)
    token_time = jnp.where(jnp.arange(12)[None] < 10, 0, time[:, None])
    before = expert(actions, token_time, condition)
    after = expert(actions.at[:, :10].add(0.5), token_time, condition)
    assert float(jnp.max(jnp.abs(before[:, 10:] - after[:, 10:]))) > 1e-5
    _, grads = nnx.value_and_grad(lambda m: jnp.mean(m(actions, token_time, condition)[:, 10:] ** 2))(expert)
    assert all(np.isfinite(g).all() for g in jax.tree.leaves(grads))
    assert float(jnp.linalg.norm(grads.time_mlp_in.kernel.value)) > 0


def test_plain_policy_normalizes_physical_prefix_and_preserves_it(monkeypatch):
    from openpi.policies.mamba_policy import create_mamba_policy
    from openpi.shared.normalize import NormStats
    from openpi.training.config import DataConfig

    config = replace(diagnostic_config(), rtc_max_delay=10)
    stats = NormStats(mean=np.arange(58, dtype=np.float32) / 10, std=np.ones(58) * 2, q01=-np.ones(58), q99=np.ones(58))
    data = DataConfig(norm_stats={"state": stats, "actions": stats})
    captured = {}

    def sample(self, rng, observation, **kwargs):
        captured.update(kwargs)
        return kwargs["prefix_actions"]

    with diagnostic_network():
        model = config.create(jax.random.key(0))
    monkeypatch.setattr(MambaPi0, "sample_actions", sample)
    raw = jax.tree.map(lambda x: np.asarray(x)[0], config.fake_obs().to_dict())
    prefix = np.arange(580, dtype=np.float32).reshape(10, 58) / 100
    policy = create_mamba_policy(model, data)
    output = policy.infer(raw, rng=jax.random.key(0), absolute_prefix=prefix)["actions"][0]
    assert captured["prefix_length"] == 10
    np.testing.assert_allclose(
        np.asarray(captured["prefix_actions"])[0, :10], (prefix - stats.mean) / (stats.std + 1e-6), atol=1e-7
    )
    np.testing.assert_array_equal(output[:10], prefix)
    with pytest.raises(ValueError, match="absolute_prefix"):
        policy.infer(raw, rng=jax.random.key(0), absolute_prefix=np.zeros((11, 58)))
