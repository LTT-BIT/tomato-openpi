"""Exercise RTC loss, token conditioning, checkpoint compatibility and sampling."""

from dataclasses import replace
import types

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.models import gemma
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.shared import nnx_utils


class FlowProbe:
    """Replace only the network; run the actual Pi0 loss and sampler."""

    def __init__(self, max_delay=3, prediction=None):
        self.rtc_max_delay = max_delay
        self.action_horizon = 4
        self.action_dim = 2
        self.prediction = prediction
        self.PaliGemma = types.SimpleNamespace(llm=lambda tokens, **kwargs: (tokens, None))

    noise = pi0.Pi0.noise

    def embed_prefix(self, observation):
        b = observation.state.shape[0]
        return jnp.zeros((b, 1, 2)), jnp.ones((b, 1), bool), jnp.zeros(1, bool)

    def embed_suffix(self, observation, actions, time):
        self.inputs, self.time = actions, time
        return actions, jnp.ones(actions.shape[:2], bool), jnp.zeros(4, bool), None

    def action_out_proj(self, output):
        return jnp.ones_like(output) if self.prediction is None else self.prediction


@pytest.mark.parametrize("delay", [-1, 4, True, 1.5])
def test_invalid_delay(delay):
    with pytest.raises(ValueError, match="rtc_max_delay"):
        pi0_config.Pi0Config(pi05=True, action_horizon=4, rtc_max_delay=delay)


def test_rtc_requires_pi05():
    with pytest.raises(ValueError, match="pi05"):
        pi0_config.Pi0Config(rtc_max_delay=1)


@pytest.mark.parametrize("enabled", [False, True])
def test_flow_inputs_masked_loss_and_gradients(monkeypatch, enabled):
    config = pi0_config.Pi0Config(pi05=True, action_dim=2, action_horizon=4)
    obs, actions = config.fake_obs(3), jnp.arange(24, dtype=jnp.float32).reshape(3, 4, 2) / 10
    rng = jax.random.key(42)
    keys = jax.random.split(rng, 4 if enabled else 3)
    noise = jax.random.normal(keys[1], actions.shape)
    time = jax.random.beta(keys[2], 1.5, 1, (3,)) * 0.999 + 0.001
    delays = jnp.array([0, 2, 3])  # Include no prefix and the largest allowed prefix.

    def randint(key, shape, minval, maxval):
        assert shape == (3,)
        assert minval == 0
        assert maxval == 4
        np.testing.assert_array_equal(jax.random.key_data(key), jax.random.key_data(keys[3]))
        return delays

    monkeypatch.setattr(jax.random, "randint", randint)
    monkeypatch.setattr(_model, "preprocess_observation", lambda rng, obs, **kwargs: obs)
    probe = FlowProbe(3 if enabled else 0)
    loss = pi0.Pi0.compute_loss(probe, rng, obs, actions)
    mask = jnp.arange(4)[None, :] >= (delays[:, None] if enabled else 0)
    mask = jnp.broadcast_to(mask, (3, 4))
    raw_loss = jnp.mean((1 - (noise - actions)) ** 2, axis=-1)
    expected_loss = jnp.where(mask, raw_loss, 0) * (12 / mask.sum())
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    expected_time = jnp.where(mask, time[:, None], 0) if enabled else time
    np.testing.assert_array_equal(probe.time, expected_time)
    expanded = expected_time[..., None] if enabled else time[:, None, None]
    np.testing.assert_array_equal(probe.inputs, expanded * noise + (1 - expanded) * actions)

    def objective(prediction):
        return pi0.Pi0.compute_loss(FlowProbe(3 if enabled else 0, prediction), rng, obs, actions).mean()

    gradient = jax.grad(objective)(jnp.ones_like(actions))
    expected_grad = jnp.where(mask[..., None], 2 * (1 - noise + actions) / (mask.sum() * 2), 0)
    np.testing.assert_allclose(gradient, expected_grad, atol=1e-7)
    if enabled:
        np.testing.assert_array_equal(probe.inputs[~mask], actions[~mask])


def test_delay_sampling_is_reproducible_and_covers_inclusive_range(monkeypatch):
    cfg = pi0_config.Pi0Config(pi05=True, action_dim=2, action_horizon=4)
    monkeypatch.setattr(_model, "preprocess_observation", lambda rng, obs, **kwargs: obs)
    probe = FlowProbe()
    obs, act = cfg.fake_obs(256), cfg.fake_act(256)
    losses = []
    times = []
    for augmentation_rng in (jax.random.key(1), jax.random.key(2)):
        losses.append(pi0.Pi0.compute_loss(probe, jax.random.key(42), obs, act, augmentation_rng=augmentation_rng))
        times.append(probe.time)
    np.testing.assert_array_equal(losses[0], losses[1])
    np.testing.assert_array_equal(times[0], times[1])
    assert set(np.asarray((times[0] == 0).sum(axis=-1))) == {0, 1, 2, 3}


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_sampler_keeps_prefix_exact_including_final_step(monkeypatch, delay):
    cfg = pi0_config.Pi0Config(pi05=True, action_dim=2, action_horizon=4)
    obs = cfg.fake_obs(2)
    prefix = jnp.arange(16, dtype=jnp.float32).reshape(2, 4, 2)
    noise = jnp.full(prefix.shape, 5.0)
    monkeypatch.setattr(_model, "preprocess_observation", lambda rng, obs, **kwargs: obs)
    sample = jax.jit(
        lambda prefix: pi0.Pi0.sample_actions(
            FlowProbe(),
            jax.random.key(0),
            obs,
            num_steps=4,
            noise=noise,
            prefix_actions=prefix,
            prefix_length=delay,
        )
    )
    result = sample(prefix)
    np.testing.assert_array_equal(result[:, :delay], prefix[:, :delay])
    np.testing.assert_array_equal(result[:, delay:], noise[:, delay:] - 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefix_length": -1},
        {"prefix_length": 4},
        {"prefix_length": True},
        {"prefix_length": 1},
        {"prefix_actions": jnp.zeros((1, 2, 2))},
    ],
)
def test_sampler_rejects_invalid_prefix(kwargs):
    cfg = pi0_config.Pi0Config(action_dim=2, action_horizon=4)
    with pytest.raises(ValueError, match="prefix"):
        pi0.Pi0.sample_actions(FlowProbe(), jax.random.key(0), cfg.fake_obs(), **kwargs)


def test_tokenwise_adarms_uses_the_same_weights_and_broadcast_semantics():
    norm = gemma.RMSNorm()
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10
    cond = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    params = norm.init(jax.random.key(0), x, cond)
    # Nonzero modulation is essential: zero-initialized gates hide broadcasting bugs.
    params["params"]["Dense_0"]["kernel"] = jnp.arange(48, dtype=jnp.float32).reshape(4, 12) / 100
    shared, gate = norm.apply(params, x, cond)
    repeated, repeated_gate = norm.apply(params, x, jnp.repeat(cond[:, None], 3, axis=1))
    np.testing.assert_allclose(repeated, shared, rtol=1e-6)
    np.testing.assert_allclose(repeated_gate, jnp.broadcast_to(gate, repeated_gate.shape), rtol=1e-6)
    varied, _ = norm.apply(params, x, jnp.repeat(cond[:, None], 3, axis=1).at[:, 0].set(0))
    assert not np.allclose(varied[:, 0], shared[:, 0])
    # Broadcasting versus token-wise Dense can differ by one float32 ULP on CPU.
    np.testing.assert_allclose(varied[:, 1:], shared[:, 1:], rtol=1e-6)


def test_real_pi05_rtc_forward_backward_and_checkpoint_shapes(monkeypatch):
    # Real Gemma, SigLIP, adaRMS and JIT on small widths; no pretrained assets.
    original_siglip = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda **kwargs: original_siglip(**{**kwargs, "variant": "mu/224"}))
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    config = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=2,
        action_horizon=4,
        max_token_len=2,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    baseline = config.create(jax.random.key(0))
    rtc_config = replace(config, rtc_max_delay=3)
    rtc = rtc_config.load(nnx.state(baseline).to_pure_dict())
    original_shapes = jax.tree.map(lambda p: p.shape, nnx.state(baseline).to_pure_dict())
    rtc_shapes = jax.tree.map(lambda p: p.shape, nnx.state(rtc).to_pure_dict())
    assert original_shapes == rtc_shapes
    assert not nnx.state(rtc, nnx.All(nnx.Param, rtc_config.get_freeze_filter())).flat_state()
    obs, act = config.fake_obs(2), config.fake_act(2)
    legacy_sample = nnx_utils.module_jit(baseline.sample_actions)(jax.random.key(2), obs, num_steps=2)
    rtc_sample = nnx_utils.module_jit(rtc.sample_actions, static_argnames=("prefix_length",))
    np.testing.assert_array_equal(rtc_sample(jax.random.key(2), obs, num_steps=2), legacy_sample)
    locked = rtc_sample(jax.random.key(2), obs, num_steps=2, prefix_actions=act, prefix_length=3)
    np.testing.assert_array_equal(locked[:, :3], act[:, :3])
    assert np.isfinite(locked).all()

    @nnx.jit
    def update(model, optimizer):
        loss, grads = nnx.value_and_grad(lambda m: m.compute_loss(jax.random.key(5), obs, act).mean())(model)
        optimizer.update(grads)
        return loss, optax.global_norm(grads)

    before = np.asarray(rtc.action_out_proj.kernel.value).copy()
    optimizer = nnx.Optimizer(rtc, optax.adam(1e-4))
    loss, grad_norm = update(rtc, optimizer)
    assert np.isfinite(loss)
    assert np.isfinite(grad_norm)
    assert grad_norm > 0
    assert not np.array_equal(before, rtc.action_out_proj.kernel.value)
