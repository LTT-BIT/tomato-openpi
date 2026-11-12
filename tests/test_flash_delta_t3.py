"""FLASH Delta/T3 causal codec, independent assets and real optimizer recovery."""

from dataclasses import fields
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_flash_training import prepared as absolute_prepared
from test_flash_training import test_fullft_update_checkpoint_and_rejected_nonfinite as fullft_check

from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.pi0_flash_delta_t3 import Pi0FlashDeltaT3Config
from openpi.shared.flash_delta_t3 import correlation
from openpi.shared.flash_delta_t3 import decode
from openpi.shared.flash_delta_t3 import encode
from openpi.training import flash_native as native
from openpi.training.flash_delta_t3_data import bind
from openpi.training.flash_delta_t3_data import load


@pytest.fixture
def prepared(tmp_path):
    config = absolute_prepared.__wrapped__(tmp_path)
    model = Pi0FlashDeltaT3Config(**{f.name: getattr(config.model, f.name) for f in fields(Pi0FlashConfig)})
    config = replace(config, model=model, data=replace(config.data, assets=replace(config.data.assets, asset_id="t3")))
    native.prepare(config)
    return bind(config)


@pytest.mark.parametrize("timestamp", [False, True])
def test_roundtrip_and_hands(timestamp):
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(2, 54, 58)), jnp.float32)
    anchor = jnp.asarray(rng.normal(size=(2, 58)), jnp.float32)
    mean = jnp.asarray(rng.normal(size=(54, 58)), jnp.float32)
    std = jnp.ones((54, 58)) * 2
    low, high = jnp.full(58, -2.0), jnp.full(58, 2.0)
    z = encode(x, anchor, mean, std, low, high, timestamp=timestamp)
    result = decode(z, anchor, mean, std, low, high, timestamp=timestamp)
    np.testing.assert_allclose(result, x, atol=1e-6)
    np.testing.assert_array_equal(z[..., 7:29], ((x - low) / (high - low + 1e-6) * 2 - 1)[..., 7:29])


def test_assets_train_only_and_bound(prepared):
    report = native.preflight(prepared)
    assert report["training_action_semantics"] == "delta_arms_absolute_hands"
    wrong = replace(
        prepared, model=Pi0FlashConfig(**{f.name: getattr(prepared.model, f.name) for f in fields(Pi0FlashConfig)})
    )
    with pytest.raises(ValueError, match="Absolute FLASH"):
        native.preflight(wrong)
    arrays = load(prepared.model)
    assert arrays["mean"].shape == (54, 58)
    assert arrays["cholesky"].shape == (406, 406)
    meta = json.loads((native.asset_directory(prepared) / "flash_t3.json").read_text())
    assert meta["training_windows"] == report["legal_windows"] * 2 // 3
    with pytest.raises(ValueError, match="digest"):
        load(replace(prepared.model, correlation_beta=0.0))
    with pytest.raises(ValueError, match="RTC"):
        replace(prepared.model, rtc_max_delay=1)
    x = np.random.default_rng(4).normal(size=(100, 406))
    np.testing.assert_array_equal(correlation(x, 0), np.eye(406, dtype=np.float32))


def test_source_is_causal_and_inference_deterministic(prepared, monkeypatch):
    from test_flash_training import shrink_network

    shrink_network(monkeypatch)
    config = replace(prepared.model, dtype="float32", max_token_len=8)
    model = config.create(jax.random.key(0))
    history = jnp.ones((1, 8, 58)) * 0.1
    obs = config.fake_obs()
    obs = replace(obs, tokenized_prompt=jnp.ones_like(obs.tokenized_prompt))
    a = model.sample_actions(jax.random.key(1), obs, history=history)
    b = model.sample_actions(jax.random.key(2), obs, history=history)
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all()
    assert a.shape == (1, 50, 58)
    assert not any(path[0].startswith("t3_") for path in nnx_params(model))


def nnx_params(model):
    from flax import nnx

    return nnx.state(model, nnx.Param).flat_state()


def test_fullft_resume_and_nonfinite(prepared, monkeypatch, tmp_path):
    fullft_check(prepared, monkeypatch, tmp_path, reference=False)
    assert list((tmp_path / "checkpoint").rglob("flash_t3.npz"))


def test_validation_changes_do_not_fit_statistics(prepared):
    import pyarrow as pa
    import pyarrow.parquet as pq

    original = load(prepared.model)
    manifest = json.loads((native.asset_directory(prepared) / "flash_manifest.json").read_text())
    record = next(r for r in manifest["records"] if r["source_recording_id"] in manifest["validation_sources"])
    table = pq.read_table(record["path"])
    shifted = np.asarray(table["action"].to_pylist()) + 100
    table = table.set_column(table.schema.get_field_index("action"), "action", pa.array(shifted.tolist()))
    pq.write_table(table, record["path"])
    other = replace(
        prepared, data=replace(prepared.data, assets=replace(prepared.data.assets, asset_id="holdout_changed"))
    )
    native.prepare(other)
    current = load(bind(other).model)
    for key in original:
        np.testing.assert_array_equal(original[key], current[key])


def test_each_experiment_has_independent_identity_and_ui_roundtrip():
    from scripts import training_ui_config as editor
    from training_configs.experiments.tomato_sharpa.flash.delta_t3 import build

    identities = set()
    for trial in ("f1", "f2", "f3", "f4"):
        config = build(trial)
        identities.add(config.data.assets.asset_id)
        result = editor.apply_overrides(config, editor.config_values(config))
        assert result.model.timestamp == config.model.timestamp
        assert result.model.correlation_beta == config.model.correlation_beta
        assert result.policy_metadata["training_action_semantics"] == "delta_arms_absolute_hands"
        assert result.model.rtc_max_delay == 0
        assert config.num_train_steps == 8000
    assert len(identities) == 4


def test_beta_zero_keeps_flash_random_stream_and_loss():
    from types import SimpleNamespace

    from openpi.models.pi0_flash import Pi0Flash
    from openpi.models.pi0_flash_delta_t3 import Pi0FlashDeltaT3
    from openpi.shared.flash_polynomial import FlashPolynomial

    probe = SimpleNamespace(
        polynomial=FlashPolynomial(),
        boundary_constraint=True,
        history_noise_std=0.5,
        consistency_weight=1.0,
        beta=0.0,
        normalize=lambda x: x,
        codec=lambda values, anchor, indices: values,
        coefficient_velocity=lambda c, t, m, k: c * 0.1,
    )
    history = jax.random.normal(jax.random.key(3), (2, 8, 58))
    target = jax.random.normal(jax.random.key(4), (2, 54, 58))
    args = (probe, jax.random.key(5), None, None)
    expected = Pi0Flash.loss_from_cache(*args, history=history, target_ext=target)
    actual = Pi0FlashDeltaT3.loss_from_cache(*args, history=history, target_ext=target)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
