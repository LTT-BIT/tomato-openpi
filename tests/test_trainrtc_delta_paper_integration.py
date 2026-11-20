"""Joint T3/RTC mathematics, real optimization/checkpoints and all four UI paths."""
# ruff: noqa: PLC0415, SLF001, RUF001

from dataclasses import replace
from pathlib import Path
import types

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_delta_paper_training import prepared as source_prepared
from test_delta_paper_training import shrink_network

from openpi.models import model as model_lib
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_delta_paper import PaperPi0
from openpi.models.pi0_delta_paper import PaperPi0Config
from openpi.shared import delta_paper as paper
from openpi.shared import nnx_utils
from openpi.training import norm_provenance
from openpi.training.delta_paper import PaperSharpaData
from openpi.training.delta_paper_stats import compute
from scripts import run_training
from scripts import training_ui as ui
from scripts import training_ui_config as editor
from scripts.create_delta_paper_experiments import create

REPO = Path(__file__).resolve().parents[1]
BASE = "training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_fullft_v1"


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    return source_prepared.__wrapped__(tmp_path, monkeypatch, request)


class CorrelatedProbe:
    """Use production Paper noise and production RTC, replacing only the network."""

    compute_loss = Pi0.compute_loss
    sample_actions = Pi0.sample_actions
    noise = PaperPi0.noise

    def __init__(self, beta, delay, prediction=None):
        self.action_horizon, self.action_dim = 11, 2
        self.paper_beta, self.rtc_max_delay = beta, delay
        self.paper_mode = "timestamp_zscore_arms"
        # Deliberately non-diagonal: iid substitution fails the formula checks.
        chol = np.eye(22, dtype=np.float32)
        chol[1:, :-1] += np.eye(21, dtype=np.float32) * 0.4
        self.paper_noise_cholesky = nnx.Variable(jnp.asarray(chol))
        self.PaliGemma = types.SimpleNamespace(llm=lambda tokens, **kwargs: (tokens, None))
        self.prediction = prediction
        self.callback = None

    def embed_prefix(self, observation):
        batch = observation.state.shape[0]
        return jnp.zeros((batch, 1, 2)), jnp.ones((batch, 1), bool), jnp.zeros(1, bool)

    def embed_suffix(self, observation, actions, time):
        self.inputs, self.time = actions, time
        if self.callback:
            jax.debug.callback(self.callback, actions, time)
        return actions, jnp.ones(actions.shape[:2], bool), jnp.zeros(self.action_horizon, bool), None

    def action_out_proj(self, output):
        return jnp.ones_like(output) if self.prediction is None else self.prediction


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("beta", [0.0, 0.5])
@pytest.mark.parametrize("delay", [0, 10])
def test_joint_noise_clean_prefix_loss_and_gradient(monkeypatch, batch, beta, delay):
    cfg = Pi0Config(pi05=True, action_dim=2, action_horizon=11)
    obs = cfg.fake_obs(batch)
    actions = jnp.arange(batch * 22, dtype=jnp.float32).reshape(batch, 11, 2) / 20
    key = jax.random.key(42)
    keys = jax.random.split(key, 4 if delay else 3)
    probe = CorrelatedProbe(beta, delay)
    gaussian = jax.random.normal(keys[1], actions.shape)
    noise = (
        (gaussian.reshape(batch, 22) @ probe.paper_noise_cholesky.value.T).reshape(actions.shape) if beta else gaussian
    )
    times = jax.random.beta(keys[2], 1.5, 1, (batch,)) * 0.999 + 0.001
    delays = jnp.array([10] if batch == 1 else [0, 3, 10])

    def randint(rng, shape, low, high):
        assert shape == (batch,)
        assert low == 0
        assert high == 11
        np.testing.assert_array_equal(jax.random.key_data(rng), jax.random.key_data(keys[3]))
        return delays

    monkeypatch.setattr(jax.random, "randint", randint)
    monkeypatch.setattr(model_lib, "preprocess_observation", lambda rng, obs, **kw: obs)
    loss = probe.compute_loss(key, obs, actions)
    suffix = jnp.broadcast_to(jnp.arange(11) >= (delays[:, None] if delay else 0), (batch, 11))
    time = jnp.where(suffix, times[:, None], 0) if delay else times
    expanded = time[..., None] if delay else time[:, None, None]
    np.testing.assert_array_equal(probe.time, time)
    np.testing.assert_array_equal(probe.inputs, expanded * noise + (1 - expanded) * actions)
    expected = jnp.where(suffix, jnp.mean((1 - noise + actions) ** 2, axis=-1), 0) * (batch * 11 / suffix.sum())
    np.testing.assert_allclose(loss, expected, rtol=1e-6)
    gradient = jax.grad(lambda pred: CorrelatedProbe(beta, delay, pred).compute_loss(key, obs, actions).mean())(
        jnp.ones_like(actions)
    )
    np.testing.assert_allclose(
        gradient, jnp.where(suffix[..., None], (1 - noise + actions) / suffix.sum(), 0), atol=1e-7
    )
    if delay:
        np.testing.assert_array_equal(probe.inputs[~suffix], actions[~suffix])
        np.testing.assert_array_equal(loss[~suffix], 0)
    if beta:
        assert not np.array_equal(noise, gaussian)
    np.testing.assert_array_equal(probe.noise(keys[1], actions.shape), noise)


@pytest.mark.parametrize("prefix", [0, 3, 10])
@pytest.mark.parametrize("explicit", [False, True])
def test_correlated_sampler_keeps_prefix_at_every_step(monkeypatch, prefix, explicit):
    probe = CorrelatedProbe(0.5, 10)
    cfg = Pi0Config(pi05=True, action_dim=2, action_horizon=11)
    obs, actions = cfg.fake_obs(2), jnp.arange(44, dtype=jnp.float32).reshape(2, 11, 2) / 10
    key = jax.random.key(7)
    noise = jnp.full(actions.shape, 5.0) if explicit else probe.noise(key, actions.shape)
    seen = []
    probe.callback = lambda x, t: seen.append((np.asarray(x), np.asarray(t)))
    monkeypatch.setattr(model_lib, "preprocess_observation", lambda rng, obs, **kw: obs)
    sample = jax.jit(
        lambda: probe.sample_actions(
            key, obs, num_steps=4, noise=noise if explicit else None, prefix_actions=actions, prefix_length=prefix
        )
    )
    result = sample().block_until_ready()
    assert len(seen) == 4
    for x, time in seen:
        np.testing.assert_array_equal(x[:, :prefix], actions[:, :prefix])
        if prefix:
            np.testing.assert_array_equal(time[:, :prefix], 0)
    np.testing.assert_array_equal(result[:, :prefix], actions[:, :prefix])
    np.testing.assert_allclose(result[:, prefix:], noise[:, prefix:] - 1, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prefix_length": -1},
        {"prefix_length": 11},
        {"prefix_length": 1},
        {"prefix_actions": jnp.zeros((1, 2, 2))},
    ],
)
def test_joint_sampler_rejects_invalid_prefix(kwargs):
    obs = Pi0Config(pi05=True, action_dim=2, action_horizon=11).fake_obs(1)
    with pytest.raises(ValueError, match="prefix"):
        CorrelatedProbe(0.5, 10).sample_actions(jax.random.key(0), obs, **kwargs)


def test_one_authoritative_loss_and_sampler_and_config_validation():
    assert PaperPi0.compute_loss is Pi0.compute_loss
    assert PaperPi0.sample_actions is Pi0.sample_actions
    with pytest.raises(ValueError, match="rtc_max_delay"):
        PaperPi0Config(pi05=True, action_dim=58, action_horizon=50, rtc_max_delay=50)


def legacy_t3_loss(model, rng, observation, actions):
    """No-RTC T3 reference from source 2ab68c8, with identical weights/inputs."""
    preprocess, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = model_lib.preprocess_observation(preprocess, observation, train=False)
    epsilon = model.noise(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    x = time[..., None, None] * epsilon + (1 - time[..., None, None]) * actions
    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, cond = model.embed_suffix(observation, x, time)
    mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    (_, out), _ = model.PaliGemma.llm(
        [prefix, suffix],
        mask=make_attn_mask(mask, jnp.concatenate([prefix_ar, suffix_ar])),
        positions=jnp.cumsum(mask, axis=1) - 1,
        adarms_cond=[None, cond],
    )
    return jnp.mean((model.action_out_proj(out[:, -model.action_horizon :]) - (epsilon - actions)) ** 2, axis=-1)


@pytest.mark.parametrize("prepared", [50], indirect=True)
def test_real_50x58_t3_rtc_jit_adamw_checkpoint_and_legacy_equivalence(prepared, monkeypatch, tmp_path):
    from openpi.training import checkpoints
    from openpi.training import data_loader
    from openpi.training import sharding
    from scripts import train

    build, _, _ = prepared
    compute(build(mode="timestamp_zscore_arms", beta=0.5), batch_size=3)
    base = build(mode="timestamp_zscore_arms", beta=0.5)
    # The same parsed UI configuration used by the four-mode workflow feeds the network.
    values = {**editor.config_values(base), "rtc_max_delay": 10, "exp_name": "joint_t3_rtc_d10"}
    config = editor.apply_overrides(base, values)
    config = replace(config, model=replace(config.model, dtype="float32", max_token_len=4), batch_size=3, num_workers=0)
    shrink_network(monkeypatch)
    model = config.model.create(jax.random.key(0))
    initial = nnx.state(model)
    assert "paper_noise_cholesky" not in initial.filter(config.trainable_filter)
    base_weights = {k: v for k, v in initial.to_pure_dict().items() if k != "paper_noise_cholesky"}
    restore_params = model_lib.restore_params
    monkeypatch.setattr(model_lib, "restore_params", lambda *a, **kw: base_weights)
    monkeypatch.setattr("openpi.training.weight_loaders.download.maybe_download", lambda path: path)
    mesh = sharding.make_mesh(1)
    state, _ = train.init_train_state(config, jax.random.key(42), mesh, resume=False)
    loader = data_loader.create_data_loader(config, shuffle=True, num_batches=1)
    obs, actions = next(iter(loader))
    assert actions.shape == (3, 50, 58)
    assert config.model.paper_beta == 0.5
    assert config.model.rtc_max_delay == 10
    updated, metrics = jax.jit(lambda s: train.train_step(config, jax.random.key(42), s, (obs, actions)))(state)
    assert int(updated.step) == 1
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["grad_norm"])
    assert metrics["grad_norm"] > 0
    np.testing.assert_array_equal(
        updated.params["paper_noise_cholesky"].value, state.params["paper_noise_cholesky"].value
    )
    assert not np.array_equal(
        updated.params["action_out_proj"]["kernel"].value, state.params["action_out_proj"]["kernel"].value
    )
    trained = nnx.merge(updated.model_def, updated.params)
    # With RTC off, the refactored inherited loss is numerically identical to the original T3 path.
    trained.rtc_max_delay = 0
    np.testing.assert_array_equal(
        trained.compute_loss(jax.random.key(5), obs, actions), legacy_t3_loss(trained, jax.random.key(5), obs, actions)
    )
    trained.rtc_max_delay = 10
    sample = nnx_utils.module_jit(trained.sample_actions, static_argnames=("prefix_length",))
    expected = sample(jax.random.key(3), obs, num_steps=2, prefix_actions=actions, prefix_length=10)
    np.testing.assert_array_equal(expected[:, :10], actions[:, :10])
    # B=1 and no-prefix/default prefix=0 are also supported with the actual 58D network.
    one_obs = jax.tree.map(lambda x: x[:1] if x is not None else None, obs)
    one = sample(jax.random.key(3), one_obs, num_steps=2)
    np.testing.assert_array_equal(one, sample(jax.random.key(3), one_obs, num_steps=2, prefix_length=0))
    view = config.data.create(config.assets_dirs, config.model)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "saved", keep_period=10, overwrite=False, resume=False
    )
    checkpoints.save_state(manager, updated, types.SimpleNamespace(data_config=lambda: view), step=1)
    manager.wait_until_finished()
    manager.close()
    asset = tmp_path / "saved/1/assets" / view.asset_id
    source = config.data.asset_directory(config.assets_dirs)
    for name in ("norm_stats.json", "norm_metadata.json", paper.BUNDLE, paper.MANIFEST):
        assert (asset / name).read_bytes() == (source / name).read_bytes()
    params = restore_params(tmp_path / "saved/1/params", restore_type=np.ndarray)
    restored = config.model.load(params)
    np.testing.assert_array_equal(
        nnx.state(restored)["paper_noise_cholesky"].value, updated.params["paper_noise_cholesky"].value
    )
    restored_sample = nnx_utils.module_jit(restored.sample_actions, static_argnames=("prefix_length",))
    np.testing.assert_array_equal(
        restored_sample(jax.random.key(3), obs, num_steps=2, prefix_actions=actions, prefix_length=10), expected
    )
    # The same loader can initialize a NEW training run from a T3 checkpoint, retaining its pinned prior.
    monkeypatch.setattr(model_lib, "restore_params", lambda *a, **kw: params)
    new_weights = replace(config.weight_loader, params_path="fixture_t3_params").load(initial.to_pure_dict())
    np.testing.assert_array_equal(new_weights["paper_noise_cholesky"], initial["paper_noise_cholesky"].value)
    np.testing.assert_array_equal(new_weights["action_out_proj"]["kernel"], params["action_out_proj"]["kernel"])


def test_cross_chunk_physical_prefix_uses_new_origin_and_timestamp(prepared):
    from openpi import transforms
    from openpi.training.delta_paper_prefix import decode_actions
    from openpi.training.delta_paper_prefix import encode_prefix

    build, _, _ = prepared
    compute(build(mode="timestamp_zscore_arms", beta=0.5), batch_size=3)
    config = build(mode="timestamp_zscore_arms", beta=0.5)
    data = config.data.create(config.assets_dirs, config.model)
    old_state = np.linspace(-0.1, 0.2, 58, dtype=np.float32)
    new_state = old_state + 0.07
    absolute = np.full((3, 58), 0.1, np.float32)
    absolute[:, paper.ARMS] += np.arange(3, dtype=np.float32)[:, None] * 0.03
    old_encoded = encode_prefix(data, old_state, absolute)
    old_physical = decode_actions(data, old_state, old_encoded)
    prefix = old_physical[-2:]
    new_encoded = encode_prefix(data, new_state, prefix)
    new_physical = decode_actions(data, new_state, new_encoded)
    np.testing.assert_allclose(new_physical[:2], prefix, atol=1e-6)
    assert not np.allclose(new_encoded[:2, paper.ARMS], old_encoded[-2:, paper.ARMS])
    # Deliberately copying old normalized values cannot satisfy the physical contract.
    wrong = np.zeros((3, 58), np.float32)
    wrong[:2] = old_encoded[-2:]
    assert not np.allclose(decode_actions(data, new_state, wrong)[:2, paper.ARMS], prefix[:, paper.ARMS])
    hands = np.r_[7:29, 36:58]
    padded = np.zeros_like(absolute)
    padded[:2] = prefix
    quantiles = transforms.Normalize(data.norm_stats, use_quantiles=True)({"actions": padded})["actions"]
    np.testing.assert_array_equal(new_encoded[:2, hands], quantiles[:2, hands])


@pytest.mark.parametrize("prepared", [50], indirect=True)
def test_ui_four_modes_catalog_snapshot_preflight_and_stat_reuse(prepared, monkeypatch, tmp_path):
    from examples.fr3_wuji.compute_norm_stats import compute as standard_stats

    build, _, _ = prepared
    baseline_path = tmp_path / "training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_fullft_v1.py"
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text((REPO / baseline_path.relative_to(tmp_path)).read_text())
    ordinary = editor.source_config(BASE)
    standard_stats(ordinary, batch_size=4)
    generated = create("test")
    module = "training_configs.experiments.tomato_sharpa.template.test.delta_paper_t3"
    assert module in {item["id"] for item in editor.personal_catalog()}
    compute(build(trial="t3", mode="timestamp_zscore_arms", beta=0.5), batch_size=4)
    real_load = paper.load_bundle
    loaded = []

    def load(*args, **kwargs):
        loaded.append(args[0])
        return real_load(*args, **kwargs)

    monkeypatch.setattr(paper, "load_bundle", load)
    for selected, is_paper in ((BASE, False), (module, True)):
        original = editor.source_config(selected)
        assets = Path(original.data.assets.assets_dir) / original.data.assets.asset_id
        before = {p.name: p.read_bytes() for p in assets.iterdir() if p.is_file()}
        for delay in (0, 10):
            loaded.clear()
            defaults = editor.defaults(selected)
            assert defaults["protocol"]["model_family"] == ("Paper" if is_paper else "pi05")
            assert ("paper_algorithm" in defaults["protocol"]) == is_paper
            values = {**defaults["values"], "rtc_max_delay": delay, "exp_name": f"case_{is_paper}_{delay}"}
            request = {"baseline": selected, "owner": "test", "values": values, "wandb_mode": "offline"}
            preview = ui.preview(request, check_data=True)
            assert preview["statistics"] == "已验证，可复用"
            shown = preview["config"]["fields"]["model"]["fields"]
            assert shown["rtc_max_delay"] == delay
            assert ("paper_beta" in shown) == is_paper
            saved = editor.personal_source(selected, "test", values)
            destination = editor.personal_path(selected, "test", values["exp_name"])
            destination.write_text(saved)
            restored = editor.source_config(".".join(destination.relative_to(tmp_path).with_suffix("").parts))
            assert restored.model.rtc_max_delay == delay
            assert restored.model.__class__ is (PaperPi0Config if is_paper else Pi0Config)
            assert isinstance(restored.data, PaperSharpaData) == is_paper
            assert restored.data.assets == original.data.assets
            assert restored.data.lerobot_datasets == original.data.lerobot_datasets
            assert norm_provenance.build_contract(restored) == norm_provenance.build_contract(original)
            assert run_training.preflight(restored)["norm_provenance"] == "verified"
            from openpi.training.reproducibility import _training_contract

            contract = _training_contract(restored, restored.data.create(restored.assets_dirs, restored.model))
            assert contract["model"]["fields"]["rtc_max_delay"] == delay
            assert ("paper_bundle_sha256" in contract) == is_paper
            if is_paper:
                assert restored.model.paper_beta == 0.5
                assert restored.model.paper_mode == "timestamp_zscore_arms"
                assert restored.model.paper_asset_sha256 == original.model.paper_asset_sha256
                assert loaded
            else:
                assert not hasattr(restored.model, "paper_beta")
                assert not loaded
                assert not (assets / paper.BUNDLE).exists()
        assert before == {p.name: p.read_bytes() for p in assets.iterdir() if p.is_file()}
    # Opt-in generation is discoverable without replacing the original five configs or assets.
    old = (generated / "delta_paper_t3.py").read_bytes()
    with_rtc = create("test", rtc_max_delay=10)
    new_module = "training_configs.experiments.tomato_sharpa.template.test.rtc_d10.delta_paper_t3_rtc_d10"
    assert new_module in {item["id"] for item in editor.personal_catalog()}
    new = editor.source_config(new_module)
    previous = editor.source_config(module)
    assert new.model.rtc_max_delay == 10
    assert new.exp_name != previous.exp_name
    assert new.data.assets == previous.data.assets
    assert new.model.paper_asset_sha256 == previous.model.paper_asset_sha256
    assert old == (generated / "delta_paper_t3.py").read_bytes()
    assert with_rtc.name == "rtc_d10"
    with pytest.raises(ValueError, match="固定"):
        editor.apply_overrides(
            previous, {**editor.config_values(previous), "action_semantics": "absolute", "asset_id": "new_stats"}
        )
    with pytest.raises(ValueError, match="仅支持epoch_shuffle"):
        editor.apply_overrides(previous, {**editor.config_values(previous), "sampling_mode": "weighted_replacement"})
    with pytest.raises(ValueError, match="shape differs"):
        editor.apply_overrides(previous, {**editor.config_values(previous), "action_horizon": 49})
    with pytest.raises(ValueError, match="mode differs"):
        run_training.preflight(
            replace(
                previous,
                model=replace(previous.model, paper_mode="global_quantile"),
                data=replace(previous.data, paper_mode="global_quantile"),
            )
        )
    with pytest.raises(ValueError, match="beta differs"):
        run_training.preflight(
            replace(previous, model=replace(previous.model, paper_beta=0), data=replace(previous.data, paper_beta=0))
        )


@pytest.mark.parametrize("rtc_delay", [0, 2])
def test_paper_beta_zero_keeps_standard_pi0_with_and_without_rtc(prepared, monkeypatch, rtc_delay):
    from dataclasses import fields

    build, _, _ = prepared
    compute(build(), batch_size=3)
    config = build()
    config = replace(config, model=replace(config.model, rtc_max_delay=rtc_delay, dtype="float32", max_token_len=4))
    shrink_network(monkeypatch)
    paper_model = config.model.create(jax.random.key(9))
    plain_config = Pi0Config(**{f.name: getattr(config.model, f.name) for f in fields(Pi0Config)})
    plain_model = plain_config.create(jax.random.key(9))
    obs, actions = config.model.fake_obs(1), config.model.fake_act(1)
    key = jax.random.key(7)
    np.testing.assert_array_equal(
        paper_model.compute_loss(key, obs, actions), plain_model.compute_loss(key, obs, actions)
    )
    np.testing.assert_array_equal(
        paper_model.sample_actions(key, obs, num_steps=2, prefix_actions=actions, prefix_length=rtc_delay),
        plain_model.sample_actions(key, obs, num_steps=2, prefix_actions=actions, prefix_length=rtc_delay),
    )
