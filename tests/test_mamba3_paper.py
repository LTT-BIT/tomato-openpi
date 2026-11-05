"""Paper assets/math reuse plus real NNX Mamba optimization; diagnostic VLM only."""
# ruff: noqa: PLC0415

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_delta_paper_training import prepared as paper_prepared

from openpi import transforms
from openpi.models.pi0_delta_paper import PaperPi0
from openpi.models.pi0_mamba import MambaPi0
from openpi.models.pi0_mamba_paper import PaperMambaPi0
from openpi.models.pi0_mamba_paper import PaperMambaPi0Config
from openpi.shared import delta_paper as paper
from openpi.training import config as cfg
from openpi.training import data_loader
from openpi.training import norm_provenance
from openpi.training.delta_paper_stats import compute
from openpi.training.mamba_experiment import with_mamba
from openpi.training.mamba_training import MambaPaperDataView
from openpi.training.mamba_training import load_mamba_checkpoint
from openpi.training.mamba_training import model_data_config
from scripts.create_delta_paper_experiments import TRIALS
from scripts.mamba3_diagnostics import diagnostic_config
from scripts.mamba3_diagnostics import diagnostic_network
from scripts.mamba3_diagnostics import record_evidence
from scripts.mamba3_smoke_runtime import run_updates


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    monkeypatch.setenv("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    with jax.default_matmul_precision("highest"):
        yield paper_prepared.__wrapped__(tmp_path, monkeypatch, request)


def diagnostic_options():
    config = diagnostic_config()
    keep = (
        "diagnostic_only",
        "paligemma_variant",
        "dtype",
        "max_token_len",
        "mamba",
        "mamba_depth",
        "mamba_mlp_dim",
        "film_mlp_dim",
        "context_heads",
        "context_vision_tokens",
        "context_language_tokens",
        "context_global_tokens",
    )
    return {name: getattr(config, name) for name in keep}


@pytest.mark.parametrize(("trial", "mode", "beta"), TRIALS)
def test_shared_statistics_transforms_noise_and_plain_equivalence(prepared, monkeypatch, trial, mode, beta):
    build, root, _ = prepared
    source = build(trial=trial, mode=mode, beta=beta)
    compute(source, batch_size=3)
    source = build(trial=trial, mode=mode, beta=beta)
    directory = source.data.asset_directory(source.assets_dirs)
    before = {p.name: paper.sha256(p) for p in directory.iterdir() if p.is_file()}
    config = with_mamba(source, model_options=diagnostic_options())
    assert config.data is source.data
    assert norm_provenance.build_contract(config) == norm_provenance.build_contract(source)
    assert config.model.paper_asset_sha256 == source.model.paper_asset_sha256
    view = model_data_config(config)
    assert isinstance(view, MambaPaperDataView)
    original = source.data.create(source.assets_dirs, source.model)
    # Same raw sample/normalization/timestamp bridge; no architecture-specific transforms.
    reference_source = data_loader.create_torch_dataset(
        original, source.model.action_horizon, source.model, sampling_mode="epoch_shuffle"
    )
    raw = reference_source[0]
    for data in (original, view):
        transformed = transforms.compose(
            [
                *data.repack_transforms.inputs,
                *data.data_transforms.inputs,
                transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                *data.model_transforms.inputs,
            ]
        )(raw)
        if data is original:
            expected = transformed
        else:
            for key in ("state", "actions"):
                np.testing.assert_array_equal(transformed[key], expected[key])
    with diagnostic_network():
        model = config.model.create(jax.random.key(3))
        rng = jax.random.key(41)
        shape = (3, config.model.action_horizon, 58)
        eps = jax.random.normal(rng, shape)
        if beta:
            _, arrays = paper.load_bundle(
                directory,
                mode=mode,
                beta=beta,
                horizon=config.model.action_horizon,
                std_floor=config.model.paper_std_floor,
            )
            expected_noise = (eps.reshape(3, -1) @ arrays["noise_cholesky"].T).reshape(shape)
            np.testing.assert_array_equal(model.noise(rng, shape), expected_noise)
            assert not nnx.state(model, nnx.Param).to_pure_dict().get("paper_noise_cholesky")
        else:
            np.testing.assert_array_equal(model.noise(rng, shape), eps)
            # T1 changes only external transforms; on identical normalized actions
            # noise/time/loss/sampling are exactly the plain Mamba path.
            plain = diagnostic_config()
            plain = replace(plain, action_horizon=config.model.action_horizon)
            baseline = plain.load(nnx.state(model).to_pure_dict(), remove_extra_params=False)
            obs = plain.fake_obs(1)
            actions = jax.random.normal(rng, (1, config.model.action_horizon, 58)) * 0.1
            np.testing.assert_array_equal(
                model.compute_loss(rng, obs, actions), baseline.compute_loss(rng, obs, actions)
            )
            np.testing.assert_array_equal(
                model.sample_actions(rng, obs, num_steps=2), baseline.sample_actions(rng, obs, num_steps=2)
            )
        with monkeypatch.context() as patch:
            observed = {}
            patch.setattr(MambaPi0, "encode_condition", lambda self, obs: None)

            def velocity(self, x, time, condition):
                observed["x"], observed["time"] = x, time
                return jnp.ones_like(x) * 0.25

            patch.setattr(MambaPi0, "velocity", velocity)
            batch = 3
            actions = jax.random.normal(jax.random.key(4), shape) * 0.2
            keys = jax.random.split(rng, 3)
            t = jax.random.beta(keys[2], 1.5, 1.0, (batch,)) * 0.999 + 0.001
            epsilon = model.noise(keys[1], shape)
            loss = model.compute_loss(rng, config.model.fake_obs(batch), actions)
            np.testing.assert_array_equal(observed["time"], t)
            np.testing.assert_array_equal(observed["x"], t[:, None, None] * epsilon + (1 - t[:, None, None]) * actions)
            np.testing.assert_allclose(loss, ((0.25 - (epsilon - actions)) ** 2).mean(-1), rtol=1e-6)
            # Explicit initial noise is already in model space; never correlate it twice.
            patch.setattr(MambaPi0, "velocity", lambda self, x, time, condition: jnp.zeros_like(x))
            explicit = jax.random.normal(rng, (1, config.model.action_horizon, 58))
            np.testing.assert_array_equal(
                model.sample_actions(rng, config.model.fake_obs(), noise=explicit, num_steps=2), explicit
            )
        from openpi.models import model as model_lib
        from openpi.training.mamba_weight_loader import PaperMambaVlmInitLoader

        target = nnx.state(model).to_pure_dict()
        reference = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), target)
        with monkeypatch.context() as patch:
            patch.setattr(model_lib, "restore_params", lambda *args, **kwargs: {"PaliGemma": target["PaliGemma"]})
            loader = PaperMambaVlmInitLoader(str(root))
            loaded = loader.load(reference)
            if beta:
                assert loaded["paper_noise_cholesky"] is reference["paper_noise_cholesky"]
            patch.setattr(
                model_lib,
                "restore_params",
                lambda *args, **kwargs: {"PaliGemma": {"img": {}, "llm": target["PaliGemma"]["llm"]}},
            )
            with pytest.raises(ValueError, match="Missing required"):
                loader.load(reference)
        assert PaperMambaPi0.compute_loss is MambaPi0.compute_loss
        assert PaperMambaPi0.sample_actions is MambaPi0.sample_actions
        assert PaperMambaPi0.noise is PaperPi0.noise
    assert before == {p.name: paper.sha256(p) for p in directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("prepared", [50], indirect=True)
@pytest.mark.parametrize(("trial", "mode", "beta"), [x for x in TRIALS if x[0] in ("t1", "t2", "t3")])
def test_real_h50_training_checkpoint_new_process_policy(prepared, monkeypatch, tmp_path, trial, mode, beta):
    from openpi.policies.mamba_policy import create_mamba_policy
    from openpi.training import optimizer
    from openpi.training.weight_loaders import NoOpWeightLoader

    if jax.default_backend() == "cpu" and os.environ.get("MAMBA_PAPER_ISOLATED") != trial:
        # The fixture replaces global Linen constructors. Run each CPU training
        # case in a clean process, independent of previous JAX/NNX test caches.
        command = [
            sys.executable,
            "-m",
            "pytest",
            f"{Path(__file__).resolve()}::test_real_h50_training_checkpoint_new_process_policy[{trial}-{mode}-{beta}-50]",
            "-q",
            "--disable-warnings",
        ]
        result = subprocess.run(
            command,
            env={**os.environ, "MAMBA_PAPER_ISOLATED": trial, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    build, root, _ = prepared
    source = build(trial=trial, mode=mode, beta=beta)
    compute(source, batch_size=3)
    source = build(trial=trial, mode=mode, beta=beta)
    rtc_delay = int(os.environ.get("MAMBA_SMOKE_RTC_MAX_DELAY", "0"))
    source = replace(source, model=replace(source.model, rtc_max_delay=rtc_delay))
    config = with_mamba(source, model_options=diagnostic_options())
    config = replace(
        config,
        batch_size=1,
        num_workers=0,
        fsdp_devices=1,
        num_train_steps=5,
        weight_loader=NoOpWeightLoader(),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=0.001, decay_steps=100, decay_lr=0.0001),
    )
    directory = (
        Path(os.environ["MAMBA_SMOKE_EVIDENCE"]) / f"train_{trial}"
        if os.environ.get("MAMBA_SMOKE_EVIDENCE")
        else tmp_path / trial
    )
    directory.mkdir()

    def numeric_model_inputs(factory, model_config):
        def tokens(data):
            return {
                **{k: v for k, v in data.items() if k != "prompt"},
                "tokenized_prompt": np.ones(model_config.max_token_len, np.int32),
                "tokenized_prompt_mask": np.ones(model_config.max_token_len, bool),
            }

        return transforms.Group(inputs=(transforms.ResizeImages(224, 224), tokens))

    monkeypatch.setattr(cfg.ModelTransformFactory, "__call__", numeric_model_inputs)
    view = model_data_config(config)
    # Use the real collection/window/transform/normalization loader, not direct fake actions.
    loader = data_loader.create_data_loader(config, shuffle=False, num_batches=1)
    assert isinstance(loader.data_config(), MambaPaperDataView)
    batch = next(iter(loader))
    assert batch[1].shape == (1, 50, 58)
    assert all(value.shape == (1, 224, 224, 3) for value in batch[0].images.values())
    assert batch[0].tokenized_prompt.shape == (1, config.model.max_token_len)
    with diagnostic_network():
        import wandb

        with wandb.init(
            project="openpi-mamba3-smoke",
            name=f"diagnostic_{trial}",
            mode="offline",
            dir=str(directory),
            config={"trial": trial, "diagnostic_only": True},
        ):
            state = run_updates(config, batch, directory, 5, diagnostic=True)
        assert int(state.step) == 6
        record = json.loads((directory / "checkpoint_roundtrip.json").read_text())
        assert record["sample_max_abs"] == 0
        assert record["resume_exact"]
        step = directory / "checkpoints/5"
        restored = load_mamba_checkpoint(config.model, step, view.asset_id)
        if beta:
            _, arrays = paper.load_bundle(
                view.paper_asset_dir, mode=mode, beta=beta, horizon=50, std_floor=config.model.paper_std_floor
            )
            np.testing.assert_array_equal(restored.paper_noise_cholesky.value, arrays["noise_cholesky"])
        for name in (paper.BUNDLE, paper.MANIFEST, "norm_metadata.json", "norm_stats.json"):
            assert paper.sha256(step / "assets" / view.asset_id / name) == paper.sha256(
                Path(view.paper_asset_dir) / name
            )
        actual_source = data_loader.create_torch_dataset(view, 50, config.model, sampling_mode="epoch_shuffle")
        raw = actual_source[2]
        vector = transforms.compose([*view.repack_transforms.inputs, *view.data_transforms.inputs])(raw)
        reference = vector["actions"].copy()
        absolute = transforms.AbsoluteActions(transforms.make_bool_mask(7, -22, 7, -22))(
            {"state": vector["state"], "actions": reference}
        )["actions"]
        normalized = transforms.Normalize(view.norm_stats, use_quantiles=True)(vector)
        bridge = view.model_transforms.inputs[0]
        target = bridge(normalized)["actions"]
        # A known normalized prediction must decode back onto the physical current state,
        # preserving absolute hands and canonical order, exactly once.
        monkeypatch.setattr(PaperMambaPi0, "sample_actions", lambda self, *args, **kwargs: jnp.asarray(target)[None])
        policy = create_mamba_policy(restored, view)
        result = policy.infer(raw, rng=jax.random.key(2))["actions"]
        np.testing.assert_allclose(result[0], absolute, atol=2e-5)
        if rtc_delay:
            from openpi.training.delta_paper_prefix import encode_prefix

            captured = {}

            def with_prefix(self, *args, **kwargs):
                captured.update(kwargs)
                return jnp.asarray(target)[None]

            monkeypatch.setattr(PaperMambaPi0, "sample_actions", with_prefix)
            prefix = absolute[:rtc_delay].copy()
            prefix[:, 0] += 0.1
            prefix[:, 29] += 0.2
            reply = policy.infer(raw, rng=jax.random.key(2), absolute_prefix=prefix)["actions"][0]
            np.testing.assert_array_equal(reply[:rtc_delay], prefix)
            expected_prefix = encode_prefix(view, vector["state"], prefix)
            np.testing.assert_array_equal(np.asarray(captured["prefix_actions"])[0], expected_prefix.astype(np.float32))
            assert captured["prefix_length"] == rtc_delay
        record_evidence(
            f"fusion_{trial}",
            {
                "diagnostic_only": True,
                "updates": 5,
                "rtc_max_delay": rtc_delay,
                "new_process_restore": record,
                "physical_roundtrip_max_abs": float(np.max(np.abs(result[0] - absolute))),
            },
        )
        if beta:
            from openpi.models import model as model_lib

            damaged = nnx.state(restored).to_pure_dict()
            damaged["paper_noise_cholesky"] = damaged["paper_noise_cholesky"].at[0, 0].add(0.1)
            with monkeypatch.context() as patch:
                patch.setattr(model_lib, "restore_params", lambda *args, **kwargs: damaged)
                with pytest.raises(ValueError, match="noise factor"):
                    load_mamba_checkpoint(config.model, step, view.asset_id)
        # Restoring with a matching relocated config is supported; mode/beta drift is rejected.
        relocated = replace(config.model, paper_asset_dir=str(step / "assets" / view.asset_id))
        load_mamba_checkpoint(relocated, step, view.asset_id)
        with pytest.raises(ValueError, match="architecture"):
            load_mamba_checkpoint(replace(config.model, paper_beta=0.25), step, view.asset_id)


def test_conversion_preserves_schedules_and_rejects_resume_or_mismatches(prepared):
    from openpi.training.config import ScheduledTrainConfig
    from openpi.training.mamba_training import MambaScheduledTrainConfig

    build, _, _ = prepared
    source = build(mode="timestamp_zscore_arms", beta=0.5)
    scheduled = ScheduledTrainConfig(
        **vars(source), save_before_step=20000, save_steps_before=(5000,), save_interval_after=1000
    )
    converted = with_mamba(scheduled, model_options=diagnostic_options())
    assert isinstance(converted, MambaScheduledTrainConfig)
    assert converted.save_steps_before == (5000,)
    assert converted.data is source.data
    rtc_source = replace(source, model=replace(source.model, rtc_max_delay=2))
    rtc_config = with_mamba(rtc_source, model_options=diagnostic_options())
    assert rtc_config.data is source.data
    assert norm_provenance.build_contract(rtc_config) == norm_provenance.build_contract(source)
    with pytest.raises(ValueError, match="new experiment"):
        with_mamba(replace(source, resume=True), model_options=diagnostic_options())
    with pytest.raises(ValueError, match="mismatch"):
        replace(converted, data=replace(source.data, paper_beta=0))


def test_mamba_paper_rejects_mismatched_algorithm_and_invalid_rtc():
    assert PaperMambaPi0Config(rtc_max_delay=10).rtc_max_delay == 10
    with pytest.raises(ValueError, match="rtc_max_delay"):
        PaperMambaPi0Config(rtc_max_delay=50)
    with pytest.raises(ValueError, match="Paper model"):
        PaperMambaPi0Config(paper_beta=1)
    with pytest.raises(ValueError, match="Compute matching"):
        PaperMambaPi0Config().create(jax.random.key(1))


def test_cpu_donation_logs_own_their_snapshot_buffers():
    from scripts.mamba3_smoke_runtime import host_copy

    tree = {"value": jnp.arange(16, dtype=jnp.float32)}
    before = host_copy(tree)
    assert before["value"].flags.owndata
    assert not np.shares_memory(before["value"], np.asarray(tree["value"]))
    updated = jax.jit(lambda values: {"value": values["value"] + 1}, donate_argnums=(0,))(tree)
    np.testing.assert_array_equal(before["value"], np.arange(16))
    np.testing.assert_array_equal(updated["value"], np.arange(16) + 1)
