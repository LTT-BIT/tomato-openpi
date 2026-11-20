"""Guard baseline equivalence, real parameter freezing, and owner isolation."""

from dataclasses import replace
import importlib

import flax.nnx as nnx
import jax
import pytest

from openpi.training import config as cfg
from openpi.training.sampling import SamplingGroup
from scripts.run_training import load_experiment
from scripts.run_training import resolve_experiment
from scripts.run_training import validate_config
from training_configs.experiments.tomato_sharpa.baseline.dataset import build_dataset
from training_configs.recipes.pi05_lora_v1 import build_config as build_lora

BASE = "training_configs.experiments.tomato_sharpa.baseline."


@pytest.mark.parametrize(
    ("module", "is_lora"),
    [
        ("tomato_fr3_sharpa_lora_v1", True),
        ("tomato_fr3_sharpa_fullft_v1", False),
    ],
)
def test_actual_model_trainable_scope(module, is_lora):
    task = "tomato_sharpa"
    config = load_experiment(f"training_configs.experiments.{task}.baseline.{module}")

    def shapes():
        return nnx.state(config.model.create(jax.random.key(config.model_seed)), nnx.Param)

    params = jax.eval_shape(shapes)
    assert params["action_in_proj"]["kernel"].value.shape[0] == config.model.action_dim
    assert params["action_out_proj"]["kernel"].value.shape[-1] == config.model.action_dim
    trainable = params.filter(config.trainable_filter).flat_state()
    frozen = params.filter(config.freeze_filter).flat_state()
    names = [str(path) for path in trainable]
    assert any("img" in path for path in names)
    assert any("llm" in path for path in names)
    assert any("action_in_proj" in path for path in names)
    assert any("action_out_proj" in path for path in names)
    if is_lora:
        assert frozen
        assert any("llm" in str(path) for path in frozen)
        assert any("lora" in path for path in names)
        assert not any("lora" in str(path) for path in frozen)
    else:
        assert not frozen
        assert len(trainable) == len(params.flat_state())
        assert not any("lora" in path for path in names)


def test_fullft_selected_data_and_reproducibility_contract(monkeypatch):
    config = load_experiment(BASE + "tomato_fr3_sharpa_fullft_v1")
    assert config.policy_metadata["source_datasets"]
    assert config.num_train_steps == 30_000
    assert (config.seed, config.model_seed, config.data_seed, config.augmentation_seed) == (42, 42, 42001, 42002)
    assert config.sampling_mode == "epoch_shuffle"
    assert config.data.base_config.prompt_from_task
    assert config.data.action_hz == 30
    assert config.model.action_dim == 58
    assert config.model.action_horizon == 50
    assert not config.resume
    assert not config.overwrite
    assert config.wandb_enabled


def test_personal_override_does_not_mutate_baseline(monkeypatch, make_personal_experiment):
    baseline = load_experiment(BASE + "tomato_fr3_sharpa_lora_v1")
    monkeypatch.setenv("PI05_SHARPA_MULTIMODAL_DATASET_ROOT", "/alice/data/tomato_v1")
    monkeypatch.setenv("PI05_SHARPA_MULTIMODAL_ASSETS_DIR", "/alice/assets")
    personal = load_experiment(make_personal_experiment(BASE + "tomato_fr3_sharpa_lora_v1", peak_lr=1e-5))
    assert personal.data.lerobot_datasets[0].root == "/alice/data/tomato_v1"
    assert personal.data.assets.assets_dir == "/alice/assets"
    assert baseline.data.lerobot_datasets[0].root != personal.data.lerobot_datasets[0].root
    assert load_experiment(BASE + "tomato_fr3_sharpa_lora_v1").data == personal.data
    assert personal.lr_schedule.peak_lr == 1e-5
    assert baseline.lr_schedule.peak_lr == 2.5e-5
    assert personal.exp_name != baseline.exp_name
    assert personal.checkpoint_dir != baseline.checkpoint_dir
    overridden = resolve_experiment(
        BASE + "tomato_fr3_sharpa_lora_v1", ["--batch-size", "4", "--exp-name", "alice_run01"]
    )
    assert overridden.batch_size == 4
    assert baseline.batch_size == 8


def test_personal_sampling_groups_survive_cli(monkeypatch, make_personal_experiment):
    module_name = make_personal_experiment(BASE + "tomato_fr3_sharpa_lora_v1")
    module = importlib.import_module(module_name)
    config = replace(
        module.build_config(),
        sampling_mode="weighted_replacement",
        sampling_groups=(
            SamplingGroup(name="A", episodes=(0, 2), weight=7),
            SamplingGroup(name="B", episodes=(1, 3), weight=3),
        ),
    )
    monkeypatch.setattr(module, "build_config", lambda: config)
    resolved = resolve_experiment(module_name, ["--exp-name", "weighted_trial", "--batch-size", "4"])
    assert resolved.sampling_mode == "weighted_replacement"
    assert resolved.sampling_groups == config.sampling_groups
    assert resolved.exp_name == "weighted_trial"
    assert resolved.batch_size == 4
    assert load_experiment(BASE + "tomato_fr3_sharpa_lora_v1").sampling_groups == ()


def test_recipe_accepts_another_embodiment_without_fr3_dimension():
    class OtherEmbodiment:
        version = "test_other_v1"
        action_dim = 14
        raw_state_dims = (14,)
        action_hz = 20.0
        camera_keys = ()

        def data_config(self, dataset):
            return cfg.FakeDataConfig(repo_id=dataset.repo_id)

    config = build_lora(name="other", exp_name="other_s42", embodiment=OtherEmbodiment(), dataset=build_dataset())
    assert config.model.action_dim == 14
    assert config.policy_metadata["embodiment_version"] == "test_other_v1"
    assert isinstance(config.data, cfg.FakeDataConfig)


def test_cli_cannot_change_model_without_regenerating_freeze_filter():
    with pytest.raises(ValueError, match="versioned recipe"):
        resolve_experiment(BASE + "tomato_fr3_sharpa_lora_v1", ["--model.paligemma-variant", "gemma_2b"])


@pytest.mark.parametrize("change", [{"exp_name": "../other"}, {"wandb_enabled": False}, {"overwrite": True}])
def test_reject_unsafe_experiment_settings(change):
    with pytest.raises(ValueError, match="identifier|W&B|overwriting"):
        validate_config(replace(load_experiment(BASE + "tomato_fr3_sharpa_lora_v1"), **change))


@pytest.mark.parametrize("field", ["log_interval", "save_interval", "keep_period"])
@pytest.mark.parametrize("value", [0, -1, 0.5, True])
def test_reject_invalid_training_intervals_before_startup(field, value):
    with pytest.raises(ValueError, match=f"{field} must be"):
        replace(load_experiment(BASE + "tomato_fr3_sharpa_lora_v1"), **{field: value})


def test_cli_rejects_zero_save_interval_without_reading_data():
    from scripts.run_training import main

    with pytest.raises(ValueError, match="save_interval must be"):
        main([BASE + "tomato_fr3_sharpa_lora_v1", "--save-interval", "0", "--dry-run"])


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_horizon_is_rejected_before_model_or_data_loading(value):
    config = load_experiment(BASE + "tomato_fr3_sharpa_lora_v1")
    with pytest.raises(ValueError, match="action_horizon must be"):
        replace(config, model=replace(config.model, action_horizon=value))
