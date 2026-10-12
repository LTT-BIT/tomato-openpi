"""Guard baseline equivalence, real parameter freezing, and owner isolation."""

from dataclasses import replace
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import pytest

from openpi.training import config as cfg
from openpi.training.reproducibility import serialize_config
from scripts.run_training import load_experiment
from scripts.run_training import resolve_experiment
from scripts.run_training import validate_config
from training_configs.datasets.tomato_fr3_v2 import build_dataset
from training_configs.recipes.pi05_lora_v2 import build_config as build_lora

BASE = "training_configs.experiments.baselines."


@pytest.mark.parametrize("module", ["tomato_fr3_lora_v2", "tomato_fr3_fullft_v3"])
def test_composition_preserves_baseline_training_semantics(module):
    current = load_experiment(BASE + module)
    snapshot = json.loads((Path(__file__).parent / "fixtures/baselines" / f"{module}.json").read_text())
    spec = current.data.lerobot_datasets[0]
    portable = replace(
        current,
        data=replace(
            current.data,
            assets=replace(current.data.assets, assets_dir=snapshot["path_placeholders"]["assets_dir"]),
            lerobot_datasets=(replace(spec, root=snapshot["path_placeholders"]["dataset_root"]),),
        ),
    )
    current_values = json.loads(serialize_config(portable))["fields"]
    for key in snapshot["excluded_fields"]:
        current_values.pop(key)
    assert current_values == snapshot["config_fields"]
    validate_config(current)


@pytest.mark.parametrize(("module", "is_lora"), [("tomato_fr3_lora_v2", True), ("tomato_fr3_fullft_v3", False)])
def test_actual_model_trainable_scope(module, is_lora):
    config = load_experiment(BASE + module)

    def shapes():
        return nnx.state(config.model.create(jax.random.key(config.model_seed)), nnx.Param)

    params = jax.eval_shape(shapes)
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
    monkeypatch.setenv("PI05_SOURCE_DATA_DIR", "/test-data/harvest-tomato")
    config = load_experiment(BASE + "tomato_fr3_fullft_v3")
    assert config.policy_metadata["source_datasets"] == [
        "/test-data/harvest-tomato/0925/A",
        "/test-data/harvest-tomato/0925/B",
        "/test-data/harvest-tomato/1002_AB_190/full",
    ]
    assert config.num_train_steps == (481_223 + config.batch_size - 1) // config.batch_size
    assert (config.seed, config.model_seed, config.data_seed, config.augmentation_seed) == (42, 42, 42001, 42002)
    assert config.sampling_mode == "epoch_shuffle"
    assert config.data.base_config.prompt_from_task
    assert config.data.action_hz == 30
    assert config.model.action_dim == 54
    assert config.model.action_horizon == 50
    assert not config.resume
    assert not config.overwrite
    assert config.wandb_enabled


def test_personal_override_does_not_mutate_baseline():
    baseline = load_experiment(BASE + "tomato_fr3_lora_v2")
    personal = load_experiment("training_configs.experiments.template.tomato_lora_lr1e5")
    assert personal.lr_schedule.peak_lr == 1e-5
    assert baseline.lr_schedule.peak_lr == 2.5e-5
    assert personal.exp_name != baseline.exp_name
    assert personal.checkpoint_dir != baseline.checkpoint_dir
    overridden = resolve_experiment(BASE + "tomato_fr3_lora_v2", ["--batch-size", "4", "--exp-name", "alice_run01"])
    assert overridden.batch_size == 4
    assert baseline.batch_size == 8


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
        resolve_experiment(BASE + "tomato_fr3_lora_v2", ["--model.paligemma-variant", "gemma_2b"])


@pytest.mark.parametrize("change", [{"exp_name": "../other"}, {"wandb_enabled": False}, {"overwrite": True}])
def test_reject_unsafe_experiment_settings(change):
    with pytest.raises(ValueError, match="identifier|W&B|overwriting"):
        validate_config(replace(load_experiment(BASE + "tomato_fr3_lora_v2"), **change))
