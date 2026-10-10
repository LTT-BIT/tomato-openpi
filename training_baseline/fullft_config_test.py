"""Check that the actual full-task model has no adapters or frozen parameters."""

import flax.nnx as nnx
import jax
import numpy as np

from openpi.models import gemma
from training_baseline.pi05_fr3_wuji_fulltask_c0_train import DATASET_SOURCES
from training_baseline.pi05_fr3_wuji_fulltask_c0_train import build_config


def test_full_model_parameters_are_trainable():
    config = build_config()
    assert not gemma.get_config(config.model.paligemma_variant).lora_configs
    assert not gemma.get_config(config.model.action_expert_variant).lora_configs

    def parameter_shapes():
        model = config.model.create(jax.random.key(config.model_seed))
        return nnx.state(model, nnx.Param)

    params = jax.eval_shape(parameter_shapes)
    all_params = params.flat_state()
    trainable = params.filter(config.trainable_filter).flat_state()
    assert len(all_params) == len(trainable) > 0
    assert sum(np.prod(value.value.shape) for value in all_params.values()) == sum(
        np.prod(value.value.shape) for value in trainable.values()
    )
    assert not params.filter(config.freeze_filter).flat_state()
    assert not any("lora" in str(path).lower() for path in all_params)
    assert any("img" in str(path) for path in trainable)
    assert any("llm" in str(path) for path in trainable)
    assert any("action_out_proj" in str(path) for path in trainable)


def test_selected_data_and_reproducibility_contract():
    config = build_config()
    assert [path.relative_to(path.parents[1]).as_posix() for path in DATASET_SOURCES[:2]] == ["0925/A", "0925/B"]
    assert DATASET_SOURCES[2].as_posix().endswith("1002_AB_190/full")
    assert len(config.data.lerobot_datasets) == 1
    assert config.sampling_mode == "epoch_shuffle"
    assert config.batch_size == 8
    assert config.num_train_steps == (481_223 + config.batch_size - 1) // config.batch_size
    assert (config.seed, config.model_seed, config.data_seed, config.augmentation_seed) == (42, 42, 42001, 42002)
    assert config.data.base_config.prompt_from_task
    assert config.data.action_hz == 30
    assert config.model.action_dim == 54
    assert config.model.action_horizon == 50
    assert not config.resume
    assert not config.overwrite
    assert config.wandb_enabled
