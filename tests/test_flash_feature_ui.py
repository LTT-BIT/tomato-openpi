"""GUI feature selection preserves data/training settings and seals each algorithm."""

from dataclasses import replace

import pytest

from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.pi0_flash_delta_t3 import Pi0FlashDeltaT3Config
from openpi.training.flash_delta_t3_ui import TRIALS
from scripts import training_ui_config as editor

PREFIX = "training_configs.experiments.tomato_sharpa.flash."
MODULES = {
    "f0": PREFIX + "tomato_fr3_sharpa_flash_fullft_v1",
    **{t: PREFIX + f"tomato_fr3_sharpa_flash_{t}_v1" for t in TRIALS},
}


@pytest.mark.parametrize("source", MODULES)
@pytest.mark.parametrize("target", MODULES)
def test_select_save_reload_preserves_training_contract(source, target, tmp_path):
    module = MODULES[source]
    original = editor.source_config(module)
    values = {
        **editor.defaults(module)["values"],
        "flash_trial": target,
        "exp_name": f"ui_{source}_{target}",
        "assets_dir": str(tmp_path),
        "asset_id": f"ui_{target}",
        "batch_size": 8,
    }
    config = editor.apply_overrides(original, values)
    assert type(config.model) is (Pi0FlashConfig if target == "f0" else Pi0FlashDeltaT3Config)
    if target != "f0":
        assert (config.model.timestamp, config.model.correlation_beta) == TRIALS[target]
        assert config.model.t3_assets == str(tmp_path / values["asset_id"])
        assert not config.model.t3_sha256
    assert config.model.rtc_max_delay == 0
    assert not config.data.extra_delta_transform
    assert config.data.lerobot_datasets == original.data.lerobot_datasets
    assert config.policy_metadata["source_datasets"] == original.policy_metadata["source_datasets"]
    assert config.policy_metadata["prompt_example"] == original.policy_metadata["prompt_example"]
    assert config.num_train_steps == original.num_train_steps
    assert config.lr_schedule == original.lr_schedule
    assert config.batch_size == 8
    scope = {}
    exec(editor.personal_source(module, "tester", values), scope)
    restored = scope["build_config"]()
    assert restored.model == config.model
    assert restored.weight_loader == config.weight_loader
    assert editor.config_values(restored)["flash_trial"] == target
    assert editor.source_config(module) == original
    if source != target:
        with pytest.raises(ValueError, match="asset_id"):
            editor.apply_overrides(original, {**values, "asset_id": original.data.assets.asset_id})
    with pytest.raises(ValueError, match="RTC"):
        editor.apply_overrides(original, {**values, "rtc_max_delay": 1})


@pytest.mark.parametrize(
    "module",
    [
        "training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1",
        PREFIX + "tomato_fr3_sharpa_reference_fullft_v1",
    ],
)
def test_unsupported_recipes_cannot_enable_flash_features(module):
    defaults = editor.defaults(module)
    assert not defaults["protocol"]["flash_features_available"]
    with pytest.raises(ValueError, match="FLASH"):
        editor.apply_overrides(
            editor.source_config(module), {**defaults["values"], "flash_trial": "f4", "asset_id": "new"}
        )


def test_legacy_values_and_custom_beta_are_preserved():
    module = MODULES["f4"]
    config = editor.source_config(module)
    values = editor.config_values(config)
    del values["flash_trial"]
    assert editor.apply_overrides(config, values).model == config.model
    custom = replace(config, model=replace(config.model, correlation_beta=0.3))
    assert editor.config_values(custom)["flash_trial"] == "custom"
    assert editor.apply_overrides(custom, editor.config_values(custom)).model == custom.model
