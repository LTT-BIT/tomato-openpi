"""Map explicit GUI feature selections to isolated FLASH F0-F4 recipes."""

from dataclasses import fields
from dataclasses import replace

from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.pi0_flash_delta_t3 import FlashDeltaT3WeightLoader
from openpi.models.pi0_flash_delta_t3 import Pi0FlashDeltaT3Config
from openpi.training.weight_loaders import PartialCheckpointWeightLoader

TRIALS = {"f1": (False, 0.0), "f2": (True, 0.0), "f3": (False, 0.5), "f4": (True, 0.5)}


def available(config):
    return (
        type(config.model) in (Pi0FlashConfig, Pi0FlashDeltaT3Config)
        and not config.model.freeze_vlm
        and config.data.action_hz == 30
        and not config.data.extra_delta_transform
    )


def selected(config):
    if type(config.model) is Pi0FlashConfig:
        return "f0"
    if type(config.model) is Pi0FlashDeltaT3Config:
        return next(
            (key for key, pair in TRIALS.items() if pair == (config.model.timestamp, config.model.correlation_beta)),
            "custom",
        )
    return "none"


def select(config, trial, asset_id):
    if trial == selected(config):
        return config
    if not available(config) or trial not in ("f0", *TRIALS):
        raise ValueError("论文勾选仅支持30Hz FLASH全参 F0-F4")
    if asset_id == config.data.assets.asset_id:
        raise ValueError("切换FLASH算法必须使用新的 asset_id")
    if type(config.weight_loader) not in (PartialCheckpointWeightLoader, FlashDeltaT3WeightLoader):
        raise ValueError("切换FLASH算法需要显式定义与基础权重匹配的加载器")
    args = {f.name: getattr(config.model, f.name) for f in fields(Pi0FlashConfig)}
    metadata = {k: v for k, v in config.policy_metadata.items() if k not in ("flash_trial", "flash_t3_version")}
    if trial == "f0":
        model = Pi0FlashConfig(**args)
        loader = PartialCheckpointWeightLoader(
            config.weight_loader.params_path, config.weight_loader.skip_on_mismatch_regex
        )
        metadata["training_action_semantics"] = "absolute"
    else:
        timestamp, beta = TRIALS[trial]
        model = Pi0FlashDeltaT3Config(**args, timestamp=timestamp, correlation_beta=beta)
        loader = FlashDeltaT3WeightLoader(config.weight_loader.params_path, config.weight_loader.skip_on_mismatch_regex)
        metadata.update(
            flash_trial=trial,
            flash_t3_version="flash-delta-t3-time-v1",
            training_action_semantics="delta_arms_absolute_hands",
        )
    return replace(
        config,
        model=model,
        weight_loader=loader,
        name=f"{config.name}_ui_{trial}",
        policy_metadata=metadata,
        freeze_filter=model.get_freeze_filter(),
    )
