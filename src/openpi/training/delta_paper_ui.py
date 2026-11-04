"""UI-only adapter: preserve data identity without changing normalization code."""

from dataclasses import fields
from dataclasses import replace
from pathlib import Path

from openpi.shared import delta_paper as paper

TRIALS = {
    "t0": ("global_quantile", 0.0),
    "t1": ("timestamp_zscore_arms", 0.0),
    "t2": ("global_quantile", 0.5),
    "t3": ("timestamp_zscore_arms", 0.5),
    "tg": ("global_zscore_arms", 0.0),
}


def selected_trial(config):
    if not getattr(config.data, "paper_contract", None):
        return "none"
    return next(
        (trial for trial, pair in TRIALS.items() if pair == (config.data.paper_mode, config.data.paper_beta)),
        "custom",
    )


def paper_available(config):
    from openpi.training.config import LeRobotFr3SharpaMultimodalDataConfig

    return (
        isinstance(config.data, LeRobotFr3SharpaMultimodalDataConfig)
        and config.policy_metadata.get("algorithm") not in ("flash", "flash_reference")
        and config.model.pi05
        and config.model.action_dim == 58
        and "lora" not in config.model.paligemma_variant
        and "lora" not in config.model.action_expert_variant
        and config.data.action_hz == 30
        and not getattr(config.data.base_config, "task_view", None)
    )


def select_trial(config, trial, asset_id):
    """Convert UI recipes without changing public baselines or existing assets."""
    if trial == selected_trial(config):
        return config
    if trial not in ("none", *TRIALS):
        raise ValueError("未知论文功能组合")
    if not paper_available(config):
        raise ValueError("论文功能仅支持30Hz Sharpa multimodal π0.5全参训练")
    if asset_id == config.data.assets.asset_id:
        raise ValueError("切换论文功能必须使用新的 asset_id,不能混用归一化统计")
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.pi0_delta_paper import PaperPi0Config
    from openpi.models.pi0_delta_paper import PaperWeightLoader
    from openpi.training.config import LeRobotFr3SharpaMultimodalDataConfig
    from openpi.training.delta_paper import PaperSharpaData
    from openpi.training.weight_loaders import PartialCheckpointWeightLoader

    def inherited(cls, obj):
        return {field.name: getattr(obj, field.name) for field in fields(cls)}

    model_args = inherited(Pi0Config, config.model)
    data_args = inherited(LeRobotFr3SharpaMultimodalDataConfig, config.data)
    loader_args = inherited(PartialCheckpointWeightLoader, config.weight_loader)
    metadata = {
        key: value
        for key, value in config.policy_metadata.items()
        if key not in ("paper_trial", "paper_algorithm", "statistics_shape", "deployment_compatibility")
    }
    if trial == "none":
        model = Pi0Config(**model_args)
        data = LeRobotFr3SharpaMultimodalDataConfig(**data_args)
        loader = PartialCheckpointWeightLoader(**loader_args)
    else:
        mode, beta = TRIALS[trial]
        data_args["extra_delta_transform"] = True
        data = PaperSharpaData(**data_args, paper_mode=mode, paper_beta=beta)
        model = PaperPi0Config(**model_args, paper_mode=mode, paper_beta=beta)
        loader = PaperWeightLoader(**loader_args)
        metadata.update(
            paper_trial=trial,
            paper_algorithm=data.paper_contract(),
            statistics_shape={"state": [58], "actions": [model.action_horizon, 58]},
            deployment_compatibility="requires_new_paper_profile; existing_profiles_reject_Hx58_stats",
        )
    return replace(
        config,
        name=f"{config.name}_ui_{trial}",
        model=model,
        data=data,
        weight_loader=loader,
        policy_metadata=metadata,
    )


def adapt_ui_config(source, result):
    """Preserve sealed sources; keep model statistics aligned with UI storage."""
    data_source = source.data
    before = [str(Path(spec.root).expanduser().resolve()) for spec in data_source.lerobot_datasets]
    after = [str(Path(spec.root).expanduser().resolve()) for spec in result.data.lerobot_datasets]
    if after != before or result.data.repo_id != data_source.repo_id:
        raise ValueError("论文实验的数据目录、顺序和repo_id由个人配置固定,请在个人文件中定义新数据版本")
    for key in ("dataset_version", "source_datasets"):
        if result.policy_metadata[key] != source.policy_metadata[key]:
            raise ValueError("论文实验的数据版本和来源由个人配置固定,请定义新的数据配置")
    if not result.data.extra_delta_transform:
        raise ValueError("论文实验固定为双臂 delta、双手 absolute")
    if result.sampling_mode != "epoch_shuffle" or result.sampling_groups:
        raise ValueError("论文实验仅支持epoch_shuffle,不支持UI加权重复采样")
    data = replace(result.data, lerobot_datasets=data_source.lerobot_datasets)
    directory = data.asset_directory(result.assets_dirs)
    model = replace(
        result.model,
        paper_asset_dir=str(directory),
        paper_asset_sha256=paper.sha256(directory / paper.BUNDLE) if (directory / paper.BUNDLE).is_file() else None,
    )
    if (directory / paper.BUNDLE).is_file():
        # Fail before saving a UI snapshot that would reuse an incompatible asset.
        paper.load_bundle(
            directory,
            mode=model.paper_mode,
            beta=model.paper_beta,
            horizon=model.action_horizon,
            std_floor=model.paper_std_floor,
            expected_sha256=model.paper_asset_sha256,
        )
    return replace(
        result,
        data=data,
        model=model,
        policy_metadata={**result.policy_metadata, "training_action_semantics": "delta_arms_absolute_hands"},
    )
