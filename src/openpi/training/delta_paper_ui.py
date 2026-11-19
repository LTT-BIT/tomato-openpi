"""Select the paper options while preserving four-camera and history recipes."""

from dataclasses import fields
from dataclasses import replace

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_delta_paper import PaperPi0Config
from openpi.models.pi0_delta_paper import PaperVisualMemoryConfig
from openpi.models.pi0_delta_paper import PaperWeightLoader
from openpi.models.pi0_visual_memory import VisualMemoryConfig
from openpi.shared import delta_paper as paper
from openpi.training.config import LeRobotFr3SharpaCropDataConfig
from openpi.training.delta_paper import PaperSharpaData
from openpi.training.delta_paper import PaperVisualMemoryData
from openpi.training.delta_paper import PaperVisualMemoryOptionsData
from openpi.training.sampling import PLAN_MODES
from openpi.training.visual_memory import LeRobotVisualMemoryDataConfig
from openpi.training.visual_memory_options import LeRobotVisualMemoryOptionsDataConfig
from openpi.training.weight_loaders import PartialCheckpointWeightLoader

TRIALS = {
    "t1": ("timestamp_zscore_arms", 0.0),
    "t2": ("global_quantile", 0.5),
    "t3": ("timestamp_zscore_arms", 0.5),
}


def selected_trial(config):
    if not hasattr(config.data, "paper_contract"):
        return "none"
    pair = config.data.paper_mode, config.data.paper_beta
    return next((trial for trial, algorithm in TRIALS.items() if algorithm == pair), "custom")


def paper_available(config):
    return (
        type(config.data)
        in (
            LeRobotFr3SharpaCropDataConfig,
            LeRobotVisualMemoryDataConfig,
            LeRobotVisualMemoryOptionsDataConfig,
            PaperSharpaData,
            PaperVisualMemoryData,
            PaperVisualMemoryOptionsData,
        )
        and type(config.model) in (Pi0Config, VisualMemoryConfig, PaperPi0Config, PaperVisualMemoryConfig)
        and isinstance(config.weight_loader, PartialCheckpointWeightLoader)
        and config.model.pi05
        and config.model.action_dim == 58
        and "lora" not in config.model.paligemma_variant
        and "lora" not in config.model.action_expert_variant
        and config.data.action_hz == 30
        and config.data.extra_delta_transform
        and len(config.data.lerobot_datasets) == 1
        and not getattr(config.data.base_config, "task_view", None)
    )


def select_trial(source, config, trial):
    previous = selected_trial(source)
    if trial == previous == "none":
        return config
    if not paper_available(config) or trial not in ("none", *TRIALS):
        raise ValueError("逐时间步归一化和相关高斯噪声仅支持兼容的 30 Hz Sharpa π0.5 全参配置")
    if trial != previous and config.data.assets.asset_id == source.data.assets.asset_id:
        raise ValueError("切换论文功能需要独立的统计 asset_id")
    if not config.data.extra_delta_transform or config.sampling_mode not in PLAN_MODES:
        raise ValueError("论文功能要求双臂 delta、双手 absolute 和可复现采样")

    def inherited(cls, obj):
        return {field.name: getattr(obj, field.name) for field in fields(cls)}

    if isinstance(config.data, LeRobotVisualMemoryOptionsDataConfig):
        data_cls, model_cls = LeRobotVisualMemoryOptionsDataConfig, VisualMemoryConfig
        paper_data_cls, paper_model_cls = PaperVisualMemoryOptionsData, PaperVisualMemoryConfig
    elif isinstance(config.model, VisualMemoryConfig):
        data_cls, model_cls = LeRobotVisualMemoryDataConfig, VisualMemoryConfig
        paper_data_cls, paper_model_cls = PaperVisualMemoryData, PaperVisualMemoryConfig
    else:
        data_cls, model_cls = LeRobotFr3SharpaCropDataConfig, Pi0Config
        paper_data_cls, paper_model_cls = PaperSharpaData, PaperPi0Config
    metadata = dict(config.policy_metadata)
    loader_args = inherited(PartialCheckpointWeightLoader, config.weight_loader)
    model_args = inherited(model_cls, config.model)
    data_args = inherited(data_cls, config.data)
    name = config.name.removesuffix(f"_paper_{previous}")
    if trial == "none":
        for key in ("paper_trial", "paper_algorithm", "statistics_shape"):
            metadata.pop(key, None)
        return replace(
            config,
            name=name,
            data=data_cls(**data_args),
            model=model_cls(**model_args),
            weight_loader=PartialCheckpointWeightLoader(**loader_args),
            policy_metadata=metadata,
        )

    mode, beta = TRIALS[trial]
    algorithm = {"paper_mode": mode, "paper_beta": beta}
    if hasattr(source.data, "paper_contract"):
        algorithm["paper_std_floor"] = source.data.paper_std_floor
        data_args.update(
            correlation_max_windows=source.data.correlation_max_windows, correlation_seed=source.data.correlation_seed
        )
    data = paper_data_cls(**data_args, **algorithm)
    directory = data.asset_directory(config.assets_dirs)
    digest = paper.sha256(directory / paper.BUNDLE) if (directory / paper.BUNDLE).is_file() else None
    model = paper_model_cls(**model_args, **algorithm, paper_asset_dir=str(directory), paper_asset_sha256=digest)
    horizon = model.action_horizon
    if digest is not None:
        paper.load_bundle(
            directory, mode=mode, beta=beta, horizon=horizon, std_floor=model.paper_std_floor, expected_sha256=digest
        )
    metadata.update(
        paper_trial=trial,
        paper_algorithm=data.paper_contract(),
        statistics_shape={"state": [58], "actions": [horizon, 58]},
    )
    return replace(
        config,
        name=f"{name}_paper_{trial}",
        data=data,
        model=model,
        weight_loader=PaperWeightLoader(**loader_args),
        policy_metadata=metadata,
    )
