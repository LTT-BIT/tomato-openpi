# ruff: noqa: RUF001
"""GUI architecture adaptation; existing Paper editing/statistics stay shared."""

from dataclasses import fields
from dataclasses import replace

from flax import nnx

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_delta_paper import PaperPi0Config
from openpi.models.pi0_delta_paper import PaperWeightLoader
from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.training import config as cfg
from openpi.training.delta_paper_ui import paper_available
from openpi.training.mamba_experiment import with_mamba
from openpi.training.mamba_training import MambaScheduledTrainConfig
from openpi.training.mamba_training import MambaSharpaData
from openpi.training.mamba_training import MambaTrainConfig
from openpi.training.mamba_weight_loader import MambaVlmInitLoader
from openpi.training.weight_loaders import PartialCheckpointWeightLoader


def architecture(config):
    return "mamba3" if isinstance(config.model, MambaPi0Config) else "transformer"


def available(config):
    return (
        paper_available(config)
        and config.freeze_filter is nnx.Nothing
        and not getattr(config.model, "diagnostic_only", False)
    )


def inherited(cls, obj):
    return {field.name: getattr(obj, field.name) for field in fields(cls)}


def editable(config):
    """Unwrap only for shared UI edits; never train this intermediate config."""
    if architecture(config) != "mamba3":
        return config
    is_paper = isinstance(config.model, PaperPi0Config)
    model_type = PaperPi0Config if is_paper else Pi0Config
    loader_type = PaperWeightLoader if is_paper else PartialCheckpointWeightLoader
    data = config.data if is_paper else cfg.LeRobotFr3SharpaMultimodalDataConfig(**vars(config.data))
    kind = cfg.ScheduledTrainConfig if isinstance(config, cfg.ScheduledTrainConfig) else cfg.TrainConfig
    metadata = {k: v for k, v in config.policy_metadata.items() if k not in ("architecture", "stage", "normalization")}
    return kind(
        **{
            **vars(config),
            "model": model_type(**inherited(model_type, config.model)),
            "data": data,
            "weight_loader": loader_type(params_path=config.weight_loader.params_path),
            "policy_metadata": metadata,
        }
    )


def select(config, source):
    """Wrap edited fields together, so model/data asset directories never drift."""
    if not available(config):
        raise ValueError("Mamba仅支持30Hz Sharpa multimodal π0.5全参训练，不支持LoRA或冻结")
    if config.model.rtc_max_delay:
        raise ValueError("Mamba当前要求RTC=0")
    if config.model.action_horizon != 50:
        raise ValueError("Mamba当前要求action_horizon=50")
    if config.sampling_mode != "epoch_shuffle" or config.sampling_groups:
        raise ValueError("Mamba当前要求epoch_shuffle，不支持加权重复采样")
    options = None
    if architecture(source) == "mamba3":
        names = {f.name for f in fields(MambaPi0Config)} - {f.name for f in fields(Pi0Config)}
        names -= {"architecture", "stage", "normalization", "action_representation"}
        options = {name: getattr(source.model, name) for name in names}
    if isinstance(config.model, PaperPi0Config):
        result = with_mamba(config, model_options=options)
    else:
        if config.data.extra_delta_transform:
            raise ValueError("普通Mamba要求absolute；双臂Delta请通过论文功能选择T0/T1/T2/T3/Tg")
        model = MambaPi0Config(**{**inherited(Pi0Config, config.model), **(options or {})})
        kind = MambaScheduledTrainConfig if isinstance(config, cfg.ScheduledTrainConfig) else MambaTrainConfig
        result = kind(
            **{
                **vars(config),
                "model": model,
                "data": MambaSharpaData(**vars(config.data)),
                "weight_loader": MambaVlmInitLoader(config.weight_loader.params_path),
                "freeze_filter": model.get_freeze_filter(),
            }
        )
    same = architecture(source) == "mamba3"
    return replace(
        result,
        name=config.name if same else config.name + "_mamba3",
        exp_name=config.exp_name,
        policy_metadata={
            **result.policy_metadata,
            **(
                {k: v for k, v in source.policy_metadata.items() if k.startswith("source_transformer_")} if same else {}
            ),
            "architecture": result.model.architecture,
            "stage": result.model.stage,
            "normalization": result.model.normalization,
            "training_status": "prepared_not_trained",
            "real_robot_status": "not_validated",
        },
    )
