"""Architecture conversion of existing Paper experiments; data/statistics stay identical."""

from dataclasses import fields
import os

from openpi.models.pi0_delta_paper import PaperPi0Config
from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.models.pi0_mamba_paper import PaperMambaPi0Config
from openpi.training.config import ScheduledTrainConfig
from openpi.training.delta_paper_experiment import build_experiment as build_paper
from openpi.training.mamba_training import MambaScheduledTrainConfig
from openpi.training.mamba_training import MambaTrainConfig
from openpi.training.mamba_weight_loader import PaperMambaVlmInitLoader
from scripts.create_delta_paper_experiments import TRIALS


def with_mamba(config, *, model_options=None):
    """Reuse a Transformer Paper config, including all four original asset files.

    A new experiment name is mandatory because architecture/optimizer trees change;
    this is initialization, never resume from a Transformer checkpoint.
    """
    if not isinstance(config.model, PaperPi0Config) or isinstance(config.model, MambaPi0Config):
        raise ValueError("with_mamba expects an existing Transformer Paper experiment")
    if config.model.rtc_max_delay or config.resume:
        raise ValueError("Mamba fusion currently requires RTC=0 and a new experiment, not Transformer resume")
    allowed = {field.name for field in fields(MambaPi0Config)} - {
        "architecture",
        "stage",
        "normalization",
        "action_representation",
        "rtc_max_delay",
    }
    options = model_options or {}
    if set(options) - allowed:
        raise ValueError("Unsupported Mamba model option")
    model = PaperMambaPi0Config(
        **{**{field.name: getattr(config.model, field.name) for field in fields(PaperPi0Config)}, **options}
    )
    config_type = MambaScheduledTrainConfig if isinstance(config, ScheduledTrainConfig) else MambaTrainConfig
    return config_type(
        **{
            **vars(config),
            "name": config.name + "_mamba3",
            "exp_name": config.exp_name + "_mamba3",
            "model": model,
            "freeze_filter": model.get_freeze_filter(),
            "weight_loader": PaperMambaVlmInitLoader(config.weight_loader.params_path),
            "policy_metadata": {
                **(config.policy_metadata or {}),
                "architecture": model.architecture,
                "stage": model.stage,
                "source_transformer_config": config.name,
                "source_transformer_experiment": config.exp_name,
                "training_status": "prepared_not_trained",
                "real_robot_status": "not_validated",
            },
        }
    )


def build_experiment(trial, *, owner=None, dataset=None):
    trials = {name: (mode, beta) for name, mode, beta in TRIALS}
    if trial not in trials:
        raise ValueError(f"Unknown Paper trial: {trial}")
    owner = owner or os.environ.get("PI05_MAMBA_OWNER", "mamba")
    source = os.environ.get("PI05_MAMBA_SOURCE_MODULE")
    if source:
        from scripts.run_training import load_experiment

        config = load_experiment(source)
        if (config.model.paper_mode, config.model.paper_beta) != trials[trial]:
            raise ValueError("Source experiment algorithm differs from selected Mamba trial")
    else:
        mode, beta = trials[trial]
        config = build_paper(owner=owner, trial=trial, mode=mode, beta=beta, dataset=dataset)
    return with_mamba(config)
