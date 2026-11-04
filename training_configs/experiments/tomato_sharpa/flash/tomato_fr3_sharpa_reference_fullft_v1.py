"""Native pi05 reference using exactly the FLASH train/validation windows and stats."""

from dataclasses import fields
from dataclasses import replace

from openpi.models.pi0_config import Pi0Config
from openpi.training.flash_native import bind_stats
from training_configs.experiments.tomato_sharpa.flash.tomato_fr3_sharpa_flash_fullft_v1 import (
    build_config as flash_config,
)


def build_config():
    config = flash_config()
    model = Pi0Config(**{f.name: getattr(config.model, f.name) for f in fields(Pi0Config)})
    return bind_stats(
        replace(
            config,
            name="pi05_sharpa_absolute_flash_reference_fullft_v1",
            exp_name="sharpa_absolute_flash_reference_fullft_s42",
            model=model,
            freeze_filter=model.get_freeze_filter(),
            policy_metadata={**config.policy_metadata, "algorithm": "flash_reference"},
        )
    )
