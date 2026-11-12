"""Independent FLASH F1-F4 experiments; same all480 sources/budget as F0."""

from dataclasses import fields
from dataclasses import replace

from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.models.pi0_flash_delta_t3 import FlashDeltaT3WeightLoader
from openpi.models.pi0_flash_delta_t3 import Pi0FlashDeltaT3Config
from openpi.training.flash_delta_t3_data import bind
from training_configs.experiments.tomato_sharpa.flash.tomato_fr3_sharpa_flash_fullft_v1 import build_config as baseline


def build(trial):
    timestamp, beta = {"f1": (False, 0.0), "f2": (True, 0.0), "f3": (False, 0.5), "f4": (True, 0.5)}[trial]
    config = baseline()
    model = Pi0FlashDeltaT3Config(
        **{
            f.name: getattr(config.model, f.name)
            for f in fields(Pi0FlashConfig)
            if f.name not in ("action_q01", "action_q99")
        },
        timestamp=timestamp,
        correlation_beta=beta,
    )
    return bind(
        replace(
            config,
            model=model,
            name=f"pi05_sharpa_flash_delta_t3_{trial}_v1",
            exp_name=f"sharpa_flash_delta_t3_{trial}_s42_run01",
            data=replace(
                config.data, assets=replace(config.data.assets, asset_id=f"normal480_flash_delta_t3_{trial}_v1")
            ),
            freeze_filter=model.get_freeze_filter(),
            weight_loader=FlashDeltaT3WeightLoader(config.weight_loader.params_path),
            policy_metadata={
                **config.policy_metadata,
                "training_action_semantics": "delta_arms_absolute_hands",
                "flash_trial": trial,
                "flash_t3_version": "flash-delta-t3-time-v1",
            },
        )
    )
