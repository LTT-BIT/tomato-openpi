"""Recorded wrist-reset full tuning with the shared baseline learning rate."""

from dataclasses import fields

from openpi.training.config import ScheduledTrainConfig
from training_configs.embodiments.dual_ur_o30 import DualURO30
from training_configs.experiments.lemon_scooping.baseline.common import build_dataset
from training_configs.experiments.lemon_scooping.baseline.common import with_recorded_split
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe


def build_config():
    base = with_recorded_split(
        build_recipe(
            name="pi05_lemon_ur_o30_fullft_v1_10hz",
            exp_name="lemon_fullft_v1_10hz_s42",
            embodiment=DualURO30(model_action_dim=32),
            dataset=build_dataset(),
        )
    )
    values = {field.name: getattr(base, field.name) for field in fields(base)}
    values.update(
        batch_size=64,
        num_workers=8,
        log_interval=20,
        save_interval=1000,
        keep_period=1000,
        project_name="pi05-lemon-full-ft",
        policy_metadata={
            **base.policy_metadata,
            "recipe_version": "lemon_fullft_record_v1",
            "baseline_version": "lemon_fullft_record_v1",
            "source_baseline": "lemon_wrist_reset_full_ft_30k",
        },
    )
    return ScheduledTrainConfig(
        **values,
        save_before_step=20000,
        save_steps_before=(5000, 10000, 15000),
        save_interval_after=1000,
    )
