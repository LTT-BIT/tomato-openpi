"""Lemon embodiment and recorded train split with the shared tomato 30k LoRA recipe."""

from training_configs.embodiments.dual_ur_o30 import DualURO30
from training_configs.experiments.lemon_scooping.baseline.common import build_dataset
from training_configs.experiments.lemon_scooping.baseline.common import with_recorded_split
from training_configs.recipes.pi05_lora_v1 import build_config as build_recipe


def build_config():
    return with_recorded_split(
        build_recipe(
            name="pi05_lemon_ur_o30_lora_v1_10hz",
            exp_name="lemon_lora_v1_10hz_s42",
            embodiment=DualURO30(model_action_dim=32),
            dataset=build_dataset(),
        )
    )
