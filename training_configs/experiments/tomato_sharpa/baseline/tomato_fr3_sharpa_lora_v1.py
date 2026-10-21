"""Official Sharpa lora: converter 116D storage projected to 58D RGB training."""

from training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL
from training_configs.experiments.tomato_sharpa.baseline.dataset import build_dataset
from training_configs.recipes.pi05_lora_v1 import build_config as build_recipe


def build_config():
    return build_recipe(
        name="pi05_fr3_sharpa_1016_2_multimodal_rgb_lora_v1_absolute_30hz",
        exp_name="tomato_sharpa_1016_2_multimodal_lora_absolute_s42",
        embodiment=FR3_SHARPA_MULTIMODAL,
        dataset=build_dataset(),
    )
