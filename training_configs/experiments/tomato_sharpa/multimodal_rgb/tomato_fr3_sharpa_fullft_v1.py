"""RGB-only full-parameter recipe for a fixed multimodal converter batch."""

from training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL
from training_configs.experiments.tomato_sharpa.multimodal_rgb.dataset import build_dataset
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe


def build_config():
    return build_recipe(
        name="pi05_fr3_sharpa_multimodal_rgb_fullft_v1_absolute_30hz",
        exp_name="tomato_sharpa_multimodal_rgb_fullft_v1_absolute_30hz_s42",
        embodiment=FR3_SHARPA_MULTIMODAL,
        dataset=build_dataset(),
    )
