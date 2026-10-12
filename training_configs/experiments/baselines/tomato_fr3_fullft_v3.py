"""Reference C0 v3 composition; use a new exp_name for a new run."""

from training_configs.datasets.tomato_fr3_v3 import build_dataset
from training_configs.embodiments.fr3_wuji import FR3_WUJI
from training_configs.recipes.pi05_fullft_v3 import build_config as build_recipe


def build_config():
    return build_recipe(
        name="pi05_fr3_wuji_fulltask_fullft_30hz",
        exp_name="tomato_all_full_fullft_30hz_s42",
        embodiment=FR3_WUJI,
        dataset=build_dataset(),
    )
