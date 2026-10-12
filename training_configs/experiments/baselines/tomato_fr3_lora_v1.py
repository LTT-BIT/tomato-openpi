"""Reference LoRA v1 composition, with portable local paths and run metadata."""

from training_configs.datasets.tomato_fr3_fulltask_v1 import build_dataset
from training_configs.embodiments.fr3_wuji import FR3_WUJI
from training_configs.recipes.pi05_lora_v1 import build_config as build_recipe


def build_config():
    return build_recipe(
        name="pi05_fr3_wuji_fulltask_lora_v1_30hz",
        exp_name="tomato_fulltask_lora_v1_30hz_s42",
        embodiment=FR3_WUJI,
        dataset=build_dataset(),
    )
