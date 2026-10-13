"""Cup grasping with the shared 30k lora recipe and single Franka adapter."""

import os

from training_configs import common
from training_configs.common import Dataset
from training_configs.embodiments.single_franka import SINGLE_FRANKA
from training_configs.recipes.pi05_lora_v1 import build_config as build_recipe

PROMPT = "Pick up the cup."


def build_dataset() -> Dataset:
    repository = common.repository_root()
    root = os.environ.get("PI05_CUP_DATASET_ROOT", str(repository / "data/cup_grasping_v1"))
    return Dataset(
        version="cup_grasping_v1_30hz",
        repo_id="single_franka/cup_grasping_v1",
        root=root,
        assets_dir=os.environ.get("PI05_CUP_ASSETS_DIR", str(repository / "assets")),
        asset_id="cup_grasping_v1_30hz",
        sources=(root,),
        prompt_example=PROMPT,
    )


def build_config():
    return build_recipe(
        name="pi05_single_franka_cup_lora_v1_30hz",
        exp_name="cup_grasping_lora_v1_30hz_s42",
        embodiment=SINGLE_FRANKA,
        dataset=build_dataset(),
    )
