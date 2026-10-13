"""Reference Sharpa fullft v1: current public recipe with the 58D physical adapter."""

import os

from training_configs import common
from training_configs.common import Dataset
from training_configs.embodiments.fr3_sharpa import FR3_SHARPA
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe

PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato "
    "with the left hand and place it in the left basket."
)


def build_dataset() -> Dataset:
    repository = common.repository_root()
    root = os.environ.get("PI05_SHARPA_DATASET_ROOT", str(repository / "data/tomato_sharpa_fulltask_v1"))
    return Dataset(
        version="tomato_sharpa_fulltask_v1_30hz",
        repo_id="fr3_sharpa/tomato_fulltask_v1",
        root=root,
        assets_dir=os.environ.get("PI05_SHARPA_ASSETS_DIR", str(repository / "assets")),
        asset_id="tomato_sharpa_fulltask_v1_30hz",
        sources=(root,),
        prompt_example=PROMPT,
    )


def build_config():
    return build_recipe(
        name="pi05_fr3_sharpa_fulltask_fullft_v1_30hz",
        exp_name="tomato_sharpa_fulltask_fullft_v1_30hz_s42",
        embodiment=FR3_SHARPA,
        dataset=build_dataset(),
    )
