"""Reference full-parameter v1 composition; use a new exp_name for a new run."""

import os
from pathlib import Path

from training_configs import common
from training_configs.common import Dataset
from training_configs.embodiments.fr3_wuji import FR3_WUJI
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe

PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato "
    "with the left hand and place it in the left basket."
)


def build_dataset() -> Dataset:
    repository = common.repository_root()
    source_root = Path(os.environ.get("PI05_SOURCE_DATA_DIR", str(repository / "data/harvest-tomato")))
    return Dataset(
        version="tomato_all_full_v1_30hz",
        repo_id="fr3_wuji/tomato_all_full_v1",
        root=os.environ.get("PI05_ALL_FULL_DATASET_ROOT", str(repository / "data/tomato_all_full_v1")),
        assets_dir=os.environ.get("PI05_ALL_FULL_ASSETS_DIR", str(repository / "assets")),
        asset_id="tomato_all_full_v1_absolute_30hz",
        sources=tuple(str((source_root / item).resolve()) for item in ("0925/A", "0925/B", "1002_AB_190/full")),
        prompt_example=PROMPT,
    )


def build_config():
    return build_recipe(
        name="pi05_fr3_wuji_fulltask_fullft_v1_absolute_30hz",
        exp_name="tomato_all_full_fullft_v1_absolute_30hz_s42",
        embodiment=FR3_WUJI,
        dataset=build_dataset(),
    )
