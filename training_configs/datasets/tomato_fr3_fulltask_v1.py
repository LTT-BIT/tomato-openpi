"""Tomato full-task dataset v1: actual data must be prepared separately."""

import os

from training_configs.datasets.common import Dataset
from training_configs.datasets.common import runtime_root

PROMPT = (
    "Pick up a tomato truss with the right hand and hold it steady between both hands, "
    "then pick a single cherry tomato from the truss with the left hand "
    "and place the picked tomato into the plate on the left."
)


def build_dataset() -> Dataset:
    runtime = runtime_root()
    root = os.environ.get("PI05_FULLTASK_DATASET_ROOT", str(runtime / "data/tomato_fulltask_v1"))
    return Dataset(
        version="tomato_fulltask_v1_30hz",
        repo_id="fr3_wuji/tomato_fulltask_v1",
        root=root,
        assets_dir=os.environ.get("PI05_FULLTASK_ASSETS_DIR", str(runtime / "assets/fulltask_c0")),
        asset_id="tomato_fulltask_v1_30hz",
        sources=(root,),
        prompt_example=PROMPT,
    )
