"""Dataset identity for a fixed, sealed automatic-conversion batch."""

import os

from training_configs import common

PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato "
    "with the left hand and place it in the left basket."
)


def build_dataset():
    repository = common.repository_root()
    root = os.environ.get("PI05_SHARPA_MULTIMODAL_DATASET_ROOT", str(repository / "data/tomato_sharpa_multimodal_rgb_v1"))
    return common.Dataset(
        version="tomato_sharpa_multimodal_rgb_absolute_v1_30hz",
        repo_id="fr3_sharpa/tomato_multimodal_rgb_v1",
        root=root,
        assets_dir=os.environ.get("PI05_SHARPA_MULTIMODAL_ASSETS_DIR", str(repository / "assets")),
        asset_id="tomato_sharpa_multimodal_rgb_absolute_v1_30hz",
        sources=(root,),
        prompt_example=PROMPT,
    )
