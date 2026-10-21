"""Sealed 10.16_2 Sharpa converter batch shared by the official RGB baselines."""

import os

from training_configs import common

PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato "
    "with the left hand and place it in the left basket."
)
SOURCE = (
    "sshfs://user@192.168.10.202/data/data-fruits/real/harvest-tomato/data/clean-rosbag/sharpa/10.16_2_data_lerobot"
)


def build_dataset():
    repository = common.repository_root()
    root = os.environ.get(
        "PI05_SHARPA_MULTIMODAL_DATASET_ROOT",
        os.environ.get("PI05_SHARPA_DATASET_ROOT", "/mnt/data/users/tairan/workspace/data/10.16_2_data_lerobot"),
    )
    return common.Dataset(
        version="tomato_sharpa_1016_2_multimodal_rgb_absolute_v1_30hz",
        repo_id="fr3_sharpa/tomato_1016_2_multimodal_rgb_v1",
        root=root,
        assets_dir=os.environ.get(
            "PI05_SHARPA_MULTIMODAL_ASSETS_DIR",
            os.environ.get("PI05_SHARPA_ASSETS_DIR", str(repository / "assets")),
        ),
        asset_id="tomato_sharpa_1016_2_multimodal_rgb_absolute_v1_30hz",
        sources=(SOURCE, root),
        prompt_example=PROMPT,
    )
