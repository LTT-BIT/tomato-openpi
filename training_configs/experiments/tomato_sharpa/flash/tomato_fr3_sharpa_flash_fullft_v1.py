"""FLASH FullFT on seven normal/no-block-one sources; fresh base, absolute 58D."""

from dataclasses import fields
from dataclasses import replace

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.training.config import LeRobotDatasetSpec
from openpi.training.flash_native import bind_stats
from training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL
from training_configs.experiments.tomato_sharpa.baseline.dataset import build_dataset
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe

ROOT = "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one"
MEMBERS = (
    "A/1022_121_430_normal_lerobot",
    "A/1022_54_120_normal_lerobot",
    "A/1026_444_513_normal_lerobot",
    "B/1022_130_423_normal_lerobot",
    "B/1022_56_106_normal_lerobot",
    "B/1026_446_510_normal_lerobot",
    "10.16before",
)


def build_config():
    dataset = replace(
        build_dataset(),
        root=ROOT + "/" + MEMBERS[0],
        version="normal480_flash_absolute_k6_s1_all_sources_v2",
        repo_id="fr3_sharpa/normal480_flash",
        asset_id="normal480_flash_absolute_k6_s1_all_sources_v2",
        sources=tuple(ROOT + "/" + part for part in MEMBERS),
    )
    config = build_recipe(
        name="pi05_sharpa_flash_absolute_k6_s1_all480_fullft_v2",
        exp_name="sharpa_flash_absolute_k6_s1_all480_fullft_s42",
        embodiment=FR3_SHARPA_MULTIMODAL,
        dataset=dataset,
    )
    model = Pi0FlashConfig(**{f.name: getattr(config.model, f.name) for f in fields(Pi0Config)}, freeze_vlm=False)
    data = replace(
        config.data,
        video_backend="pyav",
        lerobot_datasets=tuple(
            LeRobotDatasetSpec(root=ROOT + "/" + part, repo_id=f"fr3_sharpa/normal480_flash_part{i:03d}")
            for i, part in enumerate(MEMBERS)
        ),
    )
    return bind_stats(
        replace(
            config,
            model=model,
            data=data,
            freeze_filter=model.get_freeze_filter(),
            batch_size=8,
            fsdp_devices=8,
            save_interval=1000,
            keep_period=None,
            weight_loader=replace(
                config.weight_loader, params_path="/mnt/data/train_platform/weights/pi05/pi05_base/params"
            ),
            policy_metadata={
                **config.policy_metadata,
                "algorithm": "flash",
                "data_policy": "all_sources",
                "dataset_roots": [ROOT + "/" + part for part in MEMBERS],
                "K": 6,
                "stride": 1,
                "training_action_semantics": "absolute",
                "vlm_frozen": False,
            },
        )
    )
