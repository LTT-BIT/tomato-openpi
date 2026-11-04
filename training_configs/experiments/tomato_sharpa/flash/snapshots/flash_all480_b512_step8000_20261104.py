# Exported UI configuration with an embedded source snapshot.
from dataclasses import replace

from scripts.training_ui_config import apply_overrides
from scripts.training_ui_config import config_from_source

_SOURCE = '"""FLASH FullFT on seven normal/no-block-one sources; fresh base, absolute 58D."""\n\nfrom dataclasses import fields\nfrom dataclasses import replace\n\nfrom openpi.models.pi0_config import Pi0Config\nfrom openpi.models.pi0_flash import Pi0FlashConfig\nfrom openpi.training.config import LeRobotDatasetSpec\nfrom openpi.training.flash_native import bind_stats\nfrom training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL\nfrom training_configs.experiments.tomato_sharpa.baseline.dataset import build_dataset\nfrom training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe\n\nROOT = "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one"\nMEMBERS = (\n    "A/1022_121_430_normal_lerobot",\n    "A/1022_54_120_normal_lerobot",\n    "A/1026_444_513_normal_lerobot",\n    "B/1022_130_423_normal_lerobot",\n    "B/1022_56_106_normal_lerobot",\n    "B/1026_446_510_normal_lerobot",\n    "10.16before",\n)\n\n\ndef build_config():\n    dataset = replace(\n        build_dataset(),\n        root=ROOT + "/" + MEMBERS[0],\n        version="normal480_flash_absolute_k6_s1_all_sources_v2",\n        repo_id="fr3_sharpa/normal480_flash",\n        asset_id="normal480_flash_absolute_k6_s1_all_sources_v2",\n        sources=tuple(ROOT + "/" + part for part in MEMBERS),\n    )\n    config = build_recipe(\n        name="pi05_sharpa_flash_absolute_k6_s1_all480_fullft_v2",\n        exp_name="sharpa_flash_absolute_k6_s1_all480_fullft_s42",\n        embodiment=FR3_SHARPA_MULTIMODAL,\n        dataset=dataset,\n    )\n    model = Pi0FlashConfig(**{f.name: getattr(config.model, f.name) for f in fields(Pi0Config)}, freeze_vlm=False)\n    data = replace(\n        config.data,\n        video_backend="pyav",\n        lerobot_datasets=tuple(\n            LeRobotDatasetSpec(root=ROOT + "/" + part, repo_id=f"fr3_sharpa/normal480_flash_part{i:03d}")\n            for i, part in enumerate(MEMBERS)\n        ),\n    )\n    return bind_stats(\n        replace(\n            config,\n            model=model,\n            data=data,\n            freeze_filter=model.get_freeze_filter(),\n            batch_size=512,\n            num_train_steps=8000,\n            fsdp_devices=8,\n            save_interval=2000,\n            keep_period=2000,\n            weight_loader=replace(\n                config.weight_loader, params_path="/mnt/data/train_platform/weights/pi05/pi05_base/params"\n            ),\n            policy_metadata={\n                **config.policy_metadata,\n                "algorithm": "flash",\n                "data_policy": "all_sources",\n                "dataset_roots": [ROOT + "/" + part for part in MEMBERS],\n                "K": 6,\n                "stride": 1,\n                "training_action_semantics": "absolute",\n                "vlm_frozen": False,\n            },\n        )\n    )\n'
_SOURCE_MODULE = "training_configs.experiments.tomato_sharpa.flash.tomato_fr3_sharpa_flash_fullft_v1"
_SOURCE_FILE = "/mnt/data/users/mengxiang/workspace/code/tomato-openpi-sharpa/training_configs/experiments/tomato_sharpa/flash/tomato_fr3_sharpa_flash_fullft_v1.py"
_UI_VALUES = {
    "batch_size": 512,
    "num_train_steps": 8000,
    "num_workers": 8,
    "fsdp_devices": 8,
    "log_interval": 10,
    "save_interval": 2000,
    "keep_period": 2000,
    "seed": 42,
    "model_seed": 42,
    "data_seed": 42001,
    "augmentation_seed": 42002,
    "warmup_steps": 1000,
    "peak_lr": 2.5e-05,
    "decay_steps": 30000,
    "decay_lr": 2.5e-06,
    "weight_decay": 1e-10,
    "clip_gradient_norm": 1.0,
    "exp_name": "mengxiang_sharpa_flash_absolute_k6_s1_all480_fullft_s42_1104_144748",
    "dataset_root": "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/A/1022_121_430_normal_lerobot",
    "repo_id": "fr3_sharpa/normal480_flash",
    "dataset_version": "normal480_flash_absolute_k6_s1_all_sources_v2",
    "sources": [
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/A/1022_121_430_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/A/1022_54_120_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/A/1026_444_513_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/B/1022_130_423_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/B/1022_56_106_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/B/1026_446_510_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/10.16before",
    ],
    "assets_dir": "/mnt/data/users/mengxiang/workspace/code/tomato-openpi-sharpa/assets",
    "asset_id": "normal480_flash_absolute_k6_s1_all_sources_v2",
    "weight_path": "/mnt/data/train_platform/weights/pi05/pi05_base/params",
    "project_name": "openpi",
    "action_horizon": 50,
    "rtc_max_delay": 0,
    "action_semantics": "absolute",
    "sampling_mode": "epoch_shuffle",
    "sampling_groups": [],
    "dataset_roots": [
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/A/1022_121_430_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/A/1022_54_120_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/A/1026_444_513_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/B/1022_130_423_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/B/1022_56_106_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/B/1026_446_510_normal_lerobot",
        "/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one/10.16before",
    ],
}
_UI_METADATA = {
    "ui_owner": "mengxiang",
    "ui_baseline": "training_configs.experiments.tomato_sharpa.flash.tomato_fr3_sharpa_flash_fullft_v1",
    "ui_baseline_sha256": "dc40a34d9444071f83081350907be22ab6ab655688898c52075d433c495e704d",
}


def build_config():
    original = config_from_source(_SOURCE, _SOURCE_MODULE, _SOURCE_FILE)
    config = apply_overrides(original, _UI_VALUES)
    return replace(config, policy_metadata={**config.policy_metadata, **_UI_METADATA})
