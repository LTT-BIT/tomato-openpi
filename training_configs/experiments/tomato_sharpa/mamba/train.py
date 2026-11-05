"""Server FullFT candidate: 311 RGB episodes, global B512 / 8 GPUs.

Prepared configuration, not a completed capacity/throughput or robot validation.
Plain Absolute by default; the GUI independently enables Paper features and RTC.
"""

from dataclasses import replace
import os

from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.training.config import LeRobotDatasetSpec
from openpi.training.delta_paper_experiment import build_no_block_one_dataset
from openpi.training.mamba_training import MambaSharpaData
from openpi.training.mamba_training import MambaTrainConfig
from openpi.training.mamba_weight_loader import MambaVlmInitLoader
from openpi.training.video_cache import VideoCacheConfig
from training_configs import common
from training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe

PROMPT = "Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket."
_sources = build_no_block_one_dataset(
    owner="mamba",
    trial="plain",
    root=os.environ.get(
        "PI05_MAMBA_DATASET_ROOT", "/mnt/workspace/train_platform/datasets/tomato_sharpa/normal/no-block-one"
    ),
)
_identity = "tomato_sharpa_no_block_one_mamba_absolute_v1_30hz"
DATASET = replace(
    _sources,
    version=_identity,
    asset_id=_identity,
    assets_dir=os.environ.get("PI05_MAMBA_ASSETS_DIR", str(common.repository_root() / "assets")),
)
base = build_recipe(
    name="pi05_mamba3_sharpa_no_block_one_fullft_v1",
    exp_name="mamba3_sharpa_no_block_one_absolute_s42_run01",
    embodiment=FR3_SHARPA_MULTIMODAL,
    dataset=DATASET,
)
data = MambaSharpaData(
    **{
        **vars(base.data),
        "lerobot_datasets": tuple(LeRobotDatasetSpec(repo_id=m.repo_id, root=m.root) for m in DATASET.members),
        "video_backend": "pyav",
        "base_config": replace(
            base.data.base_config, video_cache=VideoCacheConfig(max_readers=768, max_bytes=2 * 1024**3)
        ),
    }
)
model = MambaPi0Config()
CONFIG = MambaTrainConfig(
    **{
        **vars(base),
        "model": model,
        "data": data,
        "batch_size": 512,
        "fsdp_devices": 8,
        "num_workers": 8,
        "num_train_steps": 8000,
        "log_interval": 10,
        "save_interval": 2000,
        "keep_period": None,
        "weight_loader": MambaVlmInitLoader(os.environ.get("PI05_MAMBA_INIT_CHECKPOINT", common.SERVER_BASE_PARAMS)),
        "policy_metadata": {
            **base.policy_metadata,
            "dataset_root": DATASET.members[0].root,
            "dataset_roots": [member.root for member in DATASET.members],
            "dataset_collection_root": DATASET.root,
            "architecture": model.architecture,
            "stage": model.stage,
            "training_action_semantics": "absolute",
            "normalization": model.normalization,
            "expected_source_episodes": 311,
            "expected_source_frames": 294206,
            "capacity_status": "batch512_candidate_not_full_model_validated",
            "training_status": "prepared_not_trained",
            "real_robot_status": "not_validated",
        },
    }
)
# Same RGB sources, ordinary z score, batch, workers and optimizer for comparison.
TRANSFORMER_CONTROL = replace(
    base,
    name="pi05_sharpa_no_block_one_transformer_control_v1",
    exp_name="sharpa_no_block_one_transformer_control_s42_run01",
    data=data,
    batch_size=CONFIG.batch_size,
    fsdp_devices=CONFIG.fsdp_devices,
    num_workers=CONFIG.num_workers,
    num_train_steps=CONFIG.num_train_steps,
    log_interval=CONFIG.log_interval,
    save_interval=CONFIG.save_interval,
    keep_period=CONFIG.keep_period,
)


def build_config():
    return CONFIG
