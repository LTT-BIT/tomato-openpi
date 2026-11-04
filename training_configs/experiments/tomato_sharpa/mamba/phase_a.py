"""Phase A FullFT B1 (not trained); explicit Absolute assets, no Paper/RTC.

Environment variables select local inputs; no server or download is started by import.
"""

from dataclasses import replace
import os

from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.training import optimizer
from openpi.training.mamba_training import MambaSharpaData
from openpi.training.mamba_training import MambaTrainConfig
from openpi.training.mamba_weight_loader import MambaVlmInitLoader
from training_configs import common
from training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe

PROMPT = "Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket."
DATASET = common.Dataset(
    version="sharpa_mamba_phase_a_absolute_v1",
    repo_id="fr3_sharpa/mamba_phase_a_absolute_v1",
    root=os.environ.get("PI05_MAMBA_DATASET_ROOT", str(common.repository_root() / "data/mamba_phase_a")),
    assets_dir=os.environ.get("PI05_MAMBA_ASSETS_DIR", str(common.repository_root() / "assets")),
    asset_id="sharpa_mamba_phase_a_absolute_v1",
    sources=(os.environ.get("PI05_MAMBA_DATASET_ROOT", str(common.repository_root() / "data/mamba_phase_a")),),
    prompt_example=PROMPT,
)
base = build_recipe(
    name="pi05_mamba3_phase_a_fullft_b1",
    exp_name="mamba3_phase_a_fullft_b1_s42",
    embodiment=FR3_SHARPA_MULTIMODAL,
    dataset=DATASET,
)
model = MambaPi0Config()
data = MambaSharpaData(**vars(base.data))
metadata = {
    **base.policy_metadata,
    "architecture": model.architecture,
    "stage": model.stage,
    "training_action_semantics": "absolute",
    "normalization": model.normalization,
    "training_status": "prepared_not_trained",
    "real_robot_status": "not_validated",
}
CONFIG = MambaTrainConfig(
    **{
        **vars(base),
        "model": model,
        "data": data,
        "policy_metadata": metadata,
        "batch_size": 1,
        "num_workers": 0,
        "num_train_steps": 10,
        "log_interval": 1,
        "save_interval": 5,
        "keep_period": None,
        "weight_loader": MambaVlmInitLoader(os.environ.get("PI05_MAMBA_INIT_CHECKPOINT", common.SERVER_BASE_PARAMS)),
        "lr_schedule": optimizer.CosineDecaySchedule(warmup_steps=1, decay_steps=100, peak_lr=2.5e-5, decay_lr=2.5e-6),
    }
)
# Same VLM/data/dtypes/optimizer and short schedule, native Transformer reference.
TRANSFORMER_CONTROL = replace(
    base,
    name="pi05_mamba3_phase_a_transformer_control_b1",
    exp_name="mamba3_transformer_control_b1_s42",
    batch_size=1,
    num_workers=0,
    num_train_steps=10,
    log_interval=1,
    lr_schedule=CONFIG.lr_schedule,
    data=data,
)


def build_config():
    return CONFIG
