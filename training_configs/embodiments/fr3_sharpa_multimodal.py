"""Versioned 116D converter storage projected to canonical 58D RGB training."""

from dataclasses import dataclass

from openpi.policies.fr3_sharpa_multimodal_policy import SOURCE_ACTION_NAMES
from openpi.policies.fr3_sharpa_multimodal_policy import SOURCE_STATE_NAMES
from openpi.training import config as cfg
from training_configs.embodiments.fr3_sharpa import Fr3Sharpa


@dataclass(frozen=True)
class Fr3SharpaMultimodal(Fr3Sharpa):
    version: str = "fr3_sharpa_multimodal_rgb_absolute_v1"
    raw_state_dims: tuple[int, ...] = (116,)

    def __post_init__(self):
        if self.action_dim != 58 or self.raw_state_dims != (116,):
            raise ValueError("Converter Sharpa adapter requires 116D raw state and 58D actions")

    @property
    def protocol_metadata(self):
        return {
            **super().protocol_metadata,
            "source_state_joint_names": list(SOURCE_STATE_NAMES),
            "source_action_joint_names": list(SOURCE_ACTION_NAMES),
            "training_modalities": ["rgb", "joint_position", "task_text"],
            "stored_only_modalities": ["depth", "tactile", "joint_velocity", "audit"],
        }

    def data_config(self, dataset):
        return cfg.LeRobotFr3SharpaMultimodalDataConfig(
            repo_id=dataset.repo_id,
            assets=cfg.AssetsConfig(assets_dir=dataset.assets_dir, asset_id=dataset.asset_id),
            base_config=cfg.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_sequence_keys=("action",),
            action_hz=self.action_hz,
            video_backend="pyav",
            lerobot_datasets=(cfg.LeRobotDatasetSpec(repo_id=dataset.repo_id, root=dataset.root, weight=1.0),),
        )


FR3_SHARPA_MULTIMODAL = Fr3SharpaMultimodal()
