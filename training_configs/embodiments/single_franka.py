"""Single Franka joint protocol from the supplied cup task configuration."""

from dataclasses import dataclass

from openpi.policies import single_franka_policy as protocol
from openpi.training import config as cfg


@dataclass(frozen=True)
class SingleFranka:
    version: str = "single_franka_v1"
    action_dim: int = protocol.PHYSICAL_DIM
    model_action_dim: int = protocol.MODEL_DIM
    raw_state_dims: tuple[int, ...] = (protocol.PHYSICAL_DIM,)
    action_hz: float = 30.0
    camera_keys: tuple[str, ...] = ("observation.images.cam_head", "observation.images.wrist_left")

    def __post_init__(self):
        if (self.action_dim, self.model_action_dim, self.raw_state_dims) != (7, 32, (7,)):
            raise ValueError(
                "Single Franka v1 requires physical 7D vectors and a padded 32D model; define a new adapter"
            )

    @property
    def protocol_metadata(self) -> dict:
        return {
            "protocol_origin": "user_supplied_single_franka_config.py",
            "source_config_sha256": "3c7096905a694437c5ce020c76817a1c3fac5924e9c8a0bf179af05d3b0834a1",
            "model_action_dim": self.model_action_dim,
            "normalization_dim": self.action_dim,
            "training_action_semantics": "absolute_joint_target",
            "camera_roles": {"base": "cam_head", "left_wrist": "wrist_left"},
        }

    def data_config(self, dataset):
        return cfg.LeRobotSingleFrankaDataConfig(
            repo_id=dataset.repo_id,
            assets=cfg.AssetsConfig(assets_dir=dataset.assets_dir, asset_id=dataset.asset_id),
            base_config=cfg.DataConfig(prompt_from_task=True),
            action_sequence_keys=("action",),
            action_hz=self.action_hz,
            video_backend="pyav",
            lerobot_datasets=(cfg.LeRobotDatasetSpec(repo_id=dataset.repo_id, root=dataset.root, weight=1.0),),
        )


SINGLE_FRANKA = SingleFranka()
