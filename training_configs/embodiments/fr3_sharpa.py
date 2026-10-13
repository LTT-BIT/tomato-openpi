"""FR3/Sharpa v1: 7+22+7+22 measured/commanded radians, arms delta and hands absolute."""

from dataclasses import dataclass

from openpi.policies import fr3_sharpa_protocol as protocol
from openpi.training import config as cfg


@dataclass(frozen=True)
class Fr3Sharpa:
    version: str = "fr3_sharpa_v1"
    action_dim: int = protocol.ACTION_DIM
    raw_state_dims: tuple[int, ...] = (protocol.STATE_DIM,)
    action_hz: float = 30.0
    camera_keys: tuple[str, ...] = (
        "observation.images.cam0",
        "observation.images.cam1",
        "observation.images.cam2",
    )

    def __post_init__(self):
        if self.action_dim != 58 or self.raw_state_dims != (58,):
            raise ValueError("FR3/Sharpa v1 has a fixed measured 58D contract; define a new adapter")

    @property
    def protocol_metadata(self) -> dict:
        return {
            "protocol_origin": "franka3-sharpa-pi05",
            "joint_names": list(protocol.ORDER),
            "joint_unit": protocol.UNIT,
            "state_semantics": "measured_joint_position",
            "source_action_semantics": "commanded_absolute_joint_position",
            "training_action_semantics": "arms_delta_hands_absolute",
            "camera_roles": {"base": "cam0", "left_wrist": "cam2", "right_wrist": "cam1"},
        }

    def data_config(self, dataset):
        return cfg.LeRobotFr3SharpaDataConfig(
            repo_id=dataset.repo_id,
            assets=cfg.AssetsConfig(assets_dir=dataset.assets_dir, asset_id=dataset.asset_id),
            base_config=cfg.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            action_sequence_keys=("action",),
            action_hz=self.action_hz,
            video_backend="pyav",
            lerobot_datasets=(cfg.LeRobotDatasetSpec(repo_id=dataset.repo_id, root=dataset.root, weight=1.0),),
        )


FR3_SHARPA = Fr3Sharpa()
