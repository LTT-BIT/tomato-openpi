"""Lemon protocol: left UR(6), left O30(20), right UR(6), right O30(20)."""

from dataclasses import dataclass

from openpi.training import config as cfg


@dataclass(frozen=True)
class DualURO30:
    model_action_dim: int = 64
    version: str = "dual_ur_o30_v1"
    action_dim: int = 12
    raw_state_dims: tuple[int, ...] = (52,)
    action_hz: float = 10.0
    camera_keys: tuple[str, ...] = tuple(f"observation.images.cam{i}" for i in range(3))

    def __post_init__(self):
        if self.model_action_dim not in (32, 64) or self.action_dim != 12 or self.raw_state_dims != (52,):
            raise ValueError("Dual UR/O30 requires state 52, action 12 and model action head 32 or 64")

    @property
    def protocol_metadata(self):
        return {
            "model_action_dim": self.model_action_dim,
            "normalization_dims": {"state": 52, "actions": 12},
            "state_order": "left_ur(6),left_o30(20),right_ur(6),right_o30(20)",
            "state_units": "rad,raw_0_255,rad,raw_0_255",
            "action_order": "left_ur(6),right_ur(6)",
            "training_action_semantics": "absolute_joint_target_rad",
            "camera_roles": {"base": "cam0", "left_wrist": "cam1", "right_wrist": "cam2"},
        }

    def data_config(self, dataset):
        return cfg.LeRobotLemonDataConfig(
            repo_id=dataset.repo_id,
            assets=cfg.AssetsConfig(assets_dir=dataset.assets_dir, asset_id=dataset.asset_id),
            base_config=cfg.DataConfig(prompt_from_task=True),
            action_hz=self.action_hz,
            video_backend="pyav",
            lerobot_datasets=(cfg.LeRobotDatasetSpec(repo_id=dataset.repo_id, root=dataset.root),),
        )
