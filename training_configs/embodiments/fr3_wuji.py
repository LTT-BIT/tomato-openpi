"""FR3/Wuji v1: left arm/hand, right arm/hand; absolute joint targets."""

from dataclasses import dataclass

from openpi.training import config as cfg


@dataclass(frozen=True)
class Fr3Wuji:
    version: str = "fr3_wuji_absolute_v1"
    action_dim: int = 54
    raw_state_dims: tuple[int, ...] = (54, 108)
    action_hz: float = 30.0
    camera_keys: tuple[str, ...] = (
        "observation.images.cam0",
        "observation.images.cam1",
        "observation.images.cam2",
    )

    def __post_init__(self):
        if self.action_dim != 54 or self.raw_state_dims != (54, 108):
            raise ValueError("FR3/Wuji v1 adapter has a fixed 54/108-dimensional contract; define a new adapter")

    @property
    def protocol_metadata(self) -> dict:
        return {"training_action_semantics": "absolute_joint_target"}

    def data_config(self, dataset):
        return cfg.LeRobotFr3WujiDataConfig(
            repo_id=dataset.repo_id,
            assets=cfg.AssetsConfig(assets_dir=dataset.assets_dir, asset_id=dataset.asset_id),
            base_config=cfg.DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            action_sequence_keys=("action",),
            action_hz=self.action_hz,
            video_backend="pyav",
            lerobot_datasets=(cfg.LeRobotDatasetSpec(repo_id=dataset.repo_id, root=dataset.root, weight=1.0),),
        )


FR3_WUJI = Fr3Wuji()
