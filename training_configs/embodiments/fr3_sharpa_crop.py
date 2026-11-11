"""Opt-in four-current-RGB Sharpa embodiment; no history or depth in pi05."""
from dataclasses import dataclass

from openpi.training import config as cfg
from training_configs.embodiments.fr3_sharpa_multimodal import Fr3SharpaMultimodal


@dataclass(frozen=True)
class Fr3SharpaCrop(Fr3SharpaMultimodal):
    version: str = 'fr3_sharpa_multimodal_rgb_crop_v1'
    camera_keys: tuple[str,...] = (
        'observation.images.cam0','observation.images.cam1','observation.images.cam2',
        'observation.images.interaction',
    )

    def data_config(self,dataset):
        return cfg.LeRobotFr3SharpaCropDataConfig(
            repo_id=dataset.repo_id,
            assets=cfg.AssetsConfig(assets_dir=dataset.assets_dir,asset_id=dataset.asset_id),
            base_config=cfg.DataConfig(prompt_from_task=True),
            action_sequence_keys=('action',),action_hz=30,video_backend='pyav',
            lerobot_datasets=(cfg.LeRobotDatasetSpec(repo_id=dataset.repo_id,root=dataset.root,weight=1.),),
        )


FR3_SHARPA_CROP=Fr3SharpaCrop()
