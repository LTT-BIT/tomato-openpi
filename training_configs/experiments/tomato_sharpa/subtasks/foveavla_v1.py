"""Own-task RGB FoveaVLA, four cameras, delta arms/absolute hands, no memory."""

from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
import json
import os
from pathlib import Path

from openpi import transforms
from openpi.models.pi0_fovea import FoveaConfig
from openpi.policies.fr3_sharpa_fovea_policy import FoveaInputs
from openpi.subtasks import fovea
from openpi.subtasks import tomato_in_hand
from openpi.training.config import LeRobotFr3SharpaCropDataConfig
from openpi.training.config import ScheduledTrainConfig
from training_configs import common
from training_configs.embodiments.fr3_sharpa_crop import FR3_SHARPA_CROP
from training_configs.recipes.pi05_fullft_v1 import build_config as recipe

DEFAULT_ROOT = "/mnt/data/train_platform/datasets/tomato_sharpa/normal/full-block-more/subtask-foveavla-v1-h15"
ASSET_ID = "tomato_foveavla_v1_h15_delta_30hz"


@dataclass(frozen=True)
class FoveaDataConfig(LeRobotFr3SharpaCropDataConfig):
    labels_root: str = DEFAULT_ROOT

    def create(self, assets_dirs, model_config):
        if (
            model_config.action_horizon != 50
            or model_config.execute_steps != fovea.EXECUTE_STEPS
            or model_config.box_loss_weight != 0.1
            or model_config.rtc_max_delay
        ):
            raise ValueError("Fovea v1 data/checkpoint contract requires horizon50, execute15, lambda0.1 and no RTC")
        config = super().create(assets_dirs, model_config)
        return replace(config, fovea_labels_root=self.labels_root, fovea_execute_steps=model_config.execute_steps)

    def create_vector_transforms(self, model_config):
        repack, vectors = super().create_vector_transforms(model_config)
        mapping = {**repack.inputs[0].structure, "foveation_tokens": "foveation_tokens"}
        return replace(repack, inputs=[transforms.RepackTransform(mapping)]), replace(
            vectors, inputs=[FoveaInputs(model_config.model_type), *vectors.inputs[1:]]
        )


def build_config():
    root = Path(os.environ.get("PI05_FOVEA_ROOT", DEFAULT_ROOT)).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        not manifest.get("complete")
        or manifest.get("version") != fovea.VERSION
        or manifest["execute_steps"] != fovea.EXECUTE_STEPS
    ):
        raise ValueError("Prepare the versioned Fovea label bundle first")
    dataset = common.Dataset(
        version=fovea.VERSION,
        repo_id="fr3_sharpa/tomato_foveavla_v1_h15",
        root=str(root / "dataset"),
        assets_dir=str(root / "assets"),
        asset_id=ASSET_ID,
        sources=(str(root / "manifest.json"),),
        prompt_example=tomato_in_hand.TASKS[1],
    )
    config = recipe(
        name="pi05_sharpa_foveavla_v1_h15",
        exp_name="tomato_foveavla_v1_h15_s42_b256_s30000_c5000",
        embodiment=FR3_SHARPA_CROP,
        dataset=dataset,
    )
    options = {f.name: getattr(config.model, f.name) for f in fields(config.model)}
    model = FoveaConfig(**{**options, "image_keys": fovea.IMAGE_KEYS})
    data_options = {f.name: getattr(config.data, f.name) for f in fields(config.data)}
    data = FoveaDataConfig(
        **{
            **data_options,
            "labels_root": str(root),
            "extra_delta_transform": True,
            "interaction_mask_key": "observation.interaction_available",
        }
    )
    config = replace(
        config,
        model=model,
        data=data,
        batch_size=256,
        num_train_steps=30000,
        fsdp_devices=8,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            **config.policy_metadata,
            "foveation": manifest["contract"],
            "fovea_manifest": str(root / "manifest.json"),
            "memory_enabled": False,
            "agent_enabled": False,
            "training_action_semantics": "delta_arm_joint_target_absolute_hand",
            "box_supervision": "task-adapted weak boxes; existing offline ROI + detected left hand + fixed target priors",
            "training_pairs": manifest["training_pairs"],
            "source_frames": manifest["frames"],
            "rgb_frame_cache": None,
            "requires_source_resolution": True,
            "norm_reference": "same119 full-frame delta vectors; checkpoint owns copied statistics",
            "gpu_memory_preflight_required": True,
        },
    )
    return ScheduledTrainConfig(
        **{f.name: getattr(config, f.name) for f in fields(config)},
        save_before_step=1,
        save_steps_before=(),
        save_interval_after=5000,
    )
