"""Compose one task profile with one explicit dataset release and one pi05 recipe."""

from dataclasses import fields
from dataclasses import replace
import os

from openpi.subtasks import tomato_in_hand
from openpi.subtasks.rgb100 import IMAGE_KEYS
from openpi.training.config import ScheduledTrainConfig
from openpi.training.current_rgb import CurrentRgbView
from openpi.training.current_rgb import LeRobotCurrentRgbThreeDataConfig
from training_configs import common
from training_configs.embodiments.fr3_sharpa_crop import FR3_SHARPA_CROP
from training_configs.experiments.tomato_sharpa.baseline.dataset import build_dataset
from training_configs.experiments.tomato_sharpa.tasks.dataset import load_release
from training_configs.experiments.tomato_sharpa.tasks.profiles import PROFILES
from training_configs.recipes import pi05_fullft_v1
from training_configs.recipes import pi05_lora_v1


def initial_config(profile):
    """Read the existing accepted release without renaming its data or statistics."""
    if profile == "single_pick":
        from training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_fullft_v1 import build_config

        return build_config()
    if profile == "truss_in_hand":
        from training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_crop import build_config

        return build_config()
    if os.environ.get("PI05_TOMATO100_CROP_BUNDLE"):
        from training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop import build_config

        return build_config()
    # Preview only. A prepared dataset release is required for a new plate experiment.
    dataset = replace(
        build_dataset(),
        root=str(common.repository_root() / "data/sharpa/truss_plate"),
        version="truss_plate_unselected",
        repo_id="fr3_sharpa/truss_plate_unselected",
        asset_id="truss_plate_unselected_delta_30hz",
        prompt_example=PROFILES[profile].prompts[1],
    )
    config = pi05_fullft_v1.build_config(
        name="pi05_sharpa_truss_plate", exp_name="tomato_truss_plate", embodiment=FR3_SHARPA_CROP, dataset=dataset
    )
    return replace(
        config,
        data=replace(config.data, extra_delta_transform=True),
        policy_metadata={**config.policy_metadata, "dataset_release_required": True},
    )


def build_config(profile, *, dataset_manifest=None, method="fullft", expected_sha256=None):
    task = PROFILES[profile]
    if method not in ("fullft", "lora"):
        raise ValueError("Choose fullft or lora")
    manifest = dataset_manifest or os.environ.get(f"PI05_{profile.upper()}_DATASET_MANIFEST")
    release = load_release(manifest, profile, expected_sha256) if manifest else None
    if release:
        dataset = common.Dataset(
            version=release["version"],
            repo_id=release["repo_id"],
            root=release["root"],
            assets_dir=str(common.repository_root() / "assets"),
            asset_id=release["asset_id"],
            sources=tuple(release["sources"]),
            prompt_example=release["prompts"][0],
        )
        base = pi05_fullft_v1.build_config(
            name=f"pi05_sharpa_{profile}", exp_name=f"tomato_{profile}", embodiment=FR3_SHARPA_CROP, dataset=dataset
        )
        base = replace(
            base,
            data=replace(
                base.data,
                extra_delta_transform=release["action_semantics"] == "delta",
                interaction_mask_key=release["interaction_mask_key"] or "observation.interaction_valid",
            ),
        )
        views = release["video_views"]
        metadata = {
            "dataset_manifest": release["manifest"],
            "dataset_manifest_sha256": release["sha256"],
            "training_episodes": release["episodes"],
            "training_frames": release["frames"],
            "prompts": release["prompts"],
            "source_datasets": release["sources"],
            "offline_roi_uses_future_annotation": release.get("offline_roi_uses_future_annotation", False),
        }
    else:
        base = initial_config(profile)
        views = 3 if profile == "single_pick" else 4
        metadata = {
            "dataset_manifest": "",
            "dataset_manifest_sha256": None,
            "prompts": list(task.prompts) if task.prompts else [base.policy_metadata["prompt_example"]],
        }
    options = CurrentRgbView(prompt_from_task=True, video_views=views)
    kwargs = {field.name: getattr(base.data, field.name) for field in fields(base.data)}
    kwargs.update(base_config=options)
    if views == 3:
        kwargs.setdefault("interaction_mask_key", "observation.interaction_valid")
        data = LeRobotCurrentRgbThreeDataConfig(**kwargs)
    else:
        data = replace(base.data, base_config=options)
    recipe = pi05_lora_v1 if method == "lora" else pi05_fullft_v1
    recipe_model = recipe.build_config(
        name="recipe", exp_name="recipe", embodiment=FR3_SHARPA_CROP, dataset=build_dataset()
    )
    model = replace(recipe_model.model, image_keys=IMAGE_KEYS[:views])
    name = f"pi05_sharpa_{profile}_{method}"
    release_suffix = f"_{release['sha256'][:12]}" if release else ""
    is_single = profile == "single_pick"
    return ScheduledTrainConfig(
        **{
            **{
                field.name: getattr(base, field.name)
                for field in fields(base)
                if field.name not in ("save_before_step", "save_steps_before", "save_interval_after")
            },
            "name": name,
            "exp_name": f"tomato_{profile}_{method}{release_suffix}_s42",
            "data": data,
            "model": model,
            "freeze_filter": model.get_freeze_filter(),
            "batch_size": (8 if method == "lora" else 32) if is_single else (8 if method == "lora" else 256),
            "fsdp_devices": 1 if method == "lora" else 8,
            "num_train_steps": 30000 if is_single else 20000,
            "save_interval": 2000 if is_single else 5000,
            "keep_period": 10000 if is_single else 5000,
            "save_before_step": 1,
            "save_steps_before": (),
            "save_interval_after": 2000 if is_single else 5000,
            "policy_metadata": {
                **base.policy_metadata,
                **metadata,
                "task_profile": profile,
                "task_label": task.label,
                "training_method": method,
                "goal": task.goal,
                "phases": list(task.phases),
                "action_prompts": task.action_prompts,
                "available_video_views": list(range(3, views + 1)),
                "camera_keys": list(tomato_in_hand.CAMERAS[:views]),
                "visual_tokens": views * 256,
                "subtask_variant": None,
                "input_acceptance_report": None,
                "dataset_prompts": metadata["prompts"],
                "dataset_offline_roi_uses_future_annotation": metadata.get(
                    "offline_roi_uses_future_annotation",
                    base.policy_metadata.get("offline_roi_uses_future_annotation", False),
                ),
                "dataset_release_required": base.policy_metadata.get("dataset_release_required", False)
                or (not release and profile == "truss_in_hand" and not base.policy_metadata.get("bundle_sha256")),
                "training_action_semantics": "delta_arm_joint_target_absolute_hand"
                if data.extra_delta_transform
                else "absolute_joint_target",
                "recipe_version": recipe_model.policy_metadata["recipe_version"],
                "baseline_version": recipe_model.policy_metadata["baseline_version"],
                "source_baseline": recipe_model.policy_metadata["source_baseline"],
                "memory_to_pi05": False,
                "agent_enabled": False,
            },
        }
    )
