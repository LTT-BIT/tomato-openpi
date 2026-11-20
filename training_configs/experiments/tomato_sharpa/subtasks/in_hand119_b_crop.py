"""Fresh five-phase in-hand119 B experiment; independent from seven-phase100."""

from dataclasses import fields
from dataclasses import replace
import json
import os
from pathlib import Path

from openpi.subtasks import tomato_in_hand as contract
from openpi.subtasks.rgb100 import IMAGE_KEYS
from openpi.subtasks.rgb100 import digest
from openpi.training.config import ScheduledTrainConfig
from training_configs import common
from training_configs.embodiments.fr3_sharpa_crop import FR3_SHARPA_CROP
from training_configs.recipes.pi05_fullft_v1 import build_config as recipe


def verify_bundle(root, dataset_root):
    manifest = json.loads((root / "bundle.json").read_text())
    info = json.loads((dataset_root / "meta/info.json").read_text())
    contract.validate_info(info)
    tasks = [json.loads(line)["task"] for line in (dataset_root / "meta/tasks.jsonl").read_text().splitlines() if line]
    if (
        manifest.get("version") != contract.VERSION
        or manifest.get("profile") != contract.PROFILE
        or manifest.get("input_preparation_complete") is not True
        or manifest.get("physical_columns_equal") is not True
        or manifest.get("raw_sources_unchanged") is not True
        or manifest.get("episodes") != contract.EPISODES
        or manifest.get("frames") != contract.FRAMES
        or manifest.get("data_policy") != "all_sources"
        or tasks != list(contract.TASKS)
        or info["total_episodes"] != contract.EPISODES
        or info["total_frames"] != contract.FRAMES
        or info["features"].get("observation.interaction_available", {}).get("dtype") != "bool"
    ):
        raise ValueError("Frozen full119 v3 four-view B bundle required")
    for name in [
        "selection.json",
        "b_coarse/meta/info.json",
        "b_coarse/meta/episodes.jsonl",
        "b_coarse/meta/tasks.jsonl",
    ]:
        if digest(root / name) != manifest["files"][name]:
            raise ValueError("Prepared metadata changed")
    return digest(root / "selection.json"), digest(root / "bundle.json")


def build_config():
    root = Path(os.environ.get("PI05_IN_HAND119_BUNDLE", contract.DEFAULT_BUNDLE)).expanduser().resolve()
    dataset_root = root / "b_coarse"
    # The UI can display a configuration before the machine has the prepared data.
    selection_sha, bundle_sha = verify_bundle(root, dataset_root) if root.exists() else (None, None)
    acceptance_report = os.environ.get("PI05_IN_HAND119_ACCEPTANCE_REPORT") or (
        common.repository_root() / "logs/in_hand119_preflight/accepted_20261116_s30000_c5000.json"
    )
    dataset = common.Dataset(
        version=contract.VERSION,
        repo_id=contract.REPO_ID,
        root=str(dataset_root),
        assets_dir=str(common.repository_root() / "assets"),
        asset_id=contract.ASSET_ID,
        sources=(str(root / "bundle.json"), str(root / "selection.json")),
        prompt_example=contract.TASKS[1],
    )
    config = recipe(
        name="pi05_sharpa_in_hand119_b_crop_fullft_delta_v3_r1",
        exp_name="tomato_in_hand119_b_crop_delta_v3_r1_s42_b256_s30000_c5000",
        embodiment=FR3_SHARPA_CROP,
        dataset=dataset,
    )
    config = replace(
        config,
        model=replace(config.model, image_keys=IMAGE_KEYS, max_token_len=256),
        data=replace(config.data, extra_delta_transform=True, interaction_mask_key="observation.interaction_available"),
        batch_size=256,
        num_train_steps=30000,
        fsdp_devices=8,
        save_interval=5000,
        keep_period=5000,
        policy_metadata={
            **config.policy_metadata,
            "subtask_variant": "in_hand_b_coarse",
            "prompt_profile": contract.PROFILE,
            "training_action_semantics": "delta_arm_joint_target_absolute_hand",
            "data_policy": "all_sources",
            "selection_sha256": selection_sha,
            "bundle_sha256": bundle_sha,
            "phases": list(contract.PHASES),
            "prompts": list(contract.TASKS),
            "max_token_len": 256,
            "visual_tokens": 1024,
            "offline_roi_uses_future_annotation": True,
            "independent_semantic_accuracy_measured": False,
            "training_episodes": contract.EPISODES,
            "training_frames": contract.FRAMES,
            "action_prompts": dict(contract.ACTION_PROMPTS),
            "rgb_frame_cache": contract.DEFAULT_CACHE,
            "input_acceptance_report": str(Path(acceptance_report).expanduser().resolve()),
            "memory_to_pi05": False,
            "agent_enabled": False,
            "gpu_memory_preflight_required": True,
        },
    )
    return ScheduledTrainConfig(
        **{f.name: getattr(config, f.name) for f in fields(config)},
        save_before_step=1,
        save_steps_before=(),
        save_interval_after=5000,
    )
