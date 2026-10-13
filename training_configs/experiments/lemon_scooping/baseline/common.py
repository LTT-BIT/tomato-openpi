"""Shared data identity and recorded split for the wrist-camera Lemon collection."""

from dataclasses import replace
import os
from pathlib import Path

from openpi.training.config import TaskDataConfig
from openpi.training.task_view import TaskView
from training_configs import common
from training_configs.common import Dataset

TASK = "Scoop the lemon slice into the cup while the left hand supports it."
PHASE_PROMPTS = (
    TASK + " Keep the right arm still. Current phase: state 0 (initial approach and establish left-hand support).",
    TASK + " Keep the left arm still. Current phase: state 1 (align the shovel to start scooping).",
    TASK + " Keep the left arm still. Current phase: state 2 (scoop until the lemon slice is supported by the shovel).",
    TASK
    + " Keep the left arm still. Current phase: state 3 (lift and move the lemon slice above the cup, ready to pour).",
    TASK + " Keep the left arm still. Current phase: state 4 (pour the lemon slice into the cup).",
)
RECORDS = Path(__file__).resolve().parent / "records"


def build_dataset():
    root = common.repository_root()
    dataset = os.environ.get("PI05_LEMON_DATASET_ROOT", str(root / "data/lemon_wrist_reset_v1"))
    return Dataset(
        version="lemon_wrist_reset_v1_10hz",
        repo_id="ur_o30/lemon_move_wrist_reset_20261008",
        root=dataset,
        assets_dir=os.environ.get("PI05_LEMON_ASSETS_DIR", str(root / "assets")),
        asset_id="lemon_wrist_reset_v1_10hz",
        sources=("wrist_reset_20261008:224 episodes; 209 training episodes; 15 excluded", dataset),
        prompt_example=TASK,
    )


def with_recorded_split(config):
    view = TaskView(split_path=str(RECORDS / "splits.json"), prompt_overrides=((0, TASK),))
    return replace(
        config,
        data=replace(config.data, base_config=TaskDataConfig(prompt_from_task=True, task_view=view)),
        policy_metadata={
            **config.policy_metadata,
            "source_record": "lemon_wrist_reset_full_ft_30k/task.json + env.sh",
            "prompt_source": "TaskView.prompt_overrides (single task)",
            "source_training_completed": "unverified",
        },
    )
