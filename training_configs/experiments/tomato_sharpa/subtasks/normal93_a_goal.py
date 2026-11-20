"""Frozen 93 normal recordings, whole-task goal, no Agent; full fine-tuning."""

from pathlib import Path

from training_configs.experiments.tomato_sharpa.subtasks.common import build


def build_config():
    return build(
        "a_goal",
        selection_path=Path(__file__).parent / "records/normal93_20261109_selection.json",
        action_semantics="delta",
    )
