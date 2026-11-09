"""Same 93 recordings with four ground-truth phase prompts; no Agent."""

from pathlib import Path

from training_configs.experiments.tomato_sharpa.subtasks.common import build


def build_config():
    return build("b_coarse", selection_path=Path(__file__).parent / "records/normal93_20261109_selection.json")
