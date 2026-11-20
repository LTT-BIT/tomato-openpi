"""A: one overall goal on every frame of each complete episode."""

from training_configs.experiments.tomato_sharpa.subtasks.common import build


def build_config():
    return build("a_goal")
