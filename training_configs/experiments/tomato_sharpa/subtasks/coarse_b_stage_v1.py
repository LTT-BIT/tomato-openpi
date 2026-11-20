"""B: coarse phase text on every frame; episodes and action windows remain whole."""

from training_configs.experiments.tomato_sharpa.subtasks.common import build


def build_config():
    return build("b_coarse")
