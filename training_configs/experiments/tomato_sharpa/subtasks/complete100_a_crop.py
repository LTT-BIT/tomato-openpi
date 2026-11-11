"""All100 four-view A goal candidate, with the same strict readiness gate as B."""
from training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop import build_config as build


def build_config():
    return build(variant='a_goal')
