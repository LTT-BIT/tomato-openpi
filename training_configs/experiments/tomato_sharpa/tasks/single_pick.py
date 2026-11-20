"""Sharpa tomato task; select a dataset release without changing this entry point."""

from training_configs.experiments.tomato_sharpa.tasks.common import build_config as build_task


def build_config():
    return build_task("single_pick")
