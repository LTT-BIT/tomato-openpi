"""Compatibility import for the server configuration; tiny runs use smoke tools."""

from training_configs.experiments.tomato_sharpa.mamba.train import CONFIG


def build_config():
    return CONFIG
