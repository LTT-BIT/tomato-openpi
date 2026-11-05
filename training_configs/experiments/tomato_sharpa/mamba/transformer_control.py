"""Ordinary Transformer control sharing server data, batch and schedule."""

from training_configs.experiments.tomato_sharpa.mamba.train import TRANSFORMER_CONTROL


def build_config():
    return TRANSFORMER_CONTROL
