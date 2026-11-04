"""Opt-in Mamba T3; original Paper statistics, not a completed training run."""

from openpi.training.mamba_experiment import build_experiment


def build_config():
    return build_experiment("t3")
