"""Five-phase Sharpa training with current RGB frames and selectable inputs."""

from dataclasses import replace

from openpi.training.current_rgb import CurrentRgbView
from training_configs.experiments.tomato_sharpa.subtasks.in_hand119_b_crop import build_config as build_baseline


def build_config():
    baseline = build_baseline()
    return replace(
        baseline,
        name="pi05_sharpa_in_hand119_current_rgb_delta_v3_r1",
        exp_name="tomato_in_hand119_current_rgb_delta_v3_r1_s42_b256_s30000_c5000",
        data=replace(baseline.data, base_config=CurrentRgbView(prompt_from_task=True)),
        policy_metadata={**baseline.policy_metadata, "input_acceptance_report": None},
    )
