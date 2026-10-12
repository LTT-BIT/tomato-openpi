"""Example of personal overrides without modifying a published recipe."""

from dataclasses import replace

from training_configs.experiments.baselines.tomato_fr3_lora_v2 import build_config as build_baseline


def build_config():
    baseline = build_baseline()
    return replace(
        baseline,
        exp_name="template_tomato_lora_lr1e5_s42_run01",
        lr_schedule=replace(baseline.lr_schedule, peak_lr=1e-5),
        policy_metadata={**baseline.policy_metadata, "owner": "template", "purpose": "learning_rate_comparison"},
    )
