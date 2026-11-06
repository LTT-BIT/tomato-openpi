"""Compose the existing Sharpa recipe with one reviewed A/B bundle."""

from dataclasses import replace
import json
import os
from pathlib import Path

from openpi.subtasks.tomato import GOAL
from openpi.subtasks.tomato import PHASES
from openpi.subtasks.tomato import PROMPTS
from openpi.training.config import TaskDataConfig
from openpi.training.task_view import TaskView
from training_configs import common
from training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe


def build(mode):
    if mode not in ("a_goal", "b_coarse"):
        raise ValueError("Unknown experiment variant")
    bundle = Path(os.environ.get("PI05_TOMATO_SUBTASK_BUNDLE", common.repository_root() / "data/tomato_truss_v1"))
    if not (bundle / "bundle.json").is_file():
        raise ValueError("Finish coarse review and prepare a unified A/B bundle before loading this experiment")
    manifest = json.loads((bundle / "bundle.json").read_text())
    if (
        manifest.get("review_complete") is not True
        or manifest.get("phases") != list(PHASES)
        or manifest.get("prompts") != list(PROMPTS)
        or manifest.get("goal") != GOAL
        or not manifest.get("token_budget")
        or any(r["max_input_tokens"] > 256 for r in manifest["token_budget"])
    ):
        raise ValueError("Bundle is not reviewed or has an incompatible prompt/token contract")
    dataset = common.Dataset(
        version=f"tomato_truss_coarse_{mode}_v1",
        repo_id=f"fr3_sharpa/tomato_truss_{mode}_v1",
        root=str(bundle / mode),
        assets_dir=str(common.repository_root() / "assets"),
        asset_id=f"tomato_truss_{mode}_v1_absolute_30hz",
        sources=(str(bundle / "bundle.json"),),
        prompt_example=GOAL if mode == "a_goal" else PROMPTS[1],
    )
    config = build_recipe(
        name=f"pi05_sharpa_truss_{mode}_fullft_v1",
        exp_name=f"tomato_truss_{mode}_v1_s42",
        embodiment=FR3_SHARPA_MULTIMODAL,
        dataset=dataset,
    )
    view = TaskView(split_path=str(bundle / "splits.json"), split="train")
    return replace(
        config,
        data=replace(config.data, base_config=TaskDataConfig(prompt_from_task=True, task_view=view)),
        policy_metadata={
            **config.policy_metadata,
            "subtask_variant": mode,
            "reviewed_bundle": str(bundle / "bundle.json"),
            "max_token_len": 256,
            "whole_episodes": True,
            "decision_agent_trained": False,
        },
    )
