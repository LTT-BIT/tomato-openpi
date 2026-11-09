"""Compose the existing Sharpa recipe with one reviewed A/B bundle."""

from dataclasses import replace
import json
import os
from pathlib import Path

from openpi.subtasks.prepare import digest
from openpi.subtasks.tomato import GOAL
from openpi.subtasks.tomato import PHASES
from openpi.subtasks.tomato import PROMPTS
from openpi.training.config import TaskDataConfig
from openpi.training.task_view import TaskView
from training_configs import common
from training_configs.embodiments.fr3_sharpa_multimodal import FR3_SHARPA_MULTIMODAL
from training_configs.recipes.pi05_fullft_v1 import build_config as build_recipe


def build(mode, *, selection_path=None):
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
    version = "tomato_truss"
    if selection_path is not None:
        selection = json.loads(Path(selection_path).read_text())
        expected = selection["episodes"]
        mapping = manifest.get("mapping", [])
        if manifest.get("selection_sha256") != digest(selection_path) or len(mapping) != len(expected):
            raise ValueError("Bundle does not match the frozen normal93 selection")
        for index, (actual, source) in enumerate(zip(mapping, expected, strict=True)):
            if (
                actual["episode_index"] != index
                or actual["source_episode_index"] != source["episode_index"]
                or actual["recording_group"] != source["source_recording_id"]
                or actual["annotation_sha256"] != source["annotation_sha256"]
            ):
                raise ValueError("Bundle mapping differs from frozen normal93 selection")
        splits = json.loads((bundle / "splits.json").read_text())["splits"]
        ids = [i for split in splits.values() for i in split]
        if (
            set(splits) != {"train", "val", "test"}
            or any(not v for v in splits.values())
            or (len(ids) != len(expected) or set(ids) != set(range(len(expected))))
        ):
            raise ValueError("A/B splits must cover every selected episode exactly once")
        group_sets = [{mapping[i]["recording_group"] for i in split} for split in splits.values()]
        if any(a & b for i, a in enumerate(group_sets) for b in group_sets[i + 1 :]):
            raise ValueError("Original recordings must not cross A/B splits")
        for variant in ("a_goal", "b_coarse"):
            info = json.loads((bundle / variant / "meta/info.json").read_text())
            tasks = [
                json.loads(line)["task"] for line in (bundle / variant / "meta/tasks.jsonl").read_text().splitlines()
            ]
            if (
                info["total_episodes"] != len(expected)
                or info["total_frames"] != sum(row["length"] for row in expected)
                or tasks != ([GOAL] if variant == "a_goal" else list(dict.fromkeys(PROMPTS)))
            ):
                raise ValueError("Prepared dataset dimensions or prompts differ from frozen A/B contract")
        version = selection["name"]
    dataset = common.Dataset(
        version=f"{version}_coarse_{mode}_v1",
        repo_id=f"fr3_sharpa/{version}_{mode}_v1",
        root=str(bundle / mode),
        assets_dir=str(common.repository_root() / "assets"),
        asset_id=f"{version}_{mode}_v1_absolute_30hz",
        sources=(str(bundle / "bundle.json"),),
        prompt_example=GOAL if mode == "a_goal" else PROMPTS[1],
    )
    config = build_recipe(
        name=(f"pi05_sharpa_{version}_{mode}_fullft_v1" if selection_path else f"pi05_sharpa_truss_{mode}_fullft_v1"),
        exp_name=f"{version}_{mode}_v1_s42",
        embodiment=FR3_SHARPA_MULTIMODAL,
        dataset=dataset,
    )
    view = TaskView(split_path=str(bundle / "splits.json"), split="train")
    return replace(
        config,
        fsdp_devices=8 if selection_path is not None else config.fsdp_devices,
        data=replace(config.data, base_config=TaskDataConfig(prompt_from_task=True, task_view=view)),
        policy_metadata={
            **config.policy_metadata,
            "subtask_variant": mode,
            "reviewed_bundle": str(bundle / "bundle.json"),
            "max_token_len": 256,
            "whole_episodes": True,
            "decision_agent_trained": False,
            "agent_enabled": False,
            "selection_sha256": manifest.get("selection_sha256"),
        },
    )
