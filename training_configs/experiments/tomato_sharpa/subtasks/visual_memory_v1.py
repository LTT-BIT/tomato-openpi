"""Independent four-RGB history experiment, retaining current-state action semantics."""

from dataclasses import fields
from dataclasses import replace
import os

from openpi.models.pi0_visual_memory import VisualMemoryConfig
from openpi.subtasks.visual_memory import VisualMemorySpec
from openpi.training.visual_memory import LeRobotVisualMemoryDataConfig
from training_configs.experiments.tomato_sharpa.subtasks.in_hand119_b_crop import build_config as build_baseline


def build_config():
    baseline = build_baseline()
    memory = VisualMemorySpec(
        frames=int(os.environ.get("PI05_VISUAL_MEMORY_FRAMES", "4")),
        stride=int(os.environ.get("PI05_VISUAL_MEMORY_STRIDE", "15")),
        pool_grid=int(os.environ.get("PI05_VISUAL_MEMORY_GRID", "4")),
    )
    model = VisualMemoryConfig(
        **{f.name: getattr(baseline.model, f.name) for f in fields(baseline.model)}, memory=memory
    )
    data = LeRobotVisualMemoryDataConfig(**{f.name: getattr(baseline.data, f.name) for f in fields(baseline.data)})
    tag = f"h{memory.frames}_s{memory.stride}_g{memory.pool_grid}"
    data = replace(data, assets=replace(data.assets, asset_id=f"tomato_visual_memory_v1_{tag}_delta_30hz"))
    return replace(
        baseline,
        name=f"pi05_sharpa_in_hand119_visual_memory_v1_{tag}",
        exp_name=f"tomato_visual_memory_v1_{tag}_s42_b256_s30000_c5000",
        model=model,
        data=data,
        policy_metadata={
            **baseline.policy_metadata,
            "memory_to_pi05": memory.frames > 1,
            "visual_memory": memory.contract(),
            "baseline_branch": "tomato-subtask-policy",
            "baseline_commit": "01ead34",
            "baseline_input_acceptance_report": baseline.policy_metadata["input_acceptance_report"],
            "input_acceptance_report": None,
            "history_state_input": False,
            "current_state_input": True,
            "fovea_box_prediction": False,
            "visual_tokens": 1024 + len(memory.history_keys) * memory.pool_grid**2,
            "gpu_memory_preflight_required": True,
            "independent_generalization_measured": False,
        },
    )
