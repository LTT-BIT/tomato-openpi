# ruff: noqa: RUF001
"""Validated UI edits and portable personal Python configurations."""

import ast
from dataclasses import asdict
from dataclasses import fields
from dataclasses import replace
from datetime import datetime
import getpass
import hashlib
import math
from pathlib import Path
import pprint
import re
from zoneinfo import ZoneInfo

from openpi.subtasks.tomato import GOAL
from openpi.subtasks.tomato_in_hand import CAMERAS
from openpi.subtasks.tomato_in_hand import TASKS
from openpi.subtasks.visual_memory import IMAGE_KEYS
from openpi.training import config as train_config
from openpi.training.delta_paper import PaperVisualMemoryData
from openpi.training.delta_paper import PaperVisualMemoryOptionsData
from openpi.training.delta_paper_ui import paper_available
from openpi.training.delta_paper_ui import select_trial
from openpi.training.delta_paper_ui import selected_trial
from openpi.training.sampling import SamplingGroup
from openpi.training.visual_memory import LeRobotVisualMemoryDataConfig
from openpi.training.visual_memory_options import LeRobotVisualMemoryOptionsDataConfig
from scripts.run_training import load_experiment
from scripts.run_training import validate_config
from training_configs import common

TASK_LABELS = {
    "tomato_wuji": "番茄 · Wuji",
    "tomato_sharpa": "番茄 · Sharpa",
    "lemon_scooping": "铲柠檬 · UR/O30",
    "cup_grasping": "抓取杯子 · Franka",
}
TRAIN_INTS = (
    "batch_size",
    "num_train_steps",
    "num_workers",
    "fsdp_devices",
    "log_interval",
    "save_interval",
    "keep_period",
    "seed",
    "model_seed",
    "data_seed",
    "augmentation_seed",
)
LR_FIELDS = ("warmup_steps", "peak_lr", "decay_steps", "decay_lr")
OPT_FIELDS = ("weight_decay", "clip_gradient_norm")
SCHEDULE_FIELDS = ("save_before_step", "save_steps_before", "save_interval_after")
TEXT_FIELDS = (
    "exp_name",
    "dataset_root",
    "repo_id",
    "dataset_version",
    "assets_dir",
    "asset_id",
    "weight_path",
    "project_name",
)
DEFAULT_AB_MODULE = "training_configs.experiments.tomato_sharpa.subtasks.visual_memory_v1"
AB_LOCKED_FIELDS = (
    "dataset_root",
    "repo_id",
    "dataset_version",
    "sources",
    "assets_dir",
    "asset_id",
    "action_horizon",
    "rtc_max_delay",
    "action_semantics",
    "sampling_mode",
    "sampling_groups",
)


def experiment_catalog():
    root = common.repository_root()
    entries = []
    for stem, label in (
        ("visual_memory_v1", "整串番茄 · 五阶段 · 119条四路RGB + 视觉历史"),
        ("in_hand119_b_crop", "整串番茄 · 手中调整五阶段 · 119条四路RGB · B · 30000步"),
        ("normal93_b_stage", "整串番茄 · B 阶段指令 · 手臂Delta · 全93条 · 全参"),
        ("normal93_a_goal", "整串番茄 · A 总体目标 · 手臂Delta · 全93条 · 全参"),
    ):
        path = Path("training_configs/experiments/tomato_sharpa/subtasks") / f"{stem}.py"
        if (root / path).is_file():
            entries.append(
                {"id": ".".join(path.with_suffix("").parts), "label": label, "task": "tomato_sharpa", "file": str(path)}
            )
    return entries


def baseline_catalog() -> list[dict]:
    root = common.repository_root()
    entries = []
    for path in sorted((root / "training_configs/experiments").glob("*/baseline/*.py")):
        if not any(
            isinstance(node, ast.FunctionDef) and node.name == "build_config"
            for node in ast.parse(path.read_text()).body
        ):
            continue
        task = path.parents[1].name
        entries.append(
            {
                "id": ".".join(path.relative_to(root).with_suffix("").parts),
                "label": f"{TASK_LABELS.get(task, task)} / {'LoRA' if 'lora' in path.stem else '全参'}",
                "task": task,
                "file": str(path.relative_to(root)),
            }
        )
    return entries


def baseline_config(module: str):
    if module not in {item["id"] for item in baseline_catalog()}:
        raise ValueError("请选择仓库中已发布的 baseline")
    return load_experiment(module)


def personal_catalog() -> list[dict]:
    root = common.repository_root().resolve()
    entries = []
    for path in sorted((root / "training_configs/experiments").glob("*/template/*/*.py")):
        if path.name.startswith("_") or not path.resolve().is_relative_to(root):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (OSError, SyntaxError, UnicodeError):
            continue
        if not any(isinstance(n, ast.FunctionDef) and n.name == "build_config" for n in tree.body):
            continue
        parts = path.relative_to(root).with_suffix("").parts
        if not all(part.isidentifier() for part in parts):
            continue
        entries.append(
            {
                "id": ".".join(parts),
                "label": f"{TASK_LABELS.get(parts[2], parts[2])} / {parts[4]} / {path.stem}",
                "task": parts[2],
                "file": str(path.relative_to(root)),
            }
        )
    return entries


def source_path(module: str) -> Path:
    entries = baseline_catalog() + experiment_catalog() + personal_catalog()
    entry = next((item for item in entries if item["id"] == module), None)
    if entry is None:
        raise ValueError("请选择当前仓库的 baseline 或个人配置")
    return common.repository_root() / entry["file"]


def config_from_source(source: str, module: str, filename: str):
    # A separate namespace keeps repeated UI copies from overwriting earlier wrappers.
    scope = {"__name__": module, "__package__": module.rpartition(".")[0], "__file__": filename}
    exec(compile(source, filename, "exec"), scope)
    return scope["build_config"]()


def source_config(module: str):
    if module in {item["id"] for item in baseline_catalog()}:
        return baseline_config(module)
    if module in {item["id"] for item in experiment_catalog()}:
        return load_experiment(module)
    path = source_path(module)
    config = config_from_source(path.read_text(), module, str(path))
    validate_config(config)
    return replace(config, policy_metadata={**config.policy_metadata, "experiment_module": module})


# Run identity and storage selection are recorded per launch, not new training recipes.
RUN_FIELDS = {"exp_name", "dataset_root", "dataset_roots", "assets_dir", "asset_id", "project_name"}


def changed_parameters(module: str, values: dict) -> list[str]:
    original = defaults(module)["values"]
    # Resolve paths and normalize tuple/list representations before comparing.
    before = config_values(apply_overrides(source_config(module), original))
    after = config_values(apply_overrides(source_config(module), values))
    return sorted(key for key in after if key not in RUN_FIELDS and after[key] != before[key])


def delta_supported(config) -> bool:
    return isinstance(config.data, train_config.LeRobotFr3WujiDataConfig | train_config.LeRobotFr3SharpaDataConfig)


def input_values(config):
    if config.policy_metadata.get("subtask_variant") != "in_hand_b_coarse" or not isinstance(
        config.data, LeRobotVisualMemoryDataConfig
    ):
        return {}
    return {
        "prompt_mode": "global" if getattr(config.data, "global_prompt", None) else "stage",
        "video_views": len(config.model.memory.image_keys),
    }


def input_suffix(options):
    parts = []
    if options["prompt_mode"] == "global":
        parts.append("global")
    if options["video_views"] == 3:
        parts.append("rgb3")
    return "".join(f"_{part}" for part in parts)


def apply_training_inputs(config, before, after):
    if before == after:
        return config
    global_prompt = GOAL if after["prompt_mode"] == "global" else None
    camera_count = after["video_views"]
    memory = replace(config.model.memory, image_keys=IMAGE_KEYS[:camera_count])
    current_fields = {field.name: getattr(config.data, field.name) for field in fields(config.data)}
    trial = selected_trial(config)
    if after == {"prompt_mode": "stage", "video_views": 4}:
        cls = PaperVisualMemoryData if trial != "none" else LeRobotVisualMemoryDataConfig
        current_fields.pop("global_prompt", None)
    else:
        cls = PaperVisualMemoryOptionsData if trial != "none" else LeRobotVisualMemoryOptionsDataConfig
        current_fields["global_prompt"] = global_prompt
    paper_suffix = f"_paper_{trial}" if trial != "none" else ""
    name = config.name.removesuffix(paper_suffix).removesuffix(input_suffix(before))
    return replace(
        config,
        name=f"{name}{input_suffix(after)}{paper_suffix}",
        model=replace(config.model, memory=memory),
        data=cls(**current_fields),
        policy_metadata={
            **config.policy_metadata,
            **after,
            "camera_keys": list(CAMERAS[:camera_count]),
            "visual_memory": memory.contract(),
            "visual_tokens": camera_count * 256 + len(memory.history_keys) * memory.pool_grid**2,
            "offline_roi_uses_future_annotation": camera_count == 4,
            "prompt_example": global_prompt or TASKS[1],
            "prompts": [global_prompt] if global_prompt else list(TASKS),
            "prompt_source": "fixed_global_prompt" if global_prompt else "meta/tasks.jsonl",
        },
    )


def locked_fields(config, trial, *, inputs_changed=False):
    metadata = config.policy_metadata
    if metadata.get("subtask_variant") == "in_hand_b_coarse":
        editable = {"rtc_max_delay"}
        if trial != "none" or selected_trial(config) != "none" or inputs_changed:
            editable.update(("assets_dir", "asset_id"))
        return [key for key in AB_LOCKED_FIELDS if key not in editable]
    return list(AB_LOCKED_FIELDS) if metadata.get("selection_sha256") else []


def config_values(config) -> dict:
    values = {key: getattr(config, key) for key in TRAIN_INTS}
    values.update({key: getattr(config.lr_schedule, key) for key in LR_FIELDS})
    values.update({key: getattr(config.optimizer, key) for key in OPT_FIELDS})
    values.update({key: getattr(config, key) for key in SCHEDULE_FIELDS if hasattr(config, key)})
    values.update(
        exp_name=config.exp_name,
        dataset_root=config.data.lerobot_datasets[0].root,
        repo_id=config.data.repo_id,
        dataset_version=config.policy_metadata["dataset_version"],
        sources=config.policy_metadata["source_datasets"],
        assets_dir=str(config.data.assets.assets_dir),
        asset_id=config.data.assets.asset_id,
        weight_path=config.weight_loader.params_path,
        project_name=config.project_name,
        action_horizon=config.model.action_horizon,
        rtc_max_delay=config.model.rtc_max_delay,
        action_semantics="delta" if getattr(config.data, "extra_delta_transform", False) else "absolute",
        sampling_mode=config.sampling_mode,
        sampling_groups=[asdict(group) for group in config.sampling_groups],
        paper_trial=selected_trial(config),
    )
    if len(config.data.lerobot_datasets) > 1:
        values["dataset_roots"] = [spec.root for spec in config.data.lerobot_datasets]
    values.update(input_values(config))
    return values


def defaults(module: str) -> dict:
    from openpi.subtasks.tomato import ACTION_PROMPTS

    config = source_config(module)
    values = config_values(config)
    owner = re.sub(r"\W", "_", getpass.getuser())
    if not owner or owner[0].isdigit():
        owner = "user_" + owner
    shared = Path(common.SERVER_BASE_PARAMS)
    return {
        "values": values,
        "source_sha256": hashlib.sha256(source_path(module).read_bytes()).hexdigest(),
        "owner": owner,
        "suggested_exp_name": f"{owner}_{config.exp_name}_{datetime.now(ZoneInfo('Asia/Shanghai')):%m%d_%H%M%S}",
        "shared_weights": str(shared) if shared.is_dir() else None,
        "protocol": {
            "action_dim": config.policy_metadata["physical_action_dim"],
            "model_action_dim": config.model.action_dim,
            "action_hz": config.data.action_hz,
            "cameras": config.policy_metadata["camera_keys"],
            "prompt": config.policy_metadata["prompt_example"],
            "global_prompt": GOAL if input_values(config) else None,
            "stage_prompt": TASKS[1] if input_values(config) else None,
            "task_view": asdict(config.data.base_config.task_view)
            if getattr(config.data.base_config, "task_view", None)
            else None,
            "actions": values["action_semantics"],
            "delta_available": delta_supported(config),
            "subtask_variant": config.policy_metadata.get("subtask_variant"),
            "data_policy": config.policy_metadata.get("data_policy"),
            "locked_fields": locked_fields(config, values["paper_trial"]),
            "paper_available": paper_available(config),
            "visual_memory": config.policy_metadata.get("visual_memory"),
            "bundle_missing": config.policy_metadata.get("subtask_variant") == "in_hand_b_coarse"
            and not config.policy_metadata.get("bundle_sha256"),
            "action_prompts": config.policy_metadata.get("action_prompts",
                ACTION_PROMPTS if config.policy_metadata.get("subtask_variant") == "b_coarse" else {}),
            "training_episodes": config.policy_metadata.get("training_episodes"),
            "training_frames": config.policy_metadata.get("training_frames"),
            "prompt_profile": config.policy_metadata.get("prompt_profile"),
        },
    }


def identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"{label} 只能包含英文字母、数字、下划线，不能以数字开头")
    return value


def apply_overrides(config, values: dict):
    """Only allow presentation fields; preserve embodiment, task view and freeze filter."""
    allowed = set(TRAIN_INTS + LR_FIELDS + OPT_FIELDS + TEXT_FIELDS + SCHEDULE_FIELDS) | {
        "action_horizon",
        "rtc_max_delay",
        "action_semantics",
        "sources",
        "sampling_mode",
        "sampling_groups",
        "dataset_roots",
        "paper_trial",
        "prompt_mode",
        "video_views",
    }
    if unknown := set(values) - allowed:
        raise ValueError(f"不支持的参数: {sorted(unknown)}")
    # Generated files store all values, so behavior does not depend on later environment variables.
    required = set(TRAIN_INTS + LR_FIELDS + OPT_FIELDS + TEXT_FIELDS) | {
        "action_horizon",
        "sources",
        "sampling_mode",
        "sampling_groups",
    }
    if missing := required - set(values):
        raise ValueError(f"缺少参数: {sorted(missing)}")
    paper_trial = values.get("paper_trial", selected_trial(config))
    if not isinstance(paper_trial, str):
        raise ValueError("paper_trial 必须为字符串")
    before_inputs = input_values(config)
    if not before_inputs and {"prompt_mode", "video_views"} & values.keys():
        raise ValueError("此配置不支持切换训练指令或视频输入")
    after_inputs = {key: values.get(key, value) for key, value in before_inputs.items()}
    if after_inputs and (
        after_inputs["prompt_mode"] not in ("stage", "global")
        or type(after_inputs["video_views"]) is not int
        or after_inputs["video_views"] not in (3, 4)
    ):
        raise ValueError("请选择阶段指令或全局 Prompt，以及三路或四路视频")
    inputs_changed = before_inputs != after_inputs
    if inputs_changed and values["asset_id"] == config.data.assets.asset_id:
        raise ValueError("切换训练指令或视频输入需要独立的统计 asset_id")
    locked = locked_fields(config, paper_trial, inputs_changed=inputs_changed)
    if locked:
        original = config_values(config)
        for key in locked:
            actual, expected = values.get(key, original[key]), original[key]
            if key in ("dataset_root", "assets_dir"):
                actual, expected = Path(actual).expanduser().resolve(), Path(expected).expanduser().resolve()
            if actual != expected:
                raise ValueError(f"当前实验固定已验收的数据、统计和监督，不能修改 {key}；请新建实验版本")
        roots = values.get("dataset_roots", [values["dataset_root"]])
        if [str(Path(root).expanduser().resolve()) for root in roots] != [
            str(Path(original["dataset_root"]).resolve())
        ]:
            raise ValueError("当前实验固定已验收的数据，不能更换或增加数据目录")
    source_semantics = "delta" if getattr(config.data, "extra_delta_transform", False) else "absolute"
    action_semantics = values.get("action_semantics", source_semantics)
    if action_semantics not in ("absolute", "delta"):
        raise ValueError("动作监督只能选择 absolute 或 delta")
    delta_available = delta_supported(config)
    if action_semantics == "delta" and not delta_available:
        raise ValueError("当前仅番茄 Wuji / Sharpa 支持 delta 动作监督")
    if action_semantics != source_semantics and values["asset_id"] == config.data.assets.asset_id:
        raise ValueError("切换动作监督后必须使用新的 asset_id，并重新计算归一化统计")
    for key in TEXT_FIELDS:
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} 不能为空")
    identifier(values["exp_name"], "实验名称")
    for key in (
        *TRAIN_INTS,
        "action_horizon",
        "warmup_steps",
        "decay_steps",
        "save_before_step",
        "save_interval_after",
    ):
        if key not in values:
            continue
        value = values[key]
        if key == "keep_period" and value is None:
            continue
        minimum = (
            0
            if key
            in (
                "num_workers",
                "warmup_steps",
                "seed",
                "model_seed",
                "data_seed",
                "augmentation_seed",
                "save_before_step",
            )
            else 1
        )
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} 必须是 >= {minimum} 的整数")
    for key in ("peak_lr", "decay_lr", *OPT_FIELDS):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} 必须是有限非负数")
    if values["peak_lr"] <= 0 or values["clip_gradient_norm"] <= 0 or values["decay_lr"] > values["peak_lr"]:
        raise ValueError("peak_lr / clip_gradient_norm 必须大于零，decay_lr 不能高于 peak_lr")
    if values["warmup_steps"] >= values["decay_steps"]:
        raise ValueError("decay_steps 必须大于 warmup_steps")
    if (
        not isinstance(values["sources"], list)
        or not values["sources"]
        or any(not isinstance(s, str) or not s.strip() for s in values["sources"])
    ):
        raise ValueError("请记录至少一个数据来源")
    if any(key in values for key in SCHEDULE_FIELDS) and not all(hasattr(config, key) for key in SCHEDULE_FIELDS):
        raise ValueError("该 baseline 不支持分阶段保存设置")
    if "save_steps_before" in values and (
        not isinstance(values["save_steps_before"], list | tuple)
        or any(type(i) is not int or i <= 0 for i in values["save_steps_before"])
    ):
        raise ValueError("save_steps_before 必须是正整数列表")
    paths = {}
    for key in ("dataset_root", "assets_dir"):
        path = Path(values[key]).expanduser()
        if not path.is_absolute():
            path = common.repository_root() / path
        paths[key] = str(path.resolve())
    asset = Path(values["asset_id"])
    if asset.is_absolute() or ".." in asset.parts or asset == Path("."):
        raise ValueError("asset_id 必须是统计目录下的相对名称")
    weight_path = values["weight_path"]
    if "://" not in weight_path:
        path = Path(weight_path).expanduser()
        weight_path = str((path if path.is_absolute() else common.repository_root() / path).resolve())
    groups = tuple(
        SamplingGroup(name=g["name"], episodes=tuple(g["episodes"]), weight=g["weight"])
        for g in values["sampling_groups"]
    )
    roots = values.get("dataset_roots", [paths["dataset_root"]])
    if not isinstance(roots, list) or not roots or any(not isinstance(root, str) or not root.strip() for root in roots):
        raise ValueError("dataset_roots 必须是非空目录列表")
    roots = [
        str(
            (
                Path(root).expanduser() if Path(root).expanduser().is_absolute() else common.repository_root() / root
            ).resolve()
        )
        for root in roots
    ]
    if len(set(roots)) != len(roots):
        raise ValueError("数据目录不能重复（含指向同一位置的链接）")
    if roots[0] != paths["dataset_root"]:
        raise ValueError("dataset_root 必须与 dataset_roots 的第一项一致")
    specs = tuple(
        replace(
            config.data.lerobot_datasets[0],
            root=root,
            repo_id=values["repo_id"] if len(roots) == 1 else f"{values['repo_id']}_part{i:03d}",
            weight=1.0,
        )
        for i, root in enumerate(roots)
    )
    data = replace(
        config.data,
        repo_id=values["repo_id"],
        lerobot_datasets=specs,
        assets=replace(config.data.assets, assets_dir=paths["assets_dir"], asset_id=values["asset_id"]),
        **({"extra_delta_transform": action_semantics == "delta"} if delta_available else {}),
    )
    name = config.name
    if action_semantics != source_semantics:
        old, new = source_semantics, action_semantics
        if f"_{old}_" in name:
            name = name.replace(f"_{old}_", f"_{new}_", 1)
        elif name.endswith(f"_{old}"):
            name = f"{name[: -len(old)]}{new}"
        else:
            name = f"{name}_{new}"
    scheduled = {
        key: tuple(values[key]) if key == "save_steps_before" else values[key]
        for key in SCHEDULE_FIELDS
        if key in values
    }
    result = replace(
        config,
        name=name,
        **{key: values[key] for key in TRAIN_INTS},
        **scheduled,
        exp_name=values["exp_name"],
        project_name=values["project_name"],
        data=data,
        model=replace(
            config.model,
            action_horizon=values["action_horizon"],
            rtc_max_delay=values.get("rtc_max_delay", config.model.rtc_max_delay),
        ),
        lr_schedule=replace(config.lr_schedule, **{key: values[key] for key in LR_FIELDS}),
        optimizer=replace(config.optimizer, **{key: values[key] for key in OPT_FIELDS}),
        weight_loader=replace(config.weight_loader, params_path=weight_path),
        sampling_mode=values["sampling_mode"],
        sampling_groups=groups,
        resume=False,
        overwrite=False,
        wandb_enabled=True,
        policy_metadata={
            **{key: value for key, value in config.policy_metadata.items() if key != "dataset_roots"},
            **(
                {"training_action_semantics": "delta_arm_joint_target_absolute_hand"}
                if action_semantics == "delta"
                else {"training_action_semantics": "absolute_joint_target"}
                if action_semantics != source_semantics
                else {}
            ),
            "dataset_root": paths["dataset_root"],
            **({"dataset_roots": roots} if len(roots) > 1 else {}),
            "dataset_repo_id": values["repo_id"],
            "dataset_version": values["dataset_version"],
            "source_datasets": values["sources"],
        },
    )
    result = apply_training_inputs(result, before_inputs, after_inputs)
    result = select_trial(config, result, paper_trial)
    validate_config(result)
    return result


def personal_path(module: str, owner: str, exp_name: str) -> Path:
    source_path(module)
    identifier(owner, "用户名")
    identifier(exp_name, "实验名称")
    task = module.split(".")[2]
    root = common.repository_root().resolve()
    destination = root / "training_configs/experiments" / task / "template" / owner / f"{exp_name}.py"
    if not destination.resolve().is_relative_to(root):
        raise ValueError("个人配置目录不能通过符号链接指向仓库之外")
    return destination


def personal_source(module: str, owner: str, values: dict) -> str:
    config = apply_overrides(source_config(module), values)
    identifier(owner, "用户名")
    path = source_path(module)
    source = path.read_text()
    values = config_values(config)
    metadata = {
        "ui_owner": owner,
        "ui_baseline": module,
        "ui_baseline_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    return (
        "# UI 保存的个人配置；来源源码独立保存，避免依赖其他个人文件。\n"
        "from dataclasses import replace\n"
        "from scripts.training_ui_config import apply_overrides, config_from_source\n"
        f"_SOURCE = {source!r}\n"
        f"_SOURCE_MODULE = {module!r}\n"
        f"_SOURCE_FILE = {str(path)!r}\n"
        f"_UI_VALUES = {pprint.pformat(values, width=100, sort_dicts=False)}\n"
        f"_UI_METADATA = {metadata!r}\n\n"
        "def build_config():\n"
        "    original = config_from_source(_SOURCE, _SOURCE_MODULE, _SOURCE_FILE)\n"
        "    config = apply_overrides(original, _UI_VALUES)\n"
        "    return replace(config, policy_metadata={**config.policy_metadata, **_UI_METADATA})\n"
    )
