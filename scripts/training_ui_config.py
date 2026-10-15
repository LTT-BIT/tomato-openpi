# ruff: noqa: RUF001
"""Validated UI edits and portable personal Python configurations."""

import ast
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime
import getpass
import hashlib
import math
from pathlib import Path
import pprint
import re
from zoneinfo import ZoneInfo

from openpi.training.sampling import SamplingGroup
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
    entries = baseline_catalog() + personal_catalog()
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
    path = source_path(module)
    config = config_from_source(path.read_text(), module, str(path))
    validate_config(config)
    return replace(config, policy_metadata={**config.policy_metadata, "experiment_module": module})


# Run identity and storage selection are recorded per launch, not new training recipes.
RUN_FIELDS = {"exp_name", "dataset_root", "assets_dir", "asset_id", "project_name"}


def changed_parameters(module: str, values: dict) -> list[str]:
    original = defaults(module)["values"]
    # Resolve paths and normalize tuple/list representations before comparing.
    before = config_values(apply_overrides(source_config(module), original))
    after = config_values(apply_overrides(source_config(module), values))
    return sorted(key for key in after if key not in RUN_FIELDS and after[key] != before[key])


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
        sampling_mode=config.sampling_mode,
        sampling_groups=[asdict(group) for group in config.sampling_groups],
    )
    return values


def defaults(module: str) -> dict:
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
            "task_view": asdict(config.data.base_config.task_view)
            if getattr(config.data.base_config, "task_view", None)
            else None,
            "actions": "absolute",
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
        "sources",
        "sampling_mode",
        "sampling_groups",
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
    spec = replace(config.data.lerobot_datasets[0], root=paths["dataset_root"], repo_id=values["repo_id"])
    data = replace(
        config.data,
        repo_id=values["repo_id"],
        lerobot_datasets=(spec,),
        assets=replace(config.data.assets, assets_dir=paths["assets_dir"], asset_id=values["asset_id"]),
    )
    scheduled = {
        key: tuple(values[key]) if key == "save_steps_before" else values[key]
        for key in SCHEDULE_FIELDS
        if key in values
    }
    result = replace(
        config,
        **{key: values[key] for key in TRAIN_INTS},
        **scheduled,
        exp_name=values["exp_name"],
        project_name=values["project_name"],
        data=data,
        model=replace(config.model, action_horizon=values["action_horizon"]),
        lr_schedule=replace(config.lr_schedule, **{key: values[key] for key in LR_FIELDS}),
        optimizer=replace(config.optimizer, **{key: values[key] for key in OPT_FIELDS}),
        weight_loader=replace(config.weight_loader, params_path=weight_path),
        sampling_mode=values["sampling_mode"],
        sampling_groups=groups,
        resume=False,
        overwrite=False,
        wandb_enabled=True,
        policy_metadata={
            **config.policy_metadata,
            "dataset_root": paths["dataset_root"],
            "dataset_repo_id": values["repo_id"],
            "dataset_version": values["dataset_version"],
            "source_datasets": values["sources"],
        },
    )
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
