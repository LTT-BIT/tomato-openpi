"""1002 派生的完整任务 C0 基准：双 LoRA / 30 Hz / 54 维 / 20k。

放在匹配版本的 Pi05 仓库根目录，参照 docs/训练教程_完整任务C0.md。
这是可执行的训练配置，不是独立于仓库的模型实现；无需注册到 config.py。
只替换数据、归一化资产和完整任务指令；模型与优化参数沿用 1002。
本文件不切分、拼接或改写数据。完整任务是否真实连续，仍需检查原始采集。
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING

import tyro

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


CONFIG_NAME = "pi05_fr3_wuji_fulltask_c0_30hz"
DEFAULT_EXP_NAME = "tomato_fulltask_c0_30hz_s42"
FULL_TASK_PROMPT = (
    "Pick up a tomato truss with the right hand and hold it steady between both hands, "
    "then pick a single cherry tomato from the truss with the left hand "
    "and place the picked tomato into the plate on the left."
)


def build_config(
    *,
    name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    repo_id: str = "fr3_wuji/tomato_fulltask_v1",
    dataset_root: str = "./data/fr3_wuji/tomato_fulltask_v1",
    assets_dir: str = "./assets/fulltask_c0",
    asset_id: str = "tomato_fulltask_v1_30hz",
) -> TrainConfig:
    """构造显式展开的统一 C0 配置。"""
    from openpi.models import gemma
    from openpi.models import pi0_config
    from openpi.training import config as cfg
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    values = {
        "name": name,
        "exp_name": exp_name,
        "repo_id": repo_id,
        "dataset_root": dataset_root,
        "assets_dir": assets_dir,
        "asset_id": asset_id,
    }
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} 必须为非空字符串")

    # 固定双 LoRA 实现, 避免同名 variant 被修改后基准静默漂移。
    for variant, expected_rank in (("gemma_2b_lora", 16), ("gemma_300m_lora", 32)):
        adapters = gemma.get_config(variant).lora_configs
        if adapters is None or set(adapters) != {"attn", "ffn"}:
            raise ValueError(f"{variant} 的 LoRA 模块与 1002 基线不一致")
        for adapter in adapters.values():
            if adapter.rank != expected_rank or adapter.alpha != float(expected_rank):
                raise ValueError(f"{variant} 的 LoRA rank/alpha 与 1002 基线不一致")

    model = pi0_config.Pi0Config(
        pi05=True,
        dtype="bfloat16",
        action_dim=54,
        action_horizon=50,
        max_token_len=256,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        pytorch_compile_mode="max-autotune",
    )

    return cfg.TrainConfig(
        name=name,
        project_name="openpi",
        exp_name=exp_name,
        model=model,
        weight_loader=weight_loaders.PartialCheckpointWeightLoader(
            params_path="gs://openpi-assets/checkpoints/pi05_base/params",
            skip_on_mismatch_regex=".*(action_in_proj|action_out_proj|state_proj).*",
        ),
        freeze_filter=model.get_freeze_filter(),
        data=cfg.LeRobotFr3WujiDataConfig(
            repo_id=repo_id,
            assets=cfg.AssetsConfig(assets_dir=assets_dir, asset_id=asset_id),
            # task 表中只允许一条完整任务指令；保留语言条件输入，取消分阶段指令。
            base_config=cfg.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            action_sequence_keys=("action",),
            action_hz=30.0,
            video_backend="pyav",
            lerobot_datasets=(
                cfg.LeRobotDatasetSpec(repo_id=repo_id, root=dataset_root, weight=1.0),
            ),
        ),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        optimizer=optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=None,
        seed=42,
        batch_size=8,
        num_workers=8,
        num_train_steps=20_000,
        fsdp_devices=1,
        log_interval=10,
        save_interval=2_000,
        keep_period=10_000,
        assets_base_dir="./assets",
        checkpoint_base_dir="./checkpoints",
        resume=False,
        overwrite=False,
        wandb_enabled=False,
        pytorch_weight_path=None,
        pytorch_training_precision="bfloat16",
        policy_metadata=None,
    )


def validate_inputs(config: TrainConfig, *, require_norm: bool = True) -> None:
    """检查完整任务标签与统计；不声称能从元数据证明轨迹完整性。"""
    import numpy as np
    from openpi.shared import normalize

    for spec in config.data.lerobot_datasets:
        if spec.root is None:
            raise ValueError("完整任务基准要求显式提供本地 dataset root")
        root = pathlib.Path(spec.root)
        required = [root / "meta" / name for name in ("info.json", "tasks.jsonl", "episodes.jsonl")]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("缺少完整任务数据文件：\n" + "\n".join(missing))
        info = json.loads(required[0].read_text())
        if info["fps"] < config.data.action_hz:
            raise ValueError("源数据 FPS 不能低于模型动作频率")
        tasks = [json.loads(line) for line in required[1].read_text().splitlines() if line.strip()]
        if len(tasks) != 1 or tasks[0].get("task") != FULL_TASK_PROMPT:
            raise ValueError("本基准要求 tasks.jsonl 仅含 FULL_TASK_PROMPT；不能直接使用四类 subtask 数据")
        episodes = [json.loads(line) for line in required[2].read_text().splitlines() if line.strip()]
        if not episodes or len(episodes) != info["total_episodes"]:
            raise ValueError("episode 数量为空或与 info.json 不一致")
        for ep in episodes:
            if "subtask" in str(ep.get("segment_id", "")).lower():
                raise ValueError("发现 subtask segment_id：请使用原始完整任务轨迹，不能仅改标签")
            if ep.get("tasks") != [FULL_TASK_PROMPT] or ep.get("length", 0) <= 0:
                raise ValueError("episode 的任务指令或长度不符合完整任务基准")
    if require_norm:
        norm_dir = pathlib.Path(config.data.assets.assets_dir) / config.data.assets.asset_id
        stats = normalize.load(norm_dir)
        for key in ("state", "actions"):
            for field in ("mean", "std", "q01", "q99"):
                value = getattr(stats[key], field)
                if value is None or value.shape != (54,) or not np.isfinite(value).all():
                    raise ValueError(f"归一化统计 {key}.{field} 必须是有限的 54 维数组")


# 保留与旧统一文件类似的校验调用接口。
_validate_inputs = validate_inputs


def main() -> None:
    """解析与仓库训练入口一致的 CLI 覆盖项并启动 JAX 训练。"""
    default = build_config()
    config = tyro.extras.overridable_config_cli({CONFIG_NAME: (CONFIG_NAME, default)})
    _validate_inputs(config)

    from scripts import train

    train.main(config)


if __name__ == "__main__":
    main()
