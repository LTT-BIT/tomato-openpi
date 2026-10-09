"""可直接执行的 π0.5 / FR3-Wuji / 1002 / 30 Hz 统一训练基准 C0。

本文件不需要注册到 ``src/openpi/training/config.py``。从仓库根目录运行:

    mkdir -p logs
    set -o pipefail
    JAX_PLATFORMS=cuda PYTHONNOUSERSITE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    WANDB_MODE=online .venv/bin/python -u pi05_fr3_wuji_1002_unified_train.py \
      pi05_fr3_wuji_1002_unified_30hz --wandb-enabled \
      2>&1 | tee "logs/tomato_1002_unified_30hz_s42_$(date +%Y%m%d_%H%M%S).log"

无网络时使用 ``WANDB_MODE=offline``, 训练完成后再执行 W&B sync。
数据集、归一化统计、预训练权重和 tokenizer 仍须在本机可访问。
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import tyro

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


CONFIG_NAME = "pi05_fr3_wuji_1002_unified_30hz"
DEFAULT_EXP_NAME = "tomato_1002_unified_30hz_s42"


def build_config(
    *,
    name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    repo_id: str = "fr3_wuji/1002_subtasks_detailed",
    dataset_root: str = "/home/descfly/data-clean/1002_subtasks_detailed",
    assets_dir: str = "/home/descfly/data-clean",
    asset_id: str = "1002_subtasks_detailed",
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


def _validate_inputs(config: TrainConfig) -> None:
    dataset = pathlib.Path(config.data.lerobot_datasets[0].root)
    norm_stats = pathlib.Path(config.data.assets.assets_dir) / config.data.assets.asset_id / "norm_stats.json"
    required = (dataset / "meta/info.json", dataset / "meta/tasks.jsonl", norm_stats)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("统一基准缺少必要输入:\n" + "\n".join(missing))


def main() -> None:
    """解析与仓库训练入口一致的 CLI 覆盖项并启动 JAX 训练。"""
    default = build_config()
    config = tyro.extras.overridable_config_cli({CONFIG_NAME: (CONFIG_NAME, default)})
    _validate_inputs(config)

    from scripts import train

    train.main(config)


if __name__ == "__main__":
    main()
