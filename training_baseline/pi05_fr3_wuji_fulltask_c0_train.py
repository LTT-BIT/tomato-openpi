# ruff: noqa: RUF002, RUF003
"""完整任务 C0 v3：全参数微调 / 30 Hz / 54 维 / 可复现无放回采样。

使用方式见同目录的训练教程；在仓库根目录通过 python -m training_baseline.pi05_fr3_wuji_fulltask_c0_train 运行。
这是可执行的训练配置，不是独立于仓库的模型实现；无需注册到 config.py。
模型使用非 LoRA 变体，VLM、视觉编码器、Action Expert 和投影层全部训练。
使用已准备好的全部数据和统计，不划分训练/验证集；启动时记录文件指纹，不改写数据。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import tyro

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


CONFIG_NAME = "pi05_fr3_wuji_fulltask_fullft_30hz"
BASELINE_VERSION = "fulltask_c0_v3_fullft"
DEFAULT_EXP_NAME = "tomato_all_full_fullft_30hz_s42"
# 指令示例，仅供参考；实际训练读取准备好的数据中的 task 文本。
FULL_TASK_PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato "
    "with the left hand and place it in the left basket."
)


# 数据仍在共享工作区；环境、缓存、统计和权重默认使用容量充足的本机根盘。
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path(os.environ.get("PI05_RUNTIME_DIR", "/home/zhiwei/pi05-runtime"))
DATASET_SOURCES = tuple(
    WORKSPACE_ROOT / "data/harvest-tomato" / relative for relative in ("0925/A", "0925/B", "1002_AB_190/full")
)
DATASET_REPO_ID = "fr3_wuji/tomato_all_full_v1"
DATASET_ROOT = str(RUNTIME_ROOT / "data/tomato_all_full_v1")
NORM_ASSETS_DIR = str(RUNTIME_ROOT / "assets")
NORM_ASSET_ID = "tomato_all_full_v1_30hz"


def build_config(
    *,
    name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    repo_id: str = DATASET_REPO_ID,
    dataset_root: str = DATASET_ROOT,
    assets_dir: str = NORM_ASSETS_DIR,
    asset_id: str = NORM_ASSET_ID,
    initial_params_path: str = "gs://openpi-assets/checkpoints/pi05_base/params",
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
        "initial_params_path": initial_params_path,
    }
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} 必须为非空字符串")

    for variant in ("gemma_2b", "gemma_300m"):
        if gemma.get_config(variant).lora_configs:
            raise ValueError(f"全参数微调要求 {variant} 不含 LoRA")

    model = pi0_config.Pi0Config(
        pi05=True,
        dtype="bfloat16",
        action_dim=54,
        action_horizon=50,
        max_token_len=256,
        discrete_state_input=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pytorch_compile_mode="max-autotune",
    )

    return cfg.TrainConfig(
        name=name,
        project_name="openpi",
        exp_name=exp_name,
        model=model,
        weight_loader=weight_loaders.PartialCheckpointWeightLoader(
            params_path=initial_params_path,
            skip_on_mismatch_regex=".*(action_in_proj|action_out_proj|state_proj).*",
        ),
        freeze_filter=model.get_freeze_filter(),
        data=cfg.LeRobotFr3WujiDataConfig(
            repo_id=repo_id,
            assets=cfg.AssetsConfig(assets_dir=assets_dir, asset_id=asset_id),
            # 从准备好的数据读取完整任务指令，不执行子任务切分或 prompt 改写。
            base_config=cfg.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            action_sequence_keys=("action",),
            action_hz=30.0,
            video_backend="pyav",
            lerobot_datasets=(cfg.LeRobotDatasetSpec(repo_id=repo_id, root=dataset_root, weight=1.0),),
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
        model_seed=42,
        data_seed=42001,
        augmentation_seed=42002,
        sampling_mode="epoch_shuffle",
        batch_size=8,
        num_workers=8,
        num_train_steps=60_153,
        fsdp_devices=1,
        log_interval=10,
        save_interval=2_000,
        keep_period=10_000,
        assets_base_dir="./assets",
        checkpoint_base_dir=str(RUNTIME_ROOT / "checkpoints"),
        resume=False,
        overwrite=False,
        wandb_enabled=True,
        pytorch_weight_path=None,
        pytorch_training_precision="bfloat16",
        policy_metadata={
            "baseline_version": BASELINE_VERSION,
            "finetuning": "full_parameters",
            "dataset_repo_id": repo_id,
            "dataset_root": dataset_root,
            "source_datasets": (
                [str(source) for source in DATASET_SOURCES] if dataset_root == DATASET_ROOT else [dataset_root]
            ),
            "prompt": FULL_TASK_PROMPT,
        },
    )


def main() -> None:
    """解析与仓库训练入口一致的 CLI 覆盖项并启动 JAX 训练。"""
    default = build_config()
    config = tyro.extras.overridable_config_cli({CONFIG_NAME: (CONFIG_NAME, default)})

    from scripts import train

    train.main(config)


if __name__ == "__main__":
    main()
