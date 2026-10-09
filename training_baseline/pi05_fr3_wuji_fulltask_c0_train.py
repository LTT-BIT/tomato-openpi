"""1002 派生的完整任务 C0 基准：双 LoRA / 30 Hz / 54 维 / 20k。

使用方式见同目录的训练教程；在仓库根目录通过 python -m training_baseline.pi05_fr3_wuji_fulltask_c0_train 运行。
这是可执行的训练配置，不是独立于仓库的模型实现；无需注册到 config.py。
只替换数据、归一化资产和完整任务指令；模型与优化参数沿用 1002。
使用已经准备好的完整任务数据和归一化统计，不执行额外的数据校验或改写。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import tyro

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


CONFIG_NAME = "pi05_fr3_wuji_fulltask_c0_30hz"
DEFAULT_EXP_NAME = "tomato_fulltask_c0_30hz_s42"
# 指令示例，仅供参考；实际训练读取准备好的数据中的 task 文本。
FULL_TASK_PROMPT = (
    "Pick up a tomato truss with the right hand and hold it steady between both hands, "
    "then pick a single cherry tomato from the truss with the left hand "
    "and place the picked tomato into the plate on the left."
)


# 填写已经准备好的数据与归一化路径；相对路径以仓库根目录为基准。
DATASET_REPO_ID = "fr3_wuji/tomato_fulltask_v1"
DATASET_ROOT = "./data/fr3_wuji/tomato_fulltask_v1"
NORM_ASSETS_DIR = "./assets/fulltask_c0"
NORM_ASSET_ID = "tomato_fulltask_v1_30hz"


def build_config(
    *,
    name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    repo_id: str = DATASET_REPO_ID,
    dataset_root: str = DATASET_ROOT,
    assets_dir: str = NORM_ASSETS_DIR,
    asset_id: str = NORM_ASSET_ID,
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
            # 从准备好的数据读取完整任务指令，不执行子任务切分或 prompt 改写。
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


def main() -> None:
    """解析与仓库训练入口一致的 CLI 覆盖项并启动 JAX 训练。"""
    default = build_config()
    config = tyro.extras.overridable_config_cli({CONFIG_NAME: (CONFIG_NAME, default)})

    from scripts import train

    train.main(config)


if __name__ == "__main__":
    main()
