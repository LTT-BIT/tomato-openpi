"""可直接执行的 π0.5 / FR3-Wuji / 0924 / 30 Hz 双 LoRA 统一基准。

此版本使用 A/B 完整任务数据和单条完整任务 prompt, 不使用 subtask。
无需注册到 ``src/openpi/training/config.py``。从仓库根目录运行:

    mkdir -p logs
    set -o pipefail
    JAX_PLATFORMS=cuda PYTHONNOUSERSITE=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
    WANDB_MODE=online .venv/bin/python -u \
      pi05_fr3_wuji_0924_unified_double_lora_train.py \
      pi05_fr3_wuji_0924_unified_double_lora_30hz --wandb-enabled \
      2>&1 | tee "logs/tomato_0924_unified_double_lora_30hz_s42_$(date +%Y%m%d_%H%M%S).log"

无网络时使用 ``WANDB_MODE=offline``, 训练完成后再执行 W&B sync。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import TYPE_CHECKING

import tyro

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


CONFIG_NAME = "pi05_fr3_wuji_0924_unified_double_lora_30hz"
DEFAULT_EXP_NAME = "tomato_0924_unified_double_lora_30hz_s42"
DATA_ROOT = "/home/descfly/data/harvest-tomato/0924-250ep"
ASSETS_DIR = "./assets/pi05_fr3_wuji_0924_30hz"
ASSET_ID = "fr3_wuji/0924_30hz"
PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand "
    "and place it in the left basket."
)
ACTION_HORIZON = 50


def build_config(
    *,
    name: str = CONFIG_NAME,
    exp_name: str = DEFAULT_EXP_NAME,
    data_root: str = DATA_ROOT,
    assets_dir: str = ASSETS_DIR,
    asset_id: str = ASSET_ID,
) -> TrainConfig:
    """构造完整任务、双 LoRA 的统一训练配置。"""
    from openpi.models import gemma
    from openpi.models import pi0_config
    from openpi.training import config as cfg
    from openpi.training import optimizer
    from openpi.training import weight_loaders

    values = {
        "name": name,
        "exp_name": exp_name,
        "data_root": data_root,
        "assets_dir": assets_dir,
        "asset_id": asset_id,
    }
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} 必须为非空字符串")

    # 固定 VLM LoRA rank 16 和 action expert LoRA rank 32。
    for variant, expected_rank in (("gemma_2b_lora", 16), ("gemma_300m_lora", 32)):
        adapters = gemma.get_config(variant).lora_configs
        if adapters is None or set(adapters) != {"attn", "ffn"}:
            raise ValueError(f"{variant} 的 LoRA 模块与统一基准不一致")
        for adapter in adapters.values():
            if adapter.rank != expected_rank or adapter.alpha != float(expected_rank):
                raise ValueError(f"{variant} 的 LoRA rank/alpha 与统一基准不一致")

    model = pi0_config.Pi0Config(
        pi05=True,
        dtype="bfloat16",
        action_dim=54,
        action_horizon=ACTION_HORIZON,
        max_token_len=256,
        discrete_state_input=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        pytorch_compile_mode="max-autotune",
    )
    root = pathlib.Path(data_root)

    return cfg.TrainConfig(
        name=name,
        project_name="openpi",
        exp_name=exp_name,
        model=model,
        weight_loader=weight_loaders.PartialCheckpointWeightLoader(
            params_path="gs://openpi-assets/checkpoints/pi05_base/params",
            skip_on_mismatch_regex=".*(action_in_proj|action_out_proj|state_proj).*",
        ),
        # 与双 LoRA C0 一致: 冻结两个 LLM 的非 LoRA 参数, 视觉编码器可训练。
        freeze_filter=model.get_freeze_filter(),
        data=cfg.LeRobotFr3WujiDataConfig(
            repo_id="fr3_wuji/0924_30hz",
            assets=cfg.AssetsConfig(assets_dir=assets_dir, asset_id=asset_id),
            base_config=cfg.DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
            action_sequence_keys=("action",),
            action_hz=30.0,
            video_backend="pyav",
            lerobot_datasets=(
                cfg.LeRobotDatasetSpec(
                    repo_id="fr3_wuji/0924_A",
                    root=str(root / "A"),
                    weight=0.7,
                ),
                cfg.LeRobotDatasetSpec(
                    repo_id="fr3_wuji/0924_B",
                    root=str(root / "B"),
                    weight=0.3,
                ),
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


def _input_fingerprint(datasets: tuple[pathlib.Path, ...], action_hz: float) -> str:
    digest = hashlib.sha256()
    for dataset in datasets:
        for relative in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
            digest.update((dataset / relative).read_bytes())
    digest.update(f"{action_hz:g}:{ACTION_HORIZON}".encode())
    digest.update(b":keep_prompts")
    return digest.hexdigest()


def _validate_inputs(config: TrainConfig) -> None:
    datasets = tuple(pathlib.Path(spec.root) for spec in config.data.lerobot_datasets)
    assets = pathlib.Path(config.data.assets.assets_dir) / config.data.assets.asset_id
    required = [
        *(dataset / relative for dataset in datasets for relative in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl")),
        assets / "norm_stats.json",
        assets / "provenance.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("统一基准缺少必要输入:\n" + "\n".join(missing))

    prompts = {
        row["task"]
        for dataset in datasets
        for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()
        if (row := json.loads(line))
    }
    if prompts != {PROMPT}:
        raise ValueError(f"完整任务 prompt 与基准不一致: {sorted(prompts)}")

    provenance = json.loads((assets / "provenance.json").read_text())
    actual_fingerprint = _input_fingerprint(datasets, config.data.action_hz)
    if actual_fingerprint != provenance.get("input_fingerprint"):
        raise ValueError("当前 A/B 数据与归一化统计的 provenance 指纹不一致")


def main() -> None:
    """解析与仓库训练入口一致的 CLI 覆盖项并启动 JAX 训练。"""
    default = build_config()
    config = tyro.extras.overridable_config_cli({CONFIG_NAME: (CONFIG_NAME, default)})
    _validate_inputs(config)

    from scripts import train

    train.main(config)


if __name__ == "__main__":
    main()
