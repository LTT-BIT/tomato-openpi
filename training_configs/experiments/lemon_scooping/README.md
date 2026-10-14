# 铲柠檬训练说明

安装与通用训练流程见[根 README](../../../README.md)，个人实验与 subtask 配置见[多人开发 README](../../../多人开发README.md)。这里仅包含训练，不包含机器人部署。

## 两份 baseline

| 设置 | LoRA | 全参数 |
| --- | --- | --- |
| 模块末段 | `lemon_ur_o30_lora_v1` | `lemon_ur_o30_fullft_v1` |
| 配置名称 | `pi05_lemon_ur_o30_lora_v1_10hz` | `pi05_lemon_ur_o30_fullft_v1_10hz` |
| 默认实验名 | `lemon_lora_v1_10hz_s42` | `lemon_fullft_v1_10hz_s42` |
| 参数来源 | 与番茄公共 LoRA recipe 对齐 | `lemon_wrist_reset_full_ft_30k/task.json`、`env.sh`、训练入口 |
| 初始化 | 官方 π0.5 base | 官方 π0.5 base |
| 全局 batch / workers | 8 / 8 | 64 / 8；原记录在 8 卡上每卡 8 |
| 总步数 | 30,000 | 30,000 |
| warmup / peak / decay end | 1,000 / 2.5e-5 / 2.5e-6 | 1,000 / 2.5e-5 / 2.5e-6 |
| decay_steps | 30,000 | 30,000 |
| 可训练范围 | 语言与动作专家 LoRA、视觉和投影 | 全部参数 |
| 日志间隔 | 10 | 20 |
| checkpoint | 沿用公共 recipe：间隔 2,000，周期保留 10,000 | 完成 5k、10k、15k；从 20k 起每 1k，及最后一步；保留这些节点 |
| W&B | `openpi`，启用 | `pi05-lemon-full-ft`，启用 |

两份均 **不设验证集、不计算验证指标、不按验证指标筛选权重**。全参原快照的“500 步验证、综合指标最佳 11 + 最新”已按当前要求取消。全参 checkpoint 目录编号表示已完成更新次数；公共 LoRA 仍沿用已有训练器的零起始 step 编号。

两份学习率均沿用公共 recipe；全参来源记录中的 1e-5 → 1e-6 仅为历史参数，不是当前默认值。

共同设置：horizon 50、bfloat16、max_token_len 256、离散状态 token、AdamW β=(0.9,0.95)、eps=1e-8、weight_decay=1e-10、clip=1、EMA 关闭。种子和可复现采样沿用当前仓库。LoRA 不复刻历史 50k、旧 9000 权重初始化或视觉冻结实验。

## 本体协议

- 双 UR + O30；状态 **52 维**：左 UR 6、左 O30 20、右 UR 6、右 O30 20。机械臂为弧度，手部为原始 0–255。
- 动作 **12 维**：左 UR 6 + 右 UR 6，绝对关节目标，单位弧度；不预测手部动作，不额外转 delta。
- 状态按 52 维、动作按 12 维分别归一化；模型动作头 **32 维**。完整 52 维状态先进入离散 token，不截成 32 维；动作补零到 32 维。
- 三路 RGB：cam0 主相机、cam1 左腕、cam2 右腕。数据与动作频率 **10 Hz**，horizon 50 覆盖约 5 秒。

相比番茄：Wuji 为 54 维动作，Sharpa 为 58 维，均为 30 Hz、absolute 动作；柠檬为 12 维绝对动作、10 Hz。相比杯子：杯子 state/action 都是 7 维、两路相机；柠檬 state/action 分别为 52/12 维、三路相机。物理维度、相机及频率不能相互套用。

## 数据、清单与统计

仅支持 LeRobot **v2.1**。默认目录是仓库内 `data/lemon_wrist_reset_v1`，可设置 `PI05_LEMON_DATASET_ROOT` 或在个人实验 `build_dataset()` 中修改。

- repo_id：`ur_o30/lemon_move_wrist_reset_20261008`。
- 当前来源：2026-10-08 新腕部视角，原集合 224 条；[清单](baseline/records/splits.json) 将原 188 train + 21 val 合为 **209 条训练数据**，另 15 条继续排除。清单中的 `train` 只是选取有效 episode，没有验证入口。
- 根目录可以保留完整 224 条数据；实际采样与统计只使用清单中的 209 条。修改数据路径不会重写数据，也不会自动修改 episode 清单。换一批数据或重新编号后，应替换清单，或在个人配置中设置 `TaskView()` 使用全部 episode。
- 统计默认保存在 `assets/lemon_wrist_reset_v1_10hz/`，包含 `norm_stats.json`、`norm_metadata.json`；`PI05_LEMON_ASSETS_DIR` 覆盖父目录。
- LoRA 和全参使用相同数据、样本与动作定义，可以共用本次重新计算的统计。**不能复用原 188 条训练集的统计**。
- 默认固定英文 prompt：`Scoop the lemon slice into the cup while the left hand supports it.`，通过 `prompt_overrides=((0, TASK),)` 显式映射 task 0。数据应使用对应任务 ID；多任务实验应提供完整映射或读取原始任务文本。

## 使用

在仓库根目录执行；全参将 MODULE 末段改为 `lemon_ur_o30_fullft_v1`：

```bash
export MODULE=training_configs.experiments.lemon_scooping.baseline.lemon_ur_o30_lora_v1
bash scripts/python.sh -m scripts.run_training "$MODULE" --dry-run
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE"
bash scripts/python.sh -m scripts.run_training "$MODULE" --check
bash scripts/python.sh -m scripts.run_training "$MODULE" --wandb-mode offline
```

联网时改为 `--wandb-mode online`。全参的 batch 64 是全局 batch；显卡数量和显存改变时在个人实验中显式调整，不能将其理解为任意一张卡都能训练。

## 可选阶段 subtask

默认两份 baseline 都是完整动作序列、单一 prompt，**不开启阶段截断或左右臂保持**。

包内提供[五时间戳标注](baseline/records/phase_annotations.json)，仅适用于这批 wrist_reset 数据。标注 JSON 的 `schema_version=3` 是阶段标注版本，**不是 LeRobot v3.0**。该标注保留原 episode/frame/时间戳，删除了旧机器视频绝对路径；原文件哈希见[来源记录](baseline/records/source_config.json)。

在本地 template 复制 baseline 后，可以替换 `config.data.base_config.task_view`；该个人文件不需要创建分支或上传：

```python
from openpi.training.task_view import TaskView
from training_configs.experiments.lemon_scooping.baseline.common import RECORDS, PHASE_PROMPTS

view = TaskView(
    split_path=str(RECORDS / "splits.json"),
    phase_annotations_path=str(RECORDS / "phase_annotations.json"),
    phase_prompts=PHASE_PROMPTS,
    phase_arm_hold=True,
)
```

将 view 放入 `TaskDataConfig(prompt_from_task=True, task_view=view)`，再通过 `dataclasses.replace` 替换个人配置的数据工厂。另设新配置名、实验名和统计 asset_id，并重新计算统计；完整修改方式见多人开发文档。

五个边界对应支撑就绪、铲取就绪、铲面承托、倒入就绪、任务完成。每帧按边界取得阶段 prompt，动作窗口超过当前阶段边界后重复端点动作，任务完成后的复位帧不作为训练样本。`phase_arm_hold=True` 另外把 state0 的右臂目标、state1–4 的左臂目标替换为当前状态；关闭它则只做边界终点保持。阶段功能不会自动生成时间戳，缺失或错位标注会报错。

## 来源与验证范围

来自 `pi05_lemon_source_bundle_20261013` 中的训练配置、任务快照和机器人协议。[来源记录](baseline/records/source_config.json) 保存文件哈希与原参数，便于核对；其中原 train/val 和验证策略只表示历史设置。

任务快照说明“如何配置”，不证明训练已经完成。当前迁移验证使用合成 v2.1 三相机视频数据、实际参数结构与 CPU 回归；生产数据、生产统计、GPU 训练及实机结果尚未在本仓库核验。源包中的旧模型部署记录不作为本次 baseline 的成功率证明。
