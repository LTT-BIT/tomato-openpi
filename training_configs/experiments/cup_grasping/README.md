# 抓取杯子训练说明

安装、日志与续训见 [仓库 README](../../../README.md)，个人配置和协作见 [多人开发 README](../../../多人开发README.md)。

## 本体与数据

单臂 Franka，原始 state/action 均为 7 维关节向量。先按 7 维计算归一化统计，再补零到模型的 32 维；沿用数据中的动作，不额外转 delta。主相机为 `observation.images.cam_head`，左腕相机为 `observation.images.wrist_left`，右腕图像为空并屏蔽。

定义来自提供的杯子任务 `config.py`，SHA-256 为 `3c7096905a694437c5ce020c76817a1c3fac5924e9c8a0bf179af05d3b0834a1`。源配置未提供完整关节顺序与单位，准备生产数据时须核对；其他训练参数复用公共 recipe：30 Hz、horizon 50、全局 batch：LoRA 8、全参 64；全参与 LoRA 均为 30,000 步。

本仓库仅接受 **LeRobot v2.1**，不兼容 v3.0，也不提供数据转换。现有杯子数据预计为 v3.0，须在独立数据仓库准备并核验 v2.1 数据，不能仅修改版本字段。检查帧数、episode 边界、关节顺序、任务文本和视频时间对齐。

两份 baseline 共用以下数据定义：

| 项目 | 默认值 |
| --- | --- |
| 数据根目录 | 仓库内 `data/cup_grasping_v1`；可用 `PI05_CUP_DATASET_ROOT` 覆盖 |
| repo_id | `single_franka/cup_grasping_v1` |
| 数据版本 / asset_id | `cup_grasping_v1_30hz` |
| 来源记录 | 默认记录数据根目录；实际采集及转换来源应在个人配置中补充 |
| 统计目录 | `assets/cup_grasping_v1_30hz`；`PI05_CUP_ASSETS_DIR` 覆盖父目录 |
| 统计文件 | `norm_stats.json` 和配套 `norm_metadata.json`，需自行生成 |
| 英文 prompt 示例 | `Pick up the cup.`；训练读取数据中实际 task，不用示例覆盖 |

## 选择配置并训练

| 方案 | baseline 模块末段 | 配置名称 | 默认实验名称 |
| --- | --- | --- | --- |
| LoRA | `cup_single_franka_lora_v1` | `pi05_single_franka_cup_lora_v1_30hz` | `cup_grasping_lora_v1_30hz_s42` |
| 全参 | `cup_single_franka_fullft_v1` | `pi05_single_franka_cup_fullft_v1_30hz` | `cup_grasping_fullft_v1_30hz_s42` |

在仓库根目录执行，切换全参时替换 MODULE 的末段：

```bash
export MODULE=training_configs.experiments.cup_grasping.baseline.cup_single_franka_lora_v1
bash scripts/python.sh -m scripts.run_training "$MODULE" --dry-run
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE"
bash scripts/python.sh -m scripts.run_training "$MODULE" --check
bash scripts/python.sh -m scripts.run_training "$MODULE" --wandb-mode offline
```

联网时使用 `--wandb-mode online`；离线训练后按根 README 同步 W&B。多人实验先复制到 `template/<负责人>/` 并指定唯一实验名称；当前 template 保持为空。

## 验证状态

配置和适配器已提供，包含合成 v2.1 双相机视频数据的统计与加载回归测试。生产数据、生产统计和预训练权重的准备尚未核验；未完成正式训练或实机验证。
