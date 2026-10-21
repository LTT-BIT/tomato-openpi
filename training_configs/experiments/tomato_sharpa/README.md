# FR3 + Sharpa 番茄训练

环境、日志和续训见[根 README](../../../README.md)，个人配置见[多人开发 README](../../../多人开发README.md)。官方 LoRA 和全参 baseline 现在直接读取 `10.16_2_data_lerobot` 的转换器格式。UI 中选择“番茄 · Sharpa / LoRA”或“番茄 · Sharpa / 全参”即可使用，不再需要 tairan 临时配置。

## 数据和向量协议

原始 LeRobot v2.1 数据为 30 Hz，`observation.state` 是 116 维位置/速度，`action` 是 58 维绝对关节目标。必须严格匹配[转换器适配器](../../../src/openpi/policies/fr3_sharpa_multimodal_policy.py)中的原始名称和顺序；不能只修改 `info.json`。

| 向量 | 原始布局 | 读取时的 Python 切片拼接 |
| --- | --- | --- |
| state | 左臂位置/速度、右臂位置/速度、左手位置/速度、右手位置/速度 | `[0:7] + [28:50] + [14:21] + [72:94]` |
| action | 左臂、右臂、左手、右手 | `[0:7] + [14:36] + [7:14] + [36:58]` |

模型输入和输出统一为左臂 7 + 左手 22 + 右臂 7 + 右手 22，共 58 维，单位 rad；状态只保留实测位置，action 默认保持 absolute 目标。相机使用 cam0 外部、cam2 左腕、cam1 右腕 RGB。深度、触觉、速度及审计列保留在文件中，不进入当前模型；底层 loader 仍可能读取这些列。名称检查不能证明数值单位、左右语义或时间对齐正确，需要数据生产者确认。

旧 `Fr3Sharpa` 58D 实测位置适配器仍保留，其严格检查不变；这两份官方 baseline 使用 `FR3_SHARPA_MULTIMODAL`。训练入口统一使用 `baseline/` 下的全参与 LoRA 两份配置，UI 和命令行使用相同入口。

## 官方配置和数据身份

| 项目 | 设置 |
| --- | --- |
| 数据来源 | `sshfs://user@192.168.10.202/data/data-fruits/real/harvest-tomato/data/clean-rosbag/sharpa/10.16_2_data_lerobot` |
| 默认训练目录 | `/mnt/data/users/tairan/workspace/data/10.16_2_data_lerobot` |
| repo_id | `fr3_sharpa/tomato_1016_2_multimodal_rgb_v1` |
| 数据版本 / asset_id | `tomato_sharpa_1016_2_multimodal_rgb_absolute_v1_30hz` |
| 统计目录 | `<当前仓库>/assets/tomato_sharpa_1016_2_multimodal_rgb_absolute_v1_30hz/` |
| 全参配置名 | `pi05_fr3_sharpa_1016_2_multimodal_rgb_fullft_v1_absolute_30hz` |
| 全参默认实验名 | `tomato_sharpa_1016_2_multimodal_fullft_absolute_s42` |
| LoRA 配置名 | `pi05_fr3_sharpa_1016_2_multimodal_rgb_lora_v1_absolute_30hz` |
| LoRA 默认实验名 | `tomato_sharpa_1016_2_multimodal_lora_absolute_s42` |

路径和身份由[共享数据定义](baseline/dataset.py)提供。优先使用 `PI05_SHARPA_MULTIMODAL_DATASET_ROOT` / `PI05_SHARPA_MULTIMODAL_ASSETS_DIR`，兼容旧 `PI05_SHARPA_DATASET_ROOT` / `PI05_SHARPA_ASSETS_DIR`。只搬迁相同数据可以保留身份；更换批次必须设置独立版本、来源和统计身份。

两种方案采用同一个英文 prompt 示例，实际训练文本来自 `meta/tasks.jsonl`：

> Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.

## 训练参数

| 方案 | 文件 | 全局 batch | FSDP 分片 | 步数 |
| --- | --- | --- | --- | --- |
| 全参 | [tomato_fr3_sharpa_fullft_v1.py](baseline/tomato_fr3_sharpa_fullft_v1.py) | 32 | 8 | 30,000 |
| LoRA | [tomato_fr3_sharpa_lora_v1.py](baseline/tomato_fr3_sharpa_lora_v1.py) | 8 | 1 | 30,000 |

全参默认按 8 张 GPU 分片，减少参数、梯度及优化器状态的每卡显存占用；选择较少 GPU 时必须覆盖 FSDP 参数，GPU 数量需能被分片数整除，全局 batch 需能被 GPU 数量整除。LoRA 保留公共配方的 batch 和 FSDP 设置，不强制使用 8 张卡。

其余沿用各自公共 recipe：pi05、bfloat16、58D 模型、horizon 50、workers 8、epoch_shuffle、seed 42/42/42001/42002、warmup 1000、peak 2.5e-5/end 2.5e-6、AdamW、无 EMA，每 2000 步保存。LoRA 使用 VLM rank 16、Expert rank 32，视觉及投影可训练；全参不使用 LoRA，全部参数可训练。基础权重为 `/mnt/data/public_ckpt/pi05/pi05_base/params`，投影形状不匹配时按公共 PartialCheckpointWeightLoader 初始化。

## 统计、启动和状态

```bash
export MODULE=training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_fullft_v1
export EXP=tairan_sharpa_1016_fullft_fsdp8_b32_s42_run01
bash scripts/python.sh -m scripts.run_training "$MODULE" --exp-name "$EXP" --dry-run
# 仅在没有配套统计时计算；工具拒绝覆盖已有目录。
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
bash scripts/python.sh -m scripts.run_training "$MODULE" --exp-name "$EXP" --check
# 只选择分配给自己的 GPU；示例默认 8 卡。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name "$EXP" --wandb-mode online
```

LoRA 将 MODULE 尾名换为 `tomato_fr3_sharpa_lora_v1`，按分配选择 GPU。W&B online 需预先登录，不能联网时显式使用 offline，恢复联网后 sync；本地日志和运行清单保存在 `logs/`，权重保存在 `checkpoints/`。

统计必须同时包含 `norm_stats.json` 和 `norm_metadata.json`，并通过数据、变换及 horizon 的绑定校验。LoRA 与全参可在绑定一致时共用统计；batch 和 FSDP 的改变不改变统计。与此前临时配置相同的批次和适配规则可以校验复用已有统计；旧 58D 或 delta 统计不能直接使用。UI 切换 delta 时必须选新 asset_id 并重新统计。

新 baseline 已准备输入适配；这不代表已经完成完整训练、效果评估或实机验证。`records/` 中的真实检查结果属于其中记录的历史批次，报告中的旧模块名保留用于追溯，可通过 Git 历史查看当时配置；它们不是本批次完整训练完成的证明。服务器既有临时配置、运行中的任务及其代码未随本地更新修改。续训必须使用原代码、配置、数据、统计及实验名；应用本地新版本后请创建新实验。

数据兼容检查使用当前 baseline，计算独立归一化统计并读取一个 batch，不加载模型权重、不启动训练：

```bash
bash scripts/python.sh -m scripts.check_converter_dataset \
  --robot sharpa --dataset /path/to/prepared_lerobot_dataset \
  --norm-assets /path/to/new-validation-assets \
  --output /path/to/new-validation-report.json
```

统计和报告使用新路径；检查其他批次不代表默认 10.16_2 批次已通过验证。
