# FR3 + Sharpa 训练说明

安装、数据校验、统计、日志与续训见 [仓库 README](../../../README.md)，个人配置和协作规则见 [多人开发 README](../../../多人开发README.md)。

## 本体来源与协议

本体协议来源：franka3-sharpa-pi05，commit `de2d187c17aaf64ee67a1501c9f7369944563729`。训练参数采用本仓库公共 recipe。

| 项目 | Sharpa v1 |
| --- | --- |
| state | 58 维实测关节位置，单位 rad |
| 原始 action | 58 维目标关节位置，单位 rad |
| 向量顺序 | 左 FR3 7 + 左 Sharpa 22 + 右 FR3 7 + 右 Sharpa 22 |
| 训练 action | 两臂和手部均为 absolute 目标位置 |
| 相机 | cam0 外部、cam2 左腕、cam1 右腕 |
| 数据与窗口 | LeRobot v2.1，30 Hz，horizon 50，episode 边界内填充 |

关节名称及顺序定义在 [fr3_sharpa_protocol.py](../../../src/openpi/policies/fr3_sharpa_protocol.py)，本体规格在 [fr3_sharpa.py](../../embodiments/fr3_sharpa.py)。向量已经是物理顺序，直接保留，不使用 Wuji 的 108 维切片或动作重排。原协议文件有 STATE_DIM=116 常量，但实际数据导出和输入变换均使用 58 维实测位置；本适配只接受实际导出的 58 维格式。

meta/info.json 的 observation.state 和 action 的 shape 均为 [58]，names 必须与协议 ORDER 一致；校验拒绝缺失名称或错序。名称校验不能证明实际数值单位、左右语义或时间对齐正确，需数据生产者确认。适配保留绝对目标值，不执行 delta 转换，也不修改原始目标位置。实际英文训练指令来自 meta/tasks.jsonl。

## Baseline

| 方案 | baseline 文件 | 默认预算 |
| --- | --- | --- |
| LoRA | [tomato_fr3_sharpa_lora_v1.py](baseline/tomato_fr3_sharpa_lora_v1.py) | 30,000 步 |
| 全参 | [tomato_fr3_sharpa_fullft_v1.py](baseline/tomato_fr3_sharpa_fullft_v1.py) | 30,000 步 |

模块前缀为 `training_configs.experiments.tomato_sharpa.baseline.`。个人实验可在自己的 clone 中从 baseline 复制到 `template/<负责人>/`；template 内容由 Git 忽略，不创建分支或上传。两种方案分别引用既有 `pi05_lora_v1` 和 `pi05_fullft_v1`。LoRA 为 VLM rank 16、Expert rank 32，视觉及投影可训练；全参无 LoRA、全部参数可训练。

除本体定义、58 维模型投影及独立数据/统计/输出身份外，训练参数与各自 Wuji baseline 一致：全局 batch：LoRA 8、全参 64；workers 8、bfloat16、学习率 peak 2.5e-5/end 2.5e-6、warmup 1000、decay 30000、AdamW、无 EMA、epoch_shuffle、seed 42/42/42001/42002、每 2000 步保存、keep_period 10000、W&B 启用。初始权重仍为官方 pi05_base，维度不匹配的投影按现有 PartialCheckpointWeightLoader 初始化。全参预算沿用公共方案，不表示当前 Sharpa 数据恰好训练一遍。

这两份 baseline 要求 30 Hz 数据。使用 20 Hz 或其他频率时，应建立对应个人配置并检查频率、horizon 与统计绑定。

| 设置 | 默认值 |
| --- | --- |
| 数据目录 | data/tomato_sharpa_fulltask_v1/ |
| 数据身份 | fr3_sharpa/tomato_fulltask_v1 |
| 数据版本 | tomato_sharpa_fulltask_v1_30hz |
| asset_id | tomato_sharpa_fulltask_v1_absolute_30hz |
| 统计目录 | assets/tomato_sharpa_fulltask_v1_absolute_30hz/ |
| LoRA 配置/实验名 | pi05_fr3_sharpa_fulltask_lora_v1_absolute_30hz / tomato_sharpa_fulltask_lora_v1_absolute_30hz_s42 |
| 全参配置/实验名 | pi05_fr3_sharpa_fulltask_fullft_v1_absolute_30hz / tomato_sharpa_fulltask_fullft_v1_absolute_30hz_s42 |

数据路径和身份为配置占位，需自行准备真实数据；sources 默认记录数据目录，正式实验填写真实生产来源。Sharpa 不使用 Wuji 54 维统计或历史固定摘要，默认要求绑定统计；同一 Sharpa 数据的 LoRA 与全参经校验后可以共用配套统计。

英文 prompt 示例分别与对应公共方案一致：

- LoRA：`Pick up a tomato truss with the right hand and hold it steady between both hands, then pick a single cherry tomato from the truss with the left hand and place the picked tomato into the plate on the left.`
- 全参：`Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.`

## 选择模块并训练

在仓库根目录设置模块，全参将尾名换为 `tomato_fr3_sharpa_fullft_v1`：

```bash
export MODULE=training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_lora_v1
export EXP=alice_tomato_sharpa_lora_s42_run01
bash scripts/python.sh -m scripts.run_training "$MODULE" --exp-name "$EXP" --dry-run
```

baseline 可用 `PI05_SHARPA_DATASET_ROOT` / `PI05_SHARPA_ASSETS_DIR` 覆盖路径。修改数据身份或参数时，按多人开发 README 复制本地个人配置；个人文件无需同步 GitLab。

新数据按根 README 计算统计，再执行 check 并训练。统计必须是本 Sharpa 数据的 58 维统计，包含 norm_stats.json 和 norm_metadata.json；同一数据、输入变换和 horizon 可在 LoRA/全参间复用。日志、W&B offline 同步和原 commit 续训规则见根 README。

以上 baseline 仍只接受 58D 实测位置格式，不能直接用于转换器的 116D 位置/速度格式。未记录完整 GPU 训练或实机验证结果。

当前两份 baseline 已切换为 absolute，统计身份与默认实验名增加 absolute。必须重新计算统计并开始新实验，不可复用旧 delta 统计或直接 resume 旧 delta checkpoint。

## 自动转换器的多模态数据：只训练 RGB

转换器保留 `observation.state[116]`（双臂、双手的位置/速度）、`action[58]`（左臂、右臂、左手、右手），以及三路 RGB、深度和二十路触觉。使用独立模块前缀 `training_configs.experiments.tomato_sharpa.multimodal_rgb.`，尾名仍为 `tomato_fr3_sharpa_lora_v1` 或 `tomato_fr3_sharpa_fullft_v1`。旧 baseline、旧 58D 协议和训练循环不变。

新适配按元数据严格核对原始关节名称/顺序，从 116D 状态中选取实测位置，并把状态与动作重排为“左臂 7 + 左手 22 + 右臂 7 + 右手 22”。模型仍使用 58D absolute 目标、30 Hz、horizon 50；相机仍为 cam0 外部、cam2 左腕、cam1 右腕。深度、触觉、速度和审计列保留在文件中，经过 repack 后不进入当前模型；底层 LeRobot loader 仍会读取额外列，因此这不是专门优化过的 RGB 列裁剪 loader。

用 `PI05_SHARPA_MULTIMODAL_DATASET_ROOT` 指定**一套已封存批次的实际数据集根目录**，不是 A/B 父目录；`PI05_SHARPA_MULTIMODAL_ASSETS_DIR` 指定统计父目录。新 asset_id 为 `tomato_sharpa_multimodal_rgb_absolute_v1_30hz`，必须重新计算与该固定批次、转换规则和 horizon 绑定的统计；不能复用旧 58D 或旧 delta 统计，也不能直接续训旧格式 checkpoint。英文任务文本继续来自该数据集的 `meta/tasks.jsonl`。

可独立做真实数据兼容性检查，不加载模型权重、不启动训练：

```bash
bash scripts/python.sh -m scripts.check_converter_dataset \
  --robot sharpa --dataset /path/to/36_68_data_lerobot \
  --norm-assets /path/to/new-validation-assets \
  --output /path/to/new-validation-report.json
```

该命令会计算独立归一化统计，核对 provenance，检查每个 episode 的首/中/尾向量映射，并实际生成一个含 RGB、位置和文本 token 的标准模型输入 batch。统计目录和报告应使用新路径，不覆盖既有训练资产；`--robot wuji` 使用现有 Wuji absolute 适配。这个检查通过表示数据/输入链路可用，不表示已经完成训练、效果评估或实机验证。可核对的真实检查结果位于本目录 `multimodal_rgb/records/`。
