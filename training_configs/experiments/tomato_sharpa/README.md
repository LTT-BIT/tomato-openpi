# FR3 + Sharpa 训练说明

本页使用统一 JAX 训练入口。首次安装从 [根 README](../../../README.md) 开始，日志与恢复使用 [训练使用说明](../../../docs/训练使用说明.md)，协作规则见 [多人开发 README](../../../docs/多人开发README.md)。

## 本体来源与协议

训练适配来自 franka3-sharpa-pi05，阅读时提交为 de2d187c17aaf64ee67a1501c9f7369944563729。源仓库实现双 FR3 + 双 Sharpa 的关节协议、ROS bag 对齐与 LeRobot 导出、58 维 LoRA 配置、离线评估及真机运行工程。此处接入训练协议、输入适配和相机映射，采用本仓库公共训练参数；数据转换仍在独立数据仓库完成。

| 项目 | Sharpa v1 |
| --- | --- |
| state | 58 维实测关节位置，单位 rad |
| 原始 action | 58 维目标关节位置，单位 rad |
| 向量顺序 | 左 FR3 7 + 左 Sharpa 22 + 右 FR3 7 + 右 Sharpa 22 |
| 训练 action | 两臂相对当前实测 state 的 delta，手部 absolute |
| 相机 | cam0 外部、cam2 左腕、cam1 右腕 |
| 数据与窗口 | LeRobot v2.1，30 Hz，horizon 50，episode 边界内填充 |

关节名称及顺序定义在 [fr3_sharpa_protocol.py](../../../src/openpi/policies/fr3_sharpa_protocol.py)，本体规格在 [fr3_sharpa.py](../../embodiments/fr3_sharpa.py)。向量已经是物理顺序，直接保留，不使用 Wuji 的 108 维切片或动作重排。原协议文件有 STATE_DIM=116 常量，但实际数据导出和输入变换均使用 58 维实测位置；本适配只接受实际导出的 58 维格式。

meta/info.json 的 observation.state 和 action 的 shape 均为 [58]，names 必须与协议 ORDER 一致；校验拒绝缺失名称或错序。名称校验不能证明实际数值单位、左右语义或时间对齐正确，需数据生产者确认。适配复制向量后做 delta，避免修改原始目标位置。实际英文训练指令来自 meta/tasks.jsonl。

## Baseline

| 方案 | baseline 文件 | 默认预算 |
| --- | --- | --- |
| LoRA | [tomato_fr3_sharpa_lora_v1.py](baseline/tomato_fr3_sharpa_lora_v1.py) | 20,000 步 |
| 全参 | [tomato_fr3_sharpa_fullft_v1.py](baseline/tomato_fr3_sharpa_fullft_v1.py) | 60,153 步 |

模块前缀为 `training_configs.experiments.tomato_sharpa.baseline.`。当前 `template/` 留空，只保留 `.gitkeep`；后续实验从 baseline 复制到 `template/<负责人>/`。两种方案分别引用既有 `pi05_lora_v1` 和 `pi05_fullft_v1`。LoRA 为 VLM rank 16、Expert rank 32，视觉及投影可训练；全参无 LoRA、全部参数可训练。

除本体定义、58 维模型投影及独立数据/统计/输出身份外，训练参数与各自 Wuji baseline 一致：batch 8、workers 8、bfloat16、学习率 peak 2.5e-5/end 2.5e-6、warmup 1000、decay 30000、AdamW、无 EMA、epoch_shuffle、seed 42/42/42001/42002、每 2000 步保存、keep_period 10000、W&B 启用。初始权重仍为官方 pi05_base，维度不匹配的投影按现有 PartialCheckpointWeightLoader 初始化。全参预算沿用公共方案，不表示当前 Sharpa 数据恰好训练一遍。

源仓库的本机路径、关闭 W&B、20 Hz Tier A/B1 混采及其他预算未迁入。20 Hz 数据不能直接用于这两份 30 Hz baseline；变更频率需在个人实验明确记录，重新检查统计与 horizon。源仓库的验证记录不代表这里已完成 GPU 训练或实机验证。

| 设置 | 默认值 |
| --- | --- |
| 数据目录 | data/tomato_sharpa_fulltask_v1/ |
| 数据身份 | fr3_sharpa/tomato_fulltask_v1 |
| 数据版本及 asset_id | tomato_sharpa_fulltask_v1_30hz |
| 统计目录 | assets/tomato_sharpa_fulltask_v1_30hz/ |
| LoRA 配置/实验名 | pi05_fr3_sharpa_fulltask_lora_v1_30hz / tomato_sharpa_fulltask_lora_v1_30hz_s42 |
| 全参配置/实验名 | pi05_fr3_sharpa_fulltask_fullft_v1_30hz / tomato_sharpa_fulltask_fullft_v1_30hz_s42 |

数据路径和身份为配置占位，需自行准备真实数据；sources 默认记录数据目录，正式实验填写真实生产来源。Sharpa 不使用 Wuji 54 维统计或历史固定摘要，默认要求绑定统计；同一 Sharpa 数据的 LoRA 与全参经校验后可以共用配套统计。

英文 prompt 示例分别与对应公共方案一致：

- LoRA：`Pick up a tomato truss with the right hand and hold it steady between both hands, then pick a single cherry tomato from the truss with the left hand and place the picked tomato into the plate on the left.`
- 全参：`Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.`

## 检查 baseline

下面的展开命令不要求数据、统计、权重或 GPU；全参将模块尾名换为 `tomato_fr3_sharpa_fullft_v1`。

```bash
.venv/bin/python -m scripts.run_training \
  training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_lora_v1 --dry-run
```

## 创建与运行个人实验

新增实验时从对应 baseline 复制，全参复制 `tomato_fr3_sharpa_fullft_v1.py`。复制后的文件包含自己的数据设置，直接引用公共 recipe 和本体，不导入 baseline。

```bash
git switch -c exp/alice-tomato-sharpa-lora
mkdir -p training_configs/experiments/tomato_sharpa/template/alice
touch training_configs/experiments/tomato_sharpa/template/__init__.py \
  training_configs/experiments/tomato_sharpa/template/alice/__init__.py
cp training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_lora_v1.py \
  training_configs/experiments/tomato_sharpa/template/alice/tomato_sharpa_lora_v1.py
```

编辑个人文件的 `build_dataset()`：root、sources、版本、repo_id、assets_dir、asset_id、prompt；在 `build_config()` 修改配置名和唯一 exp_name。需要覆盖参数或记录 owner / purpose 时，将 `return build_recipe(...)` 改为 `config = build_recipe(...)`，再用 `dataclasses.replace` 返回个人配置，完整示例见多人开发 README。相同数据沿用真实身份，新数据建立新版本和统计身份。

baseline 使用可选的 `PI05_SHARPA_DATASET_ROOT` / `PI05_SHARPA_ASSETS_DIR`，复制后可直接设置自己的仓库内路径，也可改为个人变量名；不要用共享环境变量改变他人正在运行的配置。

新数据先计算统计；已有对应的 `norm_stats.json` / `norm_metadata.json` 时跳过计算，直接 `--check`。

```bash
.venv/bin/python -m scripts.run_training \
  training_configs.experiments.tomato_sharpa.template.alice.tomato_sharpa_lora_v1 --dry-run
.venv/bin/python -m scripts.compute_experiment_norm \
  training_configs.experiments.tomato_sharpa.template.alice.tomato_sharpa_lora_v1 --stats-batch-size 512
.venv/bin/python -m scripts.run_training \
  training_configs.experiments.tomato_sharpa.template.alice.tomato_sharpa_lora_v1 --check
```

统计复用 `examples/fr3_wuji/compute_norm_stats.py` 中已支持两种本体的共享实现。统计和元数据按目标锁成对原子发布，不预先创建最终 asset_id 目录。metadata 绑定数据、horizon、实际输入变换及关节顺序；迁移统计时保留两个文件。

提交并固定配置后启动，GPU 数量与 batch 按真实显存调整：

```bash
git add training_configs/experiments/tomato_sharpa/template
git commit -m "添加个人Sharpa训练配置"
.venv/bin/wandb login
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.run_training \
  training_configs.experiments.tomato_sharpa.template.alice.tomato_sharpa_lora_v1 --wandb-mode online
```

离线改用 `--wandb-mode offline`，提前准备权重和 tokenizer；输出位置、逐段 sync 和原 commit 续训规则与其他本体相同。移动前启动的实验按原 commit 和原模块路径续训。GitLab 自动验证保持暂停。当前完成配置与 CPU 验证，真实数据准备、完整 GPU 训练及实机验证尚未完成。
