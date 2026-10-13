# Tomato OpenPI 训练仓库

基于 OpenPI 的 π0.5 机器人训练框架，支持 LoRA 和全参数微调。不同本体共用模型与训练循环，通过本体适配器和独立实验配置选择数据、动作维度及训练参数。标准训练入口使用 JAX。

## 1. 获取代码与安装环境

准备 Linux 训练机、Git、uv，以及支持当前 JAX CUDA 12 / PyTorch CUDA 12.8 依赖的 NVIDIA 驱动。安装脚本默认使用 Python 3.11；正式训练所需 GPU 数量和显存取决于方案、batch 和数据。

```bash
git clone --branch main \
  ssh://git@gitlab.litchibot.top:2222/fruit-train/harvesting/tomato-openpi.git
cd tomato-openpi

bash scripts/setup_training_env.sh
source scripts/training_env.sh
```

安装脚本按 `uv.lock` 安装依赖，并打印 JAX、PyTorch 和设备信息。环境使用仓库内 `.venv/`，缓存放在 `.cache/`；数据、统计、日志、checkpoint 和 W&B 分别放在仓库根目录的 `data/`、`assets/`、`logs/`、`checkpoints/`、`wandb/`。新终端进入仓库后执行 `source scripts/training_env.sh`。

可以先展开参考配置，确认环境和入口可用，无需准备数据或权重：

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.run_training \
  training_configs.experiments.baselines.tomato_fr3_lora_v1 --dry-run
```

## 2. 新建自己的实验

每次实验使用独立配置文件和实验名。以下用 `alice` 作示例，替换成自己的负责人名称：

```bash
git switch -c exp/alice-tomato-lora
mkdir -p training_configs/experiments/alice
touch training_configs/experiments/alice/__init__.py
cp training_configs/experiments/template/tomato_lora_lr1e5.py \
  training_configs/experiments/alice/tomato_lora_lr1e5.py
```

编辑复制后的实验文件：

| 修改位置 | 设置内容 |
| --- | --- |
| `build_dataset()` | 数据版本、`repo_id`、实际数据路径、原始来源、统计目录和 `asset_id` |
| `PROMPT` | 本次任务的英文指令示例；训练实际读取数据中的 task |
| recipe 导入 | `pi05_lora_v1` 或 `pi05_fullft_v1` |
| `embodiment` | 本体规格和数据适配器；当前提供 FR3/Wuji |
| `build_config()` | 配置名、唯一 `exp_name`、负责人及实验目的 |
| `dataclasses.replace` 覆盖 | 学习率、batch、训练步数、seed 等本次实验参数 |

模板支持用环境变量覆盖本机数据和统计位置：

```bash
export PI05_DATASET_ROOT="$PWD/data/alice_tomato_v1"
export PI05_ASSETS_DIR="$PWD/assets"
export EXPERIMENT=training_configs.experiments.alice.tomato_lora_lr1e5
```

数据身份应与实际内容一致：更换来源、筛选规则或变换时更新数据版本、`repo_id`、`asset_id` 和统计；同一数据仅搬迁目录时修改路径即可。个人实验的具体数据设置都在同一个文件中。

## 3. 准备数据、统计与初始权重

数据集和权重需要单独准备。本体适配器决定输入格式；当前 FR3/Wuji 使用本地 LeRobot v2.1 数据，包含 `meta/`、`data/`、`videos/`，30 Hz、54 维 action、54 或 108 维原始 state，以及 cam0/cam1/cam2 图像。真实英文指令来自 `meta/tasks.jsonl`。转换、验证和合并工具位于 `examples/fr3_wuji/`。

先完成实验文件的数据设置，再计算新数据的归一化统计：

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.compute_experiment_norm \
  "$EXPERIMENT" --stats-batch-size 512
```

统计保存在 `<assets_dir>/<asset_id>/norm_stats.json`，已有统计拒绝覆盖。将文件 SHA-256 填入实验 Dataset 的 `expected_norm_sha256`；复用统计时确认对应的数据、动作表示和 horizon 一致。当前统计入口支持 FR3/Wuji，其他本体需补充适配。

公共方案默认加载 `gs://openpi-assets/checkpoints/pi05_base/params`，首次使用需要下载并缓存初始权重；离线训练前须准备好权重及 tokenizer。数据格式、统计校验和权重设置详见 [训练使用说明](docs/训练使用说明.md)。

## 4. 检查与启动训练

```bash
# 展开最终配置，检查实验名、路径和训练参数。
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.run_training \
  "$EXPERIMENT" --dry-run

# 检查已准备的数据元信息和归一化统计。
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.run_training \
  "$EXPERIMENT" --check

# 首次在线使用 W&B 时登录。
"$UV_PROJECT_ENVIRONMENT/bin/wandb" login

# 按服务器分配选择 GPU；batch 必须能被可见设备数整除。
CUDA_VISIBLE_DEVICES=0 "$UV_PROJECT_ENVIRONMENT/bin/python" -u -m scripts.run_training \
  "$EXPERIMENT" --wandb-mode online
```

启动器只导入所选实验模块，调用它的 `build_config()`，然后执行输入检查和共享训练流程。可临时使用 `--batch-size 4`、`--num-train-steps 20000` 等 CLI 覆盖；正式实验将参数写入自己的配置并提交。

没有网络时显式改用 `--wandb-mode offline`，恢复联网后同步：

```bash
"$UV_PROJECT_ENVIRONMENT/bin/wandb" sync \
  "$PI05_REPOSITORY_DIR/wandb/offline-run-实际编号"
```

## 5. 查看结果与恢复训练

| 输出 | 默认位置 |
| --- | --- |
| 本地终端日志及运行清单 | `logs/<配置名>/<实验名>/` |
| checkpoint 与恢复审计 | `checkpoints/<配置名>/<实验名>/` |
| W&B 运行记录 | `wandb/` |

表中路径均相对于仓库根目录。权重保存在每个 checkpoint 的 `<step>/params/`，完整恢复状态在 `<step>/train_state/`，对应统计在 `<step>/assets/`。

运行清单记录 Git commit、最终配置、数据来源、实际英文 prompt、统计摘要和运行结果。已有实验拒绝覆盖，重复训练使用新的 `exp_name`。

恢复未完成的同一实验时，使用原 commit、配置、数据、统计和完整 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 "$UV_PROJECT_ENVIRONMENT/bin/python" -u -m scripts.run_training \
  "$EXPERIMENT" --resume --wandb-mode online
```

训练期间保持 checkout 不变；修改预算、数据或源码后建立新实验。恢复条件及离线日志同步见 [训练使用说明](docs/训练使用说明.md)。

## 目录与团队协作

| 路径 | 用途 |
| --- | --- |
| `training_configs/experiments/baselines/` | LoRA v1、全参 v1 公共参考配置 |
| `training_configs/experiments/template/` | 可复制的个人实验模板 |
| `training_configs/experiments/<负责人>/` | 个人数据设置与实验参数 |
| `training_configs/recipes/` | 带版本的公共训练方案 |
| `training_configs/embodiments/` | 本体规格与数据适配 |
| `src/openpi/` | 模型、训练循环及共享数据变换 |
| `scripts/` | 环境、统计计算和统一训练入口 |
| `examples/fr3_wuji/` | FR3/Wuji 训练数据工具 |

个人实验通过短期分支和合并请求进入 `main`；公共 recipe 或训练功能变更单独评审。每位训练者使用独立 checkout、GPU 分配和仓库内输出目录，训练固定已提交的 commit。数据集、权重、日志及密钥不提交 Git。

完整配置示例、版本规则和协作流程见 [多人开发 README](docs/多人开发README.md)。

本项目基于 Physical Intelligence OpenPI，遵循 [LICENSE](LICENSE) 与 [Gemma 许可](LICENSE_GEMMA.txt)。
