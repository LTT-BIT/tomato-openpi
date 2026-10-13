# Tomato OpenPI 训练仓库

基于 OpenPI 的 π0.5 机器人训练框架，支持 LoRA 和全参数微调。不同本体共用模型与训练循环，通过本体适配器和独立实验配置选择数据、动作维度及训练参数。标准训练入口使用 JAX。

## 1. 获取代码与安装环境

准备 Linux 训练机、Git、uv，以及支持当前 JAX CUDA 12 / PyTorch CUDA 12.8 依赖的 NVIDIA 驱动。安装脚本默认使用 Python 3.11；正式训练所需 GPU 数量和显存取决于方案、batch 和数据。

Ubuntu/Debian 新机器先安装基础工具和视频读取所需系统库，再安装与 CI 一致的 uv。NVIDIA 驱动按机器显卡安装，先确认 `nvidia-smi` 正常；训练依赖中的 CUDA 包由 uv 安装。安装 uv 后重新打开终端并确认 `uv --version` 可用。[uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/)

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates openssh-client ffmpeg libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.12.10/install.sh | sh
```

克隆前将本机 SSH 公钥加入 GitLab 账号并取得项目读取权限；可用 `ssh -T -p 2222 git@gitlab.litchibot.top` 检查连接。[GitLab SSH 设置](https://docs.gitlab.com/user/ssh/)

```bash
git clone --branch main \
  ssh://git@gitlab.litchibot.top:2222/fruit-train/harvesting/tomato-openpi.git
cd tomato-openpi

bash scripts/setup_training_env.sh
```

安装脚本按 `uv.lock` 安装依赖，并打印 JAX、PyTorch 和设备信息。环境使用仓库内 `.venv/`，缓存放在 `.cache/`；数据、统计、日志、checkpoint 和 W&B 分别放在仓库根目录的 `data/`、`assets/`、`logs/`、`checkpoints/`、`wandb/`。安装完成后直接运行下面的命令；启动入口自动设置仓库内的缓存和输出路径。

可以先展开参考配置，确认环境和入口可用，无需准备数据或权重：

```bash
.venv/bin/python -m scripts.run_training \
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
| `embodiment` | 本体规格和数据适配器；当前提供 FR3/Wuji 与 FR3/Sharpa |
| `build_config()` | 配置名、唯一 `exp_name`、负责人及实验目的 |
| `dataclasses.replace` 覆盖 | 学习率、batch、训练步数、seed 等本次实验参数 |

在个人实验文件中将默认数据目录设为 `repository / "data/alice_tomato_v1"`，统计父目录设为 `repository / "assets"`，并填写对应数据身份。数据和参数随实验文件一起保存。

数据身份应与实际内容一致：更换来源、筛选规则或变换时更新数据版本、`repo_id`、`asset_id` 和统计；同一数据仅搬迁目录时修改路径即可。个人实验的具体数据设置都在同一个文件中。

## 3. 准备数据、统计与初始权重

数据集和权重需要单独准备。数据转换、筛选与合并在独立的数据仓库完成，训练仓库接收已准备的数据。本体适配器决定输入格式；当前 FR3/Wuji 使用本地 LeRobot v2.1 数据，包含 `meta/`、`data/`、`videos/`，30 Hz、54 维 action、54 或 108 维原始 state，以及 cam0/cam1/cam2 图像。真实英文指令来自 `meta/tasks.jsonl`。本仓库保留数据校验及归一化统计工具。

先完成实验文件的数据设置，再计算新数据的归一化统计：

```bash
.venv/bin/python -m scripts.compute_experiment_norm \
  training_configs.experiments.alice.tomato_lora_lr1e5 --stats-batch-size 512
```

工具将 `norm_stats.json` 和 `norm_metadata.json` 成对发布到 `<assets_dir>/<asset_id>/`，同一目标的并发计算会报错，已有目标目录拒绝覆盖（包括空目录）；新统计使用新的 asset_id，不要预先创建其最终目录。计算或写入失败不会留下可被训练读取的半成品。训练前会核对统计文件、数据身份、元信息与 Parquet 文件摘要、动作维度、horizon 和输入变换；复制统计时保留这两个文件。相同数据搬迁目录或换学习率可复用，数据文件或输入变换变化后需重新计算。也可将统计文件 SHA-256 填入 Dataset 的 `expected_norm_sha256`，固定发布版本。

某个关节全程不变时，工具正常保存零方差统计，并在元数据中记录相应维度；归一化使用现有 epsilon 保护。当前统计入口支持 FR3/Wuji 与 FR3/Sharpa，其他本体需补充适配。旧统计缺少数据绑定记录时默认拒绝；已固定摘要的全参历史基准保留明确的兼容设置，详见训练使用说明。

FR3/Sharpa 使用独立 58 维 state/action 协议（7+22+7+22），cam2 左腕、cam1 右腕。LoRA 和全参 baseline 分别为 `training_configs.experiments.baselines.tomato_fr3_sharpa_lora_v1` 与 `training_configs.experiments.baselines.tomato_fr3_sharpa_fullft_v1`；个人模板为 `training_configs/experiments/template/tomato_sharpa_lora_v1.py` 与 `tomato_sharpa_fullft_v1.py`。训练参数复用现有公共方案，具体数据要求和命令见 [Sharpa 训练说明](docs/Sharpa训练说明.md)。

公共方案默认加载 `gs://openpi-assets/checkpoints/pi05_base/params`，首次使用需要下载并缓存初始权重；离线训练前须准备好权重及 tokenizer。数据格式、统计校验和权重设置详见 [训练使用说明](docs/训练使用说明.md)。

## 4. 检查与启动训练

```bash
# 展开最终配置，检查实验名、路径和训练参数。
.venv/bin/python -m scripts.run_training \
  training_configs.experiments.alice.tomato_lora_lr1e5 --dry-run

# 检查已准备的数据元信息和归一化统计。
.venv/bin/python -m scripts.run_training \
  training_configs.experiments.alice.tomato_lora_lr1e5 --check

# 首次在线使用 W&B 时登录。
.venv/bin/wandb login

# 按服务器分配选择 GPU；batch 必须能被可见设备数整除。
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.run_training \
  training_configs.experiments.alice.tomato_lora_lr1e5 --wandb-mode online
```

启动器只导入所选实验模块，调用它的 `build_config()`，然后执行输入检查和共享训练流程。可临时使用 `--batch-size 4`、`--num-train-steps 20000` 等 CLI 覆盖；正式实验将参数写入自己的配置并提交。

log_interval、save_interval 必须为正整数，keep_period 为 None 或正整数；非法间隔在配置加载时就报错。准备就绪后提交自己的实验配置，正式训练使用固定 commit，不在运行中修改或拉取代码。

没有网络时显式改用 `--wandb-mode offline`，恢复联网后同步：

```bash
.venv/bin/wandb sync \
  wandb/offline-run-实际编号
```

每次启动或续训保存一个独立 W&B 日志段，同一实验用固定 group 关联。offline 续训后需要逐段 sync；实际目录和恢复步数记录在本地运行清单及 checkpoint 的 wandb_sessions/ 中，云端显示为同一分组内的多个 run，不自动拼成一条历史曲线。

## 5. 查看结果与恢复训练

| 输出 | 默认位置 |
| --- | --- |
| 本地终端日志及运行清单 | `logs/<配置名>/<实验名>/` |
| checkpoint 与恢复审计 | `checkpoints/<配置名>/<实验名>/` |
| W&B 运行记录 | `wandb/` |
| JAX 编译缓存 | `.cache/jax/` |

表中路径均相对于仓库根目录。权重保存在每个 checkpoint 的 `<step>/params/`，完整恢复状态在 `<step>/train_state/`，对应统计在 `<step>/assets/`。

运行清单记录 Git commit、最终配置、数据来源、实际英文 prompt、统计摘要和运行结果。已有实验拒绝覆盖，重复训练使用新的 `exp_name`。

恢复未完成的同一实验时，使用原 commit、配置、数据、统计和完整 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.run_training \
  training_configs.experiments.alice.tomato_lora_lr1e5 --resume --wandb-mode online
```

离线恢复将模式改为 offline 即可；训练状态从 checkpoint 恢复，W&B 创建新的日志段。没有有效 checkpoint 时 --resume 报错；已完成原预算时直接退出。迁移续训须复制完整实验目录及旧日志段，保持相同源码和数据内容。

训练期间保持 checkout 不变；修改预算、数据或源码后建立新实验。恢复条件及离线日志同步见 [训练使用说明](docs/训练使用说明.md)。

## 目录与团队协作

| 路径 | 用途 |
| --- | --- |
| `training_configs/experiments/baselines/` | Wuji 与 Sharpa 各自的 LoRA v1、全参 v1 公共参考配置 |
| `training_configs/experiments/template/` | 可复制的个人实验模板 |
| `training_configs/experiments/<负责人>/` | 个人数据设置与实验参数 |
| `training_configs/recipes/` | 带版本的公共训练方案 |
| `training_configs/embodiments/` | 本体规格与数据适配 |
| `src/openpi/` | 模型、训练循环及共享数据变换 |
| `scripts/` | 环境、统计计算和统一训练入口 |
| `examples/fr3_wuji/` | FR3/Wuji 数据校验与归一化统计 |

个人实验通过短期分支和合并请求进入 `main`；公共 recipe 或训练功能变更单独评审。每位训练者使用独立 checkout、GPU 分配和仓库内输出目录，训练固定已提交的 commit。数据集、权重、日志及密钥不提交 Git。

完整配置示例、版本规则和协作流程见 [多人开发 README](docs/多人开发README.md)。

GitLab 自动验证目前暂停，推送分支或创建、更新合并请求不会创建新的流水线。检查脚本保留，需要时可在安装环境后手动执行 `bash scripts/check_training.sh`。恢复自动验证的方法与 Runner 要求见 [多人开发 README](docs/多人开发README.md)。

本项目基于 Physical Intelligence OpenPI，遵循 [LICENSE](LICENSE) 与 [Gemma 许可](LICENSE_GEMMA.txt)。
