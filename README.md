# Tomato OpenPI 训练仓库

基于 OpenPI 的 π0.5 训练仓库，使用统一 JAX 入口，支持 FR3/Wuji、FR3/Sharpa 的 LoRA 与全参数微调。数据转换在独立仓库完成，这里负责数据校验、归一化统计和训练。

本页说明下载安装和运行；创建个人配置、修改公共代码及分支管理见 [多人开发 README](多人开发README.md)。

## 1. 下载与安装

准备 Linux 训练机、NVIDIA 驱动和项目 SSH 读取权限。先确认 `nvidia-smi` 正常。依赖使用 Python 3.11、JAX CUDA 12、PyTorch CUDA 12.8；GPU 数量与显存按实际训练方案选择。

Ubuntu/Debian 安装基础工具、视频依赖和 uv：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates openssh-client ffmpeg libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.12.10/install.sh | sh
```

重新打开终端，确认 `uv --version` 可用。将本机 SSH 公钥加入 GitLab 账号后执行：

```bash
ssh -T -p 2222 git@gitlab.litchibot.top
git clone --branch main \
  ssh://git@gitlab.litchibot.top:2222/fruit-train/tomato-openpi.git
cd tomato-openpi
bash scripts/setup_training_env.sh
```

安装脚本按 `uv.lock` 创建 `.venv/`，打印 JAX 设备和 Torch CUDA 状态。正式训练前应看到 GPU 设备及 `CUDA=True`。以后在仓库根目录直接使用 `.venv/bin/python`，入口自动设置仓库内缓存和输出位置。

初始权重为 `gs://openpi-assets/checkpoints/pi05_base/params`；首次使用需要下载权重及 tokenizer，离线机器须提前准备缓存。允许动作/状态投影中形状不匹配的参数随机初始化，其余兼容参数加载基础权重。

## 2. 选择任务与 baseline

| 任务 | 说明和配置 | 当前状态 |
| --- | --- | --- |
| 番茄-Wuji | [tomato_wuji](training_configs/experiments/tomato_wuji/README.md) | LoRA / 全参 v1 baseline |
| 番茄-Sharpa | [tomato_sharpa](training_configs/experiments/tomato_sharpa/README.md) | LoRA / 全参 v1 baseline；生产数据待准备 |
| 铲柠檬 | [lemon_scooping](training_configs/experiments/lemon_scooping/README.md) | 本体、数据及 baseline 待定义 |
| 抓取杯子 | [cup_grasping](training_configs/experiments/cup_grasping/README.md) | 本体、数据及 baseline 待定义 |

每个任务下的 `baseline/` 存放参考配置，`template/` 当前为空，后续按负责人保存实验。数据、统计和真实训练结果不随 Git 仓库分发。

当前四份 baseline：

| 任务 / 方案 | baseline 文件 | 动作维度 | 总步数 | 默认数据目录 |
| --- | --- | ---: | ---: | --- |
| 番茄-Wuji LoRA | [tomato_fr3_lora_v1.py](training_configs/experiments/tomato_wuji/baseline/tomato_fr3_lora_v1.py) | 54 | 20,000 | `data/tomato_fulltask_v1` |
| 番茄-Wuji 全参 | [tomato_fr3_fullft_v1.py](training_configs/experiments/tomato_wuji/baseline/tomato_fr3_fullft_v1.py) | 54 | 60,153 | `data/tomato_all_full_v1` |
| 番茄-Sharpa LoRA | [tomato_fr3_sharpa_lora_v1.py](training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_lora_v1.py) | 58 | 20,000 | `data/tomato_sharpa_fulltask_v1` |
| 番茄-Sharpa 全参 | [tomato_fr3_sharpa_fullft_v1.py](training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_fullft_v1.py) | 58 | 60,153 | `data/tomato_sharpa_fulltask_v1` |

四份配置共用以下训练参数；本体协议、数据身份、统计及英文 prompt 见对应任务说明。

| 参数 | 默认值 |
| --- | --- |
| 模型 / 精度 | π0.5，bfloat16；max_token_len=256，discrete_state_input=True |
| LoRA | VLM `gemma_2b_lora` rank/alpha=16/16；Expert `gemma_300m_lora` rank/alpha=32/32；视觉及投影可训练 |
| 全参 | `gemma_2b` + `gemma_300m`，所有参数可训练 |
| 频率 / horizon | 30 Hz / 50；双臂 delta，手部 absolute |
| batch / workers / FSDP | 全局 batch 8 / workers 8 / fsdp_devices=1 |
| 采样 / seed | epoch_shuffle；seed=42、model_seed=42、data_seed=42001、augmentation_seed=42002 |
| 学习率 | cosine；warmup=1,000，peak=2.5e-5，decay_steps=30,000，end=2.5e-6 |
| 优化器 | AdamW；β1=0.9、β2=0.95、eps=1e-8、weight_decay=1e-10、clip=1.0 |
| EMA | 关闭 |
| 日志 / checkpoint | log_interval=10、save_interval=2,000、keep_period=10,000 |
| W&B | 启用，project=openpi；联网 online，离线显式 offline |

全参 60,153 步来自历史 Wuji 数据预算，并不会随数据量自动调整；LoRA 的 20,000 步预算也早于学习率计划的 30,000 步终点。baseline 是参考参数，复制后应按本次数据和训练计划调整。

以下以 Wuji LoRA 为例，选择模块并展开配置，无需数据、权重或 GPU：

```bash
export MODULE=training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1
export EXP=alice_tomato_wuji_lora_s42_run01
.venv/bin/python -m scripts.run_training "$MODULE" --exp-name "$EXP" --dry-run
```

更换方案时修改 MODULE。个人修改数据、学习率或预算时，按 [多人开发 README](多人开发README.md) 复制 baseline 为独立配置，再将 MODULE 指向个人模块。

## 3. 准备数据与统计

将数据仓库产出的 LeRobot v2.1 数据放入 `data/`，包含 `meta/`、`data/`、`videos/`。在所选实验的 `build_dataset()` 中确认 root、repo_id、来源、版本、assets_dir、asset_id 和英文 prompt 示例。`root` 是实际目录，`repo_id` 是数据身份；训练实际英文指令来自 `meta/tasks.jsonl`。

数据应满足任务的关节顺序、单位、相机映射和 30 Hz 频率。新数据使用新的版本和 asset_id；同一内容仅搬迁路径时保留身份。

新数据先计算统计：

```bash
.venv/bin/python -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 512
```

统计保存到 `<assets_dir>/<asset_id>/`，包含 `norm_stats.json` 和 `norm_metadata.json`。已有配套统计时跳过计算，迁移时两个文件一起复制。不要提前创建最终 asset_id 目录：工具拒绝覆盖已有目录，包括空目录；同一目标的重复计算进程也会被拒绝。

统计绑定数据文件、输入变换和 horizon。改变这些内容后重新计算；仅改学习率或 batch 可复用。常量关节允许 std=0，归一化使用 epsilon。Wuji 全参 baseline 固定了历史统计摘要并显式允许旧格式，若复制后换成新数据，应删除旧摘要和旧格式兼容设置，计算新统计。

## 4. 检查与训练

```bash
.venv/bin/python -m scripts.run_training "$MODULE" --exp-name "$EXP" --check
.venv/bin/wandb login
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.run_training \
  "$MODULE" --exp-name "$EXP" --wandb-mode online
```

`--check` 验证数据元信息、fps、维度、相机、英文 task、统计数值及数据/变换绑定，需要已有统计，不构建模型。正式启动还核对实际帧数并记录数据文件与采样审计。它们不能代替关节语义、单位及视频/动作时间对齐的验收。

GPU 编号按分配修改，全局 batch 必须能被可见设备数整除；单 GPU 命令仅为示例。可临时用 `--batch-size 4` 等 CLI 参数调试，正式实验将设置写入个人模块并提交，训练期间固定该 commit。

无网络时使用 `--wandb-mode offline`，本地日志和 W&B 仍保存。恢复联网后逐个同步离线段：

```bash
.venv/bin/wandb login
.venv/bin/wandb sync wandb/offline-run-实际编号
```

## 5. 输出与续训

所有默认目录均位于当前仓库：

| 内容 | 位置 |
| --- | --- |
| 环境 / 缓存 | `.venv/`、`.cache/`；JAX 编译缓存 `.cache/jax/` |
| 数据 / 统计 | `data/`、`assets/` |
| 本地日志及运行清单 | `logs/<配置名>/<实验名>/` |
| checkpoint | `checkpoints/<配置名>/<实验名>/<step>/` |
| W&B | `wandb/` |

每个 checkpoint 的 `params/` 保存权重，`train_state/` 保存优化器等恢复状态，`assets/` 保存配套统计。运行清单记录 commit、最终配置、数据来源、实际 prompt、统计摘要和运行结果。

恢复未完成实验时使用原 commit、模块、配置、数据、统计和实验名：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.run_training \
  "$MODULE" --exp-name "$EXP" --resume --wandb-mode online
```

迁移续训保留完整实验目录，包括 checkpoint、`reproducibility/`、`wandb_id.txt` 和 `wandb_sessions/`，以及未同步的 W&B 日志。只有 params 不足以精确续训。缺少有效 checkpoint 会报错；已完成原预算直接退出。改变代码、数据或预算后创建新实验，不能在新 main 上直接恢复旧运行。

每次启动或续训创建一个独立 W&B run，同一实验通过固定 group 关联；云端不会自动拼成一条曲线。各段的恢复步数和目录记录在 `wandb_sessions/`，离线段需逐个 sync。

历史训练参数、checkpoint 与指标见 [历史实验记录](docs/records/历史实验记录.md)。本仓库不包含这些历史运行产物。

GitLab 自动验证目前暂停。本地代码检查命令为 `bash scripts/check_training.sh`，协作与验证规则见 [多人开发 README](多人开发README.md)。本项目遵循 [LICENSE](LICENSE) 和 [Gemma 许可](LICENSE_GEMMA.txt)。
