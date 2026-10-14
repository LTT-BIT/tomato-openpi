# Tomato OpenPI 训练仓库

基于 OpenPI 的 π0.5 训练仓库，使用统一 JAX 入口，支持 FR3/Wuji、FR3/Sharpa、单臂 Franka、双 UR/O30 的 LoRA 与全参数微调。数据转换在独立仓库完成，这里负责数据校验、归一化统计和训练。

本页说明下载安装和运行；本地个人配置、公共 baseline 变更和多人协作见 [多人开发 README](多人开发README.md)。

多人在同一服务器训练时，各自在自己的目录独立 clone。日常训练直接使用公共 baseline，或在本地 `template/<负责人>/` 保存个人配置；个人配置已被 Git 忽略，不创建分支、不提交、不上传。只有团队决定新增或修改公共 baseline 时才进入公共开发流程。目录和配置类的组合关系见[开发结构说明](多人开发README.md#目录与配置的组合关系)。

## 500028 服务器：复用公共环境

服务器环境已经准备好，直接使用以下路径和命令。账号需属于 `ai_researchers` 组。

| 内容 | 路径 |
| --- | --- |
| 公共环境 | `/mnt/data/public_tools/training/environments/pi05/20261011` |
| 基础权重 | `/mnt/data/public_ckpt/pi05/pi05_base/params` |
| 仓库启动入口 | `bash scripts/python.sh` |

**下载代码**（已有仓库直接进入自己的仓库目录）：

```bash
git clone --branch main \
  ssh://git@gitlab.litchibot.top:2222/fruit-train/tomato-openpi.git
cd tomato-openpi
```

**启动图形界面**，在自己的仓库根目录执行：

```bash
export PI05_SHARED_ENV=/mnt/data/public_tools/training/environments/pi05/20261011
# 联网记录 W&B 时首次登录；选择 offline 时不需要登录。
bash scripts/python.sh -m wandb login
bash scripts/python.sh -m scripts.training_ui --port 8765
```

在 VS Code 的“端口”面板转发 `8765`，打开终端打印的完整地址（包含 `#token=...`）。如果本地转发端口不同，替换地址中的端口号，保留 token。也可以使用 SSH 隧道 `ssh -L 8765:127.0.0.1:8765 zhiwei500028`。服务只监听服务器本机地址，每位使用者从自己的 clone 启动；同一服务器端口被占用时换一个端口。

服务器无需重新安装 Python 环境或前端依赖。在界面中点击“使用服务器公共权重”，会填写 `/mnt/data/public_ckpt/pi05/pi05_base/params`。tokenizer 首次使用仍需下载或提前准备仓库缓存。

## 图形界面：配置和启动训练

自己的电脑完成下方环境安装后，同样执行 `bash scripts/python.sh -m scripts.training_ui --port 8765`，打开终端打印的地址。

1. **选择 baseline**：八份公共方案自动读取默认参数，显示本体维度、频率和英文 prompt 示例。柠檬保留原有 episode 筛选及保存时间表。
2. **选择数据**：输入或浏览训练服务器上的数据目录，确认数据版本、repo_id、来源和统计位置。浏览器不上传本机数据。更换数据路径或 horizon 时，界面会为新实验选择独立统计名称；也可手动选择已有的匹配统计。
3. **修改参数**：可修改 batch、步数、学习率计划、workers、FSDP、随机种子、保存间隔、基础权重和 W&B 项目。支持 A/B 权重及 episode 分组有放回采样。本体、动作语义、模型结构和任务视图沿用所选 baseline。
4. **选择 GPU 和 W&B 模式**：界面读取 `nvidia-smi` 的显存和利用率，可多选。全局 batch 必须能被卡数整除；卡数必须能被 FSDP 分片数整除。online 需要预先在终端登录，offline 明确保留本地记录，之后按下文同步。
5. **保存配置并启动**：自动保存个人 Python 文件，后台依次校验数据、缺少时计算统计、核对统计绑定并启动训练。可先点击“检查数据与统计”或“查看最终配置”。任何阶段失败都会显示错误和日志，不继续下一阶段。

个人配置保存到所选任务的 `template/<用户名>/<实验名>.py`，与 `baseline/` 同级；已被 Git 忽略，不用创建分支或上传。文件内 `_UI_VALUES` 是实际参数，保留 baseline 的组合代码及来源摘要。实验名和用户名使用英文字母、数字、下划线，不能以数字开头；已有配置不覆盖，另取实验名。

关闭网页或退出 UI 服务不会停止后台任务；任务记录在重开界面后仍可读取。界面可停止任务、按原配置重试或续训。停止不保证额外保存一次权重，只能从已有 checkpoint 恢复；还未产生 checkpoint 但已创建实验目录时，请换实验名。续训不会应用当前表单修改，配置被改动时会拒绝续训。新增其他个人实验文件不影响本次续训；公共源码更新仍会触发严格校验，旧版本训练需保留原 checkout。共享磁盘上的任务需要回到原服务器 UI 操作。

UI 流程日志和状态在 `logs/training_ui/<任务ID>/`；正式训练仍按现有规则写入 `logs/`、`checkpoints/`、`wandb/`，不会写入公共环境。GPU 占用检查用于提示，本仓库 UI 会预留自己任务使用的卡；其他 clone、其他用户及命令行训练仍需协调。运行期间不要修改正在使用的源码或配置。

UI 只管理本仓库训练，不安装驱动或环境、不转换数据、不修改 W&B 登录凭据。除首次准备环境、登录 W&B 和启动 UI 外，日常数据准备检查、统计计算、参数选择及训练操作都可在页面完成。下面的命令行流程仍可使用。

## 自己的电脑：安装本地环境


适用于 Linux + NVIDIA GPU 训练机。先确认 `nvidia-smi` 正常，并将本机 SSH 公钥加入 GitLab 账号。环境使用 Python 3.11、JAX CUDA 12 和 PyTorch CUDA 12.8，驱动需支持对应版本。

Ubuntu/Debian 安装基础工具和视频依赖：

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates openssh-client ffmpeg libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/0.12.10/install.sh | sh
```

重新打开终端，确认 `uv --version` 可用，然后下载仓库、安装环境：

```bash
git clone --branch main \
  ssh://git@gitlab.litchibot.top:2222/fruit-train/tomato-openpi.git
cd tomato-openpi
unset PI05_SHARED_ENV
bash scripts/setup_training_env.sh --local
```

安装脚本按 `uv.lock` 在当前仓库创建 `.venv/`，并打印 JAX 设备和 Torch CUDA 状态。正式训练前应看到 GPU 设备及 `CUDA=True`。以后使用 `bash scripts/python.sh` 自动读取本地 `.venv`。

默认基础权重为 `gs://openpi-assets/checkpoints/pi05_base/params`，首次使用会下载权重和 tokenizer；离线训练前需准备好缓存。

## 任务与 baseline

| 任务 | 说明和配置 | 当前状态 |
| --- | --- | --- |
| 番茄-Wuji | [tomato_wuji](training_configs/experiments/tomato_wuji/README.md) | LoRA / 全参 v1 baseline |
| 番茄-Sharpa | [tomato_sharpa](training_configs/experiments/tomato_sharpa/README.md) | LoRA / 全参 v1 baseline；生产数据待准备 |
| 铲柠檬 | [lemon_scooping](training_configs/experiments/lemon_scooping/README.md) | 双 UR/O30 LoRA / 全参 v1 baseline；209 条有效数据全部训练 |
| 抓取杯子 | [cup_grasping](training_configs/experiments/cup_grasping/README.md) | 单臂 Franka LoRA / 全参 v1 baseline；v2.1 数据待准备 |

每个任务下的 `baseline/` 存放团队发布的参考配置，`template/` 供每个人在自己的机器上保存个人实验配置。`template/` 内容由 `.gitignore` 排除，不上传 GitLab；数据、统计和训练结果也不随仓库分发。

当前八份 baseline：

| 任务 / 方案 | baseline 文件 | 动作维度 | 总步数 | 默认数据目录 |
| --- | --- | ---: | ---: | --- |
| 番茄-Wuji LoRA | [tomato_fr3_lora_v1.py](training_configs/experiments/tomato_wuji/baseline/tomato_fr3_lora_v1.py) | 54 | 30,000 | `data/tomato_fulltask_v1` |
| 番茄-Wuji 全参 | [tomato_fr3_fullft_v1.py](training_configs/experiments/tomato_wuji/baseline/tomato_fr3_fullft_v1.py) | 54 | 30,000 | `data/tomato_all_full_v1` |
| 番茄-Sharpa LoRA | [tomato_fr3_sharpa_lora_v1.py](training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_lora_v1.py) | 58 | 30,000 | `data/tomato_sharpa_fulltask_v1` |
| 番茄-Sharpa 全参 | [tomato_fr3_sharpa_fullft_v1.py](training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_fullft_v1.py) | 58 | 30,000 | `data/tomato_sharpa_fulltask_v1` |
| 抓取杯子 LoRA | [cup_single_franka_lora_v1.py](training_configs/experiments/cup_grasping/baseline/cup_single_franka_lora_v1.py) | 7（模型补零至 32） | 30,000 | `data/cup_grasping_v1` |
| 抓取杯子 全参 | [cup_single_franka_fullft_v1.py](training_configs/experiments/cup_grasping/baseline/cup_single_franka_fullft_v1.py) | 7（模型补零至 32） | 30,000 | `data/cup_grasping_v1` |
| 铲柠檬 LoRA | [lemon_ur_o30_lora_v1.py](training_configs/experiments/lemon_scooping/baseline/lemon_ur_o30_lora_v1.py) | 12（模型 32，状态 52） | 30,000 | `data/lemon_wrist_reset_v1` |
| 铲柠檬 全参 | [lemon_ur_o30_fullft_v1.py](training_configs/experiments/lemon_scooping/baseline/lemon_ur_o30_fullft_v1.py) | 12（模型 32，状态 52） | 30,000 | `data/lemon_wrist_reset_v1` |

番茄与杯子六份配置共用以下训练参数；柠檬 LoRA 复用公共训练配方，柠檬全参学习率与公共配方一致，日志与 checkpoint 保存设置沿用来源任务记录，详见[柠檬说明](training_configs/experiments/lemon_scooping/README.md)。以下公共参数中的频率与动作语义以各任务协议为准。

本体协议、数据身份、统计及英文 prompt 见对应任务说明。

公共训练参数：

| 参数 | 默认值 |
| --- | --- |
| 模型 / 精度 | π0.5，bfloat16；max_token_len=256，discrete_state_input=True |
| LoRA | VLM `gemma_2b_lora` rank/alpha=16/16；Expert `gemma_300m_lora` rank/alpha=32/32；视觉及投影可训练 |
| 全参 | `gemma_2b` + `gemma_300m`，所有参数可训练 |
| 频率 / horizon | 30 Hz / 50；所有 baseline 使用 absolute 动作目标；数据 action 必须为绝对目标值 |
| batch / workers / FSDP | 全局 batch：LoRA 8、全参 64 / workers 8 / fsdp_devices=1 |
| 采样 / seed | epoch_shuffle；seed=42、model_seed=42、data_seed=42001、augmentation_seed=42002 |
| 学习率 | cosine；warmup=1,000，peak=2.5e-5，decay_steps=30,000，end=2.5e-6 |
| 优化器 | AdamW；β1=0.9、β2=0.95、eps=1e-8、weight_decay=1e-10、clip=1.0 |
| EMA | 关闭 |
| 日志 / checkpoint | log_interval=10、save_interval=2,000、keep_period=10,000 |
| W&B | 启用，project=openpi；联网 online，离线显式 offline |

需要 A/B 按 7:3 等比例有放回采样时，只需在本地个人配置中设置 `sampling_mode="weighted_replacement"` 和 `sampling_groups`，见[采样配置示例](多人开发README.md#可选-ab-采样比例)。不设置时沿用 baseline 的逐轮打乱采样。

LoRA 和全参默认均训练 30,000 步，学习率计划的 decay_steps 同为 30,000；步数不随数据量自动调整。个人实验可按实际训练计划覆盖。

以下以 Wuji LoRA 为例，选择模块并展开配置，无需数据、权重或 GPU：

```bash
export MODULE=training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1
export EXP=alice_tomato_wuji_lora_s42_run01
bash scripts/python.sh -m scripts.run_training "$MODULE" --exp-name "$EXP" --dry-run
```

更换方案时修改 MODULE。个人修改数据路径、学习率或预算时，按 [多人开发 README](多人开发README.md) 在本地复制 baseline，再将 MODULE 指向本地模块；不需要创建分支或同步 GitLab。

## 数据与统计

将数据仓库产出的 LeRobot v2.1 数据放入 `data/`，包含 `meta/`、`data/`、`videos/`。在所选实验的 `build_dataset()` 中确认 root、repo_id、来源、版本、assets_dir、asset_id 和英文 prompt 示例。`root` 是实际目录，`repo_id` 是数据身份；训练实际英文指令来自 `meta/tasks.jsonl`。

数据应满足任务的关节顺序、单位、相机映射和采样频率：番茄与杯子为 30 Hz，柠檬为 10 Hz。新数据使用新的版本和 asset_id；同一内容仅搬迁路径时保留身份。

新数据先计算统计：

```bash
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 512
```

统计保存到 `<assets_dir>/<asset_id>/`，包含 `norm_stats.json` 和 `norm_metadata.json`。已有配套统计时跳过计算，迁移时两个文件一起复制。不要提前创建最终 asset_id 目录：工具拒绝覆盖已有目录，包括空目录；同一目标的重复计算进程也会被拒绝。

统计绑定数据文件、输入变换和 horizon。改变这些内容后重新计算；仅改学习率或 batch 可复用。常量关节允许 std=0，归一化使用 epsilon。番茄 Wuji / Sharpa 已切换为 absolute，四份 baseline 使用带 absolute 的新 asset_id 与默认实验名。必须重新计算统计，不能复用旧 delta 统计或对旧 checkpoint 直接 resume。

## 本机检查与训练

服务器启动命令见上方“500028 服务器”部分，包含公共基础权重路径。以下命令用于本机默认基础权重。

```bash
bash scripts/python.sh -m scripts.run_training "$MODULE" --exp-name "$EXP" --check
bash scripts/python.sh -m wandb login
CUDA_VISIBLE_DEVICES=0 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name "$EXP" --wandb-mode online
```

`--check` 验证数据元信息、fps、维度、相机、英文 task、统计数值及数据/变换绑定，需要已有统计，不构建模型。正式启动还核对实际帧数并记录数据文件与采样审计。它们不能代替关节语义、单位及视频/动作时间对齐的验收。

GPU 编号按分配修改，全局 batch 必须能被可见设备数整除；单 GPU 命令仅为示例。可临时用 `--batch-size 4` 等 CLI 参数调试，正式实验将设置写入本地个人模块。个人模块不提交；训练启动器会把它自动复制到日志目录，并在运行清单中记录文件哈希和当前公共仓库 commit。训练期间不要修改正在运行的目录。

无网络时使用 `--wandb-mode offline`，本地日志和 W&B 仍保存。恢复联网后逐个同步离线段：

```bash
bash scripts/python.sh -m wandb login
bash scripts/python.sh -m wandb sync wandb/offline-run-实际编号
```

## 输出与续训

训练输出和可写缓存默认位于当前仓库；服务器公共环境与基础权重使用上方服务器部分列出的共享路径：

| 内容 | 位置 |
| --- | --- |
| 本地环境 | 自己电脑使用 `.venv/`；服务器使用公共环境 |
| 缓存 | `.cache/`；JAX 编译缓存 `.cache/jax/` |
| 数据 / 统计 | `data/`、`assets/` |
| 本地日志、运行清单及配置源码快照 | `logs/<配置名>/<实验名>/` |
| checkpoint | `checkpoints/<配置名>/<实验名>/<step>/` |
| W&B | `wandb/` |

每个 checkpoint 的 `params/` 保存权重，`train_state/` 保存优化器等恢复状态，`assets/` 保存配套统计。运行清单记录 commit、最终配置、数据来源、实际 prompt、统计摘要和运行结果。

恢复未完成实验时使用原 commit、模块、配置、数据、统计和实验名。服务器使用上方服务器训练命令并加 `--resume`；本机执行：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name "$EXP" --resume --wandb-mode online
```

迁移续训保留完整实验目录，包括 checkpoint、`reproducibility/`、`wandb_id.txt` 和 `wandb_sessions/`，以及未同步的 W&B 日志。只有 params 不足以精确续训。缺少有效 checkpoint 会报错；已完成原预算直接退出。改变代码、数据或预算后创建新实验，不能在新 main 上直接恢复旧运行。

每次启动或续训创建一个独立 W&B run，同一实验通过固定 group 关联；云端不会自动拼成一条曲线。各段的恢复步数和目录记录在 `wandb_sessions/`，离线段需逐个 sync。

历史训练参数、checkpoint 与指标见 [历史实验记录](docs/records/历史实验记录.md)。本仓库不包含这些历史运行产物。

GitLab 自动验证目前暂停。本地代码检查命令为 `bash scripts/check_training.sh`，协作与验证规则见 [多人开发 README](多人开发README.md)。本项目遵循 [LICENSE](LICENSE) 和 [Gemma 许可](LICENSE_GEMMA.txt)。
