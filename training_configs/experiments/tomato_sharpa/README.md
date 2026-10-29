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

## 本地 Delta 论文算法实验（不修改 baseline）

共享代码新增显式选择的 `PaperPi0Config` / `PaperSharpaData`，原八份 baseline、recipe、本体配置和原 Pi0 默认实现保持原样。个人配置仍只放在 `training_configs/experiments/tomato_sharpa/template/<负责人>/`，被 Git 忽略；不要强制提交。算法代码在 `src/openpi/models/pi0_delta_paper.py`、`src/openpi/training/delta_paper*.py`、`src/openpi/shared/delta_paper.py`；统计、预检查、数据加载和保存入口只有选择新类型时才进入新路径。

### 配置位置与对照

本机已在 `template/ma39nong/` 创建五份个人文件。其他机器首次获得共享代码后，可在自己的 clone 中创建，已有负责人目录会拒绝覆盖：

```bash
bash scripts/python.sh -m scripts.create_delta_paper_experiments --owner ma39nong
```

| 文件 | 双臂14维归一化 | Flow Matching 噪声 |
| --- | --- | --- |
| `delta_paper_t0.py` | 全局每维 quantile | 独立高斯 |
| `delta_paper_t1.py` | 每未来时间位置 mean/std | 独立高斯 |
| `delta_paper_t2.py` | 全局每维 quantile | 完整50×58相关噪声，beta=0.5 |
| `delta_paper_t3.py` | 每未来时间位置 mean/std | 完整50×58相关噪声，beta=0.5 |
| `delta_paper_tg.py` | 全局每维 mean/std | 独立高斯；区分z-score与逐时间归一化的贡献 |

所有组都只将双臂转换为“未来绝对目标减本轮起点实测状态”，双手44维仍为Absolute及原quantile规则。近零双臂std使用1e-4 rad下限。原始数据仍为116D状态、58D绝对动作，适配后58D；英文prompt来自训练数据的 `meta/tasks.jsonl`。

所有组**统一只训练完整50步窗口**，不足50步的episode尾部起点不参与训练或统计。这是为保证统计与标签完全一致而建立的新对照，T0不是旧模型原样重训；与旧baseline相比还包含窗口覆盖变化，不能把差异都归因于噪声或归一化。T1对T0同时包含quantile改z-score的变化，若要单独判断逐时间位置的贡献，应比较T1与TG。

相关矩阵从同一训练集的归一化动作窗口按 `[time,dimension]` 展平估计，大小2900×2900；按特征中心化并标准化为相关矩阵，常量维度独立，使用 `beta*Sigma+(1-beta)*I` 做Cholesky。训练和推理初始化使用同一噪声定义；beta=0保持原随机流与计算路径。默认均匀无放回抽取最多20000个完整窗口拟合相关性，seed=42；逐时间统计及global统计使用全部训练窗口。计算可能占用约GB级RAM及较长CPU时间，可在个人配置调整 `correlation_max_windows`，需新统计身份。当前实验只支持单个30Hz数据集、epoch_shuffle、JAX全参入口；不支持TaskView、联合数据集或PyTorch。

修改数据来源/版本、root、repo_id、asset_id和统计父目录：在**个人文件的 `build_dataset()`** 返回值上用 `dataclasses.replace` 调整；只搬迁相同数据可用 `PI05_SHARPA_MULTIMODAL_DATASET_ROOT`，论文统计父目录可用 `PI05_PAPER_ASSETS_DIR`。换批次须记录新的来源、版本与asset_id。算法开关在个人 `build_config()` 的 `mode/beta`；batch、FSDP、预算等也用replace覆盖。不要修改公共baseline。UI可以选择个人模块，但本轮不增加算法按钮；数据/统计路径应先在个人文件或上述环境变量中确定，不要只改UI统计路径而保留旧模型资产路径，预检查会拒绝这种不匹配。

### 统计、检查和训练

默认沿用Sharpa全参：batch32、FSDP8、30000步、原学习率、独立seeds、本地日志及W&B。下面以T3为例；已有本机配置不再执行初始化命令：

```bash
export MODULE=training_configs.experiments.tomato_sharpa.template.ma39nong.delta_paper_t3
bash scripts/python.sh -m scripts.run_training "$MODULE" --dry-run
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
bash scripts/python.sh -m scripts.run_training "$MODULE" --check
# 数据、资产、GPU检查通过后再单独启动；下面会开始实际训练。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name ma39nong_sharpa_delta_paper_t3_s42_run01 --wandb-mode online
```

离线服务器显式改为 `--wandb-mode offline`，之后按原规则sync。正式运行仍写入原 `logs/`、`checkpoints/`、`wandb/`；启动器会保存实际个人源码、公共commit、数据、prompt及统计审计。改变数据、统计或算法后不能resume旧运行，须新实验名。

每组使用独立asset_id并原子发布，拒绝覆盖已有统计目录。资产包含 `norm_stats.json`、`norm_metadata.json`、`delta_paper.npz` 和 `delta_paper.json`，绑定来源、算法和摘要，checkpoint会完整保存它们；相关因子是非训练变量，随checkpoint保存且不进入优化器。新的actions统计采用 `(50,58)` 版本布局，双手global统计按时间广播；原 `(58,)` 资产不变。

**新训练权重不能直接交给现有部署profile。** 本轮只改训练仓库，未改任何串行、RTC或Soft部署。现有部署会拒绝新的统计形状，后续需单独接入新模型类型、匹配统计及相关噪声采样，不得只换checkpoint路径。

本轮状态为代码与离线验证准备完成：已验证合成parquet的真实转换、完整50×58统计/2900×2900因子、实际loader及采样审计、缩小Gemma/SigLIP的梯度与优化器更新、checkpoint保存/恢复和beta=0等价性。没有计算服务器真实训练集的新统计，没有启动真实数据完整训练或实机测试。此前 `template/timestamp_norm_trial.py` 及其已有结果保留原样。
