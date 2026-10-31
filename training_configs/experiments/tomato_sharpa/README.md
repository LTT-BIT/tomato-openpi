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

生成器会在本人目录创建五份个人文件。首次获得共享代码后，可在自己的 clone 中创建，已有负责人目录会拒绝覆盖：

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

相关矩阵从同一训练集的归一化动作窗口按 `[time,dimension]` 展平估计，大小2900×2900；按特征中心化并标准化为相关矩阵，常量维度独立，使用 `beta*Sigma+(1-beta)*I` 做Cholesky。训练和推理初始化使用同一噪声定义；beta=0保持原随机流与计算路径。默认均匀无放回抽取最多20000个完整窗口拟合相关性，seed=42；逐时间统计及global统计使用全部训练窗口。计算可能占用约GB级RAM及较长CPU时间，可在个人配置调整 `correlation_max_windows`，需新统计身份。当前实验支持单个或显式列出的多个30Hz数据集，采用epoch_shuffle和JAX全参入口；不支持TaskView、加权重复采样或PyTorch。多源按完整窗口统一打乱，每个有效起点同权，不按源等概率重复采样。

修改数据来源/版本、root、repo_id、asset_id和统计父目录：在**个人文件的 `build_dataset()`** 返回值上用 `dataclasses.replace` 调整；只搬迁相同数据可用 `PI05_SHARPA_MULTIMODAL_DATASET_ROOT`，论文统计父目录可用 `PI05_PAPER_ASSETS_DIR`。换批次须记录新的来源、版本与asset_id。算法开关在个人 `build_config()` 的 `mode/beta`；batch、FSDP、预算等也用replace覆盖。不要修改公共baseline。UI可以选择个人模块，无需另加算法按钮；论文UI适配保留个人配置中的数据源身份和顺序，修改统计存储位置时同步更新模型资产路径和摘要。更换数据须在个人文件定义新数据版本，UI不能直接改论文数据目录、顺序、repo_id或切换加权采样。

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

**新训练权重不能直接交给没有该组合适配的部署profile。** 本轮只改训练仓库，未改任何串行、RTC或Soft部署。旧标准profile会拒绝新的统计形状；已有独立Paper与Train RTC模式的外部工程，也仍需单独接入组合模型、匹配统计及相关噪声采样，不得只换checkpoint路径。

本轮状态为代码与离线验证准备完成：已验证合成parquet的真实转换、完整50×58统计/2900×2900因子、实际loader及采样审计、缩小Gemma/SigLIP的梯度与优化器更新、checkpoint保存/恢复和beta=0等价性。没有计算服务器真实训练集的新统计，没有启动真实数据完整训练或实机测试。此前 `template/timestamp_norm_trial.py` 及其已有结果保留原样。


### no-block-one 六份摘番茄数据联合实验

数据保持在 `/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one`，不搬迁、复制、合并、建软链接或修改原始meta/parquet/video。共享helper `build_no_block_one_dataset` 显式列出以下顺序；不自动扫描新加入的目录：

| 相对原目录 | episode | 原帧数 | 完整50步窗口 |
| --- | ---: | ---: | ---: |
| A/1022_121_430_normal_lerobot | 168 | 157442 | 149210 |
| A/1022_54_120_normal_lerobot | 41 | 37145 | 35136 |
| A/1026_444_513_normal_lerobot | 47 | 37240 | 34937 |
| B/1022_130_423_normal_lerobot | 34 | 40221 | 38555 |
| B/1022_56_106_normal_lerobot | 8 | 10607 | 10215 |
| B/1026_446_510_normal_lerobot | 13 | 11551 | 10914 |
| 合计 | 311 | 294206 | 278967 |

以上是2026-10-29读取实际元数据得到的数量，不能代替完整读取与训练验证。六份数据的帧率、维度、关节名字顺序和相机角色已做只读检查，均一致；来源的任务文本均为本页相同英文摘番茄指令。

每个源先独立取完整窗口，再在内存联合；episode编号只在采样审计里映射，不写回数据。训练对全部278967个完整起点做epoch_shuffle，统计也按同样起点权重覆盖全部六份；相关性从全部完整窗口均匀无放回抽最多20000个。大批次自然贡献更多窗口，不把小批次放大为相同权重。手部Absolute、双臂Delta及逐时间统计规则不变，不加入Multi-Sample Flow Matching。

统计使用独立 `tomato_sharpa_no_block_one_<owner>_delta_paper_t3_v1_30hz` 身份，记录六个原始目录、逐源帧数/episode/有效窗口数和文件摘要；任一源内容、顺序或算法改变都会导致检查拒绝复用。原10.16_2实验仍保留，不能复用它的统计或实验名称。

服务器同步共享代码后，第一次可生成个人配置；若负责人目录已存在，生成器会拒绝覆盖。来源分支曾使用被Git忽略的个人配置；这些文件不随共享代码迁移。服务器例子：

```bash
bash scripts/python.sh -m scripts.create_delta_paper_experiments --owner mengxiang --dataset no-block-one
export MODULE=training_configs.experiments.tomato_sharpa.template.mengxiang.delta_paper_no_block_one_t3
bash scripts/python.sh -m scripts.run_training "$MODULE" --dry-run
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
bash scripts/python.sh -m scripts.run_training "$MODULE" --check
# 下列命令才开始真实训练；保持原全参配方、8卡、batch32和30000步。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name mengxiang_sharpa_no_block_one_delta_paper_t3_s42_run01 --wandb-mode online
```

无法联网时显式采用 `--wandb-mode offline`，日志仍写本地，之后sync。此处不修改baseline、main、共享环境或数据权限。当前仅完成多源代码与合成数据离线验证、服务器六源元数据只读预检查，尚未同步服务器代码、计算真实联合统计或启动训练。新训练权重依旧需要单独的论文部署profile适配。


### UI使用现有论文统计

服务器启动 `bash scripts/python.sh -m scripts.training_ui --port 8765`，通过SSH转发端口后，在个人配置选择 `mengxiang / delta_paper_no_block_one_t3`。保留六个数据目录、逐时间/相关噪声T3算法、horizon50、epoch_shuffle以及现有asset_id，可修改实验名、batch/FSDP和保存步数等训练参数。UI预检查应显示现有统计通过校验，不需要重新计算。选择8张卡、batch32/FSDP8、W&B offline后，点击开始才会启动训练。

UI专用适配位于 `src/openpi/training/delta_paper_ui.py`；默认baseline UI行为不变，论文统计代码及其源码摘要也未改变，所以能复用命令行计算的资产。改变统计目录/asset_id时模型路径和摘要同步更新；原目录不存在统计时可由UI自动计算。UI保存的个人副本继续保留完整T3类型与六源身份，不依赖其他个人文件。代码同步或修改后须重启本工程UI进程；不停止其他用户的UI或训练。

## T3 + trainRTC 集成

本集成基于主线 `897cb917` 和来源 T3 `2ab68c8b`。保留来源的统计、完整窗口、相关矩阵算法与实现摘要；原有 baseline 不变。普通 Pi0 的 `noise()` 默认生成原标准高斯；Paper 仅覆盖该噪声函数，训练损失和采样继承主线 RTC。仅支持 JAX π0.5 Sharpa 全参，不实现论文第9页的 soft / correlation-aware inpainting，也不增加异步机器人执行服务。

### UI 四种选择

| 场景 | 配置入口 | RTC 上限 | 模型/统计 |
| --- | --- | ---: | --- |
| 普通训练 | 正式 Sharpa 全参 baseline 或普通个人副本 | 0 | 原生 Pi0Config；普通统计，无 Paper 资产 |
| 仅原始 trainRTC | 与上一行相同 | 10 | 仍是原生 Pi0Config / 原数据工厂；标准高斯 |
| 仅 T3 | 个人 `delta_paper_t3` 或六源对应文件 | 0 | PaperPi0Config / PaperSharpaData；逐时间双臂 z-score + beta=.5 |
| T3+RTC | 与上一行相同的 T3 配置与统计身份 | 10 | 相同 T3 资产；同一 forward 叠加 clean-prefix / suffix-only loss |

先选择配置，再修改现有 RTC 字段，查看最终配置、数据预检查，通过后使用新的实验名启动。Paper 配置的双臂 delta/双手 absolute、源目录顺序、repo_id、数据版本和 epoch_shuffle 固定；UI 会拒绝 absolute 或加权重复采样。配置摘要显示有效 RTC、Paper mode/beta/摘要与统计路径。`beta=0` 不等于退出 Paper：时间归一化和完整窗口仍可能存在。切换 Paper / 非 Paper 时不得混用统计和 checkpoint。

### 独立生成、校验和启动

原生成器默认五组输出完全保留，RTC 默认为0。已有本人 T3 配置可直接在 UI 改 RTC 并用新实验名；也可显式生成一套新 RTC 个人配置，不覆盖原五组：

```bash
# 单源；数据来源默认保持10.16_2定义，修改数据须在个人build_dataset里定义独立身份。
bash scripts/python.sh -m scripts.create_delta_paper_experiments --owner alice --rtc-max-delay 10
# 或六源预设（与单源命令二选一；同一rtc_d10目录不会被覆盖）：
bash scripts/python.sh -m scripts.create_delta_paper_experiments \
  --owner alice --dataset no-block-one --rtc-max-delay 10
export MODULE=training_configs.experiments.tomato_sharpa.template.alice.rtc_d10.delta_paper_no_block_one_t3_rtc_d10
bash scripts/python.sh -m scripts.run_training "$MODULE" --dry-run
# 如果已有匹配的T3资产，只做check；没有资产时才先计算统计，工具拒绝覆盖已有目录。
# bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
bash scripts/python.sh -m scripts.run_training "$MODULE" --check
```

选择单源时 MODULE 改为 `...template.alice.rtc_d10.delta_paper_t3_rtc_d10`。新配置保留原 trial 的数据与 asset_id，name/exp_name 增加 `_rtc_d10`，因此 RTC 本身不要求新统计且 checkpoint/run 独立。UI 个人配置列表同时发现原文件及 `rtc_d10/` 子目录。统计路径由本人配置与 `PI05_PAPER_ASSETS_DIR` 决定；源数据、来源顺序、英文 prompt、asset 摘要会随运行记录保存，英文 prompt 来自各数据集的 `meta/tasks.jsonl`。

真正开始训练仍使用统一入口，按分配的 GPU 和确认后的最终配置执行，并开启本地日志与 W&B online；无法联网显式用 `--wandb-mode offline`，恢复后执行 `bash scripts/python.sh -m wandb sync wandb/offline-run-...`。此文的示例不代表已经启动8卡训练。

已有 T3 checkpoint 可用匹配资产和 `PaperWeightLoader(params_path=".../step/params")` 初始化新的 RTC 实验；该加载器保留由已校验 bundle 初始化的非训练 Cholesky。优化器/步数重新开始，`resume=False`。真正中断恢复必须维持原代码、数据、RTC、T3 参数和 reproducibility 合同，不能绕过严格检查。新 checkpoint 保存 Cholesky 变量及 `norm_stats.json`、`norm_metadata.json`、`delta_paper.npz`、`delta_paper.json`；历史资产不被改写。

### 模型接口与跨 chunk 前缀

采样继承 `sample_actions(rng, observation, num_steps=10, noise=None, prefix_actions=None, prefix_length=0)`。默认 noise 经过一次 `z @ L.T`，显式 noise 已在模型尺度下，不再次相关化。前缀长度必须是 `0..rtc_max_delay` 的静态整数，JIT 使用 `static_argnames=("prefix_length",)`；初始及每次积分后均固定前缀。30Hz下上限10覆盖约333ms的训练延迟，部署选择仍须根据实际延迟测量。

旧 chunk 的归一化后缀不能直接作为新 chunk 的归一化前缀。先按旧时间索引逆 ArmNormalization、逆 quantile，并加回旧物理 state 得到绝对目标；再减新物理 state（仅双臂），以新时间索引0..N-1做 quantile 和 T3 编码。双手始终 absolute/quantile，不按时间做 z-score。

`openpi.training.delta_paper_prefix` 提供无机器人依赖的验证 helper，接受 `config.data.create(...)` 得到的 PaperDataView 和 canonical LA7/LH22/RA7/RH22 物理向量：

```python
from openpi.training.delta_paper_prefix import decode_actions, encode_prefix

old_absolute = decode_actions(data, old_physical_state, old_model_actions)
physical_prefix = old_absolute[-prefix_length:]
new_prefix = encode_prefix(data, new_physical_state, physical_prefix)  # 返回H×58，只有前N步承诺
# prefix_length>0；为0时不需要构造前缀。
actions = sample(rng, new_observation, prefix_actions=new_prefix[None], prefix_length=prefix_length)
```

`physical_state` 必须是未归一化的58D实测关节位置，不能传 `observation.state` 的模型归一化值。新 prefix 补齐到完整 horizon 后按新时间位置编码；模型输出解码时必须使用同一新起点 state。此 helper 没有连接任何执行队列，实际异步时间对齐、超时处理与实机成功率仍须由外部部署单独接入验证。

### 验证状态

本次使用临时合成 parquet/fixture，验证标准 Pi0 与原始 RTC 回归、T3数值/完整窗口/六源统计、50×58联合 JIT 前后向和单步优化、非训练Cholesky、四文件checkpoint恢复、旧T3无RTC损失等价、跨chunk物理前缀round-trip、UI四模式保存/重载/preflight及只改RTC的资产复用。没有本地历史真实T3资产可做实盘预检查；受保护的三个统计/数据源码与来源分支逐字一致，未清除或跳过digest。

训练/模型采样支持与离线验证完成不代表真实数据完整训练、8卡吞吐验证或机器人实机验证。服务器既有任务、数据、资产与checkpoint未修改。本仓库未接入外部RTC队列。
