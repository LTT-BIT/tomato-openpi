# FR3 + Sharpa 番茄训练

## normal93 无 Agent 训练准备

分支 `tomato-subtasks-ab-train`，从四类指令统一后的 `56ca763` 建立。此实验不调用任何阶段 Agent；B 训练使用人工审核的阶段文本，未来部署由人观察并点击阶段，在线 prompt 与动作队列切换属于后续部署任务。

固定清单为 `subtasks/records/normal93_20261109_selection.json`：28 上 `normal/no-block-more/A`、`B` 的93个独立录制，410456帧，30Hz。排除 `A/1106_740_796_normal/episode_000009`（仅三段）及 `B/1107_696_696_normal/episode_000000`（历史剔除、无发布粗标注）。不用 speedup 或 C。清单绑定各条来源ID、帧数和标注SHA，新增、缺失或改动均需人工检查并新建实验版本，不自动纳入。

A/B 均使用全部93条原始轨迹训练，410456帧，不预留验证或测试集。保留完整视频和所有物理列；仅更新合并索引、task_index，新增 coarse_phase_index。B 的七个位置映射到四种无标点文本，两次 return 共用 held object 指令。50步动作窗口允许跨阶段，尾部沿用 episode 内末帧补齐，不在阶段边界制造保持动作。`splits.json` 仅包含 train=0..92，是全部训练数据的显式清单。

两套新配置分别为 `training_configs.experiments.tomato_sharpa.subtasks.normal93_a_goal` 和 `normal93_b_stage`。默认 π0.5 全参、58D absolute、batch64、FSDP8、30000步，种子42/42/42001/42002，学习率与原 fullft recipe 相同；这只是待启动默认值，不是训练已完成或显存容量验收。没有开启论文增强或 Agent。数据身份、统计目录、实验目录均与旧单果/FLASH/Agent实验分开；归一化覆盖全部93条，A/B分别保存绑定统计。

在28的既有工程 `/home/mengxiang/workspace/code/tomato-openpi-sharpa` 使用公共环境：

```bash
export PI05_TOMATO_SUBTASK_BUNDLE="$PWD/data/tomato_truss_normal93_20261109_annotation_r2_all_sources_v2"
bash scripts/python.sh -m scripts.prepare_tomato_ab \
  --source-root /mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-more \
  --output "$PI05_TOMATO_SUBTASK_BUNDLE" \
  --tokenizer "$HOME/.cache/openpi/big_vision/paligemma_tokenizer.model"
# 生成新目录，拒绝覆盖。准备过程核验全部源SHA和真实token预算，不修改源数据。
for mode in a_goal b_stage; do
  MODULE="training_configs.experiments.tomato_sharpa.subtasks.normal93_${mode}"
  bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
done
bash scripts/python.sh -m scripts.check_tomato_ab --require-norm \
  --output outputs/normal93_annotation_r2_ab_verification.json
```

检查器比较所有源与派生 Parquet 的物理列、逐帧阶段/task映射、全部派生媒体SHA、全部93条实际加载结果和全部阶段边界的50步动作窗口，并为A/B各读取一个真实归一化模型batch。不加载模型权重、不运行训练；效果后续通过实机测试判断。

本次已接收 B/1106_750_791_normal 原 episode4 的人工 revision2：最终放回为 `[2908,2990)`，82帧（2.73秒）。冻结清单仅该条标注SHA改变，93条成员、物理轨迹、四种prompt和全部训练策略不变。新包采用 `annotation_r2_all_sources_v2` 身份，旧包保留，不会覆盖；默认配置读取清单中的 bundle_name。B 在帧2908至2988从harvest更新为return。

本次使用独立 `all_sources_v2` 数据、统计和实验身份。旧75/9/9包和75条统计保留为历史，当前配置拒绝加载它们。93条物理示教和prompt没有变化，仅纠正训练选择及其统计。batch64、FSDP8、30000步、学习率、采样方式和各随机种子均保持本分支既定值，不因修改数据选择而更改。

2026-11-09 已在28重建并验收全数据版本：A/B均实际加载93条、410456帧，归一化覆盖相同的全部训练数据且数值SHA一致；逐帧指令、原始物理列、媒体SHA、阶段边界50步窗口及真实归一化batch全部通过。76项相关测试通过。结果摘要见 `subtasks/records/normal93_20261109_all_sources_validation.json`，完整报告位于服务器 `outputs/normal93_annotation_r2_ab_verification.json`。未加载模型权重、未启动正式训练、未修改部署端。

正式训练另行明确参数后，沿用 `scripts.run_training`；默认使用新的 all_sources_v2 包；若设置 bundle 环境变量，必须指向该新包，W&B online/offline与本地日志同时保存，不覆盖旧实验目录。

## 历史粗阶段 A/B 准备设计（不用于当前全数据训练）

新配置位于 `subtasks/`，不替换下方历史单果 baseline。先完成人工粗阶段审核，再准备数据，当前没有启动训练或移动生产数据。粗阶段是4种动作按顺序出现7段：拿起、摘取、放回、调整、再次拿起、继续摘取、放回裸杆。异常轨迹不强套模板，退回人工判断；细阶段本轮不用于训练。

- A：`training_configs.experiments.tomato_sharpa.subtasks.coarse_a_goal_v1`，全帧使用 `Harvest all tomatoes into the left basket and return stem to right plate.`。这是紧凑的总体目标，没有枚举左右手分工和中途调整；完整流程保存在 bundle 的 `procedure` 和人工阶段标注中。它不等同于把完整流程文本都输入模型。
- B：`training_configs.experiments.tomato_sharpa.subtasks.coarse_b_stage_v1`，逐帧读取当前粗阶段英文。左右手分工、盘上调整使用对应动作文本，两次放回共用同一句指令，见 `src/openpi/subtasks/tomato.py`。模型学到的是阶段条件动作，实机需要阶段调度器。

两者保持完整 episode、图像、时间戳、state/action、horizon 50，不切视频、不制造边界静止动作。B 在交界附近仍保留跨阶段的真实未来动作窗口；因此当前指令不保证覆盖窗口内每个动作，这需要后续对比评估，不能靠补造保持动作解决。统计通过既有 TaskView 仅对 train split 计算。验证/测试应同样使用 `splits.json` 的 val/test 条目，不用训练集结果代替评估。普通训练入口不自动评测整条任务成功率。

任务描述与58维离散状态共同受256-token限制。准备工具使用实际 PaliGemma 分词器检查每个状态离散值、模板和文本组成的上界，超过即拒绝。A 最多256，B 最多255；没有增加长度配置，也不依赖静默截断。具体分词器指纹和检查结果在 `subtasks/records/token_budget.json`。更换分词器、状态维度或英文后必须重查。完整旧长描述保留于 `PROCEDURE` 元数据，不作为256-token模型输入。

准备输入参考 `subtasks/preparation_plan.example.json`：每条填写本地可访问的 LeRobot 包、原 episode 编号、粗阶段 JSON 和原始 rosbag 的稳定 `recording_group`。normal/speedup 必须共享录制组；`recording_namespace` 是采集来源命名空间，同一录制两版必须一致，不能填 normal/speedup。工具核对 SHA256、全段人工确认、7段完整覆盖与来源一致性，要求至少10个独立录制；完成全部计划数据审核后才将 `review_complete` 设为 true。这个字段是人工对清单完整性的声明，工具不能知道遗漏的采集文件。

```bash
# 全部数据审核完成后执行，当前不要运行；输出必须是尚不存在的新目录。
bash scripts/python.sh -m openpi.subtasks.prepare \
  --plan /absolute/path/to/reviewed-plan.json \
  --output /absolute/path/to/tomato_truss_v1 \
  --tokenizer /absolute/path/to/paligemma_tokenizer.model
```

输出集中在同一位置：`a_goal/`、`b_coarse/`、`splits.json`、`bundle.json`。两版完整复制视频，不重编码或裁剪；仅更改新副本的索引、task_index，并增加 coarse_phase_index。原始文件不写入。bundle 记录来源 SHA、原 episode 到新编号映射、人工标注 SHA、英文和 A/B 共用的80/10/10分组。normal/speedup 可以都入清单但不应当作独立录制；同源两版共享 split。源目录应是可访问文件路径，不能直接填写 sftp URL。工具会复制两套媒体，请选择容量足够的目标；这不是现在替用户搬运或汇总数据。

准备后设置 `PI05_TOMATO_SUBTASK_BUNDLE=/absolute/path/to/tomato_truss_v1`，分别加载 A/B 配置，沿用下文检查、计算独立统计及训练的步骤。未准备 bundle 时配置会明确拒绝；不要复用单果任务统计、旧实验目录或柠檬分类头。两套实验使用同一 fullft recipe 和种子，各有独立 assets/日志/权重，不自动启动。

## 阶段决策 agent

从用户提供的 `/home/user/lemon_vjepa2_training_20261106` 迁移并适配 V-JEPA 2 的因果视频历史、冻结视频编码、分类头和原始录制分组思想，来源文件指纹在 `subtasks/records/migration.json`。源包 `serve_lemon_phase_agent.py` 接的是旧静态图像模型，未复制；柠檬六类头、12维UR状态、旧权重及数据均未迁入。

新模型输入三路16帧历史，按时间从已发生的观测采样10Hz、覆盖1.5秒，早期不足重复首帧；不跨 episode。当前116维转换器状态按已有适配顺序提取58维实测位置。输出7个阶段位置，区分两次拿起/摘取。分类头只能用番茄已审核数据重新训练；V-JEPA预训练权重从指定本地目录加载并校验固定SHA，无自动下载。

```bash
# 数据准备及V-JEPA本地权重就绪后才能训练agent，以下当前未运行。
bash scripts/python.sh -m scripts.train_tomato_phase_agent \
  --bundle /absolute/path/to/tomato_truss_v1 \
  --encoder /absolute/path/to/pinned-vjepa2 \
  --output /absolute/path/to/new-agent-run --device cuda --wandb-mode offline
```

脚本按固定2Hz时间表提取三路因果视频及关节历史；归一化只用训练组；验证 macro F1 选最优头，测试组仅在选完后评估，日志同时本地及W&B保存，离线日志之后同步。V-JEPA 编码和头训练需要运行环境提供 torch、cv2、Transformers 的 VJEPA2Model；当前机器尚未完成这个依赖/权重/GPU链路验收，不能把迁移代码与合成测试称为真实模型已可用。

运行时通过 `PhaseModel.predict()` 产生概率，再交给 `PhaseDecision`；`python -m scripts.tomato_phase_decision` 提供概率输入的JSONL决策接口，本身不看图像。连续3次概率达到0.8才允许进入相邻下一阶段，不按时间跳转，不跨阶段跳转；疑似异常或低置信度返回 `needs_review`。最后阶段是“放回裸杆”，不是完成检测；不会自动宣称任务成功或停止机器人。阈值只是初始实验参数，需在验证集校准。真机相机同步、丢帧/延迟处理、动作执行端、完成判定及与π0.5共用GPU的性能尚未接入或验证。

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

#### 四类动作 prompt 与七阶段进度

当前 B 使用四类动作到四种无标点指令的一对一映射，定义在 `openpi.subtasks.tomato.ACTION_PROMPTS`。两次放回共用 `Right hand returns the held object to the right plate`，不再用不同裸杆指令。七个 `PHASES` 和 `coarse_phase_index` 保留，`PROMPTS` 由七位置的 `COARSE_SEQUENCE` 映射得到，因此仍有七个位置但只有四种不同文本。Agent 输出同时包含 `phase`、`phase_index`、`action_class`、`prompt`；其完成状态仍需独立判断，不因进入最后阶段自动完成。A 的总指令未改；旧五指令训练副本或 checkpoint 不会自动迁移，需使用新文本重新准备实验数据。该修改未运行训练或实机控制。
