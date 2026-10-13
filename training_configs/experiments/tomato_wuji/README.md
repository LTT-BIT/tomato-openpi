# 番茄-Wuji 训练说明

安装、数据校验、统计、日志与续训见 [仓库 README](../../../README.md)，个人配置和协作规则见 [多人开发 README](../../../多人开发README.md)。

## 本体与数据协议

使用 FR3 + Wuji，LeRobot v2.1，30 Hz、horizon 50，`meta/`、`data/`、`videos/` 必须齐备。原始 action 54 维，原始 state 可为 54 或 108 维；适配后的向量顺序为左臂 7、左手 20、右臂 7、右手 20。双臂使用相对当前 state 的 delta，手部使用 absolute；cam0 外部、cam1 左腕、cam2 右腕。关节顺序、单位和数据语义需由数据生产者确认，不能仅靠维度判断。

本体规格见 [fr3_wuji.py](../../embodiments/fr3_wuji.py)，完整适配见 [fr3_wuji_policy.py](../../../src/openpi/policies/fr3_wuji_policy.py)。Sharpa 有自己的关节和相机协议，不能直接复用 Wuji 数据或统计。

## Baseline

| 方案 | 文件 | 默认预算 |
| --- | --- | --- |
| LoRA | [tomato_fr3_lora_v1.py](baseline/tomato_fr3_lora_v1.py) | 30,000 步 |
| 全参 | [tomato_fr3_fullft_v1.py](baseline/tomato_fr3_fullft_v1.py) | 30,000 步 |

模块前缀为 `training_configs.experiments.tomato_wuji.baseline.`。两份文件分别组合公共 `pi05_lora_v1` / `pi05_fullft_v1`；全局 batch：LoRA 8、全参 64；workers 8、bfloat16、warmup 1000、peak lr 2.5e-5、end lr 2.5e-6、decay 30000、无 EMA、epoch_shuffle、seed 42/42/42001/42002、每 2000 步保存、keep_period 10000、W&B 启用。

| 设置 | LoRA | 全参 |
| --- | --- | --- |
| 默认数据目录 | `data/tomato_fulltask_v1` | `data/tomato_all_full_v1` |
| repo_id | `fr3_wuji/tomato_fulltask_v1` | `fr3_wuji/tomato_all_full_v1` |
| 数据版本 / asset_id | `tomato_fulltask_v1_30hz` | `tomato_all_full_v1_30hz` |
| 统计目录 | `assets/fulltask_c0/tomato_fulltask_v1_30hz/` | `assets/tomato_all_full_v1_30hz/` |
| 配置名 | `pi05_fr3_wuji_fulltask_lora_v1_30hz` | `pi05_fr3_wuji_fulltask_fullft_v1_30hz` |
| 默认实验名 | `tomato_fulltask_lora_v1_30hz_s42` | `tomato_all_full_fullft_v1_30hz_s42` |
| 准备状态 | 占位；原记录未确认数据准备或训练完成 | 历史数据已准备、训练已完成；新机器需迁移 |

路径相对于仓库根目录。LoRA 的 sources 默认为数据目录；全参历史来源为 `data/harvest-tomato/0925/A`、`0925/B`、`1002_AB_190/full`，441 episode、481,223 帧。记录 sources 不会自动转换或合并数据。

可选本机覆盖变量：LoRA 为 `PI05_FULLTASK_DATASET_ROOT` / `PI05_FULLTASK_ASSETS_DIR`；全参为 `PI05_ALL_FULL_DATASET_ROOT` / `PI05_ALL_FULL_ASSETS_DIR`，来源父目录为 `PI05_SOURCE_DATA_DIR`。旧 `PI05_V2_*` / `PI05_V3_*` 不再使用。个人实验可直接修改自己的路径，不必沿用公共基准的变量名。

全参历史统计的规范对象审计摘要固定为 `f933853e8e1bfbf8a7f37e377bd9a5a87c57d37a437685a95e464bcc65c24e23`，通过 `expected_norm_audit_sha256` 校验，且显式允许 `allow_legacy_norm_stats=True`；该值不是 JSON 文件 SHA-256。复制全参 baseline 后若改用新数据，删除历史摘要和旧统计兼容设置，使用新的 asset_id 计算绑定统计，勿继承旧来源。

英文 prompt 示例保留各自基准：

- LoRA：`Pick up a tomato truss with the right hand and hold it steady between both hands, then pick a single cherry tomato from the truss with the left hand and place the picked tomato into the plate on the left.`
- 全参：`Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.`

实际英文指令来自数据的 `meta/tasks.jsonl`。

## 选择模块并训练

在仓库根目录设置模块，全参将尾名换为 `tomato_fr3_fullft_v1`：

```bash
export MODULE=training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1
export EXP=alice_tomato_wuji_lora_s42_run01
.venv/bin/python -m scripts.run_training "$MODULE" --exp-name "$EXP" --dry-run
```

按根 README 执行新数据统计 → check → 训练；已有配套统计时跳过计算。修改数据和参数时按多人开发 README 复制为个人配置，再更新 MODULE。当前 template 为空。

## 原始采集对照校验

需要将数据与原始 FR3/Wuji 采集记录额外对照时，可以运行独立校验工具：

```bash
.venv/bin/python -m examples.fr3_wuji.verify_dataset \
  --src-root data/raw_captures \
  --root data/tomato_fulltask_v1 \
  --repo-id fr3_wuji/tomato_fulltask_v1 \
  --episodes 2 --samples 6
```

两个路径都必填：src-root 是原始逐 episode 采集目录的父目录，root 是准备好的目标 LeRobot 数据集；前者需带采集时的 dataloader.py。工具按源采集目录的数字顺序与目标 episode 顺序对照，因此不能直接用于已筛选、重排或合并后顺序不同的数据。它抽样检查 episode 长度、episode 索引及 state/action 数值一致性，另外选取运动图像与前后邻帧比较对齐。默认仅检查前两个 episode，不是全量验收；缺少原始数据时跳过此项，仍可执行实验 --check。

输出 FAIL 会报错退出；OK 表示本次抽样通过。图像比较只排除相同或几乎静止的源帧，discriminability 默认是源图像间平均像素差阈值 2（uint8 范围 0–255），不随目标图像误差增大；重新编码造成的疑似对齐失败仍需人工复核。静止画面等情况无法形成有效邻帧比较时显示 VECTOR_OK_ALIGNMENT_UNVERIFIED，表示向量抽样通过、图像对齐未验证，不能将零次对齐比较当成通过。这个工具不生成或转换训练数据。

## 实验记录

历史完整任务全参实验的训练参数、源码 commit、checkpoint 和 W&B 状态见 [历史实验记录](../../../docs/records/历史实验记录.md)。该记录属于原 v3 运行，不代表当前 v1 配置已重新完成训练。LoRA 完整任务 baseline 尚无训练完成记录。
