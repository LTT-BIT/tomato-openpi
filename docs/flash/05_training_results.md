# 标准 FLASH P1 离线验证结果

2026-11-03，实验名 `flash_p1_standard_k6_offline_smoke`。当前完成固定 K=6 标准 Expert 的机械验收：**合成/真实三路RGB＋LeRobot动作与实测历史 Forward/Backward、一次SGD、完整权重保存恢复、离线采样均通过**。没有启动完整训练，没有实机执行。P0阶段结果保留在03，P0.5局部误差见06。

## 数据、配置与权重

- 基线 `origin/main=897cb917097c1d56a3fa7a517bf561b9e878036e`；本轮在既有 `paper-flash-test` 上开发，不合并Delta分支。
- 模型 `Pi0FlashConfig`：π₀.₅、Absolute `(50,58)`、7个系数token、history8、ridge0.1、history噪声0.5（仅训练）、consistency权重1、NFE1、无RTC/LoRA。
- 实际加载 `/home/user/checkpoints/pi05_base/params` 完整原生 Orbax；48个leaf加载、3个32D→58D投影leaf重新初始化；所有VLM参数完整覆盖。缺失、额外、非白名单shape mismatch均拒绝。
- 冻结VLM `2,923,335,408` 参数；可训练Expert/时间MLP/投影 `430,151,738` 参数。权重保存float32，Transformer算术bfloat16。
- 训练统计只使用 normal `A/1022_54_120_normal_lerobot` 首/中/末3个抽样Episode，共 `3749` 行；held-out根为normal `B/1022_56_106_normal_lerobot`。不混用raw/normal/speedup。这里只检查机械split，不是正式benchmark划分，也没有跑held-out validation loss。
- 真实batch sample `8fc6041fb6830400`，发布ep0，current frame302，history rows295..302，target context rows300..353。第二次因果采样 current frame352，history345..352。各自三路当前RGB来自视频帧，核查timestamp，未来RGB不输入模型。
- action与state分别fit q01/q99，系数拟合/history共用action统计，不clip。统计 SHA256 `6c97c5918c6dc657d94d4c2d06fd8fdbd2f855c6b00273164b81309fb133aca3`；完整统计、样本fingerprint、源路径及视频PTS见机读结果。

英文任务prompt来自真实 `meta/tasks.jsonl`：

> Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.

## Forward / Backward 与损失

| Batch | FM Loss | Consistency Loss | Total Loss | 梯度 |
| --- | ---:| ---:| ---:| --- |
| 合成batch | 0.42880195 | 1.38876140 | 1.81756330 | 有限 |
| 真实RGB/LeRobot batch | 2.74194098 | 3.62039518 | 6.36233616 | 有限 |

这是**归一化系数空间**的MSE，不能直接与main离散动作FM loss比较。真实梯度范数 `19.35956383`；仅执行 **1步** clipped SGD（学习率1e-5、global norm clip1），用于确认可更新Expert且不动VLM。没有训练曲线、收敛或任务成功率结论。prefix不做随机增强，这次验证不评价增强策略。

VLM加载及更新前后SHA256完全相同：`de5ce5077e3f8e7f6dc08bec28893027c9a3aa486ff4dc4d43218a24f7f33324`。可训练参数摘要变化；两组梯度及采样均未发现NaN/Inf。数据准备拒绝非有限输入。未来action/state扰动测试在CPU数学套件覆盖；cached-vs-full-prefix测试是独立的缓存一致性检查，不能混称。

## 采样、显存与耗时

- 输出 `(1, 50, 58)`，物理absolute radians，有限；最大绝对值 `3.495307` rad。只做接口和数值验收，不表示动作已学好。
- 同一真实观察缓存KV推理与完整prefix推理最大差 `0.0`。
- RTX5090 32GB、B=1、三路224×224、256 prompt slots、NFE=1，warm-up后5次同步计时：缓存Expert中位数 **2.523ms**；包含SigLIP/VLM prefix的模型侧端到端 **43.417ms**。不含视频解码、文件读取和tokenizer；没有跑E0延迟，因此不声称相比baseline加速多少。
- JAX peak bytes-in-use `16,962,903,296` bytes（**15.798GiB**）；同次nvidia-smi总占用约18.1GB，包括桌面与allocator池，口径不同。

首次不donate的NNX JIT在prefix阶段GPU OOM；修正为模型state缓冲区donation，避免同一大权重输入/输出重复驻留。Checkpoint恢复先释放旧GPU参数再上传，避免两份完整VLM峰值。不修改模型数学算法来绕开显存问题。

## Checkpoint 往返

最终新目录 `/home/user/tomato-openpi-paper-flash/.cache/flash_p1_checkpoint_20261103_r3`；此前中断产生的目录原样保留，未覆盖历史checkpoint。约12GB完整参数以标准Orbax保存；sidecar含config、normalization、step1、无momentum SGD身份、prompt及batch来源。该机械optimizer没有momentum状态；正式训练应使用现有训练checkpoint机制保存优化器/数据采样状态。

完整参数保存/恢复SHA256一致：`2cc9a3f149d1620089fe656bd490dc4ccfc8393b6235782ea903ba8a241665c3`。恢复后重新构建模型并实际采样，最大差 **0.0**；不是仅检查文件存在。

## C¹ 实际效果与预测限制

使用同一真实Episode的frame302与352，分别输入各自当前真实RGB、实测历史，比较前一多项式tick50与下一块tick0（同一物理时刻）。未来真值仅在两次预测完成之后用于指标。

- 训练target双端KKT校正的GPU实际position/velocity与GT及相邻gap已测；float64预计算后float32矩阵收缩必须指定 `Precision.HIGHEST`，否则默认GPU低精度float32乘法可产生约1e-4rad锚点残差。
- 最终高精度路径下手部target共享边界position最大误差 `1.192e-07` rad，velocity最大误差 `4.594e-07` rad/s。
- 一步SGD后的**预测**手部position gap最大仍为 `1.141925` rad，velocity gap最大 `17.144300` rad/s。这暴露了未训练模型的真实限制；target C1机制通过，不等于采样已连续。

完整开/关target约束、双臂/双手、真实误差/gap见 `results/p1/causal_boundaries.json`。P0.5全样本已进一步证明oracle C1能改善端点但恶化附近个别手指极值。

## 测试与复现

已有 `/home/user/sharpa-pi05/.venv`，没有安装/升级包。最终CPU套件 **23 passed**（8条现有JAX/Flax deprecation warnings），GPU polynomial套件 **3 passed**；Ruff check/format及git diff --check通过。测试涵盖官方parity、真实action/state映射、因果历史、统计隔离、KKT实际重建、完整冻结分区、严格loader及非覆盖checkpoint。

```bash
cd /home/user/tomato-openpi-paper-flash
PY=/home/user/sharpa-pi05/.venv/bin/python
export PYTHONPATH=src:packages/openpi-client/src:.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# 输出目录必须新建；复跑时更换后缀，禁止覆盖已有checkpoint。
$PY scripts/flash/prepare_smoke_batch.py --frame 302 --output .cache/flash_p1_inputs_REPRO
$PY scripts/flash/prepare_smoke_batch.py --frame 352 --output .cache/flash_p1_inputs_next_REPRO
$PY scripts/flash/smoke_expert.py --inputs .cache/flash_p1_inputs_REPRO \
  --checkpoint .cache/flash_p1_checkpoint_REPRO --output .cache/flash_p1_run_REPRO
$PY scripts/flash/sample_boundaries.py --checkpoint .cache/flash_p1_checkpoint_REPRO \
  --first .cache/flash_p1_inputs_REPRO --next .cache/flash_p1_inputs_next_REPRO \
  --output .cache/flash_p1_run_REPRO/causal_boundaries.json
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 $PY -m pytest -q scripts/flash
$PY -m pytest -q scripts/flash/test_flash_math.py
/home/user/sharpa-pi05/.venv/bin/ruff check scripts/flash src/openpi/models/pi0_flash.py \
  src/openpi/shared/flash_polynomial.py src/openpi/training/flash_data.py src/openpi/training/flash_checkpoint.py
git diff --check
```

W&B显式offline run `q4omb2pj`，本地保存于 `.cache/flash_p1_run_r3/wandb/`，没有宣称已云同步；后续可 `wandb sync` 对应offline-run目录。本地完整日志、真实batch、图片和约12GB checkpoint留在Git忽略的 `.cache/`；Git只提交代码、统计/索引结果及诊断图，不上传数据集和权重。

## 是否具备小规模训练条件

**模型层面具备受控小规模离线训练条件**：真实prefix/历史、loss/gradient、可训练更新、冻结、恢复、采样链路都已验证。下一步接专用历史batch训练入口并封存原始Episode分组train/validation划分，可先20–100步；不得直接套用只传 `(obs,actions)` 的原loader，模型会明确拒绝缺history的调用。

尚未完成正式E0/E1/E2训练对比、validation action error或收敛评估。P0.5补读源时间戳后确认：speedup与normal在同源时戳下动作标签最大差0.578046rad，且native50步覆盖的原始源时间范围不同；本轮P1只用固定normal版本，不混合这些标签。正式对比前需核对converter取样契约，并按原始Episode分组封存数据。固定K6的手部急变表示误差仍存在。本轮没有全量训练、没有生产RTC/ROS2修改、没有发布机器人动作。标准最小实现完成后才另行设计表示改进消融。
