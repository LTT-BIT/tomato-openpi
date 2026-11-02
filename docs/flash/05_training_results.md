# 训练验证状态与后续对照协议

**本轮没有启动 E0/E1/E2 训练，没有新 Expert forward/backward 或训练 smoke test 结果。** P0 数学与实现检查通过，但真实数据的局部急变重建和稀疏振荡问题尚未解决，因此没有把 P0 宣称为质量通过，也没有绕过用户要求的前置验证进入第二阶段。

| 实验 | 实现/训练状态 | Flow loss / validation error | latency / GPU memory / 参数 |
| --- | --- | --- | --- |
| E0 main π₀.₅ | main 审计完成，本轮未训练 | 未测 | 未测 |
| E1 Polynomial | P0 工具完成，Expert 尚未实现 | 未测 | 未测 |
| E2 Polynomial + History-Anchored Flow | 已核对官方目标，尚未实现 | 未测 | 未测 |

P0 重建误差是**真实动作投影的表示误差**，不是模型 validation action error。已离线记录的幅度/速度误差也仅属于重建，不代表学习性能、控制性能或实机安全性。

## 可复核的验证

使用已有解释器 `/home/user/sharpa-pi05/.venv/bin/python`，没有向该环境安装/更新包，也没有系统安装。该环境包含 numpy、pyarrow、matplotlib、torch、jax、flax。P0 不用 GPU。

```bash
cd /home/user/tomato-openpi-paper-flash
PY=/home/user/sharpa-pi05/.venv/bin/python
# 只读抽取；已有 manifest 时拒绝覆盖，重做须使用新的 --output。
$PY scripts/flash/collect_samples.py --output .cache/flash_samples
OPENBLAS_NUM_THREADS=1 $PY scripts/flash/reconstruct.py \
  --samples .cache/flash_samples --output docs/flash/results
PYTHONPATH=src:packages/openpi-client/src:. $PY -m unittest discover -s scripts/flash -p 'test_*.py' -v
/home/user/sharpa-pi05/.venv/bin/ruff check scripts/flash
/home/user/sharpa-pi05/.venv/bin/ruff format --check scripts/flash
git diff --check
```

结果：69 个 Episode / 23 个 dataset roots，2,195 个完整窗口；72 个组合（3 类型 × 4 degree × 3 stride × 2 fitting mode）完成。8 项测试通过（7 项数学/语义检查和 1 项覆盖全部 69 个真实 Episode 的 main adapter 对照），包括官方 basis/一阶二阶导数数值对照。详细结果见 `results/reconstruction.json`，每关节极值见 `results/per_joint_max.csv`。

真实数据对照确认独立名称映射与 main 的 Fr3SharpaMultimodalInputs 在全部抽取 Episode 的 state/action 上一致。数学测试为精确 polynomial recovery/scale round-trip、物理速度单位、C1 约束、固定物理范围的稀疏网格、absolute/state-relative/temporal-delta 区分、无效输入、官方 pure function parity。官方 parity 默认读取参考仓库；其他机器需设置 `FLASH_REFERENCE`，参考仓库缺失时该测试会 skip，不能声称完成官方对照。

Baseline 配置实例化验证命令（未加载权重、未训练）：

```bash
PYTHONPATH=src:packages/openpi-client/src:. $PY - <<'PYCODE'
from pathlib import Path
from training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_lora_v1 import build_config
c = build_config()
d = c.data.create(Path(c.assets_base_dir), c.model)
assert (c.model.action_horizon, c.model.action_dim) == (50, 58)
assert c.model.pi05 and d.use_quantile_norm and not c.data.extra_delta_transform
print(c.name, c.model, c.weight_loader, d.norm_stats is not None)
PYCODE
```

## 下一次小规模训练的固定协议

- 先完成 `04_action_expert_design.md` 的前置条件；config、实验名称、source parent IDs/split、seed、prompt、norm hash、checkpoint hash 一并保存。
- 按原始 parent episode 划分 train/validation；同源 raw/normal/speedup 不能跨 split；不在 validation 上挑 degree 或归一化统计。
- E0/E1/E2 共享 batch、数据抽样顺序、优化步数、学习率、冻结策略。另报告 GPU 时间与额外 consistency forward 成本，不能把相同步数等同相同计算量。
- 第一轮建议仅 20–100 optimizer steps 的离线诊断，不进行全量训练；先验证实际真实样本 loss 有限、梯度有限且更新新 Expert，固定 VLM 权重确实不变，保存与恢复可用。
- 汇总各自 FM/consistency loss、共同物理 validation action error、幅度/速度误差、实际预测 Chunk 连续性；latency 做 warm-up 后设备同步并报告 NFE、batch、prefix cache 范围；记录 peak GPU memory、总/可训练参数量。
- 训练日志本地保存并启用 W&B；无法联网时显式 `WANDB_MODE=offline`，日后 sync。本轮无训练，所以没有 W&B run。
- 使用全新输出目录，不覆盖历史 Checkpoint。不接入实体机器人或生产 RTC/ROS2。

本轮修改清单：`scripts/flash/{collect_samples,polynomial,reconstruct,test_polynomial,test_real_data}.py`，`docs/flash/01..05` 五份报告、`docs/flash/results/` 真实数值/图表，以及根 README 的实验入口。真实抽取数据仅留在 Git 忽略的 `.cache/flash_samples/`，不提交数据集或权重。
