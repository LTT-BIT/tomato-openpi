# π₀.₅ + 标准 FLASH Action Expert（P1 最小实现）

2026-11-03：按用户新指令，在 `paper-flash-test` 实现固定 K=6 的标准 FLASH。已从 P0 的“设计待实现”推进到离线模型验收；不修改 main 的 `pi0.py`、`pi0_config.py` 或生产 RTC/ROS2。训练结果和限制见 [05](05_training_results.md)，局部表示误差见 [06](06_local_diagnosis.md)。

## 实现架构与文件

`Pi0FlashConfig/Pi0Flash` 继承 main 的 Pi0 配置/模型，保留 SigLIP、PaliGemma/Gemma 与原 Action Expert 的结构及预训练兼容参数。Expert 输入从 50 个离散动作 tokens 改为 **7 个58D系数 tokens**，仍通过原 Gemma suffix attention 访问每一层 prefix KV。物理 `action_horizon=50` 与 `coefficient_tokens=7` 独立管理，不把物理接口改成7步。

| 新增文件 | 已实现内容 |
| --- | --- |
| `src/openpi/models/pi0_flash.py` | K6、58D Absolute 配置校验；真实 VLM prefix/cache；adaRMS 时间条件；history source；FM/consistency；单步采样及物理 `(B,50,58)` 输出 |
| `src/openpi/shared/flash_polynomial.py` | 官方 Legendre/basis/OLS row scaling、history ridge、双端 KKT、重建、解析速度；float64 预计算，JAX float32 矩阵采用 HIGHEST contraction precision |
| `src/openpi/training/flash_data.py` | 真实缓存指纹/字段校验、canonical 排列、因果实测历史、完整扩展 target、仅训练样本 action/state 分位数统计及哈希 |
| `src/openpi/training/flash_checkpoint.py` | 严格预训练覆盖、VLM 冻结滤镜与 SHA256、独立新目录 Orbax 保存/恢复 |
| `scripts/flash/prepare_smoke_batch.py` | 只读提取真实当前帧三路 RGB、核验 video timestamp、动作/状态 batch、实际 task prompt；不读取未来图像 |
| `scripts/flash/smoke_expert.py` | 完整预训练模型的合成/真实梯度、一次 SGD、冻结检查、采样、计时/显存、完整 checkpoint 往返 |
| `scripts/flash/sample_boundaries.py` | 两个真实相邻 Chunk 分别以各自当前 RGB/history 采样，评估实际共享时刻的预测 gap 和 target C1 效果 |

没有引入 RoboVerse/Isaac Sim 环境，没有从 Delta 分支复制实现，没有自适应阶数、分组高阶或残差专家。

## 官方算法对应

参考官方 commit `f47e490f2b820f835d1b9145da3cdb3df69f50a0`，`flash_policy.py`。算法独立用 NumPy/JAX 实现，测试直接执行官方纯 basis 函数做 parity。

- 执行 ticks `[0,49]`；prefix/suffix各1，外围 fit_pad各1，target 实际 ticks `[-2,51]` 共54行；锚点为 tick0和50（扩展矩阵行2和52）。真实两端中心差分用于 **target 构造**。
- 动作先使用 main 的 q01/q99 affine（epsilon=1e-6，不clip），再 OLS 拟合，除以 OLS 逆矩阵各行 norm。历史 measured joint position 采用 **action** 统计，而非 state 统计。
- 严格复现官方 prefix=1 的 history 参数化：真实八帧 `[-7,0]` 实测位置拟合到目标中间 grid 首8行 `[-1,6]`；这是 source 的时间重参数化，不是看到未来六帧。ridge=0.1。
- 训练 source=`C_h+0.5*noise`，t~Uniform[0,1]；`C_t=(1-t)*source+t*C_target`。FM MSE 回归 `C_target-source`；另外执行 t=0 forward，consistency MSE 监督 `source+v(source,0)`；权重1。
- 推理无历史扰动：`C_pred=C_h+v(C_h,0,prefix_KV)`。重建并反归一化为物理 absolute joint radians `(B,50,58)`。
- KKT只修正训练 target。官方本身没有在 inference 用未来 endpoint 真值投影；本实现同样不这样做，也不声称未训练预测保证C1。

## 信息泄漏与历史一致性

`make_batch` 取八帧 **实测 state position**，只使用当前行及过去七行；原116D中的velocity不进入history。两级扩展未来动作仅作为 `target_ext`。模型的 `sample_actions(rng,observation,*,history)` 根本没有 target 参数。

训练与部署共用同一个 `fit_history(normalize(history))`。没有用上一块预测动作替代实测历史，没有用 action labels 伪造 history。缺少历史时明确拒绝：不静默补未来值，也不让旧 `(obs,actions)` loader 假装已支持 FLASH。

真实数据测试修改当前之后的所有 action labels 与 measured states，history和source不变；当前图像从同一Episode的当前 frame解码，timestamp与数值行核对。state 的离散 prompt 使用训练样本单独拟合的 state q01/q99；action/history使用训练 action q01/q99。

## 权重与冻结

实际加载 `/home/user/checkpoints/pi05_base/params` 完整原生 Orbax 权重。只允许32D→58D action input/output projection的形状变化，其他缺失/额外/形状不符全部报错。

冻结 vision 与原 Gemma 的 **2,923,335,408** 参数；`*_1` Expert、时间MLP与action投影可训练，共 **430,151,738** 参数。VLM采用原始float32参数保存，不通过降精度改写权重；Transformer算术保留原bfloat16配置。加载前后与实际SGD前后的VLM内容SHA256均一致。

每个观察只编码一次prefix，FM与consistency使用同一cache；cache stop-gradient且冻结VLM。更新Expert后缓存仍有效。推理缓存/完整prefix结果相等的测试与数据层未来扰动测试分别记录，不能混称。

## FullFT 可选训练模式

`Pi0FlashConfig(freeze_vlm=False)` 支持与原 π₀.₅ FullFT 相同的可训练范围：视觉、语言、Expert、时间 MLP 与动作投影，共 3,353,487,146 参数、51 个参数叶。默认仍为 `True`，用于复现历史冻结实验（VLM 32 叶冻结、Expert 等19叶可训练）。上面的冻结说明记录 P1/P2/P3 的实验状态。

`src/openpi/training/flash_train.py` 统一构造 AdamW 与梯度过滤器。每步在 `nnx.value_and_grad` 内调用 `loss_components`，当前 RGB/语言 prefix 与 KV 因而保留回传路径，FM 与 Consistency 共同向 VLM 回传。FullFT 不能沿用历史脚本预先编码的跨更新缓存，也不能只改 optimizer filter；旧 P2/P3 脚本继续显式使用冻结配置。训练条件仍只有当前 Observation 和八帧实测历史，54 帧扩展 Action 只进入监督目标，输出保持 `(B,50,58)`。

每次更新前检查模型与配置模式、优化器参数覆盖、Loss/Gradient 是否有限；非有限值不会推进 Adam 或修改参数。FullFT checkpoint 必须包含更新后的视觉与语言参数，禁止复用 P3 的 Expert-only 增量格式；新的冒烟入口可按需保存一份完整模型与 Adam，默认不保存大型权重。

`scripts/flash/train_fullft.py` 是 **fresh pi05_base、Step 0 的 1–20 步冒烟入口**，不从 Step 251 继续。复用 normal 清单、源级 Split 和训练集归一化，记录训练窗口、英文 prompt、种子与 `fold_in(key(42002), update_index)` 策略；augmentation/flow 在 loss 内拆分随机数。使用 baseline AdamW（β=.9/.95，eps=1e-8，wd=1e-10，clip=1）与 cosine schedule（warmup=1000，peak=2.5e-5，end=2.5e-6，30000步），未沿用 P2 的恒定学习率。初始 Batch=1，无梯度累积；这与正式 baseline 的 Batch=64 还不构成公平训练预算。日志写本地及 W&B offline，执行后需同步对应 W&B run。

显式运行示例（本轮未执行真实数据 FullFT 更新；需先验证可用显存）：

```bash
PYTHONPATH=src:packages/openpi-client/src:. XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/user/sharpa-pi05/.venv/bin/python scripts/flash/train_fullft.py \
  --data .cache/flash_p2_data --images .cache/flash_p2_images \
  --pretrained /home/user/checkpoints/pi05_base/params \
  --output .cache/flash_fullft_smoke_001 --steps 1
```

新增 `scripts/flash/test_fullft.py` 使用真实 dummy Gemma 注意力与小型替代视觉编码器验证：两步后视觉/语言/Expert/时间投影均实际更新；冻结模式 VLM 不变；保存全部参数及 Adam 后，以相同随机数恢复下一步结果一致；采样 `(1,50,58)` 有限；NaN 拒绝更新；误用 Expert-only optimizer 被拒绝。独立的正式尺寸抽象模型检查确认全部51叶参数纳入 FullFT。小模型一致性不能外推为完整模型跨进程或跨 GPU 的 bitwise 一致性；正式尺寸真实数据梯度、显存与学习曲线仍待28服务器验证。

回归命令：

```bash
PYTHONPATH=src:packages/openpi-client/src:. JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
  /home/user/sharpa-pi05/.venv/bin/python -m pytest scripts/flash/test_*.py -q
```

验证结果：上述完整回归 **38 passed**，232 条 Flax/JAX 兼容性弃用警告；Ruff check、format check、`git diff --check` 和冒烟入口 `--help` 均通过。未执行正式尺寸真实数据 FullFT 更新，没有新增或覆盖 P2/P3 Checkpoint。

## 历史 P1 训练接口边界

这是可直接调用的离线模型与batch API，尚未把历史字段接到原 `scripts/train.py` 的 `(obs,actions)` 默认loader，也没有注册正式 E0/E1/E2 长训练配置。已具备受控小规模离线训练的模型条件；下一阶段应接专用历史batch训练入口、封存按原始Episode分组的数据划分和训练预算，再运行20–100步验证。

P0.5 的急变局部误差不会因为模型forward通过而消失。标准六阶版本完成后，可另开消融比较更短分段或残差表示；本轮没有实现这些改进，也没有通过删除/平滑关键失败样本美化结果。
