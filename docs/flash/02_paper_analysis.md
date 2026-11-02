# FLASH 论文与官方实现分析

论文：[FLASH v2](https://arxiv.org/html/2605.15492v2)。官方：[NTUMARS/FLASH-Policy](https://github.com/NTUMARS/FLASH-Policy)。本地参考目录 `/home/user/flash-policy-reference`，commit **f47e490f2b820f835d1b9145da3cdb3df69f50a0**。只读分析；未复制整个仓库，未安装其 simulator 栈。

结论：值得作为 π₀.₅ 的表示与生成方式实验；尚无证据证明它能直接改善 58D 番茄采摘模型。官方以 ResNet-18 多视角 encoder + DiT 为主，不能视为现成的 π₀.₅ Action Expert 插件。论文报告的累计 episode 推理时间不能等同于本项目单次 VLM 调用延迟或端到端控制延迟。

## 算法与对应代码

主要实现：`roboverse_learn/il/policies/flash/flash_policy.py`，Gaussian 对照：`flash_g_policy.py`；配置：`roboverse_learn/il/configs/policy_config/flash.yaml`。

1. **Legendre representation**：在 s∈[0,1] 上使用 P_k(2s−1)；K 表示最高次数，实际 K+1 个系数。官方默认 K=6，即 7 个 coefficient tokens。`_build_legendre_basis` 同时生成一阶、二阶导数。
2. **Fitting/reconstruction**：构造 S，OLS 求 C；`_trajectory_to_coefficients` 与 `_coefficients_to_trajectory` 完成转换。官方在构造函数预计算矩阵；本次 NumPy 实现用 SVD pinv 取得 OLS，避免直接对不必要的正规方程求逆。
3. **Coefficient scaling**：每行 OLS 投影矩阵的 L2 norm 作 scale；标准化后的系数为 C/scale。动作先在共同 action 空间归一化；history 的实测位置也应使用 action normalizer，不能误用独立 state normalizer。
4. **History-Anchored Flow**：`_history_to_coefficients`，在目标 polynomial 的共享网格子集拟合实测 proprioceptive history，Tikhonov λ=0.1。训练对 history coefficients 加 std=0.5 的 Gaussian 扰动，推理默认不加。
5. **Flow loss**：官方 t 从 0 到 1 均匀采样，C_t=(1−t)C_h+tC_target，目标 v=C_target−C_h。另一次 t=0 forward 加 one-step coefficient consistency MSE，权重 1.0；不是只对轨迹做拟合。
6. **单步推理**：`conditional_sample` 的 Euler：C_pred=C_h+v(C_h,0,conditioning)；解码到动作，再取执行区间；默认 NFE=1。π₀.₅ 当前 t 方向相反，移植时必须统一，而不能照搬符号。
7. **Cross-horizon C1**：overlap + outer fit padding；`_apply_boundary_constraint` 通过带 OLS 度量的 KKT 修正，对两端位置和速度设精确锚点。默认 `poly_prefix=1,poly_suffix=1,fit_pad=1,boundary_constraint=true`。

重要限制：官方 `compute_loss` 对 **训练 target coefficients** 应用边界约束；`predict_action` 并没有用未来真值再次投影预测系数。因此训练 target 的 C1 不等于任意相邻模型输出严格 C1。部署算法必须用已知过去边界实现可因果的接续策略，并单独验证误差影响。

## 可复用与应保留部分

| 模块 | 建议 |
| --- | --- |
| Basis/derivative、OLS、row scaling | 可独立移植；本轮已实现纯 NumPy P0，并与官方纯函数做数值对照 |
| History ridge fit、source construction | E2 单独实现，严格按时间和 action 坐标系对齐 |
| FM + consistency objectives | 可移植数学定义；保留 VLM conditioning，记录实际 NFE 和训练额外 forward 成本 |
| 两级扩展/KKT | 可用作 target 构造及约束消融；预测时不能使用未来真值 |
| ResNet encoder / 完整 RoboVerse DiT | 不整体替换当前 VLM；优先利用现有 Gemma Expert attention 接口 |
| 速度 feed-forward、torque 控制栈 | 仅离线导数研究；本轮不改生产 RTC/ROS2，不发布动作 |

## 依赖与范围

官方仓库还依赖 RoboVerse/Isaac Sim/MuJoCo、Torch、Hydra、zarr、diffusers、torchcfm 等。没有必要把这些仿真依赖引入 openpi。算法是确定性矩阵运算 + flow 网络；JAX/NNX 可独立实现。

官方源码为 Apache-2.0，若后续复制具体实现，须保留相应版权及第三方许可。本次 P0 是 NumPy 独立实现，官方 basis 仅通过 AST 提取纯函数做本地测试，不写入训练仓库。

论文 sparse sampling 会扩展物理预测范围；本轮 stride=2/4 **固定 50 个原始 tick 范围，只减少拟合节点**，用于隔离拟合误差。它不复现论文的延长 horizon 加速，后续须另设实验。
