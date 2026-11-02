# π₀.₅ + FLASH Action Expert 设计（尚未实现）

当前状态：完成 P0；手部急变窗口及稀疏高阶振荡仍未解决，暂未通过进入 Expert 训练的质量门槛。以下为具体拟修改点，不代表已经可运行的功能。

## 最小架构

保持 `Pi0` 的图像编码、语言/状态 prefix 和 Gemma VLM KV 条件传递。Expert 改为接收 K+1 个 58D coefficient tokens，沿用兼容的 300M Gemma Expert，而非把官方 ResNet + DiT 塞到 VLM 后面。新增系数输入/输出投影，输出 C_pred∈R^(B,K+1,58)，固定 basis 重建 `(B,50,58)`，保留对外物理动作接口。

VLM 可复用原有权重，也可以显式冻结保持数值不变。冻结滤镜必须排除 action expert 的 `*_1` 参数，并涵盖视觉和 embedding；不能粗略冻结整个 `llm` 或假定 LoRA recipe 会冻结所有 VLM。初始训练可固定 VLM，比对 optimizer step 前后权重 checksum。是否冻结是实验配置，不改变原有 main 的默认行为。

## 拟新增/适配文件

| 文件 | 具体职责 |
| --- | --- |
| `src/openpi/models/pi0_flash.py`（新增） | 独立 Pi05Flash 模型/配置，重写 coefficient suffix、FM loss、采样与重建；physical horizon=50 和 expert token length=K+1 分开管理 |
| `src/openpi/shared/flash_polynomial.py`（新增） | JAX float32 固定矩阵，构造阶段 float64 稳定求解；独立 test 对齐本轮 NumPy oracle |
| `src/openpi/training/flash_data.py`（新增） | canonical state/action history、完整窗口与 episode 边界掩码、history action-normalization、训练 split 与 padding |
| `src/openpi/training/flash_weight_loader.py`（新增） | 复用 VLM/Expert 兼容权重，对新增系数投影明确初始化并打印完整加载覆盖率；任何非白名单缺失报错 |
| `training_configs/experiments/tomato_sharpa/flash/`（新增） | E0/E1/E2 显式配置、唯一输出路径、数据/split/norm hash、seed、NFE、训练预算 |
| `src/openpi/training/data_loader.py`（仅必要时 opt-in） | 供新 data factory 读取过去状态和 target 两级扩展，默认路径不改 |
| `src/openpi/models/model.py`（仅必要时 opt-in） | 支持独立 FlashObservation 的 history 字段；旧 Observation 不应被迫要求 history |
| `tests/test_flash_model.py`（新增） | 实际 prefix-conditioning、forward/backward、history 因果性、权重保持、sample shape、save/load 与数值稳定性 |

本轮上述 Expert 文件都未修改/创建；已实现的文件只有 `scripts/flash/` 下 P0 工具及实验记录。没有从 Delta 分支移植任何实现。

## E1 / E2 的明确分离

E0：main 原始 action FM。E1：同一 VLM/Expert 结构改 coefficient representation，从 Gaussian source 出发；优先保持与 E0 相同 t 分布和 NFE 以隔离表示影响，另做一阶推理消融。E2：共享 E1 的 polynomial、normalizer、target constraints；source 改 history ridge coefficients，并加入 t=0 consistency。E2 用官方 0→1 方向与 uniform t，单独标明这些目标差异。

E2 history 必须来自当前时刻及之前的**实测位置**，而不是未来 ground truth 或假装为实测的上一个预测 chunk。存储 state 116D 中只取 position 58D，按 canonical 顺序，并用 action 统计归一化。历史窗口长度/采样率/共享网格由配置声明；缺历史只能使用明确记录的 episode-start 策略或剔除样本。

完整 C1 target 需要未来 overlap/pad，但不可以把这些值作为 inference 条件。当前模型 Observation 只有当步 state，历史 loader/字段不能省略。当前 action pipeline 是 absolute；如以后添加 state-relative Delta，需要围绕当前实测 state 逆变换后再做物理连续性指标，不能 cumsum。

## 归一化与损失

先用 **training split** 的 action q01/q99 归一化完整动作和实测 history，再拟合系数，最后按 OLS 每行 norm 缩放。固定 basis scale 与数据统计各自保存，不混为一个统计文件。验证 split 不参与 q01/q99 估计，不做未声明 clipping。

E2：source=C_h+σϵ（仅训练），target=C_target，C_t=(1−t)source+t target；L_fm=MSE(v,target−source)；L_cons=MSE(source+v(source,0),target)；L=L_fm+λL_cons。单步 inference 使用无噪 history，C_pred=C_h+v(C_h,0)。σ=0.5、history ridge=0.1、λ=1 为官方起点，必须记录而非视为本任务最优。

不同表示的原始 FM loss 尺度不同，**不能直接拿 E0 action MSE 与 E1/E2 coefficient MSE 排名**。各自记录目标 loss，并用相同反归一化物理动作误差、速度/幅度误差和验证集评价。

## 推进前需要解决

1. 定位的手部 command 阶跃不能由平均 RMSE 掩盖，也不能私自改成 measured-state target。确定可接受误差与表示方案（更短分段、多项式+残差等），保留目标不变的对照。
2. 固定 horizon 的 stride=4 高阶拟合不稳定，先降低 degree/stride 或改变表示，检查所有关节极值。
3. 明确可因果预测连续性机制；oracle target 的零边界误差不能代替预测测试。
4. 统一并封存 E0/E1/E2 的新数据来源、原始 parent episode split、prompt、norm 统计、pretrained checkpoint；normal/speedup 同源 episode 必须分到同一侧，防止数据泄漏。
5. 先真实小 batch forward/backward、sample `(B,50,58)`、checkpoint 迁移/恢复；然后最多小规模训练。未达到这些条件，不启动正式 FLASH 训练。
