# main Baseline 审计

审计日期：2026-11-02（Asia/Shanghai）。范围仅为 `origin/main` 的实际文件。

## Git 隔离与来源

- 原目录 `/home/user/tomato-openpi`：分支 `delta-paper-training-test`，HEAD `2ab68c8b8090284fca6d258f86bf5098d2f1302b`；建立 Worktree 前 `git status --porcelain=v1` 为空。
- `origin` 为 `ssh://git@gitlab.litchibot.top:2222/fruit-train/tomato-openpi.git`；`github` 为用户提供的 GitHub 仓库。严格使用指定的 **origin/main**。
- 执行 `git fetch origin main` 后 `origin/main=897cb917097c1d56a3fa7a517bf561b9e878036e`。
- 创建前 `paper-flash-test`、目标目录及同路径 Worktree 均不存在。
- 执行 `git worktree add -b paper-flash-test /home/user/tomato-openpi-paper-flash origin/main`。
- 创建后 HEAD 与 origin/main 相同，工作区干净。后续实验提交仅在该分支增加。
- 没有 merge/cherry-pick Delta 分支。原目录文件及分支不变；fetch 和 Worktree 注册会更新共享 Git 元数据，这是预期行为。
- 本轮不 Push。最终提交号由 Git 历史和交付消息记录，避免文档自引用。

## 已有模型与数据链路

| 项目 | main 实际实现 |
| --- | --- |
| VLM | `src/openpi/models/pi0.py`：SigLIP `So400m/14` + PaliGemma/Gemma 2B；Gemma 宽 2048、18 层 |
| Expert | 同文件 `PaliGemma.llm` 的第二组 Gemma 参数（`*_1`），`gemma_300m`，宽 1024、18 层；不是单独的外接 DiT |
| 信息交互 | 图像/语言组成 prefix；含状态的离散 prompt 用于 π₀.₅。Action suffix 可注意 prefix，各层共享兼容的 KV/head 结构；prefix 不注意 suffix；推理缓存 prefix KV |
| π₀.₅ 时间条件 | `time_mlp_in/out` → Action Expert 的 adaRMSNorm |
| 物理接口 | Sharpa recipe 明确 `action_dim=58`、`action_horizon=50`，30 Hz；不是默认 Pi0Config 的 32D |
| 版本配置 | `training_configs/recipes/common.py`：π₀.₅、bfloat16、256 prompt tokens、RTC delay 默认 0 |
| LoRA | `pi05_lora_v1.py`：VLM rank 16、Expert rank 32，视觉和投影仍可训练 |
| Full FT | `pi05_fullft_v1.py`：全参数、batch 64；本轮未执行 |
| 归一化 | `src/openpi/training/config.py` 对 PI05 使用 quantile；`transforms.py`：`2*(x-q01)/(q99-q01+1e-6)-1`，该函数不裁剪 |
| 当前资产状态 | 实例化 Sharpa 官方配置确认 `use_quantile_norm=True`，但本 Worktree 对应资产未提供，`norm_stats=None`。不能据此启动正式训练 |
| 动作表示 | Sharpa 两个 embodiment 均 `extra_delta_transform=False`，commanded absolute joint target、rad |
| 权重 | `PartialCheckpointWeightLoader`；recipe 预训练路径 `/mnt/data/public_ckpt/pi05/pi05_base/params`，本轮未加载或改写权重 |

源代码：`src/openpi/models/pi0_config.py`、`gemma.py`、`pi0.py`，`src/openpi/training/{config,data_loader,weight_loaders}.py`，`src/openpi/transforms.py`。

### 58D 不等于相同排列

模型 canonical 顺序由 `src/openpi/policies/fr3_sharpa_protocol.py` 固定：左臂 7 + 左手 22 + 右臂 7 + 右手 22。双臂索引为 `[0:7,29:36]`，双手为 `[7:29,36:58]`。

converter 的 action 存储为左臂 7 + 右臂 7 + 左手 22 + 右手 22；state 为 116D，各部位 position/velocity 分块。`fr3_sharpa_multimodal_policy.py` 将实测 position 提取并映射为 canonical 58D，action 同样重排。深度、触觉、速度等虽有存储，当前 RGB Baseline **不将它们送入 VLM**。

P0 按 metadata 中的关节名称独立建立映射，并验证 58 个名称与 main 的 ORDER 完全相符；没有借用 Delta 实验代码。

### Flow 与 Checkpoint

训练：采样 `t~Beta(1.5,1)*0.999+0.001`；`x_t=t*noise+(1-t)*actions`；目标速度 `noise-actions`；投影 Expert suffix 输出并在动作维平均 MSE。RTC 可选功能已在 main 中，但本轮不启用、不改动。

推理：默认 10 Euler steps，从 t=1 积分到 0；prefix KV 只建立一次；最终 `(B,50,58)`。

Partial loader 只允许指定投影的 shape mismatch 保留随机初始化，其余不匹配报错。新建系数模块若出现 checkpoint 中不存在的新参数，不能假设现有 loader 会全部自动补齐，必须专项测试。

### 数据与字段

官方 main Sharpa 配置默认数据是旧封存批次 `/mnt/data/users/tairan/workspace/data/10.16_2_data_lerobot`（可由环境变量覆盖），并非本次指定的新 normal/speedup 根目录。E0/E1/E2 必须统一重新配置来源，不能混用旧统计。

LeRobot loader 使用 `action` 字段，未来 offset 按 source FPS 与 action_hz 计算。字段包含 `observation.state`、三路 `observation.images.cam0/cam1/cam2`、action、timestamp、frame/episode/task index；main 相机角色为 base=cam0、left wrist=cam2、right wrist=cam1。

英文任务 prompt 示例：

> Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.

注意 OpenPI `DeltaActions` 是相对于当前 state 的差值，恢复时加同一个 state；它不是逐帧差分，不能 cumsum。本轮真实 action 是绝对目标，没有应用 Delta 转换。
