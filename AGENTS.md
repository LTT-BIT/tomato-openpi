# 本仓库开发约定

- 工作目录规则：默认在用户指定的现有目录和分支中继续开发。未经用户明确要求，不新建 Git worktree、不另建 clone 或复制工程、不迁移工作目录；不能仅以隔离实验或方便开发为由另开目录。
- 开始工作前核对当前路径、分支、未提交修改及 `git worktree list`。分支已在其他 worktree 检出时，先说明占用路径和原因，不自行另建目录，也不强制检出、重置或删除现有 worktree。涉及搬回原目录或清理 worktree 时，先确认用户指定的目标目录，并保全未提交修改、实验结果和 Checkpoint。
- 当前工作区：`paper-flash-test` 位于唯一 Git 工作区 `/home/user/tomato-openpi`，后续在此目录开发和切换分支。独立 FLASH worktree 及旧路径 `/home/user/tomato-openpi-paper-flash` 的兼容软链接均已移除，不得重新创建。
- FLASH 当前训练起点：按用户要求已清理本机 P1/P2/P3 实验权重、Step100/250 优化器状态及中断保存文件；报告和数据保留为历史记录，不再声称这些 Checkpoint 本机可恢复。后续正式训练从 `pi05_base`、全新优化器和 Global Step=0 开始，不自动接续旧诊断步数。基础权重、真实数据、Split 和 Normalization 不属于清理对象；以后按实际恢复需要制定 Checkpoint 保留策略，避免反复保存多份未变化的完整权重。
- 用户要求：以后每轮修改完成并验证后，自动保存文件、提交 Git，并推送到 GitHub 对应分支完成同步，无需再次询问；用户当轮另有明确要求时以当轮要求为准。
- Git 提交说明使用中文，明确写清修改了哪些内容、修改原因及验证命令和结果；未完成或未验证的部分如实备注。
- 本仓库 GitHub 远程为 `github`（`https://github.com/LTT-BIT/tomato-openpi.git`）；当前 FLASH 实验推送到 `paper-flash-test`。推送失败时保留本地提交，明确报告失败原因及尚未同步的状态，不得声称已上传。
- 不新增开发日志或过程性文档；改动及验证结果写入 Git 提交说明，保留可核对的实验配置和真实结果记录。
- 训练版本必须记录数据来源、配置名称、实验名称、归一化统计和英文 prompt。
- 更新训练行为时同步维护根 README、多人开发文档或对应任务训练说明；区分准备完成、训练完成和实机验证。
- 只提交本轮及本任务相关内容，保留用户其他未提交修改；不把数据集、权重、密钥提交 Git。
- 训练日志同时保存到本地并启用 W&B；无法联网时明确采用 offline 模式，之后执行 sync。

- FLASH 正式训练适配只在 `paper-flash-test` 修改；`train-rtc-paper-delta` 只保留 Delta/T3/RTC。统一 UI 的接入不构成跨分支混入算法的授权。每次编辑前核对 `git branch --show-current`，不以当前检出分支代替任务指定分支。
- 28 服务器沿用 `/home/mengxiang/workspace/code/tomato-openpi-sharpa`，数据父目录 `/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one`。不另开工程、不清理其他仓库的真实 checkpoint；本轮验收后启动 UI，不自动启动正式训练。

- 用户最新要求：正式 FLASH 与 π05 对照沿用昨天 T3 的全数据方式，normal480 全部来源参与训练、完整合法窗口等权 epoch_shuffle，不强制沿用 P 系列的独立验证源。旧划分资产保留，新建 all_sources 统计资产；训练内监测不得称为独立验证或泛化结果。
