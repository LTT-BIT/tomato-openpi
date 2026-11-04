# 本仓库开发约定

- 工作目录规则：默认沿用用户指定的现有目录与分支；未经用户明确要求，不新建 worktree、另建 clone、复制工程或迁移工作目录。先检查路径、分支、未提交修改与 `git worktree list`，遇到分支占用先解释原因，不强制检出、重置或删除其他工作区。
- 当前整合分支为 `train-rtc-paper-delta`，工作目录 `/home/user/tomato-openpi`。它整合原 Delta 论文实验与 T3+trainRTC；不要再创建旧分支名称。按用户本轮明确要求接入可选 FLASH 正式训练与统一 UI，但不合并 FLASH 历史报告或诊断权重；保持 main 和 paper-flash-test 分支不变。
- 用户要求：每轮修改完成并验证后自动保存、提交并推送 GitHub 对应分支，无需再次询问；使用中文提交说明，写清改动、原因、测试命令、结果和未验证范围。GitHub 远程为 `github`（`https://github.com/LTT-BIT/tomato-openpi.git`）；推送失败须明确报告，不得声称已同步。
- 不新增开发日志或过程性文档；改动及验证结果写入 Git 提交说明，保留可核对的实验配置和真实结果记录。
- 训练版本必须记录数据来源、配置名称、实验名称、归一化统计和英文 prompt。
- 更新训练行为时同步维护根 README、多人开发文档或对应任务训练说明；区分准备完成、训练完成和实机验证。
- 只提交本轮及本任务相关内容，保留用户其他未提交修改；不把数据集、权重、密钥提交 Git。
- 训练日志同时保存到本地并启用 W&B；无法联网时明确采用 offline 模式，之后执行 sync。

- 28 服务器统一使用 `/home/mengxiang/workspace/code/tomato-openpi-sharpa`，数据父目录 `/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one`（当前七源480集）。部署只在此现有仓库操作，不另开工程；重启 UI 前确认没有活动训练任务。本轮只验收后启动 UI，不自动启动正式训练。另一个旧工程有有效 baseline 和独有配置，不得当作重复缓存删除。
