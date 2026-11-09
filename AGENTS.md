# 本仓库开发约定

- 本分支 `tomato-subtasks-ab-train` 用于无 Agent 的整串番茄 A/B 训练；用户明确要求分支不使用 codex 前缀。在现有 `/home/user/tomato-openpi` 开发，27/28 共用 `/home/mengxiang/workspace/code/tomato-openpi-sharpa`，不另建 clone/worktree。本轮只准备、校验和同步，不启动正式训练，不改部署端。
- 用户已授权验证通过后自动中文提交并推送 GitHub 对应分支；本分支推送 `github/tomato-subtasks-ab-train`，不强推。数据和权重不纳入 Git。
- 本实验固定 `records/normal93_20261109_selection.json`：采集 A/B 组共93条完整七段 normal 轨迹，不用 speedup、C、三段轨迹和已剔除无标注条目。新清单必须新建实验版本；不按目录扫描悄悄改变训练集。
- 用户最新要求：A/B动作模型按正常全数据方式训练，93条全部参与，不留val/test；归一化覆盖全部93条。不能自行新增划分或更改既定训练参数。旧75/9/9统计仅保留为历史，新all_sources_v2身份禁止复用旧统计。
- 本轮默认B主方案、A对照；使用scripts.tomato_ab_run显式check/start，默认只检查，不自动排队或开训。28 GPU被占用时继续完成CPU准备，不停止他人进程，不把CPU权重/图检查声称GPU显存验收。W&B默认offline并保留本地日志。
- 用户要求：每轮开发完成并验证后提交一次 Git，使用中文提交说明。
- 不新增开发日志或过程性文档；改动及验证结果写入 Git 提交说明，保留可核对的实验配置和真实结果记录。
- 训练版本必须记录数据来源、配置名称、实验名称、归一化统计和英文 prompt。
- 更新训练行为时同步维护根 README、多人开发文档或对应任务训练说明；区分准备完成、训练完成和实机验证。
- 只提交本轮及本任务相关内容，保留用户其他未提交修改；不把数据集、权重、密钥提交 Git。
- 训练日志同时保存到本地并启用 W&B；无法联网时明确采用 offline 模式，之后执行 sync。

- 番茄粗阶段当前采用四类动作与四种无标点 prompt 一一对应；两个 return 共用 `Right hand returns the held object to the right plate`。七阶段 PHASES 仅表达进度，不意味着七种不同指令。修改时需同时核对标注端 vocabulary、训练 ACTION_PROMPTS、token预算和阶段 Agent 输出，不能恢复第七段裸杆 prompt 特判。不要自动覆盖旧 checkpoint 或训练副本；新四类 Agent＋状态机属于另一个待实现实验。

- 用户已选27服务器承载本次训练；连接47.116.73.163端口27、mengxiang。27与28共用/mnt/data，修改工程会同步可见。UI和CLI通过training_environment统一固定JAX=cuda、PREALLOCATE=false、MEM_FRACTION=0.90，并经公共run-python预加载NCCL；UI本身与数据检查仍用CPU。本轮只配置并打开27训练UI，不启动训练。
