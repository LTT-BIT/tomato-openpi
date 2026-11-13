# 本仓库开发约定

- 本分支 `tomato-subtask-policy` 用于无 Agent 的整串番茄 A/B 训练；用户明确要求分支不使用 codex 前缀。在现有 `/home/user/tomato-openpi` 开发，27/28 共用 `/home/mengxiang/workspace/code/tomato-openpi-sharpa`，不另建 clone/worktree。本轮只准备、校验和同步，不启动正式训练，不改部署端。
- 用户已授权验证通过后自动中文提交并推送 GitHub 对应分支；本分支推送 `github/tomato-subtask-policy`，不强推。数据和权重不纳入 Git。
- 历史normal93实验固定 `records/normal93_20261109_selection.json`，已有配置、统计、数据和权重保留，不按目录扫描悄悄改变成员。
- 2026-11-11用户新规定：后续π₀.₅训练使用Agent冻结清单中的100条完整七阶段录制，全部100条参加动作训练，不留val/test，归一化重新覆盖全部100条。新增独立complete100 RGB+crop版本，不覆盖旧normal93，不复用93条统计；Agent的84/8/8划分不改。当前只做准备，不启动正式训练，不改部署。
- 本轮默认B主方案、A对照；使用scripts.tomato_ab_run显式check/start，默认只检查，不自动排队或开训。28 GPU被占用时继续完成CPU准备，不停止他人进程，不把CPU权重/图检查声称GPU显存验收。W&B默认offline并保留本地日志。
- 用户要求：每轮开发完成并验证后提交一次 Git，使用中文提交说明。
- 不新增开发日志或过程性文档；改动及验证结果写入 Git 提交说明，保留可核对的实验配置和真实结果记录。
- 训练版本必须记录数据来源、配置名称、实验名称、归一化统计和英文 prompt。
- 更新训练行为时同步维护根 README、多人开发文档或对应任务训练说明；区分准备完成、训练完成和实机验证。
- 只提交本轮及本任务相关内容，保留用户其他未提交修改；不把数据集、权重、密钥提交 Git。
- 训练日志同时保存到本地并启用 W&B；无法联网时明确采用 offline 模式，之后执行 sync。

- 番茄粗阶段当前采用四类动作与四种无标点 prompt 一一对应；两个 return 共用 `Right hand returns the held object to the right plate`。七阶段 PHASES 仅表达进度，不意味着七种不同指令。修改时需同时核对标注端 vocabulary、训练 ACTION_PROMPTS、token预算和阶段 Agent 输出，不能恢复第七段裸杆 prompt 特判。不要自动覆盖旧 checkpoint 或训练副本；新四类 Agent＋状态机属于另一个待实现实验。

- 用户已选27服务器承载本次训练；连接47.116.73.163端口27、mengxiang。27与28共用/mnt/data，修改工程会同步可见。UI和CLI通过training_environment统一固定JAX=cuda、PREALLOCATE=false、MEM_FRACTION=0.90，并经公共run-python预加载NCCL；UI本身与数据检查仍用CPU。本轮只配置并打开27训练UI，不启动训练。

- normal93 A/B现采用手臂14维Delta、手部44维Absolute，原始数据仍为绝对目标；使用独立_delta_30hz统计和_delta配置/实验名，不复用历史Absolute统计/checkpoint。训练数据及其余参数不变。

- 用户已要求部分对齐既有训练：normal93 A/B均为batch512、8000总步数、FSDP8；保留原学习率计划（warmup1000/decay_steps30000）、普通Delta与其统计，不迁移T3额外归一化或噪声。本轮只同步配置和UI，不开训。

- 2026-11-10用户授权阶段指令可控性与局部图充分离线测试，并明确GPU用28、不用27。B训练已完成，checkpoint7999只读。当前仅诊断，不改默认模型/部署、不连接机器人；93条全为训练数据，不能把本次探针结果称为独立泛化或闭环成功率。

- 2026-11-12：100条r11离线四视图通过全量传输哈希核验，使用独立 `scripts.tomato100_run` 配置28服务器；默认check、B主方案。保留不可变candidate数据manifest，通过哈希绑定的训练准备记录显式接纳已复核离线输入，不宣称独立语义100%或在线ROI验证。GPU目标batch容量仍待实测；不自动开训，不覆盖normal93。

- 2026-11-13用户要求：complete100四视图改为batch256、16000步、每4000完成步保存并保留checkpoint（4000/8000/12000/16000）；FSDP8及原学习率不变，实验名新增_b256_s16000_c4000。用户在28空闲后要求开训，执行默认B方案；旧“只准备不启动”是历史限制，本轮授权覆盖该限制。不启动A或修改normal93/部署。
