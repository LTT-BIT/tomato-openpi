# 四类 V-JEPA 2 到 OpenPI 的配置与接口

仅提供阶段结果校验、关节重排、保守七阶段Prompt provider和Chunk元信息接口。默认配置 `configs/tomato_vjepa_agent.json` 中 enabled=false、robot_control_enabled=false、checkpoint=null。不加载Agent到旧 `PhaseModel`：旧类只接收七分类checkpoint，本次Agent为四类current/future模型。

V-JEPA独立训练/推理说明位于 `/home/user/vjepa2_training/docs/TOMATO_LOCAL_AGENT.md`。两模型不联合反向传播。本接口未改VLM/Action Expert、58D动作、50步窗口、A/B训练、Delta/Absolute、归一化、RTC或执行队列。

## 协议与状态

协议 `tomato-stage-prediction-v1`。固定labels为pickup/harvest/return/adjust，响应同时带head logits和概率、预测模式、未来时距、有效标志、模型版本、session/observation ID及输入/结果Unix时间戳。具体字段由 `openpi.subtasks.agent_bridge.validate_prediction` 校验；七维、NaN、错误类别顺序、概率与logits不一致、错ID及过期时间戳被拒绝。

三路cam0/cam1/cam2，16帧10Hz因果历史；机器人状态只用实测58个位置。OpenPI排列是左臂+左手、右臂+右手；Agent排列是左臂、右臂、左手、右手。必须使用 `named_agent_history` 按名称重排，不能直接拼接。

```python
from openpi.subtasks.agent_bridge import PredictionPromptProvider, bind_chunk

provider = PredictionPromptProvider(threshold=.8, confirmations=2, max_age_s=2.)
provider.reset("human-started-session")
# response来自独立Agent进程，transport/timeout由调用方提供。
context = provider.accept_prediction(response, observation_id=expected_id, now=unix_now)
request_prompt = context.prompt
chunk_metadata = bind_chunk(context, chunk_id)
```

初始化明确假定人工从pickup_1启动，每次重置须使用新的session ID，防止旧任务Chunk混入。按合法有序后继推进：pickup_1、harvest_1、return_truss、adjust、pickup_2、harvest_2、return_stem。动作指令严格复用 `ACTION_PROMPTS`；两次return均为held object，无第七段裸杆特判。必须连续确认当前头的下一类；中断、其他类别和过长观测间隔打断确认。未来头仅预警，不能单独提前更新prompt。默认2Hz下两次连续确认最少跨0.5s，另有采样/计算延迟，阈值并未优化。

四类观察不能证明中途重启时处于哪一轮；接口拒绝按任意类别猜测任务进度。最后阶段始终task_complete=false，放回完成与任务成功需要独立可靠事件。Future-only模型不能驱动当前Prompt。

## 推理与动作边界

`bind_chunk` 为每次调用记录phase、prompt、版本、session、模型版本、observation ID和输入时刻。`chunk_matches_prompt` 判断结果是否仍对应当前上下文。版本在有序阶段变化时增加，即使两个阶段文本相同也能区分。

此接口不自动提交新指令、不清空旧动作队列、不丢弃已执行/排队动作、不实现受控停机。部署接线时，调用方必须在既有安全重规划边界使用context.prompt，并为请求与返回保存相同metadata；遇到旧版本结果应由执行器现有安全机制处理。不要把比较函数返回false等同于可立即删除动作。

如果服务失败、校验失败或观测过期，调用方须停止提交依赖该结果的新请求并走原执行器故障流程。本轮未修改执行器，也没有验证实机故障处置。真实ROS观测缓存、网络超时客户端、KV/动作缓存失效和RTC旧Chunk协调尚未接线。

## 验证

`tests/test_tomato_agent_bridge.py`：七位置/四文本、重复阶段、非法跳转、未来只预警、版本绑定、错误/过期/乱序响应以及关节重排。Mock均为合成协议数据。

真实三条录制smoke产生的checkpoint也已通过独立Agent进程→真实测试观测→本Provider的离线验证；记录在V-JEPA工作区runs/setup。该验证未运行π0.5动作推理，不属于完整控制闭环或实机验收。
