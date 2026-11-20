# FR3 + Sharpa 番茄训练

## complete100 RGB＋局部crop：训练准备

### 4000步后切换精确逐帧缓存

2026-11-13实际验收：r2覆盖100条425556帧、400路视频数组（256232425472字节），2000次视频参考像素检查通过；100条共300组完整模型输入（含dtype、四路像素、state/action/token/mask）完全一致。32批读取测试的后24批均值0.1599秒/batch，首批冷启动68.51秒；这是CPU供数速度，不是GPU训练步耗时。143项回归通过，证据为`subtasks/records/complete100_rgb_cache_20261113_validation_r2.json`。初版r1因误用JAX resize被实际输入比较拦下，未启用，已替换为实际训练PIL实现。当前原进程继续运行（抽查step3020），原生对话自动检查`4000-rgb`每10分钟检查完整4000 checkpoint后执行一次受保护接续；尚未切换，续训后的真实GPU速度和完成时间待测。

为解决长GOP的随机解码等待，单独创建`complete100_rgb224_cache_r2`，四路视频顺序解码一次，按原float32/255→adapter uint8→训练实际使用的openpi_client PIL resize_with_pad224路径保存逐帧无损npy。源视频、原数据包及统计只读。每段完整核对帧数/30Hz时间轴、源视频SHA及缓存SHA，并独立随机寻帧核对5个像素参考；最终必须覆盖100条全部四路，缓存文件只读，数据集仍使用原metadata/state/action/prompt。

`PI05_RGB_FRAME_CACHE`显式启用缓存；无此变量时原读取路径不变。`scripts.prepare_rgb_cache_resume`比较全部100条300个实际模型输入（图像、mask、state、动作窗口、tokens逐位一致），再用原seed和原16000步有限采样计划从4000位置实测至少24批。只有稳态读取均值低于5.5秒/批、数据/统计/采样/训练契约完全相同时，才生成哈希绑定迁移报告。原复现manifest和样本索引不覆盖；`PI05_INPUT_EXECUTION_MIGRATION`仅放行报告绑定的读取执行代码与已审阅的UI/离线ROI准备工具代码差异，模型、损失、数据、统计、种子和训练设置改变仍拒绝续训。

`scripts.tomato100_cached_resume check --report <报告>`只检查；`handoff`必须在4000完整提交（含train_state和params）后才能执行。切换仅向原PID575579发送正常退出信号（优先SIGINT；若/proc确认继承SIG_IGN则使用未被忽略的SIGTERM），先核对进程启动身份、命令与GPU独占；未提交checkpoint或发现其他GPU任务就拒绝操作。GPU释放后恢复完整checkpoint、optimizer和step，原采样位置为4000×256；总目标仍16000步，保留4000/8000/12000/16000。新进程自动登记UI实时日志，旧UI记录标为主动切换停止；失败时保留4000，不反复自动开训。吞吐及ETA以缓存验收和续训实测为准，准备缓存期间原训练继续。

### UI查看已启动的CLI训练（2026-11-13）

原训练UI现在支持接入现有CLI运行：在训练服务器运行`bash scripts/python.sh -m scripts.training_ui --attach-manifest <本次运行JSON> --attach-console <本次控制台日志> --attach-pid <训练PID>`。接入前核对PID身份、实验名、日志归属与仓库路径；同一运行重复接入不重复登记。不会重新开训，也不接管训练进程。列表状态根据原运行manifest及进程身份刷新，详情直接读取原实时日志；只读任务不提供UI停止或续训按钮，完成/失败/进程退出均按真实来源显示。已有UI服务更新后需重载服务代码，训练进程不重启。

### 2026-11-13训练设置更新

用户指定complete100 A/B为batch256、16000步、FSDP8，每4000完成步保存并保留一次checkpoint：4000/8000/12000/16000。使用已有ScheduledTrainConfig按完成步数编号，keep_period=4000，避免中间checkpoint被只保留最新的策略删除。原学习率计划warmup1000/decay30000不变；样本总量仍为4096000。新exp_name追加`_b256_s16000_c4000`，与昨日准备名及旧normal93隔离。下面11月12日的batch512/8000与GPU占用结论属于历史记录。

28的8卡已核验空闲，本轮启动默认B；A仅同步配置，不自动排队。命令为`bash scripts/python.sh -m scripts.tomato100_run start --variant b`。保留原四视图、100条数据、统计与英文prompt，无深度或Memory，W&B offline及本地日志。启动及显存实际结果以运行日志为准。

实际B训练已启动，代码提交`34738e9`，训练PID575579，W&B offline。首步loss1.6016/grad_norm1.8457，第10步loss1.6188/grad_norm1.8203，均为有限值；8卡各约66181MiB显存，batch256已经真实执行，无OOM。完整16000步尚未完成；不会自动启动A。启动配置、数据/统计身份、日志及checkpoint完整路径见`subtasks/records/complete100_b256_launch_20261113.json`。控制台为`logs/complete100_b256_s16000_c4000_20261113.console.log`。

### 28服务器100条四视图训练入口（2026-11-12）

当前分支`tomato-subtask-policy`。数据已同步至`/home/mengxiang/workspace/data/sharpa/no-block-more/complete100_rgb_crop_offline_r11`：100条、425556帧，A/B各四路视频，全包1513文件9126087596字节，逐文件SHA256一致。服务器现有normal源视频也与本地300份源视频逐一核对一致；原始数据只读，没有硬链接。传输、来源差异和哈希接纳记录为`subtasks/records/complete100_server28_preparation_20261112_r11.json`。

100条均为normal，不使用speedup或C；90条来自`tomato_subtasks_normal_20261107`，10条来自`tomato_independent10_20261109`。对比前次normal93：共同90条的长度与保存标注哈希一致；移出B/1101_572_601_normal_lerobot的0/1/2，加入11月9日10条。训练帧数410456→425556，统计覆盖全部100条。因此数据成员、统计与第四路视觉均有变化；严格crop消融需另用相同100条的三视图对照。

默认B四阶段英文prompt，A总体目标prompt不变。原三路顺序为cam0全局、cam2左腕、cam1右腕，第四路来自cam0局部。物理输入state58D、动作50×58、手臂14维Delta/手部44维Absolute、30Hz、Gemma2B/300M、全参数微调、base初始化、epoch_shuffle、三个独立seed42/42001/42002、batch512/8000步/FSDP8、warmup1000/decay30000、peak_lr2.5e-5/decay_lr2.5e-6均保留。视觉token768→1024、文本256不变，前缀总预算1024→1280；crop增强不做破坏交互区域的随机几何裁切。没有深度或π₀.₅历史输入。

不可变数据manifest仍记录生成时的`candidate/training_ready=false`。新入口显式加载哈希绑定的准备记录，接纳已完成全帧连续性/像素/几何检查及编码Agent视觉复核的r11离线输入，不更改冻结文件或旧验证结论。直接导入配置仍需显式`PI05_TOMATO100_PREPARATION_REPORT`；缺失、数据哈希变化、传输不完整都会拒绝。接纳范围不包含独立语义准确率100%、在线因果ROI或GPU目标batch容量。r11坐标使用前后帧，像素来自当前帧，部署后续单独处理。

在服务器现有仓库`/home/mengxiang/workspace/code/tomato-openpi-sharpa`运行：

```bash
# 默认只检查；A对照改为 --variant a
bash scripts/python.sh -m scripts.tomato100_run check --variant b
# GPU空闲且容量确认后，显式启动新实验；本次未执行
bash scripts/python.sh -m scripts.tomato100_run start --variant b
```

默认数据路径已固定，可用`--bundle`指定同一哈希数据的其他挂载位置；A/B资产分别为`assets/tomato_complete100_rgb_crop_{a,b}_saved_labels_r1_offline_r11_delta_30hz`。新实验名为`tomato_complete100_{a,b}_crop_delta_r1_offline_r11_s42`，拒绝覆盖或恢复已有checkpoint。W&B默认offline、本地日志保留；`logs/complete100_preflight/`记录最终配置。check可在GPU占用时完成CPU数据和统计验证，显式报告未启动与显存未验收；start在GPU占用时立即拒绝，不排队或停止他人任务。目标batch512的显存不能沿用三视图实验结论，8卡目前被占用，需空闲后实测。

服务器A/B实际预检均通过全部100条、425556帧和归一化哈希；实际训练数据路径各抽查12个样本（共24），四路解码并预处理为224×224、mask全true、state58D、动作50×58及文本256上限均通过。133项回归与6项W&B日志测试通过。服务器GitHub拉取遇代理403，改用SSH传输经验证的Git bundle同步同一已推送提交；没有更改代理。服务器结果及完整快照路径见`subtasks/records/complete100_server28_training_20261112_r11.json`。正式训练未启动，8卡仍占用，GPU容量未验收。

### 当前版本：前后帧辅助修框，全部帧保持局部RGB（2026-11-12）

当前候选为`/home/user/datasets/derived/tomato_complete100_offline_local_candidates_r11`，A/B四视图包为`tomato_complete100_four_view_offline_local_candidate_r11`。100条全部425556帧均从**各自对应时刻的native cam0 RGB**裁剪局部图：全图回退0、黑帧0、缺图0。原三路RGB、实测state、动作、标签和100条成员不改，无深度、无π₀.₅历史token。

这次修复坐标而非替换画面：复用r5当前检测交互框；2625个缺失框通过前后语义手框锚点、相邻RGB双向光流及高相关局部模板跟踪补充。双向轨迹明显不一致或无足够图像证据时不自动认可；剩余266帧的native RGB和候选crop已逐帧查看，记录为编码Agent视觉复核，**不是独立人工真值**。其中52帧补入当前画面可见的右手交互上下文；全部52张修正后的实际解码图再次查看。短缺口两端角色一致时修正丢失期间的短暂选手抖动。没有用阶段标签、prompt或动作命令生成框。

离线窗口提前考虑角色变化：中心可行区间前后传播，限制每轴3px/帧；裁剪边长用对称0.8%衰减包络提前扩大、缓慢收紧，保持220–380px正方形。全帧检查相邻中心位移不超过上一框边长3%、边长变化不超过2%，超限均为0；所有准备交互框完整包含，最大中心单步3.81px。425556帧全部解码并检查时间戳，11691个编码像素核对包含全部原无效帧；没有沿用旧像素或把全图改名局部图。

**这里100%通过的范围是全帧完整性、相对于准备标注的覆盖、裁剪连续性及当前像素来源，不是独立逐帧语义准确率100%。** 这些坐标使用未来帧与整段离线窗口规划，`generation_mode=offline_bidirectional_annotation`、`available_at`标为录制末尾，索引保留原始角色/失败原因、跟踪锚点、复核来源与未来信息使用范围。默认因果索引读取会拒绝它，只有显式离线数据契约允许读取。不能用于宣称在线自动ROI性能或Agent无未来信息的阶段预测；实时部署仍需当前/过去帧定位器并另行验证训练与在线输入差异。

已实际查看跨全部100条的400张视频解码样本、266张异常图和52张修正图，导出第48条原23–32秒的三列对照：native／r6全图回退／r11连续局部。A/B实际LeRobot300样本、全部物理列及600份原三路视频哈希检查通过，A/B共851112个availability和local-valid均true。训练资产后缀`_offline_r11`，归一化数值不变且重新绑定数据身份。167项回归通过。证据见`subtasks/records/complete100_offline_local_20261112_validation_r11.json`。

本节为生成时的候选状态：冻结manifest保留`training_ready=false`。当前训练输入接纳由上节独立哈希准备记录管理；独立语义准确率未测、目标batch512的GPU显存预检待做，本次未启动正式π₀.₅训练或修改部署。视觉1024、文本256、58D state、50×58动作、batch512/8000步/FSDP8和原学习率计划保持。

复做先运行`scripts.prepare_tomato100_offline_roi`生成离线标注，再核对`records/complete100_offline_roi_exception_review_20261112_r10.json`中的真实逐帧复核及其哈希，运行`scripts.export_tomato100_offline_crops`导出局部视频，随后使用`scripts.materialize_tomato100_four_view`和实际加载检查。新的标注版本不能直接套用旧复核记录；导出器必须验证原标注manifest身份、每个未自动通过帧有且仅有一次明确复核，并重新验收全部窗口。

本轮3个失败探针及被替代的r5/r6四视图包和检查记录共7项已移入系统回收站（18032613664字节），可恢复，未永久删除也未释放磁盘空间。当前r11候选及其r10标注、原始三视图基础包、r5定位依赖和r6仅裁剪对照保留。恢复位置见`records/complete100_derived_cleanup_20261112_r3.json`；下文旧四视图路径为历史记录，恢复后才能复做。

### r6对照：每步有图，但局部与全图切换过大（2026-11-12）

用户要求100条全部425556个时间步都有可用图像。新版本将图像可用性和局部定位可靠性分开：`observation.interaction_available`决定第四路image mask，`observation.interaction_valid`仅记录原局部ROI证据。正常帧从当前native RGB按原可靠框裁剪；局部证据不足或窗口尚未覆盖目标时，第四路改为**同一时刻cam0全图**，保持宽高比补边到224。索引记录`image_available`、`local_roi_valid`、`input_mode`、`input_box`及`fallback_reason`。不复用旧画面，不把回退当局部成功，不删时间点，不改原数据。

已完成100条、425556帧全部解码，黑帧0；局部裁剪415987帧，全图回退9569帧。11691个编码像素核对包含全部回退帧；A/B共851112个availability值全true，200份局部视频复制哈希一致，实际LeRobot300样本及600份原三路视频/全部物理列核对通过。155项回归和5项归一化测试通过，准备配置最终10项检查通过。A/B新归一化绑定已验证；具体证据在`subtasks/records/complete100_available_images_20261112_validation_r6.json`。共有2678次局部/全图切换，最长全图回退107帧，不能作为流畅性通过的证据。

新数据位于`/home/user/datasets/derived/tomato_complete100_available_candidates_r6`，A/B包为`tomato_complete100_four_view_available_candidate_r6`；只适用于`current_rgb_local_or_full_fallback_v5`，统计版本后缀为`_available_r6`。准备脚本`scripts.prepare_tomato100_available_crops`复用r5已核验的逐帧定位索引，重新从原始当前RGB渲染全部视频；每帧核验解码、时间戳和非黑图，所有回退帧及每200帧局部图核对重编码像素。原三路视频和物理列由实际LeRobot检查再次核对。

训练入口对新版本要求`full_image_coverage=true`、`image_available_frames=425556`及独立availability列；旧候选继续使用原mask规则，不自动把旧mask置true。`full_roi_coverage=false`可以和100%图像输入同时成立。**全图兜底解决输入缺失，不解决定位失败；局部/全图切换可能产生明显缩放变化，尚未通过连续平滑性与独立语义质量验收。** 因此仍保留`training_ready=false`，不启动正式训练。不能将r5的流畅性结论沿用到r6；后续应改善局部定位并检查切换过渡，使回退尽量少且稳定。

复做命令（输出目录必须不存在）：

```bash
PYTHONPATH=src:packages/openpi-client/src python -m scripts.prepare_tomato100_available_crops \
  --selection training_configs/experiments/tomato_sharpa/subtasks/records/complete100_rgb_crop_20261111_saved_labels_r1.json \
  --parent /home/user/datasets/derived/tomato_complete100_transport_candidates_r5 \
  --out /home/user/datasets/derived/tomato_complete100_available_candidates_r6 --workers 8
```

### r5定位与平滑对照：修复漏手、窗口跟随和篮内红色误判（2026-11-12）

最新100条、425556帧为`/home/user/datasets/derived/tomato_complete100_transport_candidates_r5`，A/B四视图包为`tomato_complete100_four_view_transport_candidate_r5`。原始数据、100条成员和动作监督均不改；无深度、无π₀.₅历史token，未启动正式策略训练或修改部署。

|全100条、同一初版对照|修复前|当前版本|
|---|---:|---:|
|无效局部crop帧|41205|9569（2.25%）|
|所需手部丢失|21116|2625|
|窗口未完整覆盖当前交互区域|20089|6944|
|非harvest阶段只选左手的离线差异提示|514|0|

无效crop减少76.8%；恢复34058个原无效帧，同时2422个原有效帧改为无效，逐帧增退及最长剩余无效段均保存在`tomato_complete100_transport_audit_r5/repair_accounting.json`。最长无效段107帧（约3.57秒），没有把这些帧伪造成有效。差异提示不是人工标注错误数，归零也不等于独立准确率100%。

检测保留原定位器的强结果；补充12个真实失败姿态关键帧（8条录制，编码Agent查看native RGB标注，非独立人工标注），混合原134帧参考训练500步修复定位器。原模型某只手置信度不足0.75时才查询修复模型，补充结果至少0.85，不降低阈值。高置信当前检测可直接确认同一手，减少无必要的光流计算；弱证据仍使用原光流检查和过期机制。

覆盖窗口根据当前/过去运动提前留边，受中心速度/加速度及每帧扩大2%、缩小0.8%的约束，并缓慢回中、延迟收紧。最小/最大窗口仍为原图高55%和短边95%，当前全帧边长中位数326。只提高覆盖率的中间版会把篮内红色误当成左手所持番茄，未通过内容检查；最终版增加手/果共同位移证据（果12px、手8px、至少4帧），平滑手框中心的姿态抖动，禁止用旧位移反复重新确认已建立的持物轨迹。当前附着证据失配6帧会过期。阶段标签、未来图和动作命令均不参与ROI生成。

最终425556帧全部解码、30Hz时间戳及当前像素来源检查通过，黑帧0、大幅中心/缩放跳变0；查看400张跨全部录制的图、49张残余失败/历史误判图和6段对照中点图。19个旧空手误判参考时刻均恢复为有效右手ROI；索引86的31个“左手＋所持果实”参考框仍全部覆盖100%。参考帧参与过方法开发，不能作为独立精度或泛化评估。证据汇总为`subtasks/records/complete100_transport_roi_20261112_validation_r5.json`，151项回归通过。

**训练保留所有100条、全部425556个观测和动作样本**。剩余9569帧只把第四路`interaction_0_rgb`的image mask置false，原三路RGB、58D state和50×58动作监督照常保留；没有删录制、删帧、重复旧像素或关闭这些样本的动作loss。实际LeRobot300样本读取、原始物理向量/600份三路视频哈希及token核对通过；归一化绑定为`assets/tomato_complete100_rgb_crop_{a,b}_saved_labels_r1_transport_r5_delta_30hz`。视觉1024、文本256、batch512/8000步/FSDP8及原学习率计划保持。`training_ready=false`、独立语义质量未认证和目标batch显存待验收的状态继续保留。

新增定位器标注包括Agent原test成员27和66，符合本轮π₀.₅全部100条动作训练的范围，但不能将该定位器作为Agent独立holdout评测的前处理。Agent的84/8/8成员未改，后续Agent对比需使用仅由其train成员训练的定位器。Agent完整历史读出仍未训练。

复做可用`scripts.cache_tomato_coverage_detections`生成已绑定原/修复模型的缓存，`scripts.prepare_tomato100_coverage_crops --cached-detections`导出连续视频，`scripts.consolidate_tomato_stable_shards`汇总；`scripts.audit_tomato_roi_parallel`复用原完整视频/像素检查并按录制并行执行，`scripts.profile_tomato_roi_repair`记录逐帧增退，`scripts.review_tomato_roi_failures`复查剩余失败及历史案例。原图/初版/最终版对照在`tomato_complete100_transport_comparison_r5/comparison30hz.mp4`。

验收新包后，被替代的四视图包及本轮试验版本共13项移入回收站，避免派生目录堆积；当前训练候选、原三视图基础包、旧ROI对照、模型/缓存依赖和误判复查证据保留。逐项恢复位置见`subtasks/records/complete100_derived_cleanup_20261112_r2.json`。下文旧四视图包路径仅是历史记录，需恢复后才能复做；当前入口以本节及准备配置为准。

### 初版对照：实测手指状态＋连续局部视频（2026-11-12）

100条、425556帧已重做为`/home/user/datasets/derived/tomato_complete100_state_guarded_candidates_r1`，四视图A/B包为`tomato_complete100_four_view_state_guarded_candidate_r1`。原始视频、Parquet、标注不改。**本次通过全量视频流畅性/像素来源检查，未独立认证逐帧语义准确率，未开训或改部署。**

2026-11-12整理派生目录：39项已被替代的试验、预览及旧四视图包移入系统回收站，共20361945088字节（约18.96GiB）；未永久删除，尚未释放磁盘空间。目录保留当前完整候选、模型、统计、验收证据，以及当前配置仍引用的人工参考、三视图基础包和复做对照所需的旧ROI基线。保留的4802个文件路径/大小/修改时间摘要及当前验证记录所列SHA均核对不变；100条crop视频、100份索引、A/B各400个视频仍齐全。原始数据及normal93未改。逐项原路径、回收站位置及核验结果见`subtasks/records/complete100_derived_cleanup_20261112_r1.json`。下文历史实验中的旧路径描述当时产物，部分现需从回收站恢复；不作为当前训练入口。

当前ROI链路：当前native RGB检测手部 → 当前帧可验证光流短桥接 → 实测四指MCP屈曲与局部RGB证据共同选择交互对象 → 有速度/加速度约束的裁剪窗口 → 当前帧224 RGB及有效性mask。禁止单凭哪只机械臂动得快就切手。左手单独运送需要测得的抓握姿态和视觉物体证据同时成立；手指弯曲本身不证明持物。两手接近时保留双手交互，已静止的篮内红色背景不能作为新的持物证据。背景仅由当前/过去帧更新，允许短暂到达/放置缓冲；阶段、prompt、未来图、动作命令均不进入ROI推理。

手部原始state列33/37/41/46分别是左index/middle/ring/pinky MCP_FE.position，逐条核对源metadata名称。原始值均值进入阈值0.70、退出0.58，连续3帧确认；只作为姿态约束。校准来自索引86的664帧人工运送区间与901帧左手未参与区间，是弱参考，不是物体持有真值。校准数据、名称、SHA与阈值记录于`subtasks/records/complete100_roi_grasp_state_calibration_20261112_r1.json`。ROI元数据里的`carry_evidence`不能作为Memory的独立抓取成功/任务完成标签。Agent的84/8/8不改；这些动作训练用的ROI标定不能直接作为Agent独立测试结果。

窗口保留当前帧像素，中心有死区与平移/加速度限制，尺寸采用扩大容差、延迟收紧；最小边长为原图高的55%，给快动作留缓冲。1.5倍是目标上下文比例，实际平滑窗口不保证逐帧精确1.5倍，且不保证与人工框逐帧相同。当前观察区域超出窗口时必须标无效，不能为了平滑裁掉操作区域后仍标有效。光流桥接最多5帧且必须通过当前图像前后向/几何检查。

**显示与训练有效性分开**：漏检或窗口未追上时，`display_crop_box`继续采当前RGB，避免黑闪；`valid=false`、训练`crop_box/interaction_box=null`，LeRobot布尔mask=false。没有重复旧图片、伪造成功或删除时间点。当前四视图模型仍是原三路RGB＋一路局部RGB、58D state、50×58动作；无深度，无π₀.₅历史token。

|全100条同口径指标|旧候选|当前版本|
|---|---:|---:|
|相邻框中心跳变超过前框边长25%|3041|0|
|相邻框边长倍率超过1.25|11787|0|
|有效性切换|8918|4724|
|算法有效帧|370399|384351|

425556帧全部真实解码，当前输出黑帧0、30Hz时间戳完整；2825帧与native裁剪像素抽查通过，包含无效mask帧。当前算法有效比例90.32%，不是准确率；无效41205帧继续mask，不自动提升训练质量状态。400张跨100条真实解码图及8段原图/旧/新对照复查，手、果实和接触细节均作为检查对象；仍有514个与非harvest阶段的潜在选手差异，不能直接当作标注错误数或已解决的精度问题。`full_roi_coverage=false`、`training_ready=false`保留，正式入口额外要求显式ready标志。

完整π₀.₅模型额外通过无效crop像素不变性检查：相同state/prompt/noise，mask=false时切换真实图/黑图/随机图，动作最大绝对差0；只证明mask隔离，不是训练效果。

当前单帧ROI（RTX5090，300帧B1）总P50/P95为8.91/11.66ms，不含解码/编码/策略。30Hz离线连续视频不等于完整机器人在线30Hz重规划验收。

|本地派生结果（均在`/home/user/datasets/derived/`）|用途|
|---|---|
|`tomato_complete100_state_guarded_candidates_r1`|全部100条连续视频、mask/坐标、检测缓存、来源回执|
|`tomato_complete100_four_view_state_guarded_candidate_r1`|新A/B四视图候选，原三路媒体独立复制|
|`tomato_complete100_state_guarded_audit_r1`|全量时序/像素与离线语义诊断|
|`tomato_complete100_state_guarded_visual_review_r1`|400张真实解码抽查图及来源索引|
|`tomato_complete100_state_guarded_comparison_r1/comparison30hz.mp4`|原图、旧版、新版并排；仅片段边界有编辑切换|
|`tomato_complete100_four_view_state_guarded_candidate_check_r1.json`|实际LeRobot四视频300样本、原始向量窗口/token核对|

数值统计继续使用同一批100条的图像无关向量统计（非旧93统计），重新生成schema2 A/B绑定，资产为`assets/tomato_complete100_rgb_crop_{a,b}_saved_labels_r1_state_guarded_r1_delta_30hz`。正式A/B入口按stream contract选择独立asset和实验名，不能误用旧裁剪契约。文本256、视觉1024、动作50位置及训练默认batch512/8000步/FSDP8/原学习率保持；新四图数据的训练显存与Agent历史读出仍未验收/训练。

复做入口：`scripts.prepare_tomato100_state_guarded_crops`可使用已核验的`--cached-detections`，每条复核源video/Parquet和缓存SHA；`scripts.consolidate_tomato_stable_shards`核验四个logical shard的25条成员；`scripts.audit_tomato_stable_roi --wait-all`逐条解码验收；`scripts.audit_tomato_roi_semantics`只做QA不改框；`scripts.review_tomato_stable_roi`与`scripts.render_tomato_roi_comparison`导出真实图/视频。汇总为`subtasks/records/complete100_state_guarded_roi_20261112_validation_r1.json`，143项回归通过。

### 2026-11-11原始候选基线（保留作对照）

2026-11-11，冻结100条完整七阶段录制，425556帧、30Hz，**全部100条参加后续π₀.₅动作训练**，不留动作val/test。Agent的84/8/8不改，旧normal93数据、配置、统计、checkpoint保留。本轮完成100条四视图候选数据、A/B新归一化绑定、真实token审计和GPU前反向检查；**ROI语义质量未通过，未启动正式π₀.₅训练、未改部署端**。

有效清单为`subtasks/records/complete100_rgb_crop_20261111_saved_labels_r1.json`，SHA256为`15be3021fd71bf701ada068709138cc2cb56dfa5750cca936db136d8c26c1000`。90条来自`/home/user/datasets/tomato_subtasks_normal_20261107`，10条来自`/home/user/datasets/tomato_independent10_20261109`。训练使用磁盘正式保存且哈希核验的Subtask；8条与Agent临时边界提案的差异已在旧解析记录中保留，不自动改成动作监督。索引86（源B/1106_750_791_normal/episode_000004）的末段return仍为[2908,2990)。原视频、Parquet、metadata和标注只读，派生媒体独立复制，无可写硬链接。

### 输入与ROI规则

保留三路RGB：global=cam0、left_wrist=cam2、right_wrist=cam1，新增从原分辨率cam0裁剪的interaction RGB。π₀.₅四图均224；Agent需要256原图时须重新从native裁剪，不能放大224训练图。实测state转58D，动作窗口50×58，手臂14D Delta、手44D Absolute；RGB-only，无深度，π₀.₅不读Memory历史。

黄色框为参与交互的手、所持物及必要接触区域的紧致并集；青色框为1.5倍上下文正方形，实际输入取青色裁剪。机械臂连杆、无关手及篮内无关果实不应主动加入紧致框。当前自动候选仍有选错手/关联篮中果实的问题，不将几何检查通过当作语义正确。

离线参考`records/complete100_episode86_manual_roi_20261111_r3.json`覆盖索引86的2990帧，134人工目视关键帧、20子段；此前检查248个关键帧/中间帧，不是逐帧独立认证。以这134帧训练了隔离的小型ROI定位器：torchvision FasterRCNN MobileNet320，右手/左手/所持物三类，COCO初始化，1200步batch4，只有颜色增强。**这是实验ROI模型训练，不是π₀.₅正式训练**，未获得独立定位精度或闭环成功率。

`scripts.tomato_roi_locator`逐帧以当前RGB检测手与物体，用当前/过去实测关节变化选择活动手，近距离双手可合并，并用附近红色区域补充候选。推理不读取阶段、prompt、未来帧或动作命令；每帧重新检测，可在丢失后恢复。原始116D实测state、源帧时间、坐标、来源和有效性均可追溯。缺失必要手部时输出`valid=false`、坐标null，视频为黑色224图，布尔mask=false；不删除时间点、不延用过期框。`available_at`表示最新输入时间戳，实际计算延迟另行计量。

该定位器仅由一条录制监督，实测state尺度使用全部100条动作训练统计。未来若用于Agent独立评估，需要另行审计train-only标定和数据划分，不能直接把这里的结果当作84/8/8测试表现。该方法不等于FoveaVLA自动注意方法或性能复现。早期GroundingDINO整臂误检诊断和第93帧失败的LK探针均保留，未作为最终训练ROI。

### 派生数据与实际验收

以下均在`/home/user/datasets/derived/`，原始数据不改：

|目录或文件|内容与状态|
|---|---|
|`tomato_complete100_saved_labels_r1_base_ab`|原三图A/B独立基础副本，100条train=0..99|
|`tomato_roi_locator_experiment_r2`|实验ROI权重、来源和训练回执；native源episode4，冻结清单episode86|
|`tomato_complete100_causal_candidates_r3`|4个shard，全部100条425556帧；224视频、逐帧坐标/有效性、SHA和QA图|
|`tomato_complete100_four_view_candidate_r1`|完整A goal/B coarse四路LeRobot候选包、splits、ROI sidecars，约8.9GB|
|`tomato_complete100_vector_stats_r1`|100条state/50步Delta动作窗口新统计、逐帧真实B文本长度及状态bins|
|`tomato_complete100_goal_token_audit_r1`|同一真实state及统计下的A goal全部425556条文本长度|
|`tomato_complete100_four_view_candidate_check_r2.json`|真实LeRobot四视频解码、300样本、独立原始向量窗口和token核对|
|`tomato_complete100_causal_candidate_audit_r1`|100条300张代表crop、逐条丢失段与角色诊断、2张针对性原图核查|
|`tomato_complete100_four_view_gpu_benchmark_r3`|真实完整模型3/4视图延迟、全参数梯度与crop输入梯度|
|`tomato_complete100_roi_b1_benchmark_r1`|B1定位＋选手＋裁剪P50/P95=4.54/5.90ms，解码编码不计入|

全部425556个帧索引、时间因果性、native紧致框范围、正方形包含关系通过。算法有效370399帧（87.04%）、无效55157帧（12.96%）；**87.04%是启发式有效比例，不是准确率**。4个shard均发布结束，源cam0与Parquet前后SHA一致。

质量审计目视检查300张代表crop，并复核原图：619个非harvest阶段的算法有效帧只选择左手，461帧距离阶段边界不超过30帧；剩余158帧不在该范围。索引70/f3116及15/f4988确认存在左手空手移动、焦点偏向左篮而右侧操作区被遗漏的情况。阶段仅用于离线QA抽样/比较，**未用阶段标签修改ROI**。尚未解决这些错误及丢失段，`crop_manifest.json`保留`full_roi_coverage=false`、`training_ready=false`、模式`causal_prediction_candidate`，正式A/B入口会拒绝。

全100条A/B原物理列保持一致，600份原三路视频副本SHA及原始媒体/标注再次核验通过。真实LeRobot每条取首/中/尾共300个样本，确认四图、224crop、布尔mask、58D state、50×58动作及实际tokenizer一致；62个采样点正确保留无效mask。动作窗口独立从原始Parquet重建，尾部复制末动作，不跨episode，不在阶段边界插入hold。

### 新归一化与token

重新统计全部425556个state和21277800个50步动作窗口向量，使用原有RunningStats（float64/5000直方图bins），q01/q99归一化、不裁剪。新`norm_stats.json` SHA256：`cadfda5b116b16f66a5b739825bd6a16321be2c09b714bef481bf4c19064cf83`，均值/标准差另用独立累加核对。先通过图像无关向量管线求统计，再绑定真实四图数据契约；不存在复用旧93统计的情况。

已生成并通过schema2 provenance校验的独立资产：`/home/user/tomato-openpi/assets/tomato_complete100_rgb_crop_a_saved_labels_r1_delta_30hz`及同名`_b_`目录。A/B数值统计相同，dataset/task-view绑定分别生成；统计绑定通过不代表ROI语义质量通过。数据包`meta/stats.json`只是存储统计，不代替这些模型归一化资产。

|完整真实输入文本（含58D离散state）|P50|P95|最大|超过256|
|---|---:|---:|---:|---:|
|A goal，425556帧|211|220|233|0|
|B coarse，425556帧|209|219|231|0|

保留`max_token_len=256`。三图视觉768→四图1024，新增crop增加256视觉token；文本槽256保持，动作50位置各投影58个连续值。四图prefix共1280槽，加50动作位置共1330；不是把58×50当作文本token。早期合成预算记录保留为历史，不代替本次真实分布。方案一Memory只给Agent；方案二32/64历史token或320文本槽仍仅为待设计候选，未实现/训练。

### GPU检查与正式训练边界

本地RTX5090加载完整公开π₀.₅ base（SigLIP、PaliGemma2B、动作专家）；原32D动作与58D不匹配，仅重新初始化3个输入/输出投影参数，其余加载预训练权重。使用真实三路RGB、人工参考crop与新归一化，B1、10次去噪，预热后各30次：三图P50/P95=54.96/56.22ms，四图=61.58/63.80ms，P50增加12.04%。该结果衡量完整架构延迟，不代表已训练四图58D策略的动作能力，不含相机、ROI、传输或动作队列延迟。

真实归一化50步目标下全参数前反向通过：loss1.8181、全部参数梯度有限；第四图输入梯度非零且有限。参数梯度使用训练增强，图像输入梯度单独关闭HSV增强求导。无optimizer step，未进行正式训练；本地B1 JAX峰值分配约14.23GiB，**不代表batch512/FSDP8显存验收**。r3是有效梯度回执，早期r2的HSV像素梯度诊断失败不作为通过依据。

28目标GPU只读检查时8卡均忙，未停止其他任务或启动远端测试。30Hz是数据/动作采样率；约62ms单次推理已超过33.3ms，不能宣称完整在线链路满足30Hz重规划。50步动作块约1.67秒，动作执行频率、策略重规划频率和ROI更新频率需要部署阶段分别确定。

单帧ROI在RTX5090预热后连续300帧实测：定位P50/P95=3.32/3.92ms，选手与RGB关联=1.14/2.10ms，裁剪=0.05/0.40ms，总计=4.54/5.90ms；263帧有效、37帧输出黑图。输入已预解码，不包含视频导出、相机或网络；与上述策略计时分开测量，不是端到端时延保证。

独立入口`subtasks.complete100_a_crop`与`subtasks.complete100_b_crop`分别使用A全程goal和B四类prompt；默认batch512、8000步、FSDP8、warmup1000/decay30000保持。环境变量分别为`PI05_TOMATO100_CROP_A_BUNDLE`、`PI05_TOMATO100_CROP_BUNDLE`。候选包目前被入口拒绝，不自动排队或开训。Agent联合历史读出、完整16×58历史接入和部署仍是后续工作。

### 重做准备

脚本以`PYTHONPATH=src python -m scripts.<名称>`运行，均显式提供源清单和新派生输出目录，不覆盖原始数据：

- `prepare_tomato100_vectors --selection ... --tokenizer ... --out ...`：全部100条新向量统计与真实B tokens；`audit_tomato100_goal_tokens --help`给出A审计参数。
- `train_tomato_roi_locator --selection ... --annotations ... --out ...`：隔离实验定位器训练，默认1200步，不调用π₀.₅。
- `prepare_tomato100_causal_crops --help`：按明确episodes/shard生成当前帧因果候选，resume必须签名一致；模型/统计/生成器/selector SHA冻结。
- `materialize_tomato100_four_view --selection ... --base ... --candidates ... --out ...`：完整四视图A/B候选；`check_tomato100_four_view_candidate --help`执行真实读取核对。
- `bind_tomato100_vector_norm --bundle ... --source-norm ... --verification ... --assets ...`：通过真实读取核验后绑定独立A/B归一化资产，不审批ROI质量。
- `audit_tomato100_roi_candidates --selection ... --candidates ... --out ...`：阶段只用于QA；`benchmark_tomato_roi_locator`与`benchmark_tomato_four_view_gpu`分别检查单帧ROI和完整模型。

准备契约见`subtasks/complete100_rgb_crop_preparation.json`；可核对汇总见`subtasks/records/complete100_causal_four_view_20261111_validation_r1.json`。之前的人工预览、source plan、合成预算及LK失败回执均保留，未覆盖或改写为自动方法成功。

## normal93 无 Agent 训练准备

分支 `tomato-subtasks-ab-train`，从四类指令统一后的 `56ca763` 建立。此实验不调用任何阶段 Agent；B 训练使用人工审核的阶段文本，未来部署由人观察并点击阶段，在线 prompt 与动作队列切换属于后续部署任务。

固定清单为 `subtasks/records/normal93_20261109_selection.json`：28 上 `normal/no-block-more/A`、`B` 的93个独立录制，410456帧，30Hz。排除 `A/1106_740_796_normal/episode_000009`（仅三段）及 `B/1107_696_696_normal/episode_000000`（历史剔除、无发布粗标注）。不用 speedup 或 C。清单绑定各条来源ID、帧数和标注SHA，新增、缺失或改动均需人工检查并新建实验版本，不自动纳入。

A/B 均使用全部93条原始轨迹训练，410456帧，不预留验证或测试集。保留完整视频和所有物理列；仅更新合并索引、task_index，新增 coarse_phase_index。B 的七个位置映射到四种无标点文本，两次 return 共用 held object 指令。50步动作窗口允许跨阶段，尾部沿用 episode 内末帧补齐，不在阶段边界制造保持动作。`splits.json` 仅包含 train=0..92，是全部训练数据的显式清单。

两套新配置分别为 `training_configs.experiments.tomato_sharpa.subtasks.normal93_a_goal` 和 `normal93_b_stage`。默认 π0.5 全参、58D（手臂14维Delta、手部44维Absolute）、batch512、FSDP8、8000步，种子42/42/42001/42002，学习率与原 fullft recipe 相同；这只是待启动默认值，不是训练已完成或显存容量验收。没有开启论文增强或 Agent。数据身份、统计目录、实验目录均与旧单果/FLASH/Agent实验分开；归一化覆盖全部93条，A/B分别保存绑定统计。

在28的既有工程 `/home/mengxiang/workspace/code/tomato-openpi-sharpa` 使用公共环境：

```bash
export PI05_TOMATO_SUBTASK_BUNDLE="$PWD/data/tomato_truss_normal93_20261109_annotation_r2_all_sources_v2"
bash scripts/python.sh -m scripts.prepare_tomato_ab \
  --source-root /mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-more \
  --output "$PI05_TOMATO_SUBTASK_BUNDLE" \
  --tokenizer "$HOME/.cache/openpi/big_vision/paligemma_tokenizer.model"
# 生成新目录，拒绝覆盖。准备过程核验全部源SHA和真实token预算，不修改源数据。
for mode in a_goal b_stage; do
  MODULE="training_configs.experiments.tomato_sharpa.subtasks.normal93_${mode}"
  bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
done
bash scripts/python.sh -m scripts.check_tomato_ab --require-norm \
  --output outputs/normal93_annotation_r2_ab_verification.json
```

检查器比较所有源与派生 Parquet 的物理列、逐帧阶段/task映射、全部派生媒体SHA、全部93条实际加载结果和全部阶段边界的50步动作窗口，并为A/B各读取一个真实归一化模型batch。不加载模型权重、不运行训练；效果后续通过实机测试判断。

本次已接收 B/1106_750_791_normal 原 episode4 的人工 revision2：最终放回为 `[2908,2990)`，82帧（2.73秒）。冻结清单仅该条标注SHA改变，93条成员、物理轨迹、四种prompt和全部训练策略不变。新包采用 `annotation_r2_all_sources_v2` 身份，旧包保留，不会覆盖；默认配置读取清单中的 bundle_name。B 在帧2908至2988从harvest更新为return。

当前normal93两配置采用手臂Delta/手部Absolute：原始action仍为绝对关节目标，左右臂目标各7维减去当前帧实测state，50步窗口均使用同一当前state；手部44维不变，推理输出通过逆变换还原绝对目标。新asset_id以`_delta_30hz`结尾，配置名及实验名新增`_delta`，必须重新计算统计，不复用Absolute统计或续训旧checkpoint。历史验收记录仍对应当时Absolute配置。

本次使用独立 `all_sources_v2` 数据、统计和实验身份。旧75/9/9包和75条统计保留为历史，当前配置拒绝加载它们。93条物理示教和prompt没有变化。当前按用户要求将训练规模对齐既有T3：batch512、FSDP8、8000步；学习率计划仍为warmup1000、peak2.5e-5、decay_steps30000、end2.5e-6，采样和种子不变。

2026-11-09 revision2 已在28完成真实验收：A/B均实际加载93条、410456帧；episode89最终阶段为82帧，B的帧2908至2988正确改为return，A总目标指令不变；原始revision2标注快照SHA核对通过。两套统计已重新生成绑定，数值与上一版全数据统计一致。全部物理列、媒体SHA、逐帧指令、阶段边界50步窗口和真实归一化batch验收通过，16项相关测试通过。摘要见 `subtasks/records/normal93_20261109_annotation_r2_validation.json`；完整报告在服务器 `outputs/normal93_annotation_r2_ab_verification.json`。未加载模型权重、未启动训练、未修改部署端。

正式训练另行明确参数后，沿用 `scripts.run_training`；默认使用新的 all_sources_v2 包；若设置 bundle 环境变量，必须指向该新包，W&B online/offline与本地日志同时保存，不覆盖旧实验目录。

## 27服务器训练UI

27使用 `mengxiang@47.116.73.163`、SSH端口27；与28共用同一工程、revision2训练数据、统计和基础权重，无需复制。27的8张RTX PRO 5000 72GB已通过JAX/CUDA/NCCL八卡AllReduce检查；这不代表batch64/FSDP8容量验收。

UI服务单独监听27的127.0.0.1:8767，本地通过SSH转发到18767，不修改28既有UI。界面默认选择B阶段指令（全93条），A总体目标保留对照，默认W&B offline；固定数据、统计、horizon、手臂Delta/手部Absolute、无RTC和等权采样，训练参数显示batch512/FSDP8/8000步，打开页面不会启动训练。

UI与CLI共用 `scripts/training_environment.py`：正式训练进程显式设置 `JAX_PLATFORMS=cuda`、所选 `CUDA_VISIBLE_DEVICES`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`，不依赖登录shell默认值。公共 `scripts/python.sh` 负责环境和NCCL；UI服务和预检进程保持CPU。正式任务state.json记录实际训练环境变量。

```bash
# 在27现有工程启动独立UI（终端保持运行）
PI05_UI_SERVER_LABEL=27 WANDB_MODE=offline bash scripts/python.sh -m scripts.training_ui --port 8767
# 在本机转发；用UI终端输出的token打开18767页面
ssh -N -p 27 -L 127.0.0.1:18767:127.0.0.1:8767 mengxiang@47.116.73.163
```

2026-11-09 已在27启动8767 UI并建立本地18767 SSH转发，真实浏览器默认B加载通过：revision2全部93条/410456帧、batch64/FSDP8/30000步、四种阶段prompt、W&B Offline，八张空闲卡已在页面勾选。点击“检查数据与统计”后返回“已验证，可复用”，norm_provenance=verified；UI配置真实roundtrip保留原训练参数。80项Python测试与12项JS交互测试通过。页面已打开并保留，未点击启动训练；27 UI进程信息与私有启动日志保存在本工程 `.runtime/training_ui_27_8767.{json,log}`，本机转发进程记录在本地 `.runtime/training_ui_27_tunnel.json`。不记录token到Git，不修改28既有服务。

2026-11-09 Delta切换已在27验收：A/B各自重新计算全93条、410456帧的独立Delta统计；逐条物理数据、媒体SHA、阶段指令与所有阶段边界50步动作监督检查通过，14维手臂差分及44维手部绝对目标与既有适配器逐值一致，真实归一化batch通过。UI默认B和Delta，数据与统计检查返回已验证，保留八卡选择与offline记录。82项Python和12项JS测试通过；摘要见 `subtasks/records/normal93_20261109_delta_validation.json`。未启动训练，尚未验证GPU训练容量或实机效果。

仅对齐训练规模，不启用旧T3的timestamp_zscore_arms或beta=0.5噪声设置。batch与总步数不改变数据变换，继续使用已验证的独立Delta统计；历史batch64验收不代表当前batch512的GPU容量已验收。

## 已准备的启动入口

本轮以 **B 阶段指令**为主方案，A 总目标保留为对照，训练规模为batch512、FSDP8、8000步；学习率、采样、随机种子和50步跨阶段监督不变。默认使用revision2全93条数据及独立统计，不接Agent，不改部署端。

在28既有工程内执行，默认命令只预检、保存最终配置和数据/统计指纹，不启动训练：

```bash
cd /home/mengxiang/workspace/code/tomato-openpi-sharpa
bash scripts/python.sh -m scripts.tomato_ab_run
# A 对照预检
bash scripts/python.sh -m scripts.tomato_ab_run check --variant a
# 查看当前GPU占用，不加载模型
bash scripts/python.sh -m scripts.tomato_ab_run gpus
```

预检快照保存到 `logs/normal93_preflight/`，正式运行由原 `scripts.run_training` 保存本地日志、最终配置、数据采样审计、W&B记录和checkpoint。入口默认W&B offline，避免未验明的网络或登录阻断启动，之后使用wandb sync上传；明确配置网络和登录后可传 `--wandb-mode online`。

以下为**正式启动命令，本轮未执行**。必须等8张卡空闲；入口发现已有计算进程或明显占用会拒绝启动，不自动排队、不停止现有任务。A/B需要顺序运行，不能同时占用同一组卡。

```bash
# B 主方案，默认GPU 0..7；不会改变训练参数
bash scripts/python.sh -m scripts.tomato_ab_run start
# B结束后，A对照使用独立实验目录
bash scripts/python.sh -m scripts.tomato_ab_run start --variant a
```

2026-11-09 开训准备实测：revision2的93条源SHA重新核验通过，A/B最终配置和归一化绑定预检通过，实验目录均未占用且输出可写；官方pi05_base已在CPU用实际训练加载器加载校验，51组参数中48组加载、3组58维投影按原设计初始化，51组均可训练。batch64/FSDP8在8个CPU抽象设备上完成损失/梯度/优化器形状追踪；没有执行数值训练或GPU容量测试。W&B offline真实写入/结束测试通过。入口与原launcher等54项测试、配置/Sharpa/统计等64项回归通过（含重复入口测试，不相加）。摘要见 `subtasks/records/normal93_20261109_training_readiness.json`。检查时28的8卡正被其他用户训练占用，资源保护验证会拒绝启动；没有停止既有任务或自动排队。

入口使用基础权重开始全新优化器，不续训、不覆盖已有checkpoint。默认实验目录已存在时会拒绝，可用 `--exp-name` 指定新的运行名。资源检查只是启动时快照，不能替代实际8卡数值训练和显存容量验收。

## 历史粗阶段 A/B 准备设计（不用于当前全数据训练）

新配置位于 `subtasks/`，不替换下方历史单果 baseline。先完成人工粗阶段审核，再准备数据，当前没有启动训练或移动生产数据。粗阶段是4种动作按顺序出现7段：拿起、摘取、放回、调整、再次拿起、继续摘取、放回裸杆。异常轨迹不强套模板，退回人工判断；细阶段本轮不用于训练。

- A：`training_configs.experiments.tomato_sharpa.subtasks.coarse_a_goal_v1`，全帧使用 `Harvest all tomatoes into the left basket and return stem to right plate.`。这是紧凑的总体目标，没有枚举左右手分工和中途调整；完整流程保存在 bundle 的 `procedure` 和人工阶段标注中。它不等同于把完整流程文本都输入模型。
- B：`training_configs.experiments.tomato_sharpa.subtasks.coarse_b_stage_v1`，逐帧读取当前粗阶段英文。左右手分工、盘上调整使用对应动作文本，两次放回共用同一句指令，见 `src/openpi/subtasks/tomato.py`。模型学到的是阶段条件动作，实机需要阶段调度器。

两者保持完整 episode、图像、时间戳、state/action、horizon 50，不切视频、不制造边界静止动作。B 在交界附近仍保留跨阶段的真实未来动作窗口；因此当前指令不保证覆盖窗口内每个动作，这需要后续对比评估，不能靠补造保持动作解决。统计通过既有 TaskView 仅对 train split 计算。验证/测试应同样使用 `splits.json` 的 val/test 条目，不用训练集结果代替评估。普通训练入口不自动评测整条任务成功率。

任务描述与58维离散状态共同受256-token限制。准备工具使用实际 PaliGemma 分词器检查每个状态离散值、模板和文本组成的上界，超过即拒绝。A 最多256，B 最多255；没有增加长度配置，也不依赖静默截断。具体分词器指纹和检查结果在 `subtasks/records/token_budget.json`。更换分词器、状态维度或英文后必须重查。完整旧长描述保留于 `PROCEDURE` 元数据，不作为256-token模型输入。

准备输入参考 `subtasks/preparation_plan.example.json`：每条填写本地可访问的 LeRobot 包、原 episode 编号、粗阶段 JSON 和原始 rosbag 的稳定 `recording_group`。normal/speedup 必须共享录制组；`recording_namespace` 是采集来源命名空间，同一录制两版必须一致，不能填 normal/speedup。工具核对 SHA256、全段人工确认、7段完整覆盖与来源一致性，要求至少10个独立录制；完成全部计划数据审核后才将 `review_complete` 设为 true。这个字段是人工对清单完整性的声明，工具不能知道遗漏的采集文件。

```bash
# 全部数据审核完成后执行，当前不要运行；输出必须是尚不存在的新目录。
bash scripts/python.sh -m openpi.subtasks.prepare \
  --plan /absolute/path/to/reviewed-plan.json \
  --output /absolute/path/to/tomato_truss_v1 \
  --tokenizer /absolute/path/to/paligemma_tokenizer.model
```

输出集中在同一位置：`a_goal/`、`b_coarse/`、`splits.json`、`bundle.json`。两版完整复制视频，不重编码或裁剪；仅更改新副本的索引、task_index，并增加 coarse_phase_index。原始文件不写入。bundle 记录来源 SHA、原 episode 到新编号映射、人工标注 SHA、英文和 A/B 共用的80/10/10分组。normal/speedup 可以都入清单但不应当作独立录制；同源两版共享 split。源目录应是可访问文件路径，不能直接填写 sftp URL。工具会复制两套媒体，请选择容量足够的目标；这不是现在替用户搬运或汇总数据。

准备后设置 `PI05_TOMATO_SUBTASK_BUNDLE=/absolute/path/to/tomato_truss_v1`，分别加载 A/B 配置，沿用下文检查、计算独立统计及训练的步骤。未准备 bundle 时配置会明确拒绝；不要复用单果任务统计、旧实验目录或柠檬分类头。两套实验使用同一 fullft recipe 和种子，各有独立 assets/日志/权重，不自动启动。

## 阶段决策 agent

从用户提供的 `/home/user/lemon_vjepa2_training_20261106` 迁移并适配 V-JEPA 2 的因果视频历史、冻结视频编码、分类头和原始录制分组思想，来源文件指纹在 `subtasks/records/migration.json`。源包 `serve_lemon_phase_agent.py` 接的是旧静态图像模型，未复制；柠檬六类头、12维UR状态、旧权重及数据均未迁入。

新模型输入三路16帧历史，按时间从已发生的观测采样10Hz、覆盖1.5秒，早期不足重复首帧；不跨 episode。当前116维转换器状态按已有适配顺序提取58维实测位置。输出7个阶段位置，区分两次拿起/摘取。分类头只能用番茄已审核数据重新训练；V-JEPA预训练权重从指定本地目录加载并校验固定SHA，无自动下载。

```bash
# 数据准备及V-JEPA本地权重就绪后才能训练agent，以下当前未运行。
bash scripts/python.sh -m scripts.train_tomato_phase_agent \
  --bundle /absolute/path/to/tomato_truss_v1 \
  --encoder /absolute/path/to/pinned-vjepa2 \
  --output /absolute/path/to/new-agent-run --device cuda --wandb-mode offline
```

脚本按固定2Hz时间表提取三路因果视频及关节历史；归一化只用训练组；验证 macro F1 选最优头，测试组仅在选完后评估，日志同时本地及W&B保存，离线日志之后同步。V-JEPA 编码和头训练需要运行环境提供 torch、cv2、Transformers 的 VJEPA2Model；当前机器尚未完成这个依赖/权重/GPU链路验收，不能把迁移代码与合成测试称为真实模型已可用。

运行时通过 `PhaseModel.predict()` 产生概率，再交给 `PhaseDecision`；`python -m scripts.tomato_phase_decision` 提供概率输入的JSONL决策接口，本身不看图像。连续3次概率达到0.8才允许进入相邻下一阶段，不按时间跳转，不跨阶段跳转；疑似异常或低置信度返回 `needs_review`。最后阶段是“放回裸杆”，不是完成检测；不会自动宣称任务成功或停止机器人。阈值只是初始实验参数，需在验证集校准。真机相机同步、丢帧/延迟处理、动作执行端、完成判定及与π0.5共用GPU的性能尚未接入或验证。

环境、日志和续训见[根 README](../../../README.md)，个人配置见[多人开发 README](../../../多人开发README.md)。官方 LoRA 和全参 baseline 现在直接读取 `10.16_2_data_lerobot` 的转换器格式。UI 中选择“番茄 · Sharpa / LoRA”或“番茄 · Sharpa / 全参”即可使用，不再需要 tairan 临时配置。

## 数据和向量协议

原始 LeRobot v2.1 数据为 30 Hz，`observation.state` 是 116 维位置/速度，`action` 是 58 维绝对关节目标。必须严格匹配[转换器适配器](../../../src/openpi/policies/fr3_sharpa_multimodal_policy.py)中的原始名称和顺序；不能只修改 `info.json`。

| 向量 | 原始布局 | 读取时的 Python 切片拼接 |
| --- | --- | --- |
| state | 左臂位置/速度、右臂位置/速度、左手位置/速度、右手位置/速度 | `[0:7] + [28:50] + [14:21] + [72:94]` |
| action | 左臂、右臂、左手、右手 | `[0:7] + [14:36] + [7:14] + [36:58]` |

模型输入和输出统一为左臂 7 + 左手 22 + 右臂 7 + 右手 22，共 58 维，单位 rad；状态只保留实测位置，action 默认保持 absolute 目标。相机使用 cam0 外部、cam2 左腕、cam1 右腕 RGB。深度、触觉、速度及审计列保留在文件中，不进入当前模型；底层 loader 仍可能读取这些列。名称检查不能证明数值单位、左右语义或时间对齐正确，需要数据生产者确认。

旧 `Fr3Sharpa` 58D 实测位置适配器仍保留，其严格检查不变；这两份官方 baseline 使用 `FR3_SHARPA_MULTIMODAL`。训练入口统一使用 `baseline/` 下的全参与 LoRA 两份配置，UI 和命令行使用相同入口。

## 官方配置和数据身份

| 项目 | 设置 |
| --- | --- |
| 数据来源 | `sshfs://user@192.168.10.202/data/data-fruits/real/harvest-tomato/data/clean-rosbag/sharpa/10.16_2_data_lerobot` |
| 默认训练目录 | `/mnt/data/users/tairan/workspace/data/10.16_2_data_lerobot` |
| repo_id | `fr3_sharpa/tomato_1016_2_multimodal_rgb_v1` |
| 数据版本 / asset_id | `tomato_sharpa_1016_2_multimodal_rgb_absolute_v1_30hz` |
| 统计目录 | `<当前仓库>/assets/tomato_sharpa_1016_2_multimodal_rgb_absolute_v1_30hz/` |
| 全参配置名 | `pi05_fr3_sharpa_1016_2_multimodal_rgb_fullft_v1_absolute_30hz` |
| 全参默认实验名 | `tomato_sharpa_1016_2_multimodal_fullft_absolute_s42` |
| LoRA 配置名 | `pi05_fr3_sharpa_1016_2_multimodal_rgb_lora_v1_absolute_30hz` |
| LoRA 默认实验名 | `tomato_sharpa_1016_2_multimodal_lora_absolute_s42` |

路径和身份由[共享数据定义](baseline/dataset.py)提供。优先使用 `PI05_SHARPA_MULTIMODAL_DATASET_ROOT` / `PI05_SHARPA_MULTIMODAL_ASSETS_DIR`，兼容旧 `PI05_SHARPA_DATASET_ROOT` / `PI05_SHARPA_ASSETS_DIR`。只搬迁相同数据可以保留身份；更换批次必须设置独立版本、来源和统计身份。

两种方案采用同一个英文 prompt 示例，实际训练文本来自 `meta/tasks.jsonl`：

> Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.

## 训练参数

| 方案 | 文件 | 全局 batch | FSDP 分片 | 步数 |
| --- | --- | --- | --- | --- |
| 全参 | [tomato_fr3_sharpa_fullft_v1.py](baseline/tomato_fr3_sharpa_fullft_v1.py) | 32 | 8 | 30,000 |
| LoRA | [tomato_fr3_sharpa_lora_v1.py](baseline/tomato_fr3_sharpa_lora_v1.py) | 8 | 1 | 30,000 |

全参默认按 8 张 GPU 分片，减少参数、梯度及优化器状态的每卡显存占用；选择较少 GPU 时必须覆盖 FSDP 参数，GPU 数量需能被分片数整除，全局 batch 需能被 GPU 数量整除。LoRA 保留公共配方的 batch 和 FSDP 设置，不强制使用 8 张卡。

其余沿用各自公共 recipe：pi05、bfloat16、58D 模型、horizon 50、workers 8、epoch_shuffle、seed 42/42/42001/42002、warmup 1000、peak 2.5e-5/end 2.5e-6、AdamW、无 EMA，每 2000 步保存。LoRA 使用 VLM rank 16、Expert rank 32，视觉及投影可训练；全参不使用 LoRA，全部参数可训练。基础权重为 `/mnt/data/public_ckpt/pi05/pi05_base/params`，投影形状不匹配时按公共 PartialCheckpointWeightLoader 初始化。

## 统计、启动和状态

```bash
export MODULE=training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_fullft_v1
export EXP=tairan_sharpa_1016_fullft_fsdp8_b32_s42_run01
bash scripts/python.sh -m scripts.run_training "$MODULE" --exp-name "$EXP" --dry-run
# 仅在没有配套统计时计算；工具拒绝覆盖已有目录。
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
bash scripts/python.sh -m scripts.run_training "$MODULE" --exp-name "$EXP" --check
# 只选择分配给自己的 GPU；示例默认 8 卡。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name "$EXP" --wandb-mode online
```

LoRA 将 MODULE 尾名换为 `tomato_fr3_sharpa_lora_v1`，按分配选择 GPU。W&B online 需预先登录，不能联网时显式使用 offline，恢复联网后 sync；本地日志和运行清单保存在 `logs/`，权重保存在 `checkpoints/`。

统计必须同时包含 `norm_stats.json` 和 `norm_metadata.json`，并通过数据、变换及 horizon 的绑定校验。LoRA 与全参可在绑定一致时共用统计；batch 和 FSDP 的改变不改变统计。与此前临时配置相同的批次和适配规则可以校验复用已有统计；旧 58D 或 delta 统计不能直接使用。UI 切换 delta 时必须选新 asset_id 并重新统计。

新 baseline 已准备输入适配；这不代表已经完成完整训练、效果评估或实机验证。`records/` 中的真实检查结果属于其中记录的历史批次，报告中的旧模块名保留用于追溯，可通过 Git 历史查看当时配置；它们不是本批次完整训练完成的证明。服务器既有临时配置、运行中的任务及其代码未随本地更新修改。续训必须使用原代码、配置、数据、统计及实验名；应用本地新版本后请创建新实验。

数据兼容检查使用当前 baseline，计算独立归一化统计并读取一个 batch，不加载模型权重、不启动训练：

```bash
bash scripts/python.sh -m scripts.check_converter_dataset \
  --robot sharpa --dataset /path/to/prepared_lerobot_dataset \
  --norm-assets /path/to/new-validation-assets \
  --output /path/to/new-validation-report.json
```

统计和报告使用新路径；检查其他批次不代表默认 10.16_2 批次已通过验证。

## 本地 Delta 论文算法实验（不修改 baseline）

共享代码新增显式选择的 `PaperPi0Config` / `PaperSharpaData`，原八份 baseline、recipe、本体配置和原 Pi0 默认实现保持原样。个人配置仍只放在 `training_configs/experiments/tomato_sharpa/template/<负责人>/`，被 Git 忽略；不要强制提交。算法代码在 `src/openpi/models/pi0_delta_paper.py`、`src/openpi/training/delta_paper*.py`、`src/openpi/shared/delta_paper.py`；统计、预检查、数据加载和保存入口只有选择新类型时才进入新路径。

### 配置位置与对照

生成器会在本人目录创建五份个人文件。首次获得共享代码后，可在自己的 clone 中创建，已有负责人目录会拒绝覆盖：

```bash
bash scripts/python.sh -m scripts.create_delta_paper_experiments --owner ma39nong
```

| 文件 | 双臂14维归一化 | Flow Matching 噪声 |
| --- | --- | --- |
| `delta_paper_t0.py` | 全局每维 quantile | 独立高斯 |
| `delta_paper_t1.py` | 每未来时间位置 mean/std | 独立高斯 |
| `delta_paper_t2.py` | 全局每维 quantile | 完整50×58相关噪声，beta=0.5 |
| `delta_paper_t3.py` | 每未来时间位置 mean/std | 完整50×58相关噪声，beta=0.5 |
| `delta_paper_tg.py` | 全局每维 mean/std | 独立高斯；区分z-score与逐时间归一化的贡献 |

所有组都只将双臂转换为“未来绝对目标减本轮起点实测状态”，双手44维仍为Absolute及原quantile规则。近零双臂std使用1e-4 rad下限。原始数据仍为116D状态、58D绝对动作，适配后58D；英文prompt来自训练数据的 `meta/tasks.jsonl`。

所有组**统一只训练完整50步窗口**，不足50步的episode尾部起点不参与训练或统计。这是为保证统计与标签完全一致而建立的新对照，T0不是旧模型原样重训；与旧baseline相比还包含窗口覆盖变化，不能把差异都归因于噪声或归一化。T1对T0同时包含quantile改z-score的变化，若要单独判断逐时间位置的贡献，应比较T1与TG。

相关矩阵从同一训练集的归一化动作窗口按 `[time,dimension]` 展平估计，大小2900×2900；按特征中心化并标准化为相关矩阵，常量维度独立，使用 `beta*Sigma+(1-beta)*I` 做Cholesky。训练和推理初始化使用同一噪声定义；beta=0保持原随机流与计算路径。默认均匀无放回抽取最多20000个完整窗口拟合相关性，seed=42；逐时间统计及global统计使用全部训练窗口。计算可能占用约GB级RAM及较长CPU时间，可在个人配置调整 `correlation_max_windows`，需新统计身份。当前实验支持单个或显式列出的多个30Hz数据集，采用epoch_shuffle和JAX全参入口；不支持TaskView、加权重复采样或PyTorch。多源按完整窗口统一打乱，每个有效起点同权，不按源等概率重复采样。

修改数据来源/版本、root、repo_id、asset_id和统计父目录：在**个人文件的 `build_dataset()`** 返回值上用 `dataclasses.replace` 调整；只搬迁相同数据可用 `PI05_SHARPA_MULTIMODAL_DATASET_ROOT`，论文统计父目录可用 `PI05_PAPER_ASSETS_DIR`。换批次须记录新的来源、版本与asset_id。算法既可在 UI 的论文功能复选框选择，也可在个人 `build_config()` 的 `mode/beta` 设置；batch、FSDP、预算等也用replace覆盖。不要修改公共baseline。UI可以从 Sharpa multimodal 全参 baseline 直接勾选论文功能，也可选择已有个人模块；论文UI适配保留个人配置中的数据源身份和顺序，修改统计存储位置时同步更新模型资产路径和摘要。更换数据须在个人文件定义新数据版本，UI不能直接改论文数据目录、顺序、repo_id或切换加权采样。

### 统计、检查和训练

默认沿用Sharpa全参：batch32、FSDP8、30000步、原学习率、独立seeds、本地日志及W&B。下面以T3为例；已有本机配置不再执行初始化命令：

```bash
export MODULE=training_configs.experiments.tomato_sharpa.template.ma39nong.delta_paper_t3
bash scripts/python.sh -m scripts.run_training "$MODULE" --dry-run
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
bash scripts/python.sh -m scripts.run_training "$MODULE" --check
# 数据、资产、GPU检查通过后再单独启动；下面会开始实际训练。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name ma39nong_sharpa_delta_paper_t3_s42_run01 --wandb-mode online
```

离线服务器显式改为 `--wandb-mode offline`，之后按原规则sync。正式运行仍写入原 `logs/`、`checkpoints/`、`wandb/`；启动器会保存实际个人源码、公共commit、数据、prompt及统计审计。改变数据、统计或算法后不能resume旧运行，须新实验名。

每组使用独立asset_id并原子发布，拒绝覆盖已有统计目录。资产包含 `norm_stats.json`、`norm_metadata.json`、`delta_paper.npz` 和 `delta_paper.json`，绑定来源、算法和摘要，checkpoint会完整保存它们；相关因子是非训练变量，随checkpoint保存且不进入优化器。新的actions统计采用 `(50,58)` 版本布局，双手global统计按时间广播；原 `(58,)` 资产不变。

**新训练权重不能直接交给没有该组合适配的部署profile。** 本轮只改训练仓库，未改任何串行、RTC或Soft部署。旧标准profile会拒绝新的统计形状；已有独立Paper与Train RTC模式的外部工程，也仍需单独接入组合模型、匹配统计及相关噪声采样，不得只换checkpoint路径。

本轮状态为代码与离线验证准备完成：已验证合成parquet的真实转换、完整50×58统计/2900×2900因子、实际loader及采样审计、缩小Gemma/SigLIP的梯度与优化器更新、checkpoint保存/恢复和beta=0等价性。没有计算服务器真实训练集的新统计，没有启动真实数据完整训练或实机测试。此前 `template/timestamp_norm_trial.py` 及其已有结果保留原样。


### no-block-one 六份摘番茄数据联合实验

数据保持在 `/mnt/data/train_platform/datasets/tomato_sharpa/normal/no-block-one`，不搬迁、复制、合并、建软链接或修改原始meta/parquet/video。共享helper `build_no_block_one_dataset` 显式列出以下顺序；不自动扫描新加入的目录：

| 相对原目录 | episode | 原帧数 | 完整50步窗口 |
| --- | ---: | ---: | ---: |
| A/1022_121_430_normal_lerobot | 168 | 157442 | 149210 |
| A/1022_54_120_normal_lerobot | 41 | 37145 | 35136 |
| A/1026_444_513_normal_lerobot | 47 | 37240 | 34937 |
| B/1022_130_423_normal_lerobot | 34 | 40221 | 38555 |
| B/1022_56_106_normal_lerobot | 8 | 10607 | 10215 |
| B/1026_446_510_normal_lerobot | 13 | 11551 | 10914 |
| 合计 | 311 | 294206 | 278967 |

以上是2026-10-29读取实际元数据得到的数量，不能代替完整读取与训练验证。六份数据的帧率、维度、关节名字顺序和相机角色已做只读检查，均一致；来源的任务文本均为本页相同英文摘番茄指令。

每个源先独立取完整窗口，再在内存联合；episode编号只在采样审计里映射，不写回数据。训练对全部278967个完整起点做epoch_shuffle，统计也按同样起点权重覆盖全部六份；相关性从全部完整窗口均匀无放回抽最多20000个。大批次自然贡献更多窗口，不把小批次放大为相同权重。手部Absolute、双臂Delta及逐时间统计规则不变，不加入Multi-Sample Flow Matching。

统计使用独立 `tomato_sharpa_no_block_one_<owner>_delta_paper_t3_v1_30hz` 身份，记录六个原始目录、逐源帧数/episode/有效窗口数和文件摘要；任一源内容、顺序或算法改变都会导致检查拒绝复用。原10.16_2实验仍保留，不能复用它的统计或实验名称。

服务器同步共享代码后，第一次可生成个人配置；若负责人目录已存在，生成器会拒绝覆盖。来源分支曾使用被Git忽略的个人配置；这些文件不随共享代码迁移。服务器例子：

```bash
bash scripts/python.sh -m scripts.create_delta_paper_experiments --owner mengxiang --dataset no-block-one
export MODULE=training_configs.experiments.tomato_sharpa.template.mengxiang.delta_paper_no_block_one_t3
bash scripts/python.sh -m scripts.run_training "$MODULE" --dry-run
bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
bash scripts/python.sh -m scripts.run_training "$MODULE" --check
# 下列命令才开始真实训练；保持原全参配方、8卡、batch32和30000步。
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/python.sh -u -m scripts.run_training \
  "$MODULE" --exp-name mengxiang_sharpa_no_block_one_delta_paper_t3_s42_run01 --wandb-mode online
```

无法联网时显式采用 `--wandb-mode offline`，日志仍写本地，之后sync。此处不修改baseline、main、共享环境或数据权限。当前仅完成多源代码与合成数据离线验证、服务器六源元数据只读预检查，尚未同步服务器代码、计算真实联合统计或启动训练。新训练权重依旧需要单独的论文部署profile适配。


### UI使用现有论文统计

服务器启动 `bash scripts/python.sh -m scripts.training_ui --port 8765`，通过SSH转发端口后，在个人配置选择 `mengxiang / delta_paper_no_block_one_t3`。保留六个数据目录、逐时间/相关噪声T3算法、horizon50、epoch_shuffle以及现有asset_id，可修改实验名、batch/FSDP和保存步数等训练参数。UI预检查应显示现有统计通过校验，不需要重新计算。选择8张卡、batch32/FSDP8、W&B offline后，点击开始才会启动训练。

UI专用适配位于 `src/openpi/training/delta_paper_ui.py`；默认baseline UI行为不变，论文统计代码及其源码摘要也未改变，所以能复用命令行计算的资产。改变统计目录/asset_id时模型路径和摘要同步更新；原目录不存在统计时可由UI自动计算。UI保存的个人副本继续保留完整T3类型与六源身份，不依赖其他个人文件。代码同步或修改后须重启本工程UI进程；不停止其他用户的UI或训练。

## T3 + trainRTC 集成

本集成基于主线 `897cb917` 和来源 T3 `2ab68c8b`。保留来源的统计、完整窗口、相关矩阵算法与实现摘要；原有 baseline 不变。普通 Pi0 的 `noise()` 默认生成原标准高斯；Paper 仅覆盖该噪声函数，训练损失和采样继承主线 RTC。仅支持 JAX π0.5 Sharpa 全参，不实现论文第9页的 soft / correlation-aware inpainting，也不增加异步机器人执行服务。

### UI 四种选择

| 场景 | 配置入口 | RTC 上限 | 模型/统计 |
| --- | --- | ---: | --- |
| 普通训练 | 正式 Sharpa 全参 baseline 或普通个人副本 | 0 | 原生 Pi0Config；普通统计，无 Paper 资产 |
| 仅原始 trainRTC | 与上一行相同 | 10 | 仍是原生 Pi0Config / 原数据工厂；标准高斯 |
| 仅 T3 | 个人 `delta_paper_t3` 或六源对应文件 | 0 | PaperPi0Config / PaperSharpaData；逐时间双臂 z-score + beta=.5 |
| T3+RTC | 与上一行相同的 T3 配置与统计身份 | 10 | 相同 T3 资产；同一 forward 叠加 clean-prefix / suffix-only loss |

现在无需先在终端运行配置生成器：选择「番茄 · Sharpa / 全参」，设置数据目录、来源、版本与英文 prompt 对应数据后，在「03 训练参数 → 论文功能」直接勾选：

| 逐时间步归一化 | 相关噪声 | GUI 训练方案 |
| --- | --- | --- |
| 关闭 | 关闭 | 普通 Pi0.5，原归一化/采样规则 |
| 开启 | 关闭 | T1，双臂逐时间步 mean/std，独立高斯 |
| 关闭 | 开启 | T2，全局 quantile，相关噪声 beta=.5 |
| 开启 | 开启 | T3，双臂逐时间步 mean/std，相关噪声 beta=.5 |

RTC 使用原有最大延迟字段，0关闭、10开启，两类功能独立组合。论文勾选仅开放给30Hz Sharpa multimodal π0.5全参；Wuji、LoRA等配置禁用并由后端拒绝。勾选会自动固定双臂 delta/双手 absolute、epoch_shuffle 并清空旧加权分组，切换功能自动分配独立asset_id。预览/检查只验证，启动时沿用现有流程保存独立个人配置、数据来源/版本、英文prompt、算法合同、统计和实验身份；不修改公共baseline或旧统计。同一算法只修改RTC可复用匹配统计。已有Paper个人配置恢复原算法，既有数据来源封存规则继续有效；新建GUI方案可直接采用当前选择的单源或多源。T0/TG仍可通过已有个人配置或可选CLI生成器使用。相关噪声固定beta=.5，其他beta仍由个人配置定义。

本轮状态为GUI功能及软件验证完成；未启动真实数据训练，未做实机验证。服务器需要同步代码并重启本工程UI进程，然后刷新页面。

先选择配置，再修改现有 RTC 字段，查看最终配置、数据预检查，通过后使用新的实验名启动。Paper 配置的双臂 delta/双手 absolute、源目录顺序、repo_id、数据版本和 epoch_shuffle 固定；UI 会拒绝 absolute 或加权重复采样。配置摘要显示有效 RTC、Paper mode/beta/摘要与统计路径。`beta=0` 不等于退出 Paper：时间归一化和完整窗口仍可能存在。切换 Paper / 非 Paper 时不得混用统计和 checkpoint。

### 独立生成、校验和启动

原生成器默认五组输出完全保留，RTC 默认为0。已有本人 T3 配置可直接在 UI 改 RTC 并用新实验名；也可显式生成一套新 RTC 个人配置，不覆盖原五组：

```bash
# 单源；数据来源默认保持10.16_2定义，修改数据须在个人build_dataset里定义独立身份。
bash scripts/python.sh -m scripts.create_delta_paper_experiments --owner alice --rtc-max-delay 10
# 或六源预设（与单源命令二选一；同一rtc_d10目录不会被覆盖）：
bash scripts/python.sh -m scripts.create_delta_paper_experiments \
  --owner alice --dataset no-block-one --rtc-max-delay 10
export MODULE=training_configs.experiments.tomato_sharpa.template.alice.rtc_d10.delta_paper_no_block_one_t3_rtc_d10
bash scripts/python.sh -m scripts.run_training "$MODULE" --dry-run
# 如果已有匹配的T3资产，只做check；没有资产时才先计算统计，工具拒绝覆盖已有目录。
# bash scripts/python.sh -m scripts.compute_experiment_norm "$MODULE" --stats-batch-size 128
bash scripts/python.sh -m scripts.run_training "$MODULE" --check
```

选择单源时 MODULE 改为 `...template.alice.rtc_d10.delta_paper_t3_rtc_d10`。新配置保留原 trial 的数据与 asset_id，name/exp_name 增加 `_rtc_d10`，因此 RTC 本身不要求新统计且 checkpoint/run 独立。UI 个人配置列表同时发现原文件及 `rtc_d10/` 子目录。统计路径由本人配置与 `PI05_PAPER_ASSETS_DIR` 决定；源数据、来源顺序、英文 prompt、asset 摘要会随运行记录保存，英文 prompt 来自各数据集的 `meta/tasks.jsonl`。

真正开始训练仍使用统一入口，按分配的 GPU 和确认后的最终配置执行，并开启本地日志与 W&B online；无法联网显式用 `--wandb-mode offline`，恢复后执行 `bash scripts/python.sh -m wandb sync wandb/offline-run-...`。此文的示例不代表已经启动8卡训练。

已有 T3 checkpoint 可用匹配资产和 `PaperWeightLoader(params_path=".../step/params")` 初始化新的 RTC 实验；该加载器保留由已校验 bundle 初始化的非训练 Cholesky。优化器/步数重新开始，`resume=False`。真正中断恢复必须维持原代码、数据、RTC、T3 参数和 reproducibility 合同，不能绕过严格检查。新 checkpoint 保存 Cholesky 变量及 `norm_stats.json`、`norm_metadata.json`、`delta_paper.npz`、`delta_paper.json`；历史资产不被改写。

### 模型接口与跨 chunk 前缀

采样继承 `sample_actions(rng, observation, num_steps=10, noise=None, prefix_actions=None, prefix_length=0)`。默认 noise 经过一次 `z @ L.T`，显式 noise 已在模型尺度下，不再次相关化。前缀长度必须是 `0..rtc_max_delay` 的静态整数，JIT 使用 `static_argnames=("prefix_length",)`；初始及每次积分后均固定前缀。30Hz下上限10覆盖约333ms的训练延迟，部署选择仍须根据实际延迟测量。

旧 chunk 的归一化后缀不能直接作为新 chunk 的归一化前缀。先按旧时间索引逆 ArmNormalization、逆 quantile，并加回旧物理 state 得到绝对目标；再减新物理 state（仅双臂），以新时间索引0..N-1做 quantile 和 T3 编码。双手始终 absolute/quantile，不按时间做 z-score。

`openpi.training.delta_paper_prefix` 提供无机器人依赖的验证 helper，接受 `config.data.create(...)` 得到的 PaperDataView 和 canonical LA7/LH22/RA7/RH22 物理向量：

```python
from openpi.training.delta_paper_prefix import decode_actions, encode_prefix

old_absolute = decode_actions(data, old_physical_state, old_model_actions)
physical_prefix = old_absolute[-prefix_length:]
new_prefix = encode_prefix(data, new_physical_state, physical_prefix)  # 返回H×58，只有前N步承诺
# prefix_length>0；为0时不需要构造前缀。
actions = sample(rng, new_observation, prefix_actions=new_prefix[None], prefix_length=prefix_length)
```

`physical_state` 必须是未归一化的58D实测关节位置，不能传 `observation.state` 的模型归一化值。新 prefix 补齐到完整 horizon 后按新时间位置编码；模型输出解码时必须使用同一新起点 state。此 helper 没有连接任何执行队列，实际异步时间对齐、超时处理与实机成功率仍须由外部部署单独接入验证。

### 验证状态

本次使用临时合成 parquet/fixture，验证标准 Pi0 与原始 RTC 回归、T3数值/完整窗口/六源统计、50×58联合 JIT 前后向和单步优化、非训练Cholesky、四文件checkpoint恢复、旧T3无RTC损失等价、跨chunk物理前缀round-trip、UI四模式保存/重载/preflight及只改RTC的资产复用。没有本地历史真实T3资产可做实盘预检查；受保护的三个统计/数据源码与来源分支逐字一致，未清除或跳过digest。

训练/模型采样支持与离线验证完成不代表真实数据完整训练、8卡吞吐验证或机器人实机验证。服务器既有任务、数据、资产与checkpoint未修改。本仓库未接入外部RTC队列。


#### 四类动作 prompt 与七阶段进度

当前 B 使用四类动作到四种无标点指令的一对一映射，定义在 `openpi.subtasks.tomato.ACTION_PROMPTS`。两次放回共用 `Right hand returns the held object to the right plate`，不再用不同裸杆指令。七个 `PHASES` 和 `coarse_phase_index` 保留，`PROMPTS` 由七位置的 `COARSE_SEQUENCE` 映射得到，因此仍有七个位置但只有四种不同文本。Agent 输出同时包含 `phase`、`phase_index`、`action_class`、`prompt`；其完成状态仍需独立判断，不因进入最后阶段自动完成。A 的总指令未改；旧五指令训练副本或 checkpoint 不会自动迁移，需使用新文本重新准备实验数据。该修改未运行训练或实机控制。


## 阶段指令可控性与局部图离线探针

B阶段指令方案已于2026-11-10 08:55在27正常完成8000步，最终checkpoint为`7999`。本节的GPU诊断使用28，不使用27；不执行机器人动作，不更新权重、统计或默认配置。

`scripts.tomato_prompt_probe`固定同一观测的三路图像、58维状态和噪声，只更换四种训练过的英文阶段指令。每条录制选择七段中点、六个边界前45/15帧及边界0/+15帧；去重后保留样本标记，两组噪声42/43。原生训练适配器、checkpoint统计、10步flow采样及Delta逆变换全部复用；checkpoint内只有统计数值文件，来源校验使用原训练assets的norm_metadata并核对两处norm_stats SHA一致，不开启legacy绕过；每个观测检查监督目标反归一化/逆Delta还原一致，首个样本重复推理检查确定性。固定图像/状态在四个条件下逐值相等。左右臂、左右手分别报告；主要1/5/15/50步误差只比较真实阶段内的前缀，保留完整50步原始输出用于诊断。不得把错误prompt对原示教的误差称为反事实动作正确率，也不把换prompt产生差异当作服从成功。

最小真实checkpoint检查（28工程内，选空闲GPU；输出必须新目录）：

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
bash scripts/python.sh -u -m scripts.tomato_prompt_probe \
  --checkpoint /absolute/path/to/finalized/checkpoint/7999 \
  --output outputs/normal93_prompt_new_check --gpu 0 --limit 3 --seeds 42 43
```

全量四分片按episode_index模4分配，分别选择28的GPU0～3；脚本启动前检查空闲状态，不停止其他进程。`scripts.summarize_tomato_prompt_probe <分片父目录> <新summary.json>`校验四分片代码、数据、统计、模型身份和两组噪声一致，并要求覆盖全部93条后汇总；`scripts.plot_tomato_prompt_examples`导出固定示例的绝对关节目标曲线。所有原始预测、日志和图像留在忽略的`outputs/normal93_*/`中，仅提交结果摘要。

`scripts.tomato_crop_audit`从原始分辨率帧输出三路全图、右腕中心半幅及右下候选ROI并排图。候选ROI只是工程可行性示意，未经训练或效果验收。当前`model.preprocess_observation`默认仅保留三路IMAGE_KEYS，额外crop键会被丢弃；局部图实验必须显式配置模型输入、训练及推理一致接通，不得仅给现有模型输入字典添加键后宣称已测试四视图。固定中心crop可能切掉边缘的枝梗，遮挡也不能靠放大恢复。


2026-11-10 已在28的GPU0～3完成全量只读探针：93条训练录制、2881个去重观测、两组噪声、四个原英文prompt，共23048个动作块；四分片身份/覆盖检查、目标反归一化和Delta还原、同观测图像/状态相等、首样本同噪声重复推理均通过。摘要见`subtasks/records/normal93_20261110_prompt_probe.json`，完整报告为服务器`outputs/normal93_prompt_full_28_20261110/summary_final.json`，原预测保留在分片npz中，未改权重。

前15步（最多0.5秒且不越阶段）pickup/return条件变化与噪声变化的RMS比值逐样本中位数：右臂1.659、右手0.972。分阶段右手比值分别为0.81/0.67/1.23/1.94/1.63/0.75/0.76，因此不能把右手总体值解读为所有阶段都不听指令。正确prompt在归一化误差上严格优于另外三种prompt的比例为右臂70.9%、右手68.9%；绝对关节误差口径分别67.3%、63.9%。这些是相对原示教的兼容性诊断，**不是切换或抓放成功率**。平均误差矩阵在各四类记录条件下均以原prompt最低，说明不能仅凭语言token数量少就判定模型忽略语言。

adjust→pickup边界，使用新prompt更接近真实后续15步示教的比例：边界帧右臂38.7%/右手59.7%，边界后15帧右臂79.6%/右手82.3%。两次观测不同，不是测得0.5秒响应延迟；单条示教也不能提供错误prompt下的正确反事实动作。全量原训练50步窗口有27342/410456（6.661%）跨阶段，15步为1.903%，5步为0.544%。已有模型未接受“不换指令无限保持当前阶段”的严格协议训练，不能把跨阶段样本比例当模型自行切换率，也不能仅mask掉跨阶段loss就声称学会保持。

固定示例`ep089_phase2_actions.png`展示在同一手持观测与相同噪声下四种指令的关节目标差异；局部图比较含3条录制×7阶段全图、中心半幅及右下候选ROI。局部图尚未进入新模型训练。下一阶段优先验证实际切换时旧动作块清理、阶段完成/保持协议及手部接触可见性；需要新的独立录制和可执行的继续/放回分支示教，才能评价闭环服从及保持能力。保留当前B，不据本次诊断切换数字编码、增加视觉token或改默认部署配置。92项相关Python测试通过。

### 交互区域逐帧裁剪样例（不训练）

参考 [FoveaVLA论文附录A](https://arxiv.org/html/2610.20064v2#A1) 和 [官方数据准备代码](https://github.com/Stanford-TML/FoveaVLA/blob/main/src/droid_foveation/repack_direct.py)：其训练框来自RoboInter的手、操作物和目标区域标注的最小包围框；视觉输入在框外扩1.5倍取正方形，边界处移动/缩小，无法容纳时补零，再从原分辨率直接缩放224。其在线策略预测下一查询时刻的框，使用PaliGemma位置token；不是按阶段选择一张固定照片。训练阶段当前框用于当前裁剪，未来框用于预测监督。论文还使用外置RGB-D，不代表我们的三路RGB加crop已复现全部方法。

`scripts.tomato_interaction_preview`仅实现离线数据预览。固定episode89（原始B/1106_750_791_normal的episode4），pickup `[0,168)`与第一次采摘接触 `[225,315)`，总258帧/8.6秒；后者不含完整harvest运输回篮流程。选框由编码助手目视指定近似关键帧组件框，帧间线性插值，再按组件并集生成交互框。**没有自动检测/跟踪器，没有学会预测框，标注未经独立人工复核，插值使用前后关键帧，不能用于声称在线定位能力。** 抓取保留右手和串，首次采摘保留双手及串/摘下果实；整段harvest若要预处理，还需单独标注接近、摘取、搬运、投放的目标区域，不可套用此短片框。

在28通过CPU完成提取（未使用27或启动GPU训练）。按同一源帧保存原三视图的224预览和第四张独立crop，另存原生裁剪、逐帧索引、原视频SHA、对照图及30 FPS视频。视频黄色框是交互框，青色框是含上下文的裁剪框；框和标题仅用于展示，不烧录进独立crop。正常三图与局部图为不同视图，未把图拼成长条后整体缩回224。此224全图仅为等比例补黑的预览，尚未接入训练增强。源全局图仅640×400，增加的是局部像素预算，无法补回遮挡或录制已丢失的细节。

28最小复现（`--videos`目录的cam0/1/2.mp4分别链接至该episode对应原视频；哈希不符立即拒绝）：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= bash scripts/python.sh \
  -m scripts.tomato_interaction_preview \
  --videos outputs/normal93_interaction_source_20261110 \
  --annotations training_configs/experiments/tomato_sharpa/subtasks/records/normal93_20261110_interaction_preview_annotations.json \
  --output outputs/normal93_interaction_preview_new
PYTHONPATH=src bash scripts/python.sh -m pytest -q tests/test_tomato_interaction_preview.py
```

实际输出为28共享工程的`outputs/normal93_interaction_preview_28_20261110_v2/`；源帧与三路时间戳、独立crop像素、导出视频帧数/帧率均已校验，结果见`subtasks/records/normal93_20261110_interaction_preview.json`。新增9项裁剪测试和既有10项prompt探针测试通过。抓取的局部图明显放大接触区；双手采摘框更宽，放大收益较小。与224等比例全图比较，线性放大约1.66～4.48倍，**不是清晰度测量或策略成功率提升**；原腕部图有时更清楚。保留B不变，不开新训练。下一步先复核短片框随动作移动的覆盖质量，再决定组件标注/自动定位方案和独立图像输入适配；部署不能依赖预先知道真实阶段或录制时间的选框表。

2026-11-13 23:52首次交接发现原进程继承nohup的SIGINT忽略掩码`0000000001001007`，60秒后控制器拒绝再启动，4000完整checkpoint保留。修复退出信号选择；`recover-stopping`只允许前次控制器已退出、尚无新PID、相同原进程身份/报告及完整4000与GPU归属再次通过时恢复该次交接，保留前次尝试记录；其他阶段仍不可重复启动。修复后重新执行完整输入/吞吐验收，结果和实际恢复情况以服务器回执为准。

2026-11-14 00:00缓存交接成功：重新完整验收100条300组输入相等，32批后24批供数均值0.1956秒/batch，13项交接测试通过。原进程正常SIGTERM退出后，新PID873695恢复4000完整模型/optimizer，日志确认sample position1024000；实际4000/4010/4020步loss与梯度有限，8卡100%利用率，UI任务`b7ebaba1ed305b17bc01d9a2c149c3e4` API状态running。初步稳定5.7～5.8秒/步，预计11月14日19～20点完成（暂估）；16000尚未完成。成功切换后自动检查`4000-rgb`已暂停。实际回执及证据见`subtasks/records/complete100_cached_resume_20261114.json`，原4000及原数据保留。


### 五阶段手中调整119条（2026-11-16独立准备）

来源是28的`/mnt/data/train_platform/datasets/tomato_sharpa/normal/full-block-more/subtask-roi`，normal全119条/357970帧，全部用于训练。v3顺序为pickup→harvest→adjust→harvest→return；adjust是在手中调整位置/朝向，最终return仍保留。四路RGB、实测state116取位置58、action58按手臂14维Delta/手部44维Absolute、50步动作窗口保持既有适配器。不使用深度、Agent或Memory，不覆盖旧七阶段100条配置/统计/checkpoint。数据目录A/B是来源分组，本次统一生成独立B阶段指令副本。

新契约`openpi.subtasks.tomato_in_hand`严格检查119条身份、v3四种原文Prompt、完整五段边界、reviewed、标注绑定的所有原视频与Parquet SHA。`scripts.prepare_tomato_in_hand119`只向新目录写独立复制的四路视频、物理值不变的Parquet以及逐帧task_index/coarse_phase_index和图像availability；availability只表示当前局部RGB存在，不代表独立语义定位准确率。ROI仍是使用前后坐标的离线裁剪，不能直接声称在线方法。

配置模块：`training_configs.experiments.tomato_sharpa.subtasks.in_hand119_b_crop`，新实验`tomato_in_hand119_b_crop_delta_v3_r1_s42_b256_s16000_c4000`，独立assets `tomato_in_hand119_rgb_crop_v3_r1_delta_30hz`。batch256/FSDP8/16000步/每4000完成步保存；原学习率warmup1000/decay30000和随机种子不变，从base权重重新训练，不自动恢复旧100条optimizer。视觉1024 tokens、文本上限256（包括58维离散state），使用实际SentencePiece审计新Prompt上界。

默认训练副本：`/home/mengxiang/workspace/data/sharpa/full-block-more/in_hand119_four_view_v3_r1`，缓存：同父目录`in_hand119_rgb224_cache_v3_r1`。先准备副本，使用`scripts.compute_experiment_norm`重算全部119条，使用`scripts.build_rgb_frame_cache`逐帧顺序解码为精确PIL224缓存；`scripts.check_tomato_in_hand119`验收原文件SHA、物理列不变、逐帧监督、所有阶段起止实际Prompt、357组完整模型输入逐位等价及32批CPU供数。训练GPU容量需区分历史同形状batch256验证与本次实际开训。

验收报告通过后默认只检查：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' bash scripts/python.sh -m scripts.tomato_in_hand119_run check
```

用户另行要求正式开训时，同入口改为`start`，锁定8卡并检查空闲，启用缓存及W&B offline，登记训练UI。该命令失败时不要重复开训，先查console和checkpoint。


2026-11-16实际准备完成（尚未正式开训）：119条/357970帧来源SHA及物理列保持一致，119条全部v3五阶段。新统计覆盖357970个state和17898500个50步动作位置，357个真实LeRobot样本对照通过。476路视频全部解码，2380次参考像素检查；357组实际完整模型输入像素/dtype/state/action/token相等，357970帧监督及1190个阶段起止Prompt检查通过；357970张局部图全帧检查黑帧为0。真实抽样文本token长度188～235，理论最坏253～255；视觉1024/文本256不变。32批CPU供数测试后24批均值0.1322秒/batch（不含GPU计算），39项回归通过。启动入口check已通过，28的8卡当时空闲；本次尚未运行GPU训练步骤或确认本次显存峰值。

验收回执：`subtasks/records/in_hand119_server28_preparation_20261116_r1.json`；完整绑定代码/数据/cache/norm的启动报告在服务器`logs/in_hand119_preflight/accepted_20261116.json`。新norm SHA为`8b06358841e0570ff842883ab8a49a0fd3c9be7b68b59820d93741cd8a06ccde`，独立统计也镜像到本地ignored assets。原ROI发布manifest保持原值，其历史training_ready=false不被改写；本次独立接纳报告通过后训练入口才放行。缓存为完整精确PIL v2，约215.54GB。源目录、旧100条数据/配置/统计/checkpoint未改写。


2026-11-16用户更新本次训练设置：使用27服务器UI手动启动，119条五阶段B配置改为30000步、每5000完成步保存并全部保留（5000/10000/15000/20000/25000/30000），batch256/FSDP8与原学习率计划不变，新实验名`tomato_in_hand119_b_crop_delta_v3_r1_s42_b256_s30000_c5000`。复用共享NAS上的数据、norm与精确缓存；原16k准备验收记录属于历史，不是此次训练完成记录。新输入接纳报告为`logs/in_hand119_preflight/accepted_20261116_s30000_c5000.json`。UI已加入119条配置，27的独立UI通过`PI05_UI_DEFAULT_BASELINE`默认选择本配置，W&B默认offline；只在用户点击启动后才创建训练任务，worker会核对接纳报告并传入该实验的RGB缓存环境。其他方案不会继承该缓存，旧100条训练不修改。本轮不点击启动。

27 UI实际就绪：`http://127.0.0.1:18767/`转发到27的localhost:8768，页面默认119条B/30000步/batch256/FSDP8/每5000保存与保留/offline，已选0～7卡；没有点击启动。27重新完成357组模型输入等价、全1190阶段端点、全357970帧监督/局部图黑帧0检查，32批后24批CPU供数均值0.4415秒/batch；UI数据统计API显示“已验证，可复用”，86项相关测试和JS语法检查通过。UI worker会校验接纳报告并使用该实验的精确缓存，未启动GPU训练。证据为`subtasks/records/in_hand119_server27_ui_20261116_r1.json`。

### 可选逐时间步归一化、相关噪声与 Train RTC

普通「番茄 · Sharpa / 全参」和个人配置共用一个页面。训练参数中的两个下拉框选择全局 Prompt 或阶段指令、三路 RGB 或四路 RGB＋ROI。每个样本只读取当前图像及 state。各输入模式、T1/T2/T3、Train RTC 和分组加权采样都使用用户选择的一个或多个兼容数据目录。

阶段指令优先读取源目录的 `annotations/subtasks/coarse/episode_*.json`，也可读取 `annotations/episode_*.json`。每段用 `start_frame`（含）、`end_frame`（不含）、`text` 定义。没有分段标注的录制使用其原逐帧 `task_index` 和 `meta/tasks.jsonl`。全局模式使用配置中的英文总体任务指令。检查数据会列出实际 Prompt。源视频和物理表格保持只读，标注文件在加载时读取一次，并绑定到统计与训练身份。

三路输入读取外部与双腕 RGB。四路另读取 `observation.images.interaction`，可使用对应的可用性布尔列；仅有第四路视频时按该视频可用处理。缺少第四路的视频目录仍可选择，四路启动预检会提示缺失的相机字段。

双臂逐时间步 Z-score 为 T1，相关高斯噪声为 T2，同时启用为 T3。T1/T3 的手部和当前 state 继续使用分位数归一化。相关高斯噪声作用于完整动作块，使用 β=0.5。Train RTC 独立设置，0 关闭，正值为可抽取的最大干净动作前缀长度，必须小于动作 horizon。

论文模式在每条录制内选择完整合法动作窗口。归一化统计和噪声估计对窗口等权，训练抽样可逐轮等权打乱或按组权重有放回采样。按目录识别分组后可修改权重。切换图像、指令或归一化选项需要独立统计资产，单独修改 RTC 和组权重可复用匹配的统计并创建新实验。

通过 `PI05_RGB_FRAME_CACHE` 指定可选缓存。缓存按录制和相机匹配，文件修改时间漂移可继续使用；缺失、纠错后不匹配或不可读的条目逐路回退视频。多 worker 保序预读并按已完成步数恢复样本位置。
