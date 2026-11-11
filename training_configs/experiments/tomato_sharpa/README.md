# FR3 + Sharpa 番茄训练

## complete100 RGB＋局部crop：训练准备

2026-11-11起，后续动作策略使用Agent冻结的100条完整七阶段录制，**全部100条参与π₀.₅训练**。Agent原84/8/8只保留为来源信息，不据此给动作训练留出验证/测试；不把这些训练录制称为动作模型盲测。旧normal93配置、归一化、checkpoint及以下历史说明保留。

当前已实现独立四视图输入接口，**尚未准备完整四视图训练数据或启动训练**。`subtasks/records/complete100_rgb_crop_20261111_selection.json`绑定100条原始身份、媒体/Parquet/标注SHA、实际源metadata以及train=0..99。逐条核验原始文件，阶段标签保留“磁盘已保存版本”与“Agent有效边界”差异，不自动把实验边界提案变成动作监督。

8条差异已解析：逐一匹配`vjepa2_training/docs/validation/tomato_reviewed_boundaries_20261109/proposals.json`中的14个临时边界提案，状态为`provisional_user_review_not_source_labels`。动作训练统一采用磁盘正式保存且哈希核验的Subtask；这些提案仍保留为Agent离线实验的对照，**不代表提案被判错或被正式批准**。新有效清单为`records/complete100_rgb_crop_20261111_saved_labels_r1.json`，每条保存选定training_segments与来源；旧未解析清单和原始文件不改。已正式修订的`B/1106_750_791_normal/episode_000004`最终return仍为`[2908,2990)`，未被恢复为旧标签。

`records/complete100_rgb_crop_20261111_source_plan_r1.json`给出兼容现有A/B准备器的100条来源计划，已经通过`openpi.subtasks.prepare.inspect_plan`只读实检：100条、425556帧、逐帧索引、实际Parquet、媒体和标注绑定均通过。它仍是**三路原图的来源计划**，不能当成四图数据已准备或执行正式训练。最新解析验收为`records/complete100_rgb_crop_20261111_label_resolution_check_r1.json`；原preparation_check保留为上一步的历史回执。

`subtasks/complete100_rgb_crop_preparation.json`是准备契约。原三路模型键为base、left_wrist、right_wrist（原始cam0/cam2/cam1），新增interaction来自原分辨率RGB。π₀.₅图像尺寸224，Agent尺寸256，禁止从已有224预览放大成Agent原图。RGB-only，不接深度。ROI需要原图坐标、相机、时间、可用时间、有效性及人工/因果生成方式；人工插值预览不等于在线ROI能力，当前完整100条ROI尚未准备。

已对索引86的2990帧做耗时实测：干净crop视频导出5.88秒、完整检查视频/检查图15.40秒、压缩NPZ导出9.95秒。轻量与完整流程的干净视频SHA完全相同；NPZ抽查64帧像素一致。320次单帧计时中，离线插值＋正方形计算P95为0.148毫秒，裁剪缩放P95为0.395毫秒。**离线插值不是因果定位；此前每条15～30分钟只是估算，没有人工选框耗时记录，不能作为实测占比。** 单机单条结果不能外推所有录制或部署硬件。记录为`records/complete100_roi_timing_and_tracking_20261111_r1.json`。

新增`scripts.export_tomato_manual_crops --selection ... --annotations ... --out /new/path`直接导出干净224视频和逐帧坐标，不生成检查视频或JPEG；仍绑定原图哈希、拒绝写入原始目录、完整解码核验输出帧数。仅用于已有完整标注的录制，不会为其他99条猜框，也尚未发布完整LeRobot四图训练包。`scripts.benchmark_tomato_roi`可重做三种导出与单帧计时。

下一步优先开发可复用的因果定位/跟踪，不继续用逐条手工标注硬铺100条。新增`scripts.tomato_causal_roi.CausalBoxTrack`和真实回放入口`scripts.probe_tomato_causal_track`：只由第0帧目视框初始化，之后仅读当前与过去帧；运动/特征/尺度检查失败即输出`valid=false, box=null`，无明确新种子不恢复。保存持久四角多边形，避免反复旋转轴对齐框引起虚假膨胀。真实探针中双目标有效93帧，3.1秒处番茄区域漂移被拒绝；有效期间跟踪P95为2.19毫秒（CPU单线程，含灰度转换/两目标），并不表示目标身份经过语义验证。与旧近似人工预览的有效帧并集IoU均值0.774，仅为已选录制内参考，不是独立精度。该原型**没有自动初始定位、丢失恢复或双手/运送角色切换，不能用于批量ROI或在线能力声明**；跟踪失败结果没有替代现有离线裁剪。

四视图完整π₀.₅推理尚未计时，不能用适配器或轻量假编码器耗时代替。速度测试本身不要求训练结束，但需完整可加载权重与可用GPU环境。本地base权重缓存为partial；本轮只读检查28服务器8张GPU均100%利用率、每卡空闲约6.5GiB，未启动推理或排队任务。30FPS数据采样率不能直接当作策略更新频率，实时验收仍为未完成。

当前局部标注已实际推进到**1条完整录制、2990帧（净新增2732帧）**：`B/1106_750_791_normal/episode_000004`，冻结清单索引86。有效标注为`records/complete100_episode86_manual_roi_20261111_r3.json`：覆盖七阶段，按双手接触、左手运送/入篮、右手放回等操作划分20段，134个关键帧。目视检查248张关键帧及中间帧；不等于逐帧人工认证或独立质量评估。运输阶段只用左手/所持果实及放置接触区域，避免被动持串右手把局部图拉宽；原258帧几何和裁剪像素保持一致。离线子段边界有视野切换，仍需完整时序复查，不能认为框已连续平滑或在线可用。

新增本地派生输出：`/home/user/datasets/derived/tomato_complete100_episode86_manual_roi_r3`保存2990帧224 RGB数组与原图坐标，`tomato_complete100_episode86_manual_review_r3`保存完整检查视频、干净局部视频、逐帧索引和248张QA图，`tomato_complete100_roi_worklist_r2/worklist.json`更新100条工作清单，尚余422566帧没有标注。**这是1条完整几何标注候选，不是完整100条训练数据**，未合入A/B训练包。原r1工作清单及258帧预览保留为历史对照。插值产生的子段边界关键帧保留父关键帧的未来依赖，`available_at`不会因为切段而错误变成当前时刻。

可用`scripts.review_tomato_manual_roi --selection ... --annotations ... --out /new/derived/path`重新生成完整录制QA视频，输出新目录，解码核验两份视频均为2990帧并再次核验原RGB哈希。最新回执为`records/complete100_episode86_manual_roi_20261111_validation_r3.json`。下面列出的r1目录仍保留，不被覆盖。

已在本地生成以下独立派生目录，**原始视频、Parquet和Subtask只读，不覆写、不与原始媒体建立可写硬链接**：

|路径（均在`/home/user/datasets/derived/`下）|内容|边界|
|---|---|---|
|`tomato_complete100_saved_labels_r1_base_ab`|完整100条A/B基础副本、来源快照、train=0..99；425556帧|原三路图，未加入完整crop；副本仅重排索引/task映射，原物理列保持一致|
|`tomato_complete100_roi_worklist_r1/worklist.json`|100条ROI工作清单、选定阶段、原视频SHA、已有框|只包含已有258帧预览框，425298帧明确缺失|
|`tomato_complete100_roi_preview_r1`|258帧原分辨率裁剪后224干净RGB、逐帧坐标/可用时间和回执|离线插值，未独立审核，不能用于在线ROI声明|
|`tomato_complete100_base_adapter_check_r1.json`|全部基础副本及真实四图输入适配检查|无权重加载、无训练或归一化模型查询|

`scripts.prepare_tomato100_rgb materialize-base`读取已解析清单和绑定的source plan，调用现有原子发布A/B准备器。输出不能位于原始数据目录内或成为原始目录的祖先，必须是新目录；媒体使用独立复制，源文件哈希在发布前再次检查。`scripts.check_tomato100_derived`比较全部100条A/B物理列、阶段/task索引、600份视频副本SHA，并再次检查原始媒体、Parquet、metadata和标注SHA。A/B包内的`meta/stats.json`只是数据存储统计，**不是π₀.₅的Delta动作窗口归一化资产**。

第一版离线ROI沿用已经确认的示例方法：在原分辨率关键帧中目视标出当前参与交互的手部、所持/操作物及必要接触区域，部件框的紧致并集是黄色框，实际裁剪采用1.5倍上下文正方形（青色框）。机械臂连杆、无关的另一只手与篮中无关番茄不加入紧致框。逐段检查关键帧及中间帧，在运动/接触变化处补关键帧；不把原示例的坐标直接复制到其他录制。最初两个片段由14个目视关键帧插值得到258帧；当前已扩展一条完整录制，尚无完整100条标注；此方法属于人工式离线标注，不能声明具备在线自动定位能力。准备契约已记录该规则，下面的通用检测仅是诊断旁支，不替代示例标注方法。

新增稀疏检测诊断入口`scripts.probe_tomato100_roi`：使用本地Grounding DINO tiny权重，在冻结100条录制的每个已保存阶段选一个中间帧，共700帧。阶段标签只用于QA抽样，不输入检测器；检测提示固定，模型文件SHA、原视频SHA、坐标、置信度与检查图一并保存。输出目录为`/home/user/datasets/derived/tomato_complete100_roi_detector_probe_r1`。**这是物体候选框诊断，approved_roi_frames=0，不是700帧合格交互ROI，也不是425556帧已覆盖**。实际目视抽查发现整臂被识别为手、篮中番茄干扰、未单独定位放回时的杆部；不按置信度自动拼框生成训练crop，不把它声明为FoveaVLA自动注意方法复现。完整训练入口拒绝`detector_probe`来源。补齐仍需可靠的手部/交互定位与逐段质量检查，当前未启动完整四图物化或训练。诊断只用本地图片推理，不上传源图；派生目录可迁移到服务器继续处理，本地不是硬性要求。

复现诊断（需独立环境中的PyTorch CUDA、Transformers 4.56.2、PyAV/Pillow；权重提前下载，本命令不联网）：

```bash
PYTHONPATH=src:packages/openpi-client/src python -m scripts.probe_tomato100_roi \
  --selection training_configs/experiments/tomato_sharpa/subtasks/records/complete100_rgb_crop_20261111_saved_labels_r1.json \
  --model-dir /path/to/local/grounding-dino-tiny \
  --out /new/derived/detector_probe
```

模型端新增`Pi0Config.image_keys`显式开关；默认三路不变，四路配置才能保留interaction。`LeRobotFr3SharpaCropDataConfig`贯通第四路repack和`Fr3SharpaCropInputs`；缺失crop/布尔有效性时拒绝。ROI无效时该路mask为false，仍可依靠原三图，不会把旧crop当新观测。模型使用固定配置的图像顺序，避免JAX PyTree对字典键排序导致局部图位置漂移。JAX与PyTorch前处理都支持显式四键；interaction保留颜色增强，但不做会裁去接触边缘的随机裁剪或旋转。旧三图路径和RTC算法未改变。

候选训练入口为`training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop`，由`PI05_TOMATO100_CROP_BUNDLE`显式指定未来完整四图包。它要求`crop_manifest.json`绑定选定100条清单、完整覆盖和ROI来源，还要求四路视频、有效性列及train=0..99；当前基础三图包会被拒绝。保留batch512/8000步/FSDP8与原学习率计划作为起点，显存尚未验收。这个入口不训练历史token，方案一的Memory仍只给Agent。旧normal93入口不替换。

原三图视觉768＋文本槽位256＋动作位置50；四图基线视觉1024＋文本槽位256＋动作位置50。新增crop增加256个视觉token，不占256文本上限。方案一历史只给Agent，π₀.₅不追加历史。方案二仅预留设计候选：额外32/64个学习型历史token，或文本上限320；这些接口尚未实现、未训练，不能把配置数字当作模型已经接入。动作50个位置各投影58维连续向量，不是2900个文本token。

实际PaliGemma分词器的预算回执在`subtasks/records/complete100_rgb_crop_20261111_token_budget.json`。包含每类257个统一状态值＋200组混合状态，共1828个合成状态组合；最大完整文本长度pickup254、harvest255、return253、adjust252。合成预算不代表真实100条的长度分布。新100条归一化尚未拟合，真实P50/P95/最大值保持null，不能套旧93条统计。

准备工具默认不创建训练数据或调度训练，输出必须为新文件：

```bash
PYTHONPATH=src python -m scripts.prepare_tomato100_rgb selection \
  --manifest /home/user/vjepa2_training/runs/tomato/complete100_20261111/dataset_manifest.json \
  --out /new/path/selection.json
PYTHONPATH=src python -m scripts.prepare_tomato100_rgb tokens \
  --tokenizer /path/paligemma_tokenizer.model \
  --out /new/path/token_budget.json
PYTHONPATH=src python -m scripts.prepare_tomato100_rgb resolve-labels \
  --selection /path/frozen_selection.json \
  --proposals /home/user/vjepa2_training/docs/validation/tomato_reviewed_boundaries_20261109/proposals.json \
  --out /new/path/resolved_selection.json \
  --plan-out /new/path/source_plan.json
PYTHONPATH=src python -m scripts.prepare_tomato100_rgb materialize-base \
  --selection /path/resolved_selection.json --plan /path/source_plan.json \
  --tokenizer /path/paligemma_tokenizer.model --out /new/derived/base_bundle
PYTHONPATH=src python -m scripts.prepare_tomato100_roi \
  --selection /path/resolved_selection.json --preview /path/preview_annotations.json \
  --out /new/derived/roi_worklist
```

将来可向tokens子命令传`--state-bins-jsonl`，每行明确提供action、observation_id（应含录制与帧身份）、normalization_sha256和新归一化后实际使用的58个state_bins；工具统计完整输入长度与超限数，不截断文本。拒绝混用归一化和重复观测。该输入必须由真实标准化/离散化管线导出，不能直接用原始关节角或合成bins充数。

后续仍需：补齐ROI并物化完整四图包、生成新100条模型归一化统计和真实token分布、检查预训练权重加载及真实归一化batch前反向、在目标训练GPU验收显存。独立CPU环境的接口/梯度测试不代表完整π₀.₅训练或GPU验收。旧三图模型默认行为不变。

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
