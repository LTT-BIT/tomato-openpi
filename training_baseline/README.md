# 完整任务训练版本

每个版本放在同级目录；配置文件与教程在各自目录根部。本目录只保留版本索引。

| 版本 | 入口 | 训练约束 |
| --- | --- | --- |
| [lora_base_v1](lora_base_v1/) | [配置](lora_base_v1/pi05_fr3_wuji_fulltask_c0_train.py) | 1002 派生的双 LoRA 基准，20k 步；默认旧采样行为，W&B 默认关闭 |
| [lora_base_v2](lora_base_v2/) | [配置](lora_base_v2/pi05_fr3_wuji_fulltask_c0_train.py) | 双 LoRA、20k 步；增加无放回采样、独立种子、数据审计与恢复，W&B 默认开启 |
| [fullft_v3](fullft_v3/) | [配置](fullft_v3/pi05_fr3_wuji_fulltask_fullft_30hz_train.py) | 本次 441 episode、60,153 步、非 LoRA 全参数微调；[实际参数](fullft_v3/本次实验_参数与约束.md) |

v1 和 v2 的模型及优化超参数大体相同，但采样、恢复审计和 W&B 默认行为不同，因此都保留。v3 的配置、教程、交接文档在 [fullft_v3](fullft_v3/)；环境、下载、数据准备和启动程序集中在 [scripts](fullft_v3/scripts/)，各文件用途见该版本的 README。

已完成实验 `tomato_all_full_fullft_30hz_s42_run01` 的审计记录了原训练代码路径和哈希。若要严格恢复该实验，检出原训练 commit `4d9b349af63cd2cdd8ffb67d579eff04a30b34e5`；当前整理后的代码用于查看历史和启动新实验。权重、数据和 W&B 日志不提交 Git。
