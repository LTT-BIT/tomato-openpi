# 完整任务 C0 配置索引

每个版本的配置和教程成对保存。顶层两个同名文件恢复为本次全参数训练**之前**的 v2 基准，便于继续核对当时的训练约束；不要用它们启动 v3 全参数训练。

| 版本 | 配置与教程 | 用途 |
| --- | --- | --- |
| v1，1002 派生 | [c0_v1_legacy](c0_v1_legacy/) | 历史双 LoRA、20k 步简明基准；源自 Git `6ba2cf3` |
| v2，可复现采样 | [顶层配置](pi05_fr3_wuji_fulltask_c0_train.py)、[顶层教程](训练教程_完整任务C0.md) | 历史双 LoRA、20k 步、无放回采样与审计；从 Git `432eb14` 原样恢复 |
| v3，全参数训练 | [c0_v3_fullft](c0_v3_fullft/) | 本次 441 episode、60,153 步实验的独立配置、[实际参数](c0_v3_fullft/本次实验_参数与约束.md)、准备脚本、训练教程和跨服务器交接 |

v3 的实际入口是 `python -m training_baseline.c0_v3_fullft.pi05_fr3_wuji_fulltask_fullft_30hz_train`，启动脚本为 `training_baseline/c0_v3_fullft/train_all_full.sh`。v1/v2 的配置不会读取本次合并数据的路径，也不会默认使用本次的全参数模型。

已完成实验 `tomato_all_full_fullft_30hz_s42_run01` 的审计记录了原训练代码文件路径及哈希。若要按原实验进行严格断点恢复，应检出训练 commit `4d9b349af63cd2cdd8ffb67d579eff04a30b34e5`；当前目录整理提交适用于查阅历史和启动新实验。权重、数据与 W&B 日志仍存放在 Git 忽略的运行目录，不在本目录提交。
