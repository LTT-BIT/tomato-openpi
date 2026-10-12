> 历史归档：原路径、命令和实验状态属于整理前仓库；当前操作见 [训练使用说明](../../../训练使用说明.md)。

# 全参数微调 v3

本目录的 [训练配置](../../../../training_baseline/fullft_v3/pi05_fr3_wuji_fulltask_fullft_30hz_train.py) 对应已完成的实验 `tomato_all_full_fullft_30hz_s42_run01`。实际参数见 [参数与约束](本次实验_参数与约束.md)，操作步骤见 [训练教程](训练教程_完整任务C0.md)，换机及权重导出见 [交接文档](跨服务器交接_完整任务C0.md)。

`scripts/` 是本版本的辅助程序，不是额外训练配置：

| 文件 | 用途 |
| --- | --- |
| `env.sh` | 设置独立运行目录、缓存、临时目录和模型缓存路径 |
| `setup_env.sh` | 按 `uv.lock` 安装独立 Python 环境并检查 GPU |
| `download_locked_packages.py` | 下载并校验锁文件中的依赖包，供网络不稳定时重用 |
| `prepare_all_full.py` | 合并三套完整任务数据并计算对应归一化统计；仅首次准备数据使用 |
| `verify_base_weights.py` | 校验官方 π0.5 初始权重并发布缓存；仅首次下载后使用 |
| `launch_when_idle.py` | 等权重齐全且 8 张 GPU 空闲后启动；本次实验曾使用 |
| `train_all_full.sh` | 直接启动训练，或传 `--resume` 恢复未完成的同一实验 |

`tests/fullft_config_test.py` 检查全参数模型、数据来源和配置约束。首次准备与历史启动工具保留为可复核记录；新阶段训练通常只需配置、环境脚本和训练入口。本次实验已经完成既定步数，不能靠原实验的 `--resume` 增加预算；严格恢复原实验须检出训练 commit `4d9b349af63cd2cdd8ffb67d579eff04a30b34e5`。
