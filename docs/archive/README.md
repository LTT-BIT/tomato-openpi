# 历史文档索引

本目录保存历史记录，原命令和路径不保证在训练专用仓库执行。当前使用 [训练说明](../训练使用说明.md) 与 [多人开发 README](../多人开发README.md)。原配置和历史脚本从原分支或 Git 历史查询；当前树已删除 training_baseline/，活动配置统一在 training_configs/。

| 记录 | 入口 |
| --- | --- |
| v1 | [README](baselines/lora_base_v1/README.md)、[教程](baselines/lora_base_v1/训练教程_完整任务C0.md) |
| v2 | [README](baselines/lora_base_v2/README.md)、[教程](baselines/lora_base_v2/训练教程_完整任务C0.md) |
| v3 | [README](baselines/fullft_v3/README.md)、[教程](baselines/fullft_v3/训练教程_完整任务C0.md) |
| v3 实际参数 | [参数与约束](baselines/fullft_v3/本次实验_参数与约束.md) |
| v3 历史交接 | [跨服务器交接](baselines/fullft_v3/跨服务器交接_完整任务C0.md) |
| 整理前根 README | [历史 README](legacy/README_root_before_training_only.md) |
| 早期 FR3 微调 | [0906 纪要](legacy/0906微调.md) |
| 归一化历史说明 | [norm_stats](legacy/norm_stats.md) |

部署专用文档与代码仍在原分支，不属于当前训练树。

[归档清单](归档清单.json) 记录原路径、源 commit、原文哈希与归档后哈希。归档附提示并修复仍存在文件的链接；已排除文件的历史链接保留作上下文。

例如查看原 v2 配置：

```bash
git show 8007c1e3ae9a4d7ab811088c811bf2c1fd2a4045:training_baseline/lora_base_v2/pi05_fr3_wuji_fulltask_c0_train.py
```

如果新仓库通过快照初始化，没有原历史 commit，回原 Pi05 仓库执行。新仓库携带原 v2/v3 参数快照和源文件哈希，用于回归与溯源。
