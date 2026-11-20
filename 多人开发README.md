# Sharpa 番茄工程协作

本分支只维护 Sharpa 番茄采摘。安装和启动见[仓库说明](README.md)，数据准备、版本发布和训练选项见[Sharpa说明](training_configs/experiments/tomato_sharpa/README.md)。

## 目录与配置的组合关系

```text
training_configs/
├── common.py                       # 数据身份和仓库路径
├── recipes/                        # π0.5全参、LoRA公共训练配方
├── embodiments/fr3_sharpa*.py       # Sharpa本体、具名向量和相机协议
└── experiments/tomato_sharpa/
    ├── tasks/
    │   ├── single_pick.py          # 普通单颗采摘入口
    │   ├── truss_plate.py          # 整串托盘调整入口
    │   ├── truss_in_hand.py        # 整串手中调整入口
    │   ├── profiles.py             # 流程、阶段、英文指令
    │   ├── dataset.py              # 数据版本协议和监督检查
    │   └── common.py               # 流程＋版本＋训练配方组合
    ├── baseline/                  # 两份普通Sharpa参考配置
    ├── subtasks/                  # 历史实验兼容入口和真实来源记录
    └── template/<负责人>/         # UI生成的本地个人配置，Git忽略
```

训练入口固定；数据数量、来源清单、版本和统计身份属于数据发布。不要为新增条数复制训练模块。新增流程必须有不同的任务语义，不能只因Prompt或视图不同新建流程入口。

## 修改与验证

- 公共模型、数据加载、归一化位于 `src/openpi/`；实验模块组合现有能力，不复制训练循环。
- `scripts.compute_experiment_norm` 和 `scripts.run_training` 使用同一个配置，统计与训练共用向量变换。共享向量统计实现为 `scripts.compute_robot_norm_stats`；旧examples路径仅作兼容导入。
- 阶段标注变化、数据增量、动作监督或视图变化，发布新数据／实验身份并重新核对统计。新版本不得修改旧数据目录、统计或checkpoint。
- UI保存完整参数及数据清单SHA256。个人配置不上传Git；公共修改在本分支中文提交并推送。数据、权重、密钥和本地缓存不进入Git。
- 修改公共行为时运行 `scripts/check_training.sh` 对应的静态检查、配置dry-run和CPU回归；GPU训练和机器人验证另行记录，不能用CPU测试替代。

## 运行中的服务器

本次本机整理期间不更新27/28共享checkout，不修改或重启四路训练、三路冻结队列及部署。训练及队列结束后另行核对最终checkpoint、运行配置和待同步提交，再同步工程。旧任务继续使用原代码、原数据与原统计。
