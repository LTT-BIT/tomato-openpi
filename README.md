# Pi05 训练仓库

集中维护 π0.5 训练、数据准备、本体适配和实验配置。不同本体共用训练框架，在同一主线维护各自配置；新增实验无需修改中央注册表。当前公共方案分别从 **LoRA v1** 和 **全参 v1** 开始编号，历史来源为原仓库 v2 双 LoRA 与 v3 全参数微调。

## 目录与修改位置

| 目录 | 内容 | 何时修改 |
| --- | --- | --- |
| `src/openpi/` | 共享模型、训练循环、采样、恢复审计及数据变换 | 公共训练功能开发 |
| `training_configs/embodiments/` | 本体规格及训练数据适配 | 新本体、关节顺序或相机映射 |
| `training_configs/common.py` | Dataset 公共结构及运行目录辅助函数 | 修改公共配置结构 |
| `training_configs/recipes/` | 带版本的公共 LoRA/全参训练方案 | 发布团队新方案 |
| `training_configs/experiments/<负责人>/` | 本次本体、方案、数据、路径、统计与参数覆盖 | 独立实验与调参 |
| `scripts/run_training.py` | 统一 JAX 启动、检查、日志与并发保护 | 公共启动功能 |
| `examples/` | 训练数据转换、合并、统计工具 | 准备训练输入 |
| `packages/openpi-client/` | 训练所需的图像工具子集 | 共享图像处理 |
| `tests/fixtures/baselines/` | 从原 v2/v3 配置生成的参数快照及来源指纹 | 基准回归核对 |
| `docs/archive/` | 历史教程及实验记录 | 查询历史事实 |

现有部署目录、服务端、机器人执行器、客户端运行时、部署打包脚本、仿真运行环境及硬件子模块已移出本分支。`src/openpi/policies/` 保留训练需要的数据适配器，模型动作采样方法保留用于模型接口与测试。

## 快速开始

安装 uv，在仓库根目录执行：

```bash
export PI05_RUNTIME_DIR=/你的本地磁盘/pi05-runtime
bash scripts/setup_training_env.sh
source scripts/training_env.sh

# 展开配置，无需数据、权重或 GPU。
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.run_training \
  training_configs.experiments.baselines.tomato_fr3_lora_v1 --dry-run

# 已准备好数据与对应统计后检查。
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.run_training \
  training_configs.experiments.baselines.tomato_fr3_lora_v1 --check

# 按团队约定分配 GPU，新训练使用独立实验名。
CUDA_VISIBLE_DEVICES=0 "$UV_PROJECT_ENVIRONMENT/bin/python" -u -m scripts.run_training \
  training_configs.experiments.baselines.tomato_fr3_lora_v1 \
  --exp-name alice_tomato_lora_v1_s42_run01 --wandb-mode online
```

默认运行目录为 `.runtime/`，实际数据、统计和初始权重需要自行准备。数据设置直接写在各自实验文件的 `build_dataset()` 中；LoRA 基准的数据位置是占位示例，全参基准保留合并完整任务数据的历史身份和统计审计哈希。完整步骤见 [训练使用说明](docs/训练使用说明.md)。

## 协作与历史

阅读 [多人开发 README](docs/多人开发README.md)。个人调参放在个人实验文件，公共 recipe 显式版本管理；使用短期开发分支，经 PR 合入主线。训练固定 commit 和工作目录，避免运行期间修改源码。

活动配置统一放在 `training_configs/`。旧 `training_baseline/` 目录已删除，旧代码通过原分支与 Git 历史查询；历史文档在归档目录，参数回归使用静态快照，不再导入旧训练入口。

`experiments/baselines/` 是两套公共基准示例，`experiments/template/` 是独立个人模板；后续每人建立自己的目录。启动程序只加载指定模块并调用其 `build_config()`，不会运行目录中的其他实验。具体工作流程见多人开发 README。

[基准迁移记录](docs/基准迁移与归档.md) 说明参数对应、历史实验状态和导出范围；[归档索引](docs/archive/README.md) 提供旧教程、实际参数和交接记录。

本轮完成仓库整理与代码验证，未重跑完整真实数据训练。历史 v3 的训练完成不代表新仓库已重新训练。PyTorch 训练实现保留，新分层配置标准入口为 JAX，不承诺两种后端的采样及恢复行为一致。

## 导出到新仓库

导出已提交的训练代码快照，排除本机数据、权重、缓存和旧 Git 历史：

```bash
git archive --format=tar.gz --output=/tmp/pi05-training-only.tar.gz HEAD
mkdir /你的新目录/pi05-training
tar -xzf /tmp/pi05-training-only.tar.gz -C /你的新目录/pi05-training
cd /你的新目录/pi05-training
git init -b main
git add .
git commit -m "初始化 Pi05 训练仓库"
```

直接推送当前分支到新 remote 会保留旧提交历史，其中仍包含过去的部署代码；希望新仓库从纯训练快照开始时采用上述方式。原 Pi05 分支和本机未跟踪的数据、权重仍保留。

本项目基于 Physical Intelligence OpenPI，保留 [LICENSE](LICENSE) 与 [Gemma 许可](LICENSE_GEMMA.txt)。
