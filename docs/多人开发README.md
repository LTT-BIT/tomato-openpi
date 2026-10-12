# Pi05 多人开发 README

## 配置归属

同一任务不同本体在同一主线管理，各本体提供自己的规格与训练数据适配，公共模型和训练循环共用。

| 改动 | 位置 | 规则 |
| --- | --- | --- |
| 个人学习率、batch、步数、seed | `training_configs/experiments/<负责人>/` | 独立文件，通过 dataclasses.replace 覆盖 |
| 团队训练方案升级 | `training_configs/recipes/` | 建立新 recipe 版本，旧实验继续引用旧版 |
| 数据及统计变化 | `training_configs/datasets/` | 新版本、新 asset_id，记录来源与统计哈希 |
| 本体维度、顺序或动作语义变化 | `training_configs/embodiments/` 及对应适配器 | 新规格版本，验证完整输入变换 |
| 训练循环、采样或恢复改进 | `src/openpi/training/` | 独立 PR，说明影响并跑回归 |

本训练仓库的各类版本独立从 v1 起步：当前公共方案是 pi05_lora_v1 与 pi05_fullft_v1；数据文件按内容命名为 tomato_fr3_fulltask_v1 和 tomato_fr3_all_full_v1。数据版本不跟随 LoRA/全参方案编号，同一数据可被两种方案复用。历史 v2/v3 只表示来源，保留在溯源记录与归档中。

公共版本的 bug 修复可以修改原文件，但必须记录影响与 commit；版本名称不能替代源码身份。历史实验使用原 commit 和最终配置快照。参数回归使用 `tests/fixtures/baselines/` 的原基准静态快照，旧代码从 Git 历史查询，不把新实验改写成旧实验。

## 新建个人实验

从模板建立自己的配置：

```bash
git switch main  # 新仓库初始化后的主线；源仓库分支名以 git branch 为准。
git switch -c exp/alice-tomato-lr
mkdir -p training_configs/experiments/alice
touch training_configs/experiments/alice/__init__.py
cp training_configs/experiments/template/tomato_lora_lr1e5.py \
  training_configs/experiments/alice/tomato_lora_lr1e5.py
```

修改实验名、owner 和 purpose，按需要覆盖参数：

```python
from dataclasses import replace
from training_configs.experiments.baselines.tomato_fr3_lora_v1 import build_config as build_baseline

def build_config():
    baseline = build_baseline()
    return replace(
        baseline,
        exp_name="alice_tomato_lora_lr1e5_s42_run01",
        lr_schedule=replace(baseline.lr_schedule, peak_lr=1e-5),
        policy_metadata={**baseline.policy_metadata, "owner": "alice", "purpose": "compare_learning_rate"},
    )
```

通过模块路径加载，不需要中央注册：

```bash
"$UV_PROJECT_ENVIRONMENT/bin/python" -m scripts.run_training \
  training_configs.experiments.alice.tomato_lora_lr1e5 --dry-run
```

个人实验可以直接引用 recipe、embodiment 和 dataset，避免依赖他人的实验文件。模板引用的是公共基准组合。

换数据时创建新的 Dataset 和版本，更新 repo_id、来源、prompt 示例、asset_id 与统计哈希；不能仅将新数据放到旧数据路径。同一内容的路径迁移可使用环境变量。

新本体实现 `recipes/common.py` 的 Embodiment 接口，提供 action_dim、raw_state_dims、action_hz、camera_keys 与 data_config()。当前启动流程要求单个显式本地 LeRobot 数据集，data_config() 需提供对应数据字段；适配器负责关节顺序、单位和动作表示。修改模型内部维度或增加 padding，需配套修改适配、统计检查和权重加载，不能直接修改 FR3 的维度数字。

## 分支与评审

每人针对一次任务建立短期 `exp/<负责人>-<主题>`、`feat/<功能>` 或 `fix/<问题>` 分支，验证后提 PR 合入主线。所有配置最终都在主线，不共享持续修改的开发分支，也不按人或本体保留永久分支。

个人实验 PR 不混入公共 recipe 修改；公共方案变更单独评审，注明影响哪些实验、是否改变冻结范围或采样行为。合并后删除任务分支，正在训练的任务继续使用固定源码。

PR 说明目的、基准、覆盖参数、数据身份、统计引用、英文 prompt 来源和验证结果。区分准备完成与训练完成；只有真实日志和完整 checkpoint 支持时才记录训练完成。本训练仓库不安排部署或实机验证。

## 多人训练隔离

运行名建议为 `<负责人>_<任务>_<方案>_s<seed>_run<编号>`，重复运行另用编号。checkpoint 路径为 `<checkpoint_base_dir>/<config.name>/<exp_name>/`，W&B 名称与实验名一致。

使用独立 checkout 或 worktree，运行期间不切分支、不拉新代码。显式分配 CUDA_VISIBLE_DEVICES；每人使用自己的运行目录、缓存和环境，数据可只读共享，发布后不能原地修改。目录锁只提供本机相同输出路径的并发保护，跨主机或共享文件系统还需团队调度和输出目录分配。

```bash
git worktree add /你的工作目录/pi05-alice <已提交分支或commit>
cd /你的工作目录/pi05-alice
export PI05_RUNTIME_DIR=/你的本地磁盘/pi05-alice-runtime
source scripts/training_env.sh
```

本地路径、数据、权重、日志、W&B 和密钥不提交 Git。配置写稳定身份与环境变量入口，换机器修改环境变量。

## 验证与提交

配置变更先执行 --dry-run；有准备好的数据和统计时再执行 --check。公共训练改动跑相关回归：

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu \
  "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest \
  tests \
  src/openpi/training/sampling_test.py src/openpi/training/reproducibility_test.py \
  src/openpi/training/weight_loaders_test.py src/openpi/training/data_loader_test.py \
  src/openpi/policies/fr3_wuji_policy_test.py src/openpi/transforms_test.py \
  src/openpi/shared/normalize_test.py scripts/train_reproducibility_test.py -q
```

验证后在 docs/开发日志.md 记录变更、结果与未完成事项，用中文提交说明，只提交本任务文件。精确恢复保留原源码、数据、统计、完整 checkpoint 和采样审计。

当前审计对训练源码、配置目录与工具生成哈希，新增其他实验文件也可能改变源码审计。续训必须使用原固定 checkout；代码升级后创建新实验，不在移动的主线上续训旧实验。
