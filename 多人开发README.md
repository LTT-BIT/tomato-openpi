# 多人开发 README

安装和训练命令见 [仓库 README](README.md)。本文件约定配置归属、协作方式和实验记录。

## 文件放在哪里

| 内容 | 位置 |
| --- | --- |
| 任务协议、默认数据及训练说明 | `training_configs/experiments/<任务>/README.md` |
| 公共参考配置 | `training_configs/experiments/<任务>/baseline/` |
| 个人实验配置与结果记录 | `training_configs/experiments/<任务>/template/<负责人>/` |
| 公共训练方案 | `training_configs/recipes/` |
| 本体规格和适配 | `training_configs/embodiments/`、`src/openpi/policies/` |
| 模型、数据加载与训练循环 | `src/openpi/` |
| 历史有效实验记录 | `docs/records/` |

当前任务为 tomato_wuji、tomato_sharpa、lemon_scooping、cup_grasping，四项均已有 LoRA / 全参 baseline；数据统一要求 LeRobot v2.1。template 目录当前仅保留空占位，有真实实验时再添加文件。启动器只导入指定模块并调用 `build_config()`，无需中央注册，不会运行其他配置。

## 创建个人实验

每次工作建立短期分支。以番茄-Wuji LoRA 为例：

```bash
git switch main
git pull --ff-only
git switch -c exp/alice-tomato-lora
mkdir -p training_configs/experiments/tomato_wuji/template/alice
touch training_configs/experiments/tomato_wuji/template/__init__.py \
  training_configs/experiments/tomato_wuji/template/alice/__init__.py
cp training_configs/experiments/tomato_wuji/baseline/tomato_fr3_lora_v1.py \
  training_configs/experiments/tomato_wuji/template/alice/tomato_lora_lr1e5.py
```

在同一个文件中修改 `build_dataset()` 的数据版本、root、repo_id、真实来源、asset_id、统计父目录和 prompt 示例。路径用 `repository_root() / "data/..."` 或 `"assets/..."`，换机器时可直接修改个人配置或使用自己定义的环境变量。

同一数据只换路径可保留身份；采集来源、筛选或变换变化时使用新版本和新统计。复制 Wuji 全参 baseline 后若换新数据，删除 `NORM_AUDIT_SHA256`、`expected_norm_audit_sha256` 和 `allow_legacy_norm_stats=True`，重新记录来源并计算统计。

个人文件直接引用公共 Dataset、本体与 recipe，避免导入其他实验文件。需要覆盖参数时，在文件顶部添加 `from dataclasses import replace`，将 `build_config()` 改成下面的形式；保留复制文件自己的 `build_dataset()`：

```python
from dataclasses import replace

def build_config():
    config = build_recipe(
        name="pi05_fr3_wuji_alice_lora_v1_30hz",
        exp_name="alice_tomato_lora_lr1e5_s42_run01",
        embodiment=FR3_WUJI,
        dataset=build_dataset(),
    )
    return replace(
        config,
        lr_schedule=replace(config.lr_schedule, peak_lr=1e-5),
        policy_metadata={**config.policy_metadata, "owner": "alice", "purpose": "compare_learning_rate"},
    )
```

通过完整模块路径检查，后续统计与训练也使用这个模块：

```bash
.venv/bin/python -m scripts.run_training \
  training_configs.experiments.tomato_wuji.template.alice.tomato_lora_lr1e5 --dry-run
```

数据和统计准备后执行 `--check`，通过后提交配置，再固定 commit 启动训练。实验名建议 `<负责人>_<任务>_<方案>_s<seed>_run<编号>`，新运行另用编号。

## 公共方案和本体变更

公共 recipe 按版本发布，旧实验保留原引用；新学习率、batch 或预算先写在个人配置中。公共 bug 修复通过独立合并请求，说明行为影响。模型变体变化需要重建 freeze_filter，不能仅通过 CLI 改变变体。原始基准参数快照保留在 [tests/fixtures/baselines](tests/fixtures/baselines/README.md)，不能从新实现重新生成期望值来掩盖回归。

新本体实现 `recipes/common.py` 的 Embodiment 接口，提供 action_dim、raw_state_dims、action_hz、camera_keys 和 data_config()。数据工厂须提供 `create_vector_transforms(model_config)`，使统计与训练复用同一输入变换；改维度、关节顺序、单位或相机时补充对应适配与验证，不能只修改数字。当前标准入口要求一个显式本地 LeRobot 数据集。

`log_interval`、`save_interval`、`action_horizon` 必须为正整数；`keep_period` 为 None 或正整数。改数据、输入变换或 horizon 后重算统计，单纯修改学习率和 batch 可复用配套统计。

## 分支、验证与合并

使用 `exp/<负责人>-<主题>`、`feat/<功能>`、`fix/<问题>` 短期分支，验证后提合并请求进入 main。个人实验不混入公共 recipe 改动；公共功能单独评审。合并后删除短期分支，不按本体或个人长期维护分叉。

配置改动执行 dry-run，有真实数据和统计时执行 check；共享代码变更运行相关回归，完整本地检查为：

```bash
bash scripts/check_training.sh
```

脚本验证锁文件、Ruff、Shell 语法、八份 baseline 展开、CPU 回归和 wheel 打包。测试使用临时数据，不启动正式训练；标记 manual 的外部数据测试需另外执行。GitLab 自动 CI 当前暂停；重新启用时删除 `.gitlab-ci.yml` 中首条无条件 `when: never`，并配置可运行作业的 Linux Docker/Kubernetes Runner，下方条件规则保留。

提交前确认只包含本任务文件，使用中文提交说明。合并请求写明改动目的、影响和验证结果；不新增开发流水账。数据、权重、日志、环境、缓存及密钥不提交 Git。

## 多人训练与有效记录

每位训练者使用独立 checkout/worktree，显式分配 `CUDA_VISIBLE_DEVICES`。训练期间不切分支或拉取代码；环境、缓存和输出保存在各自仓库内。数据和统计发布后不原地修改，共享存储上的统计计算及 GPU 使用由团队统一分配。

```bash
git worktree add /你的工作目录/pi05-alice "<已提交分支或commit>"
cd /你的工作目录/pi05-alice
bash scripts/setup_training_env.sh
```

实验配置及其旁边的结果记录只保留可核对的信息：数据生产来源和版本、源码 commit、配置名、实验名、统计身份/摘要、实际英文 prompt、关键参数、checkpoint、W&B run，以及指标与评估条件。缺失项明确写未记录。区分数据准备完成、训练完成、实机验证；配置检查通过不能写成训练完成，训练 loss 不能代替任务成功率。

训练同时保存本地日志并启用 W&B；无法联网明确用 offline，联网后逐段 sync。精确续训保持原 commit、配置、数据和完整 checkpoint；模块移动或新增配置也可能改变源码审计，因此运行期间固定整个 checkout。详细输出和续训步骤见仓库 README。

## subtask 与不同 prompt

配置文件里的 `Dataset.prompt_example` 是记录用示例，改它不会改变实际训练 prompt。普通入口按每帧 `task_index` 从数据集 `meta/tasks.jsonl` 读取文本；数据里已有不同任务标签时，不需要另一套训练脚本。

要只训练某些 subtask，或在个人参数文件中覆盖任务文本，可以使用 `TaskView`。例如复制一份个人实验，将 `build_config()` 写为：

```python
from dataclasses import replace
from openpi.training.config import TaskDataConfig
from openpi.training.task_view import TaskView
from training_configs.experiments.lemon_scooping.baseline.lemon_ur_o30_lora_v1 import build_config as baseline


def build_config():
    config = baseline()
    view = TaskView(
        split_path=config.data.base_config.task_view.split_path,  # 保留有效 episode 清单
        task_indices=(0,),                                    # 使用数据中实际存在的任务 ID
        prompt_overrides=((0, "Scoop the lemon slice into the cup."),),
        task_boundary_hold=True,                              # 动作窗口不跨任务标签边界
    )
    return replace(
        config,
        name="pi05_lemon_subtask_v1",
        exp_name="lemon_subtask_owner_run01",
        data=replace(
            config.data,
            assets=replace(config.data.assets, asset_id="lemon_subtask_v1"),
            base_config=TaskDataConfig(prompt_from_task=True, task_view=view),
        ),
        policy_metadata={
            **config.policy_metadata,
            "dataset_version": "lemon_subtask_v1",
            "prompt_source": "TaskView.prompt_overrides",
            "prompt_example": "Scoop the lemon slice into the cup.",
        },
    )
```

然后照常使用个人模块执行 `--dry-run`、计算新统计、`--check` 和训练。其他本体的数据工厂同样可以接收 `TaskDataConfig`；不需要 episode 清单时省略 `split_path`。`task_indices=()` 表示不按任务 ID 筛选；`prompt_overrides=()` 表示沿用数据任务文本。

**字段不能代替数据标注**：如果整条 episode 只有一个任务 ID，仅增加三句 prompt 并不能让模型知道哪些帧属于哪个 subtask。应在数据仓库提供逐帧任务标签，或使用明确阶段边界的视图。已有逐帧标签时，`task_boundary_hold=True` 将窗口限制在当前连续任务片段的最后一帧；不设置时仍允许跨任务标签取后续动作。

柠檬的五时间戳入口见[任务说明](training_configs/experiments/lemon_scooping/README.md)：`phase_annotations_path` 指向实际标注，`phase_prompts` 定义五个阶段的英文指令，`phase_arm_hold` 可启用双 UR 特定的手臂保持。它根据已知边界生成训练样本，不训练阶段分类器。阶段模式与 task_indices/prompt_overrides 模式分开使用。

当前所有 baseline 都没有验证集。柠檬清单把原 188 train + 21 val 合并成 209 个训练 episode；`split_path` 在这里仅用于选取有效数据。将来换数据时同步修改目录、repo_id、清单、来源和统计身份；不能只换路径却保留旧 episode 编号。筛选、阶段边界、prompt 映射的配置或内容变化会使旧统计失效，续训校验也会检查样本视图身份，需作为新实验重新计算统计。
