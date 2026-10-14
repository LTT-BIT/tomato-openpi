# 多人开发 README

安装和训练命令见 [仓库 README](README.md)。本文件约定配置归属、协作方式和实验记录。

## 文件放在哪里

| 内容 | 位置 |
| --- | --- |
| 任务协议、默认数据及训练说明 | `training_configs/experiments/<任务>/README.md` |
| 公共参考配置 | `training_configs/experiments/<任务>/baseline/` |
| 本地个人实验配置 | `training_configs/experiments/<任务>/template/<负责人>/`，Git 忽略，不上传 |
| 公共训练方案 | `training_configs/recipes/` |
| 本体规格和适配 | `training_configs/embodiments/`、`src/openpi/policies/` |
| 模型、数据加载与训练循环 | `src/openpi/` |
| 历史有效实验记录 | `docs/records/` |

当前任务为 tomato_wuji、tomato_sharpa、lemon_scooping、cup_grasping，四项均已有 LoRA / 全参 baseline；数据统一要求 LeRobot v2.1。日常训练直接使用 baseline。确实需要个人参数时，在本机或自己的服务器目录向 template 添加文件；这些文件由 `.gitignore` 排除，不创建分支、不提交、不上传。启动器只导入指定模块并调用 `build_config()`，无需中央注册，不会运行其他配置。

## 目录与配置的组合关系

```text
training_configs/
├── common.py                  # Dataset 数据身份、仓库路径与缓存设置
├── recipes/                   # 怎么训练
│   ├── common.py              # Recipe，组装公共训练配置
│   ├── pi05_fullft_v1.py       # 全参方案：batch 64、30k
│   └── pi05_lora_v1.py         # LoRA 方案：batch 8、30k
├── embodiments/               # 机器人协议：维度、频率、相机、数据工厂
│   ├── fr3_wuji.py
│   ├── fr3_sharpa.py
│   ├── single_franka.py
│   └── dual_ur_o30.py
└── experiments/               # 具体任务和实验
    ├── tomato_wuji/
    ├── tomato_sharpa/
    ├── cup_grasping/
    └── lemon_scooping/
        # 四项任务都按下面的结构组织：
        ├── README.md          # 任务协议和使用说明
        ├── baseline/          # 公共全参、LoRA 参考配置
        └── template/          # 本地个人实验；内容不进入 Git
            └── <负责人>/     # 需要时在自己的 clone 中创建
```

柠檬的 `baseline/common.py` 另外共享数据定义与样本选择，`baseline/records/` 保存真实来源、episode 清单和阶段标注。`records/` 中的原始参数表示历史记录，不一定等于当前 baseline 默认值。

这些配置主要通过**组合**组织：实验选择一个 recipe、一个本体和一份数据定义，生成最终训练配置；并非目录每多一层就继承一层 Python 类。

```text
scripts/python.sh                       选择 Python 环境和当前仓库源码
  → scripts.run_training MODULE         导入指定实验模块
    → 实验模块.build_config()
      ├── build_dataset() → Dataset     具体数据目录、身份和统计位置
      ├── 本体对象                      维度、相机、频率、动作转换
      └── recipe.build_config()         模型、优化器、学习率、训练预算
        → TrainConfig                   实验还可用 replace 覆盖参数
          → CLI 覆盖与检查 → scripts/train.py → 数据加载与训练循环
```

`compute_experiment_norm` 也加载同一个实验模块，复用对应数据选择和向量变换，保证统计与训练一致。

### 主要类和字段分别负责什么

| 类 / 对象 | 位置 | 职责 |
| --- | --- | --- |
| `Dataset` | `training_configs/common.py` | 数据目录、来源、版本、统计身份等声明；不是实际读取样本的 Dataset |
| `Recipe` / `Embodiment` | `training_configs/recipes/common.py` | 前者组装训练配置；后者是本体需要提供的接口约定 |
| `Fr3Wuji`、`Fr3Sharpa`、`SingleFranka`、`DualURO30` | `training_configs/embodiments/` | 定义四种本体协议，并创建对应的数据工厂 |
| `TrainConfig` | `src/openpi/training/config.py` | 最终训练配置，包含模型、数据工厂、优化器、学习率、batch、步数和输出路径 |
| `ScheduledTrainConfig` | 同上 | 扩展 TrainConfig 的 checkpoint 保存时间表，当前用于柠檬全参 |
| `DataConfigFactory` 及各 `LeRobot…DataConfig` | 同上 | 根据本体和模型生成数据读取与变换配置；四类 baseline 使用各自的工厂 |
| `DataConfig` / `TaskDataConfig` | 同上 | 数据读取与变换设置；TaskDataConfig 额外携带任务视图 |
| `TaskView` | `src/openpi/training/task_view.py` | episode 选择、subtask 筛选、prompt 覆盖和阶段边界规则 |
| `Pi0Config` | `src/openpi/models/pi0_config.py` | 模型维度、horizon、模型变体、精度等设置 |

常用字段的嵌套位置：

```text
TrainConfig
├── model                         Pi0Config
├── data                          本体对应的 DataConfigFactory
│   ├── lerobot_datasets          LeRobotDatasetSpec：repo_id、root 等
│   ├── assets                    AssetsConfig：统计父目录、asset_id
│   └── base_config               DataConfig 或 TaskDataConfig
│       └── task_view             使用 TaskDataConfig 时才有
├── lr_schedule                   学习率计划
├── optimizer                     优化器设置
├── weight_loader                 基础权重加载方式与路径
└── batch_size / num_train_steps / exp_name / …
```

`src/openpi/training/config.py` 还保留部分上游数据工厂；它们的存在不表示当前四项任务会使用它们。实际使用哪一个，由实验选定的本体和 recipe 决定。

### 平时改哪里

| 要改什么 | 修改位置 |
| --- | --- |
| 某次实验的 batch、学习率、步数、实验名、基础权重 | 自己的 `template/<负责人>/<实验>.py` 中 build_config，用 replace 覆盖 |
| 数据目录、数据身份和统计位置 | 该实验的 build_dataset；柠檬独立实验需要独立数据定义时复制公共定义到个人文件 |
| subtask、真实训练 prompt | 个人配置中的 TaskDataConfig / TaskView，或数据任务标签；不是 Dataset.prompt_example |
| 所有人共用的默认训练方案 | recipes，通过公共变更评审 |
| 新本体或关节、相机、动作语义 | embodiments 及对应数据工厂、policies 变换；补充协议验证 |
| 训练算法、采样或保存行为 | scripts/train.py、src/openpi/training 等共享代码 |
| 使用本地还是公共 Python 环境 | scripts/python.sh 的选择规则和 PI05_SHARED_ENV；不放在实验参数中 |

## 创建本地个人实验

个人配置只保存在自己的 clone 中，不为它创建分支，也不上传 GitLab。以番茄-Wuji LoRA 为例：

```bash
mkdir -p training_configs/experiments/tomato_wuji/template/alice
cp training_configs/experiments/tomato_wuji/baseline/tomato_fr3_lora_v1.py \
  training_configs/experiments/tomato_wuji/template/alice/tomato_lora_lr1e5.py
```

Python 可以直接导入该目录，不需要添加 `__init__.py`。`.gitignore` 已排除 template 下的个人文件；用 `git status` 不应看到它们。不要使用 `git add -f` 强制提交。

在同一个文件中修改 `build_dataset()` 的数据版本、root、repo_id、真实来源、asset_id、统计父目录和 prompt 示例。路径用 `repository_root() / "data/..."` 或 `"assets/..."`，换机器时可直接修改个人配置或使用自己定义的环境变量。

同一数据只换路径可保留身份；采集来源、筛选或变换变化时使用新版本和新统计。当前番茄 baseline 已移除历史 delta 统计兼容和固定摘要，使用带 absolute 的新统计身份；不要恢复旧统计兼容设置。

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
bash scripts/python.sh -m scripts.run_training \
  training_configs.experiments.tomato_wuji.template.alice.tomato_lora_lr1e5 --dry-run
```

数据和统计准备后执行 `--check`，通过后直接启动训练。个人配置不提交；启动器会在日志目录自动保存个人配置源码副本，并在运行清单中记录源码哈希、公共仓库 commit 和最终展开配置。实验名建议 `<负责人>_<任务>_<方案>_s<seed>_run<编号>`，新运行另用编号。

## 公共方案和本体变更

公共 recipe 按版本发布，旧实验保留原引用；新学习率、batch 或预算可先在本地个人配置中验证。确认需要让所有人复用时，先讨论数据身份、参数和命名，再由维护者新增或升级 baseline。公共 bug 修复和 baseline 变更通过独立合并请求，说明行为影响。模型变体变化需要重建 freeze_filter，不能仅通过 CLI 改变变体。原始基准参数快照保留在 [tests/fixtures/baselines](tests/fixtures/baselines/README.md)，不能从新实现重新生成期望值来掩盖回归。

新本体实现 `recipes/common.py` 的 Embodiment 接口，提供 action_dim、raw_state_dims、action_hz、camera_keys 和 data_config()。数据工厂须提供 `create_vector_transforms(model_config)`，使统计与训练复用同一输入变换；改维度、关节顺序、单位或相机时补充对应适配与验证，不能只修改数字。当前标准入口要求一个显式本地 LeRobot 数据集。

`log_interval`、`save_interval`、`action_horizon` 必须为正整数；`keep_period` 为 None 或正整数。改数据、输入变换或 horizon 后重算统计，单纯修改学习率和 batch 可复用配套统计。

## 公共代码的分支、验证与合并

本地个人配置不使用分支。只有新增 baseline、修改公共 recipe、本体适配或训练代码时，维护者才使用 `feat/<功能>`、`fix/<问题>` 等短期分支，验证后合入 main。合并后删除短期分支，不按本体或个人长期维护分叉。

配置改动执行 dry-run，有真实数据和统计时执行 check；共享代码变更运行相关回归，完整本地检查为：

```bash
bash scripts/check_training.sh
```

脚本验证锁文件、Ruff、Shell 语法、八份 baseline 展开、CPU 回归和 wheel 打包。测试使用临时数据，不启动正式训练；标记 manual 的外部数据测试需另外执行。GitLab 自动 CI 当前暂停；重新启用时删除 `.gitlab-ci.yml` 中首条无条件 `when: never`，并配置可运行作业的 Linux Docker/Kubernetes Runner，下方条件规则保留。

提交前确认只包含本任务文件，使用中文提交说明。合并请求写明改动目的、影响和验证结果；不新增开发流水账。数据、权重、日志、环境、缓存及密钥不提交 Git。

## 多人训练与有效记录

**默认每人独立 clone 到自己的目录，日常训练停留在相同的稳定 main，不要求创建分支或使用 worktree。** 例如：

```text
/home/zhiwei/workspace/tomato-openpi/     # zhiwei 的代码和输出
/home/pengyang/workspace/tomato-openpi/   # pengyang 的代码和输出
```

两份 clone 来自同一 GitLab 仓库，也各自拥有独立文件和 Git 状态。个人 template 配置只存在于各自目录，互不影响，也不需要 GitLab 同步。不要多人共用同一个代码目录。

训练期间保持**正在训练的那份目录**不变，不在其中切分支、拉代码或编辑源码。已加载的 Python 代码通常不会自动更新，但后续导入、创建 worker、读取文件或重启续训可能接触到新内容。其他独立目录可以正常开发。

服务器可共用只读 Python 环境和基础权重，启动器加载各自仓库源码；缓存、日志和输出仍各自保存。显式分配 `CUDA_VISIBLE_DEVICES`，协调 GPU 与存储资源。共享数据和统计发布后不原地修改。

### 可选：同一个人同时训练和开发

如果自己的一份目录正在训练，又需要开发另一分支，可以另 clone 一份，也可以用 worktree 创建第二份工作目录。worktree 共用 Git 对象和分支记录，但工作文件独立；不是多人训练的必选项。

以下命令不会切换当前训练目录的分支：

```bash
# 在自己的仓库执行；分支名和新目录需未被占用。
git worktree add -b feat/alice-new-feature ../tomato-openpi-dev HEAD
cd ../tomato-openpi-dev
# 服务器直接复用公共环境。
export PI05_SHARED_ENV=/mnt/data/public_tools/training/environments/pi05/20261011
bash scripts/python.sh -m scripts.run_training \
  training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1 --dry-run
```

新 worktree 不会自动复制原目录里未跟踪的数据、统计、日志或本地 `.venv`；正式训练前需单独准备数据与统计。自己的电脑可在新目录执行 `bash scripts/setup_training_env.sh --local`。不要删除仍有进程使用的工作目录。

个人配置虽然不上传，启动器仍会随本地日志自动保存其源码副本。训练记录还需保留可核对的信息：数据生产来源和版本、公共源码 commit、配置名、实验名、统计身份/摘要、实际英文 prompt、关键参数、checkpoint、W&B run，以及指标与评估条件。缺失项明确写未记录。区分数据准备完成、训练完成、实机验证；配置检查通过不能写成训练完成，训练 loss 不能代替任务成功率。

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

所有任务的全参数 baseline 默认全局 `batch_size=64`，LoRA 默认 `batch_size=8`；全局 batch 是所有参与训练 GPU 的合计，不是每张 GPU 的 batch。个人实验可显式覆盖，并使用新的实验名称；已经开始的实验续训时保持原 batch。

## 公共环境管理

500028 服务器的公共环境位于 `/mnt/data/public_tools/training/environments/pi05/20261011`，基础权重位于 `/mnt/data/public_ckpt/pi05/pi05_base/params`，使用入口见[根 README](README.md#500028-服务器复用公共环境)。环境按版本固定，升级依赖发布新目录，不原地更新正在使用的版本。

- `zhiwei` 维护环境，`ai_researchers` 组成员读取和执行。新文件应允许组读取，目录和程序允许组执行，不允许组修改。
- 每人维护自己的源码 checkout，通过 `bash scripts/python.sh` 启动，由脚本向公共 `run-python` 传入当前仓库路径；不要对公共环境执行 `pip install -e`，避免绑定某个人的代码。
- 个人缓存和训练输出位于自己的仓库。实验记录同时写明源码 commit 和公共环境版本；换环境后先验证再开始新实验。
- 原始离线包来自 `/home/pengyang/workspace/pi05_env_bundle_20261011`。SHA-256 校验、源码验证快照及测试输出保存于服务器 `/mnt/data/public_tools/training/validation/pi05-20261014`。环境包含旧版本留下的额外仿真包，不表示本仓库提供仿真或部署入口。

环境内 `system/` 补齐了宿主机缺失的 FFmpeg 6.1.1 动态库，`run-python` 自动设置其查找路径；没有修改服务器系统安装。不要绕过该启动器直接调用共享 Python，否则可能缺少视频解码库或仓库源码路径。
