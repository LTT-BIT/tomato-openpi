# Pi05 多人开发 README

## 配置归属

同一任务不同本体在同一主线管理，各本体提供自己的规格与训练数据适配，公共模型和训练循环共用。

| 改动 | 位置 | 规则 |
| --- | --- | --- |
| 个人学习率、batch、步数、seed | `training_configs/experiments/<负责人>/` | 独立文件，通过 dataclasses.replace 覆盖 |
| 团队训练方案升级 | `training_configs/recipes/` | 建立新 recipe 版本，旧实验继续引用旧版 |
| 数据、路径及统计变化 | 同一个个人实验文件的 `build_dataset()` | 新数据建立新版本、新 asset_id，记录来源与统计哈希 |
| 本体维度、顺序或动作语义变化 | `training_configs/embodiments/` 及对应适配器 | 新规格版本，验证完整输入变换 |
| 训练循环、采样或恢复改进 | `src/openpi/training/` | 独立 PR，说明影响并跑回归 |

本训练仓库的各类版本独立从 v1 起步：当前公共方案是 pi05_lora_v1 与 pi05_fullft_v1。数据版本在每个实验文件的 Dataset 中定义，不跟随方案编号，同一数据可被两种方案复用。历史 v2/v3 只表示来源，保留在溯源记录与归档中。

公共版本的 bug 修复可以修改原文件，但必须记录影响与 commit；版本名称不能替代源码身份。历史实验使用原 commit 和最终配置快照。参数回归使用 `tests/fixtures/baselines/` 的原基准静态快照，旧代码从 Git 历史查询，不把新实验改写成旧实验。

## experiments 怎样工作

experiments 是完整实验配置的集合，不是训练结果目录。子目录只是组织文件的约定，启动器不扫描所有目录，也不把它们当作先后阶段。

```text
training_configs/
  common.py                      # 统一 Dataset 结构，无具体数据设置
  embodiments/                   # 公共本体规格与适配
  recipes/                       # 公共 LoRA v1、全参 v1 方案
  experiments/
    baselines/                   # 两套可运行的公共参考示例
    template/                    # 可复制的独立个人实验模板
    alice/                       # Alice 自己的实验，按需创建
    bob/                         # Bob 自己的实验，按需创建
```

baselines 用于核对公共基准；template 用于复制，已包含自己的数据设置，不依赖基准实验文件。alice/bob 是负责人目录的示例，当前不会自动创建。目录名不是程序要求，只有完整 Python 模块路径需要唯一。

例如指定 `training_configs.experiments.alice.tomato_lora_lr1e5`，启动器就导入 `experiments/alice/tomato_lora_lr1e5.py`，调用它的 build_config()。这个函数先构造本次 Dataset，再选择本体和 recipe，覆盖本次参数，返回完整 TrainConfig；启动器随后应用 CLI 覆盖、检查输入、记录配置并调用共享训练循环。一次命令只执行一个所选实验，其他文件不会一起训练。dry-run 只打印配置，check 只验证输入。

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

在复制后的同一个文件里修改数据、路径、统计身份、prompt 示例、实验名、owner 和 purpose。模板已提供完整结构，核心组合方式如下：

```python
from dataclasses import replace
import os
from training_configs.common import Dataset, repository_root
from training_configs.embodiments.fr3_wuji import FR3_WUJI
from training_configs.recipes.pi05_lora_v1 import build_config as build_recipe

def build_config():
    repository = repository_root()
    root = os.environ.get("PI05_DATASET_ROOT", str(repository / "data/alice_tomato_v1"))
    dataset = Dataset(
        version="alice_tomato_v1_30hz",
        repo_id="fr3_wuji/alice_tomato_v1",
        root=root,
        assets_dir=os.environ.get("PI05_ASSETS_DIR", str(repository / "assets")),
        asset_id="alice_tomato_v1_30hz",
        sources=(root,),
        prompt_example="Pick a tomato and place it in the basket.",
        # 完成统计后补填实际 expected_norm_sha256。
    )
    config = build_recipe(
        name="pi05_fr3_wuji_alice_lora_v1_30hz",
        exp_name="alice_tomato_lora_lr1e5_s42_run01",
        embodiment=FR3_WUJI,
        dataset=dataset,
    )
    return replace(
        config,
        lr_schedule=replace(config.lr_schedule, peak_lr=1e-5),
        policy_metadata={**config.policy_metadata, "owner": "alice", "purpose": "compare_learning_rate"},
    )
```

通过模块路径加载，不需要中央注册：

```bash
.venv/bin/python -m scripts.run_training \
  training_configs.experiments.alice.tomato_lora_lr1e5 --dry-run
```

个人实验只需引用公共 recipe、embodiment 和 Dataset 结构，不导入他人的实验文件。配置中的 prompt_example 用于记录，真实训练从数据 task 读取英文指令。

换数据时创建新的 Dataset 和版本，更新 repo_id、来源、prompt 示例、asset_id 与统计哈希；不能仅将新数据放到旧数据路径。同一内容的路径迁移可使用环境变量。

数据格式转换、筛选和合并统一在独立数据仓库完成，训练仓库保留元信息/归一化绑定检查和原始采集对照校验。常规流程是准备数据与实验配置 → 新数据计算统计 → --check → 正式启动时再检查并记录数据/采样审计。--check 需要统计；原始采集对照工具是额外抽样核验，前提是有采集数据且 episode 顺序对应。详细步骤及校验范围见训练使用说明。

新统计同时保存 norm_stats.json 和 norm_metadata.json，提交实验配置前执行 --check，确认统计与数据文件、输入变换及 horizon 对应。复制统计需保留两个文件；学习率、batch 等训练参数不影响统计复用。常量关节允许生成零方差统计，其维度记录在元数据中。个人新实验不要沿用历史基准的 allow_legacy_norm_stats=True；旧格式兼容必须人工确认来源并固定摘要，具体条件见训练使用说明。

新本体实现 `recipes/common.py` 的 Embodiment 接口，提供 action_dim、raw_state_dims、action_hz、camera_keys 与 data_config()。当前启动流程要求单个显式本地 LeRobot 数据集，data_config() 需提供对应数据字段；适配器负责关节顺序、单位和动作表示。为归一化绑定检查，数据工厂需提供 create_vector_transforms(model_config)，返回归一化前的 repack/data 两组变换，不能在此加载模型、统计或 tokenizer；实际训练 create() 复用同一变换构造方法，统计计算入口也需对应适配。修改模型内部维度或增加 padding，需配套修改适配、统计检查和权重加载，不能直接修改 FR3 的维度数字。

## 分支与评审

每人针对一次任务建立短期 `exp/<负责人>-<主题>`、`feat/<功能>` 或 `fix/<问题>` 分支，验证后提 PR 合入主线。所有配置最终都在主线，不共享持续修改的开发分支，也不按人或本体保留永久分支。

个人实验 PR 不混入公共 recipe 修改；公共方案变更单独评审，注明影响哪些实验、是否改变冻结范围或采样行为。合并后删除任务分支，正在训练的任务继续使用固定源码。

PR 说明目的、基准、覆盖参数、数据身份、统计引用、英文 prompt 来源和验证结果。区分准备完成与训练完成；只有真实日志和完整 checkpoint 支持时才记录训练完成。本训练仓库不安排部署或实机验证。

## 多人训练隔离

运行名建议为 `<负责人>_<任务>_<方案>_s<seed>_run<编号>`，重复运行另用编号。checkpoint 路径为 `<checkpoint_base_dir>/<config.name>/<exp_name>/`，W&B 名称与实验名一致。

使用独立 checkout 或 worktree，运行期间不切分支、不拉新代码。显式分配 CUDA_VISIBLE_DEVICES；每个 checkout 使用自己的 .venv/、.cache/、logs/、checkpoints/ 和 wandb/，数据与统计放在各自仓库的 data/ 和 assets/，发布后不能原地修改。目录锁只提供本机相同输出路径的并发保护，跨主机或共享文件系统还需团队调度和输出目录分配。

```bash
git worktree add /你的工作目录/pi05-alice "<已提交分支或commit>"
cd /你的工作目录/pi05-alice
bash scripts/setup_training_env.sh
```

数据、权重、日志、W&B 和密钥不提交 Git。个人实验文件保存数据身份与默认路径，默认路径按仓库位置自动确定，数据放在 data/、统计放在 assets/，换电脑克隆后使用对应目录即可。同一内容只更换路径不需要新数据版本。

## 验证与提交

配置变更先执行 --dry-run；有准备好的数据和统计时再执行 --check。公共训练改动跑相关回归：

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest \
  tests \
  src/openpi/training/sampling_test.py src/openpi/training/reproducibility_test.py \
  src/openpi/training/weight_loaders_test.py src/openpi/training/data_loader_test.py \
  src/openpi/policies/fr3_wuji_policy_test.py src/openpi/transforms_test.py \
  src/openpi/shared/normalize_test.py scripts/train_reproducibility_test.py -q -m 'not manual'
```

验证后在 docs/开发日志.md 记录变更、结果与未完成事项，用中文提交说明，只提交本任务文件。精确恢复保留原源码、数据、统计、完整 checkpoint 和采样审计。

### GitLab 自动验证

`.gitlab-ci.yml` 在分支推送、合并请求创建/更新时执行；有打开的合并请求时只运行合并请求流水线，避免重复运行。分支也可在 GitLab 手动运行流水线。GitLab 和 GitHub 共用 `scripts/check_training.sh`，本地安装环境后也可执行：

```bash
bash scripts/check_training.sh
```

检查包括 uv 锁一致性、配置/启动器/统计工具的 Ruff 与格式、Shell 语法、两个基准与模板 dry-run、数据与统计绑定/常量关节/采样/模型冻结/中断恢复等 CPU 回归，以及 wheel 打包。测试使用临时小数据、小模型及本地生成的 tokenizer，不读取真实数据、下载预训练权重或启动正式训练；依赖下载和 wheel 构建仍需要网络。外部真实数据测试标记 manual，自动验证使用 -m 'not manual' 排除；准备好对应数据后可单独手动执行。JUnit 报告与 wheel 在流水线保留一周。开发者仍需针对新增共享源码运行相应检查。

项目需要启用一个能运行 `python:3.11-bookworm` 容器的 Linux Docker/Kubernetes Runner，允许无标签作业；本方案不要求 GPU、训练数据或 W&B 密钥。Runner 需要访问 PyPI、GitHub、PyTorch 包源及 Debian 包源。依赖仍按训练 uv.lock 安装，含 CUDA 包，即使测试只用 CPU，首次安装仍需较大空间；建议至少 16 GB 内存、30 GB 可用磁盘。若作业一直 pending，先检查 Settings → CI/CD → Runners 是否有可用 Runner；代码里的 CI 文件不会自动创建 Runner。

合并请求在 CI/CD → Pipelines 查看结果，JUnit 结果可在 Tests 查看。若团队希望验证失败时禁止合并，维护者可在项目设置启用 Pipelines must succeed；本轮只配置仓库流水线，不修改项目权限或合并规则。

触发与去重规则参考 [GitLab workflow 文档](https://docs.gitlab.com/ci/yaml/workflow/)，依赖缓存方式参考 [uv 的 GitLab 集成文档](https://docs.astral.sh/uv/guides/integration/gitlab/)。

当前审计对训练源码、配置目录与工具生成哈希，新增其他实验文件也可能改变源码审计。续训必须使用原固定 checkout；代码升级后创建新实验，不在移动的主线上续训旧实验。
