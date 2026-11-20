#!/usr/bin/env bash
# Shared local/GitLab/GitHub checks; use the already installed locked environment.
set -euo pipefail
REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_DIR"
export PYTHONNOUSERSITE=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GIT_LFS_SKIP_SMUDGE=1
export WANDB_MODE=offline
export UV_CACHE_DIR="$REPOSITORY_DIR/.cache/uv"
export UV_PROJECT_ENVIRONMENT="$REPOSITORY_DIR/.venv"
export PYTHONPATH="$REPOSITORY_DIR/src:$REPOSITORY_DIR/packages/openpi-client/src"
mkdir -p .cache/ci

uv lock --check --offline
# Lint maintained Sharpa workflows; all historical regressions still run below.
# Do not traverse ignored personal template directories.
CHECK_PATHS=(
  training_configs/common.py training_configs/recipes
  training_configs/embodiments/fr3_sharpa.py training_configs/embodiments/fr3_sharpa_multimodal.py
  training_configs/experiments/tomato_sharpa/baseline training_configs/experiments/tomato_sharpa/tasks
  tests/test_sharpa_task_releases.py tests/test_training_configs.py tests/test_training_ui.py
  tests/test_training_launcher.py tests/test_dataset_collection.py tests/test_sharpa_training.py
  tests/test_experiment_norm.py tests/test_absolute_baselines.py tests/test_repository_paths.py
  tests/test_tomato_in_hand204.py tests/test_tomato_paper_training.py
  scripts/prepare_tomato_in_hand.py scripts/prepare_tomato_in_hand204.py
  src/openpi/subtasks/tomato_in_hand.py
  scripts/create_delta_paper_experiments.py scripts/training_ui_datasets.py scripts/training_ui_groups.py
  src/openpi/models/pi0_delta_paper.py src/openpi/shared/delta_paper.py
  src/openpi/training/current_rgb.py src/openpi/training/frame_cache.py
  src/openpi/training/delta_paper.py src/openpi/training/delta_paper_stats.py
  src/openpi/training/delta_paper_ui.py src/openpi/training/delta_paper_experiment.py
  src/openpi/training/delta_paper_prefix.py
  scripts/training_ui.py scripts/training_ui_config.py
  scripts/run_training.py scripts/publish_tomato_dataset.py scripts/compute_experiment_norm.py scripts/train.py scripts/train_reproducibility_test.py
  src/openpi/training/norm_provenance.py scripts/compute_robot_norm_stats.py
  src/openpi/shared/local_storage.py src/openpi/training/tracking.py
  src/openpi/policies/fr3_sharpa_policy.py src/openpi/policies/fr3_sharpa_protocol.py
  src/openpi/policies/single_franka_policy.py src/openpi/policies/lemon_policy.py
  src/openpi/training/task_view.py src/openpi/training/task_view_norm.py
  src/openpi/training/config.py src/openpi/training/checkpoints.py
  src/openpi/training/sampling.py src/openpi/training/sampling_test.py
  src/openpi/training/data_loader.py src/openpi/training/reproducibility.py
  src/openpi/training/reproducibility_test.py
  src/openpi/transforms_test.py src/openpi/training/data_loader_test.py
  scripts/verify_dataset.py
)
bash scripts/python.sh -m ruff check "${CHECK_PATHS[@]}"
bash scripts/python.sh -m ruff format --check "${CHECK_PATHS[@]}"
bash -n scripts/setup_training_env.sh scripts/check_training.sh scripts/python.sh scripts/training_runtime.sh

for experiment in \
  training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_lora_v1 \
  training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_fullft_v1 \
  training_configs.experiments.tomato_sharpa.tasks.single_pick \
  training_configs.experiments.tomato_sharpa.tasks.truss_plate \
  training_configs.experiments.tomato_sharpa.tasks.truss_in_hand; do
  bash scripts/python.sh -m scripts.run_training "$experiment" --dry-run
done

bash scripts/python.sh -m pytest tests \
  src/openpi/models/pi0_rtc_test.py src/openpi/models/pi0_test.py \
  src/openpi/training/sampling_test.py src/openpi/training/reproducibility_test.py \
  src/openpi/training/weight_loaders_test.py src/openpi/training/data_loader_test.py \
  src/openpi/transforms_test.py \
  src/openpi/shared/normalize_test.py scripts/train_reproducibility_test.py \
  -q -m 'not manual' --junitxml=.cache/ci/junit.xml
uv build --wheel --out-dir .cache/ci/wheels
