#!/usr/bin/env bash
# Source this file to keep the environment, caches and outputs in this checkout.
export PI05_REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
unset PI05_RUNTIME_DIR
export UV_PROJECT_ENVIRONMENT="$PI05_REPOSITORY_DIR/.venv"
export UV_CACHE_DIR="$PI05_REPOSITORY_DIR/.cache/uv"
export XDG_CACHE_HOME="$PI05_REPOSITORY_DIR/.cache"
export OPENPI_DATA_HOME="$PI05_REPOSITORY_DIR/.cache/openpi"
export HF_HOME="$PI05_REPOSITORY_DIR/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PI05_REPOSITORY_DIR/.cache/torch"
export WANDB_DIR="$PI05_REPOSITORY_DIR"
export WANDB_CACHE_DIR="$PI05_REPOSITORY_DIR/.cache/wandb"
export WANDB_DATA_DIR="$PI05_REPOSITORY_DIR/.cache/wandb-data"
export TMPDIR="$PI05_REPOSITORY_DIR/.cache/tmp"
export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
mkdir -p "$TMPDIR" "$PI05_REPOSITORY_DIR/logs" "$PI05_REPOSITORY_DIR/wandb" \
  "$PI05_REPOSITORY_DIR/checkpoints" "$PI05_REPOSITORY_DIR/data" "$PI05_REPOSITORY_DIR/assets"
