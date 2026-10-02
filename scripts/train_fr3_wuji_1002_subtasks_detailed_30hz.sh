#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
test -f /home/descfly/data-clean/1002_subtasks_detailed/norm_stats.json
export JAX_PLATFORMS=cuda
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p "$ROOT/logs"
TRAIN_LOG="$ROOT/logs/1002_subtasks_detailed_$(date +%Y%m%d_%H%M%S)_$$.log"
"$ROOT/.venv/bin/python" -u "$ROOT/scripts/train.py" \
  pi05_fr3_wuji_1002_subtasks_detailed_30hz \
  --exp-name tomato_1002_subtasks_detailed_lora_30hz --wandb-enabled "$@" 2>&1 | tee "$TRAIN_LOG"
