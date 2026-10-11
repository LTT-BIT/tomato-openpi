#!/usr/bin/env bash
set -euo pipefail
BASELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASELINE_DIR/../../.."
source "$BASELINE_DIR/env.sh"
export WANDB_MODE="${WANDB_MODE:-online}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
EXP="${EXP:-tomato_all_full_fullft_30hz_s42_run01}"
mkdir -p logs
"$UV_PROJECT_ENVIRONMENT/bin/python" -u -m training_baseline.fullft_v3.pi05_fr3_wuji_fulltask_fullft_30hz_train \
  pi05_fr3_wuji_fulltask_fullft_30hz --exp-name "$EXP" --wandb-enabled "$@" \
  2>&1 | tee "logs/${EXP}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$.log"
