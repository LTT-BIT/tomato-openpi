#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
test -f /home/descfly/data-clean/1001_subtasks/norm_stats.json
export JAX_PLATFORMS=cuda
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/train.py" \
  pi05_fr3_wuji_1001_subtasks_30hz \
  --exp-name tomato_1001_subtasks_lora_30hz "$@"
