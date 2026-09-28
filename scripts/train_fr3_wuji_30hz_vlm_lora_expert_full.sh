#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

export JAX_PLATFORMS=cuda
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"$ROOT/.venv/bin/python" "$ROOT/scripts/prepare_fr3_wuji_20hz.py" \
  --action-hz 30 \
  --data-root /home/descfly/lpy/Convert_data/outputs/fr3_wuji/0924-250ep \
  --dataset-names A B \
  --output "$ROOT/assets/pi05_fr3_wuji_0924_30hz/fr3_wuji/0924_30hz" \
  --keep-prompts
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/train.py" \
  pi05_fr3_wuji_0924_30hz_vlm_lora_expert_full \
  --exp-name tomato_vlm_lora_expert_full_0924_30hz "$@"
