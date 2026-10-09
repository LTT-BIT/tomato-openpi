#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"

DATASET="$ROOT/data/fr3_wuji/0924_A_ep93_96"
test -f "$DATASET/meta/info.json" || { echo "Missing subset: $DATASET" >&2; exit 1; }
test -f "$ROOT/assets/pi05_fr3_wuji_0924_30hz/fr3_wuji/0924_30hz/norm_stats.json" || {
  echo "Missing 0924/30 Hz normalization statistics" >&2
  exit 1
}

export JAX_PLATFORMS=cuda
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_CLIENT_MEM_FRACTION=0.90

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/train.py" \
  pi05_fr3_wuji_0924_A_ep93_96_30hz_vision_trainable_overfit \
  --exp-name tomato_A_ep93_96_vision_trainable_overfit_30hz "$@"
