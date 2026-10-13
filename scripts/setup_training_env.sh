#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_DIR"
export UV_PROJECT_ENVIRONMENT="$REPOSITORY_DIR/.venv"
export UV_CACHE_DIR="$REPOSITORY_DIR/.cache/uv"
export TMPDIR="$REPOSITORY_DIR/.cache/tmp"
export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$TMPDIR"
uv sync --frozen --python "${PI05_PYTHON:-3.11}" "$@"
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import scripts; import jax, torch; print("JAX", jax.__version__, jax.devices()); print("Torch", torch.__version__, "CUDA", torch.cuda.is_available())'
