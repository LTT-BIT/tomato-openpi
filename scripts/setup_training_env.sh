#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/training_runtime.sh"
force_local=false
if [[ "${1:-}" == --local ]]; then
    force_local=true
    shift
    unset PI05_SHARED_ENV
fi
if [[ "$force_local" == true ]]; then
    PI05_REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    PI05_RUNTIME_SHARED=""
else
    pi05_select_runtime
fi
REPOSITORY_DIR="$PI05_REPOSITORY_DIR"
if [[ -n "$PI05_RUNTIME_SHARED" ]]; then
    if [[ $# -gt 0 ]]; then
        echo 'Shared environment is read-only; uv sync options require --local.' >&2
        exit 2
    fi
    echo "Using shared environment: $PI05_RUNTIME_SHARED"
    exec bash "$REPOSITORY_DIR/scripts/python.sh" -c 'import scripts; import jax, torch; print("JAX", jax.__version__, jax.devices()); print("Torch", torch.__version__, "CUDA", torch.cuda.is_available())'
fi
if [[ -L "$REPOSITORY_DIR/.venv" ]]; then
    echo 'Refusing uv sync into a symlinked .venv; use PI05_SHARED_ENV for shared environments.' >&2
    exit 1
fi
cd "$REPOSITORY_DIR"
export UV_PROJECT_ENVIRONMENT="$REPOSITORY_DIR/.venv"
export UV_CACHE_DIR="$REPOSITORY_DIR/.cache/uv"
export TMPDIR="$REPOSITORY_DIR/.cache/tmp"
export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$TMPDIR"
uv sync --frozen --python "${PI05_PYTHON:-3.11}" "$@"
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import scripts; import jax, torch; print("JAX", jax.__version__, jax.devices()); print("Torch", torch.__version__, "CUDA", torch.cuda.is_available())'
