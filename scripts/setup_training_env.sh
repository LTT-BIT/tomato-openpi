#!/usr/bin/env bash
set -euo pipefail
REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_DIR"
source scripts/training_env.sh
uv sync --frozen --python "${PI05_PYTHON:-3.11}" "$@"
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import jax, torch; print("JAX", jax.__version__, jax.devices()); print("Torch", torch.__version__, "CUDA", torch.cuda.is_available())'
