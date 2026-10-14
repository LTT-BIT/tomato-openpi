#!/usr/bin/env bash
# Run this checkout with an explicit shared environment, existing .venv, or server default.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/training_runtime.sh"
pi05_select_runtime
if [[ -n "$PI05_RUNTIME_SHARED" ]]; then
    exec "$PI05_RUNTIME_SHARED/run-python" "$PI05_REPOSITORY_DIR" "$@"
fi
python="$PI05_REPOSITORY_DIR/.venv/bin/python"
if [[ ! -x "$python" ]]; then
    echo 'No usable Python environment; run bash scripts/setup_training_env.sh first.' >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PI05_REPOSITORY_DIR:$PI05_REPOSITORY_DIR/src:$PI05_REPOSITORY_DIR/packages/openpi-client/src"
cd "$PI05_REPOSITORY_DIR"
exec "$python" "$@"
