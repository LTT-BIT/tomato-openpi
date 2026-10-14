#!/usr/bin/env bash
# Internal helper shared by the installer and Python launcher.
pi05_select_runtime() {
    PI05_REPOSITORY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    local shared_default=/mnt/data/public_tools/training/environments/pi05/20261014
    PI05_RUNTIME_SHARED=""
    if [[ -n "${PI05_SHARED_ENV:-}" ]]; then
        PI05_RUNTIME_SHARED="$PI05_SHARED_ENV"
    elif [[ -e "$PI05_REPOSITORY_DIR/.venv" || -L "$PI05_REPOSITORY_DIR/.venv" ]]; then
        : # Preserve an existing checkout environment, including reporting a broken one.
    elif [[ -x "$shared_default/run-python" ]]; then
        PI05_RUNTIME_SHARED="$shared_default"
    fi
    if [[ -n "$PI05_RUNTIME_SHARED" ]]; then
        if [[ ! -x "$PI05_RUNTIME_SHARED/run-python" ]]; then
            echo "Shared environment unavailable: $PI05_RUNTIME_SHARED/run-python" >&2
            return 1
        fi
        PI05_RUNTIME_SHARED="$(cd -- "$PI05_RUNTIME_SHARED" && pwd)"
    fi
}
