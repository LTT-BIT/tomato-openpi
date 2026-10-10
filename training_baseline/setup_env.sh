#!/usr/bin/env bash
set -euo pipefail
BASELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASELINE_DIR/.."
source "$BASELINE_DIR/env.sh"
SYNC_OPTIONS=()
if [[ -d "$PI05_RUNTIME_DIR/locked-wheels" ]]; then
    SYNC_OPTIONS+=(--find-links "$PI05_RUNTIME_DIR/locked-wheels")
fi
uv sync --frozen --python "${PI05_PYTHON:-3.11}" "${SYNC_OPTIONS[@]}"
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false "$UV_PROJECT_ENVIRONMENT/bin/python" - <<'PY'
import jax
import jax.numpy as jnp
import torch

devices = jax.devices()
assert devices and all(device.platform == "gpu" for device in devices)
print("JAX", jax.__version__, devices)
for device in devices:
    with jax.default_device(device):
        matrix = jnp.ones((32, 32))
        assert float((matrix @ matrix).block_until_ready()[0, 0]) == 32
assert torch.cuda.is_available()
print("Torch", torch.__version__, "CUDA devices", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    matrix = torch.ones((32, 32), device=f"cuda:{index}")
    assert float((matrix @ matrix)[0, 0]) == 32
print("PASS: GPU matrix multiplication on every visible JAX and Torch device")
PY
