"""Explicit CUDA environment shared by the UI and command-line training entry."""

import os

CUDA_SETTINGS = {
    "JAX_PLATFORMS": "cuda",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
}


def cuda_environment(gpus, base=None):
    env = {
        **(os.environ if base is None else base),
        **CUDA_SETTINGS,
        "CUDA_VISIBLE_DEVICES": ",".join(gpus),
        "PYTHONUNBUFFERED": "1",
    }
    env.pop("JAX_PLATFORM_NAME", None)
    return env
