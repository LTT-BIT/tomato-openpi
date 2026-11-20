"""Explicit CUDA environment shared by the UI and command-line training entry."""

import os

from training_configs.common import CUDA_MEMORY_SETTINGS

CUDA_SETTINGS = {
    "JAX_PLATFORMS": "cuda",
    **CUDA_MEMORY_SETTINGS,
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
