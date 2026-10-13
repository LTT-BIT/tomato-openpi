"""Single Franka: seven joint values, head/left-wrist images, no extra action delta."""

import dataclasses

import numpy as np

from openpi import transforms

PHYSICAL_DIM = 7
MODEL_DIM = 32


def _vector(value, label):
    value = np.array(value, dtype=np.float32, copy=True)
    if value.ndim == 0 or value.shape[-1] != PHYSICAL_DIM:
        raise ValueError(f"Expected single Franka {PHYSICAL_DIM}D {label}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"Single Franka {label} contains NaN or Inf")
    return value


def _image(value):
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError(f"Expected an RGB image, got {value.shape}")
    if value.shape[0] == 3:
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        value = (255 * value).astype(np.uint8)
    return value


@dataclasses.dataclass(frozen=True)
class SingleFrankaInputs(transforms.DataTransformFn):
    """Keep physical vectors for normalization; model transforms pad them afterwards."""

    def __call__(self, data: dict) -> dict:
        head = _image(data["observation/image"])
        result = {
            "state": _vector(data["observation/state"], "state"),
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": _image(data["observation/wrist_image"]),
                "right_wrist_0_rgb": np.zeros_like(head),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            result["actions"] = _vector(data["actions"], "action")
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class SingleFrankaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 0 or actions.shape[-1] != MODEL_DIM:
            raise ValueError(f"Expected single Franka {MODEL_DIM}D model output, got {actions.shape}")
        return {"actions": actions[..., :PHYSICAL_DIM]}
