"""Dual UR/O30: 52 observed joints and 12 absolute arm targets, three RGB cameras."""

import dataclasses

import numpy as np

from openpi import transforms

STATE_DIM = 52
ACTION_DIM = 12


def vector(value, dimension, label):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 0 or value.shape[-1] != dimension or not np.isfinite(value).all():
        raise ValueError(f"Expected finite {dimension}D Lemon {label}, got {value.shape}")
    return value


def image(value):
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[0] == 3:
        value = np.moveaxis(value, 0, -1)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        value = (255 * value).astype(np.uint8)
    return value


@dataclasses.dataclass(frozen=True)
class LemonInputs(transforms.DataTransformFn):
    def __call__(self, data):
        result = {
            "state": vector(data["observation/state"], STATE_DIM, "state"),
            "image": {
                role: image(data[key])
                for role, key in (
                    ("base_0_rgb", "observation/image"),
                    ("left_wrist_0_rgb", "observation/left_wrist_image"),
                    ("right_wrist_0_rgb", "observation/right_wrist_image"),
                )
            },
            "image_mask": dict.fromkeys(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), np.True_),
        }
        if "actions" in data:
            result["actions"] = vector(data["actions"], ACTION_DIM, "actions")
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class LemonOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :ACTION_DIM]}
