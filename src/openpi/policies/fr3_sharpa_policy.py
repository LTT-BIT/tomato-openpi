"""Training transforms for measured 58D state and commanded 58D FR3/Sharpa actions."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.fr3_sharpa_protocol import ACTION_DIM
from openpi.policies.fr3_sharpa_protocol import ORDER


def _parse_image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _vector(value, label):
    # DeltaActions updates actions in place; preserve the original measured/commanded sample.
    value = np.array(value, dtype=np.float32, copy=True)
    if value.ndim == 0 or value.shape[-1] != ACTION_DIM:
        raise ValueError(f"Expected FR3/Sharpa {ACTION_DIM}D {label}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"FR3/Sharpa {label} contains NaN or Inf")
    return value


@dataclasses.dataclass(frozen=True)
class Fr3SharpaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    # Include physical ordering in serialized normalization/recovery contracts.
    joint_order: tuple[str, ...] = ORDER

    def __post_init__(self):
        if self.joint_order != ORDER:
            raise ValueError("FR3/Sharpa v1 uses a fixed physical joint order; define a new adapter")

    def __call__(self, data: dict) -> dict:
        result = {
            "state": _vector(data["observation/state"], "measured state"),
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist_image"]),
            },
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.True_},
        }
        if "actions" in data:
            result["actions"] = _vector(data["actions"], "commanded action")
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class Fr3SharpaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 0 or actions.shape[-1] < ACTION_DIM:
            raise ValueError(f"Expected at least {ACTION_DIM} FR3/Sharpa action values, got {actions.shape}")
        return {"actions": actions[..., :ACTION_DIM]}
