"""Explicit adapter for converter 116D state / 58D commanded joint blocks."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model
from openpi.policies import fr3_sharpa_policy
from openpi.policies import fr3_sharpa_protocol as protocol

LEFT_ARM = protocol.ORDER[:7]
LEFT_HAND = protocol.ORDER[7:29]
RIGHT_ARM = protocol.ORDER[29:36]
RIGHT_HAND = protocol.ORDER[36:]
SOURCE_STATE_NAMES = tuple(
    f"{joint}.{kind}"
    for joints in (LEFT_ARM, RIGHT_ARM, LEFT_HAND, RIGHT_HAND)
    for kind in ("position", "velocity")
    for joint in joints
)
SOURCE_ACTION_NAMES = tuple(
    f"{joint}.{kind}"
    for joints, kind in ((LEFT_ARM, "position"), (RIGHT_ARM, "position"), (LEFT_HAND, "command"), (RIGHT_HAND, "command"))
    for joint in joints
)


@dataclasses.dataclass(frozen=True)
class Fr3SharpaMultimodalInputs(transforms.DataTransformFn):
    """Keep the physical 58D model contract without rewriting multimodal storage."""

    model_type: model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        if state.ndim == 0 or state.shape[-1] != 116 or not np.isfinite(state).all():
            raise ValueError("Expected finite converter 116D Sharpa state")
        canonical = {
            **data,
            "observation/state": np.concatenate(
                [state[..., :7], state[..., 28:50], state[..., 14:21], state[..., 72:94]], axis=-1
            ),
        }
        if "actions" in data:
            action = np.asarray(data["actions"])
            if action.ndim == 0 or action.shape[-1] != 58 or not np.isfinite(action).all():
                raise ValueError("Expected finite converter 58D Sharpa action")
            canonical["actions"] = np.concatenate(
                [action[..., :7], action[..., 14:36], action[..., 7:14], action[..., 36:58]], axis=-1
            )
        return fr3_sharpa_policy.Fr3SharpaInputs(self.model_type)(canonical)
