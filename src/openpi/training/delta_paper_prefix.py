"""Physical-prefix bridge for external RTC clients; no robot execution queue.

Inputs use canonical LA7/LH22/RA7/RH22 order and physical units. Only the
returned prefix length is committed; padding supplies the full model horizon.
"""

from dataclasses import replace

import numpy as np

from openpi import transforms
from openpi.shared.delta_paper import ARMS
from openpi.shared.delta_paper import ArmNormalization


def _contract(data):
    horizon, dimension = data.norm_stats["actions"].q01.shape
    if dimension != 58 or not getattr(data, "paper_bundle_sha256", None):
        raise ValueError("RTC prefix requires a validated PaperDataView with Hx58 statistics")
    bridges = [item for item in data.model_transforms.inputs if isinstance(item, ArmNormalization)]
    deltas = [item for item in data.data_transforms.inputs if isinstance(item, transforms.DeltaActions)]
    expected_mask = np.zeros(58, bool)
    expected_mask[ARMS] = True
    if (
        len(bridges) != 1
        or bridges[0].bundle_sha256 != data.paper_bundle_sha256
        or len(deltas) != 1
        or not np.array_equal(deltas[0].mask, expected_mask)
    ):
        raise ValueError("RTC prefix requires the matching Paper bridge and arm-only delta transform")
    return horizon, bridges[0], deltas[0]


def _state(value):
    state = np.asarray(value, dtype=np.float32)
    if state.shape != (58,) or not np.isfinite(state).all():
        raise ValueError("RTC reference state must be finite physical canonical 58D positions")
    return state


def encode_prefix(data, physical_state, absolute_prefix):
    """Re-anchor absolute queued targets on NEW state and NEW timestamp indices.

    Return full Hx58 model-space actions. Do not use normalized observation.state
    here. Callers pass only len(absolute_prefix) as the committed prefix length.
    """
    horizon, bridge, delta = _contract(data)
    state = _state(physical_state)
    absolute = np.asarray(absolute_prefix, dtype=np.float32)
    if absolute.ndim != 2 or absolute.shape[1] != 58 or not 1 <= len(absolute) <= horizon:
        raise ValueError("RTC absolute prefix must have shape (1..H, 58)")
    if not np.isfinite(absolute).all():
        raise ValueError("RTC absolute prefix contains nonfinite values")
    padded = np.zeros((horizon, 58), dtype=np.float32)
    padded[: len(absolute)] = absolute
    mixed = delta({"state": state, "actions": padded})["actions"]
    normalized = transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm)({"actions": mixed})
    return bridge(normalized)["actions"]


def decode_actions(data, physical_state, model_actions):
    """Decode a full chunk against the SAME physical start state used to encode it."""
    horizon, bridge, delta = _contract(data)
    state = _state(physical_state)
    actions = np.asarray(model_actions, dtype=np.float32)
    if actions.shape != (horizon, 58) or not np.isfinite(actions).all():
        raise ValueError("RTC model actions must be finite full-horizon Hx58 values")
    quantiles = replace(bridge, inverse=True)({"actions": actions})
    mixed = transforms.Unnormalize({"actions": data.norm_stats["actions"]}, use_quantiles=data.use_quantile_norm)(
        quantiles
    )
    return transforms.AbsoluteActions(delta.mask)({"state": state, "actions": mixed["actions"]})["actions"]
