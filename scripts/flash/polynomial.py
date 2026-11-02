"""NumPy Legendre representation in physical seconds, independent of robot runtime.

K is the highest degree; coefficient count is K+1. Delta conversion is explicit:
OpenPI's state-relative DeltaActions is NOT a temporal difference and must not be cumulatively summed.
"""

import numpy as np
from numpy.polynomial import legendre


def absolute_actions(actions, state, mode, mask=None):
    actions = np.asarray(actions, dtype=np.float64)
    if mode == "absolute":
        return actions.copy()
    if state is None:
        raise ValueError("Delta actions require a measured state anchor")
    mask = np.ones(actions.shape[-1], bool) if mask is None else np.asarray(mask, bool)
    out = actions.copy()
    if mode == "state_relative":
        out[..., mask] += np.asarray(state)[..., None, mask]
    elif mode == "temporal_delta":
        out[..., mask] = np.asarray(state)[..., None, mask] + np.cumsum(out[..., mask], axis=-2)
    else:
        raise ValueError("Unknown action representation")
    return out


def basis(ticks, degree, lower, upper, fps, derivative=0):
    if degree < 0 or upper <= lower or fps <= 0:
        raise ValueError("Invalid polynomial grid")
    x = 2 * (np.asarray(ticks) - lower) / (upper - lower) - 1
    eye = np.eye(degree + 1)
    return (
        np.stack([legendre.legval(x, legendre.legder(c, m=derivative)) for c in eye], axis=-1)
        * (2 * fps / (upper - lower)) ** derivative
    )


def operators(degree, stride=1, *, constrained=False, fps=30):
    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be positive integer")
    # Two-tier extension: overlap [-1,50], outer fitting pad [-2,51].
    lo, hi = (-2, 51) if constrained else (0, 49)
    fit_ticks = np.unique(np.r_[np.arange(lo, hi + 1, stride), hi])
    if len(fit_ticks) < degree + 1:
        raise ValueError("Too few sparse nodes for degree")
    s = basis(fit_ticks, degree, lo, hi, fps)
    inverse = np.linalg.pinv(s)
    scale = np.maximum(np.linalg.norm(inverse, axis=1), 1e-8)
    ev = basis(np.arange(50), degree, lo, hi, fps)
    vel = basis(np.arange(50), degree, lo, hi, fps, 1)
    boundary = basis([0, 50], degree, lo, hi, fps)
    boundary_vel = basis([0, 50], degree, lo, hi, fps, 1)
    correction = None
    if constrained:
        a = np.concatenate([boundary, boundary_vel])
        weighted = np.linalg.solve(s.T @ s, a.T)
        correction = weighted @ np.linalg.inv(a @ weighted)
    return {
        "ticks": fit_ticks,
        "inverse": inverse,
        "scale": scale,
        "ev": ev,
        "vel": vel,
        "boundary": boundary,
        "boundary_vel": boundary_vel,
        "correction": correction,
        "condition": float(np.linalg.cond(s)),
    }


def fit(windows, op, anchor_position=None, anchor_velocity=None):
    """Fit B x sampled nodes x D to RAW coefficients. Scaling remains explicit."""
    c = np.einsum("kh,bhd->bkd", op["inverse"], windows)
    if op["correction"] is not None:
        if anchor_position is None or anchor_velocity is None:
            raise ValueError("C1 requires position and velocity anchors")
        a = np.concatenate([op["boundary"], op["boundary_vel"]])
        residual = np.einsum("hk,bkd->bhd", a, c) - np.concatenate([anchor_position, anchor_velocity], axis=1)
        c -= np.einsum("kh,bhd->bkd", op["correction"], residual)
    return c
