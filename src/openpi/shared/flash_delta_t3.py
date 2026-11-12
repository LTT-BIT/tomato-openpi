"""Window-anchored Delta and 54-position time-domain FLASH codec."""

import jax.numpy as jnp
import numpy as np

ARMS = np.r_[0:7, 29:36]
MASK = np.isin(np.arange(58), ARMS)
VERSION = "flash-delta-t3-time-v1"


def encode(values, anchor, mean, std, low, high, *, timestamp):
    delta = values - jnp.where(jnp.asarray(MASK), anchor[..., None, :], 0)
    quantile = (delta - low) / (high - low + 1e-6) * 2 - 1
    return jnp.where(jnp.asarray(MASK), (delta - mean) / jnp.maximum(std, 1e-4), quantile) if timestamp else quantile


def decode(values, anchor, mean, std, low, high, *, timestamp):
    delta = (values + 1) / 2 * (high - low + 1e-6) + low
    if timestamp:
        delta = jnp.where(jnp.asarray(MASK), values * jnp.maximum(std, 1e-4) + mean, delta)
    return delta + jnp.where(jnp.asarray(MASK), anchor[..., None, :], 0)


def correlation(samples, beta):
    x = np.asarray(samples, np.float64)
    x = x - x.mean(0)
    scale = x.std(0)
    x = np.where(scale > 1e-8, x / np.where(scale > 1e-8, scale, 1), 0)
    matrix = x.T @ x / len(x)
    np.fill_diagonal(matrix, 1)
    return np.linalg.cholesky((1 - beta) * np.eye(x.shape[1]) + beta * matrix).astype(np.float32)
