"""FLASH polynomial operators, matching NTUMARS/FLASH-Policy f47e490f.

K is degree (K+1 coefficient tokens). All fitting uses normalized absolute actions.
The official prefix=1 history source reparameterizes measured ticks -7..0 onto
first eight middle-grid rows (target ticks -1..6). This is deliberately the
published implementation, NOT extrapolation at historical physical times.
Boundary projection constructs TRAINING TARGETS only; future endpoint labels
must never be provided to inference. It does not guarantee sampled C1 outputs.
All polynomial contractions request HIGHEST precision: GPU reduced-precision
float32 matmuls can otherwise visibly disturb actual boundary anchors.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import legendre


@dataclasses.dataclass(frozen=True, eq=False)
class FlashPolynomial:
    degree: int = 6
    horizon: int = 50
    history_steps: int = 8
    history_reg_lambda: float = 0.1
    fps: float = 30.0

    def __post_init__(self):
        if self.degree < 3 or self.horizon != 50 or self.history_steps != 8:
            raise ValueError("Standard FLASH adapter requires K>=3, horizon=50, history=8")
        if self.history_reg_lambda <= 0 or self.fps <= 0:
            raise ValueError("Positive history regularizer and fps required")
        ticks = np.arange(-2, self.horizon + 2, dtype=np.float64)
        s = (ticks + 2) / (self.horizon + 3)
        eye = np.eye(self.degree + 1)
        basis = np.stack([legendre.legval(2 * s - 1, c) for c in eye], axis=-1)
        ds = np.stack([2 * legendre.legval(2 * s - 1, legendre.legder(c)) for c in eye], axis=-1)
        inverse = np.linalg.pinv(basis)
        scale = np.maximum(np.linalg.norm(inverse, axis=1), 1e-08)
        hist_basis = basis[1 : 1 + self.history_steps]
        hist_inverse = np.linalg.solve(
            hist_basis.T @ hist_basis + self.history_reg_lambda * np.eye(self.degree + 1), hist_basis.T
        )
        boundary = np.concatenate([basis[[2, 52]], ds[[2, 52]]])
        weighted = np.linalg.solve(basis.T @ basis, boundary.T)
        correction = np.linalg.solve(boundary @ weighted, weighted.T).T
        for name, value in {
            "basis_ext": basis,
            "inverse": inverse,
            "scale": scale,
            "history_inverse": hist_inverse,
            "boundary": boundary,
            "correction": correction,
            "derivative_ext": ds,
        }.items():
            object.__setattr__(self, name, value.astype(np.float32))

    def fit_target(self, target_ext, *, boundary_constraint=True):
        """(B,54,D) -> scaled (B,7,D); target ticks -2..51."""
        if target_ext.shape[-2] != 54:
            raise ValueError("Target must include real context ticks -2..51")
        raw = jnp.einsum("ch,bhd->bcd", jnp.asarray(self.inverse), target_ext, precision=jax.lax.Precision.HIGHEST)
        if boundary_constraint:
            position = target_ext[:, jnp.array([2, 52])]
            velocity = (target_ext[:, jnp.array([3, 53])] - target_ext[:, jnp.array([1, 51])]) * 53 / 2
            desired = jnp.concatenate([position, velocity], axis=1)
            residual = (
                jnp.einsum("kc,bcd->bkd", jnp.asarray(self.boundary), raw, precision=jax.lax.Precision.HIGHEST)
                - desired
            )
            raw -= jnp.einsum(
                "ck,bkd->bcd", jnp.asarray(self.correction), residual, precision=jax.lax.Precision.HIGHEST
            )
        return raw / jnp.asarray(self.scale)[None, :, None]

    def fit_history(self, history):
        """Measured positions only, same routine during training and deployment."""
        if history.shape[-2] != self.history_steps:
            raise ValueError("History must contain eight measured positions, all <= current tick")
        return (
            jnp.einsum("ch,bhd->bcd", jnp.asarray(self.history_inverse), history, precision=jax.lax.Precision.HIGHEST)
            / jnp.asarray(self.scale)[None, :, None]
        )

    def reconstruct(self, coeff, *, extended=False):
        basis = self.basis_ext if extended else self.basis_ext[2:52]
        return jnp.einsum(
            "hc,bcd->bhd",
            jnp.asarray(basis),
            coeff * jnp.asarray(self.scale)[None, :, None],
            precision=jax.lax.Precision.HIGHEST,
        )

    def velocity(self, coeff, *, extended=False):
        """Analytic derivative in normalized-action units per physical second."""
        basis = self.derivative_ext if extended else self.derivative_ext[2:52]
        return jnp.einsum(
            "hc,bcd->bhd",
            jnp.asarray(basis) * self.fps / 53,
            coeff * jnp.asarray(self.scale)[None, :, None],
            precision=jax.lax.Precision.HIGHEST,
        )


def normalize_actions(actions, q01, q99):
    """Main q01/q99 affine semantics: no clipping and epsilon in denominator."""
    return (actions - q01) / (q99 - q01 + 1e-06) * 2.0 - 1.0


def unnormalize_actions(actions, q01, q99):
    return (actions + 1.0) / 2.0 * (q99 - q01 + 1e-06) + q01
