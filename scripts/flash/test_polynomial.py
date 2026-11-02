"""Algebra/semantics checks complement (and do not replace) real-data reconstruction."""

import ast
import os
from pathlib import Path
import unittest

import numpy as np
from polynomial import absolute_actions
from polynomial import basis
from polynomial import fit
from polynomial import operators
import pytest
import torch


class PolynomialTest(unittest.TestCase):
    def test_official_basis_parity(self):
        path = (
            Path(os.environ.get("FLASH_REFERENCE", "/home/user/flash-policy-reference"))
            / "roboverse_learn/il/policies/flash/flash_policy.py"
        )
        if not path.exists():
            self.skipTest("Set FLASH_REFERENCE to the official checkout for parity validation")
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashDiTImagePolicy")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_legendre_basis")
        method.decorator_list = []
        namespace = {"torch": torch, "Tuple": tuple}
        # Only execute the pure basis function; do not import RoboVerse or simulator dependencies.
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        for degree in (3, 6, 9, 12):
            matrices = namespace[method.name](torch.linspace(0, 1, 54, dtype=torch.float64), degree)
            for derivative, matrix in enumerate(matrices):
                np.testing.assert_allclose(
                    matrix.numpy(), basis(np.arange(54), degree, 0, 53, 53, derivative), atol=1e-9
                )

    def test_exact_polynomial_recovery_and_scaling(self):
        for degree in (3, 6, 9, 12):
            op = operators(degree)
            true = np.arange((degree + 1) * 2, dtype=float).reshape(1, degree + 1, 2) / 100
            samples = np.einsum("hk,bkd->bhd", op["ev"], true)
            fitted = fit(samples, op)
            np.testing.assert_allclose(fitted, true, atol=1e-12)
            normalized = fitted / op["scale"][None, :, None]
            np.testing.assert_allclose(normalized * op["scale"][None, :, None], true, atol=1e-12)

    def test_c1_shared_boundary_and_velocity_seconds(self):
        op = operators(6, constrained=True, fps=30)
        ticks = op["ticks"]
        q = np.stack([np.sin(ticks / 20), np.cos(ticks / 30)], axis=-1)[None]
        pos = np.array([[[0.3, 0.4], [0.5, 0.7]]])
        vel = np.array([[[0.1, 0.2], [0.3, 0.4]]])
        c = fit(q, op, pos, vel)
        np.testing.assert_allclose(np.einsum("hk,bkd->bhd", op["boundary"], c), pos, atol=1e-11)
        np.testing.assert_allclose(np.einsum("hk,bkd->bhd", op["boundary_vel"], c), vel, atol=1e-11)
        # x(tick)=tick/30 has derivative 1 rad/s at 30Hz.
        line = (ticks / 30)[None, :, None]
        c = fit(line, op, np.array([[[0.0], [50 / 30]]]), np.ones((1, 2, 1)))
        np.testing.assert_allclose(np.einsum("hk,bkd->bhd", op["vel"], c), 1, atol=1e-12)

    def test_sparse_grid_has_same_physical_span(self):
        for stride in (1, 2, 4):
            for degree in (3, 6, 9, 12):
                op = operators(degree, stride)
                assert op["ticks"][0] == 0
                assert op["ticks"][-1] == 49
                assert op["ev"].shape == (50, degree + 1)

    def test_absolute_and_delta_are_distinct(self):
        actions = np.ones((1, 3, 2))
        state = np.array([[10.0, 20.0]])
        np.testing.assert_array_equal(absolute_actions(actions, None, "absolute"), actions)
        rel = absolute_actions(actions, state, "state_relative", [True, False])
        np.testing.assert_array_equal(rel[0, :, 0], [11, 11, 11])
        np.testing.assert_array_equal(rel[0, :, 1], [1, 1, 1])
        temporal = absolute_actions(actions, state, "temporal_delta", [True, False])
        np.testing.assert_array_equal(temporal[0, :, 0], [11, 12, 13])
        with pytest.raises(ValueError, match="measured state anchor"):
            absolute_actions(actions, None, "state_relative")

    def test_invalid_fits_fail(self):
        with pytest.raises(ValueError, match="Too few sparse nodes"):
            operators(12, 10)
        with pytest.raises(ValueError, match="stride must be"):
            operators(6, 0)
        with pytest.raises(ValueError, match="C1 requires"):
            fit(np.zeros((1, 54, 2)), operators(6, constrained=True))

    def test_derivative_finite_difference(self):
        t = np.array([3.0, 15.0, 33.0])
        eps = 1e-4
        finite = (basis(t + eps, 9, 0, 49, 30) - basis(t - eps, 9, 0, 49, 30)) / (2 * eps / 30)
        np.testing.assert_allclose(basis(t, 9, 0, 49, 30, 1), finite, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
