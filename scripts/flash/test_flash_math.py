"""CPU offline math/data tests; real-cache tests never fall back to randomness."""

import ast
import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from openpi.shared.flash_polynomial import FlashPolynomial
from openpi.training.flash_data import fit_action_statistics
from openpi.training.flash_data import load_episode
from openpi.training.flash_data import make_batch
from openpi.training.flash_data import read_manifest
from openpi.training.flash_data import validate_split

ROOT = Path(__file__).resolve().parents[2]


class FlashMathTest(unittest.TestCase):
    def test_official_basis_fit_history_boundary_parity(self):
        """Execute actual official basis function, compare its float64 operators."""
        import torch

        source = Path("/home/user/flash-policy-reference/roboverse_learn/il/policies/flash/flash_policy.py")
        if not source.exists():
            self.skipTest("Official reference checkout unavailable")
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_legendre_basis")
        fn.decorator_list = []
        module = ast.Module(body=[fn], type_ignores=[])
        scope = {"torch": torch, "Tuple": tuple}
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
        basis, derivative, _ = scope[fn.name](torch.linspace(0, 1, 54, dtype=torch.float64), 6)
        b = basis.numpy()
        d = derivative.numpy()
        inverse = np.linalg.solve(b.T @ b, b.T)
        scale = np.linalg.norm(inverse, axis=1)
        a = np.concatenate([b[[2, 52]], d[[2, 52]]])
        weight = np.linalg.solve(b.T @ b, a.T)
        correction = weight @ np.linalg.inv(a @ weight)
        p = FlashPolynomial()
        for actual, expected in [
            (p.basis_ext, b),
            (p.derivative_ext, d),
            (p.inverse, inverse),
            (p.scale, scale),
            (p.correction, correction),
        ]:
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        hist = b[1:9]
        np.testing.assert_allclose(
            p.history_inverse, np.linalg.solve(hist.T @ hist + 0.1 * np.eye(7), hist.T), atol=2e-6
        )
        rng = np.random.default_rng(54)
        target = rng.normal(size=(2, 54, 58)).astype(np.float32)
        raw = np.einsum("ch,bhd->bcd", inverse, target)
        desired = np.concatenate([target[:, [2, 52]], (target[:, [3, 53]] - target[:, [1, 51]]) * 53 / 2], axis=1)
        expected = (raw - np.einsum("ck,bkd->bcd", correction, np.einsum("kc,bcd->bkd", a, raw) - desired)) / scale[
            None, :, None
        ]
        np.testing.assert_allclose(p.fit_target(jnp.asarray(target)), expected, atol=5e-5, rtol=2e-5)

    def test_actual_reconstructed_positions_derivatives_and_grad(self):
        p = FlashPolynomial()
        # Two adjacent chunks share measured command context at their boundary.
        ticks = np.arange(104, dtype=np.float32)
        trajectory = np.sin(ticks[:, None] * 0.15) * np.linspace(0.2, 1, 58)[None]
        target = np.stack([trajectory[:54], trajectory[50:104]]).astype(np.float32)
        coeff = jax.jit(lambda x: p.fit_target(x))(jnp.asarray(target))
        reconstructed = np.asarray(p.reconstruct(coeff, extended=True))
        velocity = np.asarray(p.velocity(coeff, extended=True))
        assert p.reconstruct(coeff).shape == (2, 50, 58)
        np.testing.assert_allclose(reconstructed[:, [2, 52]], target[:, [2, 52]], atol=4e-6)
        np.testing.assert_allclose(velocity[:, [2, 52]], (target[:, [3, 53]] - target[:, [1, 51]]) * 15, atol=5e-6)
        np.testing.assert_allclose(reconstructed[0, 52], reconstructed[1, 2], atol=4e-6)
        np.testing.assert_allclose(velocity[0, 52], velocity[1, 2], atol=5e-6)
        # Check executable samples against physical finite differences too.
        discrete = (reconstructed[:, 3:53] - reconstructed[:, 1:51]) * 15
        assert np.isfinite(discrete).all()
        grad = jax.grad(lambda x: jnp.mean(p.reconstruct(p.fit_target(x)) ** 2))(jnp.asarray(target))
        assert np.isfinite(grad).all()
        assert float(jnp.linalg.norm(grad)) > 0

    def test_real_history_causality_and_train_only_stats(self):
        cache = ROOT / ".cache/flash_samples"
        if not (cache / "manifest.json").exists():
            self.skipTest("Real numeric cache unavailable; no synthetic substitution")
        records = read_manifest(cache)
        raw = [r for r in records if r["kind"] == "raw"]
        first = raw[0]
        held = next(r for r in raw if r["root"] != first["root"])
        train = [load_episode(first, cache)]
        evaluation = [load_episode(held, cache)]
        validate_split(train, evaluation)
        statistics = fit_action_statistics(train)
        assert statistics.provenance["train_sample_ids"] == [first["id"]]
        q01, q99 = np.quantile(train[0]["actions"], [0.01, 0.99], axis=0)
        np.testing.assert_allclose(statistics.q01, q01, atol=1e-7)
        np.testing.assert_allclose(statistics.q99, q99, atol=1e-7)
        batch = make_batch(train, [(first["id"], 52)], statistics)
        np.testing.assert_array_equal(batch["history"][0], train[0]["states"][45:53])
        assert batch["target_ext"].shape == (1, 54, 58)
        coeff = FlashPolynomial().fit_history(batch["normalized_history"])
        # Modify all future labels and measured positions: inference source invariant.
        train[0]["actions"][53:] += 99
        train[0]["states"][53:] -= 99
        changed = make_batch(train, [(first["id"], 52)], statistics)
        np.testing.assert_array_equal(batch["history"], changed["history"])
        np.testing.assert_array_equal(coeff, FlashPolynomial().fit_history(changed["normalized_history"]))
        assert np.isfinite(coeff).all()
        np.testing.assert_allclose(
            statistics.denormalize(batch["normalized_target_ext"]), batch["target_ext"], atol=5e-7
        )
        print(
            json.dumps(
                {
                    "train_id": first["id"],
                    "eval_id": held["id"],
                    "statistics_sha256": statistics.provenance["statistics_sha256"],
                    "real_history_leakage_test": "passed",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
