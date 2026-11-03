"""Numerical and real-artifact regression checks for the P0.5 audit."""

import json
from pathlib import Path

from diagnose_local import exact_source_mapping
from diagnose_local import metrics
from diagnose_local import solver_operator
import numpy as np
from polynomial import basis


def test_quantiles_count_absolute_errors_not_signed_cancellation():
    values = np.array([-4.0, -2.0, 0.0, 1.0, 3.0])
    out = metrics(values)
    assert out["mae"] == 2.0
    assert out["max"] == 4.0
    np.testing.assert_allclose(out["p95"], 3.8)
    np.testing.assert_allclose(out["p99"], 3.96)


def test_stable_solvers_and_regularization():
    ticks = np.unique(np.r_[np.arange(0, 50, 4), 49])
    s = basis(ticks, 12, 0, 49, 30)
    qr = solver_operator(s, "qr")
    svd = solver_operator(s, "svd")
    np.testing.assert_allclose(qr, svd, atol=3e-13)
    np.testing.assert_allclose(qr @ s, np.eye(13), atol=3e-13)
    ridge = solver_operator(s, "ridge", 0.01)
    assert np.linalg.norm(ridge, ord=2) < np.linalg.norm(svd, ord=2)
    assert np.isfinite(ridge).all()


def test_source_mapping_keeps_int64_nanosecond_precision():
    reference = np.array([1793597090600296960 + i for i in range(6)], dtype=np.int64)
    source = reference[[0, 2, 5]]
    np.testing.assert_array_equal(exact_source_mapping(source, reference), [0, 2, 5])
    # Casting to float would collapse all six source timestamps to one value.
    assert len(np.unique(reference.astype(float))) == 1


def test_real_diagnostic_artifact_agrees_with_p0():
    root = Path(__file__).resolve().parents[2] / "docs/flash/results"
    result = json.loads((root / "p05/diagnosis.json").read_text())
    p0 = json.loads((root / "reconstruction.json").read_text())
    assert result["episodes"] == 69
    assert result["windows"] == 2195
    assert result["manifest_sha256"] == p0["source_manifest_sha256"]
    for row in result["stats"]:
        assert row["p95"] <= row["p99"] <= row["max"]
        if row["kind"] == "all":
            continue
        old = next(
            r
            for r in p0["results"]
            if (r["kind"], r["degree"], r["stride"], r["mode"]) == (row["kind"], row["degree"], 1, "ols50")
        )
        for name in ["mae", "rmse"]:
            np.testing.assert_allclose(row[name], old["groups"][row["group"]]["errors"][name], atol=1e-13)
    assert result["qr_svd_max_prediction_difference"] < 1e-10
    assert len(result["source_time_provenance"]) == 69
    for pair in result["matched_originals"]:
        assert pair["mapping_status"] == "exact_int64_source_timestamp_ns"
        assert pair["mapping_backward_steps"] == 0
        if pair["kind"] == "raw":
            assert pair["matched_action_max_error"] == 0.0
    assert max(pair["matched_action_max_error"] for pair in result["matched_originals"]) > 0.5
    for row in result["boundary"]:
        assert np.isfinite(row["max"])
        if row["mode"] == "oracle_c1" and row["metric"] in ["boundary_position_error", "boundary_velocity_error"]:
            assert row["max"] < 1e-10
