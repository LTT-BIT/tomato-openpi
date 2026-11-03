"""P2.5 time alignment, causal counterfactuals and exact P2 sampler-prefix tests."""

import json
from pathlib import Path
import unittest

import numpy as np
from plan_p3 import extend_plan
from prepare_p25 import anchors_for_episode
from prepare_p25 import digest
import pytest
from report_p25 import aligned_difference
from report_p25 import unconstrained_reference

from openpi.training.flash_dataset import FlashLeRobotDataset

ROOT = Path(__file__).resolve().parents[2]


class ExtendedValidationTest(unittest.TestCase):
    def test_overlap_uses_identical_physical_times_and_boundary_50(self):
        absolute = np.arange(200)[:, None] * np.arange(1, 59)[None, :]
        old = absolute[20:74]
        for delta in (10, 15, 50):
            new = absolute[20 + delta : 74 + delta]
            assert np.count_nonzero(aligned_difference(old, new, delta)) == 0
            np.testing.assert_array_equal(aligned_difference(old, new + 3, delta), 3)
        with pytest.raises(ValueError, match="Only 10/15"):
            aligned_difference(old, old, 49)

    def test_anchor_selection_is_deterministic_and_replans_in_bounds(self):
        states = np.zeros((300, 58), dtype=np.float32)
        states[150:, 8] = 1
        anchors = anchors_for_episode(states)
        assert anchors == anchors_for_episode(states)
        assert abs(anchors[0][1] - anchors[1][1]) >= 50
        assert all(start >= 7 and start + 101 < len(states) for _, start in anchors)

    def test_unconstrained_svd_reference_recovers_degree_six_in_float64(self):
        x = np.linspace(-1, 1, 54)
        target = np.broadcast_to((x**6 - x**3)[None, :, None], (2, 54, 58))
        reconstructed = unconstrained_reference(target, [-2.0] * 58, [3.0] * 58)
        np.testing.assert_allclose(reconstructed, target, atol=1e-12, rtol=1e-12)

    def test_real_frozen_extended_plan_and_resume_prefix(self):
        data = ROOT / ".cache/flash_p2_data"
        plan_path = ROOT / ".cache/flash_p25/plan.json"
        if not plan_path.exists():
            self.skipTest("P2.5 real data snapshot unavailable")
        d = FlashLeRobotDataset(data, ROOT / ".cache/flash_p2_images")
        p = json.loads(plan_path.read_text())
        assert digest({k: v for k, v in p.items() if k != "sha256"}) == p["sha256"]
        assert {w["source_recording_id"] for w in p["windows"]} == set(d.split["validation_sources"])
        assert 500 <= len(p["windows"]) <= 1000
        for w in p["windows"]:
            assert d.split["assignments"][w["id"]] == "validation"
            assert w["start"] >= 7
            assert w["start"] + 51 < d.records[w["id"]]["rows"]
        original = json.loads((data / "window_plan.json").read_text())
        candidate = extend_plan(d, original)
        assert candidate == extend_plan(d, original)
        assert candidate["windows"][:100] == original["train"]
        assert len({(x["id"], x["start"]) for x in candidate["windows"]}) == 1000
        for x in candidate["windows"]:
            assert d.split["assignments"][x["id"]] == "train"
            assert x["start"] + 51 < d.records[x["id"]]["rows"]
        assert candidate["next_global_step"] == 101
        assert candidate["model_updates_executed"] == 0


if __name__ == "__main__":
    unittest.main()
