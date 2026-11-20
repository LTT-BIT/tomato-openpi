import numpy as np

from scripts.tomato_coverage_roi import CoverageWindow
from scripts.tomato_coverage_roi import EvidenceHandTracker
from scripts.tomato_coverage_roi import StableRoi
from scripts.tomato_repair_locator import RgbLocator
from scripts.tomato_repair_locator import merge_detections
from scripts.tomato_stable_roi import SmoothWindow


def test_coverage_window_recovers_switch_without_jumping_or_fabricating_coverage():
    old, new = SmoothWindow(), CoverageWindow()
    counts = [0, 0]
    previous = None
    targets = [np.array([430.0, 200.0, 510.0, 280.0])] * 30
    targets += [np.array([100.0, 210.0, 480.0, 300.0])] * 100
    for target in targets:
        counts[0] += not old.update(target, 640, 400)[1]
        box, covered = new.update(target, 640, 400)
        counts[1] += not covered
        side = box[2] - box[0]
        assert side == box[3] - box[1]
        assert (box[:2] >= 0).all()
        assert (box[2:] <= [640, 400]).all()
        if covered:
            assert (target[:2] >= box[:2]).all()
            assert (target[2:] <= box[2:]).all()
        if previous is not None:
            old_side = previous[2] - previous[0]
            assert max(side / old_side, old_side / side) <= 1.025
            assert np.linalg.norm((box[:2] + box[2:] - previous[:2] - previous[2:]) / 2) / old_side < 0.04
        previous = box
    assert counts[1] < counts[0]


def test_uncontainable_target_and_missing_hand_do_not_become_valid():
    window = CoverageWindow()
    for _ in range(100):
        _, covered = window.update(np.array([0.0, 0.0, 640.0, 400.0]), 640, 400)
        assert not covered
    assert not window.update(None, 640, 400)[1]
    roi = StableRoi()
    row = roi.select([], np.zeros(116), 640, 400, np.zeros((400, 640, 3), np.uint8))
    assert not row["valid"]
    assert row["crop_box"] is None
    assert row["interaction_box"] is None


def test_small_stationary_noise_keeps_detail_and_is_causal():
    targets = [np.array([440.0, 210.0, 510.0, 280.0]) + np.tile([i % 2, 0], 2) for i in range(60)]

    def replay(sequence):
        window = CoverageWindow()
        return [window.update(t, 640, 400)[0].tolist() for t in sequence]

    boxes = replay(targets)
    assert max(b[2] - b[0] for b in boxes) == 220
    assert max(b[0] for b in boxes) - min(b[0] for b in boxes) <= 2
    assert boxes == replay([*targets, np.array([10.0, 10.0, 80.0, 80.0])])[: len(targets)]


def test_repair_retains_strong_original_and_does_not_relax_thresholds():
    original = [{"label": 1, "score": 0.92, "box": [1, 2, 3, 4]}, {"label": 2, "score": 0.6, "box": [5, 6, 7, 8]}]
    repair = [{"label": 1, "score": 0.99, "box": [9, 9, 10, 10]}, {"label": 2, "score": 0.91, "box": [11, 12, 13, 14]}]
    assert merge_detections(original, repair) == [original[0], repair[1]]
    repair[1]["score"] = 0.84
    assert merge_detections(original, repair) == original


def test_strong_current_hand_does_not_need_flow_but_missing_hand_stays_invalid(monkeypatch):
    tracker = EvidenceHandTracker()
    image = np.zeros((400, 640), np.uint8)

    def forbidden(*args):
        raise AssertionError("Strong current evidence should skip LK")

    monkeypatch.setattr(tracker, "flow_box", forbidden)
    for x in (440, 441):
        box, source = tracker.update(image, [{"score": 0.95, "box": [x, 200, x + 60, 270]}])
        assert source == "current_detector"
        assert box is not None
    monkeypatch.undo()
    assert tracker.update(image, [])[0] is None


def test_repair_network_only_runs_on_frames_missing_a_strong_hand():
    right = {"label": 1, "score": 0.95, "box": [100, 100, 170, 180]}
    left = {"label": 2, "score": 0.96, "box": [10, 100, 80, 180]}
    calls = []

    class Primary:
        def predict(self, images):
            return [[right, left], [left]]

    class Repair:
        def predict(self, images):
            calls.append(images)
            return [[right]]

    locator = RgbLocator.__new__(RgbLocator)
    locator.primary, locator.repair = Primary(), Repair()
    assert locator.predict(["first", "second"]) == [[right, left], [right, left]]
    assert calls == [["second"]]
