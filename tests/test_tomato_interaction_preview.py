"""Checks for data-preview geometry, pixels, and offline annotation coverage."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.tomato_interaction_preview import interaction_at
from scripts.tomato_interaction_preview import native_crop
from scripts.tomato_interaction_preview import square_box


@pytest.mark.parametrize("box", [[0, 0, 30, 20], [610, 380, 640, 400], [100, 10, 630, 50]])
def test_edge_crop_retains_region_without_distortion(box):
    left, top, right, bottom = square_box(box, 640, 400)
    assert right - left == bottom - top
    assert left <= box[0] < box[2] <= right
    assert top <= box[1] < box[3] <= bottom
    if box[2] - box[0] <= 400:
        assert 0 <= left < right <= 640
        assert 0 <= top < bottom <= 400
    image = np.arange(400 * 640 * 3, dtype=np.uint8).reshape(400, 640, 3)
    crop = native_crop(image, (left, top, right, bottom))
    x1, y1, x2, y2 = box
    np.testing.assert_array_equal(crop[y1 - top : y2 - top, x1 - left : x2 - left], image[y1:y2, x1:x2])


def test_nominal_scale_and_black_padding():
    assert square_box([100, 100, 200, 160], 640, 400) == (75, 55, 225, 205)
    box = square_box([0, 0, 640, 400], 640, 400)
    crop = native_crop(np.full((400, 640, 3), 150, dtype=np.uint8), box)
    assert crop.shape == (640, 640, 3)
    assert np.count_nonzero(crop[:, :, 0] == 150) == 400 * 640
    assert np.count_nonzero(crop[:, :, 0] == 0) == 240 * 640


@pytest.mark.parametrize("box", [[3, 0, 2, 4], [0, 0, 641, 400], [0, 0, float("nan"), 5]])
def test_invalid_boxes_fail(box):
    with pytest.raises(ValueError, match="finite|ordered"):
        square_box(box, 640, 400)


def test_component_union_and_offline_interpolation():
    clip = {
        "start": 0,
        "end": 11,
        "roles": ["hand", "object"],
        "keyframes": [
            {"frame": 0, "boxes": {"hand": [0, 0, 10, 10], "object": [20, 20, 30, 30]}},
            {"frame": 10, "boxes": {"hand": [10, 10, 20, 20], "object": [20, 20, 30, 30]}},
        ],
    }
    np.testing.assert_equal(interaction_at(clip, 5, 640, 400), [5, 5, 30, 30])
    for frame in (-1, 11):
        with pytest.raises(ValueError, match="outside"):
            interaction_at(clip, frame, 640, 400)
    clip["keyframes"][-1]["frame"] = 9
    with pytest.raises(ValueError, match="endpoints"):
        interaction_at(clip, 5, 640, 400)


def test_real_preview_all_frames_keep_required_components():
    path = (
        Path(__file__).resolve().parents[1]
        / "training_configs/experiments/tomato_sharpa/subtasks/records/normal93_20261110_interaction_preview_annotations.json"
    )
    spec = json.loads(path.read_text())
    total = 0
    for clip in spec["clips"]:
        for frame in range(clip["start"], clip["end"]):
            region = interaction_at(clip, frame, *spec["source_size"])
            left, top, right, bottom = square_box(region, *spec["source_size"])
            assert left <= region[0] < region[2] <= right
            assert top <= region[1] < region[3] <= bottom
            total += 1
    assert total == 258
