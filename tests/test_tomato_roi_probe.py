import pytest
from scripts.probe_tomato100_roi import checked_detections, sample_frames


def test_saved_segments_only_select_qa_frames():
    row = dict(length=70, training_segments=[dict(start_frame=i*10, end_frame=(i+1)*10,
        segment_index=i+1, label='pickup') for i in range(7)])
    assert [r['frame'] for r in sample_frames(row)] == [4, 14, 24, 34, 44, 54, 64]
    row['training_segments'][1]['start_frame'] = 9
    with pytest.raises(ValueError, match='partition'):
        sample_frames(row)


def test_detection_confidence_cannot_approve_training_roi():
    result = checked_detections(['gripper'], [0.99], [[-2, 3, 110, 70]], 100, 80)
    assert result[0]['box_xyxy'] == [0, 3, 100, 70]
    assert result[0]['approved_for_training'] is False
    assert result[0]['review_status'] == 'unreviewed_detector_candidate'
    assert checked_detections(['bad'], [.9], [[-3, 1, -2, 5]], 100, 80) == []


def test_corrupt_detection_rejected():
    with pytest.raises(ValueError, match='Nonfinite'):
        checked_detections(['x'], [.9], [[0, 0, float('nan'), 2]], 100, 80)
    with pytest.raises(ValueError, match='length'):
        checked_detections(['x'], [], [], 100, 80)
    with pytest.raises(ValueError, match='score'):
        checked_detections(['x'], [1.1], [[0, 0, 2, 2]], 100, 80)
