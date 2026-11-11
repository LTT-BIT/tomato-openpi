import json
from pathlib import Path
import numpy as np
from scripts.prepare_tomato100_roi import worklist
from scripts.tomato_interaction_preview import annotation_available_frame, interaction_at, square_box

RECORDS=Path(__file__).resolve().parents[1]/'training_configs/experiments/tomato_sharpa/subtasks/records'


def test_episode86_complete_geometry_preserves_successful_preview():
    selection=json.loads((RECORDS/'complete100_rgb_crop_20261111_saved_labels_r1.json').read_text())
    spec=json.loads((RECORDS/'complete100_episode86_manual_roi_20261111_r3.json').read_text())
    old=json.loads((RECORDS/'normal93_20261110_interaction_preview_annotations.json').read_text())
    listing=worklist(selection,spec,preview_sha256='test')
    row=next(r for r in listing['episodes'] if r['roi_clips'])
    assert row['episode_index']==86 and row['offline_preview_frames']==2990
    assert listing['remaining_frames']==422566 and not listing['training_ready']
    for stage, segment in enumerate(selection['episodes'][86]['training_segments'],1):
        clips=[c for c in spec['clips'] if c['stage_index']==stage]
        assert clips[0]['start']==segment['start_frame'] and clips[-1]['end']==segment['end_frame']
        assert all(a['end']==b['start'] for a,b in zip(clips,clips[1:]))
    for clip in spec['clips']:
        for frame in range(clip['start'],clip['end']):
            tight=interaction_at(clip,frame,640,400)
            x1,y1,x2,y2=square_box(tight,640,400)
            assert x2-x1==y2-y1
            assert x1<=tight[0] and y1<=tight[1] and x2>=tight[2] and y2>=tight[3]
    for original in old['clips']:
        for frame in range(original['start'],original['end']):
            extended=next(c for c in spec['clips'] if c['start']<=frame<c['end'])
            np.testing.assert_allclose(interaction_at(original,frame,640,400),
                                       interaction_at(extended,frame,640,400))
    transport=next(c for c in spec['clips'] if c['start']==315)
    assert transport['roles']==['left_hand_and_carried_fruit','placement_contact']
    assert annotation_available_frame(transport,315)==330
    assert 'right_hand' not in transport['roles']


def test_interpolated_key_keeps_future_source_provenance():
    import pytest
    clip={'keyframes':[{'frame':0},{'frame':5,'available_at_frame':10},{'frame':9}]}
    assert annotation_available_frame(clip,5)==10
    assert annotation_available_frame(clip,7)==10
    assert annotation_available_frame(clip,9)==9
    clip['keyframes'][1]['available_at_frame']=4
    with pytest.raises(ValueError,match='availability'):
        annotation_available_frame(clip,5)
