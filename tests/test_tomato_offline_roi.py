import numpy as np
import pytest
from scripts.tomato_offline_roi import flow_step,template_step,track,plan_windows


def test_tracking_follows_current_pixels_in_both_time_directions():
    import cv2
    rng=np.random.default_rng(42)
    a=rng.integers(0,255,(160,220),np.uint8)
    b=cv2.warpAffine(a,np.float32([[1,0,3],[0,1,2]]),(220,160))
    box=np.array([60,50,120,110],float)
    forward=flow_step(a,b,box)
    assert forward is not None
    np.testing.assert_allclose(forward,box+[3,2,3,2],atol=.2)
    reverse=flow_step(b,a,forward)
    np.testing.assert_allclose(reverse,box,atol=.2)
    template=template_step(a,b,box)
    np.testing.assert_array_equal(template,box+[3,2,3,2])
    path=track({0:a,1:b},0,box,[1])
    np.testing.assert_allclose(path[1],forward,atol=.2)


def test_tracking_rejects_unrelated_or_textureless_evidence():
    rng=np.random.default_rng(5)
    a=rng.integers(0,255,(160,220),np.uint8)
    b=np.zeros_like(a)
    box=np.array([60,50,120,110],float)
    assert flow_step(a,b,box) is None
    assert template_step(a,b,box) is None
    assert template_step(b,b,box) is None
    assert 1 not in track({0:a,1:b},0,box,[1])


def test_offline_window_anticipates_role_change_without_clipping_or_zoom_jump():
    tight=np.tile([130,180,430,270],(200,1)).astype(float)
    tight[100:]=[330,180,440,270]
    boxes,quality=plan_windows(tight,640,400)
    assert quality['all_boxes_contained']
    assert quality['center_steps_over_3pct']==quality['scale_steps_over_2pct']==0
    assert np.all(boxes[:,2]-boxes[:,0]<=380)
    # Future motion changes earlier coordinates: this is explicitly offline.
    stationary,_=plan_windows(np.tile(tight[0],(200,1)),640,400)
    assert np.any(boxes[85]!=stationary[85])


def test_offline_window_does_not_fabricate_coverage_for_impossible_annotations():
    with pytest.raises(ValueError,match='too large'):
        plan_windows([[0,100,500,300]],640,400)
    with pytest.raises(ValueError,match='Finite'):
        plan_windows([[0,0,float('nan'),30]],640,400)


@pytest.mark.parametrize('tight', [[405,135,640,354], [0,0,200,150], [200,230,360,400]])
def test_offline_window_contains_annotations_at_native_image_edges(tight):
    boxes,quality=plan_windows(np.tile(tight,(90,1)),640,400,minimum=180.,maximum=300.,context_ratio=1.12)
    assert quality['all_boxes_contained']
    assert quality['center_steps_over_3pct']==quality['scale_steps_over_2pct']==0
    assert np.all(boxes[:,:2]>=0) and np.all(boxes[:,2:]<=[640,400])


def test_future_annotation_requires_explicit_offline_reader_and_complete_provenance(tmp_path):
    import gzip,json
    from scripts.materialize_tomato100_four_view import read_candidate_index
    row=dict(source_frame=0,timestamp=0,available_at=1/30,window_support_range=[0,1],
             generation_mode='offline_bidirectional_annotation',valid=True,
             source_size=[640,400],interaction_box=[330,200,420,270],crop_box=[270,100,490,320],
             image_available=True,local_roi_valid=True,input_mode='offline_local_crop',input_box=[270,100,490,320])
    path=tmp_path/'index.gz'
    def write(second):
        with gzip.open(path,'wt') as f:
            f.write(json.dumps(row)+'\n');f.write(json.dumps(second)+'\n')
    second={**row,'source_frame':1,'timestamp':1/30}
    write(second)
    with pytest.raises(ValueError,match='causality'):read_candidate_index(path,2)
    assert len(read_candidate_index(path,2,allow_offline=True))==2
    second['window_support_range']=[0,20];write(second)
    with pytest.raises(ValueError,match='provenance'):read_candidate_index(path,2,allow_offline=True)


def test_export_requires_review_of_every_unverified_frame_and_keeps_current_identity():
    from scripts.export_tomato100_offline_crops import finalize_rows
    rows=[dict(source_frame=0,source_size=[640,400],valid=False,image_available=True,
               interaction_box=[160,210,240,260],roles=[2],repair={'repair_verified':False})]
    with pytest.raises(ValueError,match='exactly one'):finalize_rows(rows,{}, {})
    reviews={0:dict(decision='crop_content_reviewed',annotation_action='include_visible_right_hand_alongside_left_context',
                    additional_hand_box=[330,200,440,280],roles=[1,2])}
    result,geometry=finalize_rows(rows,reviews,{})
    assert geometry['all_boxes_contained'] and result[0]['source_frame']==0
    assert result[0]['valid'] and result[0]['input_mode']=='offline_local_crop'
    assert result[0]['interaction_box']==[160.,200.,440.,280.]
    assert result[0]['local_evidence_source']=='coding_agent_exception_crop_review'
