import numpy as np
from scripts.tomato_roi_locator import CausalRoleSelector,near_red_regions
from scripts.prepare_tomato100_vectors import window_indices


def detection(label,box,score=.95):return dict(label=label,box=box,score=score)


def test_lost_and_reacquired_use_only_current_boxes():
    s=CausalRoleSelector(np.ones(58));q=np.zeros(116)
    one=[detection(1,[70,30,95,55])]
    assert s.select(one,q,120,100)['valid']
    lost=s.select([],q,120,100)
    assert not lost['valid'] and lost['crop_box'] is None
    recovered=s.select([detection(1,[72,31,97,56])],q,120,100)
    assert recovered['valid'] and recovered['interaction_box']==[72,31,97,56]


def test_whole_arm_is_not_accepted_as_hand_or_used_as_fallback():
    s=CausalRoleSelector(np.ones(58))
    result=s.select([detection(1,[0,0,120,100])],np.zeros(116),120,100)
    assert not result['valid'] and result['crop_box'] is None


def test_close_hands_and_near_object_are_contained():
    s=CausalRoleSelector(np.ones(58))
    result=s.select([detection(1,[65,30,85,55]),detection(2,[42,33,62,58]),detection(3,[60,52,75,66])],np.zeros(116),120,100)
    assert result['roles']==[1,2]
    x1,y1,x2,y2=result['interaction_box']
    assert x1<=42 and y1<=30 and x2>=85 and y2>=66


def test_measured_motion_selects_active_far_hand_without_phase_labels():
    s=CausalRoleSelector(np.ones(58));q=np.zeros(116)
    items=[detection(1,[85,30,110,55]),detection(2,[10,30,35,55])]
    assert s.select(items,q,140,100)['roles']==[1]
    q[:7]=.1
    assert s.select(items,q,140,100)['roles']==[2]


def test_visible_red_fruit_extends_crop_but_far_red_does_not():
    im=np.zeros((100,120,3),np.uint8);im[55:65,25:40]=[220,0,0];im[2:7,2:7]=[220,0,0]
    regions=near_red_regions(im,[40,30,70,55])
    assert len(regions)==1
    np.testing.assert_array_equal(regions[0],[25,55,40,65])


def test_action_windows_pad_inside_episode_without_phase_holds():
    result=window_indices(3,[0,1,2],5)
    np.testing.assert_array_equal(result,[[0,1,2,2,2],[1,2,2,2,2],[2,2,2,2,2]])
