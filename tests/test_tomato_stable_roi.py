import numpy as np
import pytest

from scripts.tomato_stable_roi import HandTracker, InteractionEvidence, SmoothWindow, StableRoi
from scripts.audit_tomato_stable_roi import check_index
from scripts.tomato_state_guarded_roi import GuardedInteraction,LEFT_MCP_COLUMNS


def test_window_bounds_abrupt_teleport_without_claiming_coverage():
    window = SmoothWindow()
    a = np.array([440,200,510,270],float)
    b = np.array([80,120,180,240],float)
    previous,_ = window.update(a,640,400)
    invalid = 0
    for target in [a]*20+[b]*90+[None]*10+[a]*90:
        box,covered = window.update(target,640,400)
        side = box[2]-box[0]
        assert side==box[3]-box[1] and (box[:2]>=0).all() and (box[2:]<=[640,400]).all()
        assert np.linalg.norm((box[:2]+box[2:]-previous[:2]-previous[2:])/2)<=.025*max(side,previous[2]-previous[0])+1.5
        assert abs(side/(previous[2]-previous[0])-1)<.02
        if covered:
            assert target is not None and (target[:2]>=box[:2]).all() and (target[2:]<=box[2:]).all()
        if target is not None and not covered:
            invalid += 1
        previous = box
    assert invalid>0 and covered


def test_stationary_detector_noise_does_not_make_window_breathe_or_shake():
    window = SmoothWindow()
    target = np.array([430,210,500,280],float)
    first,_ = window.update(target,640,400)
    rng = np.random.default_rng(5)
    for _ in range(100):
        dx,dy = rng.uniform(-1.3,1.3,2)
        noisy = target+np.array([dx,dy,dx,dy])
        box,covered = window.update(noisy,640,400)
        np.testing.assert_array_equal(box,first)
        assert covered


def test_static_basket_fruit_does_not_initiate_left_transport():
    interaction = InteractionEvidence()
    red = np.array([70,240,90,260],float)
    for frame in range(80):
        left = np.array([40+frame*.5,190,100+frame*.5,250])
        assert interaction.choose({1:np.array([440,220,510,280]),2:left},[red])==(1,)
    assert interaction.carried is None


def test_basket_pixel_and_hand_box_jitter_is_not_carried_fruit():
    interaction = InteractionEvidence()
    for frame in range(90):
        jitter = (-1,0,1,0)[frame%4]
        left = np.array([50+jitter,190,110+jitter,250],float)
        red = np.array([70+jitter,240,90+jitter,260],float)
        assert interaction.choose({1:np.array([440,220,510,280]),2:left},[red])==(1,)
    assert interaction.carried is None


def test_carried_red_and_hand_motion_confirm_left_then_release():
    interaction = InteractionEvidence()
    for frame in range(35):
        left = np.array([80+2*frame,170,140+2*frame,240],float)
        red = np.array([110+2*frame,230,125+2*frame,245],float)
        roles = interaction.choose({1:np.array([440,200,500,270]),2:left},[red])
    assert roles==(2,)
    for _ in range(40):
        roles = interaction.choose({1:np.array([440,200,500,270]),2:left},[])
    assert roles==(1,)


def test_single_frame_pair_candidate_does_not_switch_role():
    interaction = InteractionEvidence()
    right = np.array([200,180,260,250],float)
    far = np.array([20,180,80,250],float)
    near = np.array([150,180,205,250],float)
    for frame in range(30):
        assert interaction.choose({1:right,2:near if frame%2 else far},[])==(1,)


def test_loss_shows_current_pixels_window_but_no_valid_training_coordinates():
    roi = StableRoi()
    image = np.zeros((400,640,3),np.uint8)
    detected = roi.select([dict(label=1,score=.95,box=[450,210,510,280])],np.zeros(116),640,400,image)
    assert detected['valid']
    lost = roi.select([],np.zeros(116),640,400,image)
    assert not lost['valid'] and lost['crop_box'] is None and lost['interaction_box'] is None
    assert lost['display_crop_box'] is not None


def test_flow_bridge_requires_current_texture_and_expires():
    import cv2
    rng = np.random.default_rng(4)
    image = np.zeros((200,300),np.uint8)
    image[70:130,90:150] = rng.integers(30,230,(60,60),np.uint8)
    tracker = HandTracker()
    box,_ = tracker.update(image,[dict(box=[85,65,155,135],score=.95)])
    assert box is not None
    for frame in range(1,6):
        current = cv2.warpAffine(image,np.float32([[1,0,frame],[0,1,0]]),(300,200))
        box,source = tracker.update(current,[])
        assert box is not None and source=='verified_current_flow'
    current = cv2.warpAffine(image,np.float32([[1,0,6],[0,1,0]]),(300,200))
    box,source = tracker.update(current,[])
    assert box is None and source=='lost'


def test_future_frames_cannot_change_previous_outputs():
    def replay(images):
        roi = StableRoi()
        return [roi.select([dict(label=1,score=.95,box=[440+i,200,500+i,270])],np.zeros(116),640,400,image)
                for i,image in enumerate(images)]
    images = [np.zeros((400,640,3),np.uint8) for _ in range(12)]
    assert replay(images)==replay(images+[np.full((400,640,3),255,np.uint8)])[:12]


def test_audit_does_not_confuse_continuous_display_with_valid_evidence():
    image = np.zeros((400,640,3),np.uint8)
    roi = StableRoi()
    lost = roi.select([],np.zeros(116),640,400,image)
    row = dict(source_frame=0,timestamp=0,available_at=0,source_size=[640,400],**lost)
    check_index([row],1)
    row['valid'] = True
    row['crop_box'] = row['display_crop_box']
    row['interaction_box'] = [0,0,640,400]
    with pytest.raises(ValueError,match='misses'):
        check_index([row],1)
    row['available_at'] = 1
    with pytest.raises(ValueError,match='causality'):
        check_index([row],1)


def test_open_measured_hand_rejects_moving_basket_color_as_left_transport():
    interaction=GuardedInteraction()
    for frame in range(40):
        state=np.zeros(116)
        state[list(LEFT_MCP_COLUMNS)]=.35
        interaction.observe_state(state)
        left=np.array([80+2*frame,170,140+2*frame,240],float)
        fruit=np.array([110+2*frame,230,125+2*frame,245],float)
        assert interaction.choose({1:np.array([440,200,500,270]),2:left},[fruit])==(1,)


def test_closed_hand_alone_is_not_transport_and_release_returns_to_right():
    interaction=GuardedInteraction()
    state=np.zeros(116);state[list(LEFT_MCP_COLUMNS)]=.82
    right=np.array([440,200,500,270])
    for frame in range(15):
        interaction.observe_state(state)
        assert interaction.choose({1:right,2:np.array([80,170,140,240])},[])==(1,)
    for frame in range(40):
        interaction.observe_state(state)
        left=np.array([80+2*frame,170,140+2*frame,240],float)
        fruit=np.array([110+2*frame,230,125+2*frame,245],float)
        roles=interaction.choose({1:right,2:left},[fruit])
    assert roles==(2,)
    state[list(LEFT_MCP_COLUMNS)]=.3
    for _ in range(3):
        interaction.observe_state(state)
        roles=interaction.choose({1:right,2:left},[fruit])
    assert roles==(1,) and interaction.carried is None


def test_audit_rejects_left_only_when_one_required_evidence_is_missing():
    row=dict(source_frame=0,timestamp=0,available_at=0,source_size=[640,400],valid=True,roles=[2],
             crop_box=[100,100,320,320],display_crop_box=[100,100,320,320],interaction_box=[150,150,200,200],
             left_grasp_state_confirmed=False,carry_evidence=True,reason=None)
    with pytest.raises(ValueError,match='grasp and visual'):
        check_index([row],1)
    row['left_grasp_state_confirmed']=True
    check_index([row],1)


def test_current_grasp_and_foreground_fruit_survive_hand_box_pose_jitter():
    interaction=GuardedInteraction()
    state=np.zeros(116);state[list(LEFT_MCP_COLUMNS)]=.82
    for frame in range(40):
        interaction.observe_state(state)
        jitter=6 if frame%2 else -6
        left=np.array([80+2*frame+jitter,170,140+2*frame+jitter,240],float)
        fruit=np.array([110+2*frame,230,125+2*frame,245],float)
        roles=interaction.choose({1:np.array([440,200,500,270]),2:left},[fruit])
    assert roles==(2,) and interaction.carried is not None


def test_arrival_at_past_static_basket_does_not_remain_transport_with_closed_empty_hand():
    interaction=GuardedInteraction()
    state=np.zeros(116);state[list(LEFT_MCP_COLUMNS)]=.82
    right=np.array([500,200,560,270])
    basket=np.array([70,238,90,260],float)
    transported=False
    for frame in range(120):
        interaction.observe_state(state)
        x=205-2*min(frame,80)
        left=np.array([x,170,x+60,240],float)
        fruit=np.array([x+30,230,x+45,245],float)
        roles=interaction.choose({1:right,2:left},[basket,fruit] if frame<80 else [basket])
        transported|=roles==(2,)
    assert transported and roles==(1,)
