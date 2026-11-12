import numpy as np

from scripts.tomato_transport_roi import TransportInteraction, LEFT_MCP_COLUMNS


def test_new_stationary_basket_fruit_does_not_initiate_possession():
    interaction = TransportInteraction()
    state = np.zeros(116)
    state[list(LEFT_MCP_COLUMNS)] = .82
    hands = {1:np.array([450.,200.,510.,270.]),2:np.array([70.,200.,140.,270.])}
    fruit = np.array([100.,258.,116.,276.])
    for frame in range(70):
        interaction.observe_state(state)
        roles = interaction.choose(hands,[] if frame<10 else [fruit])
        assert roles==(1,)
        assert interaction.carried is None


def test_true_hand_fruit_motion_with_pose_box_jitter_still_confirms_transport():
    interaction = TransportInteraction()
    state = np.zeros(116)
    state[list(LEFT_MCP_COLUMNS)] = .82
    for frame in range(40):
        interaction.observe_state(state)
        jitter = 6 if frame%2 else -6
        left = np.array([80+2*frame+jitter,170,140+2*frame+jitter,240],float)
        fruit = np.array([110+2*frame,230,125+2*frame,245],float)
        roles = interaction.choose({1:np.array([440.,200.,500.,270.]),2:left},[fruit])
    assert roles==(2,) and interaction.carried is not None
    state[list(LEFT_MCP_COLUMNS)] = .3
    for _ in range(3):
        interaction.observe_state(state)
        roles = interaction.choose({1:np.array([440.,200.,500.,270.]),2:left},[fruit])
    assert roles==(1,) and interaction.carried is None
