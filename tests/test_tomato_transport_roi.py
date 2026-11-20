import numpy as np

from scripts.tomato_transport_roi import LEFT_MCP_COLUMNS
from scripts.tomato_transport_roi import TransportInteraction


def test_new_stationary_basket_fruit_does_not_initiate_possession():
    interaction = TransportInteraction()
    state = np.zeros(116)
    state[list(LEFT_MCP_COLUMNS)] = 0.82
    hands = {1: np.array([450.0, 200.0, 510.0, 270.0]), 2: np.array([70.0, 200.0, 140.0, 270.0])}
    fruit = np.array([100.0, 258.0, 116.0, 276.0])
    for frame in range(70):
        interaction.observe_state(state)
        roles = interaction.choose(hands, [] if frame < 10 else [fruit])
        assert roles == (1,)
        assert interaction.carried is None


def test_true_hand_fruit_motion_with_pose_box_jitter_still_confirms_transport():
    interaction = TransportInteraction()
    state = np.zeros(116)
    state[list(LEFT_MCP_COLUMNS)] = 0.82
    for frame in range(40):
        interaction.observe_state(state)
        jitter = 6 if frame % 2 else -6
        left = np.array([80 + 2 * frame + jitter, 170, 140 + 2 * frame + jitter, 240], float)
        fruit = np.array([110 + 2 * frame, 230, 125 + 2 * frame, 245], float)
        roles = interaction.choose({1: np.array([440.0, 200.0, 500.0, 270.0]), 2: left}, [fruit])
    assert roles == (2,)
    assert interaction.carried is not None
    state[list(LEFT_MCP_COLUMNS)] = 0.3
    for _ in range(3):
        interaction.observe_state(state)
        roles = interaction.choose({1: np.array([440.0, 200.0, 500.0, 270.0]), 2: left}, [fruit])
    assert roles == (1,)
    assert interaction.carried is None
