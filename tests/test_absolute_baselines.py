"""All published baselines supervise source absolute targets, independent of state."""

from pathlib import Path

import numpy as np
import pytest

from openpi import transforms
from scripts.run_training import load_experiment

MODULES = [
    str(path.with_suffix("")).replace("/", ".")
    for path in sorted(Path("training_configs/experiments").glob("*/baseline/*.py"))
    if path.name.startswith(("tomato_", "cup_", "lemon_"))
]


@pytest.mark.parametrize("module", MODULES)
def test_absolute_targets_are_preserved_for_statistics_and_training(module):
    config = load_experiment(module)
    dim = config.policy_metadata["physical_action_dim"]
    state_dim = config.policy_metadata["raw_state_dims"][0]
    actions = np.arange(3 * dim, dtype=np.float32).reshape(3, dim) / 100
    state = np.arange(state_dim, dtype=np.float32) + 20
    raw = {
        "action": actions.copy(),
        "observation.state": state,
        "prompt": "Perform the task.",
        **{key: np.zeros((8, 8, 3), dtype=np.uint8) for key in config.policy_metadata["camera_keys"]},
    }
    repack, vectors = config.data.create_vector_transforms(config.model)
    assert config.policy_metadata["training_action_semantics"].startswith("absolute_joint_target")
    assert not any(isinstance(t, transforms.DeltaActions) for t in vectors.inputs)
    assert not any(isinstance(t, transforms.AbsoluteActions) for t in vectors.outputs)
    apply = transforms.compose([*repack.inputs, *vectors.inputs])
    expected = actions
    if "tomato_wuji" in module:
        expected = np.concatenate([actions[:, :7], actions[:, 14:34], actions[:, 7:14], actions[:, 34:54]], axis=-1)
    if "tomato_sharpa" in module:
        expected = np.concatenate([actions[:, :7], actions[:, 14:36], actions[:, 7:14], actions[:, 36:58]], axis=-1)
    np.testing.assert_array_equal(apply(raw)["actions"], expected)
    np.testing.assert_array_equal(apply({**raw, "observation.state": state + 100})["actions"], expected)
    np.testing.assert_array_equal(raw["action"], actions)
    if "tomato_" in module:
        assert "absolute" in config.data.assets.asset_id
        assert "absolute" in config.exp_name
        assert not config.policy_metadata["allow_legacy_norm_stats"]
        assert config.policy_metadata["expected_norm_audit_sha256"] is None
