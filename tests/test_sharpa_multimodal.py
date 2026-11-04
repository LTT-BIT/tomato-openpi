"""Converter-storage adaptation must not weaken the existing physical baseline."""

import json

import numpy as np
import pytest

from openpi import transforms
from openpi.models import model
from openpi.policies.fr3_sharpa_multimodal_policy import SOURCE_ACTION_NAMES
from openpi.policies.fr3_sharpa_multimodal_policy import SOURCE_STATE_NAMES
from openpi.policies.fr3_sharpa_multimodal_policy import Fr3SharpaMultimodalInputs
from scripts.run_training import load_experiment
from scripts.run_training import preflight

MODULE = "training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_lora_v1"


def test_converter_blocks_and_cameras_are_selected_without_changing_source():
    config = load_experiment(MODULE)
    repack, vectors = config.data.create_vector_transforms(config.model)
    state = np.arange(116, dtype=np.float32)
    action = np.arange(3 * 58, dtype=np.float32).reshape(3, 58)
    raw = {
        "observation.state": state,
        "action": action,
        "prompt": "Pick the tomato.",
        "observation.images.cam0": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation.images.cam1": np.full((4, 5, 3), 33, dtype=np.uint8),
        "observation.images.cam2": np.full((4, 5, 3), 77, dtype=np.uint8),
        "observation.depths.cam0": np.ones((4, 5), dtype=np.uint16),
        "observation.tactile.left.index.wrench": np.ones(6),
    }
    result = transforms.compose([*repack.inputs, *vectors.inputs])(raw)
    order = config.policy_metadata["joint_names"]
    state_indices = [SOURCE_STATE_NAMES.index(f"{name}.position") for name in order]
    action_indices = [{name.rsplit(".", 1)[0]: i for i, name in enumerate(SOURCE_ACTION_NAMES)}[name] for name in order]
    np.testing.assert_array_equal(result["state"], state[state_indices])
    np.testing.assert_array_equal(result["actions"], action[:, action_indices])
    assert result["image"]["left_wrist_0_rgb"][0, 0, 0] == 77
    assert result["image"]["right_wrist_0_rgb"][0, 0, 0] == 33
    assert set(result) == {"state", "actions", "image", "image_mask", "prompt"}
    result["state"][0] = -9
    result["actions"][0, 0] = -9
    assert state[0] == action[0, 0] == 0
    assert raw["observation.depths.cam0"].dtype == np.uint16


def test_only_official_sharpa_baselines_are_published_and_share_statistics():
    from scripts.training_ui_config import baseline_catalog

    lora = load_experiment(MODULE)
    fullft = load_experiment(MODULE.replace("lora", "fullft"))
    assert lora.data == fullft.data
    assert lora.policy_metadata["raw_state_dims"] == fullft.policy_metadata["raw_state_dims"] == [116]
    assert lora.checkpoint_dir != fullft.checkpoint_dir
    for config in (lora, fullft):
        assert config.policy_metadata["training_action_semantics"] == "absolute_joint_target"
        assert config.model.action_dim == 58
        assert config.model.action_horizon == 50
    assert (lora.batch_size, lora.fsdp_devices) == (8, 1)
    assert (fullft.batch_size, fullft.fsdp_devices) == (32, 8)
    entries = [entry["id"] for entry in baseline_catalog() if entry["task"] == "tomato_sharpa"]
    assert set(entries) == {MODULE, MODULE.replace("lora", "fullft")}


def test_preflight_checks_raw_names_and_rejects_wrong_order(tmp_path, monkeypatch):
    monkeypatch.setenv("PI05_SHARPA_MULTIMODAL_DATASET_ROOT", str(tmp_path))
    c = load_experiment(MODULE)
    for name in ("meta", "data", "videos"):
        (tmp_path / name).mkdir()
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_frames": 2,
        "total_episodes": 1,
        "features": {
            "observation.state": {"shape": [116], "names": list(SOURCE_STATE_NAMES)},
            "action": {"shape": [58], "names": list(SOURCE_ACTION_NAMES)},
            **{key: {} for key in c.policy_metadata["camera_keys"]},
        },
    }
    path = tmp_path / "meta/info.json"
    path.write_text(json.dumps(info))
    (tmp_path / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 2}\n')
    (tmp_path / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Pick the tomato."}\n')
    assert preflight(c, require_norm=False)["frames"] == 2
    info["features"]["action"]["names"].reverse()
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="joint names/order"):
        preflight(c, require_norm=False)


@pytest.mark.parametrize("width", [58, 108])
def test_multimodal_adapter_does_not_guess_other_state_layouts(width):
    with pytest.raises(ValueError, match="116D"):
        Fr3SharpaMultimodalInputs(model.ModelType.PI05)({"observation/state": np.zeros(width)})
