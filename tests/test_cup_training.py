"""Verify single-arm dimensions, camera masks, normalization and shared recipe parameters."""

from dataclasses import replace
import functools
import json
from pathlib import Path

import numpy as np
import pytest

from examples.fr3_wuji import compute_norm_stats
from openpi import transforms
from openpi.policies import single_franka_policy as policy
from openpi.shared import normalize
from openpi.training import config as cfg
from openpi.training import data_loader
from openpi.training.reproducibility import serialize_config
from scripts import run_training
from training_configs import common
from training_configs.embodiments.single_franka import SingleFranka

BASE = "training_configs.experiments.cup_grasping.baseline."


@pytest.mark.parametrize("scheme", ["lora", "fullft"])
def test_cup_changes_only_adapter_and_experiment_identity(scheme):
    reference = run_training.load_experiment(
        f"training_configs.experiments.tomato_wuji.baseline.tomato_fr3_{scheme}_v1"
    )
    cup = run_training.load_experiment(BASE + f"cup_single_franka_{scheme}_v1")
    run_training.validate_config(cup)
    assert cup.model.action_dim == 32
    assert cup.policy_metadata["physical_action_dim"] == cup.policy_metadata["normalization_dim"] == 7
    assert cup.policy_metadata["raw_state_dims"] == [7]
    assert cup.num_train_steps == 30_000
    expected = json.loads(serialize_config(reference))["fields"]
    actual = json.loads(serialize_config(cup))["fields"]
    for key in ("name", "exp_name", "data", "policy_metadata"):
        expected.pop(key)
        actual.pop(key)
    actual["model"]["fields"]["action_dim"] = expected["model"]["fields"]["action_dim"]
    assert actual == expected


def test_physical_vectors_remain_absolute_and_pad_only_after_normalization():
    state = np.arange(7, dtype=np.float32)
    action = np.arange(21, dtype=np.float32).reshape(3, 7)
    result = policy.SingleFrankaInputs()(
        {
            "observation/state": state,
            "actions": action,
            "observation/image": np.zeros((8, 9, 3), dtype=np.uint8),
            "observation/wrist_image": np.ones((3, 8, 9), dtype=np.float32),
            "prompt": "Pick up the cup.",
        }
    )
    np.testing.assert_array_equal(result["state"], state)
    np.testing.assert_array_equal(result["actions"], action)
    assert result["image"]["left_wrist_0_rgb"].shape == (8, 9, 3)
    assert np.all(result["image"]["left_wrist_0_rgb"] == 255)
    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert np.all(result["image"]["right_wrist_0_rgb"] == 0)
    stats = {
        key: normalize.NormStats(mean=np.zeros(7), std=np.ones(7), q01=-np.ones(7), q99=np.ones(7))
        for key in ("state", "actions")
    }
    normalized = transforms.Normalize(stats, use_quantiles=True)(result)
    unpadded_actions = normalized["actions"].copy()
    padded = transforms.PadStatesAndActions(32)(normalized)
    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (3, 32)
    assert np.all(padded["state"][7:] == 0)
    assert np.all(padded["actions"][..., 7:] == 0)
    np.testing.assert_allclose(policy.SingleFrankaOutputs()(padded)["actions"], unpadded_actions, rtol=1e-6)
    np.testing.assert_array_equal(action, np.arange(21).reshape(3, 7))


@pytest.mark.parametrize("field", ["observation/state", "actions"])
@pytest.mark.parametrize("value", [np.zeros(8), np.zeros(32), np.full(7, np.nan)])
def test_invalid_physical_vectors_are_rejected(field, value):
    raw = {
        "observation/state": np.zeros(7),
        "actions": np.zeros((3, 7)),
        "observation/image": np.zeros((8, 9, 3)),
        "observation/wrist_image": np.zeros((8, 9, 3)),
    }
    raw[field] = value
    with pytest.raises(ValueError, match="7D|NaN"):
        policy.SingleFrankaInputs()(raw)


def test_dimension_contract_rejects_model_and_physical_overrides():
    with pytest.raises(ValueError, match="physical 7D"):
        SingleFranka(action_dim=8)
    config = run_training.load_experiment(BASE + "cup_single_franka_lora_v1")
    with pytest.raises(ValueError, match="Model action_dim"):
        run_training.validate_config(replace(config, model=replace(config.model, action_dim=7)))


def test_real_cup_lerobot_statistics_use_seven_joints_and_share_between_recipes(tmp_path, monkeypatch):
    from lerobot.common.datasets import lerobot_dataset

    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    config = run_training.load_experiment(BASE + "cup_single_franka_lora_v1")
    config = replace(config, model=replace(config.model, action_horizon=3))
    root = Path(config.data.lerobot_datasets[0].root)
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": [f"joint{i}" for i in range(7)]},
        "action": {"dtype": "float32", "shape": (7,), "names": [f"joint{i}" for i in range(7)]},
        **{
            key: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channel"]}
            for key in config.policy_metadata["camera_keys"]
        },
    }
    monkeypatch.setattr(
        lerobot_dataset, "encode_video_frames", functools.partial(lerobot_dataset.encode_video_frames, vcodec="h264")
    )
    dataset = lerobot_dataset.LeRobotDataset.create(
        config.data.repo_id, fps=30, root=root, features=features, video_backend="pyav"
    )
    targets = []
    for episode, length in enumerate((3, 2)):
        for row in range(length):
            state = np.arange(7, dtype=np.float32) / 10 + episode + row
            action = state + 0.5
            targets.append(action.copy())
            dataset.add_frame(
                {
                    "observation.state": state,
                    "action": action,
                    "task": "Pick up the cup.",
                    **{
                        key: np.full((64, 64, 3), color, dtype=np.uint8)
                        for key, color in zip(config.policy_metadata["camera_keys"], (0, 190), strict=True)
                    },
                }
            )
        dataset.save_episode()
    dataset.stop_image_writer()
    assert len(list(root.rglob("*.mp4"))) == 4
    # Keep the real post-normalization padding but avoid downloading a tokenizer.
    monkeypatch.setattr(
        cfg.ModelTransformFactory,
        "__call__",
        lambda self, model: transforms.Group(inputs=[transforms.PadStatesAndActions(model.action_dim)]),
    )
    compute_norm_stats.compute(config, batch_size=2)
    assert run_training.preflight(config)["norm_provenance"] == "verified"
    fullft = run_training.load_experiment(BASE + "cup_single_franka_fullft_v1")
    fullft = replace(fullft, model=replace(fullft.model, action_horizon=3))
    assert run_training.preflight(fullft)["norm_provenance"] == "verified"
    directory = Path(config.data.assets.assets_dir) / config.data.assets.asset_id
    stats = normalize.load(directory)
    assert stats["state"].mean.shape == stats["actions"].mean.shape == (7,)
    np.testing.assert_allclose(stats["state"].mean, np.mean(np.asarray(targets) - 0.5, axis=0), atol=1e-6)
    factory = config.data.create(config.assets_dirs, config.model)
    reference = data_loader.create_torch_dataset(factory, 3, config.model, sampling_mode="epoch_shuffle")
    np.testing.assert_array_equal(reference[2]["action"], np.repeat(targets[2][None, :], 3, axis=0))
    prepared = data_loader.transform_dataset(reference, factory)[0]
    assert prepared["state"].shape == (32,)
    assert prepared["actions"].shape == (3, 32)
    assert np.all(prepared["actions"][..., 7:] == 0)
    assert not prepared["image_mask"]["right_wrist_0_rgb"]
    np.testing.assert_allclose(prepared["image"]["left_wrist_0_rgb"].mean(), 190, atol=3)
    np.testing.assert_allclose(
        prepared["actions"][0, :7],
        transforms.Normalize(stats, use_quantiles=True)({"actions": targets[0]})["actions"],
        atol=1e-6,
    )
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["action"]["shape"] = [32]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="Dataset action shape"):
        run_training.preflight(config, require_norm=False)
    info["features"]["action"]["shape"] = [7]
    info["codebase_version"] = "v3.0"
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="v2.1"):
        run_training.preflight(config, require_norm=False)
