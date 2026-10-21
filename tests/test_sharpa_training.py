"""Protect physical Sharpa ordering and recipe equivalence without robot hardware."""

from dataclasses import replace
import functools
import json
from pathlib import Path

import numpy as np
import pytest

from examples.fr3_wuji import compute_norm_stats
from openpi import transforms
from openpi.models import model
from openpi.policies import fr3_sharpa_policy
from openpi.policies import fr3_sharpa_protocol as protocol
from openpi.policies.fr3_sharpa_multimodal_policy import SOURCE_ACTION_NAMES
from openpi.policies.fr3_sharpa_multimodal_policy import SOURCE_STATE_NAMES
from openpi.shared import normalize
from openpi.training import config as cfg
from openpi.training import data_loader
from openpi.training.reproducibility import serialize_config
from scripts import run_training
from training_configs import common
from training_configs.embodiments.fr3_sharpa import Fr3Sharpa

BASE = "training_configs.experiments.tomato_sharpa.baseline."


@pytest.mark.parametrize("scheme", ["lora", "fullft"])
def test_sharpa_changes_only_adapter_and_experiment_identity(scheme):
    wuji = run_training.load_experiment(f"training_configs.experiments.tomato_wuji.baseline.tomato_fr3_{scheme}_v1")
    sharpa = run_training.load_experiment(BASE + f"tomato_fr3_sharpa_{scheme}_v1")
    run_training.validate_config(sharpa)
    assert sharpa.num_train_steps == wuji.num_train_steps == 30_000
    wuji_fields = json.loads(serialize_config(wuji))["fields"]
    sharpa_fields = json.loads(serialize_config(sharpa))["fields"]
    for key in ("name", "exp_name", "data", "policy_metadata"):
        wuji_fields.pop(key)
        sharpa_fields.pop(key)
    assert sharpa_fields["model"]["fields"]["action_dim"] == 58
    sharpa_fields["model"]["fields"]["action_dim"] = 54
    if scheme == "fullft":
        assert sharpa_fields.pop("batch_size") == 32
        assert sharpa_fields.pop("fsdp_devices") == 8
        wuji_fields.pop("batch_size")
        wuji_fields.pop("fsdp_devices")
    assert sharpa_fields == wuji_fields
    wuji_data = json.loads(serialize_config(wuji.data))["fields"]
    sharpa_data = json.loads(serialize_config(sharpa.data))["fields"]
    for key in ("repo_id", "assets", "lerobot_datasets"):
        wuji_data.pop(key)
        sharpa_data.pop(key)
    assert sharpa_data == wuji_data
    assert sharpa.policy_metadata["raw_state_dims"] == [116]
    assert sharpa.policy_metadata["dataset_version"] == "tomato_sharpa_1016_2_multimodal_rgb_absolute_v1_30hz"
    assert sharpa.policy_metadata["prompt_example"].isascii()
    assert sharpa.policy_metadata["joint_names"] == list(protocol.ORDER)
    assert sharpa.policy_metadata["joint_unit"] == "rad"
    assert not sharpa.policy_metadata["allow_legacy_norm_stats"]


@pytest.mark.parametrize("scheme", ["lora", "fullft"])
def test_copied_sharpa_baselines_keep_recipe_and_independent_paths(tmp_path, scheme):
    from scripts import training_ui_config as ui

    module = BASE + f"tomato_fr3_sharpa_{scheme}_v1"
    baseline = run_training.load_experiment(module)
    values = {
        **ui.defaults(module)["values"],
        "dataset_root": str(tmp_path / "personal_dataset"),
        "assets_dir": str(tmp_path / "personal_assets"),
        "exp_name": "alice_trial_s42_run01",
    }
    source = ui.personal_source(module, "alice", values)
    template = ui.config_from_source(source, "personal_experiment", str(tmp_path / "personal.py"))
    run_training.validate_config(template)
    assert template.data.lerobot_datasets[0].root == values["dataset_root"]
    assert template.data.assets.assets_dir == values["assets_dir"]
    assert template.exp_name != baseline.exp_name
    assert template.checkpoint_dir != baseline.checkpoint_dir
    assert template.lr_schedule == baseline.lr_schedule
    assert template.num_train_steps == baseline.num_train_steps
    assert template.policy_metadata["ui_owner"] == "alice"
    assert run_training.load_experiment(module).data == baseline.data


def test_joint_blocks_cameras_and_absolute_actions():
    assert len(protocol.SHARPA_JOINTS) == 22
    assert len(set(protocol.ORDER)) == protocol.ACTION_DIM == protocol.STATE_DIM == 58
    assert protocol.ORDER[0] == "left_fr3_joint1"
    assert protocol.ORDER[7] == "left_thumb_CMC_FE"
    assert protocol.ORDER[29] == "right_fr3_joint1"
    assert protocol.ORDER[36] == "right_thumb_CMC_FE"
    config = run_training.load_experiment(BASE + "tomato_fr3_sharpa_lora_v1")
    repack, vectors = config.data.create_vector_transforms(config.model)
    state = np.arange(116, dtype=np.float32) + 100
    actions = np.arange(2 * 58, dtype=np.float32).reshape(2, 58)
    raw = {
        "observation.state": state,
        "action": actions,
        "prompt": "Pick the tomato.",
        "observation.images.cam0": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation.images.cam1": np.full((4, 5, 3), 11, dtype=np.uint8),
        "observation.images.cam2": np.full((3, 4, 5), 0.8, dtype=np.float32),
    }
    result = transforms.compose([*repack.inputs, *vectors.inputs])(raw)
    state_indices = [SOURCE_STATE_NAMES.index(f"{name}.position") for name in protocol.ORDER]
    action_indices = [
        {name.rsplit(".", 1)[0]: i for i, name in enumerate(SOURCE_ACTION_NAMES)}[name] for name in protocol.ORDER
    ]
    np.testing.assert_array_equal(result["state"], state[state_indices])
    expected = actions[:, action_indices]
    np.testing.assert_array_equal(result["actions"], expected)
    assert np.all(result["image"]["left_wrist_0_rgb"] == 204)
    assert result["image"]["left_wrist_0_rgb"].shape == (4, 5, 3)
    assert np.all(result["image"]["right_wrist_0_rgb"] == 11)
    assert all(result["image_mask"].values())
    assert result["prompt"] == raw["prompt"]
    restored = transforms.compose(vectors.outputs)(result)
    np.testing.assert_array_equal(restored["actions"], expected)
    np.testing.assert_array_equal(raw["action"], actions)


@pytest.mark.parametrize("field", ["observation/state", "actions"])
@pytest.mark.parametrize("dim", [54, 108, 116])
def test_sharpa_rejects_other_embodiment_vectors(field, dim):
    raw = {
        "observation/state": np.zeros(58),
        "actions": np.zeros((2, 58)),
        **{
            key: np.zeros((4, 5, 3))
            for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image")
        },
    }
    raw[field] = np.zeros(dim)
    with pytest.raises(ValueError, match="58D"):
        fr3_sharpa_policy.Fr3SharpaInputs(model.ModelType.PI05)(raw)


@pytest.mark.parametrize("change", [{"action_dim": 54}, {"raw_state_dims": (58, 116)}])
def test_sharpa_contract_requires_new_adapter_for_dimension_changes(change):
    with pytest.raises(ValueError, match="fixed measured 58D"):
        Fr3Sharpa(**change)


@pytest.mark.parametrize("field", ["action", "observation.state"])
def test_preflight_rejects_same_dimension_wrong_joint_order(tmp_path, monkeypatch, field):
    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    monkeypatch.setenv("PI05_SHARPA_MULTIMODAL_DATASET_ROOT", str(tmp_path / "dataset"))
    config = run_training.load_experiment(BASE + "tomato_fr3_sharpa_lora_v1")
    root = Path(config.data.lerobot_datasets[0].root)
    for folder in ("meta", "data", "videos"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_frames": 4,
        "total_episodes": 1,
        "features": {
            "action": {"shape": [58], "names": list(SOURCE_ACTION_NAMES)},
            "observation.state": {"shape": [116], "names": list(SOURCE_STATE_NAMES)},
            **{key: {"shape": [4, 5, 3]} for key in config.policy_metadata["camera_keys"]},
        },
    }
    path = root / "meta/info.json"
    path.write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 4}\n')
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Pick the tomato."}\n')
    assert run_training.preflight(config, require_norm=False)["frames"] == 4
    names = info["features"][field]["names"]
    names[0], names[29] = names[29], names[0]
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="joint names/order differ"):
        run_training.preflight(config, require_norm=False)
    info["features"][field].pop("names")
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="joint names/order differ"):
        run_training.preflight(config, require_norm=False)


@pytest.mark.parametrize("sampling_mode", ["epoch_shuffle", "weighted_replacement"])
def test_real_sharpa_lerobot_videos_windows_and_shared_statistics(tmp_path, monkeypatch, sampling_mode):
    from lerobot.common.datasets import lerobot_dataset

    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    monkeypatch.setenv("PI05_SHARPA_MULTIMODAL_DATASET_ROOT", str(tmp_path / "dataset"))
    config = run_training.load_experiment(BASE + "tomato_fr3_sharpa_lora_v1")
    config = replace(config, model=replace(config.model, action_horizon=3))
    from openpi.training.sampling import SamplingGroup

    if sampling_mode == "weighted_replacement":
        config = replace(
            config,
            sampling_mode=sampling_mode,
            sampling_groups=(SamplingGroup("A", (0,), 7), SamplingGroup("B", (1,), 3)),
        )
    root = Path(config.data.lerobot_datasets[0].root)
    features = {
        "observation.state": {"dtype": "float32", "shape": (116,), "names": list(SOURCE_STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (58,), "names": list(SOURCE_ACTION_NAMES)},
        **{
            key: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channel"]}
            for key in config.policy_metadata["camera_keys"]
        },
    }
    # Real encoder/decoder, using fast H.264 instead of the recorder's AV1 default.
    monkeypatch.setattr(
        lerobot_dataset,
        "encode_video_frames",
        functools.partial(lerobot_dataset.encode_video_frames, vcodec="h264"),
    )
    dataset = lerobot_dataset.LeRobotDataset.create(
        config.data.repo_id, fps=30, root=root, features=features, video_backend="pyav"
    )
    targets = []
    for episode, length in enumerate((3, 2)):
        for row in range(length):
            state = np.arange(116, dtype=np.float32) / 100 + row / 10 + episode
            action = np.arange(58, dtype=np.float32) / 100 + row / 10 + episode + 0.2
            targets.append(action.copy())
            dataset.add_frame(
                {
                    "observation.state": state,
                    "action": action,
                    "task": "Pick the tomato.",
                    **{
                        key: np.full((64, 64, 3), color, dtype=np.uint8)
                        for key, color in zip(config.policy_metadata["camera_keys"], (0, 40, 190), strict=True)
                    },
                }
            )
        dataset.save_episode()
    dataset.stop_image_writer()
    assert len(list(root.rglob("*.mp4"))) == 6
    monkeypatch.setattr(cfg.ModelTransformFactory, "__call__", lambda *args: transforms.Group())
    data_config = config.data.create(config.assets_dirs, config.model)
    reference = data_loader.create_torch_dataset(data_config, 3, config.model, sampling_mode="epoch_shuffle")
    assert len(reference) == 5
    # Clamped within the first episode rather than reaching the second episode's targets.
    np.testing.assert_array_equal(reference[2]["action"], np.repeat(targets[2][None, :], 3, axis=0))
    np.testing.assert_array_equal(reference[3]["action"], np.stack([targets[3], targets[4], targets[4]]))
    compute_norm_stats.compute(config, batch_size=2)
    assert run_training.preflight(config)["norm_provenance"] == "verified"
    if sampling_mode == "weighted_replacement":
        check = run_training.preflight(config)
        assert [group["probability"] for group in check["sampling"]["groups"]] == pytest.approx([0.7, 0.3])
        weighted_config = replace(config, num_workers=0, num_train_steps=2000)
        loader = data_loader.create_data_loader(weighted_config)
        summary = loader.prepare_reproducibility(weighted_config, resuming=False)
        total = sum(summary["group_samples"].values())
        assert summary["group_samples"]["A"] / total == pytest.approx(0.7, abs=0.02)
    fullft = run_training.load_experiment(BASE + "tomato_fr3_sharpa_fullft_v1")
    fullft = replace(fullft, model=replace(fullft.model, action_horizon=3))
    assert run_training.preflight(fullft)["norm_provenance"] == "verified"
    # Statistics from a different action transform cannot be silently reused.
    with pytest.raises(ValueError, match="vector transforms differ"):
        run_training.preflight(replace(config, data=replace(config.data, extra_delta_transform=True)))
    directory = Path(config.data.assets.assets_dir) / config.data.assets.asset_id
    stats = normalize.load(directory)
    assert stats["state"].mean.shape == stats["actions"].mean.shape == (58,)
    repack, vectors = config.data.create_vector_transforms(config.model)
    sample = transforms.compose([*repack.inputs, *vectors.inputs])(reference[0])
    np.testing.assert_allclose(sample["image"]["left_wrist_0_rgb"].mean(), 190, atol=3)
    np.testing.assert_allclose(sample["image"]["right_wrist_0_rgb"].mean(), 40, atol=3)
    np.testing.assert_allclose(
        sample["actions"][0],
        targets[0][list(range(7)) + list(range(14, 36)) + list(range(7, 14)) + list(range(36, 58))],
        atol=1e-6,
    )
