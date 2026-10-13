"""Lemon protocol, train/val isolation, phase supervision and normalization bindings."""

from dataclasses import replace
import json
from pathlib import Path

from flax import nnx
import jax
import numpy as np
import pytest

from examples.fr3_wuji import compute_norm_stats
from openpi import transforms
from openpi.shared import normalize
from openpi.training import config as cfg
from openpi.training import data_loader
from openpi.training import reproducibility
from openpi.training.task_view import FIELDS
from openpi.training.task_view import TaskView
from openpi.training.task_view import TaskViewDataset
from scripts import run_training
from scripts.train import should_save_checkpoint
from training_configs import common
from training_configs.experiments.lemon_scooping.baseline.common import PHASE_PROMPTS

BASE = "training_configs.experiments.lemon_scooping.baseline."


@pytest.mark.parametrize("method", ["lora", "fullft"])
def test_recorded_parameters_and_real_parameter_scope(method):
    config = run_training.load_experiment(BASE + f"lemon_ur_o30_{method}_v1")
    run_training.validate_config(config)
    assert config.model.action_dim == 32
    assert config.policy_metadata["normalization_dims"] == {"state": 52, "actions": 12}
    assert config.data.action_hz == 10
    assert config.num_train_steps == 30000
    assert config.batch_size == (8 if method == "lora" else 64)
    assert config.lr_schedule.peak_lr == (2.5e-5 if method == "lora" else 1e-5)
    params = jax.eval_shape(lambda: nnx.state(config.model.create(jax.random.key(42)), nnx.Param))
    trainable = params.filter(config.trainable_filter).flat_state()
    names = [str(path) for path in trainable]
    assert any("img" in path for path in names)
    assert any("llm" in path for path in names)
    assert any("lora" in path for path in names) == (method == "lora")
    if method == "fullft":
        assert not params.filter(config.freeze_filter).flat_state()
        assert not should_save_checkpoint(config, 999, 0)
        assert should_save_checkpoint(config, 4999, 0)
        assert should_save_checkpoint(config, 19999, 0)
        assert should_save_checkpoint(config, 20999, 0)
        assert should_save_checkpoint(config, 29999, 0)
        assert config.keep_period == 1000
    else:
        assert not hasattr(config, "evaluation")


@pytest.fixture
def dataset_and_config(tmp_path, monkeypatch):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    config = run_training.load_experiment(BASE + "lemon_ur_o30_lora_v1")
    root = Path(config.data.lerobot_datasets[0].root)
    features = {
        "observation.state": {"dtype": "float32", "shape": (52,), "names": [f"s{i}" for i in range(52)]},
        "action": {"dtype": "float32", "shape": (12,), "names": [f"a{i}" for i in range(12)]},
        **{
            f"observation.images.cam{i}": {
                "dtype": "video",
                "shape": (64, 64, 3),
                "names": ["height", "width", "channels"],
            }
            for i in range(3)
        },
    }
    ds = LeRobotDataset.create(
        config.data.repo_id, root=root, fps=10, robot_type="dual_ur_o30", features=features, use_videos=True
    )
    for episode in range(3):
        for frame in range(6):
            ds.add_frame(
                {
                    "observation.state": np.arange(52, dtype=np.float32) + episode * 100 + frame,
                    "action": np.arange(12, dtype=np.float32) + episode * 1000 + frame,
                    "task": "Scoop the lemon." if frame < 3 else "Pour into the cup.",
                    **{f"observation.images.cam{i}": np.full((64, 64, 3), i * 80, np.uint8) for i in range(3)},
                }
            )
        ds.save_episode()
    ds.stop_image_writer()
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"splits": {"train": [0, 2], "val": [1]}}))
    view = TaskView(split_path=str(splits))
    config = replace(
        config,
        model=replace(config.model, action_horizon=3),
        data=replace(config.data, base_config=cfg.TaskDataConfig(prompt_from_task=True, task_view=view)),
    )
    monkeypatch.setattr(
        cfg.ModelTransformFactory,
        "__call__",
        lambda self, model: transforms.Group(inputs=[transforms.PadStatesAndActions(model.action_dim)]),
    )
    return ds, config, tmp_path


def test_selected_norm_and_training_and_resume_audit(dataset_and_config):
    ds, config, tmp_path = dataset_and_config
    compute_norm_stats.compute(config, batch_size=4)
    report = run_training.preflight(config)
    assert report["training_frames"] == 12
    assert report["training_episodes"] == 2
    norm_dir = Path(config.data.assets.assets_dir) / config.data.assets.asset_id
    stats = normalize.load(norm_dir)
    assert stats["state"].mean.shape == (52,)
    assert stats["actions"].mean.shape == (12,)
    np.testing.assert_allclose(stats["state"].mean, np.arange(52) + 102.5)
    data = config.data.create(config.assets_dirs, config.model)
    view = data_loader.create_torch_dataset(data, 3, config.model, sampling_mode="epoch_shuffle")
    assert isinstance(view, TaskViewDataset)
    assert list(view.ids[:, 0]) == [0] * 6 + [2] * 6
    np.testing.assert_array_equal(view[5]["action"], np.repeat(view.vectors(5)["action"][:1], 3, axis=0))
    prepared = data_loader.transform_dataset(view, data)[0]
    assert prepared["state"].shape == (52,)  # Full state was tokenized before model padding.
    assert prepared["actions"].shape == (3, 32)
    assert np.all(prepared["actions"][:, 12:] == 0)
    np.testing.assert_allclose(prepared["image"]["right_wrist_0_rgb"].mean(), 160, atol=3)
    identity, ids = reproducibility._source_identity(data, view)  # noqa: SLF001 -- exercise resume data identity
    assert identity["frames"] == 12
    assert len(ids) == 12
    assert identity["task_view"]["split"] == "train"
    old = identity["task_view"]
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"splits": {"train": [0], "val": [1, 2]}}))
    with pytest.raises(ValueError, match="transforms differ"):
        run_training.preflight(config)
    assert config.data.base_config.task_view.identity() != old


def test_phase_boundary_arm_hold_and_prompt_overrides(dataset_and_config):
    ds, config, tmp_path = dataset_and_config
    document = {
        "schema_version": 3,
        "boundary_fields": list(FIELDS),
        "episodes": [
            {
                "episode_index": ep,
                "status": "complete",
                "frame_count": 6,
                "frame_indices": dict(zip(FIELDS, range(1, 6), strict=True)),
                **dict(zip(FIELDS, np.arange(1, 6) / 10, strict=True)),
            }
            for ep in (0, 2)
        ],
    }
    path = tmp_path / "phases.json"
    path.write_text(json.dumps(document))
    view = replace(
        config.data.base_config.task_view,
        phase_annotations_path=str(path),
        phase_prompts=PHASE_PROMPTS,
        phase_arm_hold=True,
    )
    source = TaskViewDataset(ds, view, 4)
    assert len(source) == 10
    first = source.vectors(0)
    assert first["prompt"] == PHASE_PROMPTS[0]
    np.testing.assert_array_equal(first["action"][:, 6:], np.repeat(first["observation.state"][None, 26:32], 4, axis=0))
    np.testing.assert_array_equal(first["action"][1:, :6], np.repeat(first["action"][1:2, :6], 3, axis=0))
    second = source.vectors(1)
    assert second["prompt"] == PHASE_PROMPTS[1]
    np.testing.assert_array_equal(second["action"][:, :6], np.repeat(second["observation.state"][None, :6], 4, axis=0))
    subset = TaskViewDataset(
        ds, TaskView(task_indices=(1,), prompt_overrides=((1, "Pour the lemon into the cup."),)), 3
    )
    assert len(subset) == 9
    assert subset[0]["prompt"] == "Pour the lemon into the cup."
    assert set(subset.ids[:, 2]) == {1}
    held = TaskViewDataset(ds, TaskView(task_indices=(0,), task_boundary_hold=True), 4)
    np.testing.assert_array_equal(held.vectors(2)["action"], np.repeat(held.vectors(2)["action"][:1], 4, axis=0))
    phase_config = replace(
        config, data=replace(config.data, base_config=cfg.TaskDataConfig(prompt_from_task=True, task_view=view))
    )
    compute_norm_stats.compute(phase_config, batch_size=3)
    assert run_training.preflight(phase_config)["training_frames"] == 10
    document["episodes"][0]["frame_indices"][FIELDS[1]] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="timeline"):
        TaskViewDataset(ds, view, 4)


def test_invalid_task_views_and_split_leakage(dataset_and_config):
    ds, config, tmp_path = dataset_and_config
    with pytest.raises(ValueError, match="English"):
        TaskView(prompt_overrides=((0, "铲柠檬"),))
    with pytest.raises(ValueError, match="absent"):
        TaskViewDataset(ds, TaskView(task_indices=(12,)), 3)
    (tmp_path / "splits.json").write_text(json.dumps({"splits": {"train": [0, 1], "val": [1]}}))
    with pytest.raises(ValueError, match="overlap"):
        TaskViewDataset(ds, config.data.base_config.task_view, 3)
