"""Exercise real multi-root videos, action boundaries, stats, sampling and resume."""

from dataclasses import replace
import functools
import json
from pathlib import Path

import numpy as np
import pytest

from examples.fr3_wuji import compute_norm_stats
from openpi import transforms
from openpi.training import config as cfg
from openpi.training import data_loader
from openpi.training.dataset_collection import collection_layout
from openpi.training.sampling import SamplingGroup
from scripts import run_training
from scripts import training_ui_config as editor
from scripts.training_ui_datasets import dataset_groups
from scripts.training_ui_datasets import discover_datasets
from training_configs import common


def drop_prompt(data):
    return {key: value for key, value in data.items() if key != "prompt"}


BASE = "training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1"


def metadata(root, *, episode=0):
    for name in ("meta", "data", "videos"):
        (root / name).mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps({"codebase_version": "v2.1", "total_frames": 2, "total_episodes": 1, "fps": 30})
    )
    (root / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": episode, "length": 2}))
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Pick tomato."}')


def test_discovery_parents_deduplication_and_full_subtasks_choice(tmp_path):
    roots = [tmp_path / p for p in ("0924/A", "0924/B", "1010/full", "1010/subtasks")]
    for root in roots:
        metadata(root)
    result = discover_datasets([str(tmp_path / "0924"), str(tmp_path / "1010"), str(roots[0])])
    assert len(result["datasets"]) == 4
    selected = [row["root"] for row in result["datasets"] if row["selected"]]
    assert selected == list(map(str, roots[:3]))
    groups = dataset_groups(selected)
    assert [g["episodes"] for g in groups["groups"]] == [[0], [1], [2]]
    assert groups["frames"] == 6


def test_collection_ids_noncontiguous_and_symlink_duplicates(tmp_path):
    roots = [tmp_path / p for p in ("a", "b")]
    for root in roots:
        metadata(root, episode=42)
    assert [row["mapping"] for row in collection_layout(roots)] == [{42: 0}, {42: 1}]
    assert collection_layout([roots[0]])[0]["mapping"] == {42: 42}
    link = tmp_path / "alias"
    link.symlink_to(roots[0], target_is_directory=True)
    with pytest.raises(ValueError, match="unique"):
        collection_layout([roots[0], link])


def test_ui_collection_roundtrip_and_snapshot(tmp_path):
    values = editor.defaults(BASE)["values"]
    roots = [str(tmp_path / "a"), str(tmp_path / "b")]
    values.update(dataset_root=roots[0], dataset_roots=roots, sources=roots, exp_name="multi_test")
    config = editor.apply_overrides(editor.source_config(BASE), values)
    assert [s.root for s in config.data.lerobot_datasets] == roots
    assert editor.apply_overrides(config, editor.config_values(config)) == config
    scope = {}
    exec(editor.personal_source(BASE, "tester", values), scope)
    assert [s.root for s in scope["build_config"]().data.lerobot_datasets] == roots
    assert "dataset_roots" not in editor.changed_parameters(BASE, values)
    with pytest.raises(ValueError, match="重复"):
        editor.apply_overrides(config, {**values, "dataset_roots": [roots[0], roots[0]]})


@pytest.mark.parametrize("mode", ["epoch_shuffle", "weighted_replacement"])
def test_real_collection_windows_norm_audit_and_resume(tmp_path, monkeypatch, mode):
    from lerobot.common.datasets import lerobot_dataset

    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    config = run_training.load_experiment(BASE)
    specs = tuple(
        replace(config.data.lerobot_datasets[0], root=str(tmp_path / f"dataset{i}"), repo_id=f"test/part{i}")
        for i in range(2)
    )
    config = replace(
        config,
        data=replace(config.data, lerobot_datasets=specs),
        model=replace(config.model, action_horizon=3),
        batch_size=2,
        num_workers=0,
        num_train_steps=2000,
        sampling_mode=mode,
        sampling_groups=(SamplingGroup("A", (0,), 7), SamplingGroup("B", (1,), 3))
        if mode == "weighted_replacement"
        else (),
    )
    if mode == "weighted_replacement":
        config = replace(config, sampling_groups=(SamplingGroup("A", (0,), 7), SamplingGroup("B", (1,), 3)))
    monkeypatch.setattr(
        lerobot_dataset, "encode_video_frames", functools.partial(lerobot_dataset.encode_video_frames, vcodec="h264")
    )
    features = {
        "observation.state": {"dtype": "float32", "shape": (108,), "names": None},
        "action": {"dtype": "float32", "shape": (54,), "names": None},
        **{
            key: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channel"]}
            for key in config.policy_metadata["camera_keys"]
        },
    }
    for i, spec in enumerate(specs):
        ds = lerobot_dataset.LeRobotDataset.create(
            spec.repo_id, fps=30, root=Path(spec.root), features=features, video_backend="pyav"
        )
        for j in range(i + 2):
            ds.add_frame(
                {
                    "observation.state": np.full(108, i + j / 10, np.float32),
                    "action": np.full(54, i + j / 10, np.float32),
                    "task": "Pick tomato." if i == 0 else "Place tomato.",
                    **{key: np.zeros((64, 64, 3), np.uint8) for key in config.policy_metadata["camera_keys"]},
                }
            )
        ds.save_episode()
        ds.stop_image_writer()
    monkeypatch.setattr(cfg.ModelTransformFactory, "__call__", lambda *args: transforms.Group(inputs=(drop_prompt,)))
    report = run_training.preflight(config, require_norm=False)
    assert (report["episodes"], report["frames"]) == (2, 5)
    if mode == "weighted_replacement":
        assert [g["probability"] for g in report["sampling"]["groups"]] == pytest.approx([0.7, 0.3])
    data_config = config.data.create(config.assets_dirs, config.model)
    source = data_loader.create_torch_dataset(data_config, 3, config.model, sampling_mode=mode)
    assert source.frame_ids.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]
    np.testing.assert_allclose(source[1]["action"], 0.1)  # Does not cross into the next dataset.
    assert source[1]["prompt"] == "Pick tomato."
    assert source[2]["prompt"] == "Place tomato."
    compute_norm_stats.compute(config, batch_size=2)
    assert run_training.preflight(config)["norm_provenance"] == "verified"
    loader = data_loader.create_data_loader(config)
    summary = loader.prepare_reproducibility(config, resuming=False)
    if mode == "weighted_replacement":
        assert summary["group_samples"]["A"] / 4000 == pytest.approx(0.7, abs=0.025)
    else:
        assert summary["planned_coverage"] == 1
    assert loader.prepare_reproducibility(config, resuming=True) == summary
    iterator = iter(loader)
    next(iterator)
    expected = next(iterator)[1]
    loader.seek(1)
    np.testing.assert_array_equal(next(iter(loader))[1], expected)
    manifest = json.loads((config.checkpoint_dir / "reproducibility/dataset_manifest.json").read_text())
    assert [s["episode_mapping"] for s in manifest["sources"]] == [[[0, 0]], [[0, 1]]]
    reversed_config = replace(config, data=replace(config.data, lerobot_datasets=tuple(reversed(specs))))
    with pytest.raises(ValueError, match="differ"):
        run_training.preflight(reversed_config)
    # Provenance rejects mutation of either source, independently of path names.
    (Path(specs[1].root) / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"Changed prompt."}\n')
    with pytest.raises(ValueError, match="contract changed"):
        loader.prepare_reproducibility(config, resuming=True)
