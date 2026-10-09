import csv
import dataclasses
import json
import pathlib

import datasets
import numpy as np
import pytest

from openpi.training import config
from openpi.training import reproducibility
from openpi.training import sampling
import openpi.transforms as transforms


class TableDataset:
    def __init__(self):
        self.hf_dataset = datasets.Dataset.from_dict({"episode_index": [1, 1, 0], "frame_index": [0, 1, 0]})

    def __len__(self):
        return len(self.hf_dataset)


@pytest.fixture
def run_inputs(tmp_path):
    cfg = dataclasses.replace(
        config.get_config("debug"),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        exp_name="audit",
        sampling_mode="epoch_shuffle",
        num_train_steps=4,
        batch_size=2,
    )
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_frames": 3, "total_episodes": 2}))
    (root / "meta/episodes.jsonl").write_text('{"episode_index":0,"length":1}\n{"episode_index":1,"length":2}\n')
    (root / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"Pick tomato."}\n')
    (root / "videos").mkdir()
    (root / "videos/test.mp4").write_bytes(b"fingerprint-test")
    data = config.DataConfig(
        repo_id="test",
        lerobot_datasets=(config.LeRobotDatasetSpec(repo_id="test", root=str(root)),),
        norm_stats={"state": transforms.NormStats(mean=np.zeros(1), std=np.ones(1))},
    )
    return cfg, data, TableDataset(), sampling.epoch_shuffle_indices(3, seed=42, num_samples=8)


def test_artifacts_and_resume(run_inputs):
    cfg, data, source, plan = run_inputs
    summary = reproducibility.prepare_run(cfg, data, source, plan, resuming=False)
    assert summary["planned_coverage"] == 1.0
    directory = cfg.checkpoint_dir / "reproducibility"
    np.testing.assert_array_equal(np.load(directory / "sample_indices.npy"), plan)
    with (directory / "samples.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    for index, row in zip(plan, rows, strict=True):
        expected = source.hf_dataset[int(index)]
        assert int(row["episode_index"]) == expected["episode_index"]
        assert int(row["frame_index"]) == expected["frame_index"]
    assert (
        reproducibility.prepare_run(
            dataclasses.replace(cfg, resume=True, overwrite=False), data, source, plan, resuming=True
        )
        == summary
    )


@pytest.mark.parametrize("change", ["seed", "budget", "norm", "data", "plan"])
def test_resume_rejects_changed_contract(run_inputs, change):
    cfg, data, source, plan = run_inputs
    reproducibility.prepare_run(cfg, data, source, plan, resuming=False)
    if change == "seed":
        cfg = dataclasses.replace(cfg, data_seed=99)
    elif change == "budget":
        cfg = dataclasses.replace(cfg, num_train_steps=5)
    elif change == "norm":
        data = dataclasses.replace(data, norm_stats=None)
    elif change == "data":
        (pathlib.Path(data.lerobot_datasets[0].root) / "videos/test.mp4").write_bytes(b"different-video")
    else:
        plan = plan[::-1]
    with pytest.raises(ValueError, match="contract changed"):
        reproducibility.prepare_run(cfg, data, source, plan, resuming=True)


def test_resume_rejects_missing_or_modified_artifacts(run_inputs):
    cfg, data, source, plan = run_inputs
    with pytest.raises(ValueError, match="Missing reproducibility"):
        reproducibility.prepare_run(cfg, data, source, plan, resuming=True)
    reproducibility.prepare_run(cfg, data, source, plan, resuming=False)
    (cfg.checkpoint_dir / "reproducibility/samples.csv").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        reproducibility.prepare_run(cfg, data, source, plan, resuming=True)


def test_stable_object_serialization():
    class Tokenizer:
        def __init__(self):
            self.max_len = 256

    assert reproducibility._json_bytes(Tokenizer()) == reproducibility._json_bytes(Tokenizer())  # noqa: SLF001
