"""Exercise statistics, saved provenance and preflight with tiny vector episodes."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.fr3_wuji import compute_norm_stats
from openpi import transforms
from openpi.shared import normalize
from openpi.training import config as cfg
from scripts import run_training
from training_configs import common


@pytest.mark.parametrize("all_constant", [False, True])
@pytest.mark.parametrize("module", ["tomato_fr3_lora_v1", "tomato_fr3_sharpa_lora_v1"])
def test_statistics_save_constant_joints_and_bind_to_vectors(tmp_path, monkeypatch, caplog, all_constant, module):
    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    config = run_training.load_experiment("training_configs.experiments.baselines." + module)
    config = replace(config, model=replace(config.model, action_horizon=3))
    root = Path(config.data.lerobot_datasets[0].root)
    dim = config.model.action_dim
    raw_dim = config.policy_metadata["raw_state_dims"][-1]
    names = {"names": config.policy_metadata["joint_names"]} if dim == 58 else {}
    for folder in ("meta", "data", "videos"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_frames": 4,
        "total_episodes": 1,
        "chunks_size": 1000,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "features": {
            "action": {"shape": [dim], **names},
            "observation.state": {"shape": [raw_dim], **names},
            **{key: {"shape": [1, 1, 3]} for key in config.policy_metadata["camera_keys"]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 4}\n')
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Pick the tomato."}\n')
    state, actions = np.zeros((4, raw_dim), dtype=np.float32), np.zeros((4, dim), dtype=np.float32)
    if not all_constant:
        state[:, 0] = np.arange(4)
        actions[:, 0] = np.arange(4) * 2
    parquet = root / "data/episode_000000.parquet"
    pq.write_table(pa.table({"observation.state": state.tolist(), "action": actions.tolist()}), parquet)

    class Reference:
        def __len__(self):
            return 4

        def __getitem__(self, row):
            return {
                "observation.state": state[row],
                "action": actions[np.minimum(np.arange(3) + row, 3)],
                "prompt": "Pick the tomato.",
                "episode_index": 0,
                **{key: np.zeros((1, 1, 3), dtype=np.uint8) for key in config.policy_metadata["camera_keys"]},
            }

    monkeypatch.setattr(cfg.ModelTransformFactory, "__call__", lambda *args: transforms.Group())
    monkeypatch.setattr(compute_norm_stats.data_loader, "create_torch_dataset", lambda *args: Reference())
    compute_norm_stats.compute(config, batch_size=3)  # Include a final partial batch and changing histogram ranges.
    directory = Path(config.data.assets.assets_dir) / config.data.assets.asset_id
    stats = normalize.load(directory)
    recorded = json.loads((directory / "norm_metadata.json").read_text())
    assert not list(directory.parent.glob(f".{directory.name}.building-*"))
    with pytest.raises(FileExistsError, match="target already exists"):
        compute_norm_stats.compute(config, batch_size=3)
    assert recorded["vector_counts"] == {"state": 4, "actions": 12}
    assert recorded["constant_dimensions"]["state"] == list(range(0 if all_constant else 1, dim))
    assert recorded["constant_dimensions"]["actions"] == list(range(0 if all_constant else 1, dim))
    assert "Zero-variance" in caplog.text
    assert np.all(stats["state"].std[1:] == 0)
    assert np.all(stats["state"].q99[1:] == stats["state"].q01[1:])
    sample = {"state": np.zeros(dim), "actions": np.zeros((3, dim))}
    normalized = transforms.Normalize(stats, use_quantiles=True)(sample)
    assert all(np.isfinite(value).all() for value in normalized.values())
    assert run_training.preflight(config)["norm_provenance"] == "verified"
    actions[0, 0] += 1
    pq.write_table(pa.table({"observation.state": state.tolist(), "action": actions.tolist()}), parquet)
    with pytest.raises(ValueError, match="data or vector transforms differ"):
        run_training.preflight(config)
    retry_config = replace(
        config, data=replace(config.data, assets=replace(config.data.assets, asset_id="retry_stats"))
    )
    retry_directory = directory.parent / "retry_stats"
    original_write = Path.write_text

    def fail_metadata(path, *args, **kwargs):
        if path.name == "norm_metadata.json":
            raise OSError("simulated metadata write failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata)
    with pytest.raises(OSError, match="metadata write failure"):
        compute_norm_stats.compute(retry_config, batch_size=3)
    assert not retry_directory.exists()
    assert not list(directory.parent.glob(".retry_stats.building-*"))
    monkeypatch.setattr(Path, "write_text", original_write)
    compute_norm_stats.compute(retry_config, batch_size=3)
    assert run_training.preflight(retry_config)["norm_provenance"] == "verified"
