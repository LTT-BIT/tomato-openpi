"""Check local data contracts, run isolation, and persisted launch results."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.shared import normalize
from scripts import run_training


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setenv("PI05_RUNTIME_DIR", str(tmp_path / "runtime"))
    config = run_training.load_experiment("training_configs.experiments.baselines.tomato_fr3_lora_v2")
    root = Path(config.data.lerobot_datasets[0].root)
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "videos").mkdir()
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_frames": 2,
        "total_episodes": 1,
        "features": {
            "action": {"shape": [54]},
            "observation.state": {"shape": [108]},
            **{key: {"shape": [8, 8, 3]} for key in config.policy_metadata["camera_keys"]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 2}\n')
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Pick the tomato."}\n')
    norm_dir = Path(config.data.assets.assets_dir) / config.data.assets.asset_id
    stats = normalize.NormStats(mean=np.zeros(54), std=np.ones(54), q01=-np.ones(54), q99=np.ones(54))
    normalize.save(norm_dir, {"state": stats, "actions": stats})
    return config, root, norm_dir


def test_preflight_reports_actual_prompt_and_norm_digest(prepared):
    config, _, norm_dir = prepared
    result = run_training.preflight(config)
    assert result["prompts"] == ["Pick the tomato."]
    assert result["norm_stats_sha256"] == hashlib.sha256((norm_dir / "norm_stats.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["fps", "action", "camera", "prompt", "norm", "norm_audit", "horizon"])
def test_preflight_rejects_incompatible_inputs(prepared, change):
    config, root, norm_dir = prepared
    path = root / "meta/info.json"
    info = json.loads(path.read_text())
    if change == "fps":
        info["fps"] = 20
    elif change == "action":
        info["features"]["action"]["shape"] = [14]
    elif change == "camera":
        info["features"].pop("observation.images.cam2")
    elif change == "prompt":
        (root / "meta/tasks.jsonl").write_text('{"task": ""}\n')
    elif change == "norm":
        config = replace(config, policy_metadata={**config.policy_metadata, "expected_norm_sha256": "wrong"})
    elif change == "norm_audit":
        config = replace(config, policy_metadata={**config.policy_metadata, "expected_norm_audit_sha256": "wrong"})
    else:
        (norm_dir / "norm_metadata.json").write_text('{"action_horizon": 10}\n')
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="differ|prompts"):
        run_training.preflight(config)


def test_lock_rejects_same_run_but_allows_another_run(prepared):
    config, _, _ = prepared
    with run_training.launch_lock(config):
        with pytest.raises(RuntimeError, match="already running"), run_training.launch_lock(config):
            pass
        with run_training.launch_lock(replace(config, exp_name="other_run")):
            pass
    with run_training.launch_lock(config):
        pass


def test_launch_writes_local_log_snapshot_and_explicit_offline_mode(prepared, monkeypatch):
    config, _, _ = prepared
    from scripts import train

    def fake_train(actual):
        assert actual is config
        print("test trainer output")

    monkeypatch.setattr(train, "main", fake_train)
    monkeypatch.setenv("WANDB_DIR", "test-wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    handlers = list(run_training.logging.getLogger().handlers)
    try:
        run_training.launch(config, wandb_mode="offline", inputs=run_training.preflight(config))
    finally:
        run_training.logging.getLogger().handlers = handlers
    directory = Path(config.checkpoint_base_dir).parent / "logs" / config.name / config.exp_name
    manifest = json.loads(next(directory.glob("*.json")).read_text())
    assert manifest["status"] == "training_completed"
    assert manifest["wandb_mode"] == "offline"
    assert manifest["config"]["fields"]["model"]["fields"]["action_dim"] == 54
    assert "test trainer output" in Path(manifest["log"]).read_text()
