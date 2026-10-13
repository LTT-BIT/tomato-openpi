"""Check local data contracts, run isolation, and persisted launch results."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from openpi.shared import normalize
from openpi.training import norm_provenance
from openpi.training import reproducibility
from scripts import run_training
from training_configs import common


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    config = run_training.load_experiment("training_configs.experiments.baselines.tomato_fr3_lora_v1")
    root = Path(config.data.lerobot_datasets[0].root)
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "data/episode_000000.parquet").write_bytes(b"fixture vector data")
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
    (norm_dir / "norm_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": norm_provenance.SCHEMA_VERSION,
                "repo_id": config.data.repo_id,
                "frames": 2,
                "episodes": 1,
                "action_horizon": config.model.action_horizon,
                "contract": norm_provenance.build_contract(config),
                "norm_stats_sha256": reproducibility.file_sha256(norm_dir / "norm_stats.json"),
            }
        )
    )
    return config, root, norm_dir


def test_preflight_reports_actual_prompt_and_norm_digest(prepared):
    config, _, norm_dir = prepared
    result = run_training.preflight(config)
    assert result["prompts"] == ["Pick the tomato."]
    assert result["norm_provenance"] == "verified"
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
    assert run_training.os.environ["WANDB_DIR"] == str(Path(config.checkpoint_base_dir).parent)
    directory = Path(config.checkpoint_base_dir).parent / "logs" / config.name / config.exp_name
    manifest = json.loads(next(directory.glob("*.json")).read_text())
    assert manifest["status"] == "training_completed"
    assert manifest["wandb_mode"] == "offline"
    assert manifest["config"]["fields"]["model"]["fields"]["action_dim"] == 54
    assert "test trainer output" in Path(manifest["log"]).read_text()


@pytest.mark.parametrize("change", ["repo_id", "frames", "episodes", "datasets", "vectors", "transform", "stats"])
def test_preflight_rejects_wrong_normalization_binding(prepared, change):
    config, root, norm_dir = prepared
    path = norm_dir / "norm_metadata.json"
    recorded = json.loads(path.read_text())
    if change == "repo_id":
        recorded["repo_id"] = "other/dataset"
    elif change in ("frames", "episodes"):
        recorded[change] += 1
    elif change == "datasets":
        recorded["datasets"] = [{"repo_id": "other/dataset", "frames": 2, "episodes": 1}]
    elif change == "vectors":
        (root / "data/episode_000000.parquet").write_bytes(b"different vectors, same dataset name and frame count")
    elif change == "transform":
        config = replace(config, data=replace(config.data, extra_delta_transform=False))
    else:
        stats = normalize.load(norm_dir)
        stats["state"].mean[0] += 1
        normalize.save(norm_dir, stats)
    path.write_text(json.dumps(recorded))
    with pytest.raises(ValueError, match="differ"):
        run_training.preflight(config)


def test_identical_dataset_can_move_and_use_another_training_recipe(prepared, tmp_path, monkeypatch):
    config, root, _ = prepared
    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    config = replace(
        config,
        data=replace(config.data, lerobot_datasets=(replace(config.data.lerobot_datasets[0], root=str(relocated)),)),
        name="another_recipe",
        exp_name="another_run",
        batch_size=4,
        num_train_steps=10,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Preflight must not build model/tokenizer transforms")

    monkeypatch.setattr(run_training.cfg.ModelTransformFactory, "__call__", fail_if_called)
    assert run_training.preflight(config)["norm_provenance"] == "verified"


@pytest.mark.parametrize("legacy_override", ["none", "flag_only", "digest_only", "pinned"])
@pytest.mark.parametrize("digest_kind", ["file", "audit"])
def test_old_normalization_requires_explicit_override_and_digest(prepared, legacy_override, digest_kind):
    config, _, norm_dir = prepared
    (norm_dir / "norm_metadata.json").unlink()
    policy = {**config.policy_metadata, "allow_legacy_norm_stats": legacy_override in ("flag_only", "pinned")}
    if legacy_override in ("digest_only", "pinned"):
        if digest_kind == "file":
            policy["expected_norm_sha256"] = reproducibility.file_sha256(norm_dir / "norm_stats.json")
        else:
            policy["expected_norm_audit_sha256"] = hashlib.sha256(
                reproducibility.serialize_config(normalize.load(norm_dir))
            ).hexdigest()
    config = replace(config, policy_metadata=policy)
    if legacy_override == "pinned":
        assert run_training.preflight(config)["norm_provenance"] == "legacy_pinned"
    else:
        with pytest.raises(ValueError, match="provenance is missing"):
            run_training.preflight(config)


def test_legacy_override_cannot_bypass_wrong_metadata(prepared):
    config, _, norm_dir = prepared
    (norm_dir / "norm_metadata.json").write_text('{"repo_id": "other/dataset", "action_horizon": 50}')
    config = replace(
        config,
        policy_metadata={
            **config.policy_metadata,
            "allow_legacy_norm_stats": True,
            "expected_norm_sha256": reproducibility.file_sha256(norm_dir / "norm_stats.json"),
        },
    )
    with pytest.raises(ValueError, match="dataset identity differs"):
        run_training.preflight(config)
