# ruff: noqa: FBT002, FBT003, PT011, SLF001
"""Keep five-phase prompts, source data and materialized frame supervision aligned."""

import copy
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.subtasks import tomato_in_hand as c
from openpi.subtasks.prepare import digest
from openpi.subtasks.prepare import write_json
from openpi.subtasks.prepare import write_jsonl
from scripts.prepare_tomato_in_hand204 import materialize


class Tokenizer:
    def encode(self, text, add_bos=True):
        return [1] * len(text.split())


def document(length=10):
    return {
        "prompt_profile": c.PROFILE,
        "stage": "coarse",
        "segments": [
            {"label": k, "text": c.ACTION_PROMPTS[k], "start_frame": 2 * i, "end_frame": 2 * i + 2, "reviewed": True}
            for i, k in enumerate(c.SEQUENCE)
        ],
    }


@pytest.mark.parametrize("mutation", ["seven", "old_prompt", "gap", "review", "length", "profile"])
def test_reject_incompatible_annotation(mutation):
    d = document()
    length = 10
    if mutation == "seven":
        d["segments"].append(copy.deepcopy(d["segments"][-1]))
    if mutation == "old_prompt":
        d["segments"][2]["text"] = "Right fingers reposition the truss on the right plate"
    if mutation == "gap":
        d["segments"][2]["start_frame"] += 1
    if mutation == "review":
        d["segments"][0]["reviewed"] = False
    if mutation == "length":
        length = 11
    if mutation == "profile":
        d["prompt_profile"] = "tomato-in-hand-five-phases-v1"
    with pytest.raises(ValueError):
        c.reviewed_segments(d, length)


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "EPISODES", 2)
    monkeypatch.setattr(c, "FRAMES", 20)
    root = tmp_path / "source" / "A" / "batch"
    features = {
        "observation.state": {"dtype": "float32", "shape": [116], "names": list(c.SOURCE_STATE_NAMES)},
        "action": {"dtype": "float32", "shape": [58], "names": list(c.SOURCE_ACTION_NAMES)},
        **{
            cam: {
                "dtype": "video",
                "shape": [224, 224, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": 30},
            }
            for cam in c.CAMERAS
        },
        **{
            k: {"dtype": "int64", "shape": [1], "names": None}
            for k in ["index", "frame_index", "episode_index", "task_index"]
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    }
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "features": features,
        "total_episodes": 2,
        "total_frames": 20,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    entries = [
        {"episode_index": ep, "length": 10, "source_recording_id": f"recording{ep}", "tasks": ["raw goal"]}
        for ep in range(2)
    ]
    write_json(root / "meta/info.json", info)
    write_jsonl(root / "meta/episodes.jsonl", entries)
    write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "raw goal"}])
    write_json(root / "visual_review.json", {})
    receipts = []
    for ep in range(2):
        data = dict(
            index=list(range(ep * 10, (ep + 1) * 10)),
            frame_index=list(range(10)),
            episode_index=[ep] * 10,
            task_index=[0] * 10,
            timestamp=np.arange(10, dtype=np.float32) / 30,
            **{
                "observation.state": np.full((10, 116), ep, dtype=np.float32).tolist(),
                "action": np.full((10, 58), ep + 0.5, dtype=np.float32).tolist(),
            },
        )
        p = root / f"data/chunk-000/episode_{ep:06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(data), p)
        paths = ["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl", str(p.relative_to(root))]
        for cam in c.CAMERAS:
            video = root / f"videos/chunk-000/{cam}/episode_{ep:06d}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"video{ep}{cam}".encode())
            paths.append(str(video.relative_to(root)))
        d = document()
        d["source"] = {
            "episode_index": ep,
            "length": 10,
            "fps": 30,
            "source_recording_id": f"recording{ep}",
            "sha256": {p: digest(root / p) for p in paths},
        }
        write_json(root / f"annotations/subtasks/coarse/episode_{ep:06d}.json", d)
        receipts.append(
            {
                "episode": ep,
                "frames": 10,
                "output_sha256": d["source"]["sha256"][paths[-1]],
                "checks": {
                    "all_boxes_contained": True,
                    "uncovered_frames": [],
                    "center_steps_over_3pct": 0,
                    "scale_steps_over_2pct": 0,
                },
            }
        )
    roi = {
        "current_rgb_pixels_only": True,
        "state_action_unchanged": True,
        "future_coordinates_used": True,
        "semantic_review_status": "sampled_coding_agent_review_complete_not_independent_per_frame_accuracy",
        "visual_review_sha256": digest(root / "visual_review.json"),
        "roi_receipts": receipts,
    }
    write_json(root / "roi_manifest.json", roi)
    publication = tmp_path / "source"
    write_json(
        publication / "manifest.json",
        {
            "packages": [
                {
                    "path": str(root),
                    "category": "normal",
                    "manifest_sha256": digest(root / "roi_manifest.json"),
                    "episodes": 2,
                    "frames": 20,
                }
            ]
        },
    )
    return c.freeze(publication), root


def test_full_materialization_preserves_raw_and_labels(tmp_path, frozen):
    selection, root = frozen
    before = {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
    dest = tmp_path / "prepared"
    m = materialize(selection, dest, Tokenizer())
    assert m["physical_columns_equal"]
    assert m["raw_sources_unchanged"]
    assert before == {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
    for ep in range(2):
        t = pq.read_table(dest / f"b_coarse/data/chunk-000/episode_{ep:06d}.parquet")
        assert t["task_index"].to_pylist() == [0, 0, 1, 1, 2, 2, 1, 1, 3, 3]
        assert t["coarse_phase_index"].to_pylist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
        assert all(t["observation.interaction_available"].to_pylist())
        np.testing.assert_array_equal(t["action"].to_pylist(), np.full((10, 58), ep + 0.5))
    with pytest.raises(FileExistsError):
        materialize(selection, dest, Tokenizer())


def test_materialization_rejects_changed_source(tmp_path, frozen):
    selection, root = frozen
    (root / "videos/chunk-000/observation.images.cam0/episode_000000.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        materialize(selection, tmp_path / "prepared", Tokenizer())
    assert not (tmp_path / "prepared").exists()


def test_materialization_rejects_source_destination(frozen):
    selection, root = frozen
    with pytest.raises(ValueError, match="separate"):
        materialize(selection, root / "prepared", Tokenizer())


def test_independent_config_preserves_training_contract(tmp_path, frozen, monkeypatch):
    selection, root = frozen
    dest = tmp_path / "prepared"
    materialize(selection, dest, Tokenizer())
    monkeypatch.setenv("PI05_IN_HAND204_BUNDLE", str(dest))
    from openpi.subtasks.tomato import COARSE_SEQUENCE
    from training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_crop import build_config

    config = build_config()
    assert config.batch_size == 256
    assert config.num_train_steps == 20000
    assert config.fsdp_devices == 8
    assert config.model.action_horizon == 50
    assert config.model.action_dim == 58
    assert config.data.extra_delta_transform
    assert config.keep_period == 5000
    assert config.save_interval_after == 5000
    assert config.save_before_step == 1
    assert config.exp_name.endswith("s20000_c5000")
    assert config.policy_metadata["rgb_frame_cache"] == c.DEFAULT_CACHE
    assert config.data.interaction_mask_key == "observation.interaction_available"
    assert config.weight_loader.params_path.endswith("pi05_base/params")
    assert len(COARSE_SEQUENCE) == 7
    assert len(c.SEQUENCE) == 5
    assert config.policy_metadata["prompts"] == list(c.TASKS)
    assert config.model.max_token_len == 256
    assert len(config.model.image_keys) == 4
    p = dest / "b_coarse/meta/tasks.jsonl"
    p.write_text(p.read_text().replace("at the center", "on the plate"))
    with pytest.raises(ValueError):
        build_config()


def test_vector_statistics_supports_available_crop_key():
    from scripts.compute_robot_norm_stats import transform_vectors

    def transform(raw):
        assert raw["observation.interaction_available"] == np.bool_(False)
        return {"state": raw["observation.state"], "actions": raw["action"]}

    result = transform_vectors(
        np.zeros((3, 116)), np.ones((3, 58)), np.array([0, 2]), 50, transform, camera_keys=c.CAMERAS
    )
    assert result["actions"].shape == (2, 50, 58)


def test_checked_launcher_uses_full_dataset_counts(tmp_path, frozen, monkeypatch):
    selection, root = frozen
    dest = tmp_path / "prepared"
    materialize(selection, dest, Tokenizer())
    cache = tmp_path / "cache"
    write_json(cache / "manifest.json", {})
    monkeypatch.setenv("PI05_IN_HAND204_BUNDLE", str(dest))
    from openpi.training import reproducibility
    from scripts import run_training
    from scripts.check_tomato_in_hand204 import MODULE
    from scripts.run_training import load_experiment
    from scripts.tomato_in_hand204_run import checked_config

    config = load_experiment(MODULE)
    report = {
        "status": "in_hand204_training_inputs_accepted",
        "episodes": 2,
        "frames": 20,
        "all_model_inputs_equal": True,
        "original_sources_unchanged": True,
        "model_samples_checked": 6,
        "per_frame_prompts_checked": 20,
        "loader_batches": 32,
        "steady_batch_seconds": 0.2,
        "bundle": str(dest.resolve()),
        "cache": str(cache.resolve()),
        "bundle_sha256": digest(dest / "bundle.json"),
        "cache_manifest_sha256": digest(cache / "manifest.json"),
        "norm": {"norm_stats_sha256": "normsha"},
        "code_inventory": reproducibility._code_inventory(config),
    }
    report_path = tmp_path / "report.json"
    write_json(report_path, report)
    monkeypatch.setattr(
        run_training, "preflight", lambda config: {"episodes": 2, "frames": 20, "norm_stats_sha256": "normsha"}
    )
    actual, _ = checked_config(dest, cache, report_path)
    assert actual.exp_name == config.exp_name
    report["code_inventory"] = {}
    write_json(report_path, report)
    with pytest.raises(ValueError, match="Code changed"):
        checked_config(dest, cache, report_path)


def test_ui_exposes204_profile_and_accepted_cache(tmp_path, frozen, monkeypatch):
    selection, root = frozen
    dest = tmp_path / "prepared"
    materialize(selection, dest, Tokenizer())
    monkeypatch.setenv("PI05_IN_HAND204_BUNDLE", str(dest))
    from scripts import tomato_in_hand204_run
    from scripts import training_ui
    from scripts import training_ui_config
    from scripts.check_tomato_in_hand204 import MODULE

    assert MODULE not in {r["id"] for r in training_ui_config.experiment_catalog()}
    assert MODULE in {r["id"] for r in training_ui_config.legacy_catalog()}
    result = training_ui_config.defaults(MODULE)
    assert result["values"]["num_train_steps"] == 20000
    assert result["values"]["save_interval_after"] == 5000
    assert result["protocol"]["training_episodes"] == 2
    assert result["protocol"]["action_prompts"] == c.ACTION_PROMPTS
    calls = []

    def checked(bundle, cache, report):
        calls.append((bundle, cache, report))
        monkeypatch.setenv("PI05_RGB_FRAME_CACHE", str(cache))

    monkeypatch.setattr(tomato_in_hand204_run, "checked_config", checked)
    config = training_ui_config.source_config(MODULE)
    training_ui.prepare_input_environment(config)
    assert len(calls) == 1
    assert calls[0][0] == dest
    assert calls[0][1] == Path(c.DEFAULT_CACHE)
    assert "s20000_c5000" in str(calls[0][2])
    assert __import__("os").environ["PI05_RGB_FRAME_CACHE"] == c.DEFAULT_CACHE
    from types import SimpleNamespace

    training_ui.prepare_input_environment(SimpleNamespace(policy_metadata={}))
    assert "PI05_RGB_FRAME_CACHE" not in __import__("os").environ


def test_three_rgb_view_preserves_sources_and_all_other_training_settings(tmp_path, frozen, monkeypatch):
    selection, _ = frozen
    source = tmp_path / "four"
    materialize(selection, source, Tokenizer())
    monkeypatch.setenv("PI05_IN_HAND204_BUNDLE", str(source))
    from scripts.prepare_in_hand204_three_rgb import prepare
    from training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_crop import build_config as crop
    from training_configs.experiments.tomato_sharpa.subtasks.in_hand204_b_three_rgb import build_config

    old_cache = tmp_path / "old_cache"
    entries = []
    for ep in range(2):
        for cam in c.CAMERAS:
            p = old_cache / cam / f"episode_{ep:06d}.npy"
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, np.zeros((10, 224, 224, 3), dtype=np.uint8))
            p.chmod(0o444)
            entries.append(
                {
                    "episode": ep,
                    "camera": cam,
                    "frames": 10,
                    "cache_file": str(p.relative_to(old_cache)),
                    "cache_sha256": digest(p),
                    "source_sha256": digest(source / f"b_coarse/videos/chunk-000/{cam}/episode_{ep:06d}.mp4"),
                    "bytes": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                    "exact_reference_samples": 5,
                }
            )
    write_json(old_cache / "manifest.json", {"complete": True, "schema": "rgb224_exact_pil_v2", "entries": entries})
    before = {str(p): digest(p) for p in source.rglob("*") if p.is_file()}
    view = tmp_path / "three"
    cache = tmp_path / "three_cache"
    prepare(source, old_cache, view, cache)
    assert before == {str(p): digest(p) for p in source.rglob("*") if p.is_file()}
    assert len(list((view / "videos").rglob("*.mp4"))) == 6
    assert (source / "b_coarse/videos/chunk-000/observation.images.interaction/episode_000000.mp4").exists()
    assert len(json.loads((cache / "manifest.json").read_text())["entries"]) == 6
    for ep in range(2):
        assert digest(view / f"data/chunk-000/episode_{ep:06d}.parquet") == digest(
            source / f"b_coarse/data/chunk-000/episode_{ep:06d}.parquet"
        )
    monkeypatch.setenv("PI05_IN_HAND204_THREE_VIEW", str(view))
    a = build_config()
    b = crop()
    assert len(a.model.image_keys) == 3
    assert a.policy_metadata["visual_tokens"] == 768
    assert a.data.extra_delta_transform
    assert a.model.action_dim == 58
    assert a.model.action_horizon == 50
    for field in [
        "batch_size",
        "num_train_steps",
        "fsdp_devices",
        "keep_period",
        "save_interval_after",
        "lr_schedule",
        "optimizer",
        "sampling_mode",
        "data_seed",
        "augmentation_seed",
    ]:
        assert getattr(a, field) == getattr(b, field)
    assert a.policy_metadata["prompts"] == b.policy_metadata["prompts"]
    assert a.weight_loader == b.weight_loader
    assert a.data.assets.asset_id != b.data.assets.asset_id


def test_three_rgb_queue_waits_for_success_and_committed_checkpoint(tmp_path):
    from scripts.queue_in_hand204_three_rgb import gpu_indices
    from scripts.queue_in_hand204_three_rgb import predecessor_ready

    binding = {
        "id": "a",
        "host": "h",
        "exp_name": "crop",
        "checkpoint_dir": str(tmp_path),
        "child_pid": 1,
        "child_start": "123",
        "gpus": ["GPU" + str(i) for i in range(8)],
    }
    current = {**binding, "status": "running"}
    assert not predecessor_ready(current, binding, "123")
    current["status"] = "completed"
    assert not predecessor_ready(current, binding, None)
    write_json(tmp_path / "20000/_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": 1})
    for name in ["params", "train_state"]:
        write_json(tmp_path / "20000" / name / "_METADATA", {})
    (tmp_path / "20000/assets").mkdir()
    assert predecessor_ready(current, binding, None)
    assert not predecessor_ready(current, binding, "123")
    for status in ["failed", "stopped", "interrupted"]:
        with pytest.raises(ValueError, match="successfully"):
            predecessor_ready({**current, "status": status}, binding, None)
    with pytest.raises(ValueError, match="identity"):
        predecessor_ready({**current, "child_start": "other"}, binding, None)
    devices = [{"index": str(i), "uuid": "GPU" + str(i)} for i in range(8)]
    assert gpu_indices(devices, binding["gpus"]) == [str(i) for i in range(8)]
    with pytest.raises(ValueError, match="UUID"):
        gpu_indices(devices, ["wrong"])


def test_generic_preparation_is_independent_of_legacy_counts(tmp_path, frozen, monkeypatch):
    from scripts.prepare_tomato_in_hand import materialize as prepare

    _, root = frozen
    monkeypatch.setattr(c, "EPISODES", 204)
    monkeypatch.setattr(c, "FRAMES", 588734)
    selected = c.freeze(root.parents[1], expected_episodes=None, expected_frames=None, version="hand_release_r2")
    result = prepare(selected, tmp_path / "new_release", Tokenizer())
    assert result["episodes"] == 2
    assert result["frames"] == 20
    assert result["version"] == "hand_release_r2"
