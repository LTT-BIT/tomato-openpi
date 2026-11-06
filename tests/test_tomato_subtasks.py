"""Whole trajectories, source grouping, state budgets and causal decisions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.subtasks.phase_model import canonical_state
from openpi.subtasks.phase_model import history_indices
from openpi.subtasks.phase_model import training_statistics
from openpi.subtasks.binding import verify_annotation
from openpi.subtasks.prepare import grouped_split
from openpi.subtasks.prepare import inspect_plan
from openpi.subtasks.prepare import prepare
from openpi.subtasks.tomato import COARSE_SEQUENCE
from openpi.subtasks.tomato import GOAL
from openpi.subtasks.tomato import PHASES
from openpi.subtasks.tomato import PROMPTS
from openpi.subtasks.tomato import SOURCE_ACTION_NAMES
from openpi.subtasks.tomato import SOURCE_STATE_NAMES
from openpi.subtasks.tomato import PhaseDecision
from openpi.subtasks.tomato import phase_at
from openpi.subtasks.tomato import reviewed_segments


def document(length=14):
    return {
        "stage": "coarse",
        "segments": [
            {"label": label, "start_frame": 2 * i, "end_frame": 2 * i + 2, "reviewed": True}
            for i, label in enumerate(COARSE_SEQUENCE)
        ],
    }


def test_frame_boundaries_are_half_open_and_require_review():
    doc = document()
    rows = reviewed_segments(doc, 14)
    assert [phase_at(rows, f) for f in (0, 1, 2, 12, 13)] == [0, 0, 1, 6, 6]
    with pytest.raises(ValueError, match="outside"):
        phase_at(rows, 14)
    doc["segments"][0]["reviewed"] = False
    with pytest.raises(ValueError, match="reviewed"):
        reviewed_segments(doc, 14)


@pytest.mark.parametrize("fps", [30, 60, 10])
def test_causal_history_is_time_based_and_never_future(fps):
    times = np.arange(3 * fps) / fps
    for end in (0, 2, len(times) - 1):
        ids = history_indices(times, end)
        assert len(ids) == 16
        assert ids[-1] == end
        assert ids.min() >= 0
        assert ids.max() <= end
        assert np.all(times[ids] <= times[end] + 1e-12)
    ids = history_indices(times, len(times) - 1)
    assert times[ids[-1]] - times[ids[0]] == pytest.approx(1.5, abs=1 / fps)


def test_split_is_reproducible_and_keeps_normal_speedup_together():
    records = [{"recording_group": f"raw{i // 2}"} for i in range(40)]
    splits = grouped_split(records)
    assert splits == grouped_split(records)
    assignment = {i: s for s, ids in splits.items() for i in ids}
    assert all(assignment[i] == assignment[i + 1] for i in range(0, 40, 2))
    with pytest.raises(ValueError, match="ten independent"):
        grouped_split(records[:4])


def test_phase_proposals_need_consecutive_evidence_cannot_skip_or_complete():
    decision = PhaseDecision(confirmations=2)
    p = [0.0] * 7
    p[1] = 1.0
    assert not decision.update(p, 0)["changed"]
    assert decision.update(p, 0.5)["phase"] == PHASES[1]
    p = [0.0] * 7
    p[5] = 1.0
    result = decision.update(p, 1.0)
    assert result["phase_index"] == 1
    assert result["needs_review"]
    assert not result["task_complete"]
    with pytest.raises(ValueError, match="increasing"):
        decision.update(p, 1.0)
    decision.reset()
    assert decision.phase == 0


def test_training_normalization_uses_only_explicit_training_features():
    mean, std = training_statistics(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(mean, [2, 3])
    np.testing.assert_allclose(std, [1, 1])


def test_agent_uses_canonical_positions_and_rejects_wrong_joint_order():
    state = np.arange(232).reshape(2, 116)
    actual = canonical_state(state, list(SOURCE_STATE_NAMES))
    expected = np.concatenate([state[:, :7], state[:, 28:50], state[:, 14:21], state[:, 72:94]], axis=1)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="source order"):
        canonical_state(state, list(reversed(SOURCE_STATE_NAMES)))


@pytest.fixture
def plan(tmp_path):
    root = tmp_path / "source"
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [116], "names": list(SOURCE_STATE_NAMES)},
            "action": {"dtype": "float32", "shape": [58], "names": list(SOURCE_ACTION_NAMES)},
            **{f"observation.images.cam{i}": {"dtype": "video", "shape": [2, 2, 3]} for i in range(3)},
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    entries = [{"episode_index": i, "length": 14, "source_recording_id": f"recording-{i // 2}"} for i in range(20)]
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": GOAL}) + "\n")
    result = {"review_complete": True, "episodes": []}
    for i, entry in enumerate(entries):
        table = pa.table(
            {
                "episode_index": [i] * 14,
                "index": np.arange(i * 14, (i + 1) * 14),
                "frame_index": np.arange(14),
                "timestamp": np.arange(14) / 30,
                "task_index": [0] * 14,
                "observation.state": np.arange(14 * 116).reshape(14, 116).tolist(),
                "action": np.arange(14 * 58).reshape(14, 58).tolist(),
            }
        )
        relative = info["data_path"].format(episode_chunk=0, episode_index=i)
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, root / relative)
        files = [relative, "meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]
        for cam in range(3):
            relative = info["video_path"].format(
                episode_chunk=0, episode_index=i, video_key=f"observation.images.cam{cam}"
            )
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_bytes(b"synthetic-media-copy-test-only")
            files.append(relative)
        doc = document()
        doc["source"] = {
            "episode_index": i,
            "length": 14,
            "fps": 30,
            "source_recording_id": entry["source_recording_id"],
            "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
        }
        annotation = tmp_path / f"coarse-{i}.json"
        annotation.write_text(json.dumps(doc))
        result["episodes"].append(
            {
                "dataset": str(root),
                "episode_index": i,
                "annotation": str(annotation),
                "recording_group": entry["source_recording_id"],
            }
        )
    return result


class SyntheticTokenizer:
    """Only for preparation mechanics; real tokenizer checked separately."""

    def encode(self, text, *, add_bos=False):
        return list(range(len(text.split()) + int(add_bos)))


def test_preparation_keeps_whole_episodes_and_all_physical_columns(plan, tmp_path):
    output = prepare(plan, tmp_path / "bundle", SyntheticTokenizer())
    for mode in ("a_goal", "b_coarse"):
        tasks = [json.loads(s)["task"] for s in (output / mode / "meta/tasks.jsonl").read_text().splitlines()]
        assert tasks == ([GOAL] if mode == "a_goal" else list(dict.fromkeys(PROMPTS)))
        for i in (0, 19):
            rel = f"data/chunk-000/episode_{i:06d}.parquet"
            original = pq.read_table(plan["episodes"][i]["dataset"] + "/" + rel)
            actual = pq.read_table(output / mode / rel)
            assert len(actual) == 14
            for column in ("frame_index", "timestamp", "observation.state", "action"):
                assert actual[column].equals(original[column])
            assert actual["coarse_phase_index"].to_pylist() == [p for p in range(7) for _ in range(2)]
            annotation = json.loads((output / mode / f"annotations/episode_{i:06d}.json").read_text())
            verify_annotation(output / mode, annotation)
            assert annotation["source"]["dataset"] == str(output / mode)
            assert annotation["source"]["episode_index"] == i
            original_doc = json.loads(Path(plan["episodes"][i]["annotation"]).read_text())
            assert annotation["segments"] == original_doc["segments"]
            snapshot = output / mode / annotation["source_migration"]["upstream_annotation"]
            assert json.loads(snapshot.read_text()) == original_doc
            assert snapshot.read_bytes() == Path(plan["episodes"][i]["annotation"]).read_bytes()
            if mode == "b_coarse":
                goal = json.loads((output / "a_goal" / f"annotations/episode_{i:06d}.json").read_text())
                assert goal["source"]["sha256"] != annotation["source"]["sha256"]

    bundle = json.loads((output / "bundle.json").read_text())
    assert bundle["mapping"][0]["annotation_role"] == "upstream_reviewed_snapshot"
    for mode in ("a_goal", "b_coarse"):
        derived = {"review_complete": True, "episodes": [
            {**row, "dataset": str(output / mode), "episode_index": i,
             "annotation": str(output / mode / f"annotations/episode_{i:06d}.json")}
            for i, row in enumerate(plan["episodes"])
        ]}
        assert len(inspect_plan(derived)) == 20
        assert bundle["mapping"][0]["active_annotations"][mode]["dataset"] == str(output / mode)
    with pytest.raises(ValueError, match="new bundle"):
        prepare(plan, output, SyntheticTokenizer())


def test_preparation_rejects_unreviewed_changed_sources_and_group_leaks(plan):
    plan["review_complete"] = False
    with pytest.raises(ValueError, match="Finish review"):
        inspect_plan(plan)
    plan["review_complete"] = True
    plan["episodes"][1]["recording_group"] = "different-group"
    with pytest.raises(ValueError, match="different groups"):
        inspect_plan(plan)
    plan["episodes"][1]["recording_group"] = plan["episodes"][0]["recording_group"]
    from pathlib import Path

    path = Path(plan["episodes"][0]["dataset"]) / "data/chunk-000/episode_000000.parquet"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        inspect_plan(plan)


def test_failed_video_publication_leaves_no_partial_bundle(plan, tmp_path):
    from pathlib import Path

    # The annotation still matches its declared files, but is missing one required
    # media hash. Publication must fail, without leaving an apparently ready bundle.
    path = Path(plan["episodes"][0]["annotation"])
    doc = json.loads(path.read_text())
    key = next(key for key in doc["source"]["sha256"] if key.startswith("videos/"))
    del doc["source"]["sha256"][key]
    path.write_text(json.dumps(doc))
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="required hashes"):
        prepare(plan, output, SyntheticTokenizer())
    assert not output.exists()
    assert not list(tmp_path.glob(".tomato-subtasks-*"))


def test_decision_jsonl_interface_resets_between_runs():
    import os
    import subprocess
    import sys

    request = {"id": 1, "timestamp": 0.5, "probabilities": [1, 0, 0, 0, 0, 0, 0]}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "scripts.tomato_phase_decision"],
        input=json.dumps(request) + '\n{"id":2,"reset":true}\n',
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": ".:src"},
    )
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert messages[0]["phase_index"] == 0
    assert messages[0]["prompt"] == PROMPTS[0]
    assert messages[1] == {"id": 2, "reset": True}


def test_preparation_requires_shared_metadata_hashes(plan):
    annotation = Path(plan["episodes"][0]["annotation"])
    doc = json.loads(annotation.read_text())
    del doc["source"]["sha256"]["meta/tasks.jsonl"]
    annotation.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="required hashes"):
        inspect_plan(plan)
