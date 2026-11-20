"""Group recovery must use explicit conversion input provenance, never old recording IDs."""

import hashlib
import json

import pytest

from scripts.training_ui_groups import detect_bundle_groups
from scripts.training_ui_groups import detect_groups


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def dataset(tmp_path):
    write_json(tmp_path / "conversion/manifest.json", {"config": {"source_root": ["/old/A-subtask", "/old/B-subtask"]}})
    write_json(tmp_path / "meta/info.json", {"codebase_version": "v2.1", "total_episodes": 3, "total_frames": 60})
    rows = [
        {"episode_index": i, "length": length, "source_start_ns": i * 100, "source_end_ns": i * 100 + 10}
        for i, length in [(0, 10), (2, 20), (5, 30)]
    ]
    (tmp_path / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps({**row, "source_recording_id": "/old/B/episode999"}) for row in rows)
    )
    for n, (group, row) in enumerate(zip(["A", "B", "A"], rows, strict=True)):
        write_json(
            tmp_path / f"conversion/records/{n}.json",
            {"source": f"/old/{group}-subtask/episode999", "episodes": [row]},
        )
    return tmp_path


def test_recovers_noncontiguous_groups_without_historical_source_directories(dataset):
    result = detect_groups(str(dataset))
    assert result["episodes"] == 3
    assert result["groups"] == [
        {"name": "A", "source_root": "/old/A-subtask", "episodes": [0, 5], "frames": 40},
        {"name": "B", "source_root": "/old/B-subtask", "episodes": [2], "frames": 20},
    ]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "length", "time", "source", "bad_json"])
def test_rejects_incomplete_or_inconsistent_conversion_records(dataset, mutation):
    path = dataset / "conversion/records/0.json"
    record = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    elif mutation == "duplicate":
        write_json(dataset / "conversion/records/extra.json", record)
    elif mutation == "bad_json":
        path.write_text("{")
    else:
        if mutation == "unknown":
            record["episodes"][0]["episode_index"] = 99
        elif mutation == "length":
            record["episodes"][0]["length"] = 9
        elif mutation == "time":
            record["episodes"][0]["source_start_ns"] = 1
        elif mutation == "source":
            record["source"] = "/old/unknown/episode999"
        write_json(path, record)
    with pytest.raises(ValueError, match="episode|source_root|Expecting"):
        detect_groups(str(dataset))


@pytest.mark.parametrize(
    "roots",
    [[], ["/old/A", "/old/A"], ["/old", "/old/A"], ["/one/A", "/two/A"], ["/old/../A"], ["relative/A"]],
)
def test_rejects_ambiguous_roots(dataset, roots):
    write_json(dataset / "conversion/manifest.json", {"config": {"source_root": roots}})
    with pytest.raises(ValueError, match="source_root|来源|组名"):
        detect_groups(str(dataset))


def test_missing_provenance_does_not_guess_groups(dataset):
    (dataset / "conversion/manifest.json").unlink()
    with pytest.raises(ValueError, match="manifest"):
        detect_groups(str(dataset))


def test_summary_mismatch_is_rejected(dataset):
    write_json(dataset / "meta/info.json", {"codebase_version": "v2.1", "total_episodes": 3, "total_frames": 59})
    with pytest.raises(ValueError, match="info.json"):
        detect_groups(str(dataset))


def test_bundle_grouping_uses_prepared_ids_and_excludes_short_paper_episodes(tmp_path):
    root = tmp_path / "b_coarse"
    episodes = [{"episode_index": i, "length": length} for i, length in [(0, 4), (1, 2), (2, 8)]]
    write_json(root / "meta/info.json", {"total_episodes": 3, "total_frames": 14})
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes))
    mapping = [
        {**row, "source_episode_index": 42, "source_dataset": f"/history/subtask-roi/{side}/batch{i}"}
        for i, (row, side) in enumerate(zip(episodes, "ABB", strict=True))
    ]
    manifest = tmp_path / "bundle.json"
    write_json(manifest, {"mapping": mapping})
    checksum = hashlib.sha256(manifest.read_bytes()).hexdigest()
    result = detect_bundle_groups(str(root), checksum, horizon=3)
    assert result["episodes"] == 2
    assert result["eligible_windows"] == 8
    assert [(group["name"], group["episodes"], group["eligible_windows"]) for group in result["groups"]] == [
        ("A", [0], 2),
        ("B", [2], 6),
    ]
    assert detect_bundle_groups(str(root), checksum, horizon=1)["eligible_windows"] == 14
    with pytest.raises(ValueError, match="来源记录已变化"):
        detect_bundle_groups(str(root), "stale_hash", horizon=3)
