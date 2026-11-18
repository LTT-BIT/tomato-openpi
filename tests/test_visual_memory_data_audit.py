"""Audit rules must reject recording leakage and find exact repeated source videos."""

import pytest

from scripts.check_visual_memory_data import candidate_split
from scripts.check_visual_memory_data import duplicate_groups


def row(index, batch, identity):
    return {
        "episode_index": index,
        "recording_group": identity,
        "source_dataset": f"/source/A/{batch}_example_normal",
        "source_episode_index": index,
        "length": 100,
        "source_sha256": {
            f"videos/chunk-000/{camera}/episode_{index:06d}.mp4": camera for camera in ("cam0", "cam1", "cam2")
        },
    }


def test_candidate_keeps_source_package_whole_without_activating_training():
    records = [row(0, "1112", "a"), row(1, "1112", "b"), row(2, "1113", "c"), row(3, "1114", "d")]
    split = candidate_split(records)
    assert split["status"] == "candidate_only_not_activated"
    assert [split["summary"][key]["recordings"] for key in ("train", "development", "test")] == [2, 1, 1]
    assert split["requires_train_only_normalization"] is True
    assert split["requires_new_training_from_unexposed_base"] is True


def test_reused_recording_id_is_rejected_even_if_episode_ids_differ():
    with pytest.raises(ValueError, match="Recording identity"):
        candidate_split([row(0, "1112", "same"), row(1, "1114", "same")])


def test_duplicate_video_hashes_detected_despite_renamed_episode_files():
    records = [row(0, "1112", "a"), row(1, "1114", "b")]
    assert duplicate_groups(records, ("cam0", "cam1", "cam2")) == [[0, 1]]
    records[1]["source_sha256"]["videos/chunk-000/cam2/episode_000001.mp4"] = "different"
    assert duplicate_groups(records, ("cam0", "cam1", "cam2")) == []


def test_audit_reader_rejects_cache_changed_after_content_verification(tmp_path):
    from types import SimpleNamespace

    from scripts.check_visual_memory_data import HashVerifiedAuditReader

    path = tmp_path / "frame.npy"
    path.write_bytes(b"changed")
    reader = SimpleNamespace(
        cache_root=tmp_path,
        cache_entries={(0, "cam0"): {"cache_file": path.name}},
        verified={path.name: (len(b"changed"), 1)},
    )
    with pytest.raises(ValueError, match="changed after full hash"):
        HashVerifiedAuditReader._query_videos(reader, {"cam0": [0.0]}, 0)  # noqa: SLF001
