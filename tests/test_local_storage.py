"""Verify concurrent exclusion and all-or-nothing normalization publication."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from openpi.shared.local_storage import new_artifact_directory


def test_bundle_is_invisible_until_both_files_are_published(tmp_path):
    destination = tmp_path / "stats"
    with new_artifact_directory(destination) as staging:
        (staging / "norm_stats.json").write_text("statistics")
        assert not destination.exists()
        (staging / "norm_metadata.json").write_text("metadata")
        assert not destination.exists()
    assert (destination / "norm_stats.json").read_text() == "statistics"
    assert (destination / "norm_metadata.json").read_text() == "metadata"
    assert not list(tmp_path.glob(".stats.building-*"))
    with pytest.raises(FileExistsError, match="already exists"), new_artifact_directory(destination):
        pass
    assert (destination / "norm_stats.json").read_text() == "statistics"


def test_failed_writer_does_not_publish_and_target_can_retry(tmp_path):
    destination = tmp_path / "stats"

    def failed_writer():
        with new_artifact_directory(destination) as staging:
            (staging / "norm_stats.json").write_text("partial statistics")
            raise ValueError("failed metadata")

    with pytest.raises(ValueError, match="failed metadata"):
        failed_writer()
    assert not destination.exists()
    assert not list(tmp_path.glob(".stats.building-*"))
    with new_artifact_directory(destination) as staging:
        (staging / "norm_stats.json").write_text("retry")
        (staging / "norm_metadata.json").write_text("retry metadata")
    assert destination.exists()


def test_second_process_is_rejected_for_same_target_but_can_write_other_target(tmp_path):
    destination = tmp_path / "stats"
    script = """
import sys
from pathlib import Path
from openpi.shared.local_storage import new_artifact_directory
try:
    with new_artifact_directory(Path(sys.argv[1])):
        raise AssertionError('Second writer entered the locked target')
except RuntimeError as error:
    assert 'already being computed' in str(error)
with new_artifact_directory(Path(sys.argv[2])) as staging:
    (staging / 'norm_stats.json').write_text('other target')
"""
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    with new_artifact_directory(destination) as staging:
        subprocess.run(
            [sys.executable, "-c", script, str(destination), str(tmp_path / "other")],
            env=environment,
            cwd=tmp_path,
            check=True,
            capture_output=True,
            timeout=30,
        )
        assert not destination.exists()
        (staging / "norm_stats.json").write_text("owner")
    assert (destination / "norm_stats.json").read_text() == "owner"
    assert (tmp_path / "other/norm_stats.json").read_text() == "other target"
