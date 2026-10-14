"""Check checkout isolation and prevent installers from mutating shared environments."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("python.sh", "training_runtime.sh", "setup_training_env.sh"):
        shutil.copy(source / name, scripts / name)
    return root


def executable(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + text)
    path.chmod(0o750)


def run(checkout, script, *args, shared=None):
    env = {**os.environ}
    env.pop("PI05_SHARED_ENV", None)
    if shared is not None:
        env["PI05_SHARED_ENV"] = str(shared)
    return subprocess.run(
        ["bash", str(checkout / "scripts" / script), *args],
        cwd=checkout.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_explicit_shared_environment_wins_and_preserves_arguments(checkout, tmp_path):
    shared = tmp_path / "shared environment"
    executable(shared / "run-python", 'printf "%s\\n" "$@"\n')
    executable(checkout / ".venv/bin/python", "exit 99\n")
    result = run(checkout, "python.sh", "-c", "value with spaces", shared=shared)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(checkout), "-c", "value with spaces"]


def test_existing_local_environment_uses_this_checkout(checkout):
    executable(checkout / ".venv/bin/python", 'printf "%s\\n" "$PWD" "$PYTHONPATH" "$@"\n')
    result = run(checkout, "python.sh", "-m", "scripts.run_training")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(checkout),
        f"{checkout}:{checkout / 'src'}:{checkout / 'packages/openpi-client/src'}",
        "-m",
        "scripts.run_training",
    ]


def test_explicit_missing_shared_environment_does_not_fall_back(checkout, tmp_path):
    executable(checkout / ".venv/bin/python", "exit 0\n")
    result = run(checkout, "python.sh", shared=tmp_path / "missing")
    assert result.returncode != 0
    assert "Shared environment unavailable" in result.stderr


def test_setup_shared_environment_only_probes_and_rejects_sync_options(checkout, tmp_path):
    shared = tmp_path / "shared"
    executable(shared / "run-python", 'printf "%s\\n" "$@"\n')
    result = run(checkout, "setup_training_env.sh", shared=shared)
    assert result.returncode == 0, result.stderr
    assert str(checkout) in result.stdout
    assert not (checkout / ".venv").exists()
    result = run(checkout, "setup_training_env.sh", "--upgrade", shared=shared)
    assert result.returncode == 2
    assert "read-only" in result.stderr


def test_setup_refuses_sync_into_symlinked_environment(checkout, tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (checkout / ".venv").symlink_to(shared, target_is_directory=True)
    result = run(checkout, "setup_training_env.sh", "--local")
    assert result.returncode != 0
    assert "symlinked .venv" in result.stderr
    assert list(shared.iterdir()) == []
