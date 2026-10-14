"""Keep checkpoint restoration independent of online/offline W&B history."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from openpi.training import config as cfg
from openpi.training import tracking


@pytest.mark.parametrize(
    "modes", [("online", "online"), ("offline", "offline"), ("offline", "online"), ("online", "offline")]
)
def test_resume_uses_new_segment_with_same_group_and_checkpoint_step(tmp_path, monkeypatch, modes):
    config = replace(cfg.get_config("debug"), checkpoint_base_dir=str(tmp_path), wandb_enabled=True)
    Path(config.checkpoint_dir).mkdir(parents=True)
    calls, finishes = [], []
    ids = iter(["first", "second"])
    monkeypatch.setattr(tracking.wandb.util, "generate_id", lambda: next(ids))

    def init(**kwargs):
        calls.append(kwargs)
        directory = tmp_path / "wandb" / kwargs["id"] / "files"
        directory.mkdir(parents=True)
        return SimpleNamespace(id=kwargs["id"], entity="team", dir=str(directory))

    monkeypatch.setattr(tracking.wandb, "init", init)
    monkeypatch.setattr(tracking.wandb, "finish", lambda **kwargs: finishes.append(kwargs))
    monkeypatch.setenv("WANDB_RESUME", "must")
    monkeypatch.setenv("WANDB_MODE", modes[0])
    with tracking.session(config, resuming=False, start_step=0):
        pass
    monkeypatch.setenv("WANDB_MODE", modes[1])
    with tracking.session(config, resuming=True, start_step=7):
        pass
    assert [call["id"] for call in calls] == ["first", "second"]
    assert all(call["group"] == "first" and "resume" not in call for call in calls)
    assert calls[1]["config"]["logging_session"]["start_step"] == 7
    assert "WANDB_RESUME" not in os.environ
    assert Path(config.checkpoint_dir, "wandb_id.txt").read_text().strip() == "first"
    records = [json.loads(path.read_text()) for path in Path(config.checkpoint_dir, "wandb_sessions").glob("*.json")]
    assert sorted(record["start_step"] for record in records) == [0, 7]
    assert all(record["status"] == "training_completed" for record in records)
    assert finishes == [{"exit_code": 0}, {"exit_code": 0}]


def test_failed_attempt_finishes_wandb_and_records_failure(tmp_path, monkeypatch):
    config = replace(cfg.get_config("debug"), checkpoint_base_dir=str(tmp_path), wandb_enabled=True)
    Path(config.checkpoint_dir).mkdir(parents=True)
    finished = []
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setattr(
        tracking.wandb, "init", lambda **kwargs: SimpleNamespace(id="failed", entity=None, dir=str(tmp_path / "files"))
    )
    monkeypatch.setattr(tracking.wandb, "finish", lambda **kwargs: finished.append(kwargs))
    with pytest.raises(KeyboardInterrupt), tracking.session(config, resuming=False, start_step=0):
        raise KeyboardInterrupt
    record = json.loads(Path(config.checkpoint_dir, "wandb_latest_session.json").read_text())
    assert record["status"] == "failed"
    assert "KeyboardInterrupt" in record["error"]
    assert finished == [{"exit_code": 1}]


def test_real_wandb_offline_creates_two_syncable_segments_without_cloud_resume(tmp_path):
    script = """
from dataclasses import replace
from pathlib import Path
import json, os, sys
import wandb
from openpi.training.config import get_config
from openpi.training.tracking import session
root = Path(sys.argv[1])
os.environ['WANDB_DIR'] = str(root)
config = replace(get_config('debug'), checkpoint_base_dir=str(root / 'checkpoints'), wandb_enabled=True)
Path(config.checkpoint_dir).mkdir(parents=True)
records = []
for resuming, step in ((False, 0), (True, 7)):
    with session(config, resuming=resuming, start_step=step) as record:
        assert wandb.run.offline
        wandb.log({'loss': 1.0}, step=step)
        records.append(record.copy())
assert records[0]['run_id'] != records[1]['run_id']
assert records[0]['root_run_id'] == records[1]['root_run_id']
for record in records:
    assert list(Path(record['run_directory']).glob('run-*.wandb'))
print(json.dumps({'segments': len(records), 'resume_step': records[1]['start_step']}))
"""
    environment = {
        **os.environ,
        "WANDB_MODE": "offline",
        "WANDB_SILENT": "true",
        "WANDB_DISABLE_GIT": "true",
        # Shared environments intentionally have no editable checkout bindings.
        "PYTHONPATH": os.pathsep.join(
            str(Path(__file__).resolve().parents[1] / relative)
            for relative in (".", "src", "packages/openpi-client/src")
        ),
    }
    environment.pop("WANDB_API_KEY", None)
    environment.pop("WANDB_RESUME", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == {"segments": 2, "resume_step": 7}
    assert "resume` will be ignored" not in result.stderr
