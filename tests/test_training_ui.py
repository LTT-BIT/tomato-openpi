"""Exercise UI config persistence, job isolation and process control without GPU training."""

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen

import pytest

from scripts import run_training
from scripts import training_ui as ui
from scripts import training_ui_config as editor
from training_configs import common

BASE = "training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1"


@pytest.mark.parametrize("entry", editor.baseline_catalog(), ids=lambda entry: entry["id"].split(".")[-1])
def test_all_baseline_defaults_roundtrip_and_personal_snapshot(entry):
    module = entry["id"]
    baseline = editor.baseline_config(module)
    values = editor.defaults(module)["values"]
    restored = editor.apply_overrides(baseline, values)
    assert restored == baseline
    values = {**values, "exp_name": "alice_ui_trial", "batch_size": 4, "peak_lr": 1e-5}
    scope = {}
    exec(compile(editor.personal_source(module, "alice", values), "personal.py", "exec"), scope)
    saved = scope["build_config"]()
    assert saved.exp_name == "alice_ui_trial"
    assert saved.batch_size == 4
    assert saved.lr_schedule.peak_lr == 1e-5
    assert saved.model == baseline.model
    assert saved.freeze_filter == baseline.freeze_filter
    assert saved.data.base_config == baseline.data.base_config
    assert saved.policy_metadata["ui_baseline"] == module
    assert editor.baseline_config(module) == baseline


@pytest.mark.parametrize(
    "change",
    [
        {"batch_size": True},
        {"batch_size": 0},
        {"num_workers": -1},
        {"peak_lr": float("nan")},
        {"decay_steps": 1000},
        {"action_horizon": 0},
        {"exp_name": "../escape"},
        {"asset_id": "../escape"},
        {"asset_id": "/escape"},
        {"sources": []},
        {"model": "unknown"},
        {"save_before_step": 100},
    ],
)
def test_invalid_edits_fail_before_saving(change):
    values = {**editor.defaults(BASE)["values"], **change}
    with pytest.raises((ValueError, TypeError)):
        editor.apply_overrides(editor.baseline_config(BASE), values)


def test_grouped_sampling_and_data_identity_are_saved():
    values = {
        **editor.defaults(BASE)["values"],
        "dataset_root": "/tmp/new-ui-dataset",
        "repo_id": "team/new",
        "dataset_version": "new_v1",
        "sources": ["capture_20261014"],
        "sampling_mode": "weighted_replacement",
        "sampling_groups": [
            {"name": "A", "episodes": [0, 2], "weight": 7},
            {"name": "B", "episodes": [1], "weight": 3},
        ],
    }
    source = editor.personal_source(BASE, "alice", values)
    scope = {}
    exec(compile(source, "personal.py", "exec"), scope)
    config = scope["build_config"]()
    assert config.data.repo_id == config.policy_metadata["dataset_repo_id"] == "team/new"
    assert config.data.lerobot_datasets[0].root == config.policy_metadata["dataset_root"] == "/tmp/new-ui-dataset"
    assert config.policy_metadata["source_datasets"] == ["capture_20261014"]
    assert [group.weight for group in config.sampling_groups] == [7, 3]
    assert config.sampling_groups[0].episodes == (0, 2)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    catalog = editor.baseline_catalog()
    monkeypatch.setattr(editor, "baseline_catalog", lambda: catalog)
    monkeypatch.setattr(common, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        ui,
        "gpu_inventory",
        lambda: {
            "gpus": [
                {"uuid": "GPU-test0", "index": "0", "name": "test", "total_mb": 80000, "used_mb": 0, "utilization": 0},
                {
                    "uuid": "GPU-test1",
                    "index": "1",
                    "name": "test",
                    "total_mb": 80000,
                    "used_mb": 20000,
                    "utilization": 90,
                },
            ],
            "error": None,
        },
    )
    values = editor.defaults(BASE)["values"]
    values["exp_name"] = "alice_test_ui"
    return {
        "baseline": BASE,
        "owner": "alice",
        "values": values,
        "gpus": ["GPU-test0"],
        "auto_norm": True,
        "wandb_mode": "offline",
        "stats_batch_size": 8,
    }


@pytest.fixture
def fake_launch(monkeypatch):
    def launch(state):
        directory = ui.job_directory(state["id"])
        directory.mkdir(parents=True)
        state.update(worker_pid=os.getpid(), worker_start=ui.process_identity(os.getpid()))
        ui.atomic_json(directory / "state.json", state)
        (directory / "ready").touch()
        (directory / "console.log").touch()
        return state

    monkeypatch.setattr(ui, "launch_worker", launch)


def test_launch_saves_personal_file_without_mutating_baseline(workspace, fake_launch):
    state = ui.start_job(workspace)
    source = Path(state["config_path"]).read_bytes()
    assert state["module"].endswith("template.alice.alice_test_ui")
    assert hashlib.sha256(source).hexdigest() == state["config_sha256"]
    assert state["wandb_mode"] == "offline"
    with pytest.raises(ValueError, match="预留"):
        ui.start_job({**workspace, "values": {**workspace["values"], "exp_name": "another"}})
    with pytest.raises(ValueError, match="正在使用"):
        ui.start_job({**workspace, "gpus": ["GPU-test1"]})
    with pytest.raises(FileExistsError):
        ui.start_job({**workspace, "gpus": ["GPU-test1"], "allow_busy": True})


def test_config_path_rejects_escape_and_symlink(workspace, tmp_path):
    with pytest.raises(ValueError, match="用户名"):
        editor.personal_path(BASE, "../bob", "trial")
    task = tmp_path / "training_configs/experiments/tomato_wuji/template"
    task.mkdir(parents=True)
    (task / "alice").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        editor.personal_path(BASE, "alice", "trial")


def test_gpu_validation(workspace, monkeypatch):
    config = editor.baseline_config(BASE)
    for selected in ([], ["missing"], ["GPU-test0", "GPU-test0"]):
        with pytest.raises(ValueError, match="GPU"):
            ui.validate_devices(selected, config)
    with pytest.raises(ValueError, match="整除"):
        ui.validate_devices(["GPU-test0"], replace(config, fsdp_devices=2))
    ui.validate_devices(["GPU-test0", "GPU-test1"], config, allow_busy=True)
    monkeypatch.setattr(ui, "gpu_inventory", lambda: {"gpus": [], "error": "nvidia-smi missing"})
    with pytest.raises(ValueError, match="nvidia-smi missing"):
        ui.validate_devices(["GPU-test0"], config)


def test_preflight_missing_and_existing_norm(workspace, monkeypatch):
    config, _ = ui.build_request(workspace)
    calls = []
    monkeypatch.setattr(
        run_training, "preflight", lambda config, require_norm=True: calls.append(require_norm) or {"frames": 2}
    )
    result = ui.preview(workspace, check_data=True)
    assert "自动计算" in result["statistics"]
    assert calls == [False]
    norm = Path(result["norm_dir"])
    norm.mkdir(parents=True)
    with pytest.raises(ValueError, match="不完整"):
        ui.preview(workspace, check_data=True)
    (norm / "norm_stats.json").write_text("{}")
    result = ui.preview(workspace, check_data=True)
    assert "可复用" in result["statistics"]
    assert calls[-2:] == [False, True]

    def invalid(config, *, require_norm=True):
        if require_norm:
            raise ValueError("statistics mismatch")
        return {}

    monkeypatch.setattr(run_training, "preflight", invalid)
    with pytest.raises(ValueError, match="mismatch"):
        ui.preview(workspace, check_data=True)


def test_worker_sequences_cpu_preparation_and_gpu_training(workspace, fake_launch, monkeypatch):
    config, _ = ui.build_request(workspace)
    state = ui.start_job(workspace)
    monkeypatch.setattr(run_training, "load_experiment", lambda module: config)
    calls = []
    monkeypatch.setattr(
        ui, "run_stage", lambda directory, state, label, command, env: calls.append((label, command, env))
    )
    ui.worker(state["id"])
    assert [row[0] for row in calls] == ["校验数据", "计算归一化统计", "核对数据与统计", "训练中"]
    assert all(row[2]["JAX_PLATFORMS"] == "cpu" and row[2]["CUDA_VISIBLE_DEVICES"] == "" for row in calls[:-1])
    assert calls[-1][2]["JAX_PLATFORMS"] == "cuda"
    assert calls[-1][2]["CUDA_VISIBLE_DEVICES"] == "GPU-test0"
    assert calls[-1][1][-2:] == ["--wandb-mode", "offline"]
    assert ui.read_job(state["id"])["status"] == "completed"


def test_worker_failure_never_reaches_training(workspace, fake_launch, monkeypatch):
    config, _ = ui.build_request(workspace)
    state = ui.start_job(workspace)
    monkeypatch.setattr(run_training, "load_experiment", lambda module: config)
    calls = []

    def fail(directory, state, label, command, env):
        calls.append(label)
        raise ValueError("invalid dataset")

    monkeypatch.setattr(ui, "run_stage", fail)
    ui.worker(state["id"])
    assert calls == ["校验数据"]
    assert ui.read_job(state["id"])["status"] == "failed"


def test_resume_checks_config_and_keeps_original_parameters(workspace, fake_launch, monkeypatch):
    config, _ = ui.build_request(workspace)
    state = ui.start_job(workspace)
    state["status"] = "failed"
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    monkeypatch.setattr(run_training, "load_experiment", lambda module: config)
    restored = ui.resume_job({"id": state["id"]})
    assert restored["module"] == state["module"]
    assert restored["resume"] is False  # No checkpoint yet: retry preflight.
    restored["status"] = "failed"
    ui.atomic_json(ui.job_directory(restored["id"]) / "state.json", restored)
    config.checkpoint_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="尚无 checkpoint"):
        ui.resume_job({"id": state["id"]})
    (config.checkpoint_dir / "1000").mkdir()
    restored = ui.resume_job({"id": state["id"]})
    assert restored["resume"] is True
    Path(state["config_path"]).write_text("# modified")
    with pytest.raises(ValueError, match="配置文件已修改"):
        ui.resume_job({"id": state["id"]})


def test_shared_disk_foreign_host_not_marked_dead(workspace, fake_launch):
    state = ui.start_job(workspace)
    state.update(host="another-host", worker_start="different")
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    assert ui.read_job(state["id"])["status"] == "queued"
    with pytest.raises(ValueError, match="原训练服务器"):
        ui.stop_job(state["id"])
    state["host"] = socket.gethostname()
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    assert ui.read_job(state["id"])["status"] == "interrupted"


def test_run_stage_stops_owned_process_group(tmp_path):
    state = {"status": "queued"}
    pid_file = tmp_path / "child_pid"
    code = f"import os,time,pathlib; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"

    def request_stop():
        for _ in range(200):
            if pid_file.exists():
                (tmp_path / "stop").touch()
                return
            time.sleep(0.01)

    thread = threading.Thread(target=request_stop)
    thread.start()
    with pytest.raises(ui.StopRequestedError):
        ui.run_stage(tmp_path, state, "test", [sys.executable, "-c", code], os.environ.copy())
    thread.join()
    assert ui.process_identity(int(pid_file.read_text())) is None


def test_http_requires_token_and_json_for_mutations(workspace, monkeypatch):
    requests = []
    monkeypatch.setattr(ui, "start_job", lambda payload: requests.append(payload) or {"status": "queued"})
    server = ui.ThreadingHTTPServer(("127.0.0.1", 0), ui.handler_class("test-token"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url, timeout=5) as response:
            assert b"app.js" in response.read()
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/api/baselines", timeout=5)
        assert error.value.code == 403
        headers = {"X-Training-Token": "test-token", "Content-Type": "application/json"}
        with urlopen(Request(url + "/api/start", data=b"{}", headers=headers), timeout=5) as response:
            assert json.load(response)["status"] == "queued"
        assert requests == [{}]
        with pytest.raises(HTTPError):
            urlopen(Request(url + "/api/start", data=b"{}", headers={"Content-Type": "application/json"}), timeout=5)
        assert requests == [{}]
        with urlopen(Request(url + "/api/baseline?id=" + BASE, headers=headers), timeout=5) as response:
            assert json.load(response)["values"]["batch_size"] == 8
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_living_training_child_keeps_gpu_reserved_after_supervisor_exit(workspace, fake_launch):
    state = ui.start_job(workspace)
    state.update(worker_start="dead-supervisor", child_pid=os.getpid(), child_start=ui.process_identity(os.getpid()))
    ui.atomic_json(ui.job_directory(state["id"]) / "state.json", state)
    assert ui.read_job(state["id"])["status"] == "orphaned"
    with pytest.raises(ValueError, match="预留"):
        ui.validate_devices(["GPU-test0"], editor.baseline_config(BASE))
