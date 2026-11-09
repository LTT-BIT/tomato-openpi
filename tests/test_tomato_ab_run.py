"""Keep preflight read-only and prevent launching onto occupied GPUs."""

import sys

import pytest

from scripts import tomato_ab_run


def devices():
    return [{"index": str(i), "used_mib": 0, "utilization": 0, "compute_pids": []} for i in range(8)]


@pytest.mark.parametrize(("field", "value"), [("used_mib", 66000), ("utilization", 100), ("compute_pids", [123])])
def test_busy_devices_are_rejected(field, value):
    inventory = devices()
    inventory[3][field] = value
    with pytest.raises(RuntimeError, match="occupied"):
        tomato_ab_run.validate_gpus(inventory, [str(i) for i in range(8)])


@pytest.mark.parametrize("selection", [["0"] * 8, ["0", "1"], [str(i) for i in range(1, 9)]])
def test_invalid_gpu_selection_is_rejected(selection):
    with pytest.raises(ValueError, match="distinct GPUs|unavailable"):
        tomato_ab_run.validate_gpus(devices(), selection)


def test_idle_devices_are_accepted():
    tomato_ab_run.validate_gpus(devices(), [str(i) for i in range(8)])


def test_start_on_busy_devices_never_loads_data_or_launches(monkeypatch):
    inventory = devices()
    inventory[0]["compute_pids"] = [123]
    monkeypatch.setattr(tomato_ab_run, "gpu_inventory", lambda: inventory)
    monkeypatch.setattr(sys, "argv", ["tomato_ab_run", "start"])
    monkeypatch.setattr(tomato_ab_run, "cpu_preflight", lambda *args: pytest.fail("must stop before data loading"))
    monkeypatch.setattr(tomato_ab_run.subprocess, "call", lambda *args, **kwargs: pytest.fail("must never launch"))
    with pytest.raises(RuntimeError, match="not started"):
        tomato_ab_run.main()


def test_default_only_checks_b_and_never_launches(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(sys, "argv", ["tomato_ab_run"])
    monkeypatch.setattr(tomato_ab_run, "ROOT", tmp_path)
    monkeypatch.setattr(tomato_ab_run, "gpu_inventory", devices)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    selected = []

    def preflight(variant, exp_name):
        selected.append(variant)
        return SimpleNamespace(fsdp_devices=8, batch_size=64), {"variant": variant}

    monkeypatch.setattr(tomato_ab_run, "cpu_preflight", preflight)
    monkeypatch.setattr(tomato_ab_run.subprocess, "call", lambda *args, **kwargs: pytest.fail("must never launch"))
    tomato_ab_run.main()
    assert selected == ["b"]
    assert len(list((tmp_path / "logs/normal93_preflight").glob("*.json"))) == 1


def test_cuda_environment_overrides_stale_cpu_and_memory_settings():
    from scripts.training_environment import cuda_environment

    source = {
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
    }
    env = cuda_environment(["0", "1"], source)
    assert env["JAX_PLATFORMS"] == "cuda"
    assert "JAX_PLATFORM_NAME" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.90"
    assert source["JAX_PLATFORMS"] == "cpu"
