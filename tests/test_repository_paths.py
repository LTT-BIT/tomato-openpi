"""Guard checkout-local defaults against old mount settings and a different cwd."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from openpi.shared import download
from scripts.run_training import load_experiment
from training_configs import common


def test_defaults_use_checkout_even_with_old_runtime_setting(tmp_path, monkeypatch):
    repository = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PI05_RUNTIME_DIR", str(tmp_path / "old-mount"))
    for key in ("PI05_FULLTASK_DATASET_ROOT", "PI05_FULLTASK_ASSETS_DIR", "PI05_SOURCE_DATA_DIR"):
        monkeypatch.delenv(key, raising=False)
    assert common.repository_root() == repository
    config = load_experiment("training_configs.experiments.baselines.tomato_fr3_lora_v1")
    assert Path(config.checkpoint_base_dir) == repository / "checkpoints"
    assert Path(config.data.lerobot_datasets[0].root) == repository / "data/tomato_fulltask_v1"
    assert Path(config.data.assets.assets_dir) == repository / "assets/fulltask_c0"
    fullft = load_experiment("training_configs.experiments.baselines.tomato_fr3_fullft_v1")
    assert fullft.policy_metadata["source_datasets"][0] == str(repository / "data/harvest-tomato/0925/A")
    assert Path(download.DEFAULT_CACHE_DIR) == repository / ".cache/openpi"


@pytest.mark.parametrize("old_settings", [False, True])
def test_environment_uses_its_checkout_and_replaces_old_mount_paths(tmp_path, old_settings):
    repository = tmp_path / "checkout with spaces"
    for name in ("scripts/__init__.py", "training_configs/__init__.py", "training_configs/common.py"):
        destination = repository / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((common.repository_root() / name).read_text())
    paths = {
        "UV_PROJECT_ENVIRONMENT": repository / ".venv",
        "UV_CACHE_DIR": repository / ".cache/uv",
        "XDG_CACHE_HOME": repository / ".cache",
        "OPENPI_DATA_HOME": repository / ".cache/openpi",
        "HF_HOME": repository / ".cache/huggingface",
        "HF_HUB_CACHE": repository / ".cache/huggingface/hub",
        "HF_DATASETS_CACHE": repository / ".cache/huggingface/datasets",
        "TORCH_HOME": repository / ".cache/torch",
        "WANDB_DIR": repository,
        "WANDB_CACHE_DIR": repository / ".cache/wandb",
        "WANDB_DATA_DIR": repository / ".cache/wandb-data",
        "TMPDIR": repository / ".cache/tmp",
    }
    environment = os.environ.copy()
    for key in (*paths, "PI05_RUNTIME_DIR"):
        if old_settings:
            environment[key] = str(tmp_path / "old-mount" / key)
        else:
            environment.pop(key, None)
    environment["PYTHONPATH"] = str(repository)
    keys = [*paths, "PI05_RUNTIME_DIR"]
    probe = f"import scripts; import os,json; print(json.dumps({{key: os.environ.get(key) for key in {keys!r}}}))"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    values = json.loads(result.stdout)
    assert values.pop("PI05_RUNTIME_DIR") is None
    assert values == {key: str(path) for key, path in paths.items()}
    for directory in ("data", "assets", "checkpoints", "logs", "wandb", ".cache/tmp"):
        assert (repository / directory).is_dir()
