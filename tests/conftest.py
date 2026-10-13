"""Exercise copied personal experiments without populating reserved task templates."""

import importlib
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def make_personal_experiment(tmp_path, monkeypatch):
    def create(baseline_module, *, peak_lr=None):
        baseline = importlib.import_module(baseline_module)
        source = Path(baseline.__file__).read_text()
        source = source.replace("import os\n", "from dataclasses import replace\nimport os\n")
        for prefix in ("PI05_FULLTASK", "PI05_SHARPA"):
            source = source.replace(f"{prefix}_DATASET_ROOT", "PI05_DATASET_ROOT")
            source = source.replace(f"{prefix}_ASSETS_DIR", "PI05_ASSETS_DIR")
        source = source.replace(
            "def build_config():\n    return build_recipe(", "def build_config():\n    config = build_recipe("
        )
        lr_override = (
            "" if peak_lr is None else f"        lr_schedule=replace(config.lr_schedule, peak_lr={peak_lr!r}),\n"
        )
        source += (
            "\n    return replace(\n"
            "        config,\n"
            '        exp_name="alice_trial_s42_run01",\n'
            + lr_override
            + '        policy_metadata={**config.policy_metadata, "owner": "alice", "purpose": "copied_baseline"},\n'
            "    )\n"
        )
        module_name = baseline_module.replace(".baseline.", ".template.alice.")
        path = tmp_path / "personal_experiment.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        return module_name

    return create
