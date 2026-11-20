"""Exercise copied personal experiments without populating reserved task templates."""

import importlib
import importlib.util
import sys

import pytest


@pytest.fixture
def make_personal_experiment(tmp_path, monkeypatch):
    def create(baseline_module, *, peak_lr=None):
        source = (
            "from dataclasses import replace\n"
            f"from {baseline_module} import build_config as build_baseline\n\n"
            "def build_config():\n"
            "    config = build_baseline()\n"
            "    return replace(config, exp_name='alice_trial_s42_run01',\n"
            + ("" if peak_lr is None else f"        lr_schedule=replace(config.lr_schedule, peak_lr={peak_lr!r}),\n")
            + "        policy_metadata={**config.policy_metadata, 'owner': 'alice', 'purpose': 'copied_baseline'})\n"
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
