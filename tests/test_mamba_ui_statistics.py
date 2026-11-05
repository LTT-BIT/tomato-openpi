"""GUI-produced production configs run existing statistics without model allocation."""

from pathlib import Path

import pytest
from test_delta_paper_training import prepared as paper_prepared

from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.training import norm_provenance
from openpi.training.mamba_training import MambaDataView
from openpi.training.mamba_training import MambaPaperDataView
from openpi.training.mamba_training import model_data_config
from scripts import compute_experiment_norm
from scripts import run_training
from scripts import training_ui_config as editor


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    return paper_prepared.__wrapped__(tmp_path, monkeypatch, request)


@pytest.mark.parametrize("prepared", [50], indirect=True)
@pytest.mark.parametrize("trial", ["none", "t1", "t3"])
def test_gui_config_statistics_dispatch_and_reopen(prepared, monkeypatch, trial):
    build, _, _ = prepared
    source = build(trial="t3", mode="timestamp_zscore_arms", beta=0.5)
    values = {
        **editor.config_values(source),
        "model_architecture": "mamba3",
        "paper_trial": trial,
        "asset_id": f"gui_{trial}",
        "exp_name": f"gui_{trial}",
        "action_semantics": "absolute" if trial == "none" else "delta",
    }
    config = editor.apply_overrides(source, values)
    assert isinstance(config.model, MambaPi0Config)
    report = run_training.preflight(config, require_norm=False)
    assert report["prompts"] == ["Pick the tomato."]
    before = norm_provenance.build_contract(config)
    monkeypatch.setattr(compute_experiment_norm, "resolve_experiment", lambda *a: config)
    monkeypatch.setattr("sys.argv", ["compute_experiment_norm", "gui_fixture", "--stats-batch-size", "3"])
    compute_experiment_norm.main()  # Same command dispatched by the UI worker.
    reopened = editor.apply_overrides(source, values)
    assert norm_provenance.build_contract(reopened) == before
    run_training.preflight(reopened)
    reopened.model.validate_training(reopened)
    view = model_data_config(reopened)
    assert isinstance(view, MambaDataView if trial == "none" else MambaPaperDataView)
    assert view.use_quantile_norm == (trial != "none")
    directory = Path(reopened.data.assets.assets_dir) / reopened.data.assets.asset_id
    assert (directory / "norm_stats.json").is_file()
    if trial != "none":
        assert reopened.model.paper_asset_sha256
        assert all((directory / n).is_file() for n in ("delta_paper.npz", "delta_paper.json", "norm_metadata.json"))
