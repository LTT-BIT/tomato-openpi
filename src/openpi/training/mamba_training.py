"""Phase A-only training/data contracts, using the existing trainer and assets."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.training import config as cfg

ARCHITECTURE_FILE = "mamba_architecture.json"
UPSTREAM = {
    "mamba3_jax": "ffa55848fc513ca91aa330bda4c33c86dab74827",
    "MambaAct": "97414cfffc82225c6be34abf486f06c7eb11d22d",
}


def architecture_record(model):
    return {"schema_version": 1, "model": dataclasses.asdict(model), "upstream": UPSTREAM}


def validate_stats(stats, dim):
    if stats is None or set(stats) != {"state", "actions"}:
        raise ValueError("Mamba Phase A requires ordinary Absolute state/actions statistics")
    for key, stat in stats.items():
        for field in ("mean", "std", "q01", "q99"):
            value = getattr(stat, field)
            if value is None or np.shape(value) != (dim,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid ordinary Absolute statistics {key}.{field}")
        if (stat.std < 0).any() or (stat.q99 < stat.q01).any():
            raise ValueError(f"Invalid normalization range {key}")


@dataclasses.dataclass(frozen=True)
class MambaDataView(cfg.DataConfig):
    mamba_identity: dict = dataclasses.field(default_factory=dict)

    def save_extra_assets(self, directory):
        norm_path = Path(directory) / "norm_stats.json"
        record = {**self.mamba_identity, "norm_stats_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest()}
        (Path(directory) / ARCHITECTURE_FILE).write_text(json.dumps(record, indent=2) + "\n")


@dataclasses.dataclass(frozen=True)
class MambaSharpaData(cfg.LeRobotFr3SharpaMultimodalDataConfig):
    def create(self, assets_dirs, model_config):
        if (
            not isinstance(model_config, Pi0Config)
            or not model_config.pi05
            or hasattr(model_config, "paper_mode")
            or self.extra_delta_transform
        ):
            raise ValueError("Phase A data requires Absolute Mamba model; Delta/Paper unsupported")
        asset_dir = Path(self.assets.assets_dir or assets_dirs) / (self.assets.asset_id or self.repo_id)
        if (asset_dir / "delta_paper_stats.npz").exists() or any(asset_dir.glob("*paper*")):
            raise ValueError("Mamba Phase A rejects Paper statistics assets")
        data = super().create(assets_dirs, model_config)
        if data.norm_stats is not None:
            validate_stats(data.norm_stats, model_config.action_dim)
        # Pi05 defaults to quantiles; Phase A explicitly uses ordinary z score.
        if isinstance(model_config, MambaPi0Config):
            return MambaDataView(
                **{**vars(data), "use_quantile_norm": False}, mamba_identity=architecture_record(model_config)
            )
        return dataclasses.replace(data, use_quantile_norm=False)


@dataclasses.dataclass(frozen=True)
class MambaTrainConfig(cfg.TrainConfig):
    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.model, MambaPi0Config) or self.freeze_filter is not self.model.get_freeze_filter():
            raise ValueError("Mamba Phase A requires FullFT with no freeze filter")
        if not isinstance(self.data, MambaSharpaData) or self.data.extra_delta_transform:
            raise ValueError("Mamba Phase A requires its validated Absolute data factory")
        if self.pytorch_weight_path is not None:
            raise ValueError("Mamba Phase A is JAX-only")


def load_mamba_checkpoint(model_config, step_dir, asset_id):
    """Architecture-gated inference restore; the old BaseModel API stays unchanged."""
    from openpi.models.model import restore_params

    directory = Path(step_dir) / "assets" / asset_id
    record = json.loads((directory / ARCHITECTURE_FILE).read_text())
    expected = architecture_record(model_config)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Mamba checkpoint architecture/config differs from requested model")
    if hashlib.sha256((directory / "norm_stats.json").read_bytes()).hexdigest() != record["norm_stats_sha256"]:
        raise ValueError("Mamba checkpoint normalization digest differs")
    return model_config.load(restore_params(Path(step_dir) / "params"), remove_extra_params=False)


def validate_training(config):
    """Invoked before trainer allocation, including direct entry and resume."""
    if not isinstance(config, MambaTrainConfig):
        raise ValueError("Use MambaTrainConfig to enforce FullFT and Absolute contracts")
    data = config.data.create(config.assets_dirs, config.model)
    validate_stats(data.norm_stats, config.model.action_dim)
    if config.resume:
        candidates = sorted((p for p in config.checkpoint_dir.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
        if not candidates:
            raise ValueError("No complete Mamba checkpoint to resume")
        directory = candidates[-1] / "assets" / data.asset_id
        record = json.loads((directory / ARCHITECTURE_FILE).read_text())
        expected = architecture_record(config.model)
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume Mamba architecture/config differs")
        if hashlib.sha256((directory / "norm_stats.json").read_bytes()).hexdigest() != record["norm_stats_sha256"]:
            raise ValueError("Resume Mamba normalization digest differs")

        current = Path(config.data.assets.assets_dir or config.assets_dirs) / data.asset_id / "norm_stats.json"
        if hashlib.sha256(current.read_bytes()).hexdigest() != record["norm_stats_sha256"]:
            raise ValueError("Resume Mamba normalization differs from current assets")
