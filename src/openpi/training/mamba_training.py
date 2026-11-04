"""Mamba training/data contracts, reusing the existing trainer and Paper assets."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.models.pi0_config import Pi0Config
from openpi.models.pi0_mamba_config import MambaPi0Config
from openpi.training import config as cfg
from openpi.training.delta_paper import PaperDataView
from openpi.training.delta_paper import PaperSharpaData

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
        save_architecture(directory, self.mamba_identity)


def save_architecture(directory, identity):
    norm_path = Path(directory) / "norm_stats.json"
    record = {**identity, "norm_stats_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest()}
    if "paper_asset_sha256" in identity["model"]:
        from openpi.shared import delta_paper as paper

        record["paper_files_sha256"] = {
            name: paper.sha256(Path(directory) / name) for name in (paper.BUNDLE, paper.MANIFEST, "norm_metadata.json")
        }
    (Path(directory) / ARCHITECTURE_FILE).write_text(json.dumps(record, indent=2) + "\n")


@dataclasses.dataclass(frozen=True)
class MambaPaperDataView(PaperDataView):
    mamba_identity: dict = dataclasses.field(default_factory=dict)

    def save_extra_assets(self, directory):
        super().save_extra_assets(directory)
        save_architecture(directory, self.mamba_identity)


def model_data_config(config):
    data = config.data.create(config.assets_dirs, config.model)
    adapt = getattr(config.model, "adapt_data_config", None)
    return data if adapt is None else adapt(data)


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
        from openpi.models.pi0_mamba_paper import PaperMambaPi0Config

        if isinstance(self.model, PaperMambaPi0Config):
            if (
                type(self.data) is not PaperSharpaData
                or not self.data.extra_delta_transform
                or self.sampling_mode != "epoch_shuffle"
            ):
                raise ValueError("Paper Mamba requires the original PaperSharpaData and epoch_shuffle")
            for name in ("paper_mode", "paper_beta", "paper_std_floor"):
                if getattr(self.data, name) != getattr(self.model, name):
                    raise ValueError(f"Paper Mamba model/data mismatch: {name}")
            if self.data.asset_directory(self.assets_dirs).resolve() != Path(self.model.paper_asset_dir).resolve():
                raise ValueError("Paper Mamba model/data asset directories differ")
        elif not isinstance(self.data, MambaSharpaData) or self.data.extra_delta_transform:
            raise ValueError("Mamba Phase A requires its validated Absolute data factory")
        if self.pytorch_weight_path is not None:
            raise ValueError("Mamba Phase A is JAX-only")


@dataclasses.dataclass(frozen=True)
class MambaScheduledTrainConfig(MambaTrainConfig, cfg.ScheduledTrainConfig):
    """Preserve optional early/late checkpoint schedules during architecture conversion."""


def architecture_matches(record, expected):
    # A content-authenticated Paper asset can move between devices/directories.
    actual_model, expected_model = dict(record.get("model", {})), dict(expected["model"])
    if "paper_asset_sha256" in expected_model:
        actual_model.pop("paper_asset_dir", None)
        expected_model.pop("paper_asset_dir", None)
    return actual_model == expected_model and all(record.get(k) == v for k, v in expected.items() if k != "model")


def load_mamba_checkpoint(model_config, step_dir, asset_id):
    """Architecture-gated inference restore; the old BaseModel API stays unchanged."""
    from openpi.models.model import restore_params

    directory = Path(step_dir) / "assets" / asset_id
    record = json.loads((directory / ARCHITECTURE_FILE).read_text())
    expected = architecture_record(model_config)
    if not architecture_matches(record, expected):
        raise ValueError("Mamba checkpoint architecture/config differs from requested model")
    if hashlib.sha256((directory / "norm_stats.json").read_bytes()).hexdigest() != record["norm_stats_sha256"]:
        raise ValueError("Mamba checkpoint normalization digest differs")
    if "paper_asset_sha256" in expected["model"]:
        from openpi.shared import delta_paper as paper

        files = record.get("paper_files_sha256", {})
        if set(files) != {paper.BUNDLE, paper.MANIFEST, "norm_metadata.json"} or any(
            paper.sha256(directory / name) != digest for name, digest in files.items()
        ):
            raise ValueError("Paper Mamba checkpoint assets differ from saved digests")
        # Relocate only for restore; identity comparison above used the saved source config.
        model_config = dataclasses.replace(model_config, paper_asset_dir=str(directory))
    params = restore_params(Path(step_dir) / "params")
    if getattr(model_config, "paper_beta", 0):
        _, arrays = paper.load_bundle(
            directory,
            mode=model_config.paper_mode,
            beta=model_config.paper_beta,
            horizon=model_config.action_horizon,
            std_floor=model_config.paper_std_floor,
            expected_sha256=model_config.paper_asset_sha256,
        )
        if "paper_noise_cholesky" not in params or not np.array_equal(
            np.asarray(params["paper_noise_cholesky"]), arrays["noise_cholesky"]
        ):
            raise ValueError("Checkpoint noise factor differs from authenticated Paper assets")
    return model_config.load(params, remove_extra_params=False)


def validate_training(config):
    """Invoked before trainer allocation, including direct entry and resume."""
    if not isinstance(config, MambaTrainConfig):
        raise ValueError("Use MambaTrainConfig to enforce FullFT and the selected algorithm contracts")
    data = model_data_config(config)
    if isinstance(config.data, PaperSharpaData):
        config.data.validate_normalization(config, {})
    else:
        validate_stats(data.norm_stats, config.model.action_dim)
    if config.resume:
        candidates = sorted((p for p in config.checkpoint_dir.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
        if not candidates:
            raise ValueError("No complete Mamba checkpoint to resume")
        directory = candidates[-1] / "assets" / data.asset_id
        record = json.loads((directory / ARCHITECTURE_FILE).read_text())
        expected = architecture_record(config.model)
        if not architecture_matches(record, expected):
            raise ValueError("Resume Mamba architecture/config differs")
        if hashlib.sha256((directory / "norm_stats.json").read_bytes()).hexdigest() != record["norm_stats_sha256"]:
            raise ValueError("Resume Mamba normalization digest differs")

        current = Path(config.data.assets.assets_dir or config.assets_dirs) / data.asset_id / "norm_stats.json"
        if hashlib.sha256(current.read_bytes()).hexdigest() != record["norm_stats_sha256"]:
            raise ValueError("Resume Mamba normalization differs from current assets")
