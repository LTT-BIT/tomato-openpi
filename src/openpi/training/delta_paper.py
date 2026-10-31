"""Opt-in data adapter, complete-window view and assets; no baseline defaults."""

from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np

from openpi import transforms
from openpi.shared import delta_paper as paper
from openpi.shared import normalize
from openpi.training import config as cfg


class CompleteWindows:
    """Select only complete source-FPS windows and retain sampling/audit IDs."""

    def __init__(self, dataset, horizon):
        self.dataset = dataset
        self.source_total_frames = len(dataset)
        self.root = dataset.root
        self.meta = dataset.meta
        ids = dataset.hf_dataset.select_columns(["episode_index", "frame_index"]).with_format("numpy")[:]
        episodes = np.asarray(ids["episode_index"])
        frames = np.asarray(ids["frame_index"])
        indices = []
        for episode in np.unique(episodes):
            positions = np.flatnonzero(episodes == episode)
            if not np.array_equal(frames[positions], np.arange(len(positions))):
                raise ValueError("Paper training requires contiguous episode frames")
            indices.extend(positions[: max(0, len(positions) - horizon + 1)])
        self.indices = np.sort(np.asarray(indices, np.int64))
        if not len(self.indices):
            raise ValueError("Dataset contains no complete action windows")
        self.hf_dataset = dataset.hf_dataset.select(self.indices.tolist())
        self.view_identity = {"version": paper.VERSION, "complete_horizon": horizon}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[int(self.indices[index])]


class PaperCollectionDataset:
    """Complete windows per source; local episode IDs map only in memory."""

    def __init__(self, datasets, specs):
        from torch.utils.data import ConcatDataset

        from openpi.training.data_loader import TransformedDataset
        from openpi.training.dataset_collection import collection_layout
        from openpi.training.dataset_collection import validate_collection_features

        self.dataset = ConcatDataset(datasets)
        self.sources = []
        self.layout = collection_layout([spec.root for spec in specs])
        validate_collection_features([item["info"] for item in self.layout])
        ids = []
        for dataset, layout in zip(datasets, self.layout, strict=True):
            source = dataset
            while isinstance(source, TransformedDataset):
                source = source._dataset  # noqa: SLF001
            if not isinstance(source, CompleteWindows):
                raise ValueError("Paper collection requires complete-window sources")
            self.sources.append(source)
            columns = source.hf_dataset.select_columns(["episode_index", "frame_index"]).with_format("numpy")[:]
            episodes = np.asarray(columns["episode_index"], np.int64)
            horizon = source.view_identity["complete_horizon"]
            for row in layout["episodes"]:
                frames = np.asarray(columns["frame_index"])[episodes == row["episode_index"]]
                if not np.array_equal(frames, np.arange(max(0, row["length"] - horizon + 1))):
                    raise ValueError("Paper collection windows differ from episode metadata")
            if source.source_total_frames != layout["info"]["total_frames"]:
                raise ValueError("Paper source frame count differs from metadata")
            ids.append(np.column_stack(([layout["mapping"][int(e)] for e in episodes], columns["frame_index"])))
        self.frame_ids = np.concatenate(ids).astype(np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


@dataclass(frozen=True)
class PaperDataView(cfg.DataConfig):
    paper_complete_windows: bool = True
    paper_asset_dir: str = ""
    paper_bundle_sha256: str = ""
    paper_manifest_sha256: str = ""
    paper_audit_sha256: str = ""

    def save_extra_assets(self, destination):
        source = Path(self.paper_asset_dir)
        metadata = json.loads((source / paper.MANIFEST).read_text())
        if (
            paper.sha256(source / paper.BUNDLE) != self.paper_bundle_sha256
            or paper.sha256(source / paper.MANIFEST) != self.paper_manifest_sha256
            or paper.sha256(source / "norm_metadata.json") != self.paper_audit_sha256
        ):
            raise ValueError("Paper assets changed during training")
        if paper.sha256(Path(destination) / "norm_stats.json") != metadata["norm_stats_sha256"]:
            raise ValueError("Checkpoint normalization does not match paper assets")
        for name in (paper.BUNDLE, paper.MANIFEST, "norm_metadata.json"):
            shutil.copyfile(source / name, Path(destination) / name)


@dataclass(frozen=True)
class PaperSharpaData(cfg.LeRobotFr3SharpaMultimodalDataConfig):
    paper_mode: str = "global_quantile"
    paper_beta: float = 0.0
    paper_std_floor: float = 1e-4
    correlation_max_windows: int = 20000
    correlation_seed: int = 42
    extra_delta_transform: bool = True

    def __post_init__(self):
        if (
            self.paper_mode not in paper.MODES
            or not 0 <= self.paper_beta < 1
            or not np.isfinite(self.paper_std_floor)
            or self.paper_std_floor <= 0
            or self.correlation_max_windows < 2
            or not self.lerobot_datasets
            or any(spec.weight != 1.0 for spec in self.lerobot_datasets)
            or not self.extra_delta_transform
            or self.action_hz != 30
            or getattr(self.base_config, "task_view", None) is not None
        ):
            raise ValueError(
                "Paper experiment requires 30Hz Sharpa Delta datasets with uniform complete-window sampling"
            )

    def paper_contract(self):
        return {
            "version": paper.VERSION,
            "mode": self.paper_mode,
            "beta": self.paper_beta,
            "std_floor": self.paper_std_floor,
            "windows": "complete_only",
            "dataset_sampling": "uniform_complete_windows",
            "correlation_max_windows": self.correlation_max_windows,
            "correlation_seed": self.correlation_seed,
        }

    def asset_directory(self, assets_dirs):
        return Path(self.assets.assets_dir or assets_dirs) / (self.assets.asset_id or self.repo_id)

    def validate_normalization(self, config, result):
        from openpi.models.pi0_delta_paper import PaperPi0Config
        from openpi.training import norm_provenance

        directory = self.asset_directory(config.assets_dirs)
        if not isinstance(config.model, PaperPi0Config) or config.sampling_mode != "epoch_shuffle":
            raise ValueError("Paper data requires its matching model and epoch_shuffle")
        if Path(config.model.paper_asset_dir).resolve() != directory.resolve():
            raise ValueError("Paper model/data asset paths differ; rebuild experiment after path changes")
        metadata, _ = paper.load_bundle(
            directory,
            mode=self.paper_mode,
            beta=self.paper_beta,
            horizon=config.model.action_horizon,
            std_floor=self.paper_std_floor,
            expected_sha256=config.model.paper_asset_sha256,
        )
        if (
            config.model.paper_beta != self.paper_beta
            or config.model.paper_mode != self.paper_mode
            or config.model.paper_std_floor != self.paper_std_floor
            or not config.model.paper_asset_sha256
        ):
            raise ValueError(
                "Paper model/data algorithm mismatch or missing digest; reload configuration after statistics"
            )
        stats = normalize.load(directory)
        for key, shape in (("state", (58,)), ("actions", (config.model.action_horizon, 58))):
            for field in ("mean", "std", "q01", "q99"):
                value = np.asarray(getattr(stats[key], field))
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid paper normalization {key}.{field}")
            if np.any(stats[key].std < 0) or np.any(stats[key].q99 < stats[key].q01):
                raise ValueError("Invalid paper normalization range")
        digest = paper.sha256(directory / "norm_stats.json")
        result.update(
            norm_stats_path=str(directory / "norm_stats.json"),
            norm_stats_sha256=digest,
            paper_bundle_sha256=metadata["bundle_sha256"],
            paper_algorithm=self.paper_contract(),
            training_windows=metadata["training_windows"],
            norm_provenance=norm_provenance.validate_metadata(config, directory, stats_sha256=digest),
        )

    def create(self, assets_dirs, model_config):
        directory = self.asset_directory(assets_dirs)
        metadata, arrays = paper.load_bundle(
            directory,
            mode=self.paper_mode,
            beta=self.paper_beta,
            horizon=model_config.action_horizon,
            std_floor=self.paper_std_floor,
            expected_sha256=getattr(model_config, "paper_asset_sha256", None),
        )
        base = super().create(assets_dirs, model_config)
        stats = base.norm_stats["actions"]
        bridge = paper.ArmNormalization(
            self.paper_mode,
            arrays["arm_mean"],
            arrays["arm_std"],
            stats.q01,
            stats.q99,
            self.paper_std_floor,
            metadata["bundle_sha256"],
        )
        group = transforms.Group(
            inputs=(bridge, *base.model_transforms.inputs),
            outputs=(*base.model_transforms.outputs, replace(bridge, inverse=True)),
        )
        return replace(
            PaperDataView(
                **{f.name: getattr(base, f.name) for f in fields(cfg.DataConfig)},
                paper_asset_dir=str(directory),
                paper_bundle_sha256=metadata["bundle_sha256"],
                paper_manifest_sha256=paper.sha256(directory / paper.MANIFEST),
                paper_audit_sha256=paper.sha256(directory / "norm_metadata.json"),
            ),
            model_transforms=group,
        )

    def compute_normalization(self, config, *, batch_size):
        from openpi.training.delta_paper_stats import compute

        return compute(config, batch_size=batch_size)
