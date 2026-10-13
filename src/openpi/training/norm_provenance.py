"""Bind normalization to vector data and its pre-normalization transforms."""

import hashlib
import inspect
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from openpi.training import reproducibility

if TYPE_CHECKING:
    from openpi.training import config as cfg

SCHEMA_VERSION = 2


def build_contract(config: "cfg.TrainConfig", *, dataset_specs=None) -> dict:
    """Fingerprint vector inputs, excluding destination paths and training hyperparameters."""
    datasets = []
    specs = dataset_specs if dataset_specs is not None else config.data.lerobot_datasets
    for spec in specs:
        if not spec.root:
            raise ValueError("Normalization provenance requires an explicit dataset root")
        root = Path(spec.root).expanduser().resolve()
        metadata = [root / "meta" / name for name in ("info.json", "episodes.jsonl", "tasks.jsonl")]
        parquet = sorted((root / "data").rglob("*.parquet"))
        if not parquet:
            raise ValueError(f"Normalization provenance requires parquet data under {root / 'data'}")
        info = json.loads(metadata[0].read_text())
        files = {path.relative_to(root).as_posix(): reproducibility.file_sha256(path) for path in (*metadata, *parquet)}
        datasets.append(
            {
                "repo_id": spec.repo_id,
                "frames": info["total_frames"],
                "episodes": info["total_episodes"],
                "fps": info["fps"],
                "files": files,
            }
        )
    if not datasets:
        raise ValueError("Normalization provenance requires a dataset")
    builder = getattr(config.data, "create_vector_transforms", None)
    if builder is None:
        raise ValueError("The data adapter must implement create_vector_transforms for normalization provenance")
    repack, vectors = builder(config.model)
    implementation = {
        f"{type(config.data).__module__}.{type(config.data).__qualname__}": hashlib.sha256(
            inspect.getsource(type(config.data)).encode()
        ).hexdigest(),
    }
    for transform in (*repack.inputs, *vectors.inputs):
        module = inspect.getmodule(type(transform))
        implementation[module.__name__] = reproducibility.file_sha256(Path(module.__file__))
    transform_values = {
        "repack": repack.inputs,
        "vectors": vectors.inputs,
        "implementation": implementation,
    }
    view = getattr(config.data.base_config, "task_view", None)
    if view is not None:
        from openpi.training import task_view

        transform_values["task_view"] = view.identity()
        implementation["openpi.training.task_view"] = reproducibility.file_sha256(Path(task_view.__file__))
    return {
        "repo_id": config.data.repo_id,
        "datasets": datasets,
        "action_dim": config.model.action_dim,
        "action_horizon": config.model.action_horizon,
        "action_hz": config.data.action_hz,
        "action_sequence_keys": list(config.data.action_sequence_keys),
        "vector_transforms": json.loads(reproducibility.serialize_config(transform_values)),
    }


def validate_metadata(config: "cfg.TrainConfig", norm_dir: Path, *, stats_sha256: str) -> str:
    """Validate modern bindings or an explicitly pinned legacy normalization file."""
    path = norm_dir / "norm_metadata.json"
    recorded = json.loads(path.read_text()) if path.exists() else {}
    if path.exists():
        if recorded.get("action_horizon") != config.model.action_horizon:
            raise ValueError("Normalization action window differs from the experiment")
        if "repo_id" in recorded and recorded["repo_id"] != config.data.repo_id:
            raise ValueError("Normalization dataset identity differs from the experiment")
        # Check available descriptive fields even for legacy files.
        specs = config.data.lerobot_datasets
        infos = [json.loads((Path(spec.root).expanduser().resolve() / "meta/info.json").read_text()) for spec in specs]
        for key in ("frames", "episodes"):
            expected = sum(info[f"total_{key}"] for info in infos)
            if key in recorded and recorded[key] != expected:
                raise ValueError(f"Normalization dataset {key} differs from the experiment")
        if "datasets" in recorded:
            expected = [
                {"repo_id": spec.repo_id, "frames": info["total_frames"], "episodes": info["total_episodes"]}
                for spec, info in zip(specs, infos, strict=True)
            ]
            actual = [
                {key: item.get(key) for key in ("repo_id", "frames", "episodes")} for item in recorded["datasets"]
            ]
            if actual != expected:
                raise ValueError("Normalization source datasets differ from the experiment")
    if recorded.get("schema_version") == SCHEMA_VERSION:
        if recorded.get("norm_stats_sha256") != stats_sha256:
            raise ValueError("Normalization file differs from its provenance metadata")
        if recorded.get("contract") != build_contract(config):
            raise ValueError("Normalization data or vector transforms differ; recompute statistics for this experiment")
        return "verified"
    if "schema_version" in recorded:
        raise ValueError("Unsupported normalization provenance schema; recompute statistics")
    policy = config.policy_metadata or {}
    if policy.get("allow_legacy_norm_stats") and (
        policy.get("expected_norm_sha256") or policy.get("expected_norm_audit_sha256")
    ):
        logging.warning("Using explicitly pinned legacy normalization; data/transform binding is unavailable")
        return "legacy_pinned"
    raise ValueError(
        "Normalization provenance is missing; recompute with scripts.compute_experiment_norm. "
        "Known legacy statistics require allow_legacy_norm_stats=True and a pinned digest."
    )
