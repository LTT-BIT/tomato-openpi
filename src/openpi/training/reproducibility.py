"""Local sampling artifacts and resume guards for opt-in reproducible training.

The plan records observation-frame coverage. Action windows can overlap and are
not counted as additional distinct observations. Dataset files are hashed once
at each launch; paths are relative so relocating identical data is supported.
"""

from __future__ import annotations

import csv
import dataclasses
import enum
import hashlib
import json
import logging
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np

from openpi.training import sampling

if TYPE_CHECKING:
    from openpi.training import config as _config


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: getattr(value, field.name) for field in dataclasses.fields(value)},
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, re.Pattern):
        return {"pattern": value.pattern, "flags": value.flags}
    if isinstance(value, type) or (callable(value) and hasattr(value, "__qualname__")):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "serialized_model_proto"):
        return {"sentencepiece_sha256": hashlib.sha256(value.serialized_model_proto()).hexdigest()}
    if hasattr(value, "__dict__"):
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "state": vars(value)}
    raise TypeError(f"Cannot fingerprint {type(value).__qualname__}; add an explicit stable serialization")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, default=_json_default, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")


def serialize_config(value: Any) -> bytes:
    """Serialize an effective configuration without building model/tokenizer assets."""
    return _json_bytes(value)


def file_sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _inventory(root: pathlib.Path, files: list[pathlib.Path]) -> dict[str, str]:
    result = {}
    for index, path in enumerate(sorted(files)):
        result[path.relative_to(root).as_posix()] = file_sha256(path)
        if (index + 1) % 100 == 0:
            logging.info("Fingerprinting %s: %d/%d files", root, index + 1, len(files))
    return result


def _code_inventory() -> dict[str, str]:
    root = pathlib.Path(__file__).resolve().parents[3]
    files = [root / "uv.lock", root / "pyproject.toml"]
    for directory in (root / "src/openpi", root / "training_configs", root / "scripts"):
        files.extend(
            path
            for path in directory.rglob("*.py")
            if not path.name.endswith("_test.py") and not path.name.startswith("test_")
        )
    return _inventory(root, files)


def _source_identity(data_config: _config.DataConfig, source: Any) -> tuple[dict, np.ndarray]:
    if data_config.repo_id == "fake":
        size = len(source)
        return {"repo_id": "fake", "frames": size}, np.column_stack((np.zeros(size, dtype=np.int64), np.arange(size)))

    spec = data_config.lerobot_datasets[0]
    root = pathlib.Path(spec.root)
    required = [root / "meta" / name for name in ("info.json", "episodes.jsonl", "tasks.jsonl")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Sampling audit requires {path}")
    files = [path for folder in ("meta", "data", "videos") for path in (root / folder).rglob("*") if path.is_file()]
    logging.info("Hashing %d dataset files (including videos) under %s", len(files), root)
    # Read IDs from the loaded table, rather than assuming metadata/file ordering.
    columns = source.hf_dataset.select_columns(["episode_index", "frame_index"]).with_format("numpy")[:]
    frame_ids = np.column_stack((columns["episode_index"], columns["frame_index"])).astype(np.int64)
    if len(frame_ids) != len(source):
        raise ValueError("Loaded frame IDs do not match dataset length")
    info = json.loads(required[0].read_text())
    if info["total_frames"] != len(source):
        raise ValueError("Dataset metadata total_frames does not match the loaded table")
    return {
        "repo_id": spec.repo_id,
        "frames": len(source),
        "episodes": info["total_episodes"],
        "files": _inventory(root, files),
        "tasks": [json.loads(line) for line in required[2].read_text().splitlines() if line.strip()],
        "frame_ids_sha256": hashlib.sha256(frame_ids.astype("<i8").tobytes()).hexdigest(),
    }, frame_ids


def _training_contract(config: _config.TrainConfig, data_config: _config.DataConfig) -> dict:
    # Destination paths and logging do not define the training trajectory. Data
    # and normalization identity are checked separately by content, not by path.
    values = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    for key in (
        "name",
        "project_name",
        "exp_name",
        "assets_base_dir",
        "checkpoint_base_dir",
        "resume",
        "overwrite",
        "wandb_enabled",
        "log_interval",
        "save_interval",
        "keep_period",
        "policy_metadata",
        "data",
    ):
        values.pop(key, None)
    values["data_transforms"] = {
        "repack": data_config.repack_transforms,
        "data": data_config.data_transforms,
        "model": data_config.model_transforms,
        "action_hz": data_config.action_hz,
        "action_sequence_keys": data_config.action_sequence_keys,
        "prompt_from_task": data_config.prompt_from_task,
        "use_quantile_norm": data_config.use_quantile_norm,
        "video_backend": data_config.video_backend,
    }
    return json.loads(_json_bytes(values))


def prepare_run(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    source: Any,
    indices: np.ndarray,
    *,
    resuming: bool,
) -> dict:
    """Persist or validate the exact finite plan before creating any iterator.

    A resumed run must retain its original budget, seeds, data, transforms, code
    and normalization. A changed experiment gets a new experiment directory.
    """
    dataset, frame_ids = _source_identity(data_config, source)
    norm_bytes = _json_bytes(data_config.norm_stats)
    dataset_bytes = _json_bytes(dataset)
    contract = {
        "sampler_version": sampling.SAMPLER_VERSION,
        "training": _training_contract(config, data_config),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "norm_sha256": hashlib.sha256(norm_bytes).hexdigest(),
        "plan_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "code": _code_inventory(),
    }
    unique = len(np.unique(indices))
    summary = {
        "sampler_version": sampling.SAMPLER_VERSION,
        "total_samples": len(indices),
        "dataset_frames": len(source),
        "planned_unique_frames": unique,
        "planned_coverage": unique / len(source),
        "steps_for_first_pass": (len(source) + config.batch_size - 1) // config.batch_size,
        "dataset_sha256": contract["dataset_sha256"],
        "norm_sha256": contract["norm_sha256"],
        "plan_indices_sha256": contract["plan_indices_sha256"],
    }
    directory = pathlib.Path(config.checkpoint_dir) / "reproducibility"
    if directory.exists():
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["contract"] != contract:
            changed = [key for key in contract if manifest["contract"].get(key) != contract[key]]
            raise ValueError(
                f"Reproducibility contract changed: {changed}. Use a new exp_name for a changed experiment."
            )
        for name, expected in manifest["artifacts"].items():
            if file_sha256(directory / name) != expected:
                raise ValueError(f"Reproducibility artifact changed: {name}")
        logging.info("Verified existing sampling plan: %s", directory)
        return summary
    if resuming:
        raise ValueError("Missing reproducibility artifacts: cannot resume a legacy run with epoch_shuffle")

    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".reproducibility-", dir=directory.parent))
    try:
        (staging / "dataset_manifest.json").write_bytes(dataset_bytes)
        (staging / "norm_stats.json").write_bytes(norm_bytes)
        (staging / "effective_config.json").write_bytes(_json_bytes(config))
        (staging / "summary.json").write_bytes(_json_bytes(summary))
        np.save(staging / "sample_indices.npy", indices, allow_pickle=False)
        with (staging / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ("position", "step", "batch_offset", "dataset_id", "dataset_index", "episode_index", "frame_index")
            )
            for position, index in enumerate(indices):
                episode, frame = frame_ids[index]
                writer.writerow(
                    (
                        position,
                        position // config.batch_size,
                        position % config.batch_size,
                        dataset["repo_id"],
                        int(index),
                        int(episode),
                        int(frame),
                    )
                )
        repo_root = pathlib.Path(__file__).resolve().parents[3]
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=False
        ).stdout.strip()
        manifest = {
            "contract": contract,
            "git_commit": revision,
            "artifacts": _inventory(staging, list(staging.iterdir())),
        }
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        staging.rename(directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    logging.info("Saved sampling audit to %s: %s", directory, summary)
    if unique < len(source):
        logging.warning(
            "Training budget covers only %d/%d observation frames; increase num_train_steps", unique, len(source)
        )
    return summary
