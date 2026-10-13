"""Load an experiment module, preflight local inputs, and log the JAX trainer."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime
import fcntl
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

import numpy as np
import tyro

from openpi.shared import normalize
from openpi.training import config as cfg
from openpi.training import norm_provenance
from openpi.training import reproducibility
from training_configs import common


def load_experiment(module_name: str) -> cfg.TrainConfig:
    if not module_name.startswith("training_configs.experiments."):
        raise ValueError("Experiment must be a module under training_configs.experiments")
    config = importlib.import_module(module_name).build_config()
    if not isinstance(config, cfg.TrainConfig):
        raise TypeError("Experiment build_config() must return TrainConfig")
    return replace(config, policy_metadata={**(config.policy_metadata or {}), "experiment_module": module_name})


def resolve_experiment(module_name: str, overrides: list[str]) -> cfg.TrainConfig:
    config = load_experiment(module_name)
    resolved = tyro.cli(cfg.TrainConfig, default=config, args=overrides)
    # A new model recipe must also regenerate its freeze filter and adapter contract.
    for field in ("paligemma_variant", "action_expert_variant", "pi05", "discrete_state_input"):
        if getattr(resolved.model, field) != getattr(config.model, field):
            raise ValueError(f"Change model.{field} in a versioned recipe, not a CLI override")
    return resolved


def validate_config(config: cfg.TrainConfig) -> None:
    for label, value in (("name", config.name), ("exp_name", config.exp_name)):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
            raise ValueError(f"{label} must be a nonempty directory-safe identifier")
    if config.overwrite:
        raise ValueError("Use a new exp_name instead of overwriting a shared experiment")
    if not config.wandb_enabled:
        raise ValueError("Training must log to W&B; use --wandb-mode offline without network")
    if config.sampling_mode != "epoch_shuffle" or len(config.data.lerobot_datasets) != 1:
        raise ValueError("The v1 baseline workflow requires epoch_shuffle and one explicit local dataset")
    if config.batch_size <= 0 or config.num_train_steps <= 0 or config.num_workers < 0 or config.fsdp_devices <= 0:
        raise ValueError("batch_size, num_train_steps, fsdp_devices must be positive; num_workers must be nonnegative")
    if any(seed is None for seed in (config.model_seed, config.data_seed, config.augmentation_seed)):
        raise ValueError("Record all independent model/data/augmentation seeds")
    metadata = config.policy_metadata or {}
    if config.model.action_dim != metadata.get("physical_action_dim"):
        raise ValueError("Model action_dim differs from the embodiment adapter; define a new embodiment")
    if not config.data.base_config or not config.data.base_config.prompt_from_task:
        raise ValueError("Read English prompts from dataset task labels")
    spec = config.data.lerobot_datasets[0]
    if not spec.root or spec.repo_id != config.data.repo_id or spec.repo_id != metadata.get("dataset_repo_id"):
        raise ValueError("Dataset identity differs from the experiment; create a new dataset configuration")


def preflight(config: cfg.TrainConfig, *, require_norm: bool = True) -> dict:
    """Check metadata/statistics without constructing models or downloading tokenizers."""
    validate_config(config)
    metadata = config.policy_metadata
    root = Path(config.data.lerobot_datasets[0].root).expanduser().resolve()
    required = [root / "meta" / name for name in ("info.json", "episodes.jsonl", "tasks.jsonl")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing prepared LeRobot data: {path}")
    info = json.loads(required[0].read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Prepared data must use the pinned LeRobot v2.1 schema")
    if info.get("fps") != config.data.action_hz:
        raise ValueError("Dataset fps differs from the configured action sampling frequency")
    features = info["features"]
    if features["action"]["shape"] != [config.model.action_dim]:
        raise ValueError("Dataset action shape differs from the embodiment")
    if features["observation.state"]["shape"] not in [[dim] for dim in metadata["raw_state_dims"]]:
        raise ValueError("Dataset raw state shape is not supported by the embodiment")
    if any(key not in features for key in metadata["camera_keys"]):
        raise ValueError("Dataset camera roles differ from the embodiment")
    if info.get("total_frames", 0) <= 0 or info.get("total_episodes", 0) <= 0:
        raise ValueError("Dataset must contain frames and episodes")
    if not (root / "data").is_dir() or not (root / "videos").is_dir():
        raise FileNotFoundError("Prepared dataset must contain data/ and videos/")
    prompts = [json.loads(line)["task"] for line in required[2].read_text().splitlines() if line.strip()]
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("Dataset task prompts must be nonempty English strings")
    if any(not prompt.isascii() or not re.search(r"[A-Za-z]", prompt) for prompt in prompts):
        raise ValueError("This workflow requires English ASCII prompts in tasks.jsonl")
    norm_dir = Path(config.data.assets.assets_dir or config.assets_dirs) / config.data.assets.asset_id
    result = {
        "dataset_root": str(root),
        "dataset_repo_id": config.data.repo_id,
        "source_datasets": metadata["source_datasets"],
        "frames": info["total_frames"],
        "episodes": info["total_episodes"],
        "prompts": prompts,
        "norm_stats_path": str(norm_dir / "norm_stats.json"),
    }
    if require_norm:
        norm_path = norm_dir / "norm_stats.json"
        digest = reproducibility.file_sha256(norm_path)
        expected = metadata.get("expected_norm_sha256")
        if expected and digest != expected:
            raise ValueError(
                "Normalization digest differs from the recorded data version; create a new dataset version"
            )
        stats = normalize.load(norm_dir)
        audit_digest = hashlib.sha256(reproducibility.serialize_config(stats)).hexdigest()
        expected_audit = metadata.get("expected_norm_audit_sha256")
        if expected_audit and audit_digest != expected_audit:
            raise ValueError("Normalization audit digest differs from the recorded data version")
        for key in ("state", "actions"):
            value = stats[key]
            for field in ("mean", "std", "q01", "q99"):
                array = getattr(value, field)
                if array is None or array.shape != (config.model.action_dim,) or not np.isfinite(array).all():
                    raise ValueError(f"Invalid normalization {key}.{field} shape or values")
            if np.any(value.std < 0) or np.any(value.q99 < value.q01):
                raise ValueError(f"Invalid normalization range for {key}")
        result["norm_provenance"] = norm_provenance.validate_metadata(config, norm_dir, stats_sha256=digest)
        result["norm_stats_sha256"] = digest
        result["norm_audit_sha256"] = audit_digest
    return result


@contextmanager
def launch_lock(config: cfg.TrainConfig):
    """Reject concurrent writers to the same checkpoint directory on this host."""
    directory = Path(config.checkpoint_base_dir).resolve() / ".training-locks"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(config.checkpoint_dir).encode()).hexdigest()
    with (directory / f"{key}.lock").open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Experiment is already running: {config.checkpoint_dir}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile = logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def launch(config: cfg.TrainConfig, *, wandb_mode: str, inputs: dict) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    directory = common.repository_root() / "logs" / config.name / config.exp_name
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"{stamp}_{os.getpid()}"
    manifest_path = prefix.with_suffix(".json")
    os.environ["WANDB_MODE"] = wandb_mode
    os.environ["WANDB_DIR"] = str(common.repository_root())
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    manifest = {
        "status": "starting",
        "started_at": stamp,
        "git_commit": revision,
        "wandb_mode": wandb_mode,
        "inputs": inputs,
        "config": json.loads(reproducibility.serialize_config(config)),
        "checkpoint_dir": str(config.checkpoint_dir),
        "log": str(prefix.with_suffix(".log")),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with prefix.with_suffix(".log").open("w", buffering=1) as stream:
        stdout, stderr = Tee(sys.stdout, stream), Tee(sys.stderr, stream)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            logging.basicConfig(level=logging.INFO, stream=stderr, force=True)
            print(f"Run manifest: {manifest_path}\nW&B: {wandb_mode}\nCheckpoint: {config.checkpoint_dir}")
            try:
                from scripts import train

                train.main(config)
            except BaseException as error:
                import traceback

                traceback.print_exc()
                manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
                raise
            else:
                manifest["status"] = "training_completed"
            finally:
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("experiment", help="training_configs.experiments.<owner>.<experiment>")
    parser.add_argument("--dry-run", action="store_true", help="Print final config without reading data or launching")
    parser.add_argument("--check", action="store_true", help="Validate prepared data and stats, then exit")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default=os.environ.get("WANDB_MODE", "online"))
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments in (["--help"], ["-h"]):
        parser.print_help()
        return
    args, overrides = parser.parse_known_args(arguments)
    config = resolve_experiment(args.experiment, overrides)
    validate_config(config)
    if args.dry_run:
        print(reproducibility.serialize_config(config).decode())
        return
    if args.wandb_mode not in ("online", "offline"):
        raise ValueError("WANDB_MODE must be online or offline; select offline explicitly without network")
    inputs = preflight(config)
    if args.check:
        print(json.dumps(inputs, ensure_ascii=False, indent=2))
        return
    if config.resume and not config.checkpoint_dir.exists():
        raise FileNotFoundError("Cannot resume a missing experiment; remove --resume for a new run")
    with launch_lock(config):
        launch(config, wandb_mode=args.wandb_mode, inputs=inputs)


if __name__ == "__main__":
    main()
