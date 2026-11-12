"""Check or explicitly start the complete100 four-view offline experiment; B is the default."""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
from zoneinfo import ZoneInfo

from scripts.tomato_ab_run import gpu_inventory
from scripts.tomato_ab_run import validate_gpus

MODULES = {
    "b": "training_configs.experiments.tomato_sharpa.subtasks.complete100_b_crop",
    "a": "training_configs.experiments.tomato_sharpa.subtasks.complete100_a_crop",
}
ROOT = Path(__file__).resolve().parents[1]


DEFAULT_BUNDLE = '/home/mengxiang/workspace/data/sharpa/no-block-more/complete100_rgb_crop_offline_r11'
PREPARATION = ROOT / 'training_configs/experiments/tomato_sharpa/subtasks/records/complete100_server28_preparation_20261112_r11.json'


def configure_inputs(bundle, report):
    os.environ['PI05_TOMATO100_CROP_BUNDLE'] = str(Path(bundle).expanduser().resolve())
    os.environ['PI05_TOMATO100_CROP_A_BUNDLE'] = os.environ['PI05_TOMATO100_CROP_BUNDLE']
    os.environ['PI05_TOMATO100_PREPARATION_REPORT'] = str(Path(report).expanduser().resolve())


def cpu_preflight(variant, exp_name=None):
    from dataclasses import replace

    from openpi.training.reproducibility import serialize_config
    from scripts.run_training import load_experiment
    from scripts.run_training import preflight
    from scripts.run_training import validate_config

    config = load_experiment(MODULES[variant])
    if exp_name:
        config = replace(config, exp_name=exp_name)
    validate_config(config)
    if config.resume or config.overwrite or config.checkpoint_dir.exists():
        raise ValueError("This entry prepares a fresh experiment; choose an unused exp-name")
    inputs = preflight(config)
    if inputs["training_episodes"] != 100 or inputs["training_frames"] != 425556:
        raise ValueError("Training must include all 100 recordings / 425556 frames")
    for directory in (Path(config.checkpoint_base_dir), ROOT / "logs", ROOT / "wandb"):
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory):
            pass
    return config, {
        "variant": variant,
        "config": json.loads(serialize_config(config)),
        "inputs": inputs,
        "checkpoint_dir": str(config.checkpoint_dir),
        "checkpoint_exists": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("check", "start", "gpus"), default="check")
    parser.add_argument("--variant", choices=("b", "a"), default="b")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--exp-name")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--preparation-report", default=str(PREPARATION))
    args = parser.parse_args()
    configure_inputs(args.bundle, args.preparation_report)
    selected = [part.strip() for part in args.gpus.split(",")]
    devices = gpu_inventory()
    if args.action == "gpus":
        print(json.dumps(devices, indent=2))
        return
    if args.action == "start":
        validate_gpus(devices, selected, batch_size=512)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config, report = cpu_preflight(args.variant, args.exp_name)
    devices = gpu_inventory()
    try:
        validate_gpus(devices, selected, fsdp_devices=config.fsdp_devices, batch_size=config.batch_size)
        gpu_status = "available_at_check_time"
    except (ValueError, RuntimeError) as error:
        gpu_status = str(error)
    report.update(
        data_and_statistics="passed",
        gpu_status=gpu_status,
        gpus=devices,
        wandb_mode=args.wandb_mode,
        training_started=False,
        gpu_capacity_verified=False,
    )
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    path = ROOT / "logs/complete100_preflight" / f"{stamp}_{args.variant}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(f"Data/statistics: passed; GPU: {gpu_status}; snapshot: {path}", flush=True)
    if args.action == "check":
        return
    lock_path = ROOT / ".training-locks/normal93_gpus.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another tomato launcher already holds the GPU lock") from error
        validate_gpus(gpu_inventory(), selected, fsdp_devices=config.fsdp_devices, batch_size=config.batch_size)
        from scripts.training_environment import cuda_environment

        env = cuda_environment(selected)
        command = [
            "bash",
            str(ROOT / "scripts/python.sh"),
            "-u",
            "-m",
            "scripts.run_training",
            MODULES[args.variant],
            "--exp-name",
            config.exp_name,
            "--wandb-mode",
            args.wandb_mode,
        ]
        raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
