"""Check or explicitly start the fixed normal93 experiment; B is the default."""

import argparse
import csv
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
from zoneinfo import ZoneInfo

MODULES = {
    "b": "training_configs.experiments.tomato_sharpa.subtasks.normal93_b_stage",
    "a": "training_configs.experiments.tomato_sharpa.subtasks.normal93_a_goal",
}
ROOT = Path(__file__).resolve().parents[1]


def gpu_inventory():
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    devices = []
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        index, uuid, name, used, total, utilization = [value.strip() for value in row]
        devices.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "used_mib": int(used),
                "total_mib": int(total),
                "utilization": int(utilization),
                "compute_pids": [],
            }
        )
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
    )
    for uuid, pid in csv.reader(raw.splitlines(), skipinitialspace=True):
        for device in devices:
            if device["uuid"] == uuid.strip():
                device["compute_pids"].append(int(pid.strip()))
    return devices


def validate_gpus(devices, selected, *, fsdp_devices=8, batch_size=64):
    if len(set(selected)) != len(selected) or len(selected) != fsdp_devices or batch_size % len(selected):
        raise ValueError(f"Select exactly {fsdp_devices} distinct GPUs for the unchanged batch/FSDP configuration")
    indexed = {device["index"]: device for device in devices}
    if set(selected) - indexed.keys():
        raise ValueError("Requested GPU indices are unavailable")
    busy = [
        i for i in selected if indexed[i]["compute_pids"] or indexed[i]["used_mib"] > 1024 or indexed[i]["utilization"]
    ]
    if busy:
        raise RuntimeError(
            f"GPU {','.join(busy)} occupied; training was not started. Existing processes were not changed."
        )


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
    if inputs["training_episodes"] != 93 or inputs["training_frames"] != 410456:
        raise ValueError("Training must include all 93 recordings / 410456 frames")
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
    args = parser.parse_args()
    selected = [part.strip() for part in args.gpus.split(",")]
    devices = gpu_inventory()
    if args.action == "gpus":
        print(json.dumps(devices, indent=2))
        return
    if args.action == "start":
        validate_gpus(devices, selected)
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
    path = ROOT / "logs/normal93_preflight" / f"{stamp}_{args.variant}.json"
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
            raise RuntimeError("Another normal93 launcher already holds the GPU lock") from error
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
