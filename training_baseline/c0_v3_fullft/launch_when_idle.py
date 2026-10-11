"""Finish the official base download, then launch this full fine-tuning run on eight idle GPUs."""

import argparse
from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def record(path: Path, state: str, **details: object) -> None:
    payload = {"time_utc": datetime.now(UTC).isoformat(), "state": state, **details}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def download_complete(runtime: Path) -> bool:
    listing = json.loads((runtime / "pi05-base-objects.json").read_text())
    items = listing["items"]
    first = max(items, key=lambda item: int(item["size"]))
    staging = runtime / "openpi-cache/openpi-assets/checkpoints/pi05_base/params.partial"
    for item in items:
        relative = Path(item["name"].removeprefix("checkpoints/pi05_base/params/"))
        path = (runtime / "base-download-probe" / relative.name) if item is first else staging / relative
        if path.with_name(path.name + ".aria2").exists():
            return False
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            return False
    return True


def idle_gpus() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [tuple(int(value.strip()) for value in line.split(",")) for line in result.stdout.splitlines()]
    return len(rows) == 8 and [row[0] for row in rows] == list(range(8)) and all(
        memory < 1024 and utilization <= 5 for _, memory, utilization in rows
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-pid", type=int, action="append", required=True)
    parser.add_argument("--experiment", default="tomato_all_full_fullft_30hz_s42_run01")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    runtime = Path(os.environ.get("PI05_RUNTIME_DIR", "/home/zhiwei/pi05-runtime"))
    repository = Path(__file__).resolve().parents[2]
    status = runtime / f"{args.experiment}-launch-status.json"
    output = runtime / f"{args.experiment}-launcher.log"
    checkpoint = runtime / "checkpoints/pi05_fr3_wuji_fulltask_fullft_30hz" / args.experiment
    destination = runtime / "openpi-cache/openpi-assets/checkpoints/pi05_base/params"
    record(status, "waiting_for_weights", download_pids=args.download_pid)
    while not destination.is_dir():
        if download_complete(runtime):
            subprocess.run(
                [sys.executable, "-m", "training_baseline.c0_v3_fullft.verify_base_weights"],
                cwd=repository,
                check=True,
            )
            break
        if not any(process_alive(pid) for pid in args.download_pid):
            raise RuntimeError("All base download processes stopped before completing the checkpoint")
        time.sleep(args.interval)

    record(status, "waiting_for_eight_idle_gpus", base_checkpoint=str(destination))
    consecutive_idle = 0
    while consecutive_idle < 3:
        disk_free = shutil.disk_usage(runtime).free
        idle = idle_gpus() and disk_free >= 300 * 1024**3
        consecutive_idle = consecutive_idle + 1 if idle else 0
        if consecutive_idle < 3:
            time.sleep(args.interval)

    if checkpoint.exists():
        raise FileExistsError(f"Experiment checkpoint directory already exists: {checkpoint}")
    environment = dict(os.environ, WANDB_MODE="offline", EXP=args.experiment)
    record(status, "launching", checkpoint=str(checkpoint), wandb_mode="offline")
    with output.open("w") as stream:
        process = subprocess.Popen(
            ["bash", "training_baseline/c0_v3_fullft/train_all_full.sh"],
            cwd=repository,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record(status, "launched", pid=process.pid, launcher_log=str(output), checkpoint=str(checkpoint))
    while True:
        if "Step 0:" in output.read_text():
            record(status, "first_step_complete", pid=process.pid, launcher_log=str(output), checkpoint=str(checkpoint))
            return
        result = process.poll()
        if result is not None:
            raise RuntimeError(f"Training exited before the first step (exit code {result}); see {output}")
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        arguments = argparse.ArgumentParser(add_help=False)
        arguments.add_argument("--experiment", default="tomato_all_full_fullft_30hz_s42_run01")
        experiment, _ = arguments.parse_known_args()
        runtime_root = Path(os.environ.get("PI05_RUNTIME_DIR", "/home/zhiwei/pi05-runtime"))
        record(runtime_root / f"{experiment.experiment}-launch-status.json", "failed", error=str(error))
        raise
