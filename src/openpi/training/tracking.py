"""Track each training attempt separately, including checkpoint resumes offline."""

from contextlib import contextmanager
import dataclasses
from datetime import UTC
from datetime import datetime
import logging
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING

import wandb

from openpi.shared.local_storage import atomic_write_json

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig


@contextmanager
def session(config: "TrainConfig", *, resuming: bool, start_step: int):
    """A new W&B run per attempt avoids unsupported offline resume and replay collisions."""
    if not config.wandb_enabled:
        wandb.init(mode="disabled")
        try:
            yield None
        finally:
            wandb.finish()
        return
    mode = os.environ.get("WANDB_MODE", "online")
    if mode not in ("online", "offline"):
        raise ValueError("WANDB_MODE must be online or offline")
    checkpoint = Path(config.checkpoint_dir)
    anchor = checkpoint / "wandb_id.txt"
    run_id = wandb.util.generate_id()
    root_id = anchor.read_text().strip() if resuming else run_id
    if not re.fullmatch(r"[A-Za-z0-9_-]+", root_id):
        raise ValueError("Invalid or empty checkpoint wandb_id.txt")
    # Cloud resume cannot restore training state or append offline history. Each
    # attempt records its own history; checkpoints alone determine start_step.
    os.environ.pop("WANDB_RESUME", None)
    run = wandb.init(
        id=run_id,
        group=root_id,
        name=config.exp_name,
        project=config.project_name,
        mode=mode,
        config={
            **dataclasses.asdict(config),
            "logging_session": {"root_run_id": root_id, "checkpoint_resume": resuming, "start_step": start_step},
        },
    )
    record = {
        "run_id": run.id,
        "root_run_id": root_id,
        "project": config.project_name,
        "entity": run.entity,
        "mode": mode,
        "checkpoint_resume": resuming,
        "start_step": start_step,
        "launch_id": os.environ.get("PI05_LAUNCH_ID"),
        "started_at": datetime.now(UTC).isoformat(),
        "run_directory": str(Path(run.dir).parent.resolve()),
        "status": "running",
    }
    record_path = checkpoint / "wandb_sessions" / f"{run.id}.json"
    latest_path = checkpoint / "wandb_latest_session.json"

    def save_record():
        atomic_write_json(record_path, record)
        atomic_write_json(latest_path, record)

    exit_code = 1
    try:
        if not resuming:
            anchor.write_text(root_id + "\n")
        save_record()
        logging.info("W&B session %s, group %s, checkpoint start step %d", run.id, root_id, start_step)
        if mode == "offline":
            logging.info("Sync this offline segment later: .venv/bin/wandb sync %s", record["run_directory"])
        yield record
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        record["status"] = "training_completed"
        exit_code = 0
    finally:
        try:
            wandb.finish(exit_code=exit_code)
        finally:
            record["finished_at"] = datetime.now(UTC).isoformat()
            save_record()
