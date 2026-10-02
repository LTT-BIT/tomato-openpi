"""补传终端中真实保留的 1001 指标。不重建缺失的训练曲线。"""

import json
from pathlib import Path

import wandb


def main():
    path = Path(__file__).resolve().parents[1] / "docs/1001_recovered_metrics.json"
    data = json.loads(path.read_text())
    with wandb.init(
        project="openpi",
        id="backfill-1001-subtasks-20261002",
        resume="allow",
        name="tomato_1001_subtasks_lora_30hz_partial_backfill",
        mode="online",
        tags=["partial-history", "backfill", "1001-subtasks"],
        notes="2026-10-01 训练当时未启用 W&B。仅补传终端保存的 19980、19990 步。不是完整曲线。",
        config={
            "training_date": data["training_date"],
            "history_complete": False,
            "metric_source": data["source"],
            "precision": data["precision"],
            "config_name": "pi05_fr3_wuji_1001_subtasks_30hz",
        },
    ) as run:
        for row in data["metrics"]:
            if row["step"] >= run.step:
                run.log({k: v for k, v in row.items() if k != "step"}, step=row["step"])
        run.summary["history_complete"] = False
        run.summary["recovered_metric_points"] = len(data["metrics"])
        run.summary["checkpoint_step"] = 19999
        artifact = wandb.Artifact("1001-recovered-terminal-metrics", type="training-log")
        artifact.add_file(str(path))
        run.log_artifact(artifact)
        print(f"Uploaded partial history: {run.url}", flush=True)


if __name__ == "__main__":
    main()
