"""Finalize verified P3 provenance; no model execution or optimizer updates."""

import argparse
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--wandb-synced", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.run / "training/result.json").read_text())
    final_path = args.run / "training_recovery_01/result.json"
    final = json.loads(final_path.read_text())
    checks = []
    for label in ("step250_fixed", "step250_uniform"):
        audit = json.loads((args.run / label / "audit.json").read_text())
        assert audit["optimizer_step"] == 250
        assert audit["checkpoint_files_unchanged"]
        assert audit["parameter_sha256"] == final["checkpoint_pre_save_hashes"]["parameters"]
        assert audit["optimizer_sha256"] == final["checkpoint_pre_save_hashes"]["optimizer"]
        assert audit["adam_state_roundtrip"]
        checks.append(audit)
    for dirname, r in [("training", original), ("training_recovery_01", final)]:
        journal = [json.loads(x) for x in (args.run / dirname / "update_journal.jsonl").read_text().splitlines()]
        completed = [x["global_step"] for x in journal if x["event"] == "completed"]
        assert completed == list(range(101, 251))
        assert len(r["train"]) == 150
    assert [(x["id"], x["start"]) for x in original["train"]] == [(x["id"], x["start"]) for x in final["train"]]
    final["checkpoint"].update(
        restore_pending=False,
        parameter_roundtrip=True,
        adam_state_roundtrip=True,
        restored_scalar_counts=checks[0]["optimizer_scalar_counts"],
        verified_evaluation_windows=[813, 505],
    )
    final["execution_accounting"] = {
        "original_p3_updates": 150,
        "authorized_recovery_p3_updates": 150,
        "total_p3_compute_updates": 300,
        "effective_checkpoint_step": 250,
        "updates_on_final_trajectory": 150,
        "source_step": 100,
        "step251_executed": False,
        "recovery_explicitly_authorized_by_user": True,
        "maximum_replay_loss_difference": max(
            abs(a["loss"] - b["loss"]) for a, b in zip(original["train"], final["train"], strict=True)
        ),
    }
    if args.wandb_synced:
        final["wandb"].update(
            sync_status="synced", url=f"https://wandb.ai/mengxiangzhang-bit/openpi-flash/runs/{final['wandb']['id']}"
        )
    final_path.write_text(json.dumps(final, indent=2))
    (args.output / "execution_accounting.json").write_text(json.dumps(final["execution_accounting"], indent=2))
    shutil.copyfile(args.run / "training/result.json", args.output / "first_interrupted_training.json")
    for name in ("launch_audit.json", "recovery_launch_audit.json"):
        shutil.copyfile(args.run / name, args.output / name)
    print(json.dumps(final["checkpoint"], indent=2))


if __name__ == "__main__":
    main()
