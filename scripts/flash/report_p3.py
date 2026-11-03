"""Step100/250 paired physical evaluation, label/state diagnostics, and curves."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_p2 import metrics
from report_p2 import write_csv
from report_p25 import GROUPS
from report_p25 import aligned_difference

from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.training.flash_dataset import FlashLeRobotDataset

TRACKED = [
    "left_ring_MCP_FE",
    "left_pinky_MCP_FE",
    "left_middle_MCP_FE",
    "left_index_MCP_FE",
    "right_index_MCP_FE",
    "left_index_DIP",
]


def evaluate_set(plan, before, after, output, dataset):
    assert len(plan["windows"]) == len(before["prediction"]) == len(after["prediction"])
    np.testing.assert_array_equal(before["truth"], after["truth"])
    np.testing.assert_array_equal(before["history"], after["history"])
    np.testing.assert_array_equal(before["oracle"], after["oracle"])
    methods = {
        "step100": (before["prediction"], before["velocity"]),
        "step250": (after["prediction"], after["velocity"]),
        "oracle_k6": (before["oracle"], before["oracle_velocity"]),
    }
    aggregate, joints, windows, sources, roots, critical = {}, {"rows": []}, [], [], [], []
    truth = before["truth"][:, 2:52]
    for name, (q, _) in methods.items():
        e = q[:, 2:52] - truth
        aggregate[name] = {g: metrics(e[..., ids]) for g, ids in GROUPS.items()}
        for j, joint in enumerate(ORDER):
            joints["rows"].append(
                {"method": name, "joint": joint, "group": "arms" if "fr3" in joint else "hands", **metrics(e[..., j])}
            )
        for i, w in enumerate(plan["windows"]):
            for group, ids in GROUPS.items():
                windows.append(
                    {
                        "method": name,
                        "id": w["id"],
                        "start": w["start"],
                        "source_recording_id": w["source_recording_id"],
                        "root": w["root"],
                        "episode_index": w["episode_index"],
                        "group": group,
                        **metrics(e[i][:, ids]),
                    }
                )
            for joint in TRACKED:
                j = ORDER.index(joint)
                peak = int(np.argmax(abs(e[i, :, j])))
                critical.append(
                    {
                        "method": name,
                        "id": w["id"],
                        "start": w["start"],
                        "source_recording_id": w["source_recording_id"],
                        "root": w["root"],
                        "episode_index": w["episode_index"],
                        "joint": joint,
                        "peak_frame": w["start"] + peak,
                        **metrics(e[i, :, j]),
                    }
                )
        for field, rows in [("source_recording_id", sources), ("root", roots)]:
            for value in sorted({w[field] for w in plan["windows"]}):
                indices = [i for i, w in enumerate(plan["windows"]) if w[field] == value]
                for group, ids in GROUPS.items():
                    rows.append(
                        {
                            "method": name,
                            field: value,
                            "windows": len(indices),
                            "group": group,
                            **metrics(e[indices][..., ids]),
                        }
                    )
    continuity, pairs = {}, []
    for name, (q, v) in methods.items():
        continuity[name] = {}
        for delta in (10, 15, 50):
            positions, velocities = [], []
            for anchor in plan["anchors"]:
                a, b = anchor["window_indices"]["0"], anchor["window_indices"][str(delta)]
                pos = aligned_difference(q[a], q[b], delta)
                vel = aligned_difference(v[a], v[b], delta)
                np.testing.assert_array_equal(
                    aligned_difference(before["truth"][a], before["truth"][b], delta), np.zeros_like(pos)
                )
                positions.append(pos)
                velocities.append(vel)
                for group, ids in GROUPS.items():
                    pairs.append(
                        {
                            "method": name,
                            "delta": delta,
                            "id": anchor["id"],
                            "start": anchor["start"],
                            "group": group,
                            "position_mae": metrics(pos[:, ids])["mae"],
                            "velocity_mae": metrics(vel[:, ids])["mae"],
                            "position_max": metrics(pos[:, ids])["max"],
                            "velocity_max": metrics(vel[:, ids])["max"],
                        }
                    )
            if positions:
                continuity[name][str(delta)] = {
                    g: {
                        "position": metrics(np.concatenate(positions)[..., ids]),
                        "velocity": metrics(np.concatenate(velocities)[..., ids]),
                    }
                    for g, ids in GROUPS.items()
                }
    stability = {}
    for group in GROUPS:
        by_method = {
            name: {x["source_recording_id"]: x for x in sources if x["method"] == name and x["group"] == group}
            for name in ("step100", "step250")
        }
        changes = np.array([by_method["step250"][s]["mae"] - v["mae"] for s, v in by_method["step100"].items()])
        stability[group] = {
            "improved_sources": int((changes < 0).sum()),
            "total_sources": len(changes),
            "mean_source_mae_delta": float(changes.mean()),
            "worst_source_mae_delta": float(changes.max()),
        }
    state_rows = []
    for i, w in enumerate(plan["windows"]):
        state = dataset.episode(w["id"])["states"][w["start"] : w["start"] + 50]
        for joint in TRACKED:
            j = ORDER.index(joint)
            delta = np.diff(truth[i, :, j])
            peak = int(np.argmax(abs(delta)))
            state_rows.append(
                {
                    "id": w["id"],
                    "start": w["start"],
                    "source_recording_id": w["source_recording_id"],
                    "episode_index": w["episode_index"],
                    "joint": joint,
                    "max_command_step_rad": float(abs(delta).max()),
                    "command_step_peak_frame": w["start"] + peak + 1,
                    "max_measured_state_step_rad": float(abs(np.diff(state[:, j])).max()),
                    "command_minus_state_mae": float(abs(truth[i, :, j] - state[:, j]).mean()),
                    "oracle_window_max": metrics(before["oracle"][i, 2:52, j] - truth[i, :, j])["max"],
                    "step100_window_max": metrics(before["prediction"][i, 2:52, j] - truth[i, :, j])["max"],
                    "step250_window_max": metrics(after["prediction"][i, 2:52, j] - truth[i, :, j])["max"],
                }
            )
    for fname, rows in [
        ("per_joint", joints["rows"]),
        ("per_window", windows),
        ("per_source", sources),
        ("per_root", roots),
        ("tracked_fingers", critical),
        ("state_command", state_rows),
    ]:
        write_csv(output / (fname + ".csv"), rows)
    if pairs:
        write_csv(output / "replan_pairs.csv", pairs)
    # Each tracked finger's SAME worst-100 window is shown before and after; no cherry-picking by250.
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout="constrained")
    examples = []
    for ax, joint in zip(axes.flat, TRACKED, strict=True):
        j = ORDER.index(joint)
        e = abs(before["prediction"][:, 2:52, j] - truth[..., j])
        i = int(np.argmax(e.max(axis=1)))
        w = plan["windows"][i]
        state = dataset.episode(w["id"])["states"][w["start"] : w["start"] + 50, j]
        ax.plot(np.arange(50) / 30, truth[i, :, j], color="black", label="GT command")
        ax.plot(np.arange(50) / 30, state, color="purple", linestyle=":", label="measured state")
        for name, (q, _) in methods.items():
            ax.plot(np.arange(50) / 30, q[i, 2:52, j], label=name)
        ax.set(
            title=f"{joint}\n{w['id']} ep{w['episode_index']} frames{w['start']}..{w['start'] + 49}",
            xlabel="chunk time(s)",
            ylabel="rad",
        )
        ax.legend(fontsize=7)
        examples.append(
            {
                "joint": joint,
                "window_index": i,
                **w,
                "step100_max": float(e[i].max()),
                "step250_max": metrics(after["prediction"][i, 2:52, j] - truth[i, :, j])["max"],
            }
        )
    fig.savefig(output / "tracked_trajectories.png", dpi=150)
    plt.close(fig)
    (output / "trajectory_examples.json").write_text(json.dumps(examples, indent=2))
    result = {
        "aggregate": aggregate,
        "continuity": continuity,
        "source_stability": stability,
        "windows": len(plan["windows"]),
        "plan_sha256": plan["sha256"],
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--training", type=Path, default=None)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    d = FlashLeRobotDataset(args.data, ".cache/flash_p2_images")
    fixed = json.loads(Path("docs/flash/results/p25/evaluation_plan.json").read_text())
    uniform = json.loads((args.run / "uniform_plan.json").read_text())
    result = {}
    for name, plan, old, new in [
        ("fixed813", fixed, Path(".cache/flash_p25/step100"), args.run / "step250_fixed"),
        ("uniform505", uniform, args.run / "step100_uniform", args.run / "step250_uniform"),
    ]:
        dest = args.output / name
        dest.mkdir(exist_ok=True)
        before = dict(np.load(old / "predictions.npz"))
        after = dict(np.load(new / "predictions.npz"))
        old_a = json.loads((old / "audit.json").read_text())
        new_a = json.loads((new / "audit.json").read_text())
        assert old_a["input_sequence_sha256"] == new_a["input_sequence_sha256"]
        assert new_a["optimizer_step"] == 250
        assert new_a["adam_state_roundtrip"]
        result[name] = evaluate_set(plan, before, after, dest, d)
        for tag, audit in [("step100", old_a), ("step250", new_a)]:
            (dest / (tag + "_audit.json")).write_text(json.dumps(audit, indent=2))
    train = json.loads(((args.training or args.run / "training") / "result.json").read_text())
    assert train["new_optimizer_updates"] == 150
    assert [e["global_step"] for e in train["train"]] == list(range(101, 251))
    assert train["freeze_audit"]["unchanged"]
    result["training"] = train
    (args.output / "results.json").write_text(json.dumps(result, indent=2))
    write_csv(args.output / "training_curve.csv", train["train"])
    write_csv(
        args.output / "probe_curve.csv",
        [
            {
                "global_step": e["global_step"],
                **{f"{label}_{key}": v for label, row in e.items() if isinstance(row, dict) for key, v in row.items()},
            }
            for e in train["probes"]
        ],
    )
    for name in ("train_plan.json", "uniform_plan.json"):
        shutil.copyfile(args.run / name, args.output / name)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for key in ("loss", "flow_matching_loss", "polynomial_consistency_loss"):
        vals = np.array([e[key] for e in train["train"]])
        steps = np.arange(101, 251)
        axes[0, 0].plot(steps, vals, alpha=0.15)
        axes[0, 0].plot(steps[14:], np.convolve(vals, np.ones(15) / 15, mode="valid"), label=key)
        for label in ("train_fixed16", "validation_fixed16"):
            axes[0, 1].plot(
                [e["global_step"] for e in train["probes"]],
                [e[label][key] for e in train["probes"]],
                label=label + " " + key,
            )
    axes[0, 0].set(
        title="150 resumed updates; 15-step means", xlabel="global optimizer step", ylabel="coefficient objective"
    )
    axes[0, 1].set(
        title="Same-draw, clean-source probes (16 each)", xlabel="global optimizer step", ylabel="coefficient objective"
    )
    axes[1, 0].plot(
        [e["global_step"] for e in train["train"]],
        [e["gradient_norm"] for e in train["train"]],
        label="preclip grad norm",
    )
    axes[1, 0].set(title="Gradient norm", xlabel="global optimizer step")
    axes[1, 1].plot(
        [e["global_step"] for e in train["train"]],
        [e["compute_seconds"] * 1000 for e in train["train"]],
        label="backward+AdamW",
    )
    axes[1, 1].set(title="First update includes compilation", xlabel="global optimizer step", ylabel="ms", yscale="log")
    for ax in axes.flat:
        ax.legend(fontsize=6)
        ax.grid(alpha=0.2)
    fig.savefig(args.output / "training_curves.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, (label, stats) in zip(
        axes, [("fixed813", result["fixed813"]), ("uniform505", result["uniform505"])], strict=True
    ):
        for offset, name in enumerate(("step100", "step250", "oracle_k6")):
            ax.bar(
                np.arange(3) + (offset - 1) * 0.25,
                [stats["aggregate"][name][g]["mae"] for g in GROUPS],
                width=0.25,
                label=name,
            )
        ax.set(title=label + " MAE", xticks=np.arange(3), xticklabels=list(GROUPS), ylabel="rad", yscale="log")
        ax.legend(fontsize=8)
    fig.savefig(args.output / "validation_comparison.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, group in enumerate(("arms", "hands")):
        for col, kind in enumerate(("position", "velocity")):
            for name in ("step100", "step250", "oracle_k6"):
                axes[row, col].plot(
                    [10, 15, 50],
                    [result["fixed813"]["continuity"][name][str(delta)][group][kind]["mae"] for delta in (10, 15, 50)],
                    marker="o",
                    label=name,
                )
            axes[row, col].set(
                title=group + " " + kind + " gap",
                xlabel="replan interval(frames)",
                ylabel="rad" if kind == "position" else "rad/s",
                yscale="log",
            )
            axes[row, col].legend(fontsize=8)
    fig.savefig(args.output / "continuity_comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: v["aggregate"] for k, v in result.items() if k != "training"}, indent=2))


if __name__ == "__main__":
    main()
