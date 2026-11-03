"""Export immutable P2 curves, physical errors, provenance and comparison figures.

Numerical predictions remain in ignored local run directories; publish derived
statistics/plots only, with original episode IDs, split/norm hashes and commands.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.policies.fr3_sharpa_protocol import ORDER


def metrics(error):
    error = np.asarray(error)
    a = abs(error)
    return {
        "mae": float(a.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }


def write_csv(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    r = json.loads((args.run / "result.json").read_text())
    plan = json.loads((args.data / "window_plan.json").read_text())
    manifest = json.loads((args.data / "manifest.json").read_text())
    json.loads((args.data / "split.json").read_text())
    records = {v["id"]: v for v in manifest["records"]}
    before = dict(np.load(args.run / "validation_step_000.npz"))
    after = dict(np.load(args.run / "validation_step_100.npz"))
    truth = before["truth"]
    predictions = {
        "before": before["prediction"][:, 2:52],
        "after100": after["prediction"][:, 2:52],
        "oracle_k6": before["oracle"],
    }
    per_joint = []
    for method, prediction in predictions.items():
        for j, joint in enumerate(ORDER):
            per_joint.append(
                {
                    "method": method,
                    "joint": joint,
                    "group": "arms" if "fr3" in joint else "hands",
                    **metrics(prediction[..., j] - truth[..., j]),
                }
            )
    write_csv(args.output / "per_joint_errors.csv", per_joint)
    windows = []
    critical = []
    fingers = [
        "left_index_MCP_FE",
        "left_index_DIP",
        "right_index_DIP",
        "right_thumb_MCP_FE",
        "left_ring_MCP_FE",
        "left_middle_MCP_FE",
    ]
    for i, index in enumerate(plan["validation"]):
        record = records[index["id"]]
        for method, prediction in predictions.items():
            windows.append(
                {
                    **index,
                    "method": method,
                    "episode_index": record["episode_index"],
                    "source_recording_id": record["source_recording_id"],
                    "frame_end": index["start"] + 49,
                    "source_path": record["source_path"],
                    **metrics(prediction[i] - truth[i]),
                }
            )
            if index["role"] == "predeclared_sharpa_transient":
                for joint in fingers:
                    j = ORDER.index(joint)
                    error = prediction[i, :, j] - truth[i, :, j]
                    peak = int(np.argmax(abs(error)))
                    critical.append(
                        {
                            **index,
                            "method": method,
                            "joint": joint,
                            "source_recording_id": record["source_recording_id"],
                            "peak_frame": index["start"] + peak,
                            **metrics(error),
                        }
                    )
    write_csv(args.output / "validation_windows.csv", windows)
    write_csv(args.output / "critical_fingers.csv", critical)
    curve = [
        {
            "step": v["step"],
            **v["metrics"],
            **{f"{group}_{key}": value for group, stats in v["physical"].items() for key, value in stats.items()},
            **{
                f"{group}_position_gap_{key}": value
                for group, stats in v["position_gap"].items()
                for key, value in stats.items()
            },
            **{
                f"{group}_velocity_gap_{key}": value
                for group, stats in v["velocity_gap"].items()
                for key, value in stats.items()
            },
        }
        for v in r["validation_curve"]
    ]
    write_csv(args.output / "validation_curve.csv", curve)
    write_csv(args.output / "fixed_curve.csv", r["fixed"])
    write_csv(args.output / "training_curve.csv", r["train"])
    for name in ["manifest.json", "split.json", "normalization.json", "window_plan.json"]:
        shutil.copyfile(args.data / name, args.output / name)
    shutil.copyfile(args.run / "result.json", args.output / "training_result.json")
    # All axes show physical radians or explicitly normalized coefficient objectives.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    colors = {"loss": "black", "flow_matching_loss": "tab:blue", "polynomial_consistency_loss": "tab:orange"}
    for key, color in colors.items():
        axes[0, 0].plot(
            np.arange(len(r["fixed"]) + 1),
            [row[key] for row in r["fixed"]] + [r["fixed_final"][key]],
            label=key,
            color=color,
        )
        values = np.array([row[key] for row in r["train"]])
        axes[0, 1].plot(np.arange(1, len(values) + 1), values, color=color, alpha=0.18)
        axes[0, 1].plot(
            np.arange(10, len(values) + 1), np.convolve(values, np.ones(10) / 10, mode="valid"), label=key, color=color
        )
        axes[1, 0].plot(
            [v["step"] for v in r["validation_curve"]],
            [v["metrics"][key] for v in r["validation_curve"]],
            marker="o",
            label=key,
            color=color,
        )
    axes[0, 0].set(
        title="Fixed real batch: same flow draw, 20 updates",
        xlabel="optimizer updates",
        ylabel="coefficient loss",
        yscale="log",
    )
    axes[0, 1].set(
        title="Fresh initialization, varying real batches",
        xlabel="optimizer updates",
        ylabel="loss; bold = 10-step mean",
        yscale="log",
    )
    axes[1, 0].set(title="Fixed held-out set: coefficient loss", xlabel="optimizer updates", ylabel="loss")
    for group in ("all", "arms", "hands"):
        axes[1, 1].plot(
            [v["step"] for v in r["validation_curve"]],
            [v["physical"][group]["mae"] for v in r["validation_curve"]],
            marker="o",
            label=group,
        )
        axes[1, 1].axhline(r["validation_curve"][0]["oracle_physical"][group]["mae"], linestyle="--", alpha=0.35)
    axes[1, 1].set(title="Held-out physical action MAE; dashed = GT oracle", xlabel="optimizer updates", ylabel="rad")
    for ax in axes.flat:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.savefig(args.output / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, key in zip(axes.flat, ["mae", "rmse", "p99", "max"], strict=True):
        x = np.arange(2)
        for offset, (name, stats) in enumerate(
            [
                ("before", r["validation_curve"][0]["physical"]),
                ("after100", r["validation_curve"][-1]["physical"]),
                ("GT oracle", r["validation_curve"][0]["oracle_physical"]),
            ]
        ):
            ax.bar(x + (offset - 1) * 0.25, [stats[g][key] for g in ("arms", "hands")], width=0.25, label=name)
        ax.set(xticks=x, xticklabels=["arms14D", "hands44D"], title=key, ylabel="rad", yscale="log")
        ax.legend(fontsize=8)
    fig.savefig(args.output / "physical_error_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    challenge_indices = [i for i, v in enumerate(plan["validation"]) if v["role"] == "predeclared_sharpa_transient"]
    choices = [
        (challenge_indices[0], "left_index_MCP_FE"),
        (challenge_indices[2], "left_index_DIP"),
        (0, "right_thumb_MCP_FE"),
        (0, "left_fr3_joint4"),
    ]
    for ax, (i, joint) in zip(axes.flat, choices, strict=True):
        j = ORDER.index(joint)
        time = np.arange(50) / 30
        ax.plot(time, truth[i, :, j], color="black", linewidth=2, label="GT command")
        for method, color in [("before", "tab:gray"), ("after100", "tab:blue"), ("oracle_k6", "tab:orange")]:
            ax.plot(
                time,
                predictions[method][i, :, j],
                label=method,
                color=color,
                linestyle="--" if method == "oracle_k6" else "-",
            )
        peak = np.argmax(abs(predictions["after100"][i, :, j] - truth[i, :, j]))
        ax.scatter(time[peak], predictions["after100"][i, peak, j], color="red", s=25, zorder=4)
        index = plan["validation"][i]
        ax.set(
            title=f"{joint}; ep{records[index['id']]['episode_index']} frames{index['start']}..{index['start'] + 49}",
            xlabel="chunk time(s)",
            ylabel="absolute target(rad)",
        )
        ax.legend(fontsize=7)
    fig.savefig(args.output / "same_observation_trajectories.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].plot(np.arange(1, len(r["fixed"]) + 1), [v["gradient_norm"] for v in r["fixed"]], label="fixed20")
    axes[0].plot(np.arange(1, len(r["train"]) + 1), [v["gradient_norm"] for v in r["train"]], label="fresh100")
    axes[0].set(title="Pre-clip gradient norm", xlabel="step")
    axes[0].legend()
    axes[1].plot([v["compute_seconds"] * 1000 for v in r["train"]], label="backward + AdamW")
    axes[1].plot([v["data_and_prefix_seconds"] * 1000 for v in r["train"]], label="data + VLM prefix")
    axes[1].set(title="Training time (first call includes compile)", xlabel="step index", ylabel="ms", yscale="log")
    axes[1].legend()
    axes[2].plot([v["peak_gpu_bytes"] / 1024**3 for v in r["train"]])
    axes[2].set(title="JAX peak bytes-in-use", xlabel="step index", ylabel="GiB", ylim=(0, 32))
    fig.savefig(args.output / "runtime_and_gradients.png", dpi=160)
    plt.close(fig)
    print(
        json.dumps(
            {
                "plots": 4,
                "per_joint_rows": len(per_joint),
                "critical_rows": len(critical),
                "validation_windows": len(plan["validation"]),
            }
        )
    )


if __name__ == "__main__":
    main()
