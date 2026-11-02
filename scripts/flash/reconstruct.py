"""Real LeRobot P0 reconstruction; fixed 50-tick horizon at all sampling strides."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from polynomial import fit
from polynomial import operators
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("protocol", ROOT / "src/openpi/policies/fr3_sharpa_protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
ORDER = protocol.ORDER
GROUPS = {
    "arms": [i for i, n in enumerate(ORDER) if "fr3" in n],
    "hands": [i for i, n in enumerate(ORDER) if "fr3" not in n],
}


def canonical_names(names):
    if len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    return [n.rsplit(".", 1)[0] for n in names]


def load(record, directory):
    f = directory / record["local_file"]
    if hashlib.sha256(f.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Sample hash mismatch")
    info = record["info"]
    features = info["features"]
    names = canonical_names(features["action"]["names"])
    if len(names) != 58 or set(names) != set(ORDER):
        raise ValueError("Unknown joint schema")
    if any("delta" in n.lower() for n in features["action"]["names"]):
        raise ValueError("Undeclared Delta action")
    # Known converter contract on main: .position/.command are absolute joint targets in radians.
    if not all(n.endswith((".position", ".command")) for n in features["action"]["names"]):
        raise ValueError("Absolute action semantics not established by metadata")
    table = pq.read_table(f)
    act = np.stack(table["action"].to_pylist()).astype(np.float64)[:, [names.index(n) for n in ORDER]]
    state_all = np.stack(table["observation.state"].to_pylist()).astype(np.float64)
    position_names = list(features["observation.state"]["names"])
    state = state_all[:, [position_names.index(n + ".position") for n in ORDER]]
    ts = np.array(table["timestamp"].to_pylist(), dtype=float).reshape(-1)
    frames = np.array(table["frame_index"].to_pylist()).reshape(-1)
    episodes = np.array(table["episode_index"].to_pylist()).reshape(-1)
    if not np.isfinite(act).all() or not np.isfinite(state).all():
        raise ValueError("Nonfinite samples")
    if len(set(episodes)) != 1 or not np.all(np.diff(frames) == 1):
        raise ValueError("Episode boundary/frame gap")
    fps = info["fps"]
    if not np.allclose(np.diff(ts), 1 / fps, atol=2e-4, rtol=2e-3):
        raise ValueError("Timestamp/FPS mismatch")
    return act, state, fps


def stats(values):
    values = np.asarray(values)
    return {
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p99_absolute": float(np.quantile(np.abs(values), 0.99)),
        "max_absolute": float(np.max(np.abs(values))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.samples / "manifest.json").read_text())
    buckets = {}
    provenance = []
    examples = {}
    worst = {}
    episode_metrics = []
    for record in manifest:
        actions, state, fps = load(record, args.samples)
        starts = np.arange(2, len(actions) - 52, 50)
        if not len(starts):
            raise ValueError("No complete 50-step windows")
        target = actions[starts[:, None] + np.arange(50)]
        target_vel = np.gradient(actions, 1 / fps, axis=0)[starts[:, None] + np.arange(50)]
        provenance.append(
            {k: record[k] for k in ("id", "host", "kind", "root", "source_path", "source_size", "sha256", "rows")}
            | {"fps": fps, "windows": len(starts), "action_semantics": "absolute_joint_target_rad"}
        )
        for degree in [3, 6, 9, 12]:
            for stride in [1, 2, 4]:
                for mode in ["ols50", "c1_extended54"]:
                    op = operators(degree, stride, constrained=mode != "ols50", fps=fps)
                    sampled = actions[starts[:, None] + op["ticks"]]
                    anchor_idx = starts[:, None] + [0, 50]
                    coeff = fit(
                        sampled, op, actions[anchor_idx], (actions[anchor_idx + 1] - actions[anchor_idx - 1]) * fps / 2
                    )
                    pred = np.einsum("hk,bkd->bhd", op["ev"], coeff)
                    vel = np.einsum("hk,bkd->bhd", op["vel"], coeff)
                    positions = np.einsum("hk,bkd->bhd", op["boundary"], coeff)
                    velocities = np.einsum("hk,bkd->bhd", op["boundary_vel"], coeff)
                    key = (record["kind"], degree, stride, mode)
                    data = buckets.setdefault(
                        key,
                        {
                            "errors": [],
                            "velocity": [],
                            "amplitude": [],
                            "position_gap": [],
                            "velocity_gap": [],
                            "endpoint": [],
                            "windows": 0,
                            "condition": op["condition"],
                        },
                    )
                    data["errors"].append(pred - target)
                    data["velocity"].append(vel - target_vel)
                    data["amplitude"].append(np.ptp(pred, axis=1) - np.ptp(target, axis=1))
                    data["endpoint"].append((pred - target)[:, [0, -1]])
                    data["position_gap"].append(positions[:-1, 1] - positions[1:, 0])
                    data["velocity_gap"].append(velocities[:-1, 1] - velocities[1:, 0])
                    data["windows"] += len(starts)
                    if stride == 1 and mode == "ols50":
                        episode_metrics.append(
                            {
                                "id": record["id"],
                                "kind": record["kind"],
                                "degree": degree,
                                "groups": {g: stats((pred - target)[..., idx]) for g, idx in GROUPS.items()},
                            }
                        )
                        hand_error = np.abs((pred - target)[:, :, GROUPS["hands"]])
                        w, t, j = np.unravel_index(np.argmax(hand_error), hand_error.shape)
                        if degree == 12 and (
                            record["kind"] not in worst or hand_error[w, t, j] > worst[record["kind"]]["error"]
                        ):
                            joint = GROUPS["hands"][j]
                            worst[record["kind"]] = {
                                "error": float(hand_error[w, t, j]),
                                "id": record["id"],
                                "joint": ORDER[joint],
                                "frame": int(starts[w] + t),
                                "window_start": int(starts[w]),
                                "target": target[w, :, joint].tolist(),
                                "reconstruction": pred[w, :, joint].tolist(),
                                "measured_state": state[starts[w] : starts[w] + 50, joint].tolist(),
                                "max_target_step_rad": float(np.max(np.abs(np.diff(target[w, :, joint])))),
                            }
                    if stride == 1 and mode == "ols50" and record["kind"] not in examples:
                        examples[record["kind"]] = {"target": target, "pred": {}, "id": record["id"]}
                    if stride == 1 and mode == "ols50" and record["id"] == examples[record["kind"]]["id"]:
                        examples[record["kind"]]["pred"][degree] = pred
    results = []
    for (kind, degree, stride, mode), data in buckets.items():
        arrays = {k: np.concatenate(v) for k, v in data.items() if isinstance(v, list)}
        item = {
            "kind": kind,
            "degree": degree,
            "coefficients": degree + 1,
            "stride": stride,
            "mode": mode,
            "windows": data["windows"],
            "condition": data["condition"],
            "groups": {},
        }
        for group, indices in GROUPS.items():
            item["groups"][group] = {key: stats(value[..., indices]) for key, value in arrays.items() if value.size}
        item["per_joint_max_rad"] = dict(
            zip(ORDER, np.max(np.abs(arrays["errors"]), axis=(0, 1)).tolist(), strict=True)
        )
        results.append(item)
    payload = {
        "horizon": 50,
        "strides": [1, 2, 4],
        "degree_means": "highest polynomial degree; K+1 coefficients",
        "source_manifest_sha256": hashlib.sha256((args.samples / "manifest.json").read_bytes()).hexdigest(),
        "worst_hand_windows_k12": worst,
        "episode_metrics": episode_metrics,
        "total_episodes": len(manifest),
        "total_windows": sum(x["windows"] for x in provenance),
        "provenance": provenance,
        "results": results,
        "continuity": "Evaluate neighboring polynomials at same shared tick 50/0; derivative rad/s. C1 anchors use oracle demonstration endpoints, not model guarantees.",
    }
    (args.output / "reconstruction.json").write_text(json.dumps(payload, indent=2))
    for kind, example in examples.items():
        # Plot the window with largest hand motion, rather than a selected easy stationary case.
        target = example["target"]
        w = np.argmax(np.max(np.ptp(target[:, :, GROUPS["hands"]], axis=1), axis=1))
        fig, axs = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
        for ax, group in zip(axs[:, 0], GROUPS, strict=True):
            joint = GROUPS[group][np.argmax(np.ptp(target[w][:, GROUPS[group]], axis=0))]
            ax.plot(np.arange(50) / 30, target[w, :, joint], color="black", label="real")
            for degree, pred in example["pred"].items():
                ax.plot(np.arange(50) / 30, pred[w, :, joint], label=f"K={degree}")
            ax.set(title=ORDER[joint], xlabel="time (s)", ylabel="absolute target (rad)")
            ax.legend()
        for ax, group in zip(axs[:, 1], GROUPS, strict=True):
            for mode in ["ols50", "c1_extended54"]:
                rows = sorted(
                    [r for r in results if r["kind"] == kind and r["stride"] == 1 and r["mode"] == mode],
                    key=lambda r: r["degree"],
                )
                ax.plot(
                    [r["degree"] for r in rows],
                    [r["groups"][group]["errors"]["rmse"] for r in rows],
                    marker="o",
                    label=mode,
                )
            ax.set(title=f"{kind} {group} reconstruction", xlabel="degree K", ylabel="RMSE (rad)")
            ax.legend()
        fig.savefig(args.output / f"{kind}_trajectories.png", dpi=150)
        plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, (kind, event) in zip(axs, worst.items(), strict=True):
        for key, label in [
            ("target", "real command"),
            ("reconstruction", "K=12 reconstruction"),
            ("measured_state", "measured joint position"),
        ]:
            ax.plot(np.arange(50) / 30, event[key], label=label)
        ax.set(title=f"{kind}: {event['joint']}\nmax error {event['error']:.3f} rad", xlabel="time (s)", ylabel="rad")
        ax.legend(fontsize=7)
    fig.savefig(args.output / "worst_hand_windows.png", dpi=150)
    plt.close(fig)
    import csv

    with (args.output / "per_joint_max.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "degree", "stride", "mode", "joint", "max_error_rad"])
        for item in results:
            for joint, error in item["per_joint_max_rad"].items():
                writer.writerow([item["kind"], item["degree"], item["stride"], item["mode"], joint, error])
    for item in results:
        if item["stride"] == 1 and item["mode"] == "ols50":
            print(item["kind"], item["degree"], item["windows"], {g: item["groups"][g]["errors"] for g in GROUPS})
    print("episodes", len(manifest), "windows", payload["total_windows"])


if __name__ == "__main__":
    main()
