"""Aggregate paired extended evaluation. Oracle never enters model conditions."""

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

from openpi.policies.fr3_sharpa_protocol import ORDER

GROUPS = {
    "all": list(range(58)),
    "arms": [i for i, j in enumerate(ORDER) if "fr3" in j],
    "hands": [i for i, j in enumerate(ORDER) if "fr3" not in j],
}
FINGERS = [
    "left_index_MCP_FE",
    "left_index_DIP",
    "right_index_DIP",
    "right_thumb_MCP_FE",
    "left_ring_MCP_FE",
    "left_middle_MCP_FE",
]


def aligned_difference(old, new, delta):
    if delta in (10, 15):
        return new[2 : 52 - delta] - old[2 + delta : 52]
    if delta == 50:
        return new[2:3] - old[52:53]
    raise ValueError("Only 10/15 rolling overlap or 50 boundary")


def unconstrained_reference(target, q01, q99):
    """Float64 SVD reference ONLY; separates degree error from C1 projection."""
    lo, hi = np.asarray(q01), np.asarray(q99)
    normalized = (target.astype(np.float64) - lo) / (hi - lo + 1e-6) * 2 - 1
    basis = np.polynomial.legendre.legvander(np.linspace(-1, 1, 54), 6)
    rhs = normalized.transpose(1, 0, 2).reshape(54, -1)
    coefficients = np.linalg.lstsq(basis, rhs, rcond=None)[0]
    reconstructed = (basis @ coefficients).reshape(54, len(target), 58).transpose(1, 0, 2)
    return (reconstructed + 1) / 2 * (hi - lo + 1e-6) + lo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads((args.run / "plan.json").read_text())
    step = dict(np.load(args.run / "step100/predictions.npz"))
    initial = dict(np.load(args.run / "initial/predictions.npz"))
    audits = {
        m: json.loads((args.run / m / "audit.json").read_text()) for m in ("initial", "step100", "step100_repeat32")
    }
    assert audits["initial"]["input_sequence_sha256"] == audits["step100"]["input_sequence_sha256"]
    np.testing.assert_array_equal(step["truth"], initial["truth"])
    np.testing.assert_array_equal(step["history"], initial["history"])
    np.testing.assert_array_equal(step["oracle"], initial["oracle"])
    assert len(plan["windows"]) == len(step["prediction"])
    methods = {
        "step100": (step["prediction"], step["velocity"]),
        "initial": (initial["prediction"], initial["velocity"]),
        "oracle_k6": (step["oracle"], step["oracle_velocity"]),
    }
    truth = step["truth"][:, 2:52]
    normalization = json.loads((args.run.parent / "flash_p2_data/normalization.json").read_text())
    unconstrained = unconstrained_reference(step["truth"], normalization["q01"], normalization["q99"])
    representation = {g: metrics((unconstrained[:, 2:52] - truth)[..., ids]) for g, ids in GROUPS.items()}
    representation_windows = []
    for name, q in [("oracle_k6_c1", step["oracle"]), ("oracle_k6_unconstrained_svd", unconstrained)]:
        for i, index in enumerate(plan["windows"]):
            representation_windows.append(
                {
                    "method": name,
                    "id": index["id"],
                    "start": index["start"],
                    "source_recording_id": index["source_recording_id"],
                    **metrics(q[i, 2:52] - truth[i]),
                }
            )
    write_csv(args.output / "representation_windows.csv", representation_windows)
    aggregate, per_joint, per_window, per_episode, per_root, critical = {}, [], [], [], [], []
    for name, (q, _) in methods.items():
        e = q[:, 2:52] - truth
        aggregate[name] = {g: metrics(e[..., ids]) for g, ids in GROUPS.items()}
        for j, joint in enumerate(ORDER):
            per_joint.append(
                {"method": name, "joint": joint, "group": "arms" if "fr3" in joint else "hands", **metrics(e[..., j])}
            )
        for i, index in enumerate(plan["windows"]):
            for group, ids in GROUPS.items():
                per_window.append(
                    {
                        **{k: v for k, v in index.items() if k != "strata"},
                        "strata": ";".join(index["strata"]),
                        "method": name,
                        "group": group,
                        **metrics(e[i, :, ids]),
                    }
                )
            if any("transient" in label for label in index["strata"]):
                for joint in FINGERS:
                    j = ORDER.index(joint)
                    peak = int(np.argmax(abs(e[i, :, j])))
                    critical.append(
                        {
                            "method": name,
                            "id": index["id"],
                            "source_recording_id": index["source_recording_id"],
                            "root": index["root"],
                            "episode_index": index["episode_index"],
                            "start": index["start"],
                            "joint": joint,
                            "peak_frame": index["start"] + peak,
                            **metrics(e[i, :, j]),
                        }
                    )
        for field, output in [("source_recording_id", per_episode), ("root", per_root)]:
            for value in sorted({w[field] for w in plan["windows"]}):
                selection = [i for i, w in enumerate(plan["windows"]) if w[field] == value]
                for group, ids in GROUPS.items():
                    output.append(
                        {
                            "method": name,
                            field: value,
                            **(
                                {
                                    "id": plan["windows"][selection[0]]["id"],
                                    "episode_index": plan["windows"][selection[0]]["episode_index"],
                                    "root": plan["windows"][selection[0]]["root"],
                                }
                                if field == "source_recording_id"
                                else {}
                            ),
                            "windows": len(selection),
                            "group": group,
                            **metrics(e[selection][..., ids]),
                        }
                    )
    continuity, per_pair, attribution = {}, [], []
    for name, (q, v) in methods.items():
        continuity[name] = {}
        counter = step if name == "step100" else initial
        counter_map = {(int(a), int(b), int(delta)): i for i, (a, b, delta) in enumerate(counter["counter_indices"])}
        for delta in (10, 15, 50):
            positions, velocities, history_terms, context_terms = [], [], [], []
            for anchor in plan["anchors"]:
                a, b = anchor["window_indices"]["0"], anchor["window_indices"][str(delta)]
                pos = aligned_difference(q[a], q[b], delta)
                vel = aligned_difference(v[a], v[b], delta)
                # Ground-truth physical time alignment must be exact.
                np.testing.assert_array_equal(
                    aligned_difference(step["truth"][a], step["truth"][b], delta), np.zeros_like(pos)
                )
                positions.append(pos)
                velocities.append(vel)
                for group, ids in GROUPS.items():
                    per_pair.append(
                        {
                            "method": name,
                            "id": anchor["id"],
                            "start": anchor["start"],
                            "stratum": anchor["stratum"],
                            "delta": delta,
                            "group": group,
                            "position_mae": metrics(pos[:, ids])["mae"],
                            "velocity_mae": metrics(vel[:, ids])["mae"],
                            "position_max": metrics(pos[:, ids])["max"],
                            "velocity_max": metrics(vel[:, ids])["max"],
                        }
                    )
                if name != "oracle_k6":
                    cq, cv = counter["counterfactual"][counter_map[(a, b, delta)]]
                    ctx = aligned_difference(q[a], cq, delta)
                    hist = pos - ctx
                    history_terms.append(hist)
                    context_terms.append(ctx)
                    for group, ids in GROUPS.items():
                        attribution.append(
                            {
                                "method": name,
                                "id": anchor["id"],
                                "start": anchor["start"],
                                "delta": delta,
                                "group": group,
                                "actual_gap_mae": metrics(pos[:, ids])["mae"],
                                "old_history_current_prefix_gap_mae": metrics(ctx[:, ids])["mae"],
                                "history_change_contribution_mae": metrics(hist[:, ids])["mae"],
                                "history_input_rmse": metrics(step["history"][a][:, ids] - step["history"][b][:, ids])[
                                    "rmse"
                                ],
                                "velocity_history_contribution_mae": metrics(
                                    (vel - aligned_difference(v[a], cv, delta))[:, ids]
                                )["mae"],
                            }
                        )
            continuity[name][str(delta)] = {
                g: {
                    "position": metrics(np.concatenate(positions)[..., ids]),
                    "velocity": metrics(np.concatenate(velocities)[..., ids]),
                }
                for g, ids in GROUPS.items()
            }
            if history_terms:
                for g, ids in GROUPS.items():
                    continuity[name][str(delta)][g]["history_change_contribution"] = metrics(
                        np.concatenate(history_terms)[..., ids]
                    )
                    continuity[name][str(delta)][g]["current_prefix_old_history_gap"] = metrics(
                        np.concatenate(context_terms)[..., ids]
                    )
    # Equal-source improvements complement frame-pooled diagnostics.
    stability = {}
    for group in GROUPS:
        before = {x["source_recording_id"]: x for x in per_episode if x["method"] == "initial" and x["group"] == group}
        after = {x["source_recording_id"]: x for x in per_episode if x["method"] == "step100" and x["group"] == group}
        delta = np.array([after[s]["mae"] - before[s]["mae"] for s in sorted(before)])
        rng = np.random.default_rng(42)
        boots = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
        stability[group] = {
            "improved_sources": int((delta < 0).sum()),
            "sources": len(delta),
            "mean_source_mae_delta": float(delta.mean()),
            "source_bootstrap_delta_interval95": np.quantile(boots, [0.025, 0.975]).tolist(),
            "worst_source_mae_delta": float(delta.max()),
        }
    repeat = dict(np.load(args.run / "step100_repeat32/predictions.npz"))
    np.testing.assert_array_equal(repeat["truth"], step["truth"][: len(repeat["truth"])])
    assert audits["step100_repeat32"]["input_hashes"] == audits["step100"]["input_hashes"][: len(repeat["truth"])]
    drift = metrics(repeat["prediction"][:, 2:52] - step["prediction"][: len(repeat["prediction"]), 2:52])
    p2_root = args.run.parent / "flash_p2_recovery_02"
    p2_plan = json.loads((args.run.parent / "flash_p2_data/window_plan.json").read_text())
    p2_predictions = dict(np.load(p2_root / "validation_step_100.npz"))
    old_index = {(w["id"], w["start"]): i for i, w in enumerate(p2_plan["validation"])}
    shared = [
        (i, old_index[(w["id"], w["start"])])
        for i, w in enumerate(plan["windows"])
        if (w["id"], w["start"]) in old_index
    ]
    p2_differences = []
    for i, j in shared:
        np.testing.assert_array_equal(step["truth"][i, 2:52], p2_predictions["truth"][j])
        p2_differences.append(step["prediction"][i, 2:52] - p2_predictions["prediction"][j, 2:52])
    p2_graph_drift = {
        "shared_windows": len(shared),
        "physical_radians": metrics(np.stack(p2_differences)),
        "scope": "same indices/data/norm/weights, P2 training-time versus P2.5 evaluation graph; exact cause not isolated",
    }
    report = {
        "aggregate": aggregate,
        "unconstrained_k6_svd_reference": representation,
        "p2_graph_drift": p2_graph_drift,
        "continuity": continuity,
        "source_stability": stability,
        "cross_process_step100_drift_rad": drift,
        "sources": plan["source_count"],
        "windows": len(plan["windows"]),
        "anchor_pairs_per_delta": len(plan["anchors"]),
        "input_identity": True,
        "training_updates": 0,
        "audits": audits,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    for filename, rows in [
        ("per_joint", per_joint),
        ("per_window", per_window),
        ("per_source", per_episode),
        ("per_root", per_root),
        ("critical_fingers", critical),
        ("replan_pairs", per_pair),
        ("history_attribution", attribution),
    ]:
        write_csv(args.output / (filename + ".csv"), rows)
    shutil.copyfile(args.run / "plan.json", args.output / "evaluation_plan.json")
    for name, a in audits.items():
        (args.output / (name + "_audit.json")).write_text(json.dumps(a, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, group in zip(axes, GROUPS, strict=True):
        rows = [x for x in per_episode if x["group"] == group]
        for name in methods:
            ax.plot(sorted(x["mae"] for x in rows if x["method"] == name), label=name)
        ax.set(title=group + ": sorted source MAE", xlabel="source rank (method-specific)", ylabel="rad", yscale="log")
        ax.legend(fontsize=8)
    fig.savefig(args.output / "source_errors.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, group in enumerate(("arms", "hands")):
        for col, kind in enumerate(("position", "velocity")):
            for name in methods:
                axes[row, col].plot(
                    [10, 15, 50],
                    [continuity[name][str(k)][group][kind]["mae"] for k in (10, 15, 50)],
                    marker="o",
                    label=name,
                )
            axes[row, col].set(
                title=group + " " + kind + " gap",
                xlabel="replan interval(frames)",
                ylabel="rad" if col == 0 else "rad/s",
                yscale="log",
            )
            axes[row, col].legend(fontsize=8)
    fig.savefig(args.output / "replan_continuity.png", dpi=150)
    plt.close(fig)
    # Worst learned hand window, alongside the Oracle representation failure.
    errors = step["prediction"][:, 2:52] - truth
    hand_errors = abs(errors[..., GROUPS["hands"]])
    wi, t, j = np.unravel_index(np.argmax(hand_errors), hand_errors.shape)
    joint = GROUPS["hands"][j]
    oracle_err = abs(step["oracle"][:, 2:52] - truth)
    oi, ot, oj = np.unravel_index(np.argmax(oracle_err), oracle_err.shape)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for ax, i, j in [(axes[0], wi, joint), (axes[1], oi, oj)]:
        ax.plot(np.arange(50) / 30, truth[i, :, j], label="GT", color="black")
        for name, (q, _) in methods.items():
            ax.plot(np.arange(50) / 30, q[i, 2:52, j], label=name)
        w = plan["windows"][i]
        ax.set(
            title=f"{ORDER[j]}\n{w['id']} ep{w['episode_index']} frames{w['start']}..{w['start'] + 49}",
            xlabel="chunk time(s)",
            ylabel="rad",
        )
        ax.plot(np.arange(50) / 30, unconstrained[i, 2:52, j], label="K6 SVD no C1", linestyle="--", color="purple")
        ax.legend(fontsize=8)
    fig.savefig(args.output / "worst_trajectories.png", dpi=150)
    plt.close(fig)
    worst = []
    for name, (q, _) in methods.items():
        e = abs(q[:, 2:52] - truth)
        for i in np.argsort(e.max(axis=(1, 2)))[-20:][::-1]:
            t, j = np.unravel_index(np.argmax(e[i]), e[i].shape)
            worst.append(
                {
                    **{k: v for k, v in plan["windows"][i].items() if k != "strata"},
                    "method": name,
                    "joint": ORDER[j],
                    "peak_frame": plan["windows"][i]["start"] + int(t),
                    "error": float(e[i, t, j]),
                }
            )
    write_csv(args.output / "worst20_windows.csv", worst)
    print(json.dumps({"aggregate": aggregate, "stability": stability, "drift": drift}, indent=2))


if __name__ == "__main__":
    main()
