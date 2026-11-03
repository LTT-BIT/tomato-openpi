"""Reproduce P0.5 from immutable real episode samples; no training or robot I/O.

Metadata SSH is optional and read-only. Quantiles pool absolute scalar errors, and
per-joint/peak-window tables deliberately retain failures hidden by those quantiles.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import functools
import operator

from collect_samples import RemoteFile
from collect_samples import ssh_json
import matplotlib.pyplot as plt
import numpy as np
from polynomial import basis
from polynomial import fit
from polynomial import operators
import pyarrow.parquet as pq
from reconstruct import GROUPS
from reconstruct import ORDER
from reconstruct import load


def metrics(x):
    x = np.asarray(x)
    a = np.abs(x).ravel()
    return {
        "mae": float(a.mean()),
        "rmse": float(np.sqrt(np.mean(x * x))),
        "p95": float(np.quantile(a, 0.95)),
        "p99": float(np.quantile(a, 0.99)),
        "max": float(a.max()),
    }


def solver_operator(s, solver, ridge=0.0):
    """Avoid normal equations in each compared least-squares solution."""
    if solver == "qr":
        q, r = np.linalg.qr(s, mode="reduced")
        return np.linalg.solve(r, q.T)
    u, singular, vt = np.linalg.svd(s, full_matrices=False)
    filt = singular / (singular**2 + ridge) if ridge else 1 / singular
    return (vt.T * filt) @ u.T


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_metadata(records):
    out = {}
    for host in sorted({r["host"] for r in records}):
        wanted = {r["root"]: [] for r in records if r["host"] == host}
        for r in records:
            if r["host"] == host:
                wanted[r["root"]].append(int(Path(r["source_path"]).stem.split("_")[-1]))
        code = f"""import json,pathlib
out={{}}
for root,ids in {wanted!r}.items():
 p=pathlib.Path(root)/'meta/episodes.jsonl'
 out[root]=[json.loads(x) for x in p.read_text().splitlines() if json.loads(x)['episode_index'] in ids]
print(json.dumps(out))"""
        out.update(ssh_json(host, code))
    return out


def collect_source_times(records, directory):
    """Read only the original time index, never video/tactile payloads."""
    directory.mkdir(parents=True, exist_ok=True)

    def fetch(record):
        path = directory / (record["id"] + ".parquet")
        if not path.exists():
            remote = RemoteFile(record["host"], record["source_path"], record["source_size"])
            try:
                table = pq.read_table(remote, columns=["source_timestamp_ns"])
            finally:
                remote.close()
            if len(table) != record["rows"]:
                raise ValueError("Source timestamp row count changed")
            pq.write_table(table, path)
        return {
            "sample_id": record["id"],
            "source_path": record["source_path"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(fetch, records))
    (directory / "manifest.json").write_text(json.dumps(result, indent=2))
    return result


def exact_source_mapping(source, reference):
    """Integer nanoseconds only: float64 loses source-time precision near 1e18."""
    source = np.asarray(source, dtype=np.int64)
    reference = np.asarray(reference, dtype=np.int64)
    if not np.all(np.diff(source) > 0) or not np.all(np.diff(reference) > 0):
        raise ValueError("Source timestamps must be strictly increasing")
    mapping = np.searchsorted(reference, source)
    if np.any(mapping >= len(reference)) or not np.array_equal(reference[mapping], source):
        raise ValueError("Source timestamps are not an exact subset of reference")
    return mapping


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, default=Path(".cache/flash_samples"))
    p.add_argument("--output", type=Path, default=Path("docs/flash/results/p05"))
    p.add_argument("--fetch-metadata", action="store_true")
    p.add_argument("--source-times", type=Path, default=Path(".cache/flash_source_times"))
    p.add_argument("--fetch-source-times", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    records = json.loads((a.samples / "manifest.json").read_text())
    if a.fetch_source_times:
        collect_source_times(records, a.source_times)
    source_manifest = json.loads((a.source_times / "manifest.json").read_text())
    source_hashes = {r["sample_id"]: r["sha256"] for r in source_manifest}
    metadata_path = a.output / "source_metadata.json"
    if a.fetch_metadata:
        metadata_path.write_text(json.dumps(collect_metadata(records), indent=2, ensure_ascii=False))
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    episodes = []
    for r in records:
        action, state, fps = load(r, a.samples)
        ep = int(Path(r["source_path"]).stem.split("_")[-1])
        meta = next((m for m in metadata.get(r["root"], []) if m["episode_index"] == ep), {})
        starts = np.arange(2, len(action) - 52, 50)
        source_file = a.source_times / (r["id"] + ".parquet")
        if hashlib.sha256(source_file.read_bytes()).hexdigest() != source_hashes[r["id"]]:
            raise ValueError("Source timestamp cache fingerprint mismatch")
        source_times = np.asarray(
            pq.read_table(source_file)["source_timestamp_ns"].to_pylist(), dtype=np.int64
        ).reshape(-1)
        episodes.append(
            {
                "record": r,
                "action": action,
                "state": state,
                "fps": fps,
                "starts": starts,
                "episode": ep,
                "meta": meta,
                "source_times": source_times,
            }
        )
    stats_rows, joint_rows, window_rows, boundaries = [], [], [], []
    max_solver_delta = 0.0
    sparse_results = []
    for degree in [3, 6, 9, 12]:
        errors = {kind: [] for kind in ["raw", "normal", "speedup"]}
        candidates = {group: [] for group in GROUPS}
        for ei, e in enumerate(episodes):
            r, actions, starts, fps = e["record"], e["action"], e["starts"], e["fps"]
            op = operators(degree, fps=fps)
            target = actions[starts[:, None] + np.arange(50)]
            pred = np.einsum("hk,bkd->bhd", op["ev"], fit(target, op))
            error = pred - target
            errors[r["kind"]].append(error)
            for group, idx in GROUPS.items():
                for wi in range(len(starts)):
                    sub = np.abs(error[wi][:, idx])
                    t, j = np.unravel_index(sub.argmax(), sub.shape)
                    candidates[group].append((float(sub[t, j]), ei, wi, t, idx[j]))
            if degree == 6:
                for constrained in [False, True]:
                    cop = operators(6, constrained=constrained, fps=fps)
                    anchors = starts[:, None] + [0, 50]
                    c = fit(
                        actions[starts[:, None] + cop["ticks"]],
                        cop,
                        actions[anchors],
                        (actions[anchors + 1] - actions[anchors - 1]) * fps / 2,
                    )
                    pos = np.einsum("hk,bkd->bhd", cop["boundary"], c)
                    vel = np.einsum("hk,bkd->bhd", cop["boundary_vel"], c)
                    near_ticks = np.array([0, 1, 48, 49])
                    near = np.einsum(
                        "hk,bkd->bhd", basis(near_ticks, 6, -2 if constrained else 0, 51 if constrained else 49, fps), c
                    )
                    true_vel = (actions[anchors + 1] - actions[anchors - 1]) * fps / 2
                    for group, idx in GROUPS.items():
                        for name, val in [
                            ("boundary_position_error", pos - actions[anchors]),
                            ("boundary_velocity_error", vel - true_vel),
                            ("near_boundary_position_error", near - actions[starts[:, None] + near_ticks]),
                            ("adjacent_position_gap", pos[:-1, 1] - pos[1:, 0]),
                            ("adjacent_velocity_gap", vel[:-1, 1] - vel[1:, 0]),
                        ]:
                            boundaries.append(
                                {
                                    "kind": r["kind"],
                                    "mode": "oracle_c1" if constrained else "ols50",
                                    "group": group,
                                    "metric": name,
                                    "values": val[..., idx],
                                }
                            )
        for kind in [*errors, "all"]:
            err = np.concatenate(
                errors[kind] if kind != "all" else functools.reduce(operator.iadd, errors.values(), [])
            )
            for group, idx in GROUPS.items():
                window_peaks = np.abs(err[..., idx]).max(axis=(1, 2))
                stats_rows.append(
                    dict(
                        kind=kind,
                        degree=degree,
                        group=group,
                        windows=len(err),
                        **metrics(err[..., idx]),
                        window_peak_p95=float(np.quantile(window_peaks, 0.95)),
                        window_peak_p99=float(np.quantile(window_peaks, 0.99)),
                        windows_peak_above_0p1_rad=int((window_peaks > 0.1).sum()),
                    )
                )
            for j, name in enumerate(ORDER):
                joint_rows.append(dict(kind=kind, degree=degree, joint=name, **metrics(err[..., j])))
        for group, items in candidates.items():
            top = sorted(items, reverse=True)[:20]
            fig, axs = plt.subplots(5, 4, figsize=(20, 17), layout="constrained")
            for rank, (item, ax) in enumerate(zip(top, axs.flat, strict=True), 1):
                peak, ei, wi, t, j = item
                e = episodes[ei]
                r = e["record"]
                start = int(e["starts"][wi])
                target = e["action"][start : start + 50, j]
                op = operators(degree, fps=e["fps"])
                pred = op["ev"] @ op["inverse"] @ target
                row = {
                    "degree": degree,
                    "group": group,
                    "rank": rank,
                    "kind": r["kind"],
                    "sample_id": r["id"],
                    "episode": e["episode"],
                    "original_episode_id": e["meta"].get("source_recording_id", "unknown"),
                    "source_path": r["source_path"],
                    "frame_start": start,
                    "frame_end": start + 49,
                    "joint": ORDER[j],
                    "peak_frame": start + t,
                    "peak_tick": t,
                    "peak_error_rad": peak,
                    "max_target_step_rad": float(np.abs(np.diff(target)).max()),
                    "edge_distance_ticks": min(t, 49 - t),
                    "source_start_ns": e["meta"].get("source_start_ns", "unknown"),
                    "window_source_start_ns": int(e["source_times"][start]),
                    "window_source_end_ns": int(e["source_times"][start + 49]),
                    "peak_source_timestamp_ns": int(e["source_times"][start + t]),
                }
                window_rows.append(row)
                ax.plot(np.arange(start, start + 50), target, "k-", label="real absolute target")
                ax.plot(np.arange(start, start + 50), pred, label=f"K={degree}")
                ax.scatter([start + t], [pred[t]], color="red", s=22, zorder=5)
                ax.axvline(start + t, color="red", alpha=0.3)
                ax.set(
                    title=f"#{rank} {r['kind']} ep{e['episode']} {ORDER[j]}\n{peak:.4f} rad at {start + t}",
                    xlabel="episode frame",
                    ylabel="rad",
                )
                if rank == 1:
                    ax.legend(fontsize=7)
            fig.savefig(a.output / f"top20_K{degree}_{group}.png", dpi=110)
            plt.close(fig)
    # All real windows at K=12: QR vs SVD and ridge with the same sparse grid.
    for stride in [1, 2, 4]:
        op = operators(12, stride)
        s = basis(op["ticks"], 12, 0, 49, 30)
        methods = [("svd", 0.0), ("qr", 0.0), ("ridge", 1e-4), ("ridge", 1e-3), ("ridge", 1e-2), ("ridge", 0.1)]
        pools = {(method, lam, kind): [] for method, lam in methods for kind in ["raw", "normal", "speedup"]}
        leverage = op["ev"] @ solver_operator(s, "svd")
        for e in episodes:
            y = e["action"][e["starts"][:, None] + op["ticks"]]
            target = e["action"][e["starts"][:, None] + np.arange(50)]
            ref = None
            for method, lam in methods:
                pred = np.einsum("hk,bkd->bhd", op["ev"], np.einsum("kh,bhd->bkd", solver_operator(s, method, lam), y))
                if method == "svd":
                    ref = pred
                if method == "qr":
                    max_solver_delta = max(max_solver_delta, float(np.abs(pred - ref).max()))
                pools[method, lam, e["record"]["kind"]].append(pred - target)
        for (method, lam, kind), vals in pools.items():
            err = np.concatenate(vals)
            for group, idx in GROUPS.items():
                sparse_results.append(
                    dict(
                        stride=stride,
                        nodes=len(s),
                        condition=float(np.linalg.cond(s)),
                        max_reconstruction_row_l1=float(np.abs(leverage).sum(axis=1).max()),
                        method=method,
                        ridge=lam,
                        kind=kind,
                        group=group,
                        **metrics(err[..., idx]),
                    )
                )
    boundary_rows = []
    for kind in ["raw", "normal", "speedup"]:
        for mode in ["ols50", "oracle_c1"]:
            for group in GROUPS:
                for name in sorted({b["metric"] for b in boundaries}):
                    vals = [
                        b["values"]
                        for b in boundaries
                        if (b["kind"], b["mode"], b["group"], b["metric"]) == (kind, mode, group, name)
                    ]
                    boundary_rows.append(
                        dict(kind=kind, mode=mode, group=group, metric=name, **metrics(np.concatenate(vals)))
                    )
    comparisons = []
    correspondence = []
    source_shift_rows = []
    originals = {}
    for e in episodes:
        original = e["meta"].get("source_recording_id")
        if original:
            originals.setdefault(original, {})[e["record"]["kind"]] = e
    for original, kinds in originals.items():
        if "normal" not in kinds:
            continue
        normal = kinds["normal"]
        for kind, e in kinds.items():
            if kind == "normal":
                continue
            mapping = exact_source_mapping(e["source_times"], normal["source_times"])
            matched_action_error = np.abs(normal["action"][mapping] - e["action"])
            matched_state_error = np.abs(normal["state"][mapping] - e["state"])
            action_peak_frame, action_peak_joint = np.unravel_index(
                matched_action_error.argmax(), matched_action_error.shape
            )
            state_peak_frame, state_peak_joint = np.unravel_index(
                matched_state_error.argmax(), matched_state_error.shape
            )
            source_gaps = np.diff(mapping)
            if kind == "speedup":
                variants = {f"source_frame_offset_{offset:+d}": mapping + offset for offset in [-2, -1, 0, 1, 2]}
                variants["next_selected_source_frame"] = np.r_[mapping[1:], mapping[-1]]
                for label, indices in variants.items():
                    valid = (indices >= 0) & (indices < len(normal["action"]))
                    valid[:2] = False
                    valid[-2:] = False
                    error = e["action"][valid] - normal["action"][indices[valid]]
                    for group, joint_indices in GROUPS.items():
                        source_shift_rows.append(
                            dict(
                                original_episode_id=original,
                                sample_id=e["record"]["id"],
                                alignment=label,
                                group=group,
                                **metrics(error[:, joint_indices]),
                            )
                        )
            identical = e["action"].shape == normal["action"].shape and np.array_equal(e["action"], normal["action"])
            comparisons.append(
                {
                    "original_episode_id": original,
                    "kind": kind,
                    "sample_id": e["record"]["id"],
                    "normal_sample_id": normal["record"]["id"],
                    "rows": len(e["action"]),
                    "normal_rows": len(normal["action"]),
                    "exact_actions": identical,
                    "matched_action_max_error": float(matched_action_error.max()),
                    "matched_action_mae": float(matched_action_error.mean()),
                    "matched_action_peak_joint": ORDER[action_peak_joint],
                    "matched_action_peak_frame": int(action_peak_frame),
                    "matched_action_peak_normal_frame": int(mapping[action_peak_frame]),
                    "matched_state_max_error": float(matched_state_error.max()),
                    "matched_state_mae": float(matched_state_error.mean()),
                    "matched_state_peak_joint": ORDER[state_peak_joint],
                    "matched_state_peak_frame": int(state_peak_frame),
                    "mapping_backward_steps": int((np.diff(mapping) < 0).sum()),
                    "source_frame_step_min": int(source_gaps.min()),
                    "source_frame_step_max": int(source_gaps.max()),
                    "mapping_status": "exact_int64_source_timestamp_ns",
                }
            )
            for row in window_rows:
                if row["degree"] != 6 or row["group"] != "hands" or row["sample_id"] != e["record"]["id"]:
                    continue
                nt = int(mapping[row["peak_frame"]])
                ns = max(2, min(len(normal["action"]) - 52, nt - row["peak_tick"]))
                op = operators(6)
                y = normal["action"][ns : ns + 50]
                error = op["ev"] @ op["inverse"] @ y - y
                left = int(mapping[row["frame_start"]])
                right = int(mapping[row["frame_end"]])
                dense = normal["action"][left : right + 1]
                dense_ticks = np.arange(len(dense))
                dense_basis = basis(dense_ticks, 6, 0, len(dense) - 1, 30)
                dense_pred = dense_basis @ solver_operator(dense_basis, "svd") @ dense
                dense_error = dense_pred - dense
                if kind == "speedup" and row["rank"] == 1:
                    j = ORDER.index(row["joint"])
                    st = row["frame_start"]
                    speed_target = e["action"][st : st + 50]
                    speed_pred = op["ev"] @ op["inverse"] @ speed_target
                    origin_ns = normal["source_times"][left]
                    normal_seconds = (normal["source_times"][left : right + 1] - origin_ns) / 1e9
                    speed_seconds = (e["source_times"][st : st + 50] - origin_ns) / 1e9
                    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
                    ax.plot(normal_seconds, dense[:, j], "k-", label="normal real, exact same source interval")
                    ax.scatter(
                        speed_seconds,
                        speed_target[:, j],
                        s=10,
                        color="green",
                        label="speedup real, exact timestamp subset",
                    )
                    ax.plot(normal_seconds, dense_pred[:, j], label="K6 fit on dense normal source interval")
                    ax.plot(speed_seconds, speed_pred[:, j], label="K6 native50 speedup fit, plotted at source times")
                    ax.axvline(speed_seconds[row["peak_tick"]], color="red", alpha=0.4)
                    ax.set(
                        xlabel="source time from window start (s)",
                        ylabel="absolute target (rad)",
                        title=f"Exact source timestamp alignment: {row['joint']}",
                    )
                    ax.legend(fontsize=8)
                    fig.savefig(a.output / "exact_source_worst_k6.png", dpi=140)
                    plt.close(fig)
                correspondence.append(
                    {
                        "original_episode_id": original,
                        "kind": kind,
                        "rank": row["rank"],
                        "frame_start": row["frame_start"],
                        "peak_frame": row["peak_frame"],
                        "normal_native50_start_exact_peak_anchor": ns,
                        "normal_peak_exact_frame": nt,
                        "normal_same_source_start": left,
                        "normal_same_source_end": right,
                        "normal_same_source_rows": len(dense),
                        "source_span_seconds": float(
                            (e["source_times"][row["frame_end"]] - e["source_times"][row["frame_start"]]) / 1e9
                        ),
                        "normal_source_span_hand_rmse": metrics(dense_error[:, GROUPS["hands"]])["rmse"],
                        "normal_source_span_hand_max": float(np.abs(dense_error[:, GROUPS["hands"]]).max()),
                        "original_peak_error": row["peak_error_rad"],
                        "normal_hand_max": float(np.abs(error[:, GROUPS["hands"]]).max()),
                        "normal_hand_rmse": metrics(error[:, GROUPS["hands"]])["rmse"],
                        "matched_action_max_error": float(
                            matched_action_error[row["frame_start"] : row["frame_end"] + 1].max()
                        ),
                        "matched_state_max_error": float(
                            matched_state_error[row["frame_start"] : row["frame_end"] + 1].max()
                        ),
                        "alignment_status": "exact_int64_source_timestamp_ns",
                    }
                )
    payload = {
        "episodes": len(episodes),
        "windows": int(sum(len(e["starts"]) for e in episodes)),
        "manifest_sha256": hashlib.sha256((a.samples / "manifest.json").read_bytes()).hexdigest(),
        "quantile_definition": "absolute scalar error pooled over windows,time,joints; per_joint.csv and top20_windows.csv preserve local failures",
        "degree_definition": "K highest degree, K+1 coefficients",
        "source_time_provenance": source_manifest,
        "stats": stats_rows,
        "boundary": boundary_rows,
        "sparse": sparse_results,
        "qr_svd_max_prediction_difference": max_solver_delta,
        "matched_originals": comparisons,
        "corresponding_windows": correspondence,
        "source_shift_diagnostics": source_shift_rows,
    }
    (a.output / "diagnosis.json").write_text(json.dumps(payload, indent=2, default=lambda x: x.item()))
    for filename, rows in [
        ("summary.csv", stats_rows),
        ("per_joint.csv", joint_rows),
        ("top20_windows.csv", window_rows),
        ("boundaries.csv", boundary_rows),
        ("sparse_solvers.csv", sparse_results),
        ("matched_originals.csv", comparisons),
        ("corresponding_windows.csv", correspondence),
        ("source_shift_diagnostics.csv", source_shift_rows),
    ]:
        write_csv(a.output / filename, rows)
    print(
        json.dumps(
            {
                "episodes": payload["episodes"],
                "windows": payload["windows"],
                "k6": [r for r in stats_rows if r["degree"] == 6 and r["group"] == "hands"],
                "qr_svd_max_difference": max_solver_delta,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
