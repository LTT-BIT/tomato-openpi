"""Aggregate frozen paired interventions; never interpret sensitivity as success."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.tomato_prompt_probe import GROUPS
from scripts.tomato_prompt_probe import PROMPT_KEYS
from scripts.tomato_prompt_probe import paired_metrics
from scripts.tomato_prompt_probe import sha

CLASSES = (0, 1, 2, 3, 0, 1, 2)


def distribution(values):
    x = np.asarray(values, dtype=float)
    return (
        {
            "n": len(x),
            "mean": float(x.mean()),
            "p10": float(np.quantile(x, 0.1)),
            "median": float(np.median(x)),
            "p90": float(np.quantile(x, 0.9)),
        }
        if len(x)
        else {"n": 0}
    )


def summarize_rows(rows, metric="metrics"):
    result = {}
    for name in GROUPS:
        result[name] = {}
        for horizon in (1, 5, 15, 50):
            items = [r[metric][name][str(horizon)] for r in rows]
            factual = [v["mse"][r["factual_prompt"]] for r, v in zip(rows, items, strict=True)]
            result[name][str(horizon)] = {
                "factual_mse": distribution(factual),
                "factual_strictly_lower_error_fraction": float(
                    np.mean([v["best_wrong_minus_factual"] > 0 for v in items])
                ),
                "factual_tied_fraction": float(np.mean([v["best_wrong_minus_factual"] == 0 for v in items])),
                "min_wrong_minus_factual": distribution([v["best_wrong_minus_factual"] for v in items]),
                "changed_prompt_rms": distribution(
                    [
                        v["prompt_change_rms"][k]
                        for r, v in zip(rows, items, strict=True)
                        for k in range(4)
                        if k != r["factual_prompt"]
                    ]
                ),
            }
    return result


def paired_noise(files):
    results = defaultdict(lambda: defaultdict(list))
    for row, path in files:
        f = np.load(path)
        a, b = f["absolute_s42"], f["absolute_s43"]
        for group, dims in GROUPS.items():
            n = min(15, row["phase_remaining_frames"])
            aa, bb = a[:, :n, dims], b[:, :n, dims]
            prompt_effect = float(np.sqrt(np.mean((aa[0] - aa[2]) ** 2)))
            prompt_effect2 = float(np.sqrt(np.mean((bb[0] - bb[2]) ** 2)))
            noise_effect = float(np.sqrt(np.mean((aa - bb) ** 2)))
            results[group]["pickup_vs_return_rms_rad"].append((prompt_effect + prompt_effect2) / 2)
            results[group]["noise_rms_rad"].append(noise_effect)
            if noise_effect > 1e-10:
                results[group]["pickup_vs_return_to_noise_ratio"].append(
                    (prompt_effect + prompt_effect2) / 2 / noise_effect
                )
    return {g: {k: distribution(v) for k, v in metric.items()} for g, metric in results.items()}


def boundary_comparison(rows):
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        for tag in row["tags"]:
            if tag["kind"] != "boundary":
                continue
            j, offset = tag["boundary"], tag["offset"]
            old, new = CLASSES[j - 1], CLASSES[j]
            for group in GROUPS:
                m = row["physical_metrics"][group]["15"]["mse"]
                out[str(j)][str(offset)][group].append({"old": m[old], "new": m[new]})
    return {
        j: {
            offset: {
                group: {
                    "samples": len(v),
                    "old_prompt_mse_rad2": distribution([x["old"] for x in v]),
                    "new_prompt_mse_rad2": distribution([x["new"] for x in v]),
                    "new_prompt_closer_to_recorded_action_fraction": float(np.mean([x["new"] < x["old"] for x in v])),
                }
                for group, v in groups.items()
            }
            for offset, groups in offsets.items()
        }
        for j, offsets in out.items()
    }


def run(root, output):
    if output.exists():
        raise FileExistsError("Choose a new result summary")
    rows = []
    steady_timings = []
    sources = []
    files = []
    seen = set()
    for folder in sorted(root.glob("shard*")):
        p = json.loads((folder / "protocol.json").read_text())
        complete = json.loads((folder / "completed.json").read_text())
        assert complete["status"] == "completed"
        assert complete["protocol_sha256"] == sha(folder / "protocol.json")
        if sources:
            previous = sources[0]
            for key in (
                "checkpoint_metadata_sha256",
                "params_metadata_sha256",
                "probe_code_sha256",
                "norm_stats_sha256",
                "selection_sha256",
                "prompt_order",
                "seeds",
            ):
                assert p[key] == previous[key], key
        sources.append(p)
        part = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        assert len(part) == complete["calls"] == len(p["plan"]) * 2
        steady_timings.extend(r["seconds"] for r in part[1:])
        for r in part:
            key = (r["episode"], r["frame"], r["seed"])
            assert key not in seen
            seen.add(key)
            action_path = folder / f"ep{r['episode']:03d}_f{r['frame']:06d}.npz"
            with np.load(action_path) as actions:
                r["physical_metrics"] = paired_metrics(
                    actions[f"absolute_s{r['seed']}"],
                    actions["target_absolute"],
                    r["factual_prompt"],
                    r["phase_remaining_frames"],
                )
            rows.append(r)
            if r["seed"] == 42:
                files.append((r, folder / f"ep{r['episode']:03d}_f{r['frame']:06d}.npz"))
    assert {r["episode"] for r in rows} == set(range(93))
    assert {r["seed"] for r in rows} == {42, 43}
    phases = {
        str(phase): summarize_rows([r for r in rows if r["phase"] == phase], "physical_metrics") for phase in range(7)
    }
    matrices = {}
    for group in GROUPS:
        matrix = []
        for label in range(4):
            selected = [r for r in rows if r["factual_prompt"] == label]
            matrix.append(np.mean([r["metrics"][group]["15"]["mse"] for r in selected], axis=0).tolist())
        matrices[group] = matrix
    result = {
        "version": "tomato-paired-prompt-probe-v1",
        "episodes": 93,
        "anchors": len(files),
        "seeds": [42, 43],
        "chunks": len(rows) * 4,
        "paired_calls": len(rows),
        "prompts": list(PROMPT_KEYS),
        "checkpoint": sources[0]["checkpoint"],
        "checkpoint_metadata_sha256": sources[0]["checkpoint_metadata_sha256"],
        "code_sha256": sources[0]["probe_code_sha256"],
        "norm_sha256": sources[0]["norm_stats_sha256"],
        "hosts": sorted({s["host"] for s in sources}),
        "scope": sources[0]["scope"],
        "normalization": "Original checkpoint stats; roundtrip to canonical absolute targets tested per anchor",
        "summary_normalized": summarize_rows(rows),
        "summary_physical_rad": summarize_rows(rows, "physical_metrics"),
        "phase_physical": phases,
        "confusion_mse_normalized_h15": matrices,
        "matched_noise_comparison_h15": paired_noise(files),
        "phase_matched_noise_comparison_h15": {
            str(phase): paired_noise([(r, p) for r, p in files if r["phase"] == phase]) for phase in range(7)
        },
        "summary_code_sha256": sha(__file__),
        "boundary_comparison": boundary_comparison(rows),
        "inference_call_s_excluding_first_per_shard": distribution(steady_timings),
        "token_counts": distribution([n for r in rows for n in r["token_counts"]]),
        "limitations": [
            "All93 are training recordings, no held-out/generalization success claim.",
            "Only original demonstration target exists; wrong-prompt correctness is unknown.",
            "Fixed-observation query is not a closed-loop prompt-switch or hold-until-switch test.",
            "Primary action prefixes stop at actual phase boundaries; no fabricated hold supervision.",
            "Pickup1/2, harvest1/2, return1/2 have identical language codes in this checkpoint.",
            "No new crop/phase-ID model was trained or deployed.",
        ],
        "inputs": [
            {
                "path": str(root / f"shard{i}"),
                "protocol_sha256": sha(root / f"shard{i}/protocol.json"),
                "metrics_sha256": sha(root / f"shard{i}/metrics.jsonl"),
            }
            for i in range(4)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    # Export a reproducible scientific figure, no screenshots or robot execution.
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (group, matrix) in zip(axes, matrices.items(), strict=True):
        m = np.array(matrix)
        m = m / (m.min(1, keepdims=True) + 1e-12)
        ax.imshow(m, cmap="viridis", vmin=1, vmax=min(3, float(m.max())))
        ax.set(
            xticks=range(4),
            xticklabels=PROMPT_KEYS,
            yticks=range(4),
            yticklabels=PROMPT_KEYS,
            title=group,
            xlabel="query prompt",
            ylabel="recorded phase",
        )
        ax.tick_params(axis="x", rotation=45)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white")
    fig.suptitle("In-training-data diagnostic: phase-pure 15-frame normalized MSE / row minimum (lower is better)")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(
        json.dumps({"episodes": 93, "anchors": len(files), "chunks": len(rows) * 4, "output": str(output)}), flush=True
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.root, a.output)
