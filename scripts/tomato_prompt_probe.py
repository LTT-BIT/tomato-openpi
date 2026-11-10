"""Read-only paired prompt interventions on a frozen tomato B checkpoint.

This is an in-training-distribution diagnostic, not a success-rate evaluation.
It never executes actions, changes prompts on a robot, or updates weights.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

PROMPT_KEYS = ("pickup", "harvest", "return", "adjust")
GROUPS = {
    "left_arm": list(range(7)),
    "left_hand": list(range(7, 29)),
    "right_arm": list(range(29, 36)),
    "right_hand": list(range(36, 58)),
}


def noise_for(seed, episode, frame):
    return np.random.default_rng(np.random.SeedSequence([seed, episode, frame])).standard_normal(
        (50, 58), dtype=np.float32
    )


def sample_plan(annotations):
    """All stage midpoints + fixed boundary offsets, chosen without predictions."""
    result = []
    for episode, segments in enumerate(annotations):
        frames = {}
        length = segments[-1]["end_frame"]
        for phase, segment in enumerate(segments):
            middle = (segment["start_frame"] + segment["end_frame"] - 1) // 2
            frames.setdefault(middle, []).append({"kind": "midpoint", "phase": phase})
        for boundary, segment in enumerate(segments[1:], 1):
            for offset in (-45, -15, 0, 15):
                frame = segment["start_frame"] + offset
                if 0 <= frame < length:
                    frames.setdefault(frame, []).append({"kind": "boundary", "boundary": boundary, "offset": offset})
        for frame, tags in sorted(frames.items()):
            phase = next(k for k, s in enumerate(segments) if s["start_frame"] <= frame < s["end_frame"])
            result.append(
                {
                    "episode": episode,
                    "frame": frame,
                    "phase": phase,
                    "phase_remaining_frames": segments[phase]["end_frame"] - frame,
                    "tags": tags,
                }
            )
    return result


def paired_metrics(predictions, target, factual_index, remaining):
    """Compare canonical normalized actions; group metrics prevent 44 hands dominating.

    Wrong-prompt expert targets are unavailable. Differences only measure sensitivity.
    Scores against the original demonstration measure compatibility, not counterfactual correctness.
    """
    p, y = np.asarray(predictions), np.asarray(target)
    if p.shape != (4, 50, 58) or y.shape != (50, 58) or not np.isfinite(p).all():
        raise ValueError("Expected finite four-prompt canonical action chunks")
    if not 0 <= factual_index < 4 or remaining < 1:
        raise ValueError("Invalid factual phase")
    out = {}
    for group, dims in GROUPS.items():
        rows = {}
        for horizon in (1, 5, 15, 50):
            n = min(horizon, remaining)
            pp, yy = p[:, :n, dims], y[:n, dims]
            mse = np.mean((pp - yy[None]) ** 2, axis=(1, 2))
            factual = p[factual_index, :n][:, dims]
            sensitivity = [float(np.sqrt(np.mean((pp[k] - factual) ** 2))) for k in range(4)]
            others = np.delete(mse, factual_index)
            rows[str(horizon)] = {
                "evaluated_frames": int(n),
                "mse": mse.tolist(),
                "factual_best": bool(mse[factual_index] <= np.min(others)),
                "best_wrong_minus_factual": float(np.min(others) - mse[factual_index]),
                "prompt_change_rms": sensitivity,
            }
        out[group] = rows
    return out


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def json_save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def gpu_guard(index):
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    row = next(line.split(", ") for line in rows.splitlines() if line.split(", ")[0] == str(index))
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True
    )
    if int(row[2]) > 1024 or any(row[1] in line for line in processes.splitlines()):
        raise RuntimeError(f"GPU {index} is occupied; no process was stopped")
    return row


def run(args):
    # Runtime environment must be selected before importing JAX. Never silently use CPU/all GPUs.
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu) or os.environ.get("JAX_PLATFORMS") != "cuda":
        raise RuntimeError("Set JAX_PLATFORMS=cuda and CUDA_VISIBLE_DEVICES to the explicit single GPU")
    inventory = gpu_guard(args.gpu)
    import jax
    import jax.numpy as jnp

    from openpi import transforms
    from openpi.models import model as model_lib
    from openpi.shared import nnx_utils
    from openpi.subtasks.tomato import ACTION_PROMPTS
    from openpi.training import data_loader
    from scripts.run_training import load_experiment
    from scripts.run_training import preflight
    from training_configs import common

    common.configure_environment()
    config = load_experiment("training_configs.experiments.tomato_sharpa.subtasks.normal93_b_stage")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "_CHECKPOINT_METADATA").is_file():
        raise ValueError("A finalized checkpoint is required")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; frozen results are never overwritten")
    args.output.mkdir(parents=True)
    checked = preflight(config)
    original_assets = Path(config.data.assets.assets_dir) / config.data.assets.asset_id
    assets = checkpoint / "assets" / config.data.assets.asset_id
    if sha(assets / "norm_stats.json") != checked["norm_stats_sha256"]:
        raise ValueError("Checkpoint statistics differ from provenance-verified training statistics")
    config = replace(
        config, data=replace(config.data, assets=replace(config.data.assets, assets_dir=str(checkpoint / "assets")))
    )
    data = config.data.create(config.assets_dirs, config.model)
    source = data_loader.create_torch_dataset(data, 50, config.model, sampling_mode=config.sampling_mode)
    root = Path(config.data.lerobot_datasets[0].root)
    manifest = json.loads((root.parent / "bundle.json").read_text())
    annotations = [
        json.loads((root / f"annotations/episode_{ep:06d}.json").read_text())["segments"] for ep in range(93)
    ]
    plan = sample_plan(annotations)
    if args.episodes:
        allowed = set(args.episodes)
        plan = [row for row in plan if row["episode"] in allowed]
    if args.limit:
        plan = plan[: args.limit]
    if not plan:
        raise ValueError("No selected samples")
    first = {}
    for i, (ep, _, _) in enumerate(source.ids):
        first.setdefault(int(ep), i)
    protocol = {
        "version": "tomato-paired-prompt-probe-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_metadata_sha256": sha(checkpoint / "_CHECKPOINT_METADATA"),
        "params_metadata_sha256": sha(checkpoint / "params" / "_METADATA"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "probe_code_sha256": sha(__file__),
        "selection_sha256": config.policy_metadata["selection_sha256"],
        "norm_stats_sha256": sha(assets / "norm_stats.json"),
        "original_norm_metadata_sha256": sha(original_assets / "norm_metadata.json"),
        "checkpoint_statistics_verified_against_original_training_assets": True,
        "bundle_sha256": sha(root.parent / "bundle.json"),
        "prompts": ACTION_PROMPTS,
        "prompt_order": list(PROMPT_KEYS),
        "plan": plan,
        "seeds": args.seeds,
        "denoising_steps": 10,
        "dtype": "checkpoint parameter dtype, bf16 model operations",
        "gpu": inventory,
        "host": socket.gethostname(),
        "train_episodes": 93,
        "training_frames": 410456,
        "scope": "All93 were training data. Fixed-observation interventions only; no closed-loop or counterfactual target claims.",
        "empty_prompt": "not used in primary comparison",
        "untrained_numeric_or_crop_inputs": "not used to claim improvement",
        "phase_window": "Primary metrics stop at phase boundary; full raw50 predictions retained separately.",
        "policy_metadata": config.policy_metadata,
        "preflight": checked,
    }
    json_save(args.output / "protocol.json", protocol)
    print(
        json.dumps(
            {"stage": "loading", "samples": len(plan), "seeds": args.seeds, "total_calls": len(plan) * len(args.seeds)}
        ),
        flush=True,
    )
    gpu_guard(args.gpu)
    # Restore actual trained parameters, not pretrained base; no optimizer/train_state loaded.
    net = config.model.load(model_lib.restore_params(checkpoint / "params"))
    net.eval()

    predict = nnx_utils.module_jit(net.sample_actions)
    normalize_in = transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm)
    unnormalize_out = transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm)
    physical = transforms.compose(data.data_transforms.outputs)
    vectors = transforms.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs])
    _, absolute_vectors = replace(config.data, extra_delta_transform=False).create_vector_transforms(config.model)
    absolute_inputs = transforms.compose([*data.repack_transforms.inputs, *absolute_vectors.inputs])
    model_input = transforms.compose(data.model_transforms.inputs)
    started = time.perf_counter()
    log = (args.output / "metrics.jsonl").open("x")
    timings, done = [], 0
    for row in plan:
        ep, frame = row["episode"], row["frame"]
        raw = source[first[ep] + frame]
        canonical = vectors(raw)
        state = canonical["state"].copy()
        normalized = normalize_in(canonical)
        target = normalized["actions"].copy()
        inputs = [model_input({**normalized, "prompt": ACTION_PROMPTS[key]}) for key in PROMPT_KEYS]
        obs = model_lib.Observation.from_dict(jax.tree.map(lambda *xs: np.stack(xs), *inputs))
        for key, image in inputs[0]["image"].items():
            assert all(np.array_equal(image, item["image"][key]) for item in inputs[1:])
        assert all(np.array_equal(inputs[0]["state"], item["state"]) for item in inputs[1:])
        factual_index = PROMPT_KEYS.index(annotations[ep][row["phase"]]["label"])
        target_absolute = absolute_inputs(raw)["actions"].copy()
        reconstructed = physical(unnormalize_out({"actions": target.copy(), "state": normalized["state"].copy()}))[
            "actions"
        ]
        np.testing.assert_allclose(reconstructed, target_absolute, atol=2e-5, rtol=2e-5)
        saved = {"target_normalized": target, "target_absolute": target_absolute, "state": state}
        for seed in args.seeds:
            noise = np.repeat(noise_for(seed, ep, frame)[None], 4, axis=0)
            begin = time.perf_counter()
            predicted = np.asarray(predict(jax.random.key(0), obs, noise=jnp.asarray(noise)))
            seconds = time.perf_counter() - begin
            if done == 0:
                repeated = np.asarray(predict(jax.random.key(0), obs, noise=jnp.asarray(noise)))
                np.testing.assert_array_equal(predicted, repeated)
            if not np.isfinite(predicted).all():
                raise ValueError("Nonfinite inference output")
            absolute = np.stack(
                [
                    physical(unnormalize_out({"actions": x, "state": normalized["state"].copy()}))["actions"]
                    for x in predicted
                ]
            )
            result = dict(
                **row,
                seed=seed,
                factual_prompt=factual_index,
                seconds=seconds,
                token_counts=[int(x["tokenized_prompt_mask"].sum()) for x in inputs],
                metrics=paired_metrics(predicted, target, factual_index, row["phase_remaining_frames"]),
                source=manifest["mapping"][ep],
            )
            log.write(json.dumps(result, allow_nan=False) + "\n")
            log.flush()
            saved[f"normalized_s{seed}"] = predicted
            saved[f"absolute_s{seed}"] = absolute
            saved[f"noise_s{seed}"] = noise[0]
            done += 1
            timings.append(seconds)
            if done % 10 == 0 or done == 1:
                print(
                    json.dumps(
                        {
                            "stage": "inference",
                            "done": done,
                            "total": len(plan) * len(args.seeds),
                            "elapsed_s": time.perf_counter() - started,
                            "recent_s_per_call": float(np.mean(timings[-10:])),
                        }
                    ),
                    flush=True,
                )
        np.savez_compressed(args.output / f"ep{ep:03d}_f{frame:06d}.npz", **saved)
    log.close()
    json_save(
        args.output / "completed.json",
        {
            "status": "completed",
            "calls": done,
            "anchors": len(plan),
            "elapsed_s": time.perf_counter() - started,
            "steady_call_median_s": float(np.median(timings[1:])),
            "protocol_sha256": sha(args.output / "protocol.json"),
            "no_training": True,
            "no_robot": True,
        },
    )
    print("COMPLETED", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--episodes", type=int, nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
