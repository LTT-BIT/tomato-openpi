"""Atomic statistics for the exact complete windows used by opt-in paper training."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi import transforms
from openpi.shared import delta_paper as paper
from openpi.shared import normalize
from openpi.shared.local_storage import new_artifact_directory
from openpi.training import data_loader
from openpi.training import norm_provenance
from openpi.training.dataset_collection import collection_layout
from openpi.training.dataset_collection import validate_collection_features
from openpi.training.delta_paper import PaperDataView
from openpi.training.delta_paper import PaperSharpaData


def compute(config, *, batch_size=128):
    data = config.data
    if not isinstance(data, PaperSharpaData) or config.sampling_mode != "epoch_shuffle" or batch_size < 1:
        raise ValueError("Paper statistics require its own data adapter, epoch_shuffle and positive batch size")
    with new_artifact_directory(data.asset_directory(config.assets_dirs)) as output:
        _compute(config, batch_size=batch_size, output=output)


def _compute(config, *, batch_size, output):
    data, horizon = config.data, config.model.action_horizon
    layout = collection_layout([spec.root for spec in data.lerobot_datasets])
    validate_collection_features([item["info"] for item in layout])
    if any(item["info"]["fps"] != 30 for item in layout):
        raise ValueError("Paper episode metadata/FPS mismatch")
    total = sum(max(0, row["length"] - horizon + 1) for item in layout for row in item["episodes"])
    source_reports = [
        {
            "repo_id": spec.repo_id,
            "root": item["root"],
            "frames": item["info"]["total_frames"],
            "episodes": item["info"]["total_episodes"],
            "training_windows": sum(max(0, row["length"] - horizon + 1) for row in item["episodes"]),
        }
        for spec, item in zip(data.lerobot_datasets, layout, strict=True)
    ]
    if total < 2:
        raise ValueError("Paper statistics require at least two complete training windows")
    contract = norm_provenance.build_contract(config)
    repack, vectors = data.create_vector_transforms(config.model)
    vector = transforms.compose([*repack.inputs, *vectors.inputs])
    cameras = config.policy_metadata["camera_keys"]
    dummy = np.zeros((1, 1, 3), np.uint8)
    source_view = PaperDataView(
        repo_id=data.repo_id,
        action_sequence_keys=("action",),
        lerobot_datasets=data.lerobot_datasets,
        action_hz=30,
        prompt_from_task=True,
        video_backend="pyav",
    )
    reference = data_loader.create_torch_dataset(source_view, horizon, config.model, sampling_mode="epoch_shuffle")
    if len(reference) != total:
        raise ValueError("Paper vector statistics and training window selection differ")
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    arms = paper.Moments()
    chosen = (
        np.sort(
            np.random.default_rng(data.correlation_seed).choice(
                total, min(total, data.correlation_max_windows), replace=False
            )
        )
        if data.paper_beta
        else np.array([], np.int64)
    )
    sampled = np.empty((len(chosen), horizon, 58), np.float32)
    offset, checked = 0, 0
    for item in layout:
        root, info = Path(item["root"]), item["info"]
        for episode in sorted(item["episodes"], key=lambda row: row["episode_index"]):
            index, length = episode["episode_index"], episode["length"]
            count = max(0, length - horizon + 1)
            if not count:
                continue
            path = root / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
            table = pq.read_table(path, columns=["observation.state", "action", "frame_index", "episode_index"])
            if (
                len(table) != length
                or not np.array_equal(table["frame_index"].to_numpy(), np.arange(length))
                or not np.all(table["episode_index"].to_numpy() == index)
            ):
                raise ValueError("Paper episode frame IDs/length differ from metadata")
            state = np.asarray(table["observation.state"].to_pylist(), np.float32)
            action = np.asarray(table["action"].to_pylist(), np.float32)
            for start in range(0, count, batch_size):
                rows = np.arange(start, min(start + batch_size, count))
                inputs = {
                    "observation.state": state[rows],
                    "action": action[rows[:, None] + np.arange(horizon)],
                    "prompt": "",
                    **dict.fromkeys(cameras, dummy),
                }
                values = vector(inputs)
                if not np.isfinite(values["state"]).all() or not np.isfinite(values["actions"]).all():
                    raise ValueError("Nonfinite paper training window")
                stats["state"].update(values["state"])
                stats["actions"].update(values["actions"])
                arms.update(values["actions"][..., paper.ARMS])
                low, high = np.searchsorted(chosen, [offset + start, offset + start + len(rows)])
                if high > low:
                    sampled[low:high] = values["actions"][chosen[low:high] - offset - start]
                for local in (0, count // 2, count - 1):
                    if start <= local < start + len(rows):
                        actual = vector(reference[offset + local])
                        for key in stats:
                            np.testing.assert_array_equal(actual[key], values[key][local - start])
                        checked += 1
            offset += count
    if offset != total or norm_provenance.build_contract(config) != contract:
        raise ValueError("Paper data changed during statistics or window count differs")
    results = {key: value.get_statistics() for key, value in stats.items()}
    action_stats = results["actions"]
    arrays = {
        field: np.broadcast_to(getattr(action_stats, field), (horizon, 58)).copy()
        for field in ("mean", "std", "q01", "q99")
    }
    if data.paper_mode == "timestamp_zscore_arms":
        arm_mean, arm_std = arms.mean, arms.std()
        arrays["mean"][:, paper.ARMS], arrays["std"][:, paper.ARMS] = arm_mean, arm_std
    else:
        arm_mean = np.broadcast_to(action_stats.mean[paper.ARMS], (horizon, 14)).copy()
        arm_std = np.broadcast_to(action_stats.std[paper.ARMS], (horizon, 14)).copy()
    results["actions"] = normalize.NormStats(**arrays)
    # Hx58 arrays distinguish this asset family from existing per-dimension deployments.
    normalize.save(output, results)
    bridge = paper.ArmNormalization(
        data.paper_mode, arm_mean, arm_std, arrays["q01"], arrays["q99"], data.paper_std_floor, "building"
    )
    if data.paper_beta:
        normalized = transforms.Normalize(results, use_quantiles=True)({"actions": sampled})
        encoded = bridge(normalized)["actions"].reshape(len(sampled), horizon * 58)
        chol = paper.correlation_factor(encoded, data.paper_beta)
    else:
        chol = np.empty((0, 0), np.float32)
    np.savez_compressed(output / paper.BUNDLE, arm_mean=arm_mean, arm_std=arm_std, noise_cholesky=chol)
    digest = paper.sha256(output / "norm_stats.json")
    (output / "norm_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": norm_provenance.SCHEMA_VERSION,
                "contract": contract,
                "norm_stats_sha256": digest,
                "config_name": config.name,
                "repo_id": data.repo_id,
                "frames": sum(item["frames"] for item in source_reports),
                "episodes": sum(item["episodes"] for item in source_reports),
                "datasets": source_reports,
                "action_horizon": horizon,
                "training_windows": total,
                "vector_counts": {"state": total, "actions": total * horizon},
                "lerobot_comparison_samples": checked,
                "sampling": "complete windows; each start once; no padding",
            },
            indent=2,
        )
        + "\n"
    )
    metadata = {
        **data.paper_contract(),
        "shape": [horizon, 58],
        "flatten_order": "time-major",
        "training_windows": total,
        "excluded_tail_starts": sum(item["frames"] for item in source_reports) - total,
        "datasets": source_reports,
        "correlation_samples": len(chosen),
        "correlation_estimator": "centered unit-diagonal correlation; constant dims independent",
        "norm_stats_sha256": digest,
        "bundle_sha256": paper.sha256(output / paper.BUNDLE),
        "source_contract": contract,
        "prompt_source": "meta/tasks.jsonl",
        "config_name": config.name,
    }
    (output / paper.MANIFEST).write_text(json.dumps(metadata, indent=2) + "\n")
    paper.load_bundle(
        output, mode=data.paper_mode, beta=data.paper_beta, horizon=horizon, std_floor=data.paper_std_floor
    )
