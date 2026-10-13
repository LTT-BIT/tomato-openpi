"""Compute supported robot vector statistics without decoding every video frame.

Uses the configured repack/state/action/delta transforms and LeRobot's forward
action windows, clamped at each episode boundary. All frames, including the final
partial batch, are included. Boundary/middle samples are checked against LeRobot.
Only the explicitly allowed image-independent robot transforms are supported.

Example:
    uv run examples/fr3_wuji/compute_norm_stats.py --config-name pi05_fr3_wuji_125ep
"""

from dataclasses import replace
import json
import logging
from pathlib import Path

from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

from openpi import transforms
from openpi.policies import fr3_sharpa_policy
from openpi.policies import fr3_wuji_policy
from openpi.policies import lemon_policy
from openpi.policies import single_franka_policy
from openpi.shared import normalize
from openpi.shared.local_storage import new_artifact_directory
from openpi.training import config as training_config
from openpi.training import data_loader
from openpi.training import norm_provenance
from openpi.training import reproducibility


def transform_vectors(
    state,
    actions,
    rows,
    horizon,
    transform,
    *,
    camera_keys=("observation.images.cam0", "observation.images.cam1", "observation.images.cam2"),
):
    indices = np.minimum(rows[:, None] + np.arange(horizon), len(state) - 1)
    # Images cannot affect state/actions for the explicitly allowed transforms below.
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    raw = {
        "observation.state": state[rows],
        "action": actions[indices],
        "prompt": "",
        **dict.fromkeys(camera_keys, image),
    }
    return transform(raw)


def main(config_name: str = "pi05_fr3_wuji_125ep", batch_size: int = 512):
    compute(training_config.get_config(config_name), batch_size=batch_size)


def compute(config: training_config.TrainConfig, *, batch_size: int = 512):
    """Compute statistics for registered or standalone training configurations."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not isinstance(
        config.data,
        training_config.LeRobotFr3WujiDataConfig
        | training_config.LeRobotFr3SharpaDataConfig
        | training_config.LeRobotSingleFrankaDataConfig,
    ):
        raise ValueError("Only FR3/Wuji, FR3/Sharpa and single Franka LeRobot data adapters are supported")
    asset_id = config.data.assets.asset_id or config.data.repo_id
    if asset_id is None:
        raise ValueError("A normalization asset_id is required")
    output = Path(config.data.assets.assets_dir or config.assets_dirs) / asset_id
    with new_artifact_directory(output) as staging:
        _compute(config, batch_size=batch_size, output=staging)
    print(f"Normalization bundle published to {output}")


def _compute(config: training_config.TrainConfig, *, batch_size: int, output: Path):
    if getattr(config.data.base_config, "task_view", None) is not None:
        from openpi.training import task_view_norm

        task_view_norm.compute(config, batch_size=batch_size, output=output)
        return
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.local_dataset_loader is not None or tuple(data_config.action_sequence_keys) != ("action",):
        raise ValueError("Requires standard LeRobot data with the action sequence key 'action'")
    if any(type(t) is not transforms.RepackTransform for t in data_config.repack_transforms.inputs):
        raise ValueError("Unsupported repack transforms")
    if any(
        type(t)
        not in (
            fr3_wuji_policy.Fr3WujiInputs,
            fr3_sharpa_policy.Fr3SharpaInputs,
            single_franka_policy.SingleFrankaInputs,
            lemon_policy.LemonInputs,
            transforms.DeltaActions,
        )
        for t in data_config.data_transforms.inputs
    ):
        raise ValueError("Unsupported transforms: image-dependent transforms cannot use vector-only statistics")
    if data_config.asset_id is None:
        raise ValueError("A normalization asset_id is required")
    dataset_specs = data_config.lerobot_datasets or (training_config.LeRobotDatasetSpec(repo_id=data_config.repo_id),)
    sources = []
    for spec in dataset_specs:
        root = Path(spec.root) if spec.root is not None else HF_LEROBOT_HOME / spec.repo_id
        info = json.loads((root / "meta/info.json").read_text())
        if data_config.action_hz is not None and data_config.action_hz != info["fps"]:
            raise ValueError("Vector-only statistics currently require action_hz to match source FPS")
        episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
        sources.append((spec, root, info, episodes))
    specs_with_roots = tuple(replace(spec, root=str(root)) for spec, root, _, _ in sources)
    contract = norm_provenance.build_contract(config, dataset_specs=specs_with_roots)
    transform = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    reference = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    norm_dim = (config.policy_metadata or {}).get("normalization_dim", config.model.action_dim)
    norm_dims = (config.policy_metadata or {}).get("normalization_dims", dict.fromkeys(stats, norm_dim))
    totals = {key: np.zeros(norm_dims[key], dtype=np.float64) for key in stats}
    squares = {key: np.zeros(norm_dims[key], dtype=np.float64) for key in stats}
    counts = dict.fromkeys(stats, 0)
    offset = 0
    checked = 0
    for spec, root, info, episodes in sources:
        source_offset = 0
        camera_keys = tuple(key for key in info["features"] if key.startswith("observation.images."))
        for episode in tqdm.tqdm(episodes, desc=f"Vector norm ({spec.repo_id})"):
            index, length = episode["episode_index"], episode["length"]
            path = root / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index)
            table = pq.read_table(path, columns=["observation.state", "action"])
            state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            assert len(table) == length
            rows = np.array([0, length // 2, length - 1])
            actual = transform_vectors(
                state, actions, rows, config.model.action_horizon, transform, camera_keys=camera_keys
            )
            for i, row in enumerate(rows):
                raw = reference[offset + source_offset + int(row)]
                assert int(raw["episode_index"]) == index
                expected = transform(raw)
                for key in stats:
                    np.testing.assert_array_equal(actual[key][i], expected[key])
                checked += 1
            for start in range(0, length, batch_size):
                rows = np.arange(start, min(start + batch_size, length))
                batch = transform_vectors(
                    state, actions, rows, config.model.action_horizon, transform, camera_keys=camera_keys
                )
                for key, running_stats in stats.items():
                    values = np.asarray(batch[key], dtype=np.float64).reshape(-1, norm_dims[key])
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite values in {spec.repo_id} episode {index}, {key}")
                    running_stats.update(values)
                    totals[key] += values.sum(axis=0)
                    squares[key] += (values**2).sum(axis=0)
                    counts[key] += len(values)
            source_offset += length
        assert source_offset == info["total_frames"]
        offset += source_offset
    assert offset == sum(source[2]["total_frames"] for source in sources) == len(reference)
    assert counts == {"state": offset, "actions": offset * config.model.action_horizon}
    result = {key: value.get_statistics() for key, value in stats.items()}
    for key, value in result.items():
        mean = totals[key] / counts[key]
        std = np.sqrt(np.maximum(0, squares[key] / counts[key] - mean**2))
        np.testing.assert_allclose(value.mean, mean, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(value.std, std, atol=1e-9, rtol=1e-9)
        assert all(np.isfinite(getattr(value, field)).all() for field in ("mean", "std", "q01", "q99"))
        if np.any(value.std < 0) or np.any(value.q99 < value.q01):
            raise ValueError(f"Invalid normalization range for {key}")
    constant_dimensions = {key: np.flatnonzero(value.std == 0).tolist() for key, value in result.items()}
    for key, dimensions in constant_dimensions.items():
        if dimensions:
            logging.warning("Zero-variance %s dimensions %s are retained; normalization uses epsilon", key, dimensions)
    if norm_provenance.build_contract(config, dataset_specs=specs_with_roots) != contract:
        raise ValueError("Normalization source data changed during computation; statistics were not saved")
    normalize.save(output, result)
    (output / "norm_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": norm_provenance.SCHEMA_VERSION,
                "contract": contract,
                "norm_stats_sha256": reproducibility.file_sha256(output / "norm_stats.json"),
                "constant_dimensions": constant_dimensions,
                "config_name": config.name,
                "repo_id": data_config.repo_id,
                "datasets": [
                    {
                        "repo_id": spec.repo_id,
                        "training_weight": spec.weight,
                        "episodes": len(episodes),
                        "frames": info["total_frames"],
                    }
                    for spec, _, info, episodes in sources
                ],
                "episodes": sum(len(source[3]) for source in sources),
                "frames": offset,
                "action_horizon": config.model.action_horizon,
                "vector_counts": counts,
                "lerobot_comparison_samples": checked,
                "accumulation_dtype": "float64",
                "sampling": (
                    "all source frames equally weighted for norm (training weights ignored); "
                    "forward action windows padded within each episode"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"PASS: {offset} frames, {checked} LeRobot sample comparisons; normalization bundle ready to publish")


if __name__ == "__main__":
    tyro.cli(main)
