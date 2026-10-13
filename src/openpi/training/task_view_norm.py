"""Normalization of explicitly selected samples using the training window builder."""

import json
from pathlib import Path

import numpy as np

from openpi import transforms
from openpi.shared import normalize
from openpi.training import data_loader
from openpi.training import norm_provenance
from openpi.training import reproducibility
from openpi.training.task_view import TaskViewDataset


def compute(config, *, batch_size, output):
    contract = norm_provenance.build_contract(config)
    data = config.data.create(config.assets_dirs, config.model)
    source = data_loader.create_torch_dataset(
        data, config.model.action_horizon, config.model, sampling_mode="epoch_shuffle"
    )
    if not isinstance(source, TaskViewDataset):
        raise ValueError("Expected task view dataset")
    transform = transforms.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs])
    cameras = config.policy_metadata["camera_keys"]
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    checked = {0, len(source) // 2, len(source) - 1}
    for start in range(0, len(source), batch_size):
        batch = {key: [] for key in stats}
        for index in range(start, min(start + batch_size, len(source))):
            raw = {**source.vectors(index), **dict.fromkeys(cameras, image)}
            vectors = transform(raw)
            if index in checked:
                actual = transform(source[index])
                for key in stats:
                    np.testing.assert_array_equal(vectors[key], actual[key])
            for key in stats:
                batch[key].append(vectors[key])
        for key, running_stats in stats.items():
            values = np.asarray(batch[key], dtype=np.float64)
            values = values.reshape(-1, values.shape[-1])
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite task view normalization input")
            running_stats.update(values)
    result = {key: value.get_statistics() for key, value in stats.items()}
    if norm_provenance.build_contract(config) != contract:
        raise ValueError("Task view/data changed during normalization")
    normalize.save(output, result)
    info = json.loads((source.root / "meta/info.json").read_text())
    Path(output, "norm_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": norm_provenance.SCHEMA_VERSION,
                "contract": contract,
                "norm_stats_sha256": reproducibility.file_sha256(Path(output, "norm_stats.json")),
                "config_name": config.name,
                "repo_id": data.repo_id,
                "frames": info["total_frames"],
                "episodes": info["total_episodes"],
                "training_frames": len(source),
                "training_episodes": len(np.unique(source.ids[:, 0])),
                "action_horizon": config.model.action_horizon,
                "vector_counts": {"state": len(source), "actions": len(source) * config.model.action_horizon},
                "lerobot_comparison_samples": len(checked),
                "sampling": "selected training frames only; phase windows and arm holds identical to training",
            },
            indent=2,
        )
        + "\n"
    )
