"""Read-only Fovea preparation check, actual video samples, transforms and model specs."""

import argparse
import json
from pathlib import Path
import time

import jax
import numpy as np

from openpi.models.model import Observation
from openpi.subtasks import fovea
from openpi.training import data_loader
from openpi.training.fovea_dataset import digest
from scripts.run_training import load_experiment
from scripts.run_training import preflight

MODULE = "training_configs.experiments.tomato_sharpa.subtasks.foveavla_v1"


def check():
    config = load_experiment(MODULE)
    inputs = preflight(config)
    data = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(
        data, config.model.action_horizon, config.model, sampling_mode=config.sampling_mode
    )
    # TransformedDataset adds the task prompt around FoveaDataset.
    source = raw._dataset  # noqa: SLF001
    transformed = data_loader.transform_dataset(raw, data)
    indices = np.unique(np.linspace(0, len(raw) - 1, 12, dtype=int))
    samples = []
    for i in indices:
        row = raw[int(i)]
        actual = transformed[int(i)]
        obs = Observation.from_dict(jax.tree.map(np.asarray, actual))
        assert set(obs.images) == set(fovea.IMAGE_KEYS)
        assert actual["actions"].shape == (50, 58)
        assert all(np.asarray(x).shape == (224, 224, 3) for x in obs.images.values())
        assert obs.foveation_tokens.shape == (4,)
        ep, frame = int(row["episode_index"]), int(row["frame_index"])
        expected = fovea.encode_box(source.labels[ep][0][frame + 8], 640, 400)
        np.testing.assert_array_equal(obs.foveation_tokens, expected)
        samples.append(
            {
                "dataset_index": int(i),
                "episode": ep,
                "frame": frame,
                "future_frame": frame + 8,
                "prompt": row["prompt"],
                "token_length": int(np.asarray(obs.tokenized_prompt_mask).sum()),
            }
        )
    # Every retained pair has both annotations and stays in the same episode.
    columns = source.hf_dataset.select_columns(["episode_index", "frame_index"]).with_format("numpy")[:]
    for ep in np.unique(columns["episode_index"]):
        frames = columns["frame_index"][columns["episode_index"] == ep]
        boxes, valid = source.labels[int(ep)]
        assert (frames + 8 < len(boxes)).all()
        assert valid[frames].all()
        assert valid[frames + 8].all()
    root = Path(config.data.labels_root)
    started = time.monotonic()
    loader = data_loader.create_torch_data_loader(
        data,
        config.model,
        50,
        2,
        sharding=jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0]),
        num_workers=2,
        num_batches=1,
        sampling_mode="epoch_shuffle",
        num_train_steps=1,
        seed=42001,
    )
    batch_obs, batch_actions = next(iter(loader))
    assert batch_actions.shape == (2, 50, 58)
    assert batch_obs.foveation_tokens.shape == (2, 4)
    return {
        "status": "PASS",
        "training_started": False,
        "hardware_executed": False,
        "source_episodes": 119,
        "source_frames": 357970,
        "training_pairs": len(raw),
        "excluded_query_starts": 357970 - len(raw),
        "all_pairs_checked": len(raw),
        "actual_video_transform_samples": samples,
        "worker_loader_batch_shape": list(batch_actions.shape),
        "worker_loader_startup_seconds": time.monotonic() - started,
        "norm_provenance": inputs["norm_provenance"],
        "manifest_sha256": digest(root / "manifest.json"),
        "memory_enabled": False,
        "weak_box_supervision": True,
        "independent_semantic_accuracy_measured": False,
        "gpu_batch_capacity_verified": False,
        "config_module": MODULE,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = check()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
