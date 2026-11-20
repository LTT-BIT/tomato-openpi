"""Read-only CPU check of the real four-view history pipeline; never starts training."""

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np
import torch

from openpi.models.model import Observation
from openpi.subtasks.visual_memory import HISTORY
from openpi.subtasks.visual_memory import VisualHistory
from openpi.training import data_loader
from openpi.training.data_loader import _collate_fn as collate_fn
from scripts.run_training import load_experiment
from scripts.run_training import preflight

MODULE = "training_configs.experiments.tomato_sharpa.subtasks.visual_memory_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")
    if args.report and args.report.exists():
        raise FileExistsError("Use a fresh report path")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("Run this check with JAX_PLATFORMS=cpu")
    cfg = load_experiment(MODULE)
    metadata = preflight(cfg)
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    source = data_loader.create_torch_dataset(data, cfg.model.action_horizon, cfg.model, sampling_mode="epoch_shuffle")
    # Prompt transform is outside the history wrapper; retain its complete source IDs.
    raw = source
    while isinstance(raw, data_loader.TransformedDataset):
        raw = raw._dataset  # noqa: SLF001
    spec = cfg.model.memory
    starts = np.r_[0, np.flatnonzero(raw.episodes[1:] != raw.episodes[:-1]) + 1]
    indices = sorted(
        {
            0,
            min(len(source) - 1, (spec.frames - 1) * spec.stride),
            len(source) // 2,
            len(source) - 1,
            int(starts[-1]),
            max(0, int(starts[-1]) - 1),
        }
    )
    transformed = data_loader.transform_dataset(source, data)
    checks = []
    for index in indices:
        item = source[index]
        obs = transformed[index]
        if set(obs["image"]) != set(spec.all_image_keys) or obs["actions"].shape != (50, 58):
            raise ValueError("Transformed history/action dimensions differ")
        if any(image.shape != (224, 224, 3) for image in obs["image"].values()):
            raise ValueError("All model images must be 224 RGB")
        history = item[HISTORY]
        # Feed retained exact timeline slots through the online queue and compare pixels/masks.
        online = VisualHistory(spec)
        frame = int(raw.frame_indices[index])
        for previous in sorted({index - lag for lag in spec.lags if frame >= lag} | {index}):
            value = raw.dataset[previous]
            result = online.observe(
                value,
                episode_id=str(raw.episodes[index]),
                frame_index=int(raw.frame_indices[previous]),
                timestamp=float(raw.frame_indices[previous]) / spec.fps,
            )
        for key in spec.history_keys:
            np.testing.assert_array_equal(result[HISTORY]["image"][key], history["image"][key])
            if result[HISTORY]["image_mask"][key] != history["image_mask"][key]:
                raise ValueError("Offline/online mask disagreement")
        checks.append(
            {
                "index": index,
                "episode": int(raw.episodes[index]),
                "frame": frame,
                "valid_history_views": sum(bool(v) for v in history["image_mask"].values()),
            }
        )
    loader_options = {"multiprocessing_context": "spawn"} if args.workers else {}
    subset = torch.utils.data.Subset(transformed, indices[:2])
    loader = torch.utils.data.DataLoader(
        subset, batch_size=2, num_workers=args.workers, collate_fn=collate_fn, **loader_options
    )
    batch = next(iter(loader))
    parsed = Observation.from_dict(batch)
    if parsed.state.shape != (2, 58) or len(parsed.images) != len(spec.all_image_keys):
        raise ValueError("Worker batch failed model observation parsing")
    report = {
        "schema": "visual_memory_input_smoke_v1",
        "config": cfg.name,
        "experiment": cfg.exp_name,
        "contract": spec.contract(),
        "source": metadata,
        "loaded_frames": len(source),
        "loaded_episodes": len(starts),
        "checks": checks,
        "workers": args.workers,
        "worker_batch_actions": list(batch["actions"].shape),
        "rgb_cache": os.environ.get("PI05_RGB_FRAME_CACHE"),
        "batch256_gpu_capacity_verified": False,
        "training_started": False,
        "robot_executed": False,
        "memory_benefit_measured": False,
        "backend": jax.default_backend(),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
