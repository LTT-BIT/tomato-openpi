"""Current RGB crops plus future-box targets; exclude invalid and episode-tail pairs."""

import hashlib
import json
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch

from openpi.subtasks import fovea


class FoveaSourceDataset(LeRobotDataset):
    def _query_videos(self, query_timestamps, ep_idx):
        # The old, pre-rendered fourth view is replaced by a source-pixel crop.
        current = {k: v for k, v in query_timestamps.items() if k != "observation.images.interaction"}
        return super()._query_videos(current, ep_idx)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class FoveaDataset:
    def __init__(self, dataset, root, execute_steps):
        self.dataset = dataset
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        if (
            self.manifest["version"] != fovea.VERSION
            or not self.manifest["complete"]
            or execute_steps != self.manifest["execute_steps"]
        ):
            raise ValueError("Complete matching foveation label bundle required")
        for name, sha in self.manifest["dataset_metadata"].items():
            if digest(dataset.root / name) != sha:
                raise ValueError("Fovea source dataset metadata changed")
        self.labels = {}
        indices = []
        start = 0
        for record in self.manifest["episodes"]:
            path = root / record["labels"]
            if digest(path) != record["sha256"]:
                raise ValueError("Fovea labels changed")
            with np.load(path, allow_pickle=False) as data:
                boxes, valid = data["boxes"], data["valid"]
            n = record["frames"]
            ep = record["episode_index"]
            if boxes.shape != (n, 4) or valid.shape != (n,) or valid.dtype != bool:
                raise ValueError("Invalid foveation label arrays")
            self.labels[ep] = (boxes, valid)
            eligible = np.flatnonzero(valid[:-execute_steps] & valid[execute_steps:])
            indices.extend((eligible + start).tolist())
            start += n
        if start != len(dataset) or len(indices) != self.manifest["training_pairs"]:
            raise ValueError("Fovea source length or valid pairing changed")
        self.indices = np.asarray(indices, dtype=np.int64)
        self.hf_dataset = dataset.hf_dataset.select(self.indices.tolist())
        self.source_total_frames = len(dataset)
        self.view_identity = {
            "version": fovea.VERSION,
            "manifest_sha256": digest(root / "manifest.json"),
            "execute_steps": execute_steps,
            "source_total_frames": len(dataset),
        }
        self.execute_steps = execute_steps

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data = dict(self.dataset[int(self.indices[index])])
        ep, frame = int(data["episode_index"]), int(data["frame_index"])
        boxes, _ = self.labels[ep]
        raw = data["observation.images.cam0"]
        if isinstance(raw, torch.Tensor):
            raw = raw.numpy()
        raw = np.asarray(raw)
        if raw.shape[0] == 3:
            raw = raw.transpose(1, 2, 0)
        if np.issubdtype(raw.dtype, np.floating):
            raw = np.clip(raw * 255, 0, 255).astype(np.uint8)
        h, w = raw.shape[:2]
        if [w, h] != self.manifest["source_size"]:
            raise ValueError("Foveal crops require source-resolution cam0; disable RGB224 cache")
        # Stable mixed crops, independent of worker scheduling and resume order.
        seed = int(self.indices[index]) + 42002
        u = np.random.default_rng(seed).random()
        box = fovea.INITIAL_BOX if frame == 0 or u < 0.1 else boxes[frame]
        crop = fovea.crop_rgb(raw, box, perturb_seed=seed if 0.1 <= u < 0.55 and frame else None)
        data["observation.images.interaction"] = torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 255
        data["observation.interaction_available"] = np.True_
        data["foveation_tokens"] = fovea.encode_box(boxes[frame + self.execute_steps], w, h)
        return data
