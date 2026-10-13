"""Explicit episode/task selection and phase prompts without rewriting LeRobot data."""

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq

from openpi.training import reproducibility


@dataclass(frozen=True)
class TaskView:
    split_path: str | None = None
    split: str = "train"
    task_indices: tuple[int, ...] = ()
    prompt_overrides: tuple[tuple[int, str], ...] = ()
    phase_annotations_path: str | None = None
    phase_prompts: tuple[str, ...] = ()
    phase_arm_hold: bool = False
    task_boundary_hold: bool = False
    allow_unannotated: bool = False

    def __post_init__(self):
        if len(set(self.task_indices)) != len(self.task_indices) or any(i < 0 for i in self.task_indices):
            raise ValueError("task_indices must contain unique nonnegative IDs")
        ids = [key for key, _ in self.prompt_overrides]
        if len(ids) != len(set(ids)) or any(key < 0 for key in ids):
            raise ValueError("prompt_overrides must contain unique nonnegative task IDs")
        for prompt in (*self.phase_prompts, *(text for _, text in self.prompt_overrides)):
            if (
                not isinstance(prompt, str)
                or not prompt.strip()
                or not prompt.isascii()
                or not re.search("[A-Za-z]", prompt)
            ):
                raise ValueError("Task prompts must be nonempty English ASCII strings")
        if self.phase_annotations_path:
            if len(self.phase_prompts) != 5 or not self.split_path:
                raise ValueError("Phase supervision requires five prompts and an explicit episode selection")
            if self.task_indices or self.prompt_overrides:
                raise ValueError("Phase prompts cannot be combined with task selection/overrides")
        elif self.phase_prompts or self.phase_arm_hold or self.allow_unannotated:
            raise ValueError("Phase options require phase_annotations_path")

    def identity(self):
        values = asdict(self)
        for key in ("split_path", "phase_annotations_path"):
            path = values.pop(key)
            values[key + "_sha256"] = reproducibility.file_sha256(Path(path)) if path else None
        return values


FIELDS = ("left_support_ready_s", "scoop_ready_s", "lemon_supported_s", "pour_ready_s", "task_complete_s")


class TaskViewDataset:
    """Share one vector/window implementation between training and normalization."""

    def __init__(self, dataset, view, horizon):
        self.dataset = dataset
        self.view = view
        self.root = Path(dataset.root)
        self.horizon = horizon
        self.view_identity = view.identity()
        self.source_total_frames = len(dataset)
        ids = dataset.hf_dataset.select_columns(["episode_index", "frame_index", "task_index"]).with_format("numpy")[:]
        all_episodes = set(map(int, ids["episode_index"]))
        selected = all_episodes
        if view.split_path:
            splits = json.loads(Path(view.split_path).read_text())["splits"]
            groups = []
            for name, values in splits.items():
                if not values or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values):
                    raise ValueError(f"Invalid/empty split {name}")
                groups.append(set(values))
            for i, group in enumerate(groups):
                if any(group & other for other in groups[i + 1 :]):
                    raise ValueError("Train/validation/test episodes must not overlap")
            selected = set(splits[view.split])
            if not selected <= all_episodes:
                raise ValueError("Split contains missing episodes")
        tasks = set(map(int, ids["task_index"]))
        if not set(view.task_indices) <= tasks or not {i for i, _ in view.prompt_overrides} <= tasks:
            raise ValueError("Requested task IDs are absent from the dataset")
        annotations = {}
        if view.phase_annotations_path:
            document = json.loads(Path(view.phase_annotations_path).read_text())
            if document.get("schema_version") != 3 or document.get("boundary_fields") != list(FIELDS):
                raise ValueError("Expected five-boundary schema v3 annotations (LeRobot data remains v2.1)")
            annotations = {row["episode_index"]: row for row in document["episodes"]}
            if len(annotations) != len(document["episodes"]):
                raise ValueError("Duplicate phase annotation")
        self.records = {}
        for episode in sorted(selected):
            table = pq.read_table(self.root / dataset.meta.get_data_file_path(episode))
            states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
            if not np.isfinite(states).all() or not np.isfinite(actions).all():
                raise ValueError("Nonfinite state/action vectors")
            frames = np.asarray(table["frame_index"].to_numpy())
            if not np.array_equal(frames, np.arange(len(table))):
                raise ValueError("Episode frame_index must be contiguous from zero")
            boundaries = None
            if view.phase_annotations_path:
                row = annotations.get(episode)
                if row is None and not view.allow_unannotated:
                    raise ValueError(f"Missing phase annotation for episode {episode}")
                if row is not None:
                    boundaries = np.asarray([row["frame_indices"][field] for field in FIELDS])
                    times = np.asarray([row[field] for field in FIELDS], dtype=float)
                    timestamps = np.asarray(table["timestamp"].to_numpy())
                    if (
                        row.get("status") != "complete"
                        or row.get("excluded_from_training")
                        or boundaries.dtype.kind not in "iu"
                        or boundaries[0] <= 0
                        or np.any(np.diff(boundaries) <= 0)
                        or boundaries[-1] >= len(table)
                        or row["frame_count"] != len(table)
                        or not np.isfinite(times).all()
                        or np.any(np.diff(times) <= 0)
                        or not np.allclose(timestamps[boundaries], times, atol=1e-5, rtol=0)
                    ):
                        raise ValueError(f"Invalid phase timeline for episode {episode}")
                    if view.phase_arm_hold and (states.shape[1:] != (52,) or actions.shape[1:] != (12,)):
                        raise ValueError("Phase arm hold is defined only for dual UR/O30 52-state/12-action")
            task_ids = np.asarray(table["task_index"].to_numpy())
            self.records[episode] = states, actions, boundaries, task_ids
        self.indices = np.array(
            [
                i
                for i, (episode, frame, task) in enumerate(
                    zip(ids["episode_index"], ids["frame_index"], ids["task_index"], strict=True)
                )
                if episode in selected
                and (not view.task_indices or task in view.task_indices)
                and (self.records[episode][2] is None or frame < self.records[episode][2][-1])
            ],
            dtype=np.int64,
        )
        if not len(self.indices):
            raise ValueError("Task selection contains no training frames")
        self.hf_dataset = dataset.hf_dataset.select(self.indices.tolist())
        self.ids = np.column_stack([ids[key][self.indices] for key in ("episode_index", "frame_index", "task_index")])
        self.tasks = dataset.meta.tasks

    def __len__(self):
        return len(self.indices)

    def vectors(self, index):
        episode, frame, task = map(int, self.ids[index])
        states, actions, boundaries, task_ids = self.records[episode]
        prompt = dict(self.view.prompt_overrides).get(task, self.tasks[task])
        stop = len(actions) - 1
        phase = None
        if self.view.task_boundary_hold:
            changes = np.flatnonzero(task_ids[frame:] != task)
            if len(changes):
                stop = frame + int(changes[0]) - 1
        if boundaries is not None:
            phase = int(np.searchsorted(boundaries, frame, side="right"))
            stop = int(boundaries[phase])
            prompt = self.view.phase_prompts[phase]
        window = actions[np.minimum(frame + np.arange(self.horizon), stop)].copy()
        if phase is not None and self.view.phase_arm_hold:
            if phase == 0:
                window[:, 6:12] = states[frame, 26:32]
            else:
                window[:, :6] = states[frame, :6]
        return {"observation.state": states[frame], "action": window, "prompt": prompt}

    def __getitem__(self, index):
        sample = dict(self.dataset[int(self.indices[index])])
        sample.update(self.vectors(index))
        sample["action_is_pad"] = np.zeros(self.horizon, dtype=bool)
        return sample
