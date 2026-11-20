"""Current RGB and prompt selection for independent LeRobot sources."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from openpi.training import config


@dataclass(frozen=True)
class CurrentRgbView(config.DataConfig):
    video_views: int = 4
    global_prompt: str | None = None
    prompt_mode: str = "stage"


@dataclass(frozen=True)
class LeRobotCurrentRgbThreeDataConfig(config.LeRobotFr3SharpaCropDataConfig):
    def create_vector_transforms(self, model_config):
        return config.LeRobotFr3SharpaMultimodalDataConfig.create_vector_transforms(self, model_config)


def fixed_global_prompt(data: config.DataConfigFactory) -> str | None:
    return getattr(data, "global_prompt", None) or getattr(data.base_config, "global_prompt", None)


def stage_annotations(root):
    root = Path(root)
    directory = root / "annotations/subtasks/coarse"
    if not directory.is_dir():
        directory = root / "annotations"
    return sorted(directory.glob("episode_*.json"))


class StagePromptDataset:
    """Resolve source-local phase text once, without rewriting tables or actions."""

    def __init__(self, dataset, *, stage=True):
        self.dataset = dataset
        self.segments = {}
        for path in stage_annotations(dataset.root) if stage else []:
            document = json.loads(path.read_text())
            episode = int(path.stem.removeprefix("episode_"))
            rows = document.get("segments", [])
            end = 0
            for row in rows:
                start, stop, text = row["start_frame"], row["end_frame"], row["text"]
                if (
                    start != end
                    or type(stop) is not int
                    or stop <= start
                    or not isinstance(text, str)
                    or not text.strip()
                ):
                    raise ValueError(f"Invalid phase annotation in {path}")
                end = stop
            if rows and end != dataset.meta.episodes[episode]["length"]:
                raise ValueError(f"Phase annotation length differs from episode in {path}")
            if rows:
                self.segments[episode] = ([row["end_frame"] for row in rows], [row["text"] for row in rows])

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        episode = int(sample["episode_index"])
        if episode in self.segments:
            ends, texts = self.segments[episode]
            prompt = texts[int(np.searchsorted(ends, int(sample["frame_index"]), side="right"))]
        else:
            prompt = self.meta.tasks[int(sample["task_index"])]
        return {
            "observation.interaction_available": np.True_,
            **sample,
            "prompt": prompt,
        }
