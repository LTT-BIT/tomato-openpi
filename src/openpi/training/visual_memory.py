"""Image-only history sampling before the existing Sharpa transforms."""

from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace

import numpy as np

from openpi import transforms
from openpi.models import pi0_config
from openpi.policies.fr3_sharpa_policy import _parse_image
from openpi.subtasks.visual_memory import HISTORY
from openpi.subtasks.visual_memory import IMAGE_KEYS
from openpi.subtasks.visual_memory import VisualMemorySpec
from openpi.subtasks.visual_memory import camera_frame
from openpi.subtasks.visual_memory import pack_history
from openpi.training import config


class VisualMemoryDataset:
    """Preserve every current/action sample; all history stays inside its recording."""

    def __init__(self, dataset, spec, mask_key="observation.interaction_available"):
        self.dataset, self.spec, self.mask_key = dataset, spec, mask_key
        if spec.image_keys != IMAGE_KEYS or dataset.meta.fps != spec.fps:
            raise ValueError("Matching four-view source FPS required")
        self.hf_dataset = dataset.hf_dataset
        self.meta = dataset.meta
        self.root = dataset.root
        cols = self.hf_dataset.select_columns(["episode_index", "frame_index"]).with_format("numpy")[:]
        self.episodes = np.asarray(cols["episode_index"], dtype=np.int64)
        self.frame_indices = np.asarray(cols["frame_index"], dtype=np.int64)
        if not len(dataset) or len(self.episodes) != len(dataset):
            raise ValueError("History/source length mismatch")
        # Source episodes must be contiguous, complete, and ordered on the FPS grid.
        starts = np.r_[0, np.flatnonzero(self.episodes[1:] != self.episodes[:-1]) + 1]
        ends = np.r_[starts[1:], len(dataset)]
        if len(set(self.episodes[starts])) != len(starts):
            raise ValueError("Disjoint fragments of the same episode are unsupported")
        for start, end in zip(starts, ends, strict=True):
            if not np.array_equal(self.frame_indices[start:end], np.arange(end - start)):
                raise ValueError("History needs complete contiguous episode frame indices")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        index = index.__index__()
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample = self.dataset[index]
        current, _ = camera_frame(sample, self.mask_key)

        def lookup(lag):
            previous = index - lag
            if previous < 0 or self.episodes[previous] != self.episodes[index]:
                return None
            return camera_frame(self.dataset[previous], self.mask_key)

        return {**sample, HISTORY: pack_history(self.spec, current, lookup)}


@dataclass(frozen=True)
class VisualMemoryInputs:
    current_transforms: tuple
    spec: VisualMemorySpec

    def __call__(self, data):
        history = data.get(HISTORY)
        if history is None or set(history) != {"image", "image_mask"}:
            raise ValueError("Explicit visual history payload required, including for episode starts")
        if any(set(history[key]) != set(self.spec.history_keys) for key in history):
            raise ValueError("Visual history camera/time slots do not match the model")
        current_transforms = list(self.current_transforms)
        # Training has action/prompt columns; offline serving may omit actions.
        first = current_transforms[0]
        flat = transforms.flatten_dict(data)
        mapping = {k: v for k, v in first.structure.items() if k not in ("actions", "prompt") or v in flat}
        current_transforms[0] = transforms.RepackTransform(mapping)
        result = transforms.compose(current_transforms)(data)
        for key in self.spec.history_keys:
            image = _parse_image(history["image"][key])
            valid = np.asarray(history["image_mask"][key])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("Historical RGB uint8 image required")
            if valid.shape != () or valid.dtype != np.bool_:
                raise ValueError("Historical validity must be scalar boolean")
            result["image"][key] = image
            result["image_mask"][key] = valid
        return result


@dataclass(frozen=True)
class VisualMemoryDataConfig(config.DataConfig):
    visual_memory: VisualMemorySpec | None = None
    history_mask_key: str = "observation.interaction_available"


@dataclass(frozen=True)
class LeRobotVisualMemoryDataConfig(config.LeRobotFr3SharpaCropDataConfig):
    def create(self, assets_dirs, model_config):
        data = super().create(assets_dirs, model_config)
        return VisualMemoryDataConfig(
            **{f.name: getattr(data, f.name) for f in fields(data)},
            visual_memory=model_config.memory,
            history_mask_key=self.interaction_mask_key,
        )

    def create_vector_transforms(self, model_config):
        if model_config.memory.image_keys != IMAGE_KEYS:
            raise ValueError("Four canonical Sharpa cameras required")
        # Build the original current-frame physical transform unchanged, then append
        # history pixels. Only the current state participates in DeltaActions.
        current = pi0_config.Pi0Config(pi05=True, action_dim=58, image_keys=IMAGE_KEYS)
        repack, vectors = super().create_vector_transforms(current)
        inputs = VisualMemoryInputs(tuple(repack.inputs) + tuple(vectors.inputs), model_config.memory)
        return transforms.Group(), replace(vectors, inputs=[inputs])
