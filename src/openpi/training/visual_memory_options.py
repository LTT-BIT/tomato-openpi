"""Opt-in training views for the prepared 119-recording Sharpa bundle."""

from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.subtasks.visual_memory import IMAGE_KEYS
from openpi.subtasks.visual_memory import RAW_KEYS
from openpi.subtasks.visual_memory import THREE_VIEWS
from openpi.training import config
from openpi.training.visual_memory import LeRobotVisualMemoryDataConfig
from openpi.training.visual_memory import VisualMemoryDataConfig
from openpi.training.visual_memory import VisualMemoryInputs
from openpi.training.visual_memory_execution import FastVisualMemoryDataset


@dataclass(frozen=True)
class FixedGlobalPrompt:
    prompt: str

    def __call__(self, data):
        return {**data, "prompt": self.prompt}


class ThreeViewVisualMemoryDataset(FastVisualMemoryDataset):
    def __init__(self, dataset, spec):
        super().__init__(dataset, replace(spec, image_keys=IMAGE_KEYS), mask_key=None)
        self.spec = spec
        # LeRobot gets video queries from this instance's metadata. The prepared
        # dataset and its on-disk metadata still contain all four views.
        features = {key: value for key, value in self.meta.features.items() if key != RAW_KEYS[-1]}
        self.meta.info = {**self.meta.info, "features": features}


def current_frame_transforms(data, model):
    memory = getattr(model, "memory", None)
    cameras = memory.image_keys if memory else model.image_keys
    current = Pi0Config(pi05=True, action_dim=58, image_keys=cameras)
    if cameras == THREE_VIEWS:
        return config.LeRobotFr3SharpaMultimodalDataConfig.create_vector_transforms(data, current)
    return config.LeRobotFr3SharpaCropDataConfig.create_vector_transforms(data, current)


@dataclass(frozen=True)
class VisualMemoryOptionsView(VisualMemoryDataConfig):
    global_prompt: str | None = None


@dataclass(frozen=True)
class LeRobotVisualMemoryOptionsDataConfig(LeRobotVisualMemoryDataConfig):
    global_prompt: str | None = None

    def create(self, assets_dirs, model_config):
        base = super().create(assets_dirs, model_config)
        return VisualMemoryOptionsView(
            **{field.name: getattr(base, field.name) for field in fields(base)}, global_prompt=self.global_prompt
        )

    def create_vector_transforms(self, model_config):
        if model_config.memory.image_keys == IMAGE_KEYS:
            return super().create_vector_transforms(model_config)
        repack, vectors = current_frame_transforms(self, model_config)
        inputs = VisualMemoryInputs(tuple(repack.inputs) + tuple(vectors.inputs), model_config.memory)
        return transforms.Group(), replace(vectors, inputs=[inputs])
