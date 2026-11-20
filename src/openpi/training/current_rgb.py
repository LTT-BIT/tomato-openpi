"""Current-frame input options for the frozen Sharpa five-phase bundle."""

from dataclasses import dataclass

from openpi.training import config


@dataclass(frozen=True)
class CurrentRgbView(config.DataConfig):
    video_views: int = 4
    global_prompt: str | None = None


@dataclass(frozen=True)
class LeRobotCurrentRgbThreeDataConfig(config.LeRobotFr3SharpaCropDataConfig):
    def create_vector_transforms(self, model_config):
        return config.LeRobotFr3SharpaMultimodalDataConfig.create_vector_transforms(self, model_config)


@dataclass(frozen=True)
class FixedGlobalPrompt:
    prompt: str

    def __call__(self, data):
        return {**data, "prompt": self.prompt}


def fixed_global_prompt(data: config.DataConfigFactory) -> str | None:
    return getattr(data, "global_prompt", None) or getattr(data.base_config, "global_prompt", None)
