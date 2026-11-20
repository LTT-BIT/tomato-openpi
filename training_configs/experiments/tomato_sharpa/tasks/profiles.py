"""Task semantics, independent of dataset size, view count and training recipe."""

from dataclasses import dataclass

from openpi.subtasks import tomato
from openpi.subtasks import tomato_in_hand
from training_configs.experiments.tomato_sharpa.baseline.dataset import PROMPT


@dataclass(frozen=True)
class TaskProfile:
    label: str
    goal: str
    phases: tuple[str, ...]
    action_prompts: dict[str, str]

    @property
    def prompts(self):
        return tuple(self.action_prompts.values())


PROFILES = {
    "single_pick": TaskProfile("单颗采摘", PROMPT, (), {}),
    "truss_plate": TaskProfile("整串采摘 · 托盘调整", tomato.GOAL, tomato.PHASES, tomato.ACTION_PROMPTS),
    "truss_in_hand": TaskProfile(
        "整串采摘 · 手中调整", tomato.GOAL, tomato_in_hand.PHASES, tomato_in_hand.ACTION_PROMPTS
    ),
}
