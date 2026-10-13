"""Published full-parameter v1 recipe (derived from historical C0 v3): all model parameters trainable, 30,000 optimization steps."""

from dataclasses import replace

from openpi.models import gemma
from training_configs.recipes.common import Recipe

RECIPE = Recipe(
    version="pi05_fullft_v1",
    source_baseline="fulltask_c0_v3_fullft",
    paligemma_variant="gemma_2b",
    action_expert_variant="gemma_300m",
    num_train_steps=30_000,
)


def build_config(**kwargs):
    for variant in ("gemma_2b", "gemma_300m"):
        if gemma.get_config(variant).lora_configs:
            raise ValueError(f"{variant}: full-parameter recipe cannot contain LoRA")
    return replace(RECIPE.build(**kwargs), batch_size=64)
