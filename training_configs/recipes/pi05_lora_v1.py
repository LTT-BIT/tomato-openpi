"""Published LoRA v1 recipe (derived from historical C0 v2): VLM rank 16, Expert rank 32, vision/projections trainable."""

from openpi.models import gemma
from training_configs.recipes.common import Recipe

RECIPE = Recipe(
    version="pi05_lora_v1",
    source_baseline="fulltask_c0_v2",
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    num_train_steps=30_000,
)


def build_config(**kwargs):
    for variant, rank in (("gemma_2b_lora", 16), ("gemma_300m_lora", 32)):
        adapters = gemma.get_config(variant).lora_configs
        if adapters is None or set(adapters) != {"attn", "ffn"}:
            raise ValueError(f"{variant}: baseline LoRA modules changed")
        if any(adapter.rank != rank or adapter.alpha != float(rank) for adapter in adapters.values()):
            raise ValueError(f"{variant}: baseline LoRA rank/alpha changed")
    return RECIPE.build(**kwargs)
