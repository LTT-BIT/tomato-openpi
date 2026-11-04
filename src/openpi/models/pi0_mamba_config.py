"""Opt-in Phase A architecture; Pi05 identity preserves tokenizer/state semantics."""

import dataclasses

from flax import nnx

from openpi.models.mamba3.config import Mamba3Config
from openpi.models.pi0_config import Pi0Config


@dataclasses.dataclass(frozen=True)
class MambaPi0Config(Pi0Config):
    # Inherited action_expert_variant is serialization compatibility only; it is
    # never constructed and never determines Mamba width or pretrained loading.
    pi05: bool = True
    action_dim: int = 58
    action_horizon: int = 50
    max_token_len: int = 256
    architecture: str = "pi05_mamba3_jax"
    stage: str = "phase_a_plain_fm"
    action_representation: str = "absolute"
    normalization: str = "ordinary_per_dimension"
    diagnostic_only: bool = False
    mamba: Mamba3Config = dataclasses.field(default_factory=Mamba3Config)
    mamba_depth: int = 4
    mamba_mlp_dim: int = 2048
    film_mlp_dim: int = 2048
    context_heads: int = 8
    context_vision_tokens: int = 44
    context_language_tokens: int = 16
    context_global_tokens: int = 4

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or not self.discrete_state_input:
            raise ValueError("Mamba Phase A requires Pi05 discrete state text")
        if (
            self.rtc_max_delay
            or self.action_representation != "absolute"
            or self.normalization != "ordinary_per_dimension"
        ):
            raise ValueError("Mamba Phase A rejects RTC, Delta and Paper normalization")
        if self.architecture != "pi05_mamba3_jax" or self.stage != "phase_a_plain_fm":
            raise ValueError("Unsupported Mamba architecture/stage")
        if "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
            raise ValueError("Mamba Phase A supports FullFT only, no LoRA")
        if self.dtype != self.mamba.dtype:
            raise ValueError("VLM and Mamba compute dtype must match")
        if not self.diagnostic_only and (
            self.action_dim != 58 or self.action_horizon != 50 or self.paligemma_variant != "gemma_2b"
        ):
            raise ValueError("Production Phase A requires complete gemma_2b VLM, H50/D58")
        for name in (
            "mamba_depth",
            "mamba_mlp_dim",
            "film_mlp_dim",
            "context_heads",
            "context_vision_tokens",
            "context_language_tokens",
            "context_global_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.mamba.hidden_size % self.context_heads:
            raise ValueError("Context heads must divide Mamba width")

    def create(self, rng):
        from openpi.models.pi0_mamba import MambaPi0

        return MambaPi0(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter(self):
        return nnx.Nothing

    def load_pytorch(self, train_config, weight_path):
        raise ValueError("Mamba Phase A is JAX-only; use the architecture-specific checkpoint loader")

    def validate_training(self, train_config):
        from openpi.training.mamba_training import validate_training

        validate_training(train_config)
