"""Opt-in standard FLASH coefficient expert on the unchanged pi05 VLM.

Absolute physical-radian history/targets in, absolute (B,50,58) actions out.
The VLM prefix is cached once, frozen, and shared by FM/consistency calls.
Future labels exist only in the loss API; inference has no target argument.
"""

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import posemb_sincos
from openpi.models.pi0_config import Pi0Config
from openpi.shared.flash_polynomial import FlashPolynomial
from openpi.shared.flash_polynomial import normalize_actions
from openpi.shared.flash_polynomial import unnormalize_actions
from openpi.training.flash_checkpoint import is_vlm_parameter


@dataclasses.dataclass(frozen=True)
class Pi0FlashConfig(Pi0Config):
    pi05: bool = True
    action_dim: int = 58
    action_horizon: int = 50
    max_token_len: int = 256
    poly_degree: int = 6
    history_steps: int = 8
    history_reg_lambda: float = 0.1
    history_noise_std: float = 0.5
    consistency_weight: float = 1.0
    boundary_constraint: bool = True
    action_q01: tuple[float, ...] = (-1.0,) * 58
    action_q99: tuple[float, ...] = (1.0,) * 58

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or (self.action_horizon, self.action_dim, self.poly_degree, self.history_steps) != (
            50,
            58,
            6,
            8,
        ):
            raise ValueError("P1 requires absolute pi05 (50,58), K=6, history=8")
        if self.rtc_max_delay or "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
            raise ValueError("Standard P1 uses no RTC or LoRA variants")
        if self.history_noise_std < 0 or self.consistency_weight < 0:
            raise ValueError("Nonnegative history noise/consistency weight required")
        lo, hi = np.asarray(self.action_q01), np.asarray(self.action_q99)
        if lo.shape != (58,) or hi.shape != (58,) or not np.isfinite([lo, hi]).all() or (hi < lo).any():
            raise ValueError("Finite 58D action quantiles q01<=q99 required")

    def create(self, rng):
        return Pi0Flash(self, rngs=nnx.Rngs(rng))

    def get_freeze_filter(self):
        return is_vlm_parameter


class Pi0Flash(Pi0):
    def __init__(self, config: Pi0FlashConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs)
        self.polynomial = FlashPolynomial(history_reg_lambda=config.history_reg_lambda)
        self.coefficient_tokens = config.poly_degree + 1
        self.history_steps = config.history_steps
        self.history_noise_std = config.history_noise_std
        self.consistency_weight = config.consistency_weight
        self.boundary_constraint = config.boundary_constraint
        # Immutable static configuration, saved in checkpoint metadata, not optimizer parameters.
        self.action_q01 = tuple(config.action_q01)
        self.action_q99 = tuple(config.action_q99)

    def normalize(self, values):
        return normalize_actions(values, jnp.asarray(self.action_q01), jnp.asarray(self.action_q99))

    def denormalize(self, values):
        return unnormalize_actions(values, jnp.asarray(self.action_q01), jnp.asarray(self.action_q99))

    def encode_condition(self, observation):
        """Real pi05 image/language prefix and original layerwise KV projection."""
        tokens, mask, ar = self.embed_prefix(observation)
        _, cache = self.PaliGemma.llm(
            [tokens, None],
            mask=make_attn_mask(mask, ar),
            positions=jnp.cumsum(mask, axis=1) - 1,
        )
        return mask, jax.tree.map(jax.lax.stop_gradient, cache)

    def coefficient_velocity(self, coeff, time, prefix_mask, kv_cache):
        if coeff.shape[1:] != (7, 58):
            raise ValueError("Expected seven 58D coefficient tokens")
        suffix = self.action_in_proj(coeff)
        time_emb = posemb_sincos(time, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        time_emb = nnx.swish(self.time_mlp_out(nnx.swish(self.time_mlp_in(time_emb))))
        suffix_mask = jnp.ones(coeff.shape[:2], dtype=jnp.bool_)
        suffix_ar = jnp.arange(self.coefficient_tokens) == 0
        mask = jnp.concatenate(
            [
                jnp.broadcast_to(
                    prefix_mask[:, None, :], (coeff.shape[0], self.coefficient_tokens, prefix_mask.shape[1])
                ),
                make_attn_mask(suffix_mask, suffix_ar),
            ],
            axis=-1,
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.arange(self.coefficient_tokens)[None, :]
        (_, output), _ = self.PaliGemma.llm(
            [None, suffix],
            mask=mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, time_emb],
        )
        return self.action_out_proj(output).astype(jnp.float32)

    def loss_from_cache(self, rng, prefix_mask, kv_cache, *, history, target_ext, train=True):
        """0->1 official FLASH FM and separate one-step consistency forward."""
        if history.shape[1:] != (8, 58) or target_ext.shape[1:] != (54, 58):
            raise ValueError("Need measured history (B,8,58) and target context (B,54,58)")
        noise_rng, time_rng = jax.random.split(rng)
        source = self.polynomial.fit_history(self.normalize(history))
        target = self.polynomial.fit_target(self.normalize(target_ext), boundary_constraint=self.boundary_constraint)
        if train and self.history_noise_std:
            source = source + self.history_noise_std * jax.random.normal(noise_rng, source.shape)
        time = jax.random.uniform(time_rng, (history.shape[0],))
        mixture = (1 - time[:, None, None]) * source + time[:, None, None] * target
        velocity = self.coefficient_velocity(mixture, time, prefix_mask, kv_cache)
        fm = jnp.mean((velocity - (target - source)) ** 2, axis=(1, 2))
        at_zero = self.coefficient_velocity(source, jnp.zeros_like(time), prefix_mask, kv_cache)
        consistency = jnp.mean((source + at_zero - target) ** 2, axis=(1, 2))
        return {
            "loss": fm + self.consistency_weight * consistency,
            "flow_matching_loss": fm,
            "polynomial_consistency_loss": consistency,
        }

    def loss_components(self, rng, observation, *, history, target_ext, train=False):
        augmentation_rng, flow_rng = jax.random.split(rng)
        observation = _model.preprocess_observation(augmentation_rng, observation, train=train)
        prefix_mask, cache = self.encode_condition(observation)
        return self.loss_from_cache(flow_rng, prefix_mask, cache, history=history, target_ext=target_ext, train=train)

    def compute_loss(
        self, rng, observation, actions=None, *, history=None, target_ext=None, train=False, augmentation_rng=None
    ):
        # Explicit metadata required: the old (obs,actions) loader cannot silently fabricate history.
        del actions
        if history is None or target_ext is None:
            raise ValueError("FLASH requires explicit causal measured history and separate target_ext")
        if augmentation_rng is not None:
            observation = _model.preprocess_observation(augmentation_rng, observation, train=train)
            mask, cache = self.encode_condition(observation)
            return self.loss_from_cache(rng, mask, cache, history=history, target_ext=target_ext, train=train)["loss"]
        return self.loss_components(rng, observation, history=history, target_ext=target_ext, train=train)["loss"]

    def sample_from_cache(self, prefix_mask, kv_cache, *, history, num_steps=1):
        """History-only source, deterministic Euler; no Gaussian noise at deployment."""
        if type(num_steps) is not int or num_steps < 1:
            raise ValueError("num_steps must be a positive static integer")
        if history.shape[1:] != (8, 58):
            raise ValueError("Need eight measured history positions")
        coeff = self.polynomial.fit_history(self.normalize(history))
        for step in range(num_steps):
            time = jnp.full((history.shape[0],), step / num_steps)
            coeff += self.coefficient_velocity(coeff, time, prefix_mask, kv_cache) / num_steps
        return self.denormalize(self.polynomial.reconstruct(coeff))

    def sample_actions(self, rng, observation, *, history, num_steps=1):
        del rng  # official inference is deterministic given measured history and observations
        observation = _model.preprocess_observation(None, observation, train=False)
        mask, cache = self.encode_condition(observation)
        return self.sample_from_cache(mask, cache, history=history, num_steps=num_steps)
