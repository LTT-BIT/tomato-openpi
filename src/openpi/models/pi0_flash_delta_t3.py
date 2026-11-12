"""Opt-in FLASH Delta/time-domain T3; deterministic history-anchored inference."""

import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models.pi0_flash import Pi0Flash
from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.shared.flash_delta_t3 import decode
from openpi.shared.flash_delta_t3 import encode


@dataclasses.dataclass(frozen=True)
class Pi0FlashDeltaT3Config(Pi0FlashConfig):
    timestamp: bool = True
    correlation_beta: float = 0.5
    t3_assets: str = ""
    t3_sha256: str = ""

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.correlation_beta < 1 or self.freeze_vlm:
            raise ValueError("Delta/T3 requires FullFT and beta in [0,1)")

    def create(self, rng):
        return Pi0FlashDeltaT3(self, rngs=nnx.Rngs(rng))


class Pi0FlashDeltaT3(Pi0Flash):
    def __init__(self, config, rngs):
        from openpi.training.flash_delta_t3_data import load

        arrays = load(config)
        super().__init__(config, rngs)
        self.timestamp = config.timestamp
        self.beta = config.correlation_beta
        for name, value in arrays.items():
            setattr(self, "t3_" + name, nnx.Variable(jnp.asarray(value)))

    def codec(self, values, anchor, indices, *, inverse=False):
        return (decode if inverse else encode)(
            values,
            anchor,
            self.t3_mean.value[indices],
            self.t3_std.value[indices],
            self.t3_low.value,
            self.t3_high.value,
            timestamp=self.timestamp,
        )

    def normalize(self, values):
        return values

    def denormalize(self, values):
        return values

    def loss_from_cache(self, rng, prefix_mask, kv_cache, *, history, target_ext, train=True):
        anchor = history[:, -1]
        source = self.polynomial.fit_history(self.codec(history, anchor, slice(1, 9)))
        target = self.polynomial.fit_target(
            self.codec(target_ext, anchor, slice(None)), boundary_constraint=self.boundary_constraint
        )
        noise_rng, time_rng = jax.random.split(rng)
        if train and self.history_noise_std:
            noise = jax.random.normal(noise_rng, source.shape)
            if self.beta:
                noise = (noise.reshape(-1, 406) @ self.t3_cholesky.value.T).reshape(source.shape)
            source = source + self.history_noise_std * noise
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

    def sample_from_cache(self, prefix_mask, kv_cache, *, history, num_steps=1):
        anchor = history[:, -1]
        encoded = self.codec(history, anchor, slice(1, 9))
        output = super().sample_from_cache(prefix_mask, kv_cache, history=encoded, num_steps=num_steps)
        return self.codec(output, anchor, slice(2, 52), inverse=True)


@dataclasses.dataclass(frozen=True)
class FlashDeltaT3WeightLoader:
    params_path: str
    skip_on_mismatch_regex: str = ".*(action_in_proj|action_out_proj|state_proj).*"

    def load(self, params):
        from openpi.training.weight_loaders import PartialCheckpointWeightLoader

        weights = {k: v for k, v in params.items() if not k.startswith("t3_")}
        result = PartialCheckpointWeightLoader(self.params_path, self.skip_on_mismatch_regex).load(weights)
        result.update({k: v for k, v in params.items() if k.startswith("t3_")})
        return result
