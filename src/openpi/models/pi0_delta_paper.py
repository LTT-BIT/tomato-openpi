"""Opt-in Pi0.5 subclass for correlated flow noise; base Pi0 is untouched."""

from dataclasses import dataclass

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.models import model as model_lib
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0_config import Pi0Config
from openpi.shared import delta_paper as paper
from openpi.training.weight_loaders import PartialCheckpointWeightLoader


@dataclass(frozen=True)
class PaperPi0Config(Pi0Config):
    paper_asset_dir: str = ""
    paper_asset_sha256: str | None = None
    paper_mode: str = "global_quantile"
    paper_beta: float = 0.0
    paper_std_floor: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if (
            not self.pi05
            or self.action_dim != 58
            or self.action_horizon <= 0
            or self.paper_mode not in paper.MODES
            or not 0 <= self.paper_beta < 1
        ):
            raise ValueError("Paper model requires Pi0.5, 58D actions and valid mode/beta")

    def create(self, rng):
        return PaperPi0(self, rngs=nnx.Rngs(rng))

    def load_pytorch(self, train_config, weight_path):
        raise ValueError("Delta paper experiments are JAX-only; PyTorch is not implemented")


@dataclass(frozen=True)
class PaperWeightLoader(PartialCheckpointWeightLoader):
    """Retain initialized nontrainable correlation state when loading base weights."""

    def load(self, params):
        base = {key: value for key, value in params.items() if key != "paper_noise_cholesky"}
        result = super().load(base)
        if "paper_noise_cholesky" in params:
            result["paper_noise_cholesky"] = params["paper_noise_cholesky"]
        return result


class PaperPi0(Pi0):
    def __init__(self, config, *, rngs):
        if not config.paper_asset_sha256:
            raise ValueError("Compute matching paper statistics and reload configuration before model creation")
        _, arrays = paper.load_bundle(
            config.paper_asset_dir,
            mode=config.paper_mode,
            beta=config.paper_beta,
            horizon=config.action_horizon,
            std_floor=config.paper_std_floor,
            expected_sha256=config.paper_asset_sha256,
        )
        super().__init__(config, rngs=rngs)
        self.paper_beta = config.paper_beta
        if self.paper_beta:
            # Variable rather than Param: checkpointed and device-resident, excluded from optimizer.
            self.paper_noise_cholesky = nnx.Variable(jnp.asarray(arrays["noise_cholesky"]))

    def noise(self, rng, shape):
        value = jax.random.normal(rng, shape)
        if not self.paper_beta:
            return value
        flat = value.reshape(*shape[:-2], self.action_horizon * self.action_dim)
        return (flat @ self.paper_noise_cholesky.value.T).reshape(shape)

    def compute_loss(self, rng, observation, actions, *, train=False, augmentation_rng=None):
        if not self.paper_beta:
            return super().compute_loss(rng, observation, actions, train=train, augmentation_rng=augmentation_rng)
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        if augmentation_rng is not None:
            preprocess_rng = augmentation_rng
        observation = model_lib.preprocess_observation(preprocess_rng, observation, train=train)
        noise = self.noise(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        target = noise - actions
        prefix, prefix_mask, prefix_ar = self.embed_prefix(observation)
        suffix, suffix_mask, suffix_ar, cond = self.embed_suffix(observation, x, time)
        mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar = jnp.concatenate([prefix_ar, suffix_ar])
        (_, output), _ = self.PaliGemma.llm(
            [prefix, suffix],
            mask=make_attn_mask(mask, ar),
            positions=jnp.cumsum(mask, axis=1) - 1,
            adarms_cond=[None, cond],
        )
        velocity = self.action_out_proj(output[:, -self.action_horizon :])
        return jnp.mean(jnp.square(velocity - target), axis=-1)

    def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
        if noise is None and self.paper_beta:
            noise = self.noise(rng, (observation.state.shape[0], self.action_horizon, self.action_dim))
        # Explicit noise is already in the selected distribution; do not correlate it twice.
        return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)
