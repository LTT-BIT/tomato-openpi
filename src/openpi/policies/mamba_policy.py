"""Local/offline Mamba policy using the same data transforms as the trainer.

This training checkout has no generic Policy/serve implementation. This adapter
returns arrays only and has no robot, transport, executor or carry cache.
"""

import jax
import jax.numpy as jnp
import numpy as np

from openpi import transforms
from openpi.models import model as model_lib
from openpi.training.mamba_training import validate_stats


def _physical_prefix(model, absolute_prefix):
    if absolute_prefix is None:
        return None, 0
    prefix = np.asarray(absolute_prefix, np.float32)
    if (
        prefix.ndim != 2
        or prefix.shape[1] != model.action_dim
        or len(prefix) > model.rtc_max_delay
        or not np.isfinite(prefix).all()
    ):
        raise ValueError("absolute_prefix must be finite canonical physical (0..rtc_max_delay, action_dim) actions")
    return (prefix, len(prefix)) if len(prefix) else (None, 0)


class MambaOfflinePolicy:
    def __init__(self, model, data_config):
        from openpi.models.pi0_mamba import MambaPi0

        if not isinstance(model, MambaPi0) or getattr(data_config, "paper_asset_dir", None):
            raise ValueError("Use an ordinary Phase A Mamba model/data contract")
        validate_stats(data_config.norm_stats, model.action_dim)
        if data_config.use_quantile_norm:
            raise ValueError("Mamba Phase A requires ordinary per-dimension statistics")
        if any(
            isinstance(t, transforms.DeltaActions | transforms.AbsoluteActions)
            for t in (*data_config.data_transforms.inputs, *data_config.data_transforms.outputs)
        ):
            raise ValueError("Mamba Phase A policy does not accept Delta transforms")
        self.model = model
        self.action_stats = data_config.norm_stats["actions"]
        self.inputs = transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                transforms.Normalize(data_config.norm_stats, use_quantiles=False),
                *data_config.model_transforms.inputs,
            ]
        )
        # State stats are deliberately excluded: only model action predictions are unnormalized.
        self.outputs = transforms.compose(
            [
                *data_config.model_transforms.outputs,
                transforms.Unnormalize({"actions": data_config.norm_stats["actions"]}, use_quantiles=False),
                *data_config.data_transforms.outputs,
            ]
        )

    def infer(self, raw_observation, *, rng, noise=None, num_steps=10, absolute_prefix=None):
        prefix, length = _physical_prefix(self.model, absolute_prefix)
        encoded = None
        if length:
            padded = np.zeros((self.model.action_horizon, self.model.action_dim), np.float32)
            padded[:length] = prefix
            encoded = transforms.Normalize({"actions": self.action_stats}, use_quantiles=False)({"actions": padded})[
                "actions"
            ]
        prepared = self.inputs(raw_observation)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None], prepared)
        observation = model_lib.Observation.from_dict(batched)
        self.model.eval()
        actions = self.model.sample_actions(
            rng,
            observation,
            noise=noise,
            num_steps=num_steps,
            prefix_actions=None if encoded is None else jnp.asarray(encoded)[None],
            prefix_length=length,
        )
        result = self.outputs({"actions": np.asarray(actions)})
        if length:
            result["actions"] = np.array(result["actions"], copy=True)
            result["actions"][:, :length] = prefix
        return result


class MambaPaperOfflinePolicy:
    """Same input transforms, then existing Paper inverse using physical current state."""

    def __init__(self, model, data_config):
        from openpi.models.pi0_mamba_paper import PaperMambaPi0
        from openpi.training.delta_paper import PaperDataView
        from openpi.training.delta_paper_prefix import _contract

        if not isinstance(model, PaperMambaPi0) or not isinstance(data_config, PaperDataView):
            raise ValueError("Paper Mamba policy requires a Paper model and data view")
        if data_config.paper_bundle_sha256 != model.config.paper_asset_sha256 or not data_config.use_quantile_norm:
            raise ValueError("Paper Mamba model/data assets or normalization differ")
        _contract(data_config)
        self.model, self.data = model, data_config
        self.vector_inputs = transforms.compose(
            [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
        )
        self.model_inputs = transforms.compose(
            [
                transforms.Normalize(data_config.norm_stats, use_quantiles=True),
                *data_config.model_transforms.inputs,
            ]
        )

    def infer(self, raw_observation, *, rng, noise=None, num_steps=10, absolute_prefix=None):
        from openpi.training.delta_paper_prefix import decode_actions
        from openpi.training.delta_paper_prefix import encode_prefix

        prefix, length = _physical_prefix(self.model, absolute_prefix)
        vectors = self.vector_inputs(raw_observation)
        physical_state = np.asarray(vectors["state"], np.float32).copy()
        encoded = encode_prefix(self.data, physical_state, prefix) if length else None
        prepared = self.model_inputs(vectors)
        observation = model_lib.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], prepared))
        self.model.eval()
        actions = self.model.sample_actions(
            rng,
            observation,
            noise=noise,
            num_steps=num_steps,
            prefix_actions=None if encoded is None else jnp.asarray(encoded)[None],
            prefix_length=length,
        )
        absolute = decode_actions(self.data, physical_state, np.asarray(actions)[0])
        if length:
            absolute[:length] = prefix
        return {"actions": absolute[None]}


def create_mamba_policy(model, data_config):
    from openpi.models.pi0_mamba_paper import PaperMambaPi0

    return (
        MambaPaperOfflinePolicy(model, data_config)
        if isinstance(model, PaperMambaPi0)
        else MambaOfflinePolicy(model, data_config)
    )
