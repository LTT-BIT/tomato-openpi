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

    def infer(self, raw_observation, *, rng, noise=None, num_steps=10):
        prepared = self.inputs(raw_observation)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[None], prepared)
        observation = model_lib.Observation.from_dict(batched)
        self.model.eval()
        actions = self.model.sample_actions(rng, observation, noise=noise, num_steps=num_steps)
        return self.outputs({"actions": np.asarray(actions)})
