"""Model contract and freeze coverage without allocating the pretrained model."""

import inspect
import unittest

from flax import nnx
import jax
import numpy as np
import pytest

from openpi.models.pi0_flash import Pi0Flash
from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.training.flash_checkpoint import is_flash_trainable_parameter
from openpi.training.flash_checkpoint import is_vlm_parameter


class FlashModelTest(unittest.TestCase):
    def test_fixed_standard_configuration(self):
        for kwargs in ({"action_dim": 32}, {"poly_degree": 9}, {"pi05": False}, {"rtc_max_delay": 1}):
            with pytest.raises(ValueError, match="P1 requires|Standard P1"):
                Pi0FlashConfig(**kwargs)
        with pytest.raises(ValueError, match="Finite 58D"):
            Pi0FlashConfig(action_q01=(float("nan"),) * 58)

    def test_full_parameter_shape_and_freeze_partition(self):
        config = Pi0FlashConfig()
        model = nnx.eval_shape(config.create, jax.random.key(42))
        frozen = nnx.state(model, nnx.All(nnx.Param, is_vlm_parameter))
        trainable = nnx.state(model, is_flash_trainable_parameter)
        assert len(frozen.flat_state()) == 32
        assert len(trainable.flat_state()) == 19
        assert sum(x.size for x in jax.tree.leaves(frozen)) == 2923335408
        assert sum(x.size for x in jax.tree.leaves(trainable)) == 430151738
        assert (model.action_horizon, model.coefficient_tokens, model.action_dim) == (50, 7, 58)
        assert frozen.flat_state().keys().isdisjoint(trainable.flat_state().keys())

    def test_future_labels_cannot_enter_sampling(self):
        params = inspect.signature(Pi0Flash.sample_actions).parameters
        assert "history" in params
        assert "target_ext" not in params
        assert "actions" not in params
        # Fail explicitly instead of treating existing (obs,actions) loader as FLASH-ready.
        model = nnx.eval_shape(Pi0FlashConfig().create, jax.random.key(0))
        with pytest.raises(ValueError, match="explicit causal measured history"):
            model.compute_loss(jax.random.key(0), None, np.zeros((1, 50, 58)))


if __name__ == "__main__":
    unittest.main()
