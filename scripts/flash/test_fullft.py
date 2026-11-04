"""Gradient-path regression: real dummy Gemma + small replacement vision encoder.

This verifies graph connectivity and optimizer behavior, not full-size GPU capacity
or pretrained-model convergence. Full-size parameter coverage is checked separately.
"""

import dataclasses
from unittest.mock import patch

from flax import linen as nn
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import siglip
from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.training.flash_checkpoint import is_flash_trainable_parameter
from openpi.training.flash_checkpoint import is_vlm_parameter
from openpi.training.flash_checkpoint import restore_offline_checkpoint
from openpi.training.flash_checkpoint import save_offline_checkpoint
from openpi.training.flash_train import create_optimizer
from openpi.training.flash_train import make_train_step


class SmallVision(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, image, *, train=False):
        del train
        return nn.Dense(self.num_classes)(image.mean(axis=(1, 2))[:, None, :]), None


@pytest.mark.parametrize("freeze_vlm", [True, False])
def test_real_attention_gradient_and_update(freeze_vlm, monkeypatch, tmp_path):
    config = Pi0FlashConfig(
        freeze_vlm=freeze_vlm, paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32"
    )
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    with patch.object(siglip, "Module", side_effect=lambda **kw: SmallVision(kw["num_classes"])):
        model = config.create(jax.random.key(42))
    observation = config.fake_obs()
    observation = dataclasses.replace(
        observation,
        images={key: jnp.full_like(value, 0.4) for key, value in observation.images.items()},
        tokenized_prompt=jnp.ones_like(observation.tokenized_prompt),
    )
    history = jnp.linspace(-0.3, 0.5, 8 * 58).reshape(1, 8, 58)
    target = jnp.linspace(-0.2, 0.6, 54 * 58).reshape(1, 54, 58)
    optimizer = create_optimizer(model, config, 2.5e-5)
    step = make_train_step(config)
    before = {path: np.array(v.value) for path, v in nnx.state(model, nnx.Param).flat_state().items()}
    metrics = step(optimizer, observation, history=history, target_ext=target, rng=jax.random.key(10))
    assert int(optimizer.step.value) == 1
    assert all(np.isfinite(float(value)) for value in metrics.values())
    # Fresh dummy adaRMS gates start at zero: upstream/time gradients open after update 1.
    step(optimizer, observation, history=history, target_ext=target, rng=jax.random.key(11))
    assert int(optimizer.step.value) == 2
    changed = {
        path
        for path, v in nnx.state(model, nnx.Param).flat_state().items()
        if not np.array_equal(before[path], np.asarray(v.value))
    }
    assert any(path[0] == "action_out_proj" for path in changed)
    assert any(path[0] == "time_mlp_out" for path in changed)
    assert any(path[:2] == ("PaliGemma", "llm") and not is_vlm_parameter(path) for path in changed)
    if freeze_vlm:
        assert not any(is_vlm_parameter(path) for path in changed)
    else:
        assert any(path[:2] == ("PaliGemma", "img") for path in changed)
        assert any(path[:2] == ("PaliGemma", "llm") and is_vlm_parameter(path) for path in changed)
        # Reject an expert-only optimizer despite matching model/config FullFT flags.
        wrong = nnx.Optimizer(model, optimizer.tx, wrt=is_flash_trainable_parameter)
        with pytest.raises(ValueError, match="coverage"):
            step(wrong, observation, history=history, target_ext=target, rng=jax.random.key(12))
    graph, template = nnx.split(optimizer)
    path = tmp_path / "checkpoint"
    save_offline_checkpoint(
        path,
        nnx.state(model).to_pure_dict(),
        optimizer_state=nnx.state(optimizer, lambda p, v: p[0] != "model").to_pure_dict(),
        metadata={"config": dataclasses.asdict(config), "normalization": {}, "step": 2, "optimizer": "AdamW"},
    )
    params, metadata, adam = restore_offline_checkpoint(path)
    template.replace_by_pure_dict(jax.device_put({"model": params, **adam}))
    restored = nnx.merge(graph, template)
    assert metadata["step"] == int(restored.step.value) == 2
    for opt in (optimizer, restored):
        step(opt, observation, history=history, target_ext=target, rng=jax.random.key(12))
    for left, right in zip(jax.tree.leaves(nnx.state(optimizer)), jax.tree.leaves(nnx.state(restored)), strict=True):
        np.testing.assert_array_equal(left, right)
    output = model.sample_actions(jax.random.key(0), observation, history=history)
    assert output.shape == (1, 50, 58)
    assert np.isfinite(np.asarray(output)).all()
    # Nonfinite inputs must not advance Adam or modify parameters.
    before_bad = {p: np.array(v.value) for p, v in nnx.state(optimizer).flat_state().items()}
    with pytest.raises(FloatingPointError, match="not updated"):
        step(optimizer, observation, history=history, target_ext=target * jnp.nan, rng=jax.random.key(13))
    for path, value in nnx.state(optimizer).flat_state().items():
        np.testing.assert_array_equal(value.value, before_bad[path])


def test_fullsize_fullft_parameter_coverage():
    config = Pi0FlashConfig(freeze_vlm=False)
    model = nnx.eval_shape(config.create, jax.random.key(42))
    assert not nnx.state(model, nnx.All(nnx.Param, config.get_freeze_filter())).flat_state()
    trainable = nnx.state(model, config.get_trainable_filter())
    assert len(trainable.flat_state()) == 51
    assert sum(x.size for x in jax.tree.leaves(trainable)) == 3353487146
