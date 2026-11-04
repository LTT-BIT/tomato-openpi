"""Observation-based FLASH updates, with differentiable prefix conditioning.

Unlike the legacy frozen-VLM diagnostics, never accept a precomputed KV cache.
Both the optimizer and differentiation use the same configuration-derived filter.
"""

from flax import nnx
import jax
import jax.numpy as jnp
import optax

from openpi.training.optimizer import AdamW


def create_optimizer(model, config, learning_rate):
    if model.freeze_vlm != config.freeze_vlm:
        raise ValueError("Model/config VLM freeze mode mismatch")
    return nnx.Optimizer(model, AdamW().create(learning_rate), wrt=config.get_trainable_filter())


def make_train_step(config):
    """Return a checked update; reject nonfinite loss/gradients BEFORE mutation.

    Caller owns PRNG splitting, window order and checkpointing. Pass real current
    observation, measured history (8,58), and separate supervision (54,58).
    Prefix is recomputed inside autodiff on every update, including FullFT.
    """
    diff = nnx.DiffState(0, config.get_trainable_filter())

    @nnx.jit
    def gradients(model, observation, history, target_ext, rng):
        def objective(m):
            values = m.loss_components(rng, observation, history=history, target_ext=target_ext, train=True)
            metrics = {key: jnp.mean(value) for key, value in values.items()}
            return metrics["loss"], metrics

        (_, metrics), grads = nnx.value_and_grad(objective, argnums=diff, has_aux=True)(model)
        metrics["gradient_norm"] = optax.global_norm(grads)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]))
        finite &= jnp.all(jnp.stack([jnp.isfinite(x) for x in metrics.values()]))
        return metrics, grads, finite

    @nnx.jit
    def apply(optimizer, grads):
        optimizer.update(grads)

    def step(optimizer, observation, *, history, target_ext, rng):
        if optimizer.model.freeze_vlm != config.freeze_vlm:
            raise ValueError("Model/config VLM freeze mode mismatch")
        # Avoid silently using a legacy expert-only optimizer in FullFT mode.
        expected = nnx.state(optimizer.model, config.get_trainable_filter()).flat_state().keys()
        actual = nnx.state(optimizer.model, optimizer.wrt).flat_state().keys()
        if expected != actual:
            raise ValueError("Optimizer parameter coverage differs from FLASH configuration")
        metrics, grads, finite = gradients(optimizer.model, observation, history, target_ext, rng)
        if not bool(jax.device_get(finite)):
            raise FloatingPointError("Nonfinite FLASH loss/gradient; optimizer not updated")
        apply(optimizer, grads)
        return metrics

    return step
