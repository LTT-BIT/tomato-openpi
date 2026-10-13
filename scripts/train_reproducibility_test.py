"""Exercise the real JAX loop and Orbax resume with a small CPU-only model."""

from contextlib import contextmanager
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import optimizer
from openpi.training import utils
from scripts import train


class TinyModel(model_lib.BaseModel):
    def __init__(self, rng):
        super().__init__(action_dim=2, action_horizon=2, max_token_len=8)
        self.weight = nnx.Param(jax.random.normal(rng, (1, 1)))

    def compute_loss(self, rng, observation, actions, *, train=False, augmentation_rng=None):
        del train
        inputs = observation.state[..., 0, None]
        if augmentation_rng is not None:
            inputs = inputs + jax.random.normal(augmentation_rng, inputs.shape)
        targets = actions[..., 0] + jax.random.normal(rng, actions.shape[:-1])
        return jnp.square(self.weight.value[0, 0] * inputs - targets)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return jnp.zeros((*observation.state.shape[:-1], 2, 2))


@pytest.mark.parametrize("scheduled", [False, True])
def test_interrupted_training_matches_uninterrupted(tmp_path, monkeypatch, scheduled):
    states = {}
    real_save = checkpoints.save_state
    real_session = train.tracking.session
    session_starts = []

    @contextmanager
    def capture_session(cfg, *, resuming, start_step):
        session_starts.append((resuming, start_step))
        with real_session(cfg, resuming=resuming, start_step=start_step) as record:
            yield record

    def tiny_init(cfg, init_rng, mesh, *, resume):
        del mesh, resume
        model = TinyModel(init_rng)
        params = nnx.state(model)
        tx = optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule)
        state = utils.TrainState(
            step=jnp.array(0),
            params=params,
            model_def=nnx.graphdef(model),
            opt_state=tx.init(params),
            tx=tx,
            ema_decay=None,
            ema_params=None,
        )
        # None permits the tiny state to use default replicated JIT placement.
        return state, None

    def capture_save(manager, state, loader, step):
        real_save(manager, state, loader, step)
        states[step] = jax.device_get(state)

    monkeypatch.setattr(train, "init_train_state", tiny_init)
    monkeypatch.setattr(train.tracking, "session", capture_session)
    monkeypatch.setattr(checkpoints, "save_state", capture_save)
    cfg = dataclasses.replace(
        config.get_config("debug"),
        checkpoint_base_dir=str(tmp_path),
        exp_name="continuous",
        overwrite=False,
        sampling_mode="epoch_shuffle",
        model_seed=42,
        data_seed=42001,
        augmentation_seed=42002,
        model=dataclasses.replace(config.get_config("debug").model, action_dim=2, action_horizon=2, max_token_len=8),
        num_train_steps=4,
        num_workers=0,
        batch_size=2,
        save_interval=1,
        log_interval=1,
    )
    if scheduled:
        cfg = config.ScheduledTrainConfig(
            **{field.name: getattr(cfg, field.name) for field in dataclasses.fields(cfg)},
            save_before_step=2,
            save_steps_before=(1,),
            save_interval_after=1,
        )
    train.main(cfg)
    expected = states[4 if scheduled else 3]

    class SimulatedInterruptionError(Exception):
        pass

    def interrupt_after_checkpoint(manager, state, loader, step):
        capture_save(manager, state, loader, step)
        if step == (2 if scheduled else 1):
            manager.wait_until_finished()
            manager.close()
            raise SimulatedInterruptionError

    interrupted_cfg = dataclasses.replace(cfg, exp_name="interrupted")
    monkeypatch.setattr(checkpoints, "save_state", interrupt_after_checkpoint)
    with pytest.raises(SimulatedInterruptionError):
        train.main(interrupted_cfg)
    monkeypatch.setattr(checkpoints, "save_state", capture_save)
    train.main(dataclasses.replace(interrupted_cfg, resume=True))
    assert session_starts == [(False, 0), (False, 0), (True, 2)]
    actual = states[4 if scheduled else 3]
    for a, b in zip(jax.tree.leaves(actual.params), jax.tree.leaves(expected.params), strict=True):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(actual.opt_state), jax.tree.leaves(expected.opt_state), strict=True):
        np.testing.assert_array_equal(a, b)
    assert int(actual.step) == int(expected.step) == 4
    # A completed finite plan must not attempt to fetch a fifth batch.
    train.main(dataclasses.replace(interrupted_cfg, resume=True))
    assert session_starts == [(False, 0), (False, 0), (True, 2)]
