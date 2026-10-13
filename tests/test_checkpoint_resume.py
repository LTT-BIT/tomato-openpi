"""Reject absent checkpoints while accepting a real checkpoint at step zero."""

import jax.numpy as jnp
import pytest

from openpi.training.checkpoints import initialize_checkpoint_dir


@pytest.mark.parametrize("directory_exists", [False, True])
def test_resume_does_not_silently_restart_before_first_checkpoint(tmp_path, directory_exists):
    directory = tmp_path / "experiment"
    if directory_exists:
        directory.mkdir()
    with pytest.raises(FileNotFoundError, match="no saved checkpoint|does not exist"):
        initialize_checkpoint_dir(directory, keep_period=None, overwrite=False, resume=True)


def test_step_zero_checkpoint_is_a_valid_resume(tmp_path):
    checkpoint = tmp_path / "experiment"
    manager, resuming = initialize_checkpoint_dir(checkpoint, keep_period=None, overwrite=False, resume=False)
    assert not resuming
    try:
        manager.save(
            0,
            {
                "assets": lambda path: None,
                "train_state": {"step": jnp.array(1)},
                "params": {"params": {"w": jnp.array(2.0)}},
            },
        )
        manager.wait_until_finished()
    finally:
        manager.close()
    manager, resuming = initialize_checkpoint_dir(checkpoint, keep_period=None, overwrite=False, resume=True)
    try:
        assert resuming
        assert manager.latest_step() == 0
        restored = manager.restore(
            0, items={"train_state": {"step": jnp.array(0)}, "params": {"params": {"w": jnp.array(0.0)}}}
        )
        assert int(restored["train_state"]["step"]) == 1
        assert float(restored["params"]["params"]["w"]) == 2.0
    finally:
        manager.close()
