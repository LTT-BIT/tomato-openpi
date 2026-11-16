"""Real Pi05 flow-loss update and checkpoint contract with a small CPU backbone."""

from dataclasses import replace
import json
from types import SimpleNamespace

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import pi0_config
from openpi.models import siglip
from openpi.models.pi0_visual_memory import VisualMemoryConfig
from openpi.shared.normalize import NormStats
from openpi.subtasks.visual_memory import VisualMemorySpec
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import utils
from scripts.train import train_step


def test_flow_update_parameter_compatibility_and_checkpoint(tmp_path, monkeypatch):
    original = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda **kw: original(**{**kw, "variant": "mu/224"}))
    spec = VisualMemorySpec(frames=2, pool_grid=1, image_keys=("base_0_rgb",))
    options = {
        "dtype": "float32",
        "action_dim": 2,
        "action_horizon": 3,
        "max_token_len": 3,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "pi05": True,
    }
    cfg = VisualMemoryConfig(**options, memory=spec)
    model = cfg.create(jax.random.key(0))
    definition, params = nnx.split(model)
    baseline = pi0_config.Pi0Config(**options, image_keys=spec.image_keys).create(jax.random.key(0))
    baseline_params = nnx.state(baseline).to_pure_dict()
    # No additional parameters: initialization from the baseline is shape-compatible.
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(a, b), params.to_pure_dict(), baseline_params)
    control_cfg = VisualMemoryConfig(**options, memory=replace(spec, frames=1))
    control = control_cfg.create(jax.random.key(0))
    control_prefix = control.embed_prefix(control_cfg.fake_obs())
    baseline_prefix = baseline.embed_prefix(control_cfg.fake_obs())
    for a, b in zip(control_prefix, baseline_prefix, strict=True):
        np.testing.assert_array_equal(a, b)
    tx = optax.sgd(1e-3)
    state = utils.TrainState(
        step=jnp.array(0), params=params, model_def=definition, opt_state=tx.init(params), tx=tx, ema_decay=None
    )
    training = config.TrainConfig(name="visual-memory-cpu-test", model=cfg, augmentation_seed=42002)
    before = np.asarray(model.action_out_proj.kernel.value).copy()
    obs = cfg.fake_obs()
    updated, metrics = jax.jit(lambda s, o: train_step(training, jax.random.key(2), s, (o, jnp.zeros((1, 3, 2)))))(
        state, obs
    )
    assert int(updated.step) == 1
    assert all(np.isfinite(v) for v in metrics.values())
    actual = nnx.merge(updated.model_def, updated.params)
    assert not np.array_equal(before, actual.action_out_proj.kernel.value)
    actions = jax.jit(lambda o: actual.sample_actions(jax.random.key(3), o, num_steps=2))(obs)
    assert actions.shape == (1, 3, 2)
    assert np.isfinite(actions).all()
    stats = {
        k: NormStats(mean=np.zeros(2), std=np.ones(2), q01=-np.ones(2), q99=np.ones(2)) for k in ("state", "actions")
    }
    data = SimpleNamespace(norm_stats=stats, asset_id="test", visual_memory=spec)
    assets = epath.Path(tmp_path / "assets")
    assets.mkdir()

    class Manager:
        def save(self, step, items):
            assert step == 1
            items["assets"](assets)

    checkpoints.save_state(Manager(), updated, SimpleNamespace(data_config=lambda: data), 1)
    record = json.loads((assets / "visual_memory_contract.json").read_text())
    assert record == {**spec.contract(), "training_checkpoint": True}
