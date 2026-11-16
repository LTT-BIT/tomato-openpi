"""Exercise the real shared training step and checkpoint asset callback."""

from dataclasses import replace
import json
from types import SimpleNamespace

from etils import epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import pi0_fovea
from openpi.models import siglip
from openpi.shared.normalize import NormStats
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import utils
from openpi.training.fovea_dataset import digest
from scripts.train import train_step


def test_training_update_metrics_and_checkpoint_contract(tmp_path, monkeypatch):
    original = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda **kw: original(**{**kw, "variant": "mu/224"}))
    cfg = pi0_fovea.FoveaConfig(
        dtype="float32",
        action_dim=2,
        action_horizon=4,
        execute_steps=2,
        max_token_len=3,
        image_keys=("base_0_rgb",),
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = cfg.create(jax.random.key(0))
    definition, params = nnx.split(model)
    tx = optax.sgd(1e-3)
    state = utils.TrainState(
        step=jnp.array(0), params=params, model_def=definition, opt_state=tx.init(params), tx=tx, ema_decay=None
    )
    training = config.TrainConfig(name="fovea-cpu-test", model=cfg, augmentation_seed=42002)
    obs = replace(cfg.fake_obs(), foveation_tokens=jnp.array([[256010, 256020, 256500, 256700]], jnp.int32))
    before = np.asarray(model.action_out_proj.kernel.value).copy()
    updated, metrics = jax.jit(lambda s, o: train_step(training, jax.random.key(2), s, (o, jnp.zeros((1, 4, 2)))))(
        state, obs
    )
    assert int(updated.step) == 1
    assert {"loss", "action_loss", "box_loss", "grad_norm", "param_norm"} == set(metrics)
    assert all(np.isfinite(v) for v in metrics.values())
    np.testing.assert_allclose(metrics["loss"], metrics["action_loss"] + 0.1 * metrics["box_loss"], rtol=1e-6)
    actual = nnx.merge(updated.model_def, updated.params)
    assert not np.array_equal(before, actual.action_out_proj.kernel.value)
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "manifest.json").write_text(json.dumps({"contract": {"version": "test"}}))
    stats = {
        k: NormStats(mean=np.zeros(2), std=np.ones(2), q01=-np.ones(2), q99=np.ones(2)) for k in ["state", "actions"]
    }
    data = SimpleNamespace(norm_stats=stats, asset_id="fovea-test", fovea_labels_root=str(labels))
    assets = epath.Path(tmp_path / "checkpoint/assets")
    assets.mkdir(parents=True)

    class Manager:
        def save(self, step, items):
            assert step == 1
            items["assets"](assets)

    checkpoints.save_state(Manager(), updated, SimpleNamespace(data_config=lambda: data), 1)
    record = json.loads((assets / "fovea_contract.json").read_text())
    assert record["training_checkpoint"] is True
    assert record["norm_stats_sha256"] == digest(assets / "fovea-test/norm_stats.json")
    assert record["labels_manifest_sha256"] == digest(labels / "manifest.json")
