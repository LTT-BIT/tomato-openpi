"""P3 resume contracts; tests execute no full-model training update."""

import json
from pathlib import Path
import tempfile
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from prepare_p3 import uniform_plan
from train_p3 import flow_key
from train_p3 import make_update
from train_p3 import random_audit

from openpi.training.flash_checkpoint import parameter_digest
from openpi.training.flash_checkpoint import restore_offline_checkpoint
from openpi.training.flash_checkpoint import save_offline_checkpoint
from openpi.training.flash_dataset import FlashLeRobotDataset
from openpi.training.optimizer import AdamW

ROOT = Path(__file__).resolve().parents[2]


class TinyFlow(nnx.Module):
    def __init__(self):
        self.projection = nnx.Linear(2, 2, rngs=nnx.Rngs(0))

    def loss_from_cache(self, rng, mask, cache, *, history, target_ext, train):
        del rng, mask, cache, train
        fm = jnp.mean((self.projection(history) - target_ext) ** 2)
        return {"loss": fm + fm, "flow_matching_loss": fm, "polynomial_consistency_loss": fm}


class P3Test(unittest.TestCase):
    def test_prng_wrap_and_global_step_are_exact(self):
        for step in (100, 101, 250, 251):
            a = random_audit(step)
            assert a["roundtrip"]
            assert a == random_audit(step)
        assert random_audit(101)["key_data"] == [837265607, 1489898847]
        assert not np.array_equal(jax.random.key_data(flow_key(101)), jax.random.key_data(flow_key(222)))

    def test_update_advances_optimizer_once_and_reports_losses(self):
        opt = nnx.Optimizer(TinyFlow(), AdamW().create(2.5e-5), wrt=nnx.Param)
        opt.step.value = jnp.asarray(100, dtype=jnp.uint32)
        metrics, norm, finite = jax.block_until_ready(
            make_update()(opt, None, None, jnp.ones((1, 2)), jnp.zeros((1, 2)), flow_key(101))
        )
        assert int(opt.step.value) == 101
        assert bool(finite)
        assert float(norm) > 0
        np.testing.assert_allclose(
            metrics["loss"], metrics["flow_matching_loss"] + metrics["polynomial_consistency_loss"]
        )

    def test_expert_increment_restores_reference_vlm_and_adam(self):
        base = {
            "PaliGemma": {"img": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)}},
            "action_in_proj": {"kernel": np.ones((2, 2), dtype=np.float32)},
        }
        delta = {"action_in_proj": {"kernel": np.full((2, 2), 7, dtype=np.float32)}}
        full = {**base, **delta}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_offline_checkpoint(
                root / "base", base, metadata={"config": {}, "normalization": {}, "step": 100, "optimizer": {}}
            )
            save_offline_checkpoint(
                root / "delta",
                delta,
                optimizer_state={"step": np.asarray(250, dtype=np.uint32)},
                metadata={
                    "config": {},
                    "normalization": {},
                    "step": 250,
                    "optimizer": {},
                    "checkpoint_kind": "flash_expert_increment",
                    "vlm_reference_checkpoint": str(root / "base"),
                    "vlm_sha256": parameter_digest(base, vlm_only=True),
                    "full_parameter_sha256": parameter_digest(full),
                },
            )
            restored, meta, moments = restore_offline_checkpoint(root / "delta")
            assert parameter_digest(restored) == parameter_digest(full)
            assert meta["step"] == int(moments["step"]) == 250

    def test_real_uniform_plan_is_frozen_disjoint_and_causal(self):
        folder = ROOT / ".cache/flash_p2_data"
        if not folder.exists():
            self.skipTest("Real snapshot unavailable")
        d = FlashLeRobotDataset(folder, ROOT / ".cache/flash_p2_images")
        fixed = json.loads((ROOT / "docs/flash/results/p25/evaluation_plan.json").read_text())
        p = uniform_plan(d, fixed)
        assert p == uniform_plan(d, fixed)
        assert len(p["windows"]) == 505
        assert p["source_count"] == 101
        old = {(w["id"], w["start"]) for w in fixed["windows"]}
        for w in p["windows"]:
            assert (w["id"], w["start"]) not in old
            assert d.split["assignments"][w["id"]] == "validation"
            assert w["start"] >= 7
            assert w["start"] + 51 < d.records[w["id"]]["rows"]
        frozen = json.loads((ROOT / ".cache/flash_p3/train_plan.json").read_text())
        candidate = json.loads((ROOT / "docs/flash/results/p25/p3_candidate_plan.json").read_text())
        assert frozen["windows"] == candidate["windows"][100:250]
        assert len(frozen["windows"]) == 150


if __name__ == "__main__":
    unittest.main()
