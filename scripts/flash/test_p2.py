"""P2 source split, real boundary/causality and Adam state resumption tests."""

from pathlib import Path
import tempfile
import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from prepare_p2 import source_split
from train_p2 import content_digest
from train_p2 import optimizer_only

from openpi.training.flash_checkpoint import restore_offline_checkpoint
from openpi.training.flash_checkpoint import save_offline_checkpoint
from openpi.training.flash_dataset import FlashLeRobotDataset
from openpi.training.optimizer import AdamW

ROOT = Path(__file__).resolve().parents[2]


class P2Test(unittest.TestCase):
    def test_group_split_is_deterministic_and_aliases_together(self):
        records = [{"id": str(i), "source_recording_id": f"origin{i // 2}"} for i in range(20)]
        split = source_split(records, 42, ["origin2"], ["origin3"])
        assert split == source_split(records, 42, ["origin2"], ["origin3"])
        assert not set(split["train_sources"]) & set(split["validation_sources"])
        for i in range(0, 20, 2):
            assert split["assignments"][str(i)] == split["assignments"][str(i + 1)]
        assert split["assignments"]["4"] == "validation"
        assert split["assignments"]["6"] == "train"

    def test_complete_real_snapshot_and_causal_window_index(self):
        folder = ROOT / ".cache/flash_p2_data"
        if not (folder / "manifest.json").exists():
            self.skipTest("P2 snapshot not available")
        dataset = FlashLeRobotDataset(folder, ROOT / ".cache/flash_p2_images")
        assert len(dataset.records) > 3
        plan = dataset.training_plan(100, 42001)
        assert plan == dataset.training_plan(100, 42001)
        assert len({(i["id"], i["start"]) for i in plan}) == 100
        for index in plan:
            r = dataset.records[index["id"]]
            assert dataset.split["assignments"][index["id"]] == "train"
            assert index["start"] >= 7
            assert index["start"] + 51 < r["rows"]
        current = plan[0]
        batch = dataset.window(current)
        episode = dataset.episode(current["id"])
        start = current["start"]
        np.testing.assert_array_equal(batch["history"][0], episode["states"][start - 7 : start + 1])
        np.testing.assert_array_equal(batch["actions"], batch["target_ext"][:, 2:52])
        original = batch["history"].copy()
        episode["actions"][start + 1 :] += 100
        episode["states"][start + 1 :] -= 100
        np.testing.assert_array_equal(original, dataset.window(current)["history"])
        train_ids = {r["id"] for r in dataset.records.values() if dataset.split["assignments"][r["id"]] == "train"}
        assert set(dataset.normalization["train_sample_ids"]) == train_ids
        assert all(dataset.split["assignments"][v["id"]] == "validation" for v in dataset.validation_plan())

    def test_adam_checkpoint_resumes_same_next_update(self):
        model = nnx.Linear(2, 3, rngs=nnx.Rngs(0))
        tx = AdamW().create(2.5e-5)
        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        def advance(opt):
            grads = nnx.grad(lambda m: jnp.mean(m(jnp.ones((1, 2))) ** 2))(opt.model)
            opt.update(grads)

        advance(optimizer)
        graph, template = nnx.split(optimizer)
        template = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), template)
        with tempfile.TemporaryDirectory() as folder:
            save_offline_checkpoint(
                Path(folder) / "checkpoint",
                nnx.state(optimizer.model).to_pure_dict(),
                optimizer_state=nnx.state(optimizer, lambda p, v: p[0] != "model").to_pure_dict(),
                metadata={"config": {}, "normalization": {}, "step": 1, "optimizer": "AdamW"},
            )
            params, metadata, opt_state = restore_offline_checkpoint(Path(folder) / "checkpoint")
            template.replace_by_pure_dict(jax.device_put({"model": params, **opt_state}))
            resumed = nnx.merge(graph, template)
            assert int(resumed.step.value) == metadata["step"] == 1
            assert content_digest(optimizer_only(resumed).to_pure_dict()) == content_digest(
                optimizer_only(optimizer).to_pure_dict()
            )
            advance(optimizer)
            advance(resumed)
            for left, right in zip(
                jax.tree.leaves(nnx.state(optimizer)), jax.tree.leaves(nnx.state(resumed)), strict=True
            ):
                np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
