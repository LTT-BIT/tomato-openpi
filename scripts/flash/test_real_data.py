"""Opt-in cached real-data integration with the actual main adapter."""

import json
from pathlib import Path
import unittest

import numpy as np
import pyarrow.parquet as pq
from reconstruct import load


class RealDataTest(unittest.TestCase):
    def test_canonical_mapping_matches_main_on_every_real_episode(self):
        directory = Path(__file__).resolve().parents[2] / ".cache/flash_samples"
        if not (directory / "manifest.json").exists():
            self.skipTest("Collect real samples first; never replace them with synthetic data")
        from openpi.models.model import ModelType
        from openpi.policies.fr3_sharpa_multimodal_policy import Fr3SharpaMultimodalInputs

        adapter = Fr3SharpaMultimodalInputs(ModelType.PI05)
        records = json.loads((directory / "manifest.json").read_text())
        assert records
        for record in records:
            canonical, state, fps = load(record, directory)
            table = pq.read_table(directory / record["local_file"])
            result = adapter(
                {
                    "observation/state": np.asarray(table["observation.state"][0].as_py()),
                    "actions": np.asarray(table["action"].to_pylist()[:50]),
                    "observation/image": np.zeros((2, 2, 3), np.uint8),
                    "observation/left_wrist_image": np.zeros((2, 2, 3), np.uint8),
                    "observation/right_wrist_image": np.zeros((2, 2, 3), np.uint8),
                }
            )
            np.testing.assert_allclose(result["state"], state[0], atol=1e-7)
            np.testing.assert_allclose(result["actions"], canonical[:50], atol=1e-7)
            assert fps == 30


if __name__ == "__main__":
    unittest.main()
