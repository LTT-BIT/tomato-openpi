"""Verify unchanged all119 physical vectors before reusing their delta statistics."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.policies.fr3_sharpa_crop_policy import Fr3SharpaCropInputs
from openpi.training.fovea_dataset import digest
from openpi.training.norm_provenance import build_contract
from openpi.training.norm_provenance import validate_metadata
from training_configs.experiments.tomato_sharpa.subtasks.foveavla_v1 import ASSET_ID
from training_configs.experiments.tomato_sharpa.subtasks.foveavla_v1 import build_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-norm", type=Path, required=True)
    parser.add_argument(
        "--refresh-binding", action="store_true", help="Refresh only this generated bundle after code formatting"
    )
    args = parser.parse_args()
    config = build_config()
    root = Path(config.data.lerobot_datasets[0].root)
    source = json.loads((args.source_norm / "norm_metadata.json").read_text())
    contract = build_contract(config)
    reference = source["contract"]
    if (
        reference["action_horizon"] != 50
        or reference["action_dim"] != 58
        or reference["action_hz"] != 30
        or reference["datasets"][0]["files"] != contract["datasets"][0]["files"]
        or source["norm_stats_sha256"] != digest(args.source_norm / "norm_stats.json")
    ):
        raise ValueError("All119 source physical files/statistics differ")
    # Compare the actual new input/delta chain with the previous physical chain.
    repack, vectors = config.data.create_vector_transforms(config.model)
    new = transforms.compose([*repack.inputs, *vectors.inputs])
    old_mapping = {k: v for k, v in repack.inputs[0].structure.items() if k != "foveation_tokens"}
    old = transforms.compose(
        [
            transforms.RepackTransform(old_mapping),
            Fr3SharpaCropInputs(Pi0Config(pi05=True).model_type),
            transforms.DeltaActions(transforms.make_bool_mask(7, -22, 7, -22)),
        ]
    )
    checked = 0
    image = np.zeros((4, 4, 3), np.uint8)
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].to_pylist(), np.float32)
        action = np.asarray(table["action"].to_pylist(), np.float32)
        for i in [0, len(state) // 2, len(state) - 1]:
            raw = {
                "observation.state": state[i],
                "action": action[np.minimum(i + np.arange(50), len(state) - 1)],
                "prompt": "test",
                "observation.interaction_available": np.True_,
                "foveation_tokens": np.array([256000, 256000, 257023, 257023], np.int32),
                **dict.fromkeys(
                    (
                        "observation.images.cam0",
                        "observation.images.cam1",
                        "observation.images.cam2",
                        "observation.images.interaction",
                    ),
                    image,
                ),
            }
            a, b = new(raw), old(raw)
            for key in ("state", "actions"):
                np.testing.assert_array_equal(a[key], b[key])
            checked += 1
    out = Path(config.data.assets.assets_dir) / ASSET_ID
    if out.exists():
        if not args.refresh_binding or digest(out / "norm_stats.json") != source["norm_stats_sha256"]:
            raise FileExistsError(out)
        previous = json.loads((out / "norm_metadata.json").read_text())
        if not previous.get("reused_all119_statistics") or previous.get("repo_id") != config.data.repo_id:
            raise ValueError("Only this experiment's generated binding can be refreshed")
    else:
        out.mkdir(parents=True)
    shutil.copy2(args.source_norm / "norm_stats.json", out / "norm_stats.json")
    record = {
        **source,
        "contract": contract,
        "repo_id": config.data.repo_id,
        "datasets": [{"repo_id": config.data.repo_id, "frames": 357970, "episodes": 119}],
        "reused_all119_statistics": True,
        "vector_equivalence_samples": checked,
        "source_norm_metadata_sha256": digest(args.source_norm / "norm_metadata.json"),
        "source_norm_path": str(args.source_norm),
        "training_pairs": config.policy_metadata["training_pairs"],
        "sampling": "Normalization covers all119 physical frames; action/box training uses valid future-box pairs.",
    }
    (out / "norm_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    status = validate_metadata(config, out, stats_sha256=digest(out / "norm_stats.json"))
    print(
        json.dumps({"status": status, "samples": checked, "path": str(out), "sha256": digest(out / "norm_stats.json")})
    )


if __name__ == "__main__":
    main()
