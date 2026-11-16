"""Bind unchanged baseline vector statistics to the independent visual-history adapter."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq

from openpi import transforms
from openpi.subtasks.visual_memory import HISTORY
from openpi.subtasks.visual_memory import RAW_KEYS
from openpi.training.norm_provenance import build_contract
from openpi.training.norm_provenance import validate_metadata
from openpi.training.reproducibility import file_sha256
from openpi.training.reproducibility import serialize_config
from training_configs.experiments.tomato_sharpa.subtasks.in_hand119_b_crop import build_config as build_baseline
from training_configs.experiments.tomato_sharpa.subtasks.visual_memory_v1 import build_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-norm", type=Path, required=True)
    args = parser.parse_args()
    baseline, cfg = build_baseline(), build_config()
    source = args.source_norm.resolve(strict=True)
    digest = file_sha256(source / "norm_stats.json")
    validate_metadata(baseline, source, stats_sha256=digest)
    contract = build_contract(cfg)
    if contract["datasets"] != build_contract(baseline)["datasets"]:
        raise ValueError("Visual memory must use the identical frozen physical source")
    old_repack, old_vectors = baseline.data.create_vector_transforms(baseline.model)
    new_repack, new_vectors = cfg.data.create_vector_transforms(cfg.model)
    original_chain = (*old_repack.inputs, *old_vectors.inputs)
    wrapper = new_vectors.inputs[0]
    if new_repack.inputs or serialize_config(wrapper.current_transforms) != serialize_config(original_chain):
        raise ValueError("Current vector transform chain differs from the baseline")
    old = transforms.compose(original_chain)
    new = transforms.compose(new_vectors.inputs)
    image = np.zeros((4, 4, 3), np.uint8)
    memory = cfg.model.memory
    history = {
        "image": dict.fromkeys(memory.history_keys, image),
        "image_mask": dict.fromkeys(memory.history_keys, np.False_),
    }
    root = Path(cfg.data.lerobot_datasets[0].root)
    checked = 0
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), np.float32)
        actions = np.asarray(table["action"].to_pylist(), np.float32)
        for index in (0, len(states) // 2, len(states) - 1):
            raw = {
                **dict.fromkeys(RAW_KEYS, image),
                "observation.state": states[index],
                "action": actions[np.minimum(index + np.arange(50), len(actions) - 1)],
                "observation.interaction_available": np.True_,
                "prompt": "test",
                HISTORY: history,
            }
            a, b = old(raw), new(raw)
            for key in ("state", "actions"):
                np.testing.assert_array_equal(a[key], b[key])
            checked += 1
    if not checked:
        raise ValueError("No physical vector equivalence samples")
    target = Path(cfg.data.assets.assets_dir) / cfg.data.assets.asset_id
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "norm_stats.json", target / "norm_stats.json")
    record = {
        "schema_version": 2,
        "contract": contract,
        "norm_stats_sha256": digest,
        "repo_id": cfg.data.repo_id,
        "config_name": cfg.name,
        "action_horizon": 50,
        "frames": 357970,
        "episodes": 119,
        "reused_baseline_statistics": True,
        "current_transform_chain_identical": True,
        "vector_equivalence_samples": checked,
        "source_norm_path": str(source),
        "source_norm_metadata_sha256": file_sha256(source / "norm_metadata.json"),
        "sampling": "All119 identical current physical vectors/action windows; image-only history added",
    }
    (target / "norm_metadata.json").write_text(json.dumps(record, indent=2) + "\n")
    status = validate_metadata(cfg, target, stats_sha256=digest)
    print(json.dumps({"status": status, "path": str(target), "samples": checked, "sha256": digest}))


if __name__ == "__main__":
    main()
