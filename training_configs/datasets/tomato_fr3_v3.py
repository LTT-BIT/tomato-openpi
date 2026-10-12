"""Historical C0 v3 merged data identity, with the recorded normalization digest."""

import os
from pathlib import Path

from training_configs.datasets.common import Dataset
from training_configs.datasets.common import runtime_root

PROMPT = (
    "Pick up a tomato truss with the right hand, then pick a cherry tomato "
    "with the left hand and place it in the left basket."
)
NORM_AUDIT_SHA256 = "f933853e8e1bfbf8a7f37e377bd9a5a87c57d37a437685a95e464bcc65c24e23"


def build_dataset() -> Dataset:
    runtime = runtime_root()
    source_root = Path(os.environ.get("PI05_SOURCE_DATA_DIR", "data/harvest-tomato"))
    return Dataset(
        version="tomato_all_full_v1_30hz",
        repo_id="fr3_wuji/tomato_all_full_v1",
        root=os.environ.get("PI05_V3_DATASET_ROOT", str(runtime / "data/tomato_all_full_v1")),
        assets_dir=os.environ.get("PI05_V3_ASSETS_DIR", str(runtime / "assets")),
        asset_id="tomato_all_full_v1_30hz",
        sources=tuple(str((source_root / item).resolve()) for item in ("0925/A", "0925/B", "1002_AB_190/full")),
        prompt_example=PROMPT,
        expected_norm_audit_sha256=NORM_AUDIT_SHA256,
    )
