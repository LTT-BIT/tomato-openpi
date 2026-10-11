"""Merge all three selected full-task datasets and compute matching vector statistics."""

import json
from pathlib import Path

from examples.fr3_wuji.compute_norm_stats import compute
from examples.fr3_wuji.merge_lerobot_datasets import merge
from training_baseline.fullft_v3.pi05_fr3_wuji_fulltask_fullft_30hz_train import DATASET_REPO_ID
from training_baseline.fullft_v3.pi05_fr3_wuji_fulltask_fullft_30hz_train import DATASET_ROOT
from training_baseline.fullft_v3.pi05_fr3_wuji_fulltask_fullft_30hz_train import DATASET_SOURCES
from training_baseline.fullft_v3.pi05_fr3_wuji_fulltask_fullft_30hz_train import FULL_TASK_PROMPT
from training_baseline.fullft_v3.pi05_fr3_wuji_fulltask_fullft_30hz_train import NORM_ASSET_ID
from training_baseline.fullft_v3.pi05_fr3_wuji_fulltask_fullft_30hz_train import NORM_ASSETS_DIR
from training_baseline.fullft_v3.pi05_fr3_wuji_fulltask_fullft_30hz_train import build_config


def main() -> None:
    destination = Path(DATASET_ROOT)
    for source in DATASET_SOURCES:
        tasks = [json.loads(line)["task"] for line in (source / "meta/tasks.jsonl").read_text().splitlines()]
        if tasks != [FULL_TASK_PROMPT]:
            raise ValueError(f"Unexpected full-task prompts at {source}: {tasks}")
    if not destination.exists():
        merge(list(DATASET_SOURCES), destination, repo_id=DATASET_REPO_ID)
    else:
        provenance = json.loads((destination / "provenance/merge.json").read_text())
        expected = [str(source.resolve()) for source in DATASET_SOURCES]
        if [entry["root"] for entry in provenance["sources"]] != expected:
            raise ValueError("Existing merged data was built from different sources; use a new data version")
        print(f"Using existing merged dataset: {destination}")
    stats_path = Path(NORM_ASSETS_DIR) / NORM_ASSET_ID / "norm_stats.json"
    if stats_path.exists():
        raise FileExistsError(f"Statistics already exist: {stats_path}; use a new data version to recompute")
    compute(build_config())


if __name__ == "__main__":
    main()
