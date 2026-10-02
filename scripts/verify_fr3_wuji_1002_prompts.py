"""验证新 prompt、原始数据保真、动作边界和实际多进程训练 batch。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.training import config
from openpi.training import data_loader


def main():
    cfg = config.get_config("pi05_fr3_wuji_1002_subtasks_detailed_30hz")
    root = Path(cfg.data.lerobot_datasets[0].root)
    source = Path("/home/descfly/data-clean/subtasks")
    prompts = json.loads((Path(__file__).resolve().parents[1] / "docs/prompts_1002_subtasks.json").read_text())
    tasks = [json.loads(s) for s in (root / "meta/tasks.jsonl").read_text().splitlines()]
    assert {str(t["task_index"]): t["task"] for t in tasks} == prompts
    episodes = [json.loads(s) for s in (root / "meta/episodes.jsonl").read_text().splitlines()]
    for e in episodes:
        assert e["tasks"] == [prompts[str(e["task_index"])]]
        rel = f"data/chunk-000/episode_{e['episode_index']:06d}.parquet"
        assert pq.read_table(root / rel).equals(pq.read_table(source / rel))
    for p in (root / "videos").rglob("*.mp4"):
        assert (
            hashlib.sha256(p.read_bytes()).digest()
            == hashlib.sha256((source / p.relative_to(root)).read_bytes()).digest()
        )
    resolved = cfg.data.create(cfg.assets_dirs, cfg.model)
    dataset = data_loader.create_torch_dataset(resolved, 50, cfg.model)
    transformed = data_loader.transform_dataset(dataset, resolved)
    offset = 0
    for e in episodes:
        if e["episode_index"] < 4 or e["length"] < 50:
            assert dataset[offset]["prompt"] == prompts[str(e["task_index"])]
            sample = transformed[offset]
            assert sample["actions"].shape == (50, 54)
            assert np.isfinite(sample["actions"]).all()
            raw = dataset[offset + e["length"] - 1]["action"].numpy()
            np.testing.assert_array_equal(raw, np.repeat(raw[:1], 50, axis=0))
        offset += e["length"]
    loader = data_loader.create_data_loader(cfg, shuffle=True, num_batches=2)
    for obs, actions in loader:
        assert actions.shape == (8, 50, 54)
        assert obs.state.shape == (8, 54)
        assert np.isfinite(actions).all()
    report = {
        "status": "passed",
        "episodes": len(episodes),
        "frames": offset,
        "prompts_verified": True,
        "all_videos_identical_to_verified_source": True,
        "batch_size": 8,
        "num_workers": 8,
        "training_step_tested": False,
    }
    (root / "conversion/training_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
