"""Check prepared subtask data, decode every video, and exercise training loader."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

from openpi.training import config
from openpi.training import data_loader


def decode_video(item):
    path, length = item
    with av.open(str(path)) as container:
        count = sum(1 for _ in container.decode(video=0))
    if count != length:
        raise ValueError(f"{path}: decoded {count}, expected {length}")
    return count


def main():
    cfg = config.get_config("pi05_fr3_wuji_1001_subtasks_30hz")
    root = Path(cfg.data.lerobot_datasets[0].root)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(s) for s in (root / "meta/episodes.jsonl").read_text().splitlines()]
    videos = []
    offset = 0
    for i, e in enumerate(episodes):
        assert e["episode_index"] == i
        assert e["length"] >= 50
        assert e["source_task_index"] in (1, 2, 3, 4)
        assert e["segment_id"].startswith("subtask_")
        kwargs = {"episode_chunk": i // info["chunks_size"], "episode_index": i}
        table = pq.read_table(root / info["data_path"].format(**kwargs))
        assert len(table) == e["length"]
        for key, expected in [("episode_index", i), ("task_index", e["task_index"])]:
            assert np.all(np.asarray(table[key]) == expected)
        np.testing.assert_array_equal(np.asarray(table["index"]), np.arange(offset, offset + len(table)))
        np.testing.assert_array_equal(np.asarray(table["frame_index"]), np.arange(len(table)))
        np.testing.assert_allclose(np.asarray(table["timestamp"]), np.arange(len(table)) / 30, atol=1e-5)
        for key, dim in [("observation.state", 108), ("action", 54)]:
            x = np.asarray(table[key].to_pylist())
            assert x.shape == (len(table), dim)
            assert np.isfinite(x).all()
        for key, feature in info["features"].items():
            if feature["dtype"] == "video":
                videos.append((root / info["video_path"].format(**kwargs, video_key=key), len(table)))
        offset += len(table)
    assert offset == info["total_frames"] == 79516
    assert len(episodes) == 392
    with ThreadPoolExecutor(max_workers=4) as pool:
        decoded = 0
        for n, count in enumerate(pool.map(decode_video, videos), 1):
            decoded += count
            if n % 150 == 0:
                print(f"Decoded {n}/{len(videos)} videos", flush=True)
    resolved = cfg.data.create(cfg.assets_dirs, cfg.model)
    for stats in resolved.norm_stats.values():
        assert stats.mean.shape == (54,)
        assert np.isfinite(stats.mean).all()
        assert np.all(stats.std > 0)
        assert np.all(stats.q99 > stats.q01)
    dataset = data_loader.create_torch_dataset(resolved, cfg.model.action_horizon, cfg.model)
    transformed = data_loader.transform_dataset(dataset, resolved)
    offset = 0
    tested_tasks = set()
    for e in episodes:
        if e["task_index"] not in tested_tasks:
            for index in (offset, offset + e["length"] - 1):
                sample = transformed[index]
                assert sample["actions"].shape == (50, 54)
                assert np.isfinite(sample["actions"]).all()
            # LeRobot should repeat the last action at the episode boundary.
            raw = dataset[offset + e["length"] - 1]["action"].numpy()
            np.testing.assert_array_equal(raw, np.repeat(raw[:1], 50, axis=0))
            tested_tasks.add(e["task_index"])
        offset += e["length"]
    # Use the actual configured batch size and worker count on CPU.
    loader = data_loader.create_data_loader(cfg, shuffle=True, num_batches=2)
    for obs, actions in loader:
        assert actions.shape == (8, 50, 54)
        assert obs.state.shape == (8, 54)
        assert np.isfinite(actions).all()
        assert np.isfinite(obs.state).all()
    result = {
        "status": "passed",
        "episodes": len(episodes),
        "frames": offset,
        "videos": len(videos),
        "decoded_video_frames": decoded,
        "full_video_decode": True,
        "tested_task_indices": sorted(tested_tasks),
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "action_horizon": 50,
        "backend": "cpu",
        "training_step_tested": False,
    }
    (root / "conversion/training_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
