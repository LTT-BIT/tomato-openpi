"""Keep read-only source comparison usable without the removed data converters."""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from examples.fr3_wuji import verify_dataset


@pytest.mark.parametrize("case", ["matching", "wrong_vectors", "shifted_images", "static_images"])
def test_source_comparison_is_independent_and_read_only(tmp_path, monkeypatch, capsys, case):
    source = tmp_path / "captures/episode_1"
    (source / "meta").mkdir(parents=True)
    (source / "meta/info.json").write_text("{}")
    target = tmp_path / "prepared"
    target.mkdir()
    vectors = np.arange(9, dtype=np.float32)[:, None]
    intensities = [0] * 9 if case == "static_images" else (np.arange(9) * 20).tolist()
    table = pa.table(
        {
            "timestamp": (np.arange(9) / 30).tolist(),
            **{key: vectors.tolist() for key in verify_dataset.VECTOR_KEYS},
            **dict.fromkeys(verify_dataset.CAMERA_KEYS, intensities),
            "unused_depth": [{"payload": [1, 2]}] * 9,
        }
    )
    pq.write_table(table, source / "episode.parquet")
    (source / "dataloader.py").write_text(
        "from types import SimpleNamespace\n"
        "import torch\n"
        "class LeRobotEpisodeDataset:\n"
        "    def __init__(self, root, vector_keys, video_keys):\n"
        "        self._tables = {}\n"
        "        self.episodes = [SimpleNamespace(parquet_path=root / 'episode.parquet')]\n"
        "        self.vector_keys, self.video_keys = vector_keys, video_keys\n"
        "    def __len__(self):\n"
        "        return len(self._table(0))\n"
        "    def __getitem__(self, row):\n"
        "        table = self._table(0)\n"
        "        assert 'unused_depth' not in table.column_names\n"
        "        result = {key: torch.tensor(table[key][row].as_py()) for key in self.vector_keys}\n"
        "        result.update({key: torch.full((3, 4, 4), table[key][row].as_py(), dtype=torch.uint8) "
        "for key in self.video_keys})\n"
        "        return result\n"
        "    def close(self):\n"
        "        self._tables.clear()\n"
    )
    original = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    class PreparedDataset:
        def __init__(self, repo_id, root):
            assert root == target
            self.meta = SimpleNamespace(
                total_episodes=1,
                total_frames=9,
                tasks=["Pick the tomato."],
                episodes={0: {"length": 9}},
                features={key: {"shape": [1]} for key in (*verify_dataset.VECTOR_KEYS, *verify_dataset.CAMERA_KEYS)},
            )

        def __getitem__(self, row):
            value = vectors[row].copy()
            if case == "wrong_vectors" and row == 4:
                value += 1
            image_row = min(row + 1, 8) if case == "shifted_images" else row
            return {
                "episode_index": torch.tensor(0),
                **{key: torch.from_numpy(value) for key in verify_dataset.VECTOR_KEYS},
                **{
                    key: torch.full((3, 4, 4), intensities[image_row] / 255, dtype=torch.float32)
                    for key in verify_dataset.CAMERA_KEYS
                },
            }

    monkeypatch.setattr(verify_dataset, "LeRobotDataset", PreparedDataset)
    args = verify_dataset.Args(src_root=source.parent, root=target, episodes=1, samples=3, neighbourhood=1)
    if case in ("wrong_vectors", "shifted_images"):
        with pytest.raises(AssertionError, match="failed verification"):
            verify_dataset.main(args)
    else:
        verify_dataset.main(args)
        output = capsys.readouterr().out
        assert ("VECTOR_OK_ALIGNMENT_UNVERIFIED" in output) == (case == "static_images")
    assert all(path.read_bytes() == content for path, content in original.items())
    assert not list(target.iterdir())


def test_source_comparison_requires_explicit_paths_and_valid_sample_counts(tmp_path):
    import tyro

    with pytest.raises(SystemExit):
        tyro.cli(verify_dataset.Args, args=[])
    with pytest.raises(ValueError, match="counts must be positive"):
        verify_dataset.main(verify_dataset.Args(src_root=tmp_path, root=tmp_path, episodes=0))
