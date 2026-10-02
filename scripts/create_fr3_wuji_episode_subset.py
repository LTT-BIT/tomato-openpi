"""Copy selected LeRobot v2.1 episodes into a standalone, reindexed dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def create_subset(source: Path, output: Path, episode_indices: list[int]) -> None:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Subset already exists: {output}")
    if not episode_indices or len(episode_indices) != len(set(episode_indices)):
        raise ValueError("Provide distinct episode indices")

    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("This subset writer requires LeRobot v2.1 data")
    episodes = {row["episode_index"]: row for row in read_jsonl(source / "meta/episodes.jsonl")}
    episode_stats = {row["episode_index"]: row for row in read_jsonl(source / "meta/episodes_stats.jsonl")}
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for episode_index in episode_indices:
        if episode_index not in episodes or episode_index not in episode_stats:
            raise ValueError(f"Missing episode metadata or stats: {episode_index}")
        chunk = episode_index // info["chunks_size"]
        data_path = source / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
        video_paths = [
            source / info["video_path"].format(episode_chunk=chunk, episode_index=episode_index, video_key=key)
            for key in video_keys
        ]
        for path in (data_path, *video_paths):
            if not path.is_file():
                raise FileNotFoundError(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        root = Path(temporary)
        selected_episodes = []
        selected_stats = []
        next_frame_index = 0
        for new_index, original_index in enumerate(episode_indices):
            original_chunk = original_index // info["chunks_size"]
            new_chunk = new_index // info["chunks_size"]
            original_data = source / info["data_path"].format(
                episode_chunk=original_chunk, episode_index=original_index
            )
            table = pq.read_table(original_data)
            length = episodes[original_index]["length"]
            if table.num_rows != length or set(table["episode_index"].to_pylist()) != {original_index}:
                raise ValueError(f"Episode frame count or index mismatch: {original_data}")
            for column, values in (
                ("episode_index", [new_index] * length),
                ("index", range(next_frame_index, next_frame_index + length)),
            ):
                position = table.schema.get_field_index(column)
                table = table.set_column(position, column, pa.array(values, type=table.schema.field(column).type))
            new_data = root / info["data_path"].format(episode_chunk=new_chunk, episode_index=new_index)
            new_data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, new_data)

            for key in video_keys:
                original_video = source / info["video_path"].format(
                    episode_chunk=original_chunk, episode_index=original_index, video_key=key
                )
                new_video = root / info["video_path"].format(
                    episode_chunk=new_chunk, episode_index=new_index, video_key=key
                )
                new_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original_video, new_video)

            selected_episodes.append({**episodes[original_index], "episode_index": new_index})
            selected_stats.append({**episode_stats[original_index], "episode_index": new_index})
            next_frame_index += length

        subset_info = {
            **info,
            "total_episodes": len(episode_indices),
            "total_frames": next_frame_index,
            "total_videos": len(episode_indices) * len(video_keys),
            "total_chunks": (len(episode_indices) - 1) // info["chunks_size"] + 1,
            "splits": {"train": f"0:{len(episode_indices)}"},
        }
        meta_dir = root / "meta"
        meta_dir.mkdir()
        (meta_dir / "info.json").write_text(json.dumps(subset_info, indent=2) + "\n")
        shutil.copy2(source / "meta/tasks.jsonl", meta_dir / "tasks.jsonl")
        write_jsonl(meta_dir / "episodes.jsonl", selected_episodes)
        write_jsonl(meta_dir / "episodes_stats.jsonl", selected_stats)
        provenance = root / "conversion/subset_provenance.json"
        provenance.parent.mkdir()
        provenance.write_text(
            json.dumps(
                {
                    "source": str(source),
                    "source_episode_indices": episode_indices,
                    "subset_episode_indices": list(range(len(episode_indices))),
                    "fps": info["fps"],
                    "total_frames": next_frame_index,
                },
                indent=2,
            )
            + "\n"
        )
        root.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    args = parser.parse_args()
    create_subset(args.source, args.output, args.episodes)
    print(f"Created {len(args.episodes)} episodes at {args.output.resolve()}")


if __name__ == "__main__":
    main()
