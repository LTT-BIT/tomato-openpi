"""Check a converted FR3/Wuji dataset against the original captures.

Vectors must match exactly. Images are re-encoded with a lossy codec, so alignment is
checked instead: a converted frame has to be closer to its own source frame than to any
neighbouring source frame. Near-static frames cannot tell neighbours apart, so the
comparison is restricted to the highest-motion rows. cam0 records at 20fps while the
dataset runs at 30fps, which makes some neighbouring source frames identical; those are
excluded from the comparison rather than counted as competitors.

Example:
    .venv/bin/python -m examples.fr3_wuji.verify_dataset \
        --src-root data/raw_captures --root data/tomato_fulltask_v1 --repo-id fr3_wuji/tomato_fulltask_v1
"""

import dataclasses
import importlib.util
import pathlib
import sys

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
import tyro

SOURCE_LOADER_FILENAME = "dataloader.py"
VECTOR_KEYS = ("observation.state", "action")
CAMERA_KEYS = ("observation.images.cam0", "observation.images.cam1", "observation.images.cam2")


def _load_source_loader(module_path: pathlib.Path):
    """Read source captures through their original timestamp-aware loader."""
    spec = importlib.util.spec_from_file_location("fr3_wuji_source_dataloader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load source dataloader: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_capture_reader(source_loader):
    """Read only vectors and camera references, excluding unused depth payloads."""
    columns = ["timestamp", *VECTOR_KEYS, *CAMERA_KEYS]

    class CaptureReader(source_loader.LeRobotEpisodeDataset):
        def _table(self, position: int):
            table = self._tables.get(position)
            if table is None:
                table = pq.read_table(self.episodes[position].parquet_path, columns=columns)
                self._tables[position] = table
            return table

    return CaptureReader


def _find_captures(src_root: pathlib.Path, max_episodes: int) -> list[pathlib.Path]:
    def episode_order(path):
        digits = "".join(c for c in path.name if c.isdigit())
        return (int(digits) if digits else 0, path.name)

    captures = sorted((path for path in src_root.iterdir() if (path / "meta/info.json").is_file()), key=episode_order)
    if not captures:
        raise FileNotFoundError(f"No LeRobot captures found under {src_root}")
    return captures[:max_episodes]


@dataclasses.dataclass
class Args:
    src_root: pathlib.Path
    root: pathlib.Path
    repo_id: str = "fr3_wuji/tomato"
    # Number of episodes to check, taken from the start of the dataset.
    episodes: int = 2
    # Number of frames sampled per episode.
    samples: int = 6
    # How far to look on each side when testing frame alignment.
    neighbourhood: int = 3
    # Rows probed while searching for motion, per sample.
    probes_per_sample: int = 8
    # Minimum mean pixel difference (uint8 units) between distinct source frames.
    # This threshold must not depend on target/source mismatch: a shifted target
    # would otherwise exclude the very neighbour that reveals the shift.
    discriminability: float = 2.0


def _to_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """LeRobot returns float CHW in [0, 1]; the captures return uint8 CHW."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image.transpose(1, 2, 0) if image.shape[0] == 3 else image


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.int16) - right.astype(np.int16)).mean())


def _pick_moving_rows(source, key: str, episode_length: int, args: Args) -> list[int]:
    """Rank rows by how much the scene changes, so neighbours are distinguishable."""
    first = args.neighbourhood
    last = episode_length - args.neighbourhood - 1
    stride = max(1, (last - first) // max(1, args.samples * args.probes_per_sample))
    scored = []
    for row in range(first, last, stride):
        motion = _distance(_to_hwc_uint8(source[row][key].numpy()), _to_hwc_uint8(source[row + 1][key].numpy()))
        scored.append((motion, row))
    scored.sort(reverse=True)
    return [row for _, row in scored[: args.samples]]


def main(args: Args) -> None:
    if min(args.episodes, args.samples, args.neighbourhood, args.probes_per_sample) <= 0:
        raise ValueError("Verification episode, sample and neighbourhood counts must be positive")
    if not np.isfinite(args.discriminability) or args.discriminability <= 0:
        raise ValueError("discriminability must be positive and finite")
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    captures = _find_captures(args.src_root, max_episodes=args.episodes)
    capture_reader = _make_capture_reader(_load_source_loader(captures[0] / SOURCE_LOADER_FILENAME))

    print(f"{dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    print(f"tasks: {dataset.meta.tasks}")
    for key in (*VECTOR_KEYS, *CAMERA_KEYS):
        print(f"  {key}: {dataset.meta.features[key]['shape']}")

    offset = 0
    for episode_position, capture in enumerate(captures):
        source = capture_reader(capture, vector_keys=VECTOR_KEYS, video_keys=CAMERA_KEYS)
        try:
            episode_length = dataset.meta.episodes[episode_position]["length"]
            if episode_length != len(source):
                raise AssertionError(f"{capture.name}: length {episode_length} != source {len(source)}")

            worst_vector = 0.0
            for row in np.linspace(0, episode_length - 1, args.samples).astype(int):
                item = dataset[offset + int(row)]
                expected = source[int(row)]
                if item["episode_index"].item() != episode_position:
                    raise AssertionError(f"{capture.name}: episode index mismatch at row {row}")
                for key in VECTOR_KEYS:
                    diff = float(np.abs(item[key].numpy() - expected[key].numpy()).max())
                    worst_vector = max(worst_vector, diff)

            worst_margin = None
            comparisons = 0
            failures = []
            for key in CAMERA_KEYS:
                # Each camera moves at its own moments, so rank rows per camera.
                for row in _pick_moving_rows(source, key, episode_length, args):
                    own = _to_hwc_uint8(source[row][key].numpy())
                    converted = _to_hwc_uint8(dataset[offset + row][key].numpy())
                    own_distance = _distance(converted, own)

                    competitors = {}
                    for shift in range(-args.neighbourhood, args.neighbourhood + 1):
                        if shift == 0:
                            continue
                        candidate = _to_hwc_uint8(source[row + shift][key].numpy())
                        # Repeated/near-static source frames cannot decide alignment.
                        if _distance(candidate, own) > args.discriminability:
                            competitors[shift] = _distance(converted, candidate)
                    if not competitors:
                        continue

                    comparisons += 1
                    closest = min(competitors, key=competitors.get)
                    margin = competitors[closest] - own_distance
                    worst_margin = margin if worst_margin is None else min(worst_margin, margin)
                    if margin <= 0:
                        failures.append((key, row, own_distance, closest, competitors[closest]))

            status = "OK" if not failures and worst_vector == 0.0 else "FAIL"
            if status == "OK" and comparisons == 0:
                status = "VECTOR_OK_ALIGNMENT_UNVERIFIED"
            margin_text = "n/a" if worst_margin is None else f"{worst_margin:.2f}"
            print(
                f"[{status}] {capture.name}: {episode_length} frames, "
                f"max vector diff {worst_vector:g}, "
                f"{comparisons} alignment checks, min margin {margin_text}"
            )
            for key, row, own_distance, closest, closest_distance in failures:
                print(
                    f"    {key} row {row}: own frame {own_distance:.2f} is not closer than "
                    f"offset {closest} at {closest_distance:.2f}"
                )
            if status == "FAIL":
                raise AssertionError(f"{capture.name} failed verification")

            offset += episode_length
        finally:
            source.close()

    print("Sampled vector checks passed. Review alignment counts; zero comparisons cannot verify alignment.")


if __name__ == "__main__":
    main(tyro.cli(Args))
