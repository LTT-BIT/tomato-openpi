"""Immutable metadata releases for prepared Sharpa data; no episode-count constants."""

import hashlib
import json
from pathlib import Path

from openpi.policies.fr3_sharpa_multimodal_policy import SOURCE_ACTION_NAMES
from openpi.policies.fr3_sharpa_multimodal_policy import SOURCE_STATE_NAMES
from training_configs.experiments.tomato_sharpa.tasks.profiles import PROFILES

SCHEMA = "sharpa_tomato_dataset_v1"
METADATA = ("info.json", "episodes.jsonl", "tasks.jsonl")
CAMERAS = tuple(f"observation.images.cam{i}" for i in range(3))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_dataset(root, profile):
    """Check the physical protocol and prompt vocabulary before publishing a release."""
    task = PROFILES[profile]
    root = Path(root).expanduser().resolve(strict=True)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
    tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line]
    features = info["features"]
    if (
        info.get("codebase_version") != "v2.1"
        or info.get("fps") != 30
        or features["observation.state"]["shape"] != [116]
        or features["observation.state"].get("names") != list(SOURCE_STATE_NAMES)
        or features["action"]["shape"] != [58]
        or features["action"].get("names") != list(SOURCE_ACTION_NAMES)
    ):
        raise ValueError("Expected named Sharpa state116/action58 at 30 Hz")
    cameras = tuple(key for key, value in features.items() if value.get("dtype") == "video")
    if set(cameras) not in (set(CAMERAS), {*CAMERAS, "observation.images.interaction"}):
        raise ValueError("Expected three RGB cameras, optionally with an interaction camera")
    ids = [row["episode_index"] for row in episodes]
    lengths = [row["length"] for row in episodes]
    if (
        not episodes
        or any(type(i) is not int or i < 0 for i in ids)
        or len(set(ids)) != len(ids)
        or any(type(n) is not int or n <= 0 for n in lengths)
        or info["total_episodes"] != len(episodes)
        or info["total_frames"] != sum(lengths)
    ):
        raise ValueError("Episode/frame totals disagree with the frozen metadata")
    prompts = [row["task"] for row in tasks]
    if (
        not prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts)
        or [row["task_index"] for row in tasks] != list(range(len(tasks)))
        or len(set(prompts)) != len(prompts)
        or (task.prompts and prompts != list(task.prompts))
    ):
        raise ValueError("Dataset task vocabulary differs from the selected task profile")
    mask = None
    if len(cameras) == 4:
        mask = next(
            (
                key
                for key in ("observation.interaction_available", "observation.interaction_valid")
                if features.get(key, {}).get("dtype") == "bool"
            ),
            None,
        )
        if mask is None:
            raise ValueError("Four-view input requires an explicit interaction availability/validity mask")
    return {
        "root": str(root),
        "episodes": len(episodes),
        "frames": sum(lengths),
        "prompts": prompts,
        "video_views": len(cameras),
        "interaction_mask_key": mask,
    }


def stage_files(root, profile):
    """Bind reviewed phase boundaries and the actual per-frame task supervision."""
    import numpy as np
    import pyarrow.parquet as pq

    from openpi.subtasks import tomato
    from openpi.subtasks import tomato_in_hand

    contract = tomato_in_hand if profile == "truss_in_hand" else tomato
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    tasks = [json.loads(line)["task"] for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line]
    files = []
    for line in (root / "meta/episodes.jsonl").read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        episode = row["episode_index"]
        annotation = root / f"annotations/episode_{episode:06d}.json"
        segments = contract.reviewed_segments(json.loads(annotation.read_text()), row["length"])
        parquet = root / info["data_path"].format(
            episode_chunk=episode // info.get("chunks_size", 1000), episode_index=episode
        )
        table = pq.read_table(parquet, columns=["task_index", "frame_index", "episode_index"])
        if len(table) != row["length"] or table["frame_index"].to_pylist() != list(range(row["length"])):
            raise ValueError("Prepared episode rows differ from reviewed boundaries")
        if set(table["episode_index"].to_pylist()) != {episode}:
            raise ValueError("Prepared episode identity changed")
        ids = table["task_index"].to_numpy()
        for segment in segments:
            prompt = contract.ACTION_PROMPTS[segment["label"]]
            if not np.all(ids[segment["start_frame"] : segment["end_frame"]] == tasks.index(prompt)):
                raise ValueError("Per-frame task_index differs from the reviewed phase prompt")
        files.extend((annotation.resolve(), parquet.resolve()))
    return files


def load_release(path, profile, expected_sha256=None):
    path = Path(path).expanduser().resolve(strict=True)
    sha = digest(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("Dataset release changed; create a new experiment")
    release = json.loads(path.read_text())
    if release.get("schema") != SCHEMA or release.get("task_profile") != profile:
        raise ValueError("Dataset release schema/task profile mismatch")
    for key in ("version", "asset_id", "repo_id"):
        value = release.get(key)
        if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"Invalid dataset release {key}")
    if release.get("action_semantics") not in ("absolute", "delta"):
        raise ValueError("Dataset release must declare action_semantics")
    files = release.get("files", {})
    root = Path(release["root"]).resolve()
    required = {str(root / "meta" / name) for name in METADATA}
    if not required <= files.keys() or not release.get("sources"):
        raise ValueError("Dataset release must bind all metadata and record its sources")
    if type(release.get("offline_roi_uses_future_annotation")) is not bool:
        raise ValueError("Dataset release must declare ROI provenance")
    provenance = release.get("provenance", [])
    if profile != "single_pick" and (not provenance or not set(provenance) <= files.keys()):
        raise ValueError("Stage datasets require bound preparation/annotation provenance")
    if profile != "single_pick":
        # Require the published supervision evidence as well as its hashes.
        bound = {str(path) for path in stage_files(root, profile)}
        if not bound <= files.keys():
            raise ValueError("Dataset release must bind reviewed annotations and frame supervision")
    for filename, expected in files.items():
        if digest(filename) != expected:
            raise ValueError(f"Dataset release file changed: {filename}")
    inspected = inspect_dataset(root, profile)
    for key, value in inspected.items():
        if release.get(key) != value:
            raise ValueError(f"Dataset release {key} differs from prepared data")
    return {**release, "manifest": str(path), "sha256": sha}
