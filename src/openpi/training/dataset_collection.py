"""Metadata-only layout for ordered local dataset collections.

For collections, global episode IDs are dense in source order then sorted local
episode order. Original files and IDs are never rewritten.
"""

import json
from pathlib import Path


def collection_layout(roots):
    paths = [Path(root).expanduser().resolve() for root in roots]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Dataset directories must be nonempty and unique (including symlink targets)")
    result = []
    offset = 0
    for root in paths:
        info = json.loads((root / "meta/info.json").read_text())
        episodes = [
            json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()
        ]
        ids = [row["episode_index"] for row in episodes]
        if not episodes or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid or duplicate episode IDs: {root}")
        if any(type(row["length"]) is not int or row["length"] <= 0 for row in episodes):
            raise ValueError(f"Invalid episode lengths: {root}")
        if len(episodes) != info["total_episodes"] or sum(row["length"] for row in episodes) != info["total_frames"]:
            raise ValueError(f"Episode metadata totals disagree: {root}")
        local_ids = sorted(ids)
        global_ids = list(range(offset, offset + len(ids))) if len(paths) > 1 else local_ids
        result.append(
            {
                "root": str(root),
                "info": info,
                "episodes": episodes,
                "mapping": dict(zip(local_ids, global_ids, strict=True)),
            }
        )
        offset += len(ids)
    return result


def validate_collection_features(infos):
    reference = infos[0]
    keys = {"observation.state", "action", *(k for k in reference["features"] if k.startswith("observation.images."))}
    for info in infos[1:]:
        cameras = {k for k in info["features"] if k.startswith("observation.images.")}
        if info["fps"] != reference["fps"] or cameras != keys - {"observation.state", "action"}:
            raise ValueError("Collection datasets must have matching FPS and camera roles")
        for key in keys:
            a, b = reference["features"].get(key, {}), info["features"].get(key, {})
            if any(a.get(field) != b.get(field) for field in ("shape", "dtype", "names")):
                raise ValueError(f"Collection feature dimensions/dtype/order differ: {key}")
