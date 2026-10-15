# ruff: noqa: RUF001
"""Recover episode groups from explicit conversion provenance, without reading videos."""

from collections import defaultdict
import json
from pathlib import Path
from pathlib import PurePosixPath


def detect_groups(dataset_root: str) -> dict:
    if not isinstance(dataset_root, str) or not dataset_root.strip():
        raise ValueError("请先填写数据集目录")
    root = Path(dataset_root).expanduser().resolve()
    manifest = root / "conversion/manifest.json"
    records = sorted((root / "conversion/records").glob("*.json"))
    if not manifest.is_file() or not records:
        raise ValueError("缺少 conversion/manifest.json 或 conversion/records 来源记录，请手动填写分组")
    source_roots = json.loads(manifest.read_text())["config"]["source_root"]
    if isinstance(source_roots, str):
        source_roots = [source_roots]
    if not isinstance(source_roots, list) or not source_roots:
        raise ValueError("转换记录没有有效的 source_root，请手动填写分组")
    # Historical paths are identifiers only; they need not exist on this machine.
    if any(not isinstance(p, str) or not p.startswith("/") or ".." in PurePosixPath(p).parts for p in source_roots):
        raise ValueError("转换来源必须是明确的绝对目录")
    sources = [PurePosixPath(p) for p in source_roots]
    if len(set(sources)) != len(sources) or any(a != b and a.is_relative_to(b) for a in sources for b in sources):
        raise ValueError("转换来源目录重复或相互包含，无法唯一识别分组")
    names = [p.name.removesuffix("-subtask") for p in sources]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("来源目录的组名不唯一，请手动填写分组")

    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError("自动分组仅支持本仓库使用的 LeRobot v2.1 数据")
    metadata = {}
    for line in (root / "meta/episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        episode, length = row["episode_index"], row["length"]
        if type(episode) is not int or episode < 0 or episode in metadata:
            raise ValueError("数据元信息包含无效或重复的 episode_index")
        if type(length) is not int or length <= 0:
            raise ValueError(f"episode {episode} 的长度无效")
        metadata[episode] = row
    if (
        not metadata
        or len(metadata) != info.get("total_episodes")
        or sum(row["length"] for row in metadata.values()) != info.get("total_frames")
    ):
        raise ValueError("info.json 与 episodes.jsonl 的 episode 数量或帧数不一致")

    groups = defaultdict(list)
    seen = set()
    for path in records:
        record = json.loads(path.read_text())
        source = record["source"]
        if not isinstance(source, str) or ".." in PurePosixPath(source).parts:
            raise ValueError(f"转换来源路径无效: {path.name}")
        matches = [i for i, parent in enumerate(sources) if PurePosixPath(source).is_relative_to(parent)]
        if len(matches) != 1:
            raise ValueError(f"转换来源不能唯一对应 source_root: {source}")
        for row in record["episodes"]:
            episode = row["episode_index"]
            if type(episode) is not int or episode not in metadata or episode in seen:
                raise ValueError(f"转换记录包含重复或未知的 episode: {episode}")
            for key in ("length", "source_start_ns", "source_end_ns"):
                if (key == "length" or key in metadata[episode]) and row.get(key) != metadata[episode].get(key):
                    raise ValueError(f"episode {episode} 的转换记录与元信息不一致: {key}")
            seen.add(episode)
            groups[matches[0]].append(episode)
    missing = set(metadata) - seen
    if missing:
        raise ValueError(f"转换记录未覆盖全部 episode，缺少 {len(missing)} 条，请手动填写分组")
    return {
        "dataset_root": str(root),
        "episodes": len(metadata),
        "frames": info["total_frames"],
        "groups": [
            {
                "name": names[i],
                "source_root": str(sources[i]),
                "episodes": sorted(episodes),
                "frames": sum(metadata[e]["length"] for e in episodes),
            }
            for i, episodes in sorted(groups.items())
        ],
    }
