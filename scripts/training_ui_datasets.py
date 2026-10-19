# ruff: noqa: RUF001
"""Discover prepared datasets beneath selected folders without modifying them."""

import json
from pathlib import Path

from openpi.training.dataset_collection import collection_layout


def dataset_groups(roots):
    layout = collection_layout(roots)
    return {
        "episodes": sum(len(item["mapping"]) for item in layout),
        "frames": sum(item["info"]["total_frames"] for item in layout),
        "groups": [
            {
                "name": f"{i + 1}:{Path(item['root']).parent.name}/{Path(item['root']).name}",
                "source_root": item["root"],
                "episodes": list(item["mapping"].values()),
                "frames": item["info"]["total_frames"],
            }
            for i, item in enumerate(layout)
        ],
    }


def discover_datasets(roots):
    if not isinstance(roots, list) or not roots or any(not isinstance(p, str) or not p.strip() for p in roots):
        raise ValueError("请填写一个或多个服务器目录")
    found = {}
    warnings = []
    visited = set()

    def visit(path, depth):
        path = path.expanduser().resolve()
        if path in visited:
            return
        visited.add(path)
        if len(visited) > 2000:
            raise ValueError("扫描目录过多，请选择更具体的数据父目录")
        if not path.is_dir():
            raise ValueError(f"目录不存在或不可访问: {path}")
        metadata = path / "meta/info.json"
        if metadata.is_file():
            try:
                info = json.loads(metadata.read_text())
                required = [path / p for p in ("meta/episodes.jsonl", "meta/tasks.jsonl", "data", "videos")]
                if info.get("codebase_version") != "v2.1" or not all(p.exists() for p in required):
                    raise ValueError("需要完整的 LeRobot v2.1 meta/data/videos")
                alternative = path.name == "subtasks" and (path.parent / "full/meta/info.json").is_file()
                found[str(path)] = {
                    "root": str(path),
                    "episodes": info["total_episodes"],
                    "frames": info["total_frames"],
                    "fps": info["fps"],
                    "selected": not alternative,
                    "note": "与 full 可能重复，默认不选；如需阶段任务请明确选择" if alternative else "",
                }
            except (OSError, ValueError, KeyError) as error:
                warnings.append(f"{path}: {error}")
            return
        if depth >= 4:
            warnings.append(f"达到扫描深度限制: {path}")
            return
        for child in sorted(path.iterdir()):
            if (
                child.is_dir()
                and not child.is_symlink()
                and not child.name.startswith(".")
                and child.name not in {"data", "videos", "meta", "conversion"}
            ):
                visit(child, depth + 1)

    for root in roots:
        visit(Path(root), 0)
    if not found:
        raise ValueError("未发现完整 LeRobot v2.1 数据集；请选择包含 meta/data/videos 的目录。" + "；".join(warnings))
    return {"datasets": list(found.values()), "warnings": warnings}
