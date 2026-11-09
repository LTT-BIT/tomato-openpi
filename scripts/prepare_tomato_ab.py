"""Prepare the frozen normal93 A/B experiment without an Agent or training."""

import argparse
import json
from pathlib import Path

from openpi.subtasks.prepare import digest
from openpi.subtasks.prepare import prepare
from openpi.subtasks.tomato import reviewed_segments

SELECTION = Path(__file__).resolve().parents[1] / (
    "training_configs/experiments/tomato_sharpa/subtasks/records/normal93_20261109_selection.json"
)


def build_plan(source_root, selection_path=SELECTION):
    source_root = Path(source_root).resolve(strict=True)
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text())
    rows = selection["episodes"]
    expected = {(r["dataset"], r["episode_index"]): r for r in rows}
    excluded = {(r["dataset"], r["episode_index"]) for r in selection["excluded"]}
    if len(expected) != len(rows) or expected.keys() & excluded:
        raise ValueError("Duplicate or overlapping selection/exclusion")
    found = {}
    for side in ("A", "B"):
        for root in sorted((source_root / side).iterdir()):
            metadata = root / "meta/episodes.jsonl"
            if not metadata.is_file():
                continue
            if not root.resolve().is_relative_to(source_root):
                raise ValueError("Dataset escapes selected source root")
            for line in metadata.read_text().splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                key = (str(root.relative_to(source_root)), entry["episode_index"])
                if key in found:
                    raise ValueError("Duplicate source episode metadata")
                found[key] = entry
    if expected.keys() - found.keys() or found.keys() - expected.keys() - excluded:
        raise ValueError("Source inventory differs from frozen selection; review before changing this experiment")
    episodes = []
    for row in rows:
        relative = Path(row["dataset"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] not in ("A", "B"):
            raise ValueError("Selection must contain only relative A/B dataset paths")
        entry = found[row["dataset"], row["episode_index"]]
        if any(entry.get(k) != row[k] for k in ("length", "source_recording_id")):
            raise ValueError("Source identity differs from frozen selection")
        root = source_root / relative
        annotation = root / f"annotations/subtasks/coarse/episode_{row['episode_index']:06d}.json"
        if digest(annotation) != row["annotation_sha256"]:
            raise ValueError("Reviewed annotation differs from frozen selection")
        reviewed_segments(json.loads(annotation.read_text()), row["length"])
        episodes.append(
            {
                "dataset": str(root),
                "episode_index": row["episode_index"],
                "annotation": str(annotation),
                "recording_namespace": "sharpa",
                "recording_group": row["source_recording_id"],
            }
        )
    return {"review_complete": True, "selection_sha256": digest(selection_path), "episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    args = parser.parse_args()
    import sentencepiece

    plan = build_plan(args.source_root)
    plan["tokenizer_sha256"] = digest(args.tokenizer)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer))
    print(prepare(plan, args.output, tokenizer))


if __name__ == "__main__":
    main()
