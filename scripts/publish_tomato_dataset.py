"""Publish a new immutable metadata release for an already prepared Sharpa dataset.

Does not annotate, copy data, compute normalization, or start training.
"""

import argparse
import json
from pathlib import Path

from training_configs.experiments.tomato_sharpa.tasks.dataset import METADATA
from training_configs.experiments.tomato_sharpa.tasks.dataset import SCHEMA
from training_configs.experiments.tomato_sharpa.tasks.dataset import digest
from training_configs.experiments.tomato_sharpa.tasks.dataset import inspect_dataset
from training_configs.experiments.tomato_sharpa.tasks.dataset import load_release
from training_configs.experiments.tomato_sharpa.tasks.dataset import stage_files
from training_configs.experiments.tomato_sharpa.tasks.profiles import PROFILES


def publish(
    *,
    root,
    profile,
    version,
    asset_id,
    action_semantics,
    sources,
    provenance,
    output,
    offline_roi_uses_future_annotation=False,
):
    inspected = inspect_dataset(root, profile)
    paths = [Path(inspected["root"]) / "meta" / name for name in METADATA]
    provenance = [str(Path(p).expanduser().resolve(strict=True)) for p in provenance]
    for filename in provenance:
        evidence = json.loads(Path(filename).read_text())
        uses_future = (
            evidence.get("offline_roi_uses_future_coordinates") is True
            or evidence.get("stream_contract") == "offline_bidirectional_local_rgb_v6"
        )
        if uses_future and not offline_roi_uses_future_annotation:
            raise ValueError("Preparation uses future ROI coordinates; declare offline ROI provenance")
    paths.extend(map(Path, provenance))
    if profile != "single_pick":
        paths.extend(stage_files(inspected["root"], profile))
    release = {
        "schema": SCHEMA,
        "task_profile": profile,
        "version": version,
        "asset_id": asset_id,
        "repo_id": f"fr3_sharpa/{version}",
        "action_semantics": action_semantics,
        "sources": list(sources),
        "provenance": provenance,
        "files": {str(p): digest(p) for p in paths},
        "offline_roi_uses_future_annotation": offline_roi_uses_future_annotation,
        **inspected,
    }
    # Exclusive publication; a version file is never overwritten.
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Publish to a new dataset version path")
    # Validate before publishing, using an exclusive temporary sibling.
    import tempfile

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, suffix=".json") as probe:
        json.dump(release, probe, ensure_ascii=False, indent=2)
        probe.flush()
        load_release(probe.name, profile)
        with output.open("x") as stream:
            stream.write(Path(probe.name).read_text() + "\n")
    return load_release(output, profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--action-semantics", choices=("absolute", "delta"), required=True)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--provenance", type=Path, action="append", default=[])
    parser.add_argument("--offline-roi-uses-future-annotation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = publish(
        root=args.dataset_root,
        profile=args.profile,
        version=args.version,
        asset_id=args.asset_id,
        action_semantics=args.action_semantics,
        sources=args.source,
        provenance=args.provenance,
        output=args.output,
        offline_roi_uses_future_annotation=args.offline_roi_uses_future_annotation,
    )
    print(json.dumps({key: result[key] for key in ("manifest", "sha256", "episodes", "frames")}, indent=2))


if __name__ == "__main__":
    main()
