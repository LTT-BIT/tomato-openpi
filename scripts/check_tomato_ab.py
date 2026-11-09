"""Verify paired physical trajectories, phase boundaries, splits and a real model batch; never train."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.subtasks.prepare import digest
from openpi.subtasks.tomato import GOAL
from openpi.subtasks.tomato import PROMPTS
from openpi.training import data_loader
from scripts.run_training import load_experiment
from scripts.run_training import preflight

MODULE = "training_configs.experiments.tomato_sharpa.subtasks.normal93_"


def check(*, require_norm=False):
    configs = {mode: load_experiment(MODULE + name) for mode, name in (("a_goal", "a_goal"), ("b_coarse", "b_stage"))}
    bundle = Path(configs["a_goal"].policy_metadata["reviewed_bundle"]).parent
    manifest = json.loads((bundle / "bundle.json").read_text())
    splits = json.loads((bundle / "splits.json").read_text())["splits"]
    frames = 0
    for row in manifest["mapping"]:
        index = row["episode_index"]
        original_root = Path(row["source_dataset"])
        original_info = json.loads((original_root / "meta/info.json").read_text())
        original = pq.read_table(
            original_root
            / original_info["data_path"].format(
                episode_chunk=row["source_episode_index"] // 1000, episode_index=row["source_episode_index"]
            )
        )
        for mode in configs:
            root = bundle / mode
            actual = pq.read_table(root / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet")
            if len(actual) != len(original) or set(actual.column_names) != set(original.column_names) | {
                "coarse_phase_index"
            }:
                raise ValueError("Physical trajectory length/columns changed")
            for key in original.column_names:
                if key not in ("episode_index", "index", "task_index", "coarse_phase_index") and not original[
                    key
                ].equals(actual[key]):
                    raise ValueError(f"Physical column changed: {key}")
            annotation = json.loads((root / f"annotations/episode_{index:06d}.json").read_text())
            for relative, expected in annotation["source"]["sha256"].items():
                if digest(root / relative) != expected:
                    raise ValueError("Prepared data changed after publication")
            phase_ids = actual["coarse_phase_index"].to_numpy()
            task_ids = actual["task_index"].to_numpy()
            texts = [GOAL] if mode == "a_goal" else list(dict.fromkeys(PROMPTS))
            for phase, segment in enumerate(annotation["segments"]):
                region = slice(segment["start_frame"], segment["end_frame"])
                if not np.all(phase_ids[region] == phase) or not np.all(
                    task_ids[region] == (0 if mode == "a_goal" else texts.index(PROMPTS[phase]))
                ):
                    raise ValueError("Per-frame phase/prompt differs from reviewed annotation")
        frames += len(original)
    result = {
        "episodes": len(manifest["mapping"]),
        "frames": frames,
        "splits": {},
        "variants": {},
        "training_started": False,
        "model_weights_loaded": False,
    }
    for mode, config in configs.items():
        basic = preflight(config, require_norm=require_norm)
        data = config.data.create(config.assets_dirs, config.model)
        for split, expected in splits.items():
            view = replace(data.task_view, split=split)
            source = data_loader.create_torch_dataset(
                replace(data, task_view=view),
                config.model.action_horizon,
                config.model,
                sampling_mode=config.sampling_mode,
            )
            if set(source.ids[:, 0]) != set(expected):
                raise ValueError("Loader selected wrong split")
            result["splits"][split] = len(expected)
            # Both sides of all six phase boundaries, plus the true episode end.
            for episode, (_, actions, _, _) in source.records.items():
                label = json.loads((bundle / mode / f"annotations/episode_{episode:06d}.json").read_text())
                starts = np.flatnonzero(source.ids[:, 0] == episode)
                first = int(starts[0])
                for frame in {
                    0,
                    len(actions) - 1,
                    *(f for s in label["segments"][1:] for f in (s["start_frame"] - 1, s["start_frame"])),
                }:
                    actual = source.vectors(first + frame)
                    expected_window = actions[np.minimum(frame + np.arange(50), len(actions) - 1)]
                    np.testing.assert_array_equal(actual["action"], expected_window)
                    phase = next(
                        i for i, s in enumerate(label["segments"]) if s["start_frame"] <= frame < s["end_frame"]
                    )
                    if actual["prompt"] != (GOAL if mode == "a_goal" else PROMPTS[phase]):
                        raise ValueError("Loader prompt differs at phase boundary")
            del source
        result["variants"][mode] = {"preflight": basic}
        if require_norm:
            loader = data_loader.create_torch_data_loader(
                data,
                config.model,
                50,
                1,
                num_batches=1,
                num_workers=0,
                framework="jax",
                seed=config.data_seed,
                sampling_mode=config.sampling_mode,
                num_train_steps=config.num_train_steps,
            )
            observation, actions = next(iter(loader))
            for value in (observation.state, actions, *observation.images.values()):
                if not np.isfinite(np.asarray(value)).all():
                    raise ValueError("Nonfinite model input")
            if (
                actions.shape != (1, 50, 58)
                or observation.state.shape != (1, 58)
                or observation.tokenized_prompt.shape != (1, 256)
            ):
                raise ValueError("Unexpected normalized model batch shape")
            result["variants"][mode]["batch"] = {
                "state": list(observation.state.shape),
                "actions": list(actions.shape),
                "tokens": list(observation.tokenized_prompt.shape),
                "images": {k: list(v.shape) for k, v in observation.images.items()},
            }
    result["status"] = "passed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-norm", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new verification report")
    result = check(require_norm=args.require_norm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ("status", "episodes", "frames", "splits")}))


if __name__ == "__main__":
    main()
