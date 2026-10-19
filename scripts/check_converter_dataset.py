"""Read and normalize a real converter dataset, then fetch one batch; never train."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.fr3_wuji import compute_norm_stats
from openpi import transforms
from openpi.training import data_loader
from scripts.run_training import load_experiment
from scripts.run_training import preflight


def check(robot, root, assets_root):
    root = root.resolve(strict=True)
    if robot == "sharpa":
        module = "training_configs.experiments.tomato_sharpa.multimodal_rgb.tomato_fr3_sharpa_lora_v1"
    else:
        module = "training_configs.experiments.tomato_wuji.baseline.tomato_fr3_lora_v1"
    config = load_experiment(module)
    config = replace(
        config,
        data=replace(
            config.data,
            assets=replace(config.data.assets, assets_dir=str(assets_root.resolve())),
            lerobot_datasets=(replace(config.data.lerobot_datasets[0], root=str(root)),),
        ),
        policy_metadata={**config.policy_metadata, "dataset_root": str(root), "source_datasets": [str(root)]},
    )
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "meta").iterdir() if p.is_file()}
    basic = preflight(config, require_norm=False)
    compute_norm_stats.compute(config, batch_size=128)
    verified = preflight(config)
    data_config = config.data.create(config.assets_dirs, config.model)
    raw = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    vector_transform = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    checked = 0
    offset = 0
    for ep in episodes:
        for i in sorted({offset, offset + ep["length"] // 2, offset + ep["length"] - 1}):
            sample = raw[i]
            selected = vector_transform(sample)
            state = np.asarray(sample["observation.state"])
            actions = np.asarray(sample["action"])
            action_names = info["features"]["action"]["names"]
            canonical = config.policy_metadata.get("joint_names")
            if canonical is None:
                canonical = [name.rsplit(".", 1)[0] for name in (
                    action_names[:7] + action_names[14:34] + action_names[7:14] + action_names[34:54]
                )]
            expected_state = [info["features"]["observation.state"]["names"].index(f"{name}.position") for name in canonical]
            lookup = {name.rsplit(".", 1)[0]: n for n, name in enumerate(action_names)}
            np.testing.assert_array_equal(selected["state"], state[expected_state])
            np.testing.assert_array_equal(selected["actions"], actions[:, [lookup[name] for name in canonical]])
            if set(selected) != {"image", "image_mask", "state", "actions", "prompt"}:
                raise ValueError("Unexpected model-bound modality")
            checked += 1
        offset += ep["length"]
    loader = data_loader.create_torch_data_loader(
        data_config, config.model, config.model.action_horizon, 1,
        num_batches=1, num_workers=0, framework="jax", seed=config.data_seed,
        sampling_mode=config.sampling_mode, num_train_steps=config.num_train_steps,
    )
    observation, actions = next(iter(loader))
    if not np.isfinite(np.asarray(observation.state)).all() or not np.isfinite(np.asarray(actions)).all():
        raise ValueError("Nonfinite normalized batch")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / "meta").iterdir() if p.is_file()}
    if before != after:
        raise ValueError("Input dataset metadata changed")
    return {
        "status": "passed", "robot": robot, "dataset": str(root), "experiment_module": module,
        "dataset_version": config.policy_metadata["dataset_version"], "exp_name": config.exp_name,
        "prompts": basic["prompts"],
        "frames": basic["frames"], "episodes": basic["episodes"], "fps": info["fps"],
        "normalization_provenance": verified["norm_provenance"],
        "loader_framework": "jax", "sampling_mode": config.sampling_mode,
        "norm_stats_sha256": hashlib.sha256(Path(verified["norm_stats_path"]).read_bytes()).hexdigest(),
        "norm_metadata_sha256": hashlib.sha256(Path(verified["norm_stats_path"]).with_name("norm_metadata.json").read_bytes()).hexdigest(),
        "raw_state_shape": info["features"]["observation.state"]["shape"],
        "model_state_shape": list(observation.state.shape), "model_action_shape": list(actions.shape),
        "model_images": {key: list(value.shape) for key, value in observation.images.items()},
        "tokenized_prompt_shape": list(observation.tokenized_prompt.shape),
        "depth_features_stored": [k for k in info["features"] if k.startswith("observation.depths.")],
        "tactile_features_stored": [k for k in info["features"] if k.startswith("observation.tactile.")],
        "boundary_middle_samples_checked": checked, "input_metadata_sha256": before,
        "training_started": False, "model_weights_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", required=True, choices=("sharpa", "wuji"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--norm-assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new report path")
    result = check(args.robot, args.dataset, args.norm_assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
