"""Four-view paper statistics, phase prompts and RTC in a CPU training step."""

from dataclasses import replace
import json
from types import SimpleNamespace

import datasets
from flax import nnx
import jax
import numpy as np
import optax
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi import transforms
from openpi.models import siglip
from openpi.shared import delta_paper as paper
from openpi.shared import normalize
from openpi.subtasks.tomato import GOAL
from openpi.subtasks.tomato_in_hand import TASKS
from openpi.training import config as cfg
from openpi.training import data_loader
from openpi.training import frame_cache
from openpi.training import sampling
from openpi.training import utils
from openpi.training.current_rgb import fixed_global_prompt
from openpi.training.delta_paper_stats import compute
from scripts import run_training
from scripts import training_ui_config as ui
from scripts.train import train_step

PREFIX = "training_configs.experiments.tomato_sharpa.subtasks."
SHARPA_BASE = "training_configs.experiments.tomato_sharpa.baseline.tomato_fr3_sharpa_fullft_v1"


@pytest.mark.parametrize("source_count", [1, 3])
def test_sharpa_baseline_t3_rtc_and_weighted_sampling_accept_selected_paths(tmp_path, source_count):
    defaults = ui.defaults(SHARPA_BASE)
    assert defaults["protocol"]["paper_available"]
    assert defaults["values"]["action_semantics"] == "absolute"
    roots = [str(tmp_path / f"arbitrary location {i}" / "recording") for i in range(source_count)]
    values = {
        **defaults["values"],
        "dataset_root": roots[0],
        "dataset_roots": roots,
        "sources": roots,
        "asset_id": "arbitrary_paths_t3",
        "action_semantics": "delta",
        "paper_trial": "t3",
        "rtc_max_delay": 10,
        "sampling_mode": "weighted_replacement",
        "sampling_groups": [{"name": "selected", "episodes": list(range(source_count)), "weight": 1}],
    }
    config = ui.config_from_source(ui.personal_source(SHARPA_BASE, "test", values), SHARPA_BASE, "saved.py")
    assert [spec.root for spec in config.data.lerobot_datasets] == roots
    assert config.model.paper_mode == "timestamp_zscore_arms"
    assert config.model.paper_beta == 0.5
    assert config.model.rtc_max_delay == 10
    assert config.sampling_mode == "weighted_replacement"
    assert config.sampling_groups[0].episodes == tuple(range(source_count))


@pytest.mark.parametrize("prompt_mode", ["stage", "global"])
@pytest.mark.parametrize("video_views", [3, 4])
def test_training_input_options_survive_paper_and_personal_configs(prompt_mode, video_views, tmp_path, monkeypatch):
    monkeypatch.setenv("PI05_IN_HAND204_BUNDLE", str(tmp_path / "missing"))
    monkeypatch.setenv("PI05_VISUAL_MEMORY_FRAMES", "4")
    module = ui.DEFAULT_AB_MODULE
    source = ui.source_config(module)
    defaults = ui.defaults(module)
    assert defaults["values"]["prompt_mode"] == "stage"
    assert defaults["values"]["video_views"] == 4
    values = {**defaults["values"], "prompt_mode": prompt_mode, "video_views": video_views, "rtc_max_delay": 8}
    if (prompt_mode, video_views) != ("stage", 4):
        with pytest.raises(ValueError, match="asset_id"):
            ui.apply_overrides(source, values)
    values.update(asset_id="smoke_config", paper_trial="t3")
    saved = ui.personal_source(module, "test", values)
    config = ui.config_from_source(saved, module, "saved.py")
    assert len(config.model.image_keys) == len(config.policy_metadata["camera_keys"]) == video_views
    assert not hasattr(config.model, "memory")
    assert fixed_global_prompt(config.data) == (GOAL if prompt_mode == "global" else None)
    assert config.model.rtc_max_delay == 8
    assert config.model.paper_beta == 0.5
    reverted = ui.apply_overrides(
        config,
        {
            **ui.config_values(config),
            "prompt_mode": "stage",
            "video_views": 4,
            "paper_trial": "none",
            "asset_id": "smoke_reverted",
        },
    )
    assert reverted.name == source.name
    assert reverted.model.image_keys == source.model.image_keys
    assert fixed_global_prompt(reverted.data) is None


@pytest.mark.parametrize("recipe", ["in_hand204_b_crop", "in_hand204_current_rgb"])
def test_paper_ui_preserves_recipe_and_checks_independent_stats(recipe, tmp_path, monkeypatch):
    monkeypatch.setenv("PI05_IN_HAND204_BUNDLE", str(tmp_path / "missing"))
    defaults = ui.defaults(PREFIX + recipe)
    source = ui.source_config(PREFIX + recipe)
    assert defaults["protocol"]["bundle_missing"]
    assert defaults["protocol"]["paper_available"]
    assert "rtc_max_delay" not in defaults["protocol"]["locked_fields"]
    assert {"sampling_mode", "sampling_groups"}.isdisjoint(defaults["protocol"]["locked_fields"])
    for trial, parameters in {
        "t1": ("timestamp_zscore_arms", 0),
        "t2": ("global_quantile", 0.5),
        "t3": ("timestamp_zscore_arms", 0.5),
    }.items():
        values = {
            **defaults["values"],
            "paper_trial": trial,
            "rtc_max_delay": 8,
            "sampling_mode": "weighted_replacement",
            "sampling_groups": [
                {"name": "A", "episodes": [0], "weight": 7},
                {"name": "B", "episodes": [1], "weight": 3},
            ],
        }
        with pytest.raises(ValueError, match="asset_id"):
            ui.apply_overrides(source, values)
        values["asset_id"] += "_" + trial
        config = ui.apply_overrides(source, values)
        assert (config.model.paper_mode, config.model.paper_beta) == parameters
        assert config.model.image_keys == source.model.image_keys
        assert config.model.rtc_max_delay == 8
        assert config.policy_metadata["action_prompts"] == source.policy_metadata["action_prompts"]
        assert config.data.lerobot_datasets == source.data.lerobot_datasets
        text = ui.personal_source(PREFIX + recipe, "test", values)
        restored = ui.config_from_source(text, PREFIX + recipe, "test.py")
        assert restored.model == config.model
        assert restored.data == config.data
        assert restored.sampling_groups == config.sampling_groups
        with pytest.raises(FileNotFoundError, match="full204"):
            run_training.preflight(config, require_norm=False)
    with pytest.raises(ValueError, match="rtc_max_delay"):
        ui.apply_overrides(source, {**defaults["values"], "rtc_max_delay": source.model.action_horizon})
    run_record = tmp_path / "run.json"
    run_record.write_text(
        json.dumps(
            {
                "module": PREFIX + recipe,
                "config_sha256": defaults["source_sha256"],
                "values": values,
            }
        )
    )
    resolved = run_training.resolve_experiment(PREFIX + recipe, ["--ui-run", str(run_record)])
    assert resolved.model.rtc_max_delay == 8
    assert resolved.model.paper_beta == 0.5
    assert resolved.sampling_mode == "weighted_replacement"


def test_current_rgb_t3_statistics_loading_and_jitted_rtc_update(tmp_path, monkeypatch):
    monkeypatch.delenv("PI05_RGB_FRAME_CACHE", raising=False)
    monkeypatch.setenv("PI05_IN_HAND204_BUNDLE", str(tmp_path / "missing"))
    monkeypatch.setenv("PI05_VISUAL_MEMORY_FRAMES", "2")
    monkeypatch.setenv("PI05_VISUAL_MEMORY_STRIDE", "1")
    module = ui.DEFAULT_AB_MODULE
    source = ui.source_config(module)
    root = tmp_path / "data"
    for name in ("meta", "data"):
        (root / name).mkdir(parents=True)
    (root / "videos").mkdir()
    cameras = source.policy_metadata["camera_keys"]
    features = {
        "action": {"shape": [58], "names": source.policy_metadata["source_action_joint_names"]},
        "observation.state": {"shape": [116], "names": source.policy_metadata["source_state_joint_names"]},
        **{key: {"shape": [16, 16, 3]} for key in cameras},
    }
    lengths = (5, 4, 2)
    episodes = [{"episode_index": index, "length": length} for index, length in enumerate(lengths)]
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "features": features,
        "total_frames": sum(lengths),
        "total_episodes": len(episodes),
        "chunks_size": 1000,
        "data_path": "data/episode_{episode_index:06d}.parquet",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    (root / "meta/tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": index, "task": task}) + "\n" for index, task in enumerate(TASKS))
    )
    samples = []
    ids = {"episode_index": [], "frame_index": []}
    for episode, length in enumerate(lengths):
        states = np.tile(np.arange(length, dtype=np.float32)[:, None] / 10, (1, 116))
        actions = np.tile(np.arange(length, dtype=np.float32)[:, None] / 5, (1, 58))
        actions += episode
        ids["episode_index"].extend([episode] * length)
        ids["frame_index"].extend(range(length))
        pq.write_table(
            pa.table(
                {
                    "observation.state": states.tolist(),
                    "action": actions.tolist(),
                    "frame_index": list(range(length)),
                    "episode_index": [episode] * length,
                }
            ),
            root / f"data/episode_{episode:06d}.parquet",
        )
        samples.extend(
            {
                "observation.state": states[frame],
                "action": actions[np.minimum(frame + np.arange(3), length - 1)],
                "task_index": min(frame, 3),
                "observation.interaction_available": np.bool_(frame != 0),
                **{key: np.full((16, 16, 3), 10 + frame, np.uint8) for key in cameras},
            }
            for frame in range(length)
        )
    metadata = SimpleNamespace(fps=30, tasks=dict(enumerate(TASKS)))
    dataset = SimpleNamespace(root=root, meta=metadata, hf_dataset=datasets.Dataset.from_dict(ids))

    class SyntheticSource:
        root, meta, hf_dataset = dataset.root, dataset.meta, dataset.hf_dataset

        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index]

    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *a, **kw: metadata)
    monkeypatch.setattr(frame_cache, "CachedLeRobotDataset", lambda *a, **kw: SyntheticSource())

    def tokenize(values):
        return {
            **{key: value for key, value in values.items() if key != "prompt"},
            "tokenized_prompt": np.ones(4, np.int32),
            "tokenized_prompt_mask": np.ones(4, bool),
        }

    monkeypatch.setattr(cfg.ModelTransformFactory, "__call__", lambda *args: transforms.Group(inputs=[tokenize]))
    source = replace(
        source,
        batch_size=1,
        fsdp_devices=1,
        num_workers=0,
        model=replace(
            source.model,
            action_horizon=3,
            max_token_len=4,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            rtc_max_delay=2,
            dtype="float32",
        ),
        data=replace(source.data, lerobot_datasets=(replace(source.data.lerobot_datasets[0], root=str(root)),)),
        policy_metadata={**source.policy_metadata, "bundle_sha256": "synthetic"},
    )
    pending = replace(source, data=replace(source.data, assets=cfg.AssetsConfig(str(tmp_path / "assets"), "t3")))
    groups = (sampling.SamplingGroup("A", (0,), 7), sampling.SamplingGroup("B", (1,), 3))
    weighted = replace(pending, sampling_mode="weighted_replacement", sampling_groups=groups)
    config = ui.select_trial(source, weighted, "t3")
    compute(config, batch_size=2)
    config = ui.select_trial(source, weighted, "t3")
    report = run_training.preflight(config)
    assert report["training_windows"] == 5
    assert report["norm_provenance"] == "verified"
    assert report["sampling"]["eligible_windows"] == 5
    assert [row["eligible_windows"] for row in report["sampling"]["groups"]] == [3, 2]
    np.testing.assert_allclose([row["probability"] for row in report["sampling"]["groups"]], [0.7, 0.3])
    with pytest.raises(ValueError, match=r"absent from data=\[2\]"):
        run_training.preflight(replace(config, sampling_groups=(groups[0], replace(groups[1], episodes=(1, 2)))))
    assert (
        run_training.preflight(ui.select_trial(source, pending, "t3"))["paper_bundle_sha256"]
        == report["paper_bundle_sha256"]
    )
    data = config.data.create(config.assets_dirs, config.model)
    assert not hasattr(data, "visual_memory")
    assert data.video_views == 4
    assert data.norm_stats["actions"].mean.shape == (3, 58)
    assert np.all(np.diff(data.norm_stats["actions"].mean[:, 0]) > 0)
    raw = data_loader.create_torch_dataset(data, 3, config.model, sampling_mode="epoch_shuffle")
    assert [raw[index]["prompt"] for index in range(len(raw))] == [*TASKS[:3], *TASKS[:2]]
    loader = data_loader.create_data_loader(config, num_batches=1)
    observation, actions = next(iter(loader))
    assert set(observation.images) == set(source.model.image_keys)
    assert actions.shape == (1, 3, 58)

    original = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda **kw: original(**{**kw, "variant": "mu/56"}))
    model = config.model.create(jax.random.key(0))
    noise_key = jax.random.key(1)
    white = jax.random.normal(noise_key, actions.shape)
    correlated = model.noise(noise_key, actions.shape)
    expected = (white.reshape(1, -1) @ model.paper_noise_cholesky.value.T).reshape(actions.shape)
    np.testing.assert_allclose(correlated, expected)
    assert not np.array_equal(correlated, white)
    definition, params = nnx.split(model)
    tx = optax.adamw(1e-3)
    training = utils.TrainState(
        step=0,
        params=params,
        model_def=definition,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        tx=tx,
        ema_decay=None,
    )
    updated, metrics = jax.jit(lambda state: train_step(config, jax.random.key(2), state, (observation, actions)))(
        training
    )
    assert int(updated.step) == 1
    assert all(np.isfinite(value) for value in metrics.values())
    np.testing.assert_array_equal(updated.params["paper_noise_cholesky"].value, params["paper_noise_cholesky"].value)
    destination = tmp_path / "checkpoint"
    normalize.save(destination, data.norm_stats)
    data.save_extra_assets(destination)
    assert (destination / paper.BUNDLE).is_file()
