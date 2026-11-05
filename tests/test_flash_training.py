"""Formal FLASH contracts: causal local data, source split, FSDP update and resume."""

from dataclasses import fields
from dataclasses import replace
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.models.pi0_config import Pi0Config
from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.shared.image_tools import resize_with_pad
from openpi.training import checkpoints
from openpi.training import sharding
from openpi.training import weight_loaders
from openpi.training.flash_native import NativeDataset
from openpi.training.flash_native import bind_stats
from openpi.training.flash_native import preflight
from openpi.training.flash_native import prepare
from scripts import training_ui_config as editor
from scripts.train import init_train_state
from scripts.train_flash import evaluate_batch
from scripts.train_flash import step
from training_configs.experiments.tomato_sharpa.flash.tomato_fr3_sharpa_flash_fullft_v1 import build_config


def shrink_network(monkeypatch):
    from openpi.models import gemma
    from openpi.models import siglip

    original = siglip.Module
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    monkeypatch.setattr(
        gemma,
        "get_config",
        lambda variant: gemma.Config(width=32, depth=1, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16),
    )
    monkeypatch.setattr(
        siglip,
        "Module",
        lambda num_classes, **kwargs: original(
            num_classes,
            width=32,
            depth=1,
            mlp_dim=64,
            num_heads=2,
            patch_size=(112, 112),
            pool_type="none",
            scan=False,
            dtype_mm="float32",
        ),
    )


@pytest.fixture
def prepared(tmp_path):
    config = build_config()
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "videos").mkdir()
    info = {
        "fps": 30,
        "codebase_version": "v2.1",
        "chunks_size": 1000,
        "data_path": "data/episode_{episode_index:06d}.parquet",
        "total_episodes": 6,
        "total_frames": 480,
        "features": {
            "action": {"shape": [58], "names": [x + ".command" for x in ORDER]},
            "observation.state": {
                "shape": [116],
                "names": [x + ".position" for x in ORDER] + [x + ".velocity" for x in ORDER],
            },
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "Pick a tomato."}))
    episodes = []
    for i in range(6):
        frames = np.arange(80)
        actions = frames[:, None] * 0.001 + np.arange(58)[None, :] * 0.01 + i * 0.1
        pq.write_table(
            pa.table(
                {
                    "action": actions.tolist(),
                    "observation.state": np.concatenate([actions - 0.005, actions * 0], axis=1).tolist(),
                    "frame_index": frames,
                    "timestamp": frames / 30,
                    "task_index": frames * 0,
                }
            ),
            root / f"data/episode_{i:06d}.parquet",
        )
        episodes.append({"episode_index": i, "length": 80, "source_recording_id": f"recording_{i // 2}"})
    (root / "meta/episodes.jsonl").write_text("\n".join(map(json.dumps, episodes)))
    config = replace(
        config,
        policy_metadata={**config.policy_metadata, "data_policy": "source_holdout"},
        batch_size=1,
        fsdp_devices=1,
        num_workers=0,
        data=replace(
            config.data,
            lerobot_datasets=(replace(config.data.lerobot_datasets[0], root=str(root)),),
            assets=replace(config.data.assets, assets_dir=str(tmp_path / "assets"), asset_id="new"),
        ),
    )
    prepare(config)
    return bind_stats(config)


def test_causal_data_split_sampler_and_fingerprints(prepared):
    report = preflight(prepared)
    assert report["episodes"] == 6
    d = NativeDataset(prepared)
    other = NativeDataset(prepared)
    assert not set(d.manifest["train_sources"]) & set(d.manifest["validation_sources"])
    for position in (0, 1, 21, 22, 87, 88, 101, 10000):
        assert d.index(position) == other.index(position)
    window = d.index(0)
    history, target, current, _ = d.window(window)
    assert history.shape == (8, 58)
    assert target.shape == (54, 58)
    np.testing.assert_array_equal(history[-1], current)
    ident, start = window["id"], window["start"]
    actions, states, _ = d.numeric[ident]
    original = history.copy()
    actions[start + 1 :] += 100
    states[start + 1 :] -= 100
    np.testing.assert_array_equal(original, d.window(window)[0])
    record = d.records[ident]
    with open(record["path"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        preflight(prepared)


@pytest.mark.parametrize("reference", [False, True])
def test_fullft_update_checkpoint_and_rejected_nonfinite(prepared, monkeypatch, tmp_path, reference):
    shrink_network(monkeypatch)
    config = replace(
        prepared,
        model=replace(prepared.model, dtype="float32", max_token_len=8),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        lr_schedule=replace(prepared.lr_schedule, warmup_steps=0, peak_lr=2.5e-4, decay_steps=100, decay_lr=1e-5),
    )
    if reference:
        config = bind_stats(
            replace(
                config,
                model=Pi0Config(**{f.name: getattr(config.model, f.name) for f in fields(Pi0Config)}),
                policy_metadata={**config.policy_metadata, "algorithm": "flash_reference"},
            )
        )
    mesh = sharding.make_mesh(1)
    state, _ = init_train_state(config, jax.random.key(42), mesh, resume=False)
    obs = config.model.fake_obs()
    obs = replace(
        obs,
        images={k: jnp.full_like(v, 0.3) for k, v in obs.images.items()},
        tokenized_prompt=jnp.ones_like(obs.tokenized_prompt),
    )
    d = NativeDataset(config)
    history, target, _, _ = d.window(d.index(0))
    batch = (obs, jnp.asarray(history[None]), jnp.asarray(target[None]))
    update = jax.jit(lambda s, b: step(config, jax.random.key(42), s, b))
    initial = {p: np.array(v.value) for p, v in state.params.flat_state().items()}
    for _ in range(2):
        state, metrics = jax.block_until_ready(update(state, batch))
        assert bool(metrics["finite"])
    changed = {p for p, v in state.params.flat_state().items() if not np.array_equal(initial[p], np.asarray(v.value))}
    assert any(p[:2] == ("PaliGemma", "img") for p in changed)
    assert any(p[:2] == ("PaliGemma", "llm") for p in changed)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "checkpoint", keep_period=None, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(manager, state, d, 2)
        manager.wait_until_finished()
        assert list((tmp_path / "checkpoint").rglob("flash_manifest.json"))
        template, _ = init_train_state(config, jax.random.key(42), mesh, resume=True)
        restored = checkpoints.restore_state(manager, template, d)
        left, _ = update(state, batch)
        right, _ = update(restored, batch)
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
            np.testing.assert_array_equal(a, b)
        bad, metrics = update(state, (obs, batch[1], batch[2] * jnp.nan))
        assert not bool(metrics["finite"])
        for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(bad), strict=True):
            np.testing.assert_array_equal(a, b)
        result = evaluate_batch(
            state.model_def,
            state.params,
            batch,
            jax.random.key(0),
            reference=reference,
            q01=tuple(d.stats["actions"].q01),
            q99=tuple(d.stats["actions"].q99),
        )
        assert result["prediction"].shape == (1, 50, 58)
        assert np.isfinite(np.asarray(result["prediction"])).all()
    finally:
        manager.close()


def test_ui_flash_enforces_standard_method():
    module = "training_configs.experiments.tomato_sharpa.flash.tomato_fr3_sharpa_flash_fullft_v1"
    config = editor.baseline_config(module)
    values = editor.defaults(module)["values"]
    assert not config.model.freeze_vlm
    assert editor.defaults(module)["protocol"]["flash"]
    for delta in ({"action_semantics": "delta"}, {"rtc_max_delay": 10}, {"paper_trial": "t3", "asset_id": "different"}):
        with pytest.raises(ValueError, match="Absolute|不支持"):
            editor.apply_overrides(config, {**values, **delta})


@pytest.mark.parametrize("shape", [(80, 160, 3), (480, 640, 3)])
def test_numpy_camera_batch_matches_jax_reference_without_backend_allocations(shape):
    rgb = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
    actual = NativeDataset.resize_image(rgb)
    expected = np.asarray(resize_with_pad(rgb.astype(np.float32) / 127.5 - 1, 224, 224))[None]
    assert isinstance(actual, np.ndarray)
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=1e-5)


def test_all_sources_train_and_monitor_without_holdout(prepared, tmp_path):
    config = replace(
        prepared,
        policy_metadata={**prepared.policy_metadata, "data_policy": "all_sources"},
        data=replace(prepared.data, assets=replace(prepared.data.assets, asset_id="all_sources")),
    )
    report = prepare(config)
    assert report["train_sources"] == 3
    assert report["validation_sources"] == 0
    assert report["evaluation_scope"] == "training_data"
    d = NativeDataset(bind_stats(config))
    assert len(d.train) == 6
    assert int(d.ends[-1]) == 6 * (80 - 58)
    assert len({d.index(i)["id"] for i in range(int(d.ends[-1]))}) == 6
    assert {w["id"] for w in d.validation} <= set(d.records)
    # Norm statistics include all six episodes, rather than the old holdout split.
    actions = np.concatenate([d.window({"id": r["id"], "start": 7})[1] for r in d.train])
    assert np.isfinite(actions).all()
    expected = np.concatenate([np.asarray(pq.read_table(r["path"])["action"].to_pylist()) for r in d.train])
    np.testing.assert_allclose(d.stats["actions"].mean, expected.mean(axis=0), atol=1e-6)
    with pytest.raises(ValueError, match="policy changed"):
        preflight(replace(config, policy_metadata={**config.policy_metadata, "data_policy": "source_holdout"}))
    with pytest.raises(FileExistsError):
        prepare(config)


def test_prefetch_real_video_order_resume_and_cpu_workers(prepared, tmp_path):
    import av

    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training.flash_loader import PrefetchedBatches
    from openpi.training.flash_loader import WorkerDataset

    # Same numeric fixture, now with real videos for all three camera roles.
    root = prepared.data.lerobot_datasets[0].root
    info_path = Path(root) / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = "videos/{video_key}/episode_{episode_index:06d}.mp4"
    info_path.write_text(json.dumps(info))
    for episode in range(6):
        for camera in ("cam0", "cam1", "cam2"):
            path = Path(root) / f"videos/observation.images.{camera}/episode_{episode:06d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=30)
                stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
                for i in range(80):
                    frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 2, np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
    config = replace(
        prepared,
        batch_size=2,
        num_workers=2,
        data=replace(prepared.data, assets=replace(prepared.data.assets, asset_id="videos")),
    )
    prepare(config)
    dataset = NativeDataset(config, tokenizer=PaligemmaTokenizer(max_len=config.model.max_token_len))
    originals = [dataset.batch([dataset.index(completed * 2 + i) for i in range(2)]) for completed in range(3, 6)]
    with PrefetchedBatches(dataset, 3, 6) as batches:
        for expected, actual in zip(originals, batches, strict=True):
            for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
                np.testing.assert_array_equal(a, b)
    # Discard an active prefetch queue after only one accepted batch.
    with PrefetchedBatches(dataset, 3, 9) as batches:
        for a, b in zip(jax.tree.leaves(originals[0]), jax.tree.leaves(next(batches)), strict=True):
            np.testing.assert_array_equal(a, b)
    # Restart at the last completed update; read-ahead must never skip a sample.
    with PrefetchedBatches(dataset, 4, 5) as batches:
        for a, b in zip(jax.tree.leaves(originals[1]), jax.tree.leaves(next(batches)), strict=True):
            np.testing.assert_array_equal(a, b)
    cached = WorkerDataset(dataset, decoder_limit=2, frame_limit=2)
    try:
        for start in (7, 20, 9, 35, 7):
            index = {"id": dataset.train[0]["id"], "start": start}
            actual = cached.resized_image(index, "base_0_rgb")
            expected = dataset.resize_image(dataset.image(index, "base_0_rgb"))
            np.testing.assert_array_equal(actual, expected)
        assert len(cached.decoders) <= 2
        assert len(cached.frames) <= 2
    finally:
        cached.close()


def test_prefetch_migration_preserves_math_and_rejects_contract_changes():
    import subprocess

    from openpi.training.flash_loader import verify_input_migration
    from openpi.training.flash_native import digest

    root = Path(__file__).resolve().parents[1]
    names = (
        "scripts/train_flash.py",
        "src/openpi/training/flash_native.py",
        "src/openpi/models/pi0_flash.py",
        "src/openpi/shared/flash_polynomial.py",
        "src/openpi/training/optimizer.py",
        "src/openpi/models/pi0.py",
        "src/openpi/models/gemma.py",
        "src/openpi/models/siglip.py",
        "scripts/train.py",
    )
    before = {name: subprocess.check_output(["git", "show", "e64c9a41:" + name], cwd=root, text=True) for name in names}
    saved = {"batch_size": 256, "training_code_sha256": digest(before)}
    current = {**saved, "training_code_sha256": "new-input-only-version"}
    assert verify_input_migration(saved, current, root)["kind"] == "cpu_prefetch_only"
    with pytest.raises(ValueError, match="RNG/batch"):
        verify_input_migration(saved, {**current, "batch_size": 512}, root)
    with pytest.raises(ValueError, match="audited"):
        verify_input_migration({**saved, "training_code_sha256": "unrecognized"}, current, root)
