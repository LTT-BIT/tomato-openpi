"""Shared finite smoke helpers. Training updates always call scripts.train.train_step."""

from contextlib import nullcontext
from dataclasses import asdict
from dataclasses import replace
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import normalize
from openpi.training import checkpoints
from openpi.training import config as cfg
from openpi.training import optimizer
from openpi.training import sharding
from openpi.training.mamba_training import MambaDataView
from openpi.training.mamba_training import architecture_record
from openpi.training.mamba_training import load_mamba_checkpoint
from openpi.training.mamba_training import model_data_config
from scripts import train
from scripts.mamba3_diagnostics import diagnostic_config
from scripts.mamba3_diagnostics import diagnostic_network


class SmokeLoader:
    def __init__(self, data):
        self.data = data

    def data_config(self):
        return self.data


def write_json(directory, name, value):
    (Path(directory) / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def group_name(path):
    name = "/".join(map(str, path))
    if name.startswith("PaliGemma/img"):
        return "SigLIP"
    if name.startswith("PaliGemma/llm"):
        return "Gemma"
    if name.startswith("context_compressor"):
        return "ContextCompressor"
    if "/mamba/" in name:
        return "MambaCore"
    if "/film_" in name:
        return "FiLM"
    if "/ffn_" in name:
        return "FFN"
    if "/time_" in name or "/cond_proj/" in name:
        return "TimeEmbedding/Projection"
    return "ActionInput/Output"


def inventory(params):
    """Count learned parameters only; fixed prior Variables never enter this inventory."""
    params = params.filter(nnx.Param)
    groups = {}
    for path, value in traverse_util.flatten_dict(params.to_pure_dict()).items():
        if value is None:
            continue
        group = groups.setdefault(
            group_name(path), {"parameters": 0, "bytes": 0, "leaves": 0, "dtypes": {}, "paths": []}
        )
        size = int(np.prod(value.shape))
        group["parameters"] += size
        group["bytes"] += size * np.dtype(value.dtype).itemsize
        group["leaves"] += 1
        group["dtypes"][str(value.dtype)] = group["dtypes"].get(str(value.dtype), 0) + size
        group["paths"].append("/".join(map(str, path)))
    return groups


def task_gradient(config, rng, state, batch):
    model = nnx.merge(state.model_def, state.params)
    model.train()
    key = jax.random.fold_in(rng, state.step)
    aug = (
        None
        if config.augmentation_seed is None
        else jax.random.fold_in(jax.random.key(config.augmentation_seed), state.step)
    )
    return nnx.value_and_grad(
        lambda m: m.compute_loss(key, *batch, train=True, augmentation_rng=aug).mean(),
        argnums=nnx.DiffState(0, config.trainable_filter),
    )(model)


def gradient_record(grads, before, after, step):
    flat_g = traverse_util.flatten_dict(grads.to_pure_dict())
    flat_b = traverse_util.flatten_dict(before.to_pure_dict())
    flat_a = traverse_util.flatten_dict(after.to_pure_dict())
    groups = {}
    for path, value in flat_g.items():
        if value is None:
            continue
        grad = np.asarray(value, dtype=np.float64)
        delta = np.asarray(flat_a[path], dtype=np.float64) - np.asarray(flat_b[path], dtype=np.float64)
        group = groups.setdefault(
            group_name(path),
            {"grad_squared": 0.0, "grad_max": 0.0, "nonfinite": 0, "nonzero_leaves": 0, "delta_squared": 0.0},
        )
        group["grad_squared"] += float((grad**2).sum())
        group["grad_max"] = max(group["grad_max"], float(np.abs(grad).max()))
        group["nonfinite"] += int((~np.isfinite(grad)).sum())
        group["nonzero_leaves"] += int(np.any(grad))
        group["delta_squared"] += float((delta**2).sum())
    for values in groups.values():
        values["grad_l2"] = values.pop("grad_squared") ** 0.5
        values["parameter_delta_l2"] = values.pop("delta_squared") ** 0.5
    return {"step": int(step), "groups": groups}


def state_digest(state):
    digest = hashlib.sha256()
    # GraphDef/tx are static pytree metadata. Actual params, moments, step/EMA are included.
    for value in jax.tree.leaves(state):
        array = np.asarray(value)
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def synthetic_batch(model_config):
    obs = model_config.fake_obs()
    obs = replace(
        obs,
        images={
            key: jax.random.uniform(jax.random.key(i + 11), value.shape, minval=-1, maxval=1)
            for i, (key, value) in enumerate(obs.images.items())
        },
        tokenized_prompt=jnp.arange(model_config.max_token_len)[None] + 3,
    )
    return obs, jax.random.normal(jax.random.key(7), (1, 50, 58)) * 0.1


def diagnostic_train_config(output, *, steps=10, seed=42):
    return cfg.TrainConfig(
        name="mamba3_phase_a_components",
        exp_name="diagnostic_only",
        model=diagnostic_config(),
        batch_size=1,
        num_workers=0,
        num_train_steps=steps,
        seed=seed,
        ema_decay=None,
        augmentation_seed=42002,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=0.001, decay_steps=100, decay_lr=0.0001),
        checkpoint_base_dir=str(Path(output) / "checkpoints"),
    )


def compile_train_step(config, state_sharding, mesh):
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    # Local locked JAX 0.5.3 CPU donation replay diverged with identical input
    # bytes and StableHLO. CPU diagnostics use the native step without buffer
    # donation; GPU diagnostics retain the production donation/sharding path.
    return jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=() if jax.default_backend() == "cpu" else (1,),
    )


def place_batch(batch, mesh):
    return jax.device_put(batch, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)))


def resume_step(config, state_sharding, mesh, state, batch, *, diagnostic):
    """GPU replays the production JIT; CPU tiny checks use an eager reference.

    The local locked CPU whole-program compiler has produced divergent update
    values for identical inputs/StableHLO. Compare checkpoint values using the
    same native train_step without whole-program fusion, never relaxed tolerances.
    """
    if diagnostic and jax.default_backend() == "cpu":
        with jax.disable_jit():
            return train.train_step(config, jax.random.key(config.seed), state, batch)
    return compile_train_step(config, state_sharding, mesh)(jax.random.key(config.seed), state, batch)


def host_copy(tree):
    # device_get on CPU can return views into donated device buffers. Retaining
    # these views across an optimizer update is unsafe even for read-only logs.
    return jax.tree.map(lambda value: np.array(value, copy=True), jax.device_get(tree))


def run_updates(config, batch, output, steps, *, diagnostic=False, checkpoint=True):
    with diagnostic_network() if diagnostic else nullcontext():
        return _run_updates(config, batch, output, steps, diagnostic=diagnostic, checkpoint=checkpoint)


def _run_updates(config, batch, output, steps, *, diagnostic=False, checkpoint=True):
    """Audit gradients separately, then time the native train_step.

    Logging snapshots own their host buffers before GPU training donates inputs.
    """
    directory = Path(output)
    mesh = sharding.make_mesh(config.fsdp_devices)
    batch = place_batch(batch, mesh)
    rng = jax.random.key(config.seed)
    started = time.perf_counter()
    with diagnostic_network() if diagnostic else nullcontext():
        state, state_sharding = train.init_train_state(
            config, jax.random.key(config.seed if config.model_seed is None else config.model_seed), mesh, resume=False
        )
    jax.block_until_ready(state)
    init_seconds = time.perf_counter() - started
    if not diagnostic:
        full_prefix_parity(
            nnx.merge(state.model_def, state.params), batch[0], config.weight_loader.params_path, directory
        )
    write_json(directory, "parameter_inventory.json", inventory(state.params))
    compiled = compile_train_step(config, state_sharding, mesh)
    compiled_grad = jax.jit(functools.partial(task_gradient, config, rng))
    metrics = []
    with (directory / "gradients.jsonl").open("w") as grad_stream:
        for i in range(steps):
            # Separately labelled audit; never confuse this with step wall time.
            _, grads = compiled_grad(state, batch)
            before = host_copy(state.params)
            grads = host_copy(grads)
            started = time.perf_counter()
            state, info = compiled(rng, state, batch)
            jax.block_until_ready(state)
            duration = time.perf_counter() - started
            numeric = {k: float(v) for k, v in info.items()}
            if not all(np.isfinite(list(numeric.values()))):
                raise FloatingPointError("Nonfinite training metric")
            metric = {"step": int(state.step), "seconds": duration, **numeric}
            metrics.append(metric)
            record = gradient_record(grads, before, jax.device_get(state.params), i)
            grad_stream.write(json.dumps(record) + "\n")
            grad_stream.flush()
            print(json.dumps(metric), flush=True)
            import wandb

            if wandb.run is not None:
                wandb.log(numeric, step=int(state.step))
    write_json(directory, "train_metrics.json", metrics)
    write_json(
        directory,
        "performance.json",
        {
            "diagnostic_only": diagnostic,
            "initialization_seconds": init_seconds,
            "first_train_step_seconds": metrics[0]["seconds"],
            "step_samples": len(metrics[3:]),
            "step_median_ms": float(np.median([m["seconds"] for m in metrics[3:]]) * 1000)
            if len(metrics) > 3
            else None,
            "step_p95_ms": float(np.percentile([m["seconds"] for m in metrics[3:]], 95) * 1000)
            if len(metrics) > 3
            else None,
            "donation_enabled": jax.default_backend() != "cpu",
            "measurement": "synchronized native train_step, after separate raw-gradient audit; CPU diagnostics disable donation; not a standalone throughput benchmark",
            "jax_memory_stats": jax.devices()[0].memory_stats(),
            "ema_tested": config.ema_decay is not None,
        },
    )
    records = [json.loads(line) for line in (directory / "gradients.jsonl").read_text().splitlines()]
    for name in (
        "SigLIP",
        "Gemma",
        "ContextCompressor",
        "MambaCore",
        "FiLM",
        "FFN",
        "TimeEmbedding/Projection",
        "ActionInput/Output",
    ):
        if not any(
            r["groups"][name]["grad_l2"] > 0
            and r["groups"][name]["parameter_delta_l2"] > 0
            and not r["groups"][name]["nonfinite"]
            for r in records[1:]
        ):
            raise AssertionError(f"No finite task-gradient/update after gate opening: {name}")
    if checkpoint:
        state = checkpoint_roundtrip(config, state, batch, directory, diagnostic)
    return state


def checkpoint_roundtrip(config, state, batch, output, diagnostic):
    if diagnostic and not hasattr(config.model, "paper_mode"):
        stat = normalize.NormStats(
            mean=np.zeros(58, np.float32),
            std=np.ones(58, np.float32),
            q01=-np.ones(58, np.float32),
            q99=np.ones(58, np.float32),
        )
        data = MambaDataView(
            repo_id="diagnostic_only",
            asset_id="diagnostic_absolute",
            norm_stats={"state": stat, "actions": stat},
            mamba_identity=architecture_record(config.model),
        )
    else:
        data = model_data_config(config)
    loader = SmokeLoader(data)
    root = output / "checkpoints"
    manager, _ = checkpoints.initialize_checkpoint_dir(root, keep_period=None, overwrite=False, resume=False)
    checkpoints.save_state(manager, state, loader, int(state.step))
    manager.wait_until_finished()
    manager.close()
    noise = jax.random.normal(jax.random.key(123), (1, 50, 58))
    mesh = sharding.make_mesh(config.fsdp_devices)
    state_sharding = sharding.fsdp_sharding(state, mesh)
    sample_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    prefix_length = min(10, config.model.rtc_max_delay)

    def sample(parameters):
        m = nnx.merge(state.model_def, parameters)
        m.eval()
        return m.sample_actions(
            jax.random.key(123),
            batch[0],
            noise=noise,
            num_steps=10,
            prefix_actions=batch[1] if prefix_length else None,
            prefix_length=prefix_length,
        )

    sample_jit = jax.jit(sample, in_shardings=(state_sharding.params,), out_shardings=sample_sharding)
    expected = np.asarray(sample_jit(state.params))
    np.testing.assert_array_equal(expected[:, :prefix_length], np.asarray(batch[1])[:, :prefix_length])
    np.save(output / "expected_sample.npy", expected)
    np.save(output / "sample_noise.npy", np.asarray(noise))
    save_batch(output / "batch.npz", batch)
    # Same replay execution mode and sharding in parent and child.
    before_digest = state_digest(state)
    mesh = sharding.make_mesh(config.fsdp_devices)
    state_sharding = sharding.fsdp_sharding(state, mesh)
    updated, update_info = resume_step(
        config, state_sharding, mesh, state, place_batch(batch, mesh), diagnostic=diagnostic
    )
    jax.block_until_ready(updated)
    write_json(
        output,
        "expected_resume.json",
        {
            "digest": state_digest(updated),
            "before_digest": before_digest,
            "batch_digest": state_digest(batch),
            "rng_digest": state_digest(jax.random.key_data(jax.random.key(config.seed))),
            "step": int(updated.step),
            "update_info": {k: float(v) for k, v in update_info.items()},
        },
    )
    from openpi.training.reproducibility import serialize_config

    (output / "effective_train_config.json").write_bytes(serialize_config(config))
    write_json(
        output,
        "restore_request.json",
        {
            "diagnostic_only": diagnostic,
            "step": int(updated.step) - 1,
            "asset_id": data.asset_id,
            "config_module": (config.policy_metadata or {}).get(
                "experiment_module", "training_configs.experiments.tomato_sharpa.mamba.train"
            ),
            "paper_model": asdict(config.model) if hasattr(config.model, "paper_mode") else None,
            "lr_schedule": asdict(config.lr_schedule),
            "seed": config.seed,
            "model_seed": config.model_seed,
            "steps": config.num_train_steps,
        },
    )
    command = [sys.executable, "-m", "scripts.smoke_mamba3_jax", "--restore-child", str(output)]
    with (output / "restore_child.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"New-process restore failed; see {output / 'restore_child.log'}")
    record = json.loads((output / "checkpoint_roundtrip.json").read_text())
    if record["sample_max_abs"] != 0 or not all(
        record[key] for key in ("resume_exact", "before_resume_exact", "batch_exact", "rng_exact", "prefix_exact")
    ):
        raise AssertionError(f"Restore/resume numerical mismatch: {record}")

    return updated


def save_batch(path, batch):
    obs, actions = batch
    values = traverse_util.flatten_dict(obs.to_dict(), sep="/")
    np.savez(path, actions=np.asarray(actions), **{k: np.asarray(v) for k, v in values.items() if v is not None})


def restore_child(output):
    request = json.loads((Path(output) / "restore_request.json").read_text())
    with diagnostic_network() if request["diagnostic_only"] else nullcontext():
        return _restore_child(output)


def _restore_child(output):
    directory = Path(output)
    request = json.loads((directory / "restore_request.json").read_text())
    diagnostic = request["diagnostic_only"]
    if (directory / "effective_train_config.json").exists():
        config = decode_smoke_config(json.loads((directory / "effective_train_config.json").read_text()))
        if hasattr(config.model, "paper_asset_dir") and not Path(config.model.paper_asset_dir).exists():
            asset_dir = directory / "checkpoints" / str(request["step"]) / "assets" / request["asset_id"]
            config = replace(
                config,
                model=replace(config.model, paper_asset_dir=str(asset_dir)),
                data=replace(
                    config.data,
                    assets=replace(config.data.assets, assets_dir=str(asset_dir.parent), asset_id=asset_dir.name),
                ),
            )
    else:
        if request.get("paper_model") is not None:
            from openpi.models.mamba3.config import Mamba3Config
            from openpi.models.pi0_mamba_paper import PaperMambaPi0Config

            values = dict(request["paper_model"])
            values["mamba"] = Mamba3Config(**values["mamba"])
            values["paper_asset_dir"] = str(
                directory / "checkpoints" / str(request["step"]) / "assets" / request["asset_id"]
            )
            model_config = PaperMambaPi0Config(**values)
            if diagnostic:
                config = replace(
                    diagnostic_train_config(directory, steps=request["steps"], seed=request["seed"]), model=model_config
                )
            else:
                import importlib

                source = importlib.import_module(request["config_module"]).build_config()
                asset_dir = Path(model_config.paper_asset_dir)
                source = replace(
                    source,
                    data=replace(
                        source.data,
                        assets=replace(source.data.assets, assets_dir=str(asset_dir.parent), asset_id=asset_dir.name),
                    ),
                    model=model_config,
                )
                config = replace(
                    source,
                    model=model_config,
                    seed=request["seed"],
                    model_seed=request["model_seed"],
                    num_train_steps=request["steps"],
                )
        elif diagnostic:
            config = diagnostic_train_config(directory, steps=request["steps"], seed=request["seed"])
        else:
            import importlib

            config = replace(
                importlib.import_module(request["config_module"]).build_config(),
                seed=request["seed"],
                model_seed=request["model_seed"],
                num_train_steps=request["steps"],
            )
        if "lr_schedule" in request:
            config = replace(config, lr_schedule=optimizer.CosineDecaySchedule(**request["lr_schedule"]))
    with np.load(directory / "batch.npz", allow_pickle=False) as values:
        batch = (
            model_lib.Observation.from_dict(
                jax.tree.map(
                    jnp.asarray, traverse_util.unflatten_dict({k: values[k] for k in values if k != "actions"}, sep="/")
                )
            ),
            jnp.asarray(values["actions"]),
        )
    with diagnostic_network() if diagnostic else nullcontext():
        mesh = sharding.make_mesh(config.fsdp_devices)
        shape, state_sharding = train.init_train_state(config, jax.random.key(config.seed), mesh, resume=True)
        batch = place_batch(batch, mesh)
        model = load_mamba_checkpoint(
            config.model, directory / "checkpoints" / str(request["step"]), request["asset_id"]
        )
    graph = shape.model_def
    params = jax.device_put(nnx.state(model), state_sharding.params)
    sample_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    noise = jnp.asarray(np.load(directory / "sample_noise.npy"))
    prefix_length = min(10, config.model.rtc_max_delay)

    def sample(parameters):
        m = nnx.merge(graph, parameters)
        m.eval()
        return m.sample_actions(
            jax.random.key(123),
            batch[0],
            noise=noise,
            num_steps=10,
            prefix_actions=batch[1] if prefix_length else None,
            prefix_length=prefix_length,
        )

    actual = np.asarray(jax.jit(sample, in_shardings=(state_sharding.params,), out_shardings=sample_sharding)(params))
    delta = float(np.max(np.abs(actual - np.load(directory / "expected_sample.npy"))))
    manager, _ = checkpoints.initialize_checkpoint_dir(
        directory / "checkpoints", keep_period=None, overwrite=False, resume=True
    )
    restored = checkpoints.restore_state(manager, shape, None, step=request["step"])
    manager.close()
    before_digest = state_digest(restored)
    updated, update_info = resume_step(config, state_sharding, mesh, restored, batch, diagnostic=diagnostic)
    jax.block_until_ready(updated)
    expected = json.loads((directory / "expected_resume.json").read_text())
    write_json(
        directory,
        "checkpoint_roundtrip.json",
        {
            "diagnostic_only": diagnostic,
            "new_process": True,
            "sample_max_abs": delta,
            "prefix_length": prefix_length,
            "prefix_exact": np.array_equal(actual[:, :prefix_length], np.asarray(batch[1])[:, :prefix_length]),
            "resume_execution": "cpu_eager_reference"
            if diagnostic and jax.default_backend() == "cpu"
            else "jitted_native_trainer",
            "resume_exact": state_digest(updated) == expected["digest"],
            "before_resume_exact": before_digest == expected.get("before_digest", before_digest),
            "batch_exact": state_digest(batch) == expected.get("batch_digest", state_digest(batch)),
            "rng_exact": state_digest(jax.random.key_data(jax.random.key(config.seed)))
            == expected.get("rng_digest", state_digest(jax.random.key_data(jax.random.key(config.seed)))),
            "update_info": {k: float(v) for k, v in update_info.items()},
            "expected_update_info": expected.get("update_info"),
            "restored_step": request["step"],
            "continued_step": int(updated.step),
            "batch_replay": "fixed saved batch; real iterator-position continuation is a separate check",
        },
    )
    return 0


def full_prefix_parity(mamba_model, observation, init_checkpoint, output):
    """Compare real initialized VLM to unmodified Pi0 Prefix-only evaluation."""
    from openpi.models.pi0 import make_attn_mask
    from openpi.models.pi0_config import Pi0Config

    loaded = model_lib.restore_params(init_checkpoint, restore_type=np.ndarray)
    action_dim = loaded["action_out_proj"]["kernel"].shape[-1]
    reference_config = Pi0Config(
        pi05=True,
        action_dim=action_dim,
        max_token_len=mamba_model.max_token_len,
        dtype=mamba_model.config.dtype,
        paligemma_variant=mamba_model.config.paligemma_variant,
    )
    reference = reference_config.load(loaded)
    tokens, mask, ar = reference.embed_prefix(observation)
    (expected, _), _ = reference.PaliGemma.llm(
        [tokens, None], mask=make_attn_mask(mask, ar), positions=jnp.cumsum(mask, axis=1) - 1
    )
    actual, _, _ = mamba_model.prefix_hidden(observation)
    a, b = (
        np.asarray(actual, dtype=np.float32)[np.asarray(mask)],
        np.asarray(expected, dtype=np.float32)[np.asarray(mask)],
    )
    maximum = float(np.max(np.abs(a - b)))
    relative = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-8))
    threshold = 1e-4 if mamba_model.config.dtype == "float32" else 1e-2
    write_json(
        output,
        "prefix_parity.json",
        {
            "diagnostic_only": False,
            "max_abs": maximum,
            "relative_l2": relative,
            "threshold": threshold,
            "valid_tokens": int(mask.sum()),
            "status": "PASS" if relative <= threshold else "FAIL",
        },
    )
    if not np.isfinite(a).all() or not np.isfinite(b).all() or relative > threshold:
        raise AssertionError("Pretrained VLM Prefix-only parity failed")


def decode_smoke_config(value):
    """Decode our own finite replay snapshot using an explicit dataclass allowlist."""
    from openpi import transforms
    from openpi.models.mamba3.config import Mamba3Config
    from openpi.models.pi0_mamba_config import MambaPi0Config
    from openpi.models.pi0_mamba_paper import PaperMambaPi0Config
    from openpi.training.delta_paper import PaperSharpaData
    from openpi.training.mamba_training import MambaScheduledTrainConfig
    from openpi.training.mamba_training import MambaSharpaData
    from openpi.training.mamba_training import MambaTrainConfig
    from openpi.training.mamba_weight_loader import MambaVlmInitLoader
    from openpi.training.mamba_weight_loader import PaperMambaVlmInitLoader
    from openpi.training.weight_loaders import NoOpWeightLoader

    classes = (
        cfg.TrainConfig,
        cfg.ScheduledTrainConfig,
        MambaTrainConfig,
        MambaScheduledTrainConfig,
        cfg.DataConfig,
        cfg.VideoCacheConfig,
        cfg.FakeDataConfig,
        cfg.AssetsConfig,
        cfg.LeRobotDatasetSpec,
        PaperSharpaData,
        MambaSharpaData,
        MambaPi0Config,
        PaperMambaPi0Config,
        Mamba3Config,
        optimizer.AdamW,
        optimizer.CosineDecaySchedule,
        optimizer.RsqrtDecaySchedule,
        NoOpWeightLoader,
        MambaVlmInitLoader,
        PaperMambaVlmInitLoader,
        transforms.Group,
    )
    allowed = {f"{item.__module__}.{item.__qualname__}": item for item in classes}
    tuple_fields = {
        "sampling_groups",
        "lerobot_datasets",
        "action_sequence_keys",
        "save_steps_before",
        "inputs",
        "outputs",
        "local_dataset_video_keys",
        "datasets",
    }

    def decode(item):
        if isinstance(item, list):
            return [decode(x) for x in item]
        if isinstance(item, dict):
            if item == {"type": f"{nnx.Nothing.__module__}.{nnx.Nothing.__qualname__}", "state": {}}:
                return nnx.Nothing()
            if set(item) == {"type", "fields"}:
                if item["type"] not in allowed:
                    raise ValueError(f"Unsupported smoke snapshot type: {item['type']}")
                values = {key: decode(val) for key, val in item["fields"].items()}
                for key in tuple_fields & values.keys():
                    values[key] = tuple(values[key])
                return allowed[item["type"]](**values)
            return {key: decode(val) for key, val in item.items()}
        if item == f"{nnx.Nothing.__module__}.{nnx.Nothing.__qualname__}":
            return nnx.Nothing
        return item

    return decode(value)
