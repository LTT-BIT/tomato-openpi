"""FullFT FLASH training using native causal data and the shared FSDP/checkpoint stack."""

from dataclasses import replace
import functools
import json
from pathlib import Path
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from openpi.models import model as model_lib
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import checkpoints
from openpi.training import sharding
from openpi.training import tracking
from openpi.training.flash_native import NativeDataset
from openpi.training.flash_native import bind_stats
from openpi.training.flash_native import preflight
from scripts.train import init_train_state


def step(config, rng, state, batch):
    observation, history, target = batch
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def objective(m):
        if config.policy_metadata.get("algorithm") == "flash_reference":
            from openpi.shared.flash_polynomial import normalize_actions

            actions = normalize_actions(
                target[:, 2:52],
                jnp.asarray(config.policy_metadata["action_q01"]),
                jnp.asarray(config.policy_metadata["action_q99"]),
            )
            fm = jnp.mean(
                m.compute_loss(
                    jax.random.fold_in(rng, state.step),
                    observation,
                    actions,
                    train=True,
                    augmentation_rng=jax.random.fold_in(jax.random.key(config.augmentation_seed), state.step),
                )
            )
            return fm, {"loss": fm, "flow_matching_loss": fm, "polynomial_consistency_loss": jnp.asarray(0.0)}
        augmented = model_lib.preprocess_observation(
            jax.random.fold_in(jax.random.key(config.augmentation_seed), state.step), observation, train=True
        )
        mask, cache = m.encode_condition(augmented)
        values = m.loss_from_cache(
            jax.random.fold_in(rng, state.step), mask, cache, history=history, target_ext=target, train=True
        )
        values = {k: jnp.mean(v) for k, v in values.items()}
        return values["loss"], values

    (_, info), grads = nnx.value_and_grad(objective, argnums=nnx.DiffState(0, config.trainable_filter), has_aux=True)(
        model
    )
    info["gradient_norm"] = optax.global_norm(grads)
    from openpi.training.flash_checkpoint import is_vlm_parameter

    info["vision_gradient_norm"] = optax.global_norm(grads.filter(lambda p, _: p[:2] == ("PaliGemma", "img")))
    info["language_gradient_norm"] = optax.global_norm(
        grads.filter(lambda p, _: p[:2] == ("PaliGemma", "llm") and is_vlm_parameter(p))
    )
    info["expert_gradient_norm"] = optax.global_norm(grads.filter(lambda p, _: not is_vlm_parameter(p)))
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads)]))
    finite &= jnp.all(jnp.stack([jnp.isfinite(v) for v in info.values()]))

    def advance(old):
        params = old.params.filter(config.trainable_filter)
        updates, adam = old.tx.update(grads, old.opt_state, params)
        nnx.update(model, optax.apply_updates(params, updates))
        return replace(old, step=old.step + 1, params=nnx.state(model), opt_state=adam)

    # A failed update returns the original parameters, moments and completed step.
    updated = jax.lax.cond(finite, advance, lambda old: old, state)
    info["finite"] = finite
    return updated, info


def evaluate_batch(model_def, params, batch, rng, *, reference=False, q01=None, q99=None):
    obs, history, target = batch
    model = nnx.merge(model_def, params)
    model.eval()
    if reference:
        from openpi.shared.flash_polynomial import FlashPolynomial
        from openpi.shared.flash_polynomial import normalize_actions
        from openpi.shared.flash_polynomial import unnormalize_actions

        low, high = jnp.asarray(q01), jnp.asarray(q99)
        values = {
            "loss": jnp.mean(
                model.compute_loss(rng, obs, normalize_actions(target[:, 2:52], low, high), train=False), axis=-1
            )
        }
        prediction = unnormalize_actions(model.sample_actions(rng, obs), low, high)
        polynomial = FlashPolynomial()
        oracle = unnormalize_actions(
            polynomial.reconstruct(polynomial.fit_target(normalize_actions(target, low, high))), low, high
        )
    else:
        values = model.loss_components(rng, obs, history=history, target_ext=target, train=False)
        prediction = model.sample_actions(rng, obs, history=history)
        oracle = model.denormalize(
            model.polynomial.reconstruct(
                model.polynomial.fit_target(model.normalize(target), boundary_constraint=model.boundary_constraint)
            )
        )
    truth = target[:, 2:52]
    return {**values, "prediction": prediction, "truth": truth, "oracle": oracle}


def validation(config, state, dataset, data_sharding, evaluator):
    errors, oracles, coefficients = [], [], []
    for offset in range(0, len(dataset.validation), config.batch_size):
        indices = dataset.validation[offset : offset + config.batch_size]
        valid = len(indices)
        indices += [indices[-1]] * (config.batch_size - valid)
        batch = jax.device_put(dataset.batch(indices), data_sharding)
        values = jax.device_get(evaluator(state.params, batch, jax.random.key(config.seed + 101)))
        errors.append(values["prediction"][:valid] - values["truth"][:valid])
        oracles.append(values["oracle"][:valid] - values["truth"][:valid])
        coefficients.extend(values["loss"][:valid].tolist())
    error, oracle = np.concatenate(errors), np.concatenate(oracles)
    arms = np.r_[0:7, 29:36]
    hands = np.r_[7:29, 36:58]
    result = {
        "normalized_action_flow_loss"
        if config.policy_metadata.get("algorithm") == "flash_reference"
        else "coefficient_loss": float(np.mean(coefficients)),
        "windows": len(error),
        "evaluation_scope": dataset.manifest["contract"].get("evaluation_scope", "held_out_sources"),
    }
    for name, indices in (("all", np.arange(58)), ("arms", arms), ("hands", hands)):
        for method, delta in (("prediction", error), ("oracle", oracle)):
            selected = delta[..., indices]
            result[f"{method}_{name}_mae"] = float(np.abs(selected).mean())
            result[f"{method}_{name}_rmse"] = float(np.sqrt(np.square(selected).mean()))
    result["per_joint_p95"] = np.quantile(np.abs(error), 0.95, axis=(0, 1)).tolist()
    result["per_joint_p99"] = np.quantile(np.abs(error), 0.99, axis=(0, 1)).tolist()
    result["per_joint_max"] = np.abs(error).max(axis=(0, 1)).tolist()
    return result


def main(config):
    config = bind_stats(config)
    inputs = preflight(config)
    if config.batch_size % jax.device_count():
        raise ValueError("Global batch must be divisible by selected GPU count")
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    dataset = NativeDataset(config, tokenizer=PaligemmaTokenizer(max_len=config.model.max_token_len))
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=config.resume
    )
    try:
        state, layout = init_train_state(config, jax.random.key(config.model_seed), mesh, resume=resuming)
        contract = {
            "dataset": inputs,
            "model_seed": config.model_seed,
            "data_seed": config.data_seed,
            "flow_seed": config.seed,
            "augmentation_seed": config.augmentation_seed,
            "batch_size": config.batch_size,
            "optimizer": str(config.optimizer),
            "schedule": str(config.lr_schedule),
            "model": str(config.model),
            "rng_policy": "fold_in(seed, completed_update); independent augmentation_seed",
            "sampler": "epoch_shuffle; cursor=completed_update*global_batch_size",
        }
        from openpi.training.flash_native import digest

        source_files = (
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
        root = Path(__file__).resolve().parents[1]
        contract["training_code_sha256"] = digest({name: (root / name).read_text() for name in source_files})
        seal_path = Path(config.checkpoint_dir) / "flash_training_contract.json"
        if resuming:
            if json.loads(seal_path.read_text()) != contract:
                raise ValueError("FLASH resume data/model/optimizer/RNG/batch contract changed")
            state = checkpoints.restore_state(manager, state, dataset)
        else:
            seal_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False))
        start = int(state.step)
        update = jax.jit(
            functools.partial(step, config),
            in_shardings=(replicated, layout, data_sharding),
            out_shardings=(layout, replicated),
            donate_argnums=(1,),
        )
        evaluator = jax.jit(
            functools.partial(
                evaluate_batch,
                state.model_def,
                reference=config.policy_metadata.get("algorithm") == "flash_reference",
                q01=tuple(dataset.stats["actions"].q01),
                q99=tuple(dataset.stats["actions"].q99),
            ),
            in_shardings=(layout.params, data_sharding, replicated),
        )
        output = Path(config.checkpoint_dir) / "flash_metrics"
        output.mkdir(exist_ok=True)
        with tracking.session(config, resuming=resuming, start_step=start):
            for completed in range(start, config.num_train_steps):
                # This cursor also determines the next window exactly after checkpoint restore.
                indices = [dataset.index(completed * config.batch_size + i) for i in range(config.batch_size)]
                batch = jax.device_put(dataset.batch(indices), data_sharding)
                before = time.perf_counter()
                with sharding.set_mesh(mesh):
                    candidate, metrics = update(jax.random.key(config.seed), state, batch)
                metrics = jax.device_get(metrics)
                if not bool(metrics["finite"]):
                    raise FloatingPointError("FLASH nonfinite loss/gradient; optimizer update rejected")
                state = candidate
                row = {k: float(v) for k, v in metrics.items() if k != "finite"}
                row.update(
                    step=int(state.step),
                    seconds=time.perf_counter() - before,
                    learning_rate=float(config.lr_schedule.create()(completed)),
                    effective_samples=int(state.step) * config.batch_size,
                )
                memory = jax.devices()[0].memory_stats() or {}
                row["peak_device_bytes"] = memory.get("peak_bytes_in_use")
                with (output / "training.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                wandb.log({"train/" + k: v for k, v in row.items() if v is not None}, step=int(state.step))
                if int(state.step) % config.log_interval == 0 or completed == start:
                    print(json.dumps(row), flush=True)
                final = int(state.step) == config.num_train_steps
                if int(state.step) % config.save_interval == 0 or final:
                    # Persist the accepted update before evaluation can fail or exhaust memory.
                    checkpoints.save_state(manager, state, dataset, int(state.step))
                    manager.wait_until_finished()
                    report = validation(config, state, dataset, data_sharding, evaluator)
                    scope = "training_evaluation" if inputs["evaluation_scope"] == "training_data" else "validation"
                    (output / f"{scope}_{int(state.step)}.json").write_text(json.dumps(report, indent=2))
                    wandb.log(
                        {scope + "/" + k: v for k, v in report.items() if isinstance(v, int | float)},
                        step=int(state.step),
                    )
        manager.wait_until_finished()
    finally:
        manager.close()
