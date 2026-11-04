"""Finite local-only Phase A verification. Exit 0=selected suite pass, 1=fail, 2=blocked.

Components use a clearly labelled synthetic tiny VLM, H50/D58, and the existing
train_step. FullFT/all never substitute a tiny model for missing real inputs.
"""

import argparse
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback


def environment():
    import jax
    import jax.numpy as jnp

    result = {
        "python": sys.version,
        "executable": sys.executable,
        "packages": {
            p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint", "numpy")
        },
        "devices": [str(d) for d in jax.devices()],
        "lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest(),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    result["matrix_probe"] = float((jnp.ones((128, 128)) @ jnp.ones((128, 128))).sum())
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    result["nvidia_smi"] = query.stdout.strip()
    return result


def run_pytest(output, paths, name, *, cpu=False):
    from scripts.mamba3_smoke_runtime import write_json

    command = [sys.executable, "-m", "pytest", *paths, "-q", "--disable-warnings", f"--junitxml={output / name}.xml"]
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"}
    env["MAMBA_SMOKE_EVIDENCE"] = str(output)
    if cpu:
        env["JAX_PLATFORMS"] = "cpu"
    with (output / f"{name}.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    write_json(output, f"{name}.json", {"command": command, "returncode": result.returncode, "cpu": cpu})
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {output / f'{name}.log'}")


def full_preflight(args, output):
    from dataclasses import replace

    from flax import nnx
    import jax

    from openpi.training.mamba_training import MambaTrainConfig
    from openpi.training.mamba_weight_loader import MambaVlmInitLoader
    from scripts import run_training
    from scripts.mamba3_smoke_runtime import inventory
    from scripts.mamba3_smoke_runtime import write_json
    from training_configs.experiments.tomato_sharpa.mamba.phase_a import CONFIG

    issues = []
    for name in ("init_checkpoint", "dataset", "norm_stats"):
        value = getattr(args, name)
        if not value or not Path(value).exists():
            issues.append(f"missing {name}: supply an existing local path")
    if not any(d.platform == "gpu" for d in jax.devices()):
        issues.append("no JAX GPU device; real FullFT suite requires GPU")
    # Shape-only production audit never materializes billions of parameters.
    abstract = nnx.eval_shape(CONFIG.model.create, jax.random.key(args.seed))
    groups = inventory(nnx.state(abstract, nnx.Param))
    write_json(output, "production_parameter_inventory.json", groups)
    param_bytes = sum(v["bytes"] for v in groups.values())
    # These are a lower bound, NOT measured peak memory: params + Adam m/v.
    persistent_min = param_bytes * 3
    limits = [d.memory_stats().get("bytes_limit", 0) for d in jax.devices() if d.platform == "gpu"]
    write_json(
        output,
        "memory_preflight.json",
        {
            "parameters_bytes": param_bytes,
            "parameters_and_adam_minimum_bytes": persistent_min,
            "device_allocator_limits": limits,
            "measurement": "shape-derived lower bound, excludes gradients/activations/workspace; no OOM trial claimed",
        },
    )
    if limits and sum(limits) < persistent_min:
        issues.append(
            "FULL_E2E_BLOCKED_BY_MEMORY_PREFLIGHT: parameters + FP32 Adam moments exceed available device budget"
        )
    if issues:
        return None, issues
    norm = Path(args.norm_stats).resolve()
    if norm.name != "norm_stats.json":
        return None, ["norm-stats must point to a regular norm_stats.json in its independent Absolute asset directory"]
    root = Path(args.dataset).resolve()
    metadata = {
        **CONFIG.policy_metadata,
        "dataset_root": str(root),
        "source_datasets": [str(root)],
        "experiment_module": "training_configs.experiments.tomato_sharpa.mamba.phase_a",
    }
    config = replace(
        CONFIG,
        seed=args.seed,
        model_seed=args.seed,
        num_train_steps=args.steps,
        checkpoint_base_dir=str(output / "training_checkpoints"),
        weight_loader=MambaVlmInitLoader(
            str(Path(args.init_checkpoint).resolve()), report_path=str(output / "weight_load_report.json")
        ),
        data=replace(
            CONFIG.data,
            assets=replace(CONFIG.data.assets, assets_dir=str(norm.parent.parent), asset_id=norm.parent.name),
            lerobot_datasets=(replace(CONFIG.data.lerobot_datasets[0], root=str(root)),),
        ),
        policy_metadata=metadata,
    )
    assert isinstance(config, MambaTrainConfig)
    inputs = run_training.preflight(config)
    write_json(output, "input_preflight.json", inputs)
    return config, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("components", "fullft", "checkpoint", "policy", "regression", "all"), default="components"
    )
    parser.add_argument("--config")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--dataset")
    parser.add_argument("--norm-stats")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--restore-child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.restore_child:
        from scripts.mamba3_smoke_runtime import restore_child

        return restore_child(args.restore_child)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.config is None:
        args.config = "mamba3_phase_a_components" if args.suite == "components" else "pi05_mamba3_phase_a_fullft_b1"
    if args.suite != "components" and args.config == "mamba3_phase_a_components":
        parser.error("Real suites require pi05_mamba3_phase_a_fullft_b1, never a diagnostic config")
    if args.suite == "components" and args.config != "mamba3_phase_a_components":
        parser.error("Components are diagnostic-only; use mamba3_phase_a_components")
    if args.steps < 10:
        parser.error("--steps must be at least 10 for the gate-opening audit")
    if args.config not in ("pi05_mamba3_phase_a_fullft_b1", "mamba3_phase_a_components"):
        parser.error("unknown Phase A config")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from scripts.mamba3_smoke_runtime import diagnostic_train_config
    from scripts.mamba3_smoke_runtime import run_updates
    from scripts.mamba3_smoke_runtime import synthetic_batch
    from scripts.mamba3_smoke_runtime import write_json

    result = {
        "schema_version": 1,
        "architecture": "pi05_mamba3_jax",
        "stage": "phase_a_plain_fm",
        "suite": args.suite,
        "overall_status": "NOT_RUN",
        "PHASE_A_COMPLETE": False,
        "tests": [],
        "diagnostic_only": args.suite == "components",
    }
    (output / "commands.txt").write_text(
        f"cwd={Path.cwd()}\npython={sys.executable}\n{shlex.join([sys.executable, '-m', 'scripts.smoke_mamba3_jax', *sys.argv[1:]])}\n"
    )
    write_json(output, "environment.json", environment())
    source_paths = [
        Path("scripts/train.py"),
        Path("src/openpi/models/pi0.py"),
        Path("src/openpi/models/gemma.py"),
        Path("src/openpi/models/siglip.py"),
        *Path("src/openpi/models/mamba3").glob("*.py"),
        *Path("src/openpi/models").glob("pi0_mamba*.py"),
        *Path("src/openpi/training").glob("mamba*.py"),
        Path("src/openpi/policies/mamba_policy.py"),
        *Path("scripts").glob("*mamba3*.py"),
        *Path("tests").glob("test_mamba3*.py"),
        *Path("training_configs/experiments/tomato_sharpa/mamba").glob("*.py"),
    ]
    write_json(output, "source_hashes.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    write_json(
        output,
        "manifest.json",
        {
            "git_commit": revision,
            "branch": subprocess.run(
                ["git", "branch", "--show-current"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "wandb_mode": "offline",
            "data_source": "synthetic diagnostic fixture" if args.suite == "components" else args.dataset,
        },
    )
    try:
        with (output / "console.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            if args.suite in ("components", "all"):
                run_pytest(output, ["tests/test_mamba3_core.py", "tests/test_mamba3_integration.py"], "components")
                if args.suite == "components":
                    config = diagnostic_train_config(output, steps=args.steps, seed=args.seed)
                    # Local logs + explicitly offline W&B; no uploads/camera logging.
                    import wandb

                    with wandb.init(
                        project="openpi-mamba3-smoke",
                        name="diagnostic_only",
                        mode="offline",
                        dir=str(output),
                        config={"diagnostic_only": True},
                    ):
                        run_updates(config, synthetic_batch(config.model), output, args.steps, diagnostic=True)
                    result["tests"].append(
                        {
                            "id": "diagnostic_train_checkpoint_resume",
                            "status": "PASS",
                            "evidence": "checkpoint_roundtrip.json",
                        }
                    )
            if args.suite in ("regression", "all"):
                run_pytest(
                    output,
                    [
                        "tests/test_trainrtc_delta_paper_integration.py",
                        "tests/test_delta_paper_training.py",
                        "src/openpi/models/pi0_rtc_test.py",
                        "src/openpi/models/pi0_test.py",
                    ],
                    "regression",
                    cpu=True,
                )
                result["tests"].append({"id": "R01_selected", "status": "PASS", "evidence": "regression.xml"})
            if args.suite in ("fullft", "checkpoint", "policy", "all"):
                config, issues = full_preflight(args, output)
                if issues:
                    result.update(overall_status="BLOCKED", blocking_reasons=issues)
                    result["tests"].extend(
                        {"id": key, "status": "BLOCKED", "reason": "; ".join(issues)}
                        for key in (
                            "D01",
                            "T01",
                            "G01_full",
                            "V01_full",
                            "V02_full",
                            "K01_full",
                            "K02_full",
                            "K03_full",
                            "P01_full",
                        )
                    )
                    write_json(output, "results.json", result)
                    return 2
                # No alternate trainer: use real data, existing transforms, init and train_step.
                import jax
                import wandb

                from openpi.policies.mamba_policy import MambaOfflinePolicy
                from openpi.training.data_loader import create_data_loader
                from openpi.training.mamba_training import validate_training
                from training_configs import common

                common.configure_environment()
                validate_training(config)
                loader = create_data_loader(config, shuffle=False, num_batches=1)
                batch = next(iter(loader))
                with wandb.init(project="openpi-mamba3-smoke", name=config.exp_name, mode="offline", dir=str(output)):
                    state = run_updates(config, batch, output, args.steps, diagnostic=False)
                # Offline policy through exact source sample + transform inversion.
                from openpi.training.data_loader import create_torch_dataset

                source = create_torch_dataset(loader.data_config(), config.model.action_horizon, config.model)
                raw = source[0]
                from openpi.training.mamba_training import load_mamba_checkpoint

                model = load_mamba_checkpoint(
                    config.model, output / "checkpoints" / str(int(state.step) - 1), loader.data_config().asset_id
                )
                policy = MambaOfflinePolicy(model, loader.data_config())
                import numpy as np

                actions = policy.infer(raw, rng=jax.random.key(args.seed))["actions"]
                if actions.shape != (1, 50, 58) or not np.isfinite(actions).all():
                    raise AssertionError("Invalid physical Absolute policy output")
                write_json(
                    output,
                    "policy_roundtrip.json",
                    {
                        "shape": list(actions.shape),
                        "finite": True,
                        "action_semantics": "absolute",
                        "diagnostic_only": False,
                    },
                )
                result["tests"].append({"id": "T01_K01_K02_K03_P01", "status": "PASS"})
                result["tests"].append({"id": "V02_full", "status": "PASS", "evidence": "prefix_parity.json"})
            result["overall_status"] = "SELECTED_SUITE_PASS"
    except Exception as error:
        result.update(overall_status="FAIL", error=f"{type(error).__name__}: {error}")
        (output / "error.txt").write_text(traceback.format_exc())
        write_json(output, "results.json", result)
        return 1
    write_json(output, "results.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
