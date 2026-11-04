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
import re
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
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "JAX_DEFAULT_MATMUL_PRECISION": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
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


def experiment_module(args):
    if args.config and args.config.startswith("training_configs.experiments."):
        return args.config
    suffix = "phase_a" if args.algorithm == "plain" else f"paper_{args.algorithm}"
    return f"training_configs.experiments.tomato_sharpa.mamba.{suffix}"


def full_preflight(args, output):
    from dataclasses import replace

    from flax import nnx
    import jax

    from openpi.models.pi0_mamba import MambaPi0
    from openpi.shared import delta_paper as paper
    from openpi.training import optimizer
    from openpi.training.mamba_training import MambaTrainConfig
    from openpi.training.mamba_weight_loader import MambaVlmInitLoader
    from openpi.training.mamba_weight_loader import PaperMambaVlmInitLoader
    from scripts import run_training
    from scripts.create_delta_paper_experiments import TRIALS
    from scripts.mamba3_smoke_runtime import inventory
    from scripts.mamba3_smoke_runtime import write_json

    config = run_training.load_experiment(experiment_module(args))
    if not isinstance(config, MambaTrainConfig) or config.model.diagnostic_only:
        raise ValueError("Real suites require a production MambaTrainConfig")
    is_paper = hasattr(config.model, "paper_mode")
    selected = {name: (mode, beta) for name, mode, beta in TRIALS}
    if is_paper != (args.algorithm != "plain") or (
        is_paper and (config.model.paper_mode, config.model.paper_beta) != selected[args.algorithm]
    ):
        raise ValueError("Selected module differs from --algorithm")
    norm = (
        Path(args.norm_stats).expanduser().resolve()
        if args.norm_stats
        else Path(config.data.assets.assets_dir or config.assets_dirs) / config.data.assets.asset_id / "norm_stats.json"
    )
    init = Path(args.init_checkpoint or config.weight_loader.params_path).expanduser().resolve()
    specs = config.data.lerobot_datasets
    metadata = dict(config.policy_metadata or {})
    if args.dataset:
        if len(specs) != 1:
            raise ValueError("Use an explicit multi-source experiment module; --dataset overrides a single source only")
        root = str(Path(args.dataset).expanduser().resolve())
        specs = (replace(specs[0], root=root),)
        metadata.update(dataset_root=root, source_datasets=[root])
    data = replace(
        config.data,
        assets=replace(config.data.assets, assets_dir=str(norm.parent.parent), asset_id=norm.parent.name),
        lerobot_datasets=specs,
    )
    model = config.model
    issues = []
    if norm.name != "norm_stats.json":
        issues.append("--norm-stats must point to norm_stats.json in its independent asset directory")
    for name, path in (
        ("init_checkpoint", init),
        ("norm_stats", norm),
        *(("dataset", Path(spec.root)) for spec in specs),
    ):
        if not path.exists():
            issues.append(f"missing {name}: {path}")
    if is_paper:
        missing = [
            name for name in (paper.BUNDLE, paper.MANIFEST, "norm_metadata.json") if not (norm.parent / name).is_file()
        ]
        issues.extend(f"missing Paper asset: {name}" for name in missing)
        digest = paper.sha256(norm.parent / paper.BUNDLE) if not missing else None
        if model.paper_asset_sha256 and digest and model.paper_asset_sha256 != digest:
            raise ValueError("Supplied statistics differ from pinned Paper bundle")
        model = replace(model, paper_asset_dir=str(norm.parent), paper_asset_sha256=digest)
    loader_type = PaperMambaVlmInitLoader if is_paper else MambaVlmInitLoader
    config = replace(
        config,
        model=model,
        data=data,
        policy_metadata=metadata,
        seed=args.seed,
        model_seed=args.seed,
        num_train_steps=args.steps,
        batch_size=1,
        num_workers=0,
        fsdp_devices=1,
        checkpoint_base_dir=str(output / "training_checkpoints"),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1, peak_lr=2.5e-5, decay_steps=max(100, args.steps + 1), decay_lr=2.5e-6
        ),
        weight_loader=loader_type(str(init), report_path=str(output / "weight_load_report.json")),
    )
    # Shapes depend on the architecture, not the prior contents. This audit invokes
    # only the plain composition under eval_shape, never a runnable unvalidated Paper model.
    abstract = nnx.eval_shape(lambda rng: MambaPi0(config.model, nnx.Rngs(rng)), jax.random.key(args.seed))
    groups = inventory(nnx.state(abstract, nnx.Param))
    write_json(output, "production_parameter_inventory.json", groups)
    param_bytes = sum(v["bytes"] for v in groups.values())
    prior_bytes = (
        (config.model.action_horizon * config.model.action_dim) ** 2 * 4 if is_paper and config.model.paper_beta else 0
    )
    minimum = param_bytes * 3 + prior_bytes
    limits = [d.memory_stats().get("bytes_limit", 0) for d in jax.devices() if d.platform == "gpu"]
    write_json(
        output,
        "memory_preflight.json",
        {
            "parameters_bytes": param_bytes,
            "fixed_prior_bytes": prior_bytes,
            "parameters_and_adam_minimum_bytes": minimum,
            "device_allocator_limits": limits,
            "measurement": "shape-derived lower bound, excludes gradients/activations/workspace; no OOM trial claimed",
        },
    )
    if jax.device_count() != 1:
        issues.append("B1 smoke requires exactly one visible device; set CUDA_VISIBLE_DEVICES before starting Python")
    if not limits:
        issues.append("no JAX GPU device; real FullFT suite requires GPU")
    elif max(limits) < minimum:
        issues.append("FULL_E2E_BLOCKED_BY_MEMORY_PREFLIGHT: parameters + Adam exceed device budget")
    if issues:
        return None, issues
    inputs = run_training.preflight(config)
    write_json(output, "input_preflight.json", inputs)
    return config, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("components", "fullft", "checkpoint", "policy", "regression", "all"), default="components"
    )
    parser.add_argument("--config")
    parser.add_argument("--algorithm", choices=("plain", "t0", "t1", "t2", "t3", "tg"), default="plain")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--dataset")
    parser.add_argument("--norm-stats")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disable-triton-gemm",
        action="store_true",
        help="Local workaround for locked JAX 0.5.3 Triton GEMM compiler failures; sets only this process environment",
    )
    parser.add_argument(
        "--deterministic-gpu",
        action="store_true",
        help="Request deterministic GPU operations for exact checkpoint replay diagnostics",
    )
    parser.add_argument("--restore-child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.disable_triton_gemm:
        flags = re.sub(
            r"(?<!\S)--xla_gpu_enable_triton_gemm=(?:true|false|0|1)(?=\s|$)", "", os.environ.get("XLA_FLAGS", "")
        ).strip()
        os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_enable_triton_gemm=false").strip()
    if args.deterministic_gpu:
        os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        flags = re.sub(
            r"(?<!\S)--xla_gpu_deterministic_ops=(?:true|false|0|1)(?=\s|$)", "", os.environ.get("XLA_FLAGS", "")
        ).strip()
        os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_deterministic_ops=true").strip()
    if args.restore_child:
        from scripts.mamba3_smoke_runtime import restore_child

        return restore_child(args.restore_child)
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.config is None:
        args.config = (
            ("mamba3_phase_a_components" if args.algorithm == "plain" else "mamba3_paper_components")
            if args.suite == "components"
            else experiment_module(args)
        )
    if args.suite != "components" and args.config.endswith("components"):
        parser.error("Real suites require a production experiment module, never a diagnostic config")
    expected_components = "mamba3_phase_a_components" if args.algorithm == "plain" else "mamba3_paper_components"
    if args.suite == "components" and args.config != expected_components:
        parser.error(f"Use --config {expected_components} for the selected diagnostic algorithm")
    if (
        args.suite != "components"
        and not args.config.startswith("training_configs.experiments.")
        and not (args.algorithm == "plain" and args.config == "pi05_mamba3_phase_a_fullft_b1")
    ):
        parser.error("Unknown configuration; supply a training_configs.experiments module")
    if args.steps < 10:
        parser.error("--steps must be at least 10 for the gate-opening audit")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    from scripts.mamba3_smoke_runtime import diagnostic_train_config
    from scripts.mamba3_smoke_runtime import run_updates
    from scripts.mamba3_smoke_runtime import synthetic_batch
    from scripts.mamba3_smoke_runtime import write_json

    result = {
        "schema_version": 1,
        "architecture": "pi05_mamba3_jax",
        "stage": "phase_a_plain_fm" if args.algorithm == "plain" else "paper_delta_fm",
        "algorithm": args.algorithm,
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
                paths = (
                    ["tests/test_mamba3_core.py", "tests/test_mamba3_integration.py"]
                    if args.algorithm == "plain"
                    else ["tests/test_mamba3_paper.py"]
                )
                run_pytest(output, paths, "components")
                if args.suite == "components" and args.algorithm != "plain":
                    result["tests"].append(
                        {"id": "T1_T2_T3_diagnostic_matrix", "status": "PASS", "evidence": "components.xml"}
                    )
                if args.suite == "components" and args.algorithm == "plain":
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

                from openpi.policies.mamba_policy import create_mamba_policy
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
                policy = create_mamba_policy(model, loader.data_config())
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
                        "model_action_semantics": config.model.action_representation,
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
