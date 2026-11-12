"""Train-only full-window statistics, isolated assets and immutable FLASH T3 binding."""

from dataclasses import fields
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from openpi.models.pi0_flash import Pi0FlashConfig
from openpi.shared.flash_delta_t3 import MASK
from openpi.shared.flash_delta_t3 import VERSION
from openpi.shared.flash_delta_t3 import correlation
from openpi.shared.flash_delta_t3 import encode
from openpi.shared.flash_polynomial import FlashPolynomial
from openpi.shared.local_storage import new_artifact_directory
from openpi.training import flash_native as native


def enabled(config):
    return hasattr(config.model, "t3_assets")


def base(config):
    return replace(
        config,
        model=Pi0FlashConfig(**{f.name: getattr(config.model, f.name) for f in fields(Pi0FlashConfig)}),
        policy_metadata={**config.policy_metadata, "_flash_t3_base_view": True},
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(model):
    directory = Path(model.t3_assets)
    meta = json.loads((directory / "flash_t3.json").read_text())
    expected = {
        "version": VERSION,
        "timestamp": model.timestamp,
        "beta": model.correlation_beta,
        "history_grid": list(range(-1, 7)),
        "target_grid": list(range(-2, 52)),
        "covariance_source": "history_coefficients",
        "history_reg_lambda": model.history_reg_lambda,
    }
    if (
        any(meta.get(k) != v for k, v in expected.items())
        or not model.t3_sha256
        or sha(directory / "flash_t3.npz") != model.t3_sha256
        or meta["sha256"] != model.t3_sha256
    ):
        raise ValueError("FLASH Delta/T3 algorithm or asset digest mismatch")
    if meta["normalization_sha256"] != sha(directory / "norm_stats.json") or meta["seal_sha256"] != sha(
        directory / "flash_seal.json"
    ):
        raise ValueError("FLASH T3 normalization/split binding mismatch")
    with np.load(directory / "flash_t3.npz", allow_pickle=False) as bundle:
        arrays = {k: bundle[k].copy() for k in bundle.files}
    for k, shape in {"mean": (54, 58), "std": (54, 58), "low": (58,), "high": (58,), "cholesky": (406, 406)}.items():
        if arrays[k].shape != shape or not np.isfinite(arrays[k]).all():
            raise ValueError("Invalid FLASH T3 statistics")
    if (
        (arrays["std"] < 0).any()
        or (arrays["high"] < arrays["low"]).any()
        or (np.diag(arrays["cholesky"]) <= 0).any()
        or not np.allclose(arrays["cholesky"], np.tril(arrays["cholesky"]))
    ):
        raise ValueError("Invalid FLASH T3 scales or Cholesky")
    return arrays


def bind(config):
    directory = native.asset_directory(config)
    path = directory / "flash_t3.npz"
    return replace(
        config, model=replace(config.model, t3_assets=str(directory), t3_sha256=sha(path) if path.exists() else "")
    )


def preflight(config, *, require_norm=True):
    report = native.preflight(base(config), require_norm=require_norm)
    if config.resume and not config.model.t3_sha256:
        raise ValueError("Missing FLASH T3 resume assets")
    if require_norm:
        load(config.model if config.model.t3_sha256 else bind(config).model)
    report.update(
        training_action_semantics="delta_arms_absolute_hands",
        t3_version=VERSION,
        timestamp=config.model.timestamp,
        correlation_beta=config.model.correlation_beta,
    )
    if require_norm:
        report["t3_sha256"] = bind(config).model.t3_sha256
    return report


def prepare(config, *, batch_size=256):
    native.preflight(base(config), require_norm=False)
    destination = native.asset_directory(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with new_artifact_directory(destination) as staging, tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        plain = base(config)
        plain = replace(
            plain, data=replace(plain.data, assets=replace(plain.data.assets, assets_dir=tmp, asset_id="base"))
        )
        native.prepare(plain)
        for path in (Path(tmp) / "base").iterdir():
            shutil.copyfile(path, staging / path.name)
        manifest = json.loads((staging / "flash_manifest.json").read_text())
        train = [r for r in manifest["records"] if r["source_recording_id"] in manifest["train_sources"]]
        total = sum(r["rows"] - 58 for r in train)
        if total < 2:
            raise ValueError("At least two training windows required")
        selected = set(np.random.default_rng(config.data_seed).choice(total, min(total, 20000), replace=False).tolist())
        count = 0
        sums = np.zeros((54, 58))
        squares = sums.copy()
        samples = []
        histories = []
        cursor = 0
        for record in train:
            actions, states, _ = native.read_numeric(record)
            for offset in range(0, record["rows"] - 58, batch_size):
                ticks = np.arange(offset, min(offset + batch_size, record["rows"] - 58)) + 7
                anchors = states[ticks]
                target = actions[ticks[:, None] + np.arange(-2, 52)] - np.where(MASK, anchors[:, None, :], 0)
                sums += target.sum(0, dtype=np.float64)
                squares += np.square(target.astype(np.float64)).sum(0)
                count += len(ticks)
                for i, t in enumerate(ticks):
                    if cursor + i in selected:
                        samples.append(target[i])
                        histories.append((states[t - 7 : t + 1], anchors[i]))
                cursor += len(ticks)
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - mean**2, 0))
        low, high = np.quantile(np.asarray(samples), [0.01, 0.99], axis=(0, 1))
        poly = FlashPolynomial(history_reg_lambda=config.model.history_reg_lambda)
        coeff = []
        for offset in range(0, len(histories), batch_size):
            group = histories[offset : offset + batch_size]
            history = np.asarray([h for h, a in group])
            anchor = np.asarray([a for h, a in group])
            encoded = np.asarray(
                encode(history, anchor, mean[1:9], std[1:9], low, high, timestamp=config.model.timestamp)
            )
            coeff.append(np.asarray(poly.fit_history(encoded)).reshape(-1, 406))
        chol = correlation(np.concatenate(coeff), config.model.correlation_beta)
        np.savez(
            staging / "flash_t3.npz",
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            cholesky=chol,
        )
        meta = {
            "version": VERSION,
            "timestamp": config.model.timestamp,
            "beta": config.model.correlation_beta,
            "history_grid": list(range(-1, 7)),
            "target_grid": list(range(-2, 52)),
            "covariance_source": "history_coefficients",
            "history_reg_lambda": config.model.history_reg_lambda,
            "training_windows": count,
            "correlation_windows": len(histories),
            "quantile_scope": "uniform_training_windows_max20000",
            "sha256": sha(staging / "flash_t3.npz"),
            "normalization_sha256": sha(staging / "norm_stats.json"),
            "seal_sha256": sha(staging / "flash_seal.json"),
        }
        (staging / "flash_t3.json").write_text(json.dumps(meta, indent=2))
    return preflight(bind(config))
