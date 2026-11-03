"""Strict pretrained loading and non-overwriting offline FLASH checkpoints.

Parameter checkpoints are compatible with ``model.restore_params``. The small
JSON sidecar records normalization/configuration and optimizer identity; real
training should continue to use ``training.checkpoints.save_state`` for its
full optimizer and data-loader state.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from flax import nnx
from flax import traverse_util
import numpy as np


def _parts(path):
    return tuple(str(p) for p in path) if not isinstance(path, str) else tuple(path.split("/"))


def is_vlm_parameter(path, value=None) -> bool:
    """NNX filter: frozen vision and original Gemma, excluding action expert."""
    del value
    parts = _parts(path)
    if parts[:2] == ("PaliGemma", "img"):
        return True
    return parts[:2] == ("PaliGemma", "llm") and not any(p.endswith("_1") for p in parts[2:])


def is_flash_trainable_parameter(path, value=None) -> bool:
    """NNX gradient filter excluding frozen VLM and non-Param buffers."""
    is_parameter = value is None or nnx.filterlib.OfType(nnx.Param)(path, value)
    return is_parameter and not is_vlm_parameter(path, value)


def parameter_digest(params, *, vlm_only=False) -> str:
    """Content hash including path, dtype and shape (CPU; no model execution)."""
    digest = hashlib.sha256()
    for path, value in sorted(traverse_util.flatten_dict(params).items()):
        if vlm_only and not is_vlm_parameter(path):
            continue
        array = np.asarray(value)
        digest.update("/".join(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        # Hash a byte view so bfloat16 does not require conversion/copies.
        digest.update(memoryview(np.ascontiguousarray(array).view(np.uint8)))
    return digest.hexdigest()


def merge_flash_params(reference, loaded, *, audit_hashes=True):
    """Strict merge: retain only dimension-changing action projections.

    Every reference leaf must be supplied by the checkpoint. In particular,
    missing VLM/expert weights cannot silently remain randomly initialized.
    Only action_in_proj and action_out_proj shapes may differ. FLASH polynomial
    matrices are deterministic non-Param buffers and do not belong in this tree.
    """
    ref = traverse_util.flatten_dict(reference)
    source = traverse_util.flatten_dict(loaded)
    result = {}
    skipped = []
    missing = []
    for path, target in ref.items():
        key = "/".join(path)
        if path not in source:
            missing.append(key)
            continue
        value = source[path]
        if tuple(value.shape) != tuple(target.shape):
            if path[0] not in ("action_in_proj", "action_out_proj"):
                raise ValueError(f"Forbidden shape mismatch at {key}: {value.shape} != {target.shape}")
            result[path] = target
            skipped.append({"path": key, "source_shape": list(value.shape), "target_shape": list(target.shape)})
        else:
            # ShapeDtypeStruct references are supported without allocating them.
            result[path] = value.astype(target.dtype) if value.dtype != target.dtype else value
    if missing:
        raise ValueError(f"Missing checkpoint parameters (including any VLM leaves): {missing}")
    unexpected = sorted("/".join(p) for p in source.keys() - ref.keys())
    if unexpected:
        raise ValueError(f"Unexpected checkpoint parameters: {unexpected}")
    merged = traverse_util.unflatten_dict(result)
    report = {
        "loaded_leaf_count": len(ref) - len(skipped),
        "vlm_leaf_count": sum(is_vlm_parameter(p) for p in ref),
        "retained_projection_leaves": skipped,
        "missing": [],
        "unexpected": [],
    }
    if audit_hashes:
        report["source_vlm_sha256"] = parameter_digest(loaded, vlm_only=True)
        report["loaded_vlm_sha256"] = parameter_digest(merged, vlm_only=True)
        if report["source_vlm_sha256"] != report["loaded_vlm_sha256"]:
            raise ValueError("VLM content changed during loading (including dtype conversion)")
    return merged, report


@dataclasses.dataclass
class FlashCheckpointWeightLoader:
    params_path: str
    audit_hashes: bool = True
    report: dict[str, Any] | None = dataclasses.field(default=None, init=False)

    def load(self, params):
        from openpi.models import model
        from openpi.shared import download

        loaded = model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        merged, self.report = merge_flash_params(params, loaded, audit_hashes=self.audit_hashes)
        return merged


def save_offline_checkpoint(directory, params, *, metadata: Mapping[str, Any], optimizer_state=None):
    """Save a *new* offline checkpoint, never overwrite an existing directory.

    ``metadata`` must state config, normalization, step, and optimizer. Optimizer
    arrays (if any) use a separate Orbax item. For SGD without momentum the
    explicit optimizer identity and empty state are sufficient for resumption.
    """
    import orbax.checkpoint as ocp

    required = {"config", "normalization", "step", "optimizer"}
    if missing := required - metadata.keys():
        raise ValueError(f"Missing checkpoint metadata: {sorted(missing)}")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(directory / "params", {"params": params})
        if optimizer_state is not None:
            checkpointer.save(directory / "optimizer", {"optimizer": optimizer_state})
    (directory / "metadata.json").write_text(json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n")
    return directory


def restore_offline_checkpoint(directory):
    """Restore params to host arrays plus explicit offline optimizer metadata."""
    import orbax.checkpoint as ocp

    from openpi.models import model

    directory = Path(directory).resolve()
    params = model.restore_params(directory / "params", restore_type=np.ndarray)
    metadata = json.loads((directory / "metadata.json").read_text())
    optimizer = None
    if (directory / "optimizer").exists():
        with ocp.PyTreeCheckpointer() as checkpointer:
            optimizer = checkpointer.restore(directory / "optimizer")["optimizer"]
    return params, metadata, optimizer
