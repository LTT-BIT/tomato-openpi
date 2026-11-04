"""Strict VLM-only initialization: preserve explicitly new NNX parameter leaves."""

from dataclasses import dataclass
import json
import logging
from pathlib import Path

from flax import traverse_util
import numpy as np

from openpi.models import model
from openpi.shared import download

logger = logging.getLogger(__name__)
NEW_ROOTS = {"context_compressor", "action_expert"}
OLD_HEADS = {
    "action_in_proj",
    "action_out_proj",
    "state_proj",
    "time_mlp_in",
    "time_mlp_out",
    "action_time_mlp_in",
    "action_time_mlp_out",
}
# Exact additional-expert module names in gemma.py. Never a broad '_1' match.
OLD_EXPERT_MODULES = {
    "qkv_einsum_1",
    "q_einsum_1",
    "kv_einsum_1",
    "attn_vec_einsum_1",
    "mlp_1",
    "pre_attention_norm_1",
    "pre_ffw_norm_1",
    "final_norm_1",
}


def is_old_expert(path):
    return path[0] in OLD_HEADS or (path[:2] == ("PaliGemma", "llm") and bool(set(path) & OLD_EXPERT_MODULES))


def merge_vlm_weights(params, loaded):
    target = traverse_util.flatten_dict(params)
    source = traverse_util.flatten_dict(loaded)
    result, reused, initialized, ignored = {}, [], [], []
    for path, ref in target.items():
        if path[0] in NEW_ROOTS:
            result[path] = ref
            initialized.append("/".join(map(str, path)))
        elif path[:2] in (("PaliGemma", "llm"), ("PaliGemma", "img")):
            if path not in source:
                raise ValueError(f"Missing required pretrained VLM weight: {'/'.join(map(str, path))}")
            value = source[path]
            if value.shape != ref.shape:
                raise ValueError(f"VLM shape mismatch: {'/'.join(map(str, path))} {value.shape} != {ref.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite pretrained VLM weight: {'/'.join(map(str, path))}")
            result[path] = value.astype(ref.dtype)
            reused.append("/".join(map(str, path)))
        else:
            raise ValueError(f"Unexpected Mamba target parameter: {'/'.join(map(str, path))}")
    for path in source.keys() - target.keys():
        if not is_old_expert(path):
            raise ValueError(f"Unexpected pretrained source parameter: {'/'.join(map(str, path))}")
        ignored.append("/".join(map(str, path)))
    if not reused:
        raise ValueError("No VLM parameters in target")
    report = {
        "target_leaves": len(target),
        "reused_vlm_leaves": len(reused),
        "reused_vlm_parameters": sum(source[tuple(p.split("/"))].size for p in reused),
        "new_leaves": len(initialized),
        "ignored_old_expert_leaves": len(ignored),
        "unexpected_missing": 0,
        "mismatched": 0,
        "reused_paths": reused,
        "initialized_paths": initialized,
        "ignored_paths": ignored,
    }
    return traverse_util.unflatten_dict(result), report


@dataclass(frozen=True)
class MambaVlmInitLoader:
    params_path: str
    report_path: str | None = None

    def load(self, params):
        loaded = model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        result, report = merge_vlm_weights(params, loaded)
        logger.info("Mamba VLM initialization: %s", report)
        if self.report_path is not None:
            Path(self.report_path).write_text(json.dumps(report, indent=2) + "\n")
        return result


@dataclass(frozen=True)
class PaperMambaVlmInitLoader(MambaVlmInitLoader):
    """Keep the validated fixed prior state, initialize the VLM with the strict loader."""

    def load(self, params):
        base = {key: value for key, value in params.items() if key != "paper_noise_cholesky"}
        result = super().load(base)
        if "paper_noise_cholesky" in params:
            result["paper_noise_cholesky"] = params["paper_noise_cholesky"]
        return result
