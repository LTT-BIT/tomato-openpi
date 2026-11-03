"""CPU safety tests for strict VLM loading and offline checkpoint restoration."""

from flax import nnx
import numpy as np
import pytest

from openpi.training import flash_checkpoint as checkpoint


def _params(action_dim=32):
    return {
        "PaliGemma": {
            "img": {"kernel": np.ones((2, 3), np.float32)},
            "llm": {
                "layers": {
                    "attn": {
                        "q_einsum": {"w": np.ones((3, 3), np.float32)},
                        "q_einsum_1": {"w": np.ones((2, 2), np.float32)},
                    }
                },
                "final_norm_1": {"scale": np.ones(2, np.float32)},
            },
        },
        "action_in_proj": {"kernel": np.zeros((action_dim, 2), np.float32)},
        "action_out_proj": {"bias": np.zeros(action_dim, np.float32)},
    }


def test_only_action_projection_mismatch_allowed():
    loaded, report = checkpoint.merge_flash_params(_params(58), _params(32))
    assert loaded["action_in_proj"]["kernel"].shape == (58, 2)
    assert len(report["retained_projection_leaves"]) == 2
    assert report["source_vlm_sha256"] == report["loaded_vlm_sha256"]
    source = _params()
    source["PaliGemma"]["img"]["kernel"] = np.zeros((1, 3), np.float32)
    with pytest.raises(ValueError, match="Forbidden shape mismatch"):
        checkpoint.merge_flash_params(_params(), source)


def test_missing_vlm_is_rejected():
    source = _params()
    del source["PaliGemma"]["img"]
    with pytest.raises(ValueError, match="Missing checkpoint parameters"):
        checkpoint.merge_flash_params(_params(), source)


def test_vlm_dtype_change_is_rejected_by_hash():
    source = _params()
    source["PaliGemma"]["img"]["kernel"] = source["PaliGemma"]["img"]["kernel"].astype(np.float64)
    with pytest.raises(ValueError, match="VLM content changed"):
        checkpoint.merge_flash_params(_params(), source)


def test_vlm_filter_covers_vision_and_excludes_expert():
    assert checkpoint.is_vlm_parameter(("PaliGemma", "img", "kernel"))
    assert checkpoint.is_vlm_parameter(("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"))
    assert not checkpoint.is_vlm_parameter(("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"))
    assert not checkpoint.is_vlm_parameter(("PaliGemma", "llm", "final_norm_1", "scale"))
    assert checkpoint.is_flash_trainable_parameter(("action_out_proj", "bias"))
    assert not checkpoint.is_flash_trainable_parameter(("poly", "matrix"), nnx.Variable(np.ones(2)))
    assert checkpoint.is_flash_trainable_parameter(("action_out_proj", "bias"), nnx.Param(np.ones(2)))


def test_roundtrip_and_no_overwrite(tmp_path):
    metadata = {
        "config": {"K": 6},
        "normalization": {"actions": "quantile"},
        "step": 1,
        "optimizer": "SGD(no momentum)",
    }
    path = checkpoint.save_offline_checkpoint(
        tmp_path / "new", _params(), metadata=metadata, optimizer_state={"count": np.asarray(1, np.int32)}
    )
    restored, actual_metadata, optimizer = checkpoint.restore_offline_checkpoint(path)
    assert checkpoint.parameter_digest(restored) == checkpoint.parameter_digest(_params())
    assert actual_metadata == metadata
    assert optimizer["count"] == 1
    with pytest.raises(FileExistsError):
        checkpoint.save_offline_checkpoint(path, _params(), metadata=metadata)
