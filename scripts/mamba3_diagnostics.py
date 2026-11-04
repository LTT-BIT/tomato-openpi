"""Explicitly synthetic, reduced VLM fixture. Never used by a training config."""

from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

from openpi.models import gemma
from openpi.models import siglip
from openpi.models.mamba3.config import Mamba3Config
from openpi.models.pi0_mamba_config import MambaPi0Config


def diagnostic_config():
    return MambaPi0Config(
        diagnostic_only=True,
        paligemma_variant="dummy",
        dtype="float32",
        max_token_len=8,
        mamba=Mamba3Config(hidden_size=16, state_size=8, head_dim=8, dtype="float32"),
        mamba_depth=2,
        mamba_mlp_dim=32,
        film_mlp_dim=32,
        context_heads=2,
        context_vision_tokens=5,
        context_language_tokens=3,
        context_global_tokens=2,
    )


@contextmanager
def diagnostic_network():
    """Shrink only inside this explicitly named fixture; production code is unchanged."""
    original_gemma = gemma.get_config
    original_siglip = siglip.Module

    def small_gemma(variant):
        if variant == "dummy":
            return replace(original_gemma(variant), width=16, depth=2, mlp_dim=32, num_heads=2, head_dim=8)
        return original_gemma(variant)

    def small_siglip(*args, **kwargs):
        return original_siglip(
            *args,
            **{
                **kwargs,
                "width": 16,
                "depth": 2,
                "mlp_dim": 32,
                "num_heads": 2,
                "patch_size": (112, 112),
                "head_zeroinit": False,
            },
        )

    with (
        patch.object(gemma, "get_config", small_gemma),
        patch.object(gemma, "PALIGEMMA_VOCAB_SIZE", 128),
        patch.object(siglip, "Module", small_siglip),
    ):
        yield


def record_evidence(name, values):
    """Persist small numerical summaries only when the smoke runner requests it."""
    import json
    import os
    from pathlib import Path

    directory = os.environ.get("MAMBA_SMOKE_EVIDENCE")
    if directory is None:
        return
    path = Path(directory) / "numerical_evidence.json"
    records = json.loads(path.read_text()) if path.exists() else {}
    records[name] = values
    path.write_text(json.dumps(records, indent=2) + "\n")
