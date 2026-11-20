"""Adapted causal V-JEPA features from the supplied lemon training package.

Source: tools/train_lemon_vjepa_phase.py, history_indices/encode_clips/head_model.
Tomato uses seven phase occurrences, three views and 58 canonical Sharpa joints.
No lemon weights, source paths, six-class labels or robot command code are used.
Imports of torch/transformers are lazy; annotation preparation needs neither.
"""

import hashlib
from pathlib import Path

import numpy as np

from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.subtasks.tomato import PHASES
from openpi.subtasks.tomato import PROMPTS
from openpi.subtasks.tomato import SOURCE_STATE_NAMES
from openpi.subtasks.tomato import STATE_DIM

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
WEIGHTS_SHA256 = "25466aef85727d16546c6cf8c99f12fcfad9cbca8225d45f23685e2e025b786b"
CAMERAS = ("cam0", "cam1", "cam2")
CLIP, SIZE, SAMPLE_HZ = 16, 256, 10


def canonical_state(state, names):
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 116 or names != list(SOURCE_STATE_NAMES) or not np.isfinite(state).all():
        raise ValueError("Expected named finite converter Sharpa states in the verified source order")
    return state[:, [names.index(f"{joint}.position") for joint in ORDER]]


def history_indices(times, end):
    """1.5 seconds of causal history at 10 Hz, for normal OR speedup input FPS."""
    times = np.asarray(times, dtype=np.float64)
    if (
        times.ndim != 1
        or not len(times)
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
        or type(end) is not int
        or not 0 <= end < len(times)
    ):
        raise ValueError("Expected increasing finite times and a valid current frame")
    targets = times[end] - np.arange(CLIP - 1, -1, -1) / SAMPLE_HZ
    # A future observation must never fill a missing historical sample.
    return np.maximum(0, np.searchsorted(times[: end + 1], targets, side="right") - 1)


def head_model(dimension):
    from torch import nn

    return nn.Sequential(nn.Linear(dimension, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, len(PHASES)))


def load_encoder(path, device="cpu"):
    import torch
    from transformers import VJEPA2Model

    # Pinned source weight identity, recorded in the supplied training_config.json.
    digest = hashlib.sha256()
    with (Path(path) / "model.safetensors").open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Encoder weights differ from the pinned V-JEPA revision")
    # Explicit local weight path: never trigger a network download at robot startup.
    model = VJEPA2Model.from_pretrained(
        path, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    return model.encoder.eval().requires_grad_(requires_grad=False).to(device)


def encode_features(encoder, clips, states, device="cpu"):
    """Three synchronized full-view RGB histories plus canonical joint history."""
    import torch

    clips, states = np.asarray(clips), np.asarray(states, dtype=np.float32)
    if clips.shape != (3, CLIP, SIZE, SIZE, 3) or clips.dtype != np.uint8:
        raise ValueError("Expected three 16-frame 256x256 uint8 RGB histories")
    if states.shape != (CLIP, STATE_DIM) or not np.isfinite(states).all():
        raise ValueError("Expected 16x58 finite canonical measured joint positions")
    with torch.inference_mode():
        pixels = torch.from_numpy(clips).to(device).permute(0, 1, 4, 2, 3).float() / 255
        mean = pixels.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
        std = pixels.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
        visual = encoder(pixel_values_videos=(pixels - mean) / std).last_hidden_state.mean(dim=1)
        result = np.concatenate([visual.float().cpu().numpy().reshape(-1), states.reshape(-1)])
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite V-JEPA features")
    return result


def training_statistics(features):
    """Call on training split only; validation/test must reuse these values."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not len(features) or not np.isfinite(features).all():
        raise ValueError("Expected finite training features")
    return features.mean(axis=0, dtype=np.float64).astype(np.float32), features.std(axis=0, dtype=np.float64).clip(
        0.01
    ).astype(np.float32)


class PhaseModel:
    """Decision-only inference with an explicitly trained tomato head."""

    def __init__(self, head_path, encoder_path, device="cpu"):
        import torch

        saved = torch.load(head_path, map_location="cpu", weights_only=True)
        expected = {
            "labels": list(PHASES),
            "prompts": list(PROMPTS),
            "joints": list(ORDER),
            "cameras": list(CAMERAS),
            "encoder_revision": REVISION,
            "clip": CLIP,
            "sample_hz": SAMPLE_HZ,
        }
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Expected a tomato/Sharpa causal seven-phase checkpoint; lemon heads are incompatible")
        self.mean = np.asarray(saved["mean"], dtype=np.float32)
        self.std = np.asarray(saved["std"], dtype=np.float32)
        dim = saved["dimension"]
        if (
            self.mean.shape != (dim,)
            or self.std.shape != (dim,)
            or not np.isfinite(self.mean).all()
            or not np.isfinite(self.std).all()
            or np.any(self.std <= 0)
        ):
            raise ValueError("Invalid training-only feature normalization")
        self.device = device
        self.encoder = load_encoder(encoder_path, device)
        self.head = head_model(dim).to(device).eval()
        self.head.load_state_dict(saved["head"])

    def predict(self, clips, states):
        import torch

        x = encode_features(self.encoder, clips, states, self.device)
        if x.shape != self.mean.shape:
            raise ValueError("Encoder/head feature dimension mismatch")
        with torch.inference_mode():
            logits = self.head(torch.from_numpy((x - self.mean) / self.std).to(self.device)[None])
            probabilities = logits.softmax(dim=-1)[0].cpu().numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite phase prediction")
        return probabilities
