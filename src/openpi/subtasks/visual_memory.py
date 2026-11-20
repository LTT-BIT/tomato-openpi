"""Bounded, image-only execution history shared by training and policy adapters.

This is a sampled, spatially pooled visual-prefix baseline, not MEM or LTM.
Indices refer to the 30 Hz observation timeline, never policy-call counts.
"""

from collections import deque
from dataclasses import asdict
from dataclasses import dataclass
import math

import numpy as np

from openpi.policies.fr3_sharpa_policy import _parse_image

VERSION = "tomato-visual-memory-pooled-v1"
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "interaction_0_rgb")
THREE_VIEWS = IMAGE_KEYS[:3]
RAW_KEYS = (
    "observation.images.cam0",
    "observation.images.cam2",
    "observation.images.cam1",
    "observation.images.interaction",
)
HISTORY = "visual_history"


@dataclass(frozen=True)
class VisualMemorySpec:
    # Includes the current observation: t-45, t-30, t-15, t at the default 30 Hz.
    frames: int = 4
    stride: int = 15
    fps: int = 30
    pool_grid: int = 4
    image_keys: tuple[str, ...] = IMAGE_KEYS

    def __post_init__(self):
        for name in ("frames", "stride", "fps", "pool_grid"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.pool_grid > 16 or 16 % self.pool_grid:
            raise ValueError("pool_grid must divide the 16x16 SigLIP patch grid")
        if (
            not isinstance(self.image_keys, tuple)
            or not self.image_keys
            or len(set(self.image_keys)) != len(self.image_keys)
            or any(not isinstance(k, str) or not k or "__history_" in k for k in self.image_keys)
        ):
            raise ValueError("Unique current camera names required")

    @property
    def lags(self):
        return tuple(range((self.frames - 1) * self.stride, 0, -self.stride))

    @property
    def history_keys(self):
        return tuple(self.key(camera, lag) for lag in self.lags for camera in self.image_keys)

    @property
    def all_image_keys(self):
        return self.image_keys + self.history_keys

    @staticmethod
    def key(camera, lag):
        return f"{camera}__history_{lag}"

    def contract(self):
        return {
            "version": VERSION,
            **asdict(self),
            "image_keys": list(self.image_keys),
            "history_state": False,
            "current_state": True,
            "history_tokens": len(self.history_keys) * self.pool_grid**2,
            "history_span_seconds": (self.frames - 1) * self.stride / self.fps,
            "roi_source": "existing_offline_roi_may_use_future_annotation"
            if "interaction_0_rgb" in self.image_keys
            else None,
            "method": "sampled_spatial_pool_visual_prefix",
        }


def camera_frame(sample, mask_key="observation.interaction_available"):
    """Copy image pixels and masks; a null mask key selects the three original RGB views."""
    image_keys = THREE_VIEWS if mask_key is None else IMAGE_KEYS
    images = {}
    masks = {}
    for camera, raw in zip(IMAGE_KEYS, RAW_KEYS, strict=True):
        if camera not in image_keys:
            continue
        image = _parse_image(sample[raw])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 1:
            raise ValueError(f"RGB uint8 image required: {raw}")
        images[camera] = image.copy()
        masks[camera] = np.True_
    if "interaction_0_rgb" in images:
        valid = np.asarray(sample[mask_key])
        if valid.shape != () or valid.dtype != np.bool_:
            raise ValueError("Scalar boolean ROI availability required")
        masks["interaction_0_rgb"] = np.bool_(valid)
    return images, masks


def pack_history(spec, current, lookup):
    """Build fixed-shape slots; unavailable exact timestamps are zeroed and masked."""
    images, masks = {}, {}
    for lag in spec.lags:
        frame = lookup(lag)
        for camera in spec.image_keys:
            key = spec.key(camera, lag)
            valid = frame is not None and bool(frame[1][camera])
            images[key] = frame[0][camera].copy() if valid else np.zeros_like(current[camera])
            masks[key] = np.bool_(valid)
    return {"image": images, "image_mask": masks}


class VisualHistory:
    """One synchronous stream, with explicit episode identity and acquisition frame IDs.

    Call observe for each acquired observation, including those between policy calls.
    Missing exact historical IDs stay masked. An episode change clears the buffer;
    task/prompt transitions within the episode do not clear it.
    """

    def __init__(self, spec=None):
        spec = spec if spec is not None else VisualMemorySpec()
        if spec.image_keys != IMAGE_KEYS:
            raise ValueError("Sharpa runtime requires the four canonical views")
        self.spec = spec
        self.reset()

    def reset(self):
        self.episode_id = None
        self.frames = deque()
        self.last_index = None
        self.last_time = None

    def observe(self, sample, *, episode_id, frame_index, timestamp):
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("Explicit nonempty episode identity required")
        if type(frame_index) is not int or frame_index < 0 or not math.isfinite(timestamp):
            raise ValueError("Nonnegative acquisition frame index and finite timestamp required")
        frame = camera_frame(sample)
        if episode_id != self.episode_id:
            self.reset()
            self.episode_id = episode_id
        if self.last_index is not None:
            if frame_index <= self.last_index or timestamp <= self.last_time:
                raise ValueError("Observations must have strictly increasing frame IDs and timestamps")
            expected = (frame_index - self.last_index) / self.spec.fps
            if abs(timestamp - self.last_time - expected) > 0.5 / self.spec.fps:
                raise ValueError("Acquisition frame IDs and timestamps disagree with configured FPS")
        if self.frames:
            previous = self.frames[-1][1][0]
            if any(frame[0][key].shape != previous[key].shape for key in IMAGE_KEYS):
                raise ValueError("Camera resolution changed within an episode; reset with a new identity")
        oldest = frame_index - (self.spec.frames - 1) * self.spec.stride
        while self.frames and self.frames[0][0] < oldest:
            self.frames.popleft()
        stored = dict(self.frames)
        history = pack_history(self.spec, frame[0], lambda lag: stored.get(frame_index - lag))
        self.frames.append((frame_index, frame))
        self.last_index, self.last_time = frame_index, timestamp
        return {**sample, HISTORY: history}


class VisualMemoryPolicy:
    """Adapter around an offline policy; robot transport/control is intentionally separate."""

    def __init__(self, policy, spec, checkpoint_contract):
        expected = {**spec.contract(), "training_checkpoint": True}
        if checkpoint_contract != expected:
            raise ValueError("A matching visual-memory training checkpoint contract is required")
        self.policy = policy
        self.history = VisualHistory(spec)

    def reset(self):
        self.history.reset()
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def infer(self, obs, *, episode_id, frame_index, timestamp, **kwargs):
        inputs = self.history.observe(obs, episode_id=episode_id, frame_index=frame_index, timestamp=timestamp)
        return self.policy.infer(inputs, **kwargs)
