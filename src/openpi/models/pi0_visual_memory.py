"""Parameter-compatible sampled visual-prefix baseline for Pi05.

Current views retain all 256 tokens. Each historical view is spatially pooled
into a 4x4 grid, tagged with camera/relative time, and read by the usual prefix
attention. This is intentionally not an implementation of MEM's temporal ViT.
"""

from dataclasses import dataclass
from dataclasses import field

from flax import nnx
import jax.numpy as jnp

from openpi.models import pi0
from openpi.models import pi0_config
from openpi.subtasks.visual_memory import VisualMemorySpec


@dataclass(frozen=True)
class VisualMemoryConfig(pi0_config.Pi0Config):
    pi05: bool = True
    memory: VisualMemorySpec = field(default_factory=VisualMemorySpec)

    def __post_init__(self):
        object.__setattr__(self, "image_keys", self.memory.all_image_keys)
        super().__post_init__()
        if not self.pi05:
            raise ValueError("The visual-memory baseline requires Pi05")

    def create(self, rng):
        return Pi0VisualMemory(self, rngs=nnx.Rngs(rng))


def pool_tokens(tokens, grid):
    b, n, width = tokens.shape
    side = int(n**0.5)
    if side * side != n or side % grid:
        raise ValueError("Historical image token grid must be square and divisible by pool_grid")
    step = side // grid
    return tokens.reshape(b, grid, step, grid, step, width).mean(axis=(2, 4)).reshape(b, grid * grid, width)


def position_tag(position, width):
    # Relative time/camera tags have no trainable parameters and are zero at 0.
    half = width // 2
    frequency = 10000 ** (-jnp.arange(half, dtype=jnp.float32) / max(half - 1, 1))
    phase = position * frequency
    return jnp.concatenate([jnp.sin(phase), jnp.cos(phase) - 1]).astype(jnp.float32)


class Pi0VisualMemory(pi0.Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        self.memory = config.memory

    def embed_prefix(self, obs):
        if self.memory.frames == 1:
            return super().embed_prefix(obs)
        tokens, masks = [], []
        # Oldest first, with explicit camera and age tags. No historical state,
        # task index, future frame, or action target is fed to the prefix.
        for lag in self.memory.lags:
            for camera_index, camera in enumerate(self.memory.image_keys):
                key = self.memory.key(camera, lag)
                encoded, _ = self.PaliGemma.img(obs.images[key], train=False)
                pooled = pool_tokens(encoded, self.memory.pool_grid)
                tag = position_tag(-lag / self.memory.fps, pooled.shape[-1])
                tag += position_tag(camera_index + 1, pooled.shape[-1])
                tokens.append(pooled + tag.astype(pooled.dtype)[None, None, :])
                masks.append(jnp.broadcast_to(obs.image_masks[key][:, None], pooled.shape[:2]))
        for camera in self.memory.image_keys:
            encoded, _ = self.PaliGemma.img(obs.images[camera], train=False)
            tokens.append(encoded)
            masks.append(jnp.broadcast_to(obs.image_masks[camera][:, None], encoded.shape[:2]))
        if obs.tokenized_prompt is not None:
            tokens.append(self.PaliGemma.llm(obs.tokenized_prompt, method="embed"))
            masks.append(obs.tokenized_prompt_mask)
        tokens = jnp.concatenate(tokens, axis=1)
        return tokens, jnp.concatenate(masks, axis=1), jnp.zeros(tokens.shape[1], dtype=jnp.bool_)
