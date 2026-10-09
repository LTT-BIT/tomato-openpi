"""Versioned, finite sampling plans independent of worker state and prefetching.

Indices describe observation/action-window start frames, not every action target
inside a window. Consecutive shuffled epochs form one stream; batches may span
epoch boundaries, so no tail frames are dropped.
"""

from collections.abc import Iterator

import numpy as np
import torch

SAMPLER_VERSION = "epoch_shuffle_pcg64_v1"


def epoch_shuffle_indices(size: int, *, seed: int, num_samples: int) -> np.ndarray:
    """Build a reproducible plan; extending the budget preserves its prefix."""
    if size <= 0 or num_samples <= 0 or not 0 <= seed < 2**32:
        raise ValueError("Require size > 0, num_samples > 0 and a uint32 seed")
    indices = np.empty(num_samples, dtype=np.int64)
    for epoch, start in enumerate(range(0, num_samples, size)):
        # Use an explicit generator and seed composition, not Python's salted hash.
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, epoch])))
        count = min(size, num_samples - start)
        indices[start : start + count] = rng.permutation(size)[:count]
    indices.flags.writeable = False
    return indices


class PlanSampler(torch.utils.data.Sampler[int]):
    """Seek by consumed samples, never by the worker's prefetched cursor.

    Call seek only before creating the training iterator. Each iterator captures
    its own start offset so a previous iterator cannot move the new cursor.
    """

    def __init__(self, indices: np.ndarray):
        self.indices = indices
        self._start = 0

    def seek(self, start: int) -> None:
        if not 0 <= start <= len(self.indices):
            raise ValueError(f"Sample offset {start} outside [0, {len(self.indices)}]")
        self._start = start

    def __iter__(self) -> Iterator[int]:
        return (int(index) for index in self.indices[self._start :])

    def __len__(self) -> int:
        return len(self.indices) - self._start
