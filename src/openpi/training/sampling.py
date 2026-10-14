"""Versioned, finite sampling plans independent of worker state and prefetching.

Indices describe observation/action-window start frames, not every action target
inside a window. Consecutive shuffled epochs form one stream; batches may span
epoch boundaries, so no tail frames are dropped.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import math

import numpy as np
import torch

SAMPLER_VERSION = "epoch_shuffle_pcg64_v1"


WEIGHTED_SAMPLER_VERSION = "weighted_replacement_pcg64_v1"
PLAN_MODES = ("epoch_shuffle", "weighted_replacement")


@dataclass(frozen=True)
class SamplingGroup:
    """One disjoint episode group; weight controls its total sampling probability."""

    name: str
    episodes: tuple[int, ...]
    weight: float

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Sampling group name must be nonempty")
        if not self.episodes or any(type(i) is not int or i < 0 for i in self.episodes):
            raise ValueError("Sampling group episodes must be nonempty nonnegative integer IDs")
        if len(set(self.episodes)) != len(self.episodes):
            raise ValueError("Duplicate episodes in sampling group")
        if isinstance(self.weight, bool) or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Sampling group weight must be finite and positive")


def validate_groups(groups: tuple[SamplingGroup, ...]) -> None:
    if not groups:
        raise ValueError("weighted_replacement requires sampling_groups")
    names, episodes = set(), set()
    for group in groups:
        if not isinstance(group, SamplingGroup):
            raise ValueError("sampling_groups must contain SamplingGroup objects")
        if group.name in names:
            raise ValueError("Duplicate sampling group name")
        if episodes.intersection(group.episodes):
            raise ValueError("Sampling group episodes overlap")
        names.add(group.name)
        episodes.update(group.episodes)


def group_frame_weights(episode_ids, groups: tuple[SamplingGroup, ...]) -> np.ndarray:
    """Divide each group probability uniformly across its loaded start frames.

    All loaded episodes must belong to exactly one group. This keeps sampling
    separate from data selection and lets normalization cover the same dataset.
    """
    validate_groups(groups)
    episode_ids = np.asarray(episode_ids)
    if episode_ids.ndim != 1 or episode_ids.dtype.kind not in "iu" or not len(episode_ids):
        raise ValueError("Sampling requires a nonempty integer episode ID vector")
    expected = {episode for group in groups for episode in group.episodes}
    actual = set(map(int, episode_ids))
    if expected != actual:
        raise ValueError(
            f"Sampling groups must cover exactly the loaded episodes; "
            f"missing from groups={sorted(actual - expected)}, absent from data={sorted(expected - actual)}"
        )
    probabilities = np.asarray([group.weight for group in groups], dtype=np.float64)
    probabilities /= probabilities.max()
    probabilities /= probabilities.sum()
    if np.any(probabilities <= 0):
        raise ValueError("Sampling group weights have an unsupported dynamic range")
    weights = np.zeros(len(episode_ids), dtype=np.float64)
    for group, probability in zip(groups, probabilities, strict=True):
        mask = np.isin(episode_ids, group.episodes)
        weights[mask] = probability / mask.sum()
    return weights


def weighted_replacement_indices(weights, *, seed: int, num_samples: int) -> np.ndarray:
    """Fixed-seed, with-replacement frame plan with a budget-independent prefix."""
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.ndim != 1
        or not len(weights)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or num_samples <= 0
        or not 0 <= seed < 2**32
    ):
        raise ValueError("Require positive finite frame weights, budget and a uint32 seed")
    weights = weights / weights.max()
    cdf = np.cumsum(weights / weights.sum())
    cdf[-1] = 1.0
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = np.searchsorted(cdf, rng.random(num_samples), side="right").astype(np.int64)
    indices.flags.writeable = False
    return indices


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
        self._first_visits = None

    def unique_count(self, consumed: int) -> int:
        """Actual distinct start frames in a consumed prefix, including replacement draws."""
        if not 0 <= consumed <= len(self.indices):
            raise ValueError("Consumed sample count outside plan")
        if self._first_visits is None:
            self._first_visits = np.sort(np.unique(self.indices, return_index=True)[1])
        return int(np.searchsorted(self._first_visits, consumed))

    def seek(self, start: int) -> None:
        if not 0 <= start <= len(self.indices):
            raise ValueError(f"Sample offset {start} outside [0, {len(self.indices)}]")
        self._start = start

    def __iter__(self) -> Iterator[int]:
        return (int(index) for index in self.indices[self._start :])

    def __len__(self) -> int:
        return len(self.indices) - self._start
