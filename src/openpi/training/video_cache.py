"""Worker-local bounded PyAV readers and decoded seek segments.

The decoding/nearest-PTS rule matches pinned LeRobot's PyAV backend. No image
resize, source rewrite, GPU decoding, or change to the training sampler occurs.
Only handles/tensors created in this process are retained, never in a dataset.
"""

import atexit
from collections import OrderedDict
from dataclasses import dataclass
import os
from pathlib import Path
import resource

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
import torchvision


@dataclass(frozen=True)
class VideoCacheConfig:
    max_readers: int = 128
    max_bytes: int = 512 * 1024**2

    def __post_init__(self):
        if type(self.max_readers) is not int or self.max_readers < 1:
            raise ValueError("max_readers must be a positive integer")
        if type(self.max_bytes) is not int or self.max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")


class PyAVCache:
    def __init__(self, config=None):
        config = config or VideoCacheConfig()
        self.config = config
        self.pid = os.getpid()
        limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        self.reader_limit = (
            min(config.max_readers, max(1, limit - 128)) if limit != resource.RLIM_INFINITY else config.max_readers
        )
        self.readers = OrderedDict()
        self.segments = OrderedDict()
        self.bytes = 0
        self.opens = self.hits = self.misses = self.evictions = 0

    def close(self):
        for reader in self.readers.values():
            reader.container.close()
        self.readers.clear()
        self.segments.clear()
        self.bytes = 0

    def _remove_segment(self, key):
        _, frames, size = self.segments.pop(key)
        self.bytes -= size
        del frames
        self.evictions += 1

    def decode(self, path, timestamps, tolerance_s):
        if os.getpid() != self.pid:
            # Spawn is used by training; also protect callers which use fork.
            self.close()
            self.pid = os.getpid()
        path = Path(path).resolve()
        stat = path.stat()
        identity = (str(path), stat.st_size, stat.st_mtime_ns)
        first, last = min(timestamps), max(timestamps)
        # A file replaced in place must never use stale frames or a stale handle.
        for key in list(self.readers):
            if key[0] == str(path) and key != identity:
                self.readers.pop(key).container.close()
        for key in list(self.segments):
            if key[0][0] == str(path) and key[0] != identity:
                self._remove_segment(key)
        for key in reversed(self.segments):
            pts, frames, _ = self.segments[key]
            if key[0] == identity and pts[0] <= first and pts[-1] >= last:
                self.hits += 1
                self.segments.move_to_end(key)
                return self._select(path, timestamps, tolerance_s, pts, frames)
        self.misses += 1
        if identity not in self.readers:
            while len(self.readers) >= self.reader_limit:
                _, reader = self.readers.popitem(last=False)
                reader.container.close()
            torchvision.set_video_backend("pyav")
            reader = torchvision.io.VideoReader(str(path), "video")
            # Bound CPU decoder threads as well as reader count in each worker.
            reader.container.streams.video[0].codec_context.thread_count = 1
            self.readers[identity] = reader
            self.opens += 1
        reader = self.readers[identity]
        self.readers.move_to_end(identity)
        pts, frames = [], []
        try:
            reader.seek(first, keyframes_only=True)
            for frame in reader:
                pts.append(frame["pts"])
                frames.append(frame["data"])
                if pts[-1] >= last:
                    break
            result = self._select(path, timestamps, tolerance_s, pts, frames)
        except BaseException:
            self.readers.pop(identity).container.close()
            raise
        size = sum(frame.numel() * frame.element_size() for frame in frames)
        if size <= self.config.max_bytes:
            # Segments from the same keyframe grow as later times are requested.
            key = (identity, pts[0])
            if key in self.segments:
                self._remove_segment(key)
            while self.segments and self.bytes + size > self.config.max_bytes:
                self._remove_segment(next(iter(self.segments)))
            if self.config.max_bytes:
                self.segments[key] = (pts, frames, size)
                self.bytes += size
        return result

    @staticmethod
    def _select(path, timestamps, tolerance_s, pts, frames):
        if not pts:
            raise ValueError(f"No video frames decoded from {path}")
        # Keep LeRobot's float32 timestamp rounding, first-min tie break, strict
        # tolerance and float32 RGB conversion. Return a fresh tensor on every hit.
        distance = torch.cdist(torch.tensor(timestamps)[:, None], torch.tensor(pts)[:, None], p=1)
        minimum, indices = distance.min(1)
        if not (minimum < tolerance_s).all():
            raise AssertionError(f"Video timestamp tolerance violated: {path}, {minimum=}, {tolerance_s=}")
        return torch.stack([frames[index] for index in indices]).to(torch.float32) / 255


_process_cache = None


def process_cache(config):
    global _process_cache  # noqa: PLW0603 -- one cache shared across datasets, never serialized
    if _process_cache is None or _process_cache.pid != os.getpid() or _process_cache.config != config:
        if _process_cache is not None:
            _process_cache.close()
        _process_cache = PyAVCache(config)
    return _process_cache


def _close_process_cache():
    if _process_cache is not None:
        _process_cache.close()


atexit.register(_close_process_cache)


class CachedLeRobotDataset(LeRobotDataset):
    """Same Arrow tables/action queries, sharing one bounded cache per worker."""

    video_cache_config = VideoCacheConfig()

    def _query_videos(self, query_timestamps, ep_idx):
        if self.video_backend != "pyav":
            return super()._query_videos(query_timestamps, ep_idx)
        cache = process_cache(self.video_cache_config)
        return {
            key: cache.decode(self.root / self.meta.get_video_file_path(ep_idx, key), times, self.tolerance_s).squeeze(
                0
            )
            for key, times in query_timestamps.items()
        }
