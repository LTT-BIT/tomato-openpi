"""Optional image cache with per-stream fallback to the source video."""

from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path

import av
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_video_sha256(path):
    """Share the video digest between cache binding and the run's dataset audit."""
    path = Path(path).resolve()
    stat = path.stat()
    return _video_digest(path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=2048)
def _video_digest(path, *signature):
    digest = sha256(path)
    stat = path.stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != signature:
        raise ValueError(f"Source video changed while fingerprinting: {path}")
    return digest


class InvalidCacheFileError(ValueError):
    pass


class _PyAVReader:
    """Keep a bounded set of open streams and convert only the requested frames to RGB."""

    def __init__(self):
        self._containers = OrderedDict()

    def read(self, path, timestamps, tolerance):
        try:
            return self._read(path, timestamps, tolerance)
        except (av.FFmpegError, OSError, IndexError, TypeError, ValueError):
            container = self._containers.pop(path, None)
            if container is not None:
                container.close()
            return None

    def _read(self, path, timestamps, tolerance):
        if path not in self._containers:
            if len(self._containers) >= 8:
                _, old = self._containers.popitem(last=False)
                old.close()
            container = av.open(str(path))
            self._containers[path] = container
            container.streams.video[0].codec_context.thread_count = 1
        self._containers.move_to_end(path)
        container = self._containers[path]
        stream = container.streams.video[0]
        first, last = min(timestamps), max(timestamps)
        offset = round(max(first, 0.0) / stream.time_base)
        container.seek(offset, stream=stream, backward=True, any_frame=False)

        # Match LeRobot's PyAV seek, float32 nearest-timestamp comparison and first-match ties.
        query_times = np.asarray(timestamps, dtype=np.float32)
        distances = np.full(query_times.shape, np.inf, dtype=np.float32)
        closest = [None] * len(timestamps)
        for frame in container.decode(stream):
            timestamp = float(frame.pts * frame.time_base)
            distance = np.abs(query_times - np.float32(timestamp))
            for index in np.flatnonzero(distance < distances):
                closest[index] = frame
                distances[index] = distance[index]
            if timestamp >= last:
                break
        if not np.all(distances < tolerance):
            return None
        pixels = np.stack([frame.to_rgb().to_ndarray() for frame in closest])
        return torch.from_numpy(pixels).permute(0, 3, 1, 2).to(torch.float32) / 255


def read_cached_frames(cache_root, maps, key, entry, timestamps, fps, tolerance):
    indices = [round(ts * fps) for ts in timestamps]
    if not indices or any(
        not 0 <= index < entry["frames"] or abs(index / fps - ts) >= tolerance
        for index, ts in zip(indices, timestamps, strict=True)
    ):
        raise ValueError("Cached timestamp outside source timeline")
    if key not in maps:
        try:
            value = np.load(cache_root / entry["cache_file"], mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, EOFError) as error:
            raise InvalidCacheFileError(str(error)) from error
        if not isinstance(value, np.memmap) or value.shape != (entry["frames"], 224, 224, 3) or value.dtype != np.uint8:
            raise InvalidCacheFileError("Invalid RGB cache shape/dtype")
        maps[key] = value
    pixels = maps[key][indices]
    return torch.from_numpy(pixels[0] if len(indices) == 1 else pixels)


class CachedLeRobotDataset(LeRobotDataset):
    def __init__(self, *args, cache_root=None, reuse_pyav=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._reuse_pyav = reuse_pyav and self.video_backend == "pyav"
        self._source_reader = None
        self.cache_entries = {}
        self._cache_maps = {}
        self.cache_root = Path(cache_root).resolve() if cache_root else None
        if self.cache_root is None:
            return
        if self.image_transforms is not None:
            logging.info("Using source RGB with image transforms")
            return
        try:
            manifest = json.loads((self.cache_root / "manifest.json").read_text())
            entries = manifest["entries"]
            if manifest["schema"] != "rgb224_exact_pil_v2" or not isinstance(entries, list):
                raise ValueError("Unsupported RGB cache manifest")
            from openpi_client import image_tools

            from openpi.policies import fr3_sharpa_policy

            pipeline = {"parse_image": sha256(fr3_sharpa_policy.__file__), "resize": sha256(image_tools.__file__)}
            if manifest["pixel_pipeline"] != pipeline:
                raise ValueError("RGB cache pixel pipeline changed")
        except (OSError, KeyError, TypeError, ValueError) as error:
            logging.warning("Using source RGB because cache %s is unavailable: %s", self.cache_root, error)
            return

        expected = {(int(ep), cam) for ep in self.meta.episodes for cam in self.meta.video_keys}
        seen = set()
        for entry in entries:
            try:
                key = (entry["episode"], entry["camera"])
                if key not in expected:
                    continue
                if key in seen:
                    self.cache_entries.pop(key, None)
                    continue
                seen.add(key)
                path = (self.cache_root / entry["cache_file"]).resolve()
                if (
                    not path.is_relative_to(self.cache_root)
                    or path.suffix != ".npy"
                    or not path.is_file()
                    or entry["frames"] != self.meta.episodes[key[0]]["length"]
                    or entry.get("exact_reference_samples", 0) < 3
                ):
                    continue
                source = self.root / self.meta.get_video_file_path(*key)
                if source_video_sha256(source) == entry["source_sha256"]:
                    self.cache_entries[key] = {**entry, "cache_file": path}
            except (OSError, KeyError, TypeError, ValueError):
                continue
        logging.info(
            "RGB cache %s: %d streams matched; %d use source video",
            self.cache_root,
            len(self.cache_entries),
            len(expected) - len(self.cache_entries),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache_maps"] = {}
        state["_source_reader"] = None
        return state

    def _query_videos(self, query_timestamps, ep_idx):
        result = {}
        fallback = {}
        for camera, times in query_timestamps.items():
            key = (int(ep_idx), camera)
            entry = self.cache_entries.get(key)
            if entry is not None:
                try:
                    result[camera] = read_cached_frames(
                        self.cache_root, self._cache_maps, key, entry, times, self.fps, self.tolerance_s
                    )
                    continue
                except InvalidCacheFileError as error:
                    self.cache_entries.pop(key)
                    self._cache_maps.pop(key, None)
                    logging.warning("Using source RGB for episode %s camera %s: %s", ep_idx, camera, error)
            if self._reuse_pyav:
                if self._source_reader is None:
                    self._source_reader = _PyAVReader()
                path = self.root / self.meta.get_video_file_path(ep_idx, camera)
                frames = self._source_reader.read(path, times, self.tolerance_s)
                if frames is not None:
                    result[camera] = frames.squeeze(0)
                    continue
            fallback[camera] = times
        if fallback:
            result.update(super()._query_videos(fallback, ep_idx))
        return result
