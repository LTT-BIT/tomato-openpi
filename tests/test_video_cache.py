"""Real PyAV equivalence, bounded resources, invalidation and spawn isolation."""

import multiprocessing
import os
import pickle
from unittest.mock import patch

import av
from lerobot.common.datasets import video_utils
import numpy as np
import pytest
import torch

from openpi.training.video_cache import CachedLeRobotDataset
from openpi.training.video_cache import PyAVCache
from openpi.training.video_cache import VideoCacheConfig
from openpi.training.video_cache import process_cache


def write_video(path, offset=0):
    rng = np.random.default_rng(offset)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.options = {"g": "4", "keyint_min": "4", "sc_threshold": "0"}
        for _ in range(16):
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "source.mp4"
    write_video(path)
    return path


@pytest.mark.parametrize("limit", [0, 32 * 32 * 3 * 4, 1024**2])
def test_random_seek_and_pixel_equivalence(video, limit):
    torch.set_num_threads(1)
    cache = PyAVCache(VideoCacheConfig(max_readers=2, max_bytes=limit))
    queries = [[1.4], [0.3], [0.1], [0.3], [1.5], [0.4], [0.39, 0.61, 0.49], [0.0], [1.4, 1.41]]
    try:
        for times in queries:
            expected = video_utils.decode_video_frames(video, times, 0.051, "pyav")
            result = cache.decode(video, times, 0.051)
            assert torch.equal(result, expected)
            result.zero_()  # A transform must not mutate retained uint8 frames.
            assert torch.equal(cache.decode(video, times, 0.051), expected)
            assert cache.bytes <= limit
        assert cache.opens == 1
        assert len(cache.readers) == 1
        if limit == 1024**2:
            assert cache.hits > 0
    finally:
        cache.close()
    assert cache.bytes == 0
    assert not cache.readers


def test_lru_reader_and_frame_eviction_and_replaced_file(video, tmp_path):
    other = tmp_path / "other.mp4"
    write_video(other, 10)
    cache = PyAVCache(VideoCacheConfig(max_readers=1, max_bytes=32 * 32 * 3 * 2))
    try:
        a = cache.decode(video, [0.1], 0.01)
        cache.decode(other, [0.1], 0.01)
        cache.decode(video, [0.1], 0.01)
        assert cache.opens == 3
        assert cache.evictions > 0
        assert len(cache.readers) == 1
        replacement = tmp_path / "new.mp4"
        write_video(replacement, 42)
        replacement.replace(video)
        b = cache.decode(video, [0.1], 0.01)
        assert not torch.equal(a, b)
        assert cache.opens == 4
        assert cache.bytes <= cache.config.max_bytes
        assert torch.equal(b, video_utils.decode_video_frames(video, [0.1], 0.01, "pyav"))
    finally:
        cache.close()


def test_tolerance_stays_strict_on_cached_frames(video):
    cache = PyAVCache()
    try:
        cache.decode(video, [0.3], 0.01)
        with pytest.raises(AssertionError, match="tolerance"):
            cache.decode(video, [0.25], 0.001)
    finally:
        cache.close()


def child_decode(path, config, queue):
    torch.set_num_threads(1)
    cache = process_cache(config)
    cache.decode(path, [0.2], 0.01)
    cache.decode(path, [0.2], 0.01)
    queue.put((os.getpid(), cache.pid, cache.opens, cache.hits))


def test_worker_cache_not_pickled_and_spawn_independent(video):
    dataset = CachedLeRobotDataset.__new__(CachedLeRobotDataset)
    dataset.video_cache_config = VideoCacheConfig(max_readers=2, max_bytes=1024**2)
    # Even after the parent opened a video, the dataset serializes no AV handles.
    cache = process_cache(dataset.video_cache_config)
    cache.decode(video, [0.2], 0.01)
    restored = pickle.loads(pickle.dumps(dataset))
    assert vars(restored) == vars(dataset)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    child = ctx.Process(target=child_decode, args=(str(video), dataset.video_cache_config, queue))
    child.start()
    child.join(timeout=45)
    try:
        assert child.exitcode == 0
        pid, owner, opens, hits = queue.get(timeout=2)
        assert pid == owner != os.getpid()
        assert opens == hits == 1
        assert cache.pid == os.getpid()
    finally:
        if child.is_alive():
            child.terminate()
            child.join()
        queue.close()
        cache.close()


def test_reader_closes_after_decode_error(video):
    cache = PyAVCache()
    with (
        patch.object(cache, "_select", side_effect=RuntimeError("decode error")),
        pytest.raises(RuntimeError, match="decode error"),
    ):
        cache.decode(video, [0.2], 0.01)
    assert not cache.readers
