"""Ordered CPU-only FLASH prefetch, with bounded per-worker video/frame caches.

The sample cursor is completed_step * global_batch, never worker read-ahead.
No padding, shuffling, image transforms or model inputs are changed here.
"""

from collections import OrderedDict
from contextlib import contextmanager
import multiprocessing
import os

import av
import jax
import numpy as np
import torch

from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.flash_native import CAMERAS
from openpi.training.flash_native import NativeDataset


def concatenate(items):
    return jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *items)


@contextmanager
def cpu_worker_environment():
    values = {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def worker_init(_worker_id):
    torch.set_num_threads(1)


class WorkerDataset:
    def __init__(self, dataset, *, decoder_limit=16, frame_limit=128):
        self.dataset = dataset
        self.decoder_limit = decoder_limit
        self.frame_limit = frame_limit
        self.decoders = OrderedDict()
        self.frames = OrderedDict()
        self.initialized = False

    def __getstate__(self):
        state = self.__dict__.copy()
        # Tokenizer and native arrays are built in the child, never GPU objects.
        import copy

        state["dataset"] = copy.copy(self.dataset)
        state["dataset"].tokenizer = None
        state["dataset"].numeric = {}
        state["dataset"].permutation = None
        state["decoders"], state["frames"] = OrderedDict(), OrderedDict()
        state["initialized"] = False
        return state

    def __len__(self):
        return int(self.dataset.ends[-1])

    def image(self, index, role):
        record = self.dataset.records[index["id"]]
        path = (
            record["root"]
            + "/"
            + record["info"]["video_path"].format(
                episode_chunk=record["episode"] // record["info"]["chunks_size"],
                episode_index=record["episode"],
                video_key=CAMERAS[role],
            )
        )
        if path not in self.decoders:
            while len(self.decoders) >= self.decoder_limit:
                _, old = self.decoders.popitem(last=False)
                old.close()
            self.decoders[path] = av.open(path)
            self.decoders[path].streams.video[0].codec_context.thread_count = 1
        self.decoders.move_to_end(path)
        container = self.decoders[path]
        stream = container.streams.video[0]
        container.seek(int(index["start"] / 30 / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            current = round(float(frame.time) * 30)
            if current < index["start"]:
                continue
            if current != index["start"]:
                raise ValueError("Current RGB and numeric frame mismatch")
            return frame.to_ndarray(format="rgb24")
        raise ValueError("Current video frame missing")

    def resized_image(self, index, role):
        key = (index["id"], index["start"], role)
        if key not in self.frames:
            self.frames[key] = NativeDataset.resize_image(self.image(index, role))
            while len(self.frames) > self.frame_limit:
                self.frames.popitem(last=False)
        self.frames.move_to_end(key)
        return self.frames[key]

    def __getitem__(self, index):
        if not self.initialized:
            self.dataset.tokenizer = PaligemmaTokenizer(max_len=self.dataset.config.model.max_token_len)
            self.initialized = True
        # Reuse the exact native observation transforms; only substitute cached IO.
        original = self.dataset.image
        original_resize = self.dataset.resize_image
        self.dataset.image = lambda i, role: self.resized_image(i, role)
        self.dataset.resize_image = lambda value: value
        try:
            return self.dataset.observation(index)
        finally:
            self.dataset.image = original
            self.dataset.resize_image = original_resize

    def close(self):
        for container in self.decoders.values():
            container.close()
        self.decoders.clear()
        self.frames.clear()


class WindowSampler:
    def __init__(self, dataset, start_step, end_step):
        self.dataset, self.start_step, self.end_step = dataset, start_step, end_step

    def __iter__(self):
        batch = self.dataset.config.batch_size
        for position in range(self.start_step * batch, self.end_step * batch):
            yield self.dataset.index(position)

    def __len__(self):
        return (self.end_step - self.start_step) * self.dataset.config.batch_size


class PrefetchedBatches:
    def __init__(self, dataset, start_step, end_step, *, prefetch_factor=2):
        if start_step < 0 or end_step < start_step:
            raise ValueError("Invalid completed-step range")
        if dataset.config.num_workers < 1:
            raise ValueError("Multiprocess FLASH loader requires num_workers >= 1")
        self.batch_size = dataset.config.batch_size
        workers = dataset.config.num_workers
        # Dispatch small tasks to all workers even when only one GPU batch is needed.
        chunk_size = max(1, self.batch_size // workers)
        while self.batch_size % chunk_size:
            chunk_size -= 1
        self.chunks = self.batch_size // chunk_size
        self.loader = torch.utils.data.DataLoader(
            WorkerDataset(dataset),
            batch_size=chunk_size,
            sampler=WindowSampler(dataset, start_step, end_step),
            num_workers=workers,
            multiprocessing_context=multiprocessing.get_context("spawn"),
            persistent_workers=True,
            prefetch_factor=prefetch_factor,
            collate_fn=concatenate,
            worker_init_fn=worker_init,
            pin_memory=False,
        )
        self.iterator = None
        self.remaining = end_step - start_step

    def __enter__(self):
        # Spawned children inherit CPU-only environment; parent settings restored.
        with cpu_worker_environment():
            self.iterator = iter(self.loader)
        return self

    def __next__(self):
        if not self.remaining:
            raise StopIteration
        result = concatenate([next(self.iterator) for _ in range(self.chunks)])
        self.remaining -= 1
        return result

    def __iter__(self):
        return self

    def __exit__(self, *_):
        # PyTorch has no public DataLoader.close; explicitly join persistent workers.
        if self.iterator is not None:
            self.iterator._shutdown_workers()  # noqa: SLF001
            self.iterator = None


def verify_input_migration(saved, current, root):
    """Explicit, narrow upgrade from the audited e64c9a41 input pipeline.

    Keep every model/optimizer/data/seed contract and objective function identical.
    The original checkpoint contract remains untouched for auditability.
    """
    import ast
    import subprocess

    from openpi.training.flash_native import digest

    before = {k: v for k, v in saved.items() if k != "training_code_sha256"}
    after = {k: v for k, v in current.items() if k != "training_code_sha256"}
    if before != after:
        raise ValueError("Input migration cannot change model/optimizer/data/RNG/batch contract")
    base = "e64c9a41ecb24494ea6c81e2df3dd4bbbfdcc45c"
    names = (
        "scripts/train_flash.py",
        "src/openpi/training/flash_native.py",
        "src/openpi/models/pi0_flash.py",
        "src/openpi/shared/flash_polynomial.py",
        "src/openpi/training/optimizer.py",
        "src/openpi/models/pi0.py",
        "src/openpi/models/gemma.py",
        "src/openpi/models/siglip.py",
        "scripts/train.py",
    )
    original = {
        name: subprocess.check_output(["git", "-C", str(root), "show", base + ":" + name], text=True) for name in names
    }
    if digest(original) != saved["training_code_sha256"]:
        raise ValueError("Checkpoint was not written by the audited prefetch migration base")
    for name in names[1:]:
        if original[name] != (root / name).read_text():
            raise ValueError(f"Input migration cannot change {name}")

    def objectives(source):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name in ("step", "evaluate_batch", "validation")
        }

    if objectives(original[names[0]]) != objectives((root / names[0]).read_text()):
        raise ValueError("Input migration cannot change loss/update/evaluation mathematics")
    return {
        "from_code_sha256": saved["training_code_sha256"],
        "to_code_sha256": current["training_code_sha256"],
        "audited_base_commit": base,
        "kind": "cpu_prefetch_only",
    }
