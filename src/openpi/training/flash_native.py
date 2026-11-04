"""Local LeRobot causal FLASH windows and immutable train-only statistics.

No SSH, video copying, padded targets, future observations or implicit history.
Sampler position is a pure function of completed updates and global batch size.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import av
import jax
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F  # noqa: N812

from openpi.models import model as model_lib
from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.shared import normalize
from openpi.shared.local_storage import new_artifact_directory

CAMERAS = {
    "base_0_rgb": "observation.images.cam0",
    "left_wrist_0_rgb": "observation.images.cam2",
    "right_wrist_0_rgb": "observation.images.cam1",
}


def is_flash(config):
    return getattr(config.model, "poly_degree", None) == 6 and hasattr(config.model, "freeze_vlm")


def asset_directory(config):
    return Path(config.data.assets.assets_dir or config.assets_dirs) / config.data.assets.asset_id


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_numeric(record):
    info = record["info"]
    names = info["features"]["action"]["names"]
    if not all(n.endswith((".position", ".command")) for n in names):
        raise ValueError("FLASH requires commanded absolute joint positions")
    action_names = [n.rsplit(".", 1)[0] for n in names]
    state_names = info["features"]["observation.state"]["names"]
    if len(action_names) != 58 or set(action_names) != set(ORDER):
        raise ValueError("FLASH joint names/order do not match Sharpa protocol")
    table = pq.read_table(
        record["path"], columns=["action", "observation.state", "frame_index", "timestamp", "task_index"]
    )
    actions = np.asarray(table["action"].to_pylist(), np.float32)[:, [action_names.index(n) for n in ORDER]]
    states = np.asarray(table["observation.state"].to_pylist(), np.float32)[
        :, [state_names.index(n + ".position") for n in ORDER]
    ]
    frames = np.asarray(table["frame_index"].to_pylist()).reshape(-1)
    times = np.asarray(table["timestamp"].to_pylist()).reshape(-1)
    if len(actions) != record["rows"] or not np.array_equal(frames, np.arange(len(actions))):
        raise ValueError("Episode row count/frame alignment mismatch")
    if (
        not np.isfinite(actions).all()
        or not np.isfinite(states).all()
        or not np.allclose(np.diff(times), 1 / 30, atol=2e-4)
    ):
        raise ValueError("Nonfinite or non-30Hz trajectory")
    tasks = np.asarray(table["task_index"].to_pylist()).reshape(-1)
    return actions, states, tasks


def inventory(config):
    records = []
    for spec in config.data.lerobot_datasets:
        root = Path(spec.root).resolve()
        info = json.loads((root / "meta/info.json").read_text())
        if info["fps"] != 30 or info["features"]["action"]["shape"] != [58]:
            raise ValueError("FLASH requires normal 30Hz, 58D LeRobot data")
        tasks = {
            str(t["task_index"]): t["task"]
            for t in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())
        }
        if not tasks or any(not t.isascii() or not t.strip() for t in tasks.values()):
            raise ValueError("Nonempty English task prompts required")
        for ep in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines()):
            if not ep.get("source_recording_id"):
                raise ValueError("Source-level split requires original source_recording_id")
            i = ep["episode_index"]
            path = root / info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i)
            stat = path.stat()
            if ep["length"] < 59:
                continue
            records.append(
                {
                    "id": digest([str(root), i])[:16],
                    "root": str(root),
                    "episode": i,
                    "rows": ep["length"],
                    "source_recording_id": ep["source_recording_id"],
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "info": info,
                    "tasks": tasks,
                }
            )
    if not records:
        raise ValueError("No legal causal 8-history/54-target windows")
    return records


def data_policy(config):
    policy = config.policy_metadata.get("data_policy", "source_holdout")
    if policy not in ("all_sources", "source_holdout"):
        raise ValueError(f"Unsupported FLASH data policy: {policy}")
    return policy


def prepare(config, *, batch_size=512):
    del batch_size
    records = inventory(config)
    groups = sorted({r["source_recording_id"] for r in records})
    policy = data_policy(config)
    if policy == "source_holdout" and len(groups) < 2:
        raise ValueError("At least two independent original recordings required")
    shuffled = np.random.default_rng(config.data_seed).permutation(groups)
    validation = set(shuffled[: max(1, round(len(groups) * 0.2))]) if policy == "source_holdout" else set()
    train = [r for r in records if r["source_recording_id"] not in validation]
    chunks, states = [], []
    for record in train:
        actions, state, _ = read_numeric(record)
        chunks.append(actions)
        states.append(state)
    stats = {}
    for name, values in (("actions", chunks), ("state", states)):
        array = np.concatenate(values)
        lo, hi = np.quantile(array, [0.01, 0.99], axis=0)
        stats[name] = normalize.NormStats(mean=array.mean(axis=0), std=array.std(axis=0), q01=lo, q99=hi)
    contract = {
        "version": 1,
        "action_semantics": "absolute",
        "horizon": 50,
        "history": 8,
        "target": 54,
        "K": 6,
        "stride": 1,
        "data_seed": config.data_seed,
        "roots": [str(Path(s.root).resolve()) for s in config.data.lerobot_datasets],
        "joint_names": list(ORDER),
        "fit_scope": "train_episodes_only",
        "data_policy": policy,
        "evaluation_scope": "held_out_sources" if validation else "training_data",
    }
    manifest = {
        "contract": contract,
        "records": records,
        "train_sources": sorted(set(groups) - validation),
        "validation_sources": sorted(validation),
    }
    manifest["sha256"] = digest(manifest)
    # Full-data runs monitor training windows; they have no held-out validation.
    evaluation_sources = validation if validation else set(groups)
    # Fix evaluation windows once, before any model is trained.
    windows = [
        {"id": r["id"], "start": int(t)}
        for r in records
        if r["source_recording_id"] in evaluation_sources
        for t in sorted(set(np.linspace(7, r["rows"] - 52, 3, dtype=int).tolist()))
    ]
    chosen = np.random.default_rng(config.data_seed + 1).permutation(len(windows))[:64]
    plan = [windows[int(i)] for i in chosen]
    with new_artifact_directory(asset_directory(config)) as staging:
        normalize.save(staging, stats)
        (staging / "flash_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        (staging / "flash_validation.json").write_text(json.dumps(plan, indent=2))
        seal = {
            "manifest_sha256": manifest["sha256"],
            "norm_sha256": hashlib.sha256((staging / "norm_stats.json").read_bytes()).hexdigest(),
            "validation_sha256": digest(plan),
        }
        (staging / "flash_seal.json").write_text(json.dumps(seal, indent=2))
    return preflight(config)


def preflight(config, *, require_norm=True):
    if config.model.rtc_max_delay or getattr(config.model, "freeze_vlm", False) or config.data.extra_delta_transform:
        raise ValueError("Standard FLASH requires FullFT Absolute without RTC")
    if config.sampling_mode != "epoch_shuffle" or config.sampling_groups:
        raise ValueError("FLASH requires deterministic epoch_shuffle, without weighted groups")
    policy = data_policy(config)
    records = inventory(config)
    result = {
        "episodes": len(records),
        "frames": sum(r["rows"] for r in records),
        "legal_windows": sum(r["rows"] - 58 for r in records),
        "source_count": len({r["source_recording_id"] for r in records}),
        "data_policy": policy,
        "evaluation_scope": "training_data" if policy == "all_sources" else "held_out_sources",
        "training_action_semantics": "absolute",
        "K": 6,
        "stride": 1,
    }
    if not require_norm:
        return result
    directory = asset_directory(config)
    manifest = json.loads((directory / "flash_manifest.json").read_text())
    seal = json.loads((directory / "flash_seal.json").read_text())
    unsigned = {k: v for k, v in manifest.items() if k != "sha256"}
    if digest(unsigned) != manifest["sha256"] or seal["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("FLASH manifest fingerprint mismatch")
    if manifest["contract"].get("data_policy", "source_holdout") != policy:
        raise ValueError("FLASH data policy changed; use a new asset_id")
    if manifest["records"] != records or manifest["contract"]["data_seed"] != config.data_seed:
        raise ValueError("Data/source order/content or split seed changed; use a new asset_id")
    if seal["norm_sha256"] != hashlib.sha256((directory / "norm_stats.json").read_bytes()).hexdigest():
        raise ValueError("FLASH normalization fingerprint mismatch")
    plan = json.loads((directory / "flash_validation.json").read_text())
    if digest(plan) != seal["validation_sha256"]:
        raise ValueError("FLASH evaluation windows changed")
    if set(manifest["train_sources"]) & set(manifest["validation_sources"]):
        raise ValueError("Recording leakage across train/validation")
    train_sources, val_sources = set(manifest["train_sources"]), set(manifest["validation_sources"])
    sources = {r["source_recording_id"] for r in records}
    if train_sources | val_sources != sources or not train_sources:
        raise ValueError("FLASH source allocation does not cover the dataset")
    if policy == "all_sources" and (val_sources or train_sources != sources):
        raise ValueError("Full-data FLASH requires every source in training and no holdout")
    if policy == "source_holdout" and not val_sources:
        raise ValueError("Source holdout requires independent validation sources")
    lookup = {r["id"]: r for r in records}
    evaluation_sources = train_sources if policy == "all_sources" else val_sources
    if not plan or any(
        w["id"] not in lookup
        or lookup[w["id"]]["source_recording_id"] not in evaluation_sources
        or not 7 <= w["start"] <= lookup[w["id"]]["rows"] - 52
        for w in plan
    ):
        raise ValueError("FLASH evaluation windows do not match their declared data scope")
    result.update(
        manifest_sha256=manifest["sha256"],
        normalization_sha256=seal["norm_sha256"],
        validation_sha256=seal["validation_sha256"],
        train_sources=len(manifest["train_sources"]),
        validation_sources=len(manifest["validation_sources"]),
    )
    return result


def bind_stats(config):
    directory = asset_directory(config)
    if not (directory / "norm_stats.json").exists():
        return config
    stats = normalize.load(directory)["actions"]
    if config.policy_metadata.get("algorithm") == "flash_reference":
        return replace(
            config,
            policy_metadata={
                **config.policy_metadata,
                "action_q01": stats.q01.tolist(),
                "action_q99": stats.q99.tolist(),
            },
        )
    return replace(
        config,
        model=replace(
            config.model, action_q01=tuple(float(x) for x in stats.q01), action_q99=tuple(float(x) for x in stats.q99)
        ),
    )


class NativeDataset:
    def __init__(self, config, *, tokenizer=None):
        preflight(config)
        torch.set_num_threads(1)
        self.config = config
        self.directory = asset_directory(config)
        self.manifest = json.loads((self.directory / "flash_manifest.json").read_text())
        self.stats = normalize.load(self.directory)
        self.validation = json.loads((self.directory / "flash_validation.json").read_text())
        self.records = {r["id"]: r for r in self.manifest["records"]}
        self.train = [r for r in self.records.values() if r["source_recording_id"] in self.manifest["train_sources"]]
        self.ends = np.cumsum([r["rows"] - 58 for r in self.train])
        self.numeric = {}
        self.permutation = None
        self.epoch = None
        self.tokenizer = tokenizer

    def index(self, position):
        epoch, offset = divmod(position, int(self.ends[-1]))
        if epoch != self.epoch:
            self.permutation = np.random.default_rng(
                np.random.SeedSequence([self.config.data_seed, epoch])
            ).permutation(int(self.ends[-1]))
            self.epoch = epoch
        value = int(self.permutation[offset])
        i = int(np.searchsorted(self.ends, value, side="right"))
        start = value - (int(self.ends[i - 1]) if i else 0) + 7
        return {"id": self.train[i]["id"], "start": start}

    def window(self, index):
        ident, start = index["id"], index["start"]
        record = self.records[ident]
        if not 7 <= start <= record["rows"] - 52:
            raise ValueError("History/target outside episode")
        if ident not in self.numeric:
            self.numeric[ident] = read_numeric(record)
        actions, states, tasks = self.numeric[ident]
        return (
            states[start - 7 : start + 1],
            actions[start - 2 : start + 52],
            states[start],
            str(record["tasks"][str(int(tasks[start]))]),
        )

    def image(self, index, role):
        record = self.records[index["id"]]
        info, i = record["info"], record["episode"]
        path = Path(record["root"]) / info["video_path"].format(
            episode_chunk=i // info["chunks_size"], episode_index=i, video_key=CAMERAS[role]
        )
        with av.open(str(path)) as container:
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

    @staticmethod
    def resize_image(rgb):
        # Match float linear resize-with-pad on CPU; no JAX CPU backend is needed
        # when training explicitly requires JAX_PLATFORMS=cuda.
        pixels = rgb.astype(np.float32) / 127.5 - 1
        height, width = pixels.shape[:2]
        ratio = max(width / 224, height / 224)
        resized_h, resized_w = int(height / ratio), int(width / ratio)
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)[None]
        value = (
            F.interpolate(tensor, size=(resized_h, resized_w), mode="bilinear", align_corners=False, antialias=True)[0]
            .permute(1, 2, 0)
            .numpy()
        )
        top, left = (224 - resized_h) // 2, (224 - resized_w) // 2
        return np.pad(
            value, ((top, 224 - resized_h - top), (left, 224 - resized_w - left), (0, 0)), constant_values=-1
        )[None]

    def observation(self, index):
        history, target, state, prompt = self.window(index)
        stat = self.stats["state"]
        normalized = (state - stat.q01) / (stat.q99 - stat.q01 + 1e-6) * 2 - 1
        tokens, mask = self.tokenizer.tokenize(prompt, normalized)
        images = {key: self.resize_image(self.image(index, key)) for key in CAMERAS}
        obs = model_lib.Observation(
            images=images,
            image_masks={k: np.ones(1, bool) for k in images},
            state=np.asarray(normalized[None], np.float32),
            tokenized_prompt=np.asarray(tokens[None], np.int32),
            tokenized_prompt_mask=np.asarray(mask[None]),
        )
        return obs, history[None], target[None]

    def batch(self, indices):
        with ThreadPoolExecutor(max_workers=min(max(self.config.num_workers, 1), len(indices))) as pool:
            values = list(pool.map(self.observation, indices))
        return jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *values)

    def data_config(self):
        import shutil
        from types import SimpleNamespace

        def extra(destination):
            for name in ("flash_manifest.json", "flash_validation.json", "flash_seal.json"):
                shutil.copyfile(self.directory / name, Path(destination) / name)

        return SimpleNamespace(
            norm_stats=self.stats, asset_id=self.config.data.assets.asset_id, save_extra_assets=extra
        )
