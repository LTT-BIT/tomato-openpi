"""Audit all119 recordings and real historical inputs without model training.

Writes only a fresh result JSON. The suggested split is not a training config;
its package prefixes are export-batch labels, not verified acquisition dates.
"""

import argparse
from collections import Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
import pyarrow.parquet as pq
import torch

from openpi import transforms
from openpi.policies import fr3_sharpa_policy
from openpi.subtasks import tomato_in_hand as contract
from openpi.subtasks.visual_memory import HISTORY
from openpi.subtasks.visual_memory import IMAGE_KEYS
from openpi.subtasks.visual_memory import VisualHistory
from openpi.training import data_loader
from openpi.training.frame_cache import read_cached_frames
from openpi.training.reproducibility import file_sha256
from openpi.training.visual_memory import VisualMemoryDataset
from scripts.check_visual_memory import MODULE
from scripts.run_training import load_experiment
from scripts.run_training import preflight


def candidate_split(rows):
    assignments = {"1112": "train", "1113": "development", "1114": "test"}
    result = {name: [] for name in assignments.values()}
    packages = defaultdict(set)
    for row in rows:
        name = Path(row["source_dataset"]).name
        part = assignments[name.split("_")[0]]
        packages[part].add(row["source_dataset"])
        result[part].append({k: row[k] for k in ("episode_index", "recording_group", "source_dataset", "length")})
    identities = [{r["recording_group"] for r in result[part]} for part in result]
    if len(set.union(*identities)) != len(rows) or any(
        a & b for i, a in enumerate(identities) for b in identities[i + 1 :]
    ):
        raise ValueError("Recording identity crosses proposed partitions")
    return {
        "status": "candidate_only_not_activated",
        "grouping": "keep each of six source packages whole; use export-batch prefixes, not acquisition dates",
        "requires_new_training_from_unexposed_base": True,
        "requires_train_only_normalization": True,
        "unseen_scene_or_object_identity_verified": False,
        "summary": {
            key: {"recordings": len(value), "frames": sum(r["length"] for r in value), "packages": len(packages[key])}
            for key, value in result.items()
        },
        "assignments": result,
    }


def duplicate_groups(rows, keys):
    seen = defaultdict(list)
    for row in rows:
        fingerprints = []
        for camera in keys:
            suffix = f"/{camera}/episode_{row['source_episode_index']:06d}.mp4"
            values = [v for k, v in row["source_sha256"].items() if k.endswith(suffix)]
            if len(values) != 1:
                raise ValueError("One source video fingerprint per camera required")
            fingerprints.append(values[0])
        seen[tuple(fingerprints)].append(row["episode_index"])
    return [indices for indices in seen.values() if len(indices) > 1]


def verify_migrated_cache(cache_root):
    """Full byte checks for an audit-only reader; never rewrite production acceptance."""
    cache_root = cache_root.resolve(strict=True)
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    accepted_path = (
        Path(__file__).resolve().parents[1] / "training_configs/experiments/tomato_sharpa/subtasks/records/"
        "in_hand119_server28_preparation_20261116_r1.json"
    )
    expected = json.loads(accepted_path.read_text())["cache_manifest_sha256"]
    if file_sha256(manifest_path) != expected:
        raise ValueError("Cache manifest differs from the previously accepted original")
    pipeline = {
        "parse_image": file_sha256(Path(fr3_sharpa_policy.__file__)),
        "resize": file_sha256(Path(image_tools.__file__)),
    }
    if (
        manifest.get("schema") != "rgb224_exact_pil_v2"
        or not manifest.get("complete")
        or manifest["pixel_pipeline"] != pipeline
    ):
        raise ValueError("Unsupported cache or pixel pipeline")
    started = time.monotonic()

    def verify(entry):
        path = (cache_root / entry["cache_file"]).resolve(strict=True)
        if not path.is_relative_to(cache_root):
            raise ValueError("Cache path escapes its root")
        before = path.stat()
        if before.st_size != entry["bytes"] or file_sha256(path) != entry["cache_sha256"]:
            raise ValueError("Cache bytes differ from original acceptance: " + entry["cache_file"])
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Cache changed while hashing")
        return (before.st_size, before.st_mtime_ns), before.st_mtime_ns != entry["mtime_ns"]

    verified, mtime_changes = {}, 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, (entry, (stat, changed)) in enumerate(
            zip(manifest["entries"], pool.map(verify, manifest["entries"]), strict=True)
        ):
            verified[entry["cache_file"]] = stat
            mtime_changes += changed
            if (index + 1) % 24 == 0:
                print(f"Cache SHA256 verified {index + 1}/{len(manifest['entries'])}", flush=True)
    return (
        manifest,
        verified,
        {
            "manifest_sha256": expected,
            "files_sha256_verified": len(verified),
            "bytes_sha256_verified": sum(e["bytes"] for e in manifest["entries"]),
            "mtime_mismatches": mtime_changes,
            "seconds": round(time.monotonic() - started, 3),
            "original_manifest_modified": False,
            "production_stat_guard_passed": mtime_changes == 0,
            "reader_scope": "audit_only_full_hash_verified_not_training_acceptance",
        },
    )


class HashVerifiedAuditReader(LeRobotDataset):
    """Use the original cache read code after full hashes, retaining a fresh stat guard."""

    def __init__(self, *args, cache_root, manifest, verified, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_root = cache_root.resolve()
        self.cache_entries = {(e["episode"], e["camera"]): e for e in manifest["entries"]}
        self.verified = verified
        self._cache_maps = {}
        for relative, digest in manifest["source_metadata"].items():
            if file_sha256(self.root / relative) != digest:
                raise ValueError("Cache source metadata differs")
        if set(self.cache_entries) != {(int(ep), cam) for ep in self.meta.episodes for cam in self.meta.video_keys}:
            raise ValueError("Cache does not cover all source videos")
        for (ep, cam), entry in self.cache_entries.items():
            if (
                entry["frames"] != self.meta.episodes[ep]["length"]
                or file_sha256(self.root / self.meta.get_video_file_path(ep, cam)) != entry["source_sha256"]
            ):
                raise ValueError("Cache source video or episode length changed")

    def _query_videos(self, query_timestamps, ep_idx):
        frames = {}
        for camera, timestamps in query_timestamps.items():
            key = (int(ep_idx), camera)
            entry = self.cache_entries[key]
            stat = (self.cache_root / entry["cache_file"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != self.verified[entry["cache_file"]]:
                raise ValueError("Cache changed after full hash validation")
            frames[camera] = read_cached_frames(
                self.cache_root, self._cache_maps, key, entry, timestamps, self.fps, self.tolerance_s
            )
        return frames


def audit(report_path, checkpoint_run, *, verify_migration=False):
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("This input audit must run on CPU")
    if report_path.exists():
        raise FileExistsError("Use a new report path")
    torch.set_num_threads(1)
    cfg = load_experiment(MODULE)
    baseline_inputs = preflight(cfg)
    root = Path(cfg.data.lerobot_datasets[0].root).resolve()
    bundle = root.parent
    manifest = json.loads((bundle / "bundle.json").read_text())
    selection = json.loads((bundle / "selection.json").read_text())
    rows = selection["episodes"]
    if len(rows) != 119 or sum(r["length"] for r in rows) != 357970:
        raise ValueError("Frozen all119 identity changed")
    verified_files = 0
    for relative, expected in manifest["files"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle) or file_sha256(path) != expected:
            raise ValueError(f"Prepared bundle changed: {relative}")
        verified_files += 1
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    cache_audit = None
    if verify_migration:
        cache_root = Path(os.environ["PI05_RGB_FRAME_CACHE"])
        cache_manifest, verified, cache_audit = verify_migrated_cache(cache_root)
        spec = data.lerobot_datasets[0]
        dataset = HashVerifiedAuditReader(
            spec.repo_id,
            root=spec.root,
            video_backend=data.video_backend,
            delta_timestamps={"action": [i / 30 for i in range(50)]},
            cache_root=cache_root,
            manifest=cache_manifest,
            verified=verified,
        )
        history = VisualMemoryDataset(dataset, cfg.model.memory, data.history_mask_key)
        source = data_loader.TransformedDataset(history, [transforms.PromptFromLeRobotTask(dataset.meta.tasks)])
    else:
        source = data_loader.create_torch_dataset(
            data, cfg.model.action_horizon, cfg.model, sampling_mode="epoch_shuffle"
        )
    raw = source
    while isinstance(raw, data_loader.TransformedDataset):
        raw = raw._dataset  # noqa: SLF001
    transformed = data_loader.transform_dataset(source, data)
    memory = cfg.model.memory
    offset = frames_checked = history_decisions = history_pixels = prompt_checks = 0
    masked_slots = valid_slots = 0
    annotation_keys, segment_keys = Counter(), Counter()
    physical_digests = defaultdict(list)
    physical_source_columns = 0
    for row in rows:
        episode, length = row["episode_index"], row["length"]
        original_annotation = Path(row["annotation"])
        if file_sha256(original_annotation) != row["annotation_sha256"]:
            raise ValueError("Source annotation changed")
        document = json.loads(original_annotation.read_text())
        segments = contract.reviewed_segments(document, length)
        if segments[1]["text"] != segments[3]["text"]:
            raise ValueError("Repeated harvest prompt contract changed")
        annotation_keys.update(document.keys())
        for segment in segments:
            segment_keys.update(segment.keys())
        table = pq.read_table(root / f"data/chunk-000/episode_{episode:06d}.parquet")
        upstream = pq.read_table(
            Path(row["source_dataset"]) / f"data/chunk-000/episode_{row['source_episode_index']:06d}.parquet"
        )
        for key in upstream.column_names:
            if key not in ("index", "episode_index", "task_index"):
                if not upstream[key].equals(table[key]):
                    raise ValueError(f"Physical source column changed: {episode}/{key}")
                physical_source_columns += 1
        state = np.asarray(table["observation.state"].to_pylist(), np.float32)
        actions = np.asarray(table["action"].to_pylist(), np.float32)
        if state.shape != (length, 116) or actions.shape != (length, 58):
            raise ValueError("Physical vector dimensions differ")
        if not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("Nonfinite physical input")
        import hashlib

        physical_digests[hashlib.sha256(state.tobytes() + actions.tobytes()).hexdigest()].append(episode)
        frame_ids = table["frame_index"].to_numpy()
        if not np.array_equal(frame_ids, np.arange(length)):
            raise ValueError("Noncontiguous frame indices")
        if not np.all(table["episode_index"].to_numpy() == episode):
            raise ValueError("Episode identity mismatch")
        if not np.allclose(table["timestamp"].to_numpy(), frame_ids / 30, rtol=0, atol=1e-4):
            raise ValueError("Timestamp grid mismatch")
        phases, tasks = table["coarse_phase_index"].to_numpy(), table["task_index"].to_numpy()
        if not table["observation.interaction_available"].to_numpy().all():
            raise ValueError("Missing ROI images")
        indices = {min(length - 1, max(memory.lags, default=0))}
        for phase, segment in enumerate(segments):
            start, end = segment["start_frame"], segment["end_frame"]
            if not np.all(phases[start:end] == phase) or not np.all(
                tasks[start:end] == contract.TASKS.index(segment["text"])
            ):
                raise ValueError("Frame-level task/phase labels disagree")
            indices.update((start, end - 1))
        # Exhaustively test the exact source indices for every history slot.
        global_ids = offset + np.arange(length)
        for lag in memory.lags:
            valid = frame_ids >= lag
            past = global_ids[valid] - lag
            if not np.all(raw.episodes[past] == episode) or not np.array_equal(
                raw.frame_indices[past], frame_ids[valid] - lag
            ):
                raise ValueError("Historical observation crosses a recording or uses a wrong timestamp")
            valid_slots += int(valid.sum()) * len(IMAGE_KEYS)
            masked_slots += int((~valid).sum()) * len(IMAGE_KEYS)
        for local_index in sorted(indices):
            index = offset + local_index
            item = source[index]
            model_input = transformed[index]
            if item["prompt"] != contract.PROMPTS[int(phases[local_index])]:
                raise ValueError("Loaded current task prompt differs")
            prompt_checks += 1
            if (
                set(model_input["image"]) != set(memory.all_image_keys)
                or model_input["state"].shape != (58,)
                or model_input["actions"].shape != (50, 58)
            ):
                raise ValueError("Model input structure differs")
            online = VisualHistory(memory)
            for previous in sorted({local_index - lag for lag in memory.lags if local_index >= lag} | {local_index}):
                result = online.observe(
                    raw.dataset[offset + previous],
                    episode_id=str(episode),
                    frame_index=previous,
                    timestamp=previous / 30,
                )
            for key in memory.history_keys:
                expected, actual = item[HISTORY], result[HISTORY]
                np.testing.assert_array_equal(expected["image"][key], actual["image"][key])
                if expected["image_mask"][key] != actual["image_mask"][key]:
                    raise ValueError("Historical image mask differs")
                if not actual["image_mask"][key] and actual["image"][key].any():
                    raise ValueError("Unavailable history must be zeroed")
                history_pixels += 1
            # Real history includes only images/masks; no historical vectors or oracle phase.
            if set(item[HISTORY]) != {"image", "image_mask"}:
                raise ValueError("History payload leaks nonvisual fields")
            history_decisions += 1
        offset += length
        frames_checked += length
        if (episode + 1) % 10 == 0:
            print(f"Audited recordings {episode + 1}/119", flush=True)
    training_exposure = {"verified": False}
    if checkpoint_run:
        identity = json.loads((checkpoint_run / "reproducibility/dataset_manifest.json").read_text())
        config = json.loads((checkpoint_run / "reproducibility/effective_config.json").read_text())
        session = json.loads((checkpoint_run / "wandb_latest_session.json").read_text())
        # Bind the completed run to these exact prepared physical files and frame IDs.
        for relative, digest in identity["files"].items():
            if file_sha256(root / relative) != digest:
                raise ValueError("Completed training used a different source identity")
        steps = sorted(int(p.name) for p in checkpoint_run.iterdir() if p.is_dir() and p.name.isdigit())
        training_exposure = {
            "verified": True,
            "run": str(checkpoint_run),
            "identity_sha256": file_sha256(checkpoint_run / "reproducibility/dataset_manifest.json"),
            "recordings": identity["episodes"],
            "frames": identity["frames"],
            "max_saved_step": max(steps),
            "session_status": session.get("status"),
            "existing_checkpoint_valid_for_proposed_test": False,
            "effective_config_top_level_keys": list(config)[:5],
        }
    duplicates = {
        "three_source_video_hashes": duplicate_groups(rows, contract.CAMERAS[:3]),
        "state_and_action_bytes": [v for v in physical_digests.values() if len(v) > 1],
    }
    proposal = candidate_split(rows)
    partition_by_episode = {
        r["episode_index"]: part for part, entries in proposal["assignments"].items() for r in entries
    }
    if any(len({partition_by_episode[i] for i in group}) > 1 for groups in duplicates.values() for group in groups):
        raise ValueError("Duplicate content crosses proposed partitions")
    report = {
        "schema": "visual_memory_all119_data_audit_v1",
        "audit_date": "2026-11-18",
        "timezone": "Asia/Shanghai",
        "code_branch": "tomato-visual-memory",
        "model_code_commit": "49252d2",
        "inputs": baseline_inputs,
        "bundle_sha256": file_sha256(bundle / "bundle.json"),
        "selection_sha256": file_sha256(bundle / "selection.json"),
        "prepared_files_hash_verified": verified_files,
        "recordings": len(rows),
        "frames_checked": frames_checked,
        "unique_recording_ids": len({r["recording_group"] for r in rows}),
        "source_packages": dict(Counter(r["source_dataset"] for r in rows)),
        "reviewed_phase_intervals": 5 * len(rows),
        "two_harvest_prompts_identical": True,
        "annotation_fields": dict(annotation_keys),
        "segment_fields": dict(segment_keys),
        "result_or_summary_labels_in_checked_schema": False,
        "physical_source_columns_equal_checks": physical_source_columns,
        "historical_view_slots_valid": valid_slots,
        "historical_view_slots_masked": masked_slots,
        "historical_indices_exhaustively_checked": True,
        "boundary_and_anchor_observations": history_decisions,
        "historical_view_pixel_mask_comparisons": history_pixels,
        "current_prompt_samples": prompt_checks,
        "exact_duplicate_checks": duplicates,
        "completed_training_exposure": training_exposure,
        "candidate_split": proposal,
        "roi_uses_future_coordinates": True,
        "fresh_source_video_decode_performed": False,
        "full_cache_file_sha256_performed": cache_audit is not None,
        "physical_result_video_review_performed": False,
        "trained_memory_model_evaluated": False,
        "cache_content_audit": cache_audit,
        "new_training_started": False,
        "robot_executed": False,
        "gpu_used": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "recordings",
                    "frames_checked",
                    "boundary_and_anchor_observations",
                    "historical_view_pixel_mask_comparisons",
                    "exact_duplicate_checks",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print("CANDIDATE_SPLIT", json.dumps(proposal["summary"], ensure_ascii=False), flush=True)
    print("REPORT", report_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-run", type=Path)
    parser.add_argument("--verify-migrated-cache", action="store_true")
    args = parser.parse_args()
    audit(args.report, args.checkpoint_run, verify_migration=args.verify_migrated_cache)


if __name__ == "__main__":
    main()
