"""Independent five-phase in-hand tomato contract; seven-phase experiments stay frozen."""

import json
from pathlib import Path

from openpi.subtasks.tomato import SOURCE_ACTION_NAMES
from openpi.subtasks.tomato import SOURCE_STATE_NAMES
from openpi.subtasks.tomato import input_text

PROFILE = "tomato-in-hand-five-phases-v3"
SEQUENCE = ("pickup", "harvest", "adjust", "harvest", "return")
PHASES = ("pickup", "harvest_1", "adjust_in_hand", "harvest_2", "return_stem")
ACTION_PROMPTS = {
    "pickup": "Right hand grasps and lifts the truss to the center",
    "harvest": "Left hand picks tomatoes into left basket while right hand holds truss",
    "adjust": "Right hand adjusts tomato truss position and orientation at the center",
    "return": "Right hand returns the held object to the right plate",
}
PROMPTS = tuple(ACTION_PROMPTS[k] for k in SEQUENCE)
TASKS = tuple(dict.fromkeys(PROMPTS))
CAMERAS = (*tuple(f"observation.images.cam{i}" for i in range(3)), "observation.images.interaction")
EPISODES = 204
FRAMES = 588734
VERSION = "tomato_in_hand204_rgb_crop_v3_r1"
DEFAULT_SOURCE = "/mnt/data/train_platform/datasets/tomato_sharpa/normal/full-block-more/subtask-roi"
DEFAULT_BUNDLE = "/mnt/data/users/mengxiang/workspace/training_data/sharpa/full-block-more/in_hand204_four_view_v3_r1"
DEFAULT_CACHE = "/mnt/data/users/mengxiang/workspace/training_data/sharpa/full-block-more/in_hand204_rgb224_cache_v3_r1"
ASSET_ID = VERSION + "_delta_30hz"
REPO_ID = "fr3_sharpa/" + VERSION


def reviewed_segments(document, length):
    rows = document.get("segments", [])
    if (
        document.get("prompt_profile") != PROFILE
        or document.get("stage") != "coarse"
        or tuple(r.get("label") for r in rows) != SEQUENCE
    ):
        raise ValueError("Reviewed v3 five-phase in-hand template required")
    end = 0
    for row in rows:
        a, b = row.get("start_frame"), row.get("end_frame")
        if (
            type(a) is not int
            or type(b) is not int
            or a != end
            or b <= a
            or row.get("reviewed") is not True
            or row.get("text") != ACTION_PROMPTS[row["label"]]
        ):
            raise ValueError("Reviewed contiguous boundaries and exact v3 prompts required")
        end = b
    if end != length:
        raise ValueError("Five phases must cover the complete recording")
    return rows


def audit_tokens(tokenizer):
    reports = []
    for prompt in TASKS:
        base = len(tokenizer.encode(input_text(prompt, [0] * 58), add_bos=True))
        increases = []
        for value in range(-1, 256):
            delta = len(tokenizer.encode(input_text(prompt, [value] + [0] * 57), add_bos=True)) - base
            uniform = len(tokenizer.encode(input_text(prompt, [value] * 58), add_bos=True))
            if uniform != base + 58 * delta:
                raise ValueError("Tokenizer state boundary behavior requires a new audit")
            increases.append(delta)
        maximum = base + 58 * max(increases)
        if maximum > 256:
            raise ValueError("Prompt/state exceeds256 tokens")
        reports.append({"prompt": prompt, "max_input_tokens": maximum})
    return reports


def validate_info(info):
    if (
        info.get("fps") != 30
        or info.get("codebase_version") != "v2.1"
        or info["features"]["observation.state"]["shape"] != [116]
        or info["features"]["observation.state"].get("names") != list(SOURCE_STATE_NAMES)
        or info["features"]["action"]["shape"] != [58]
        or info["features"]["action"].get("names") != list(SOURCE_ACTION_NAMES)
        or {k for k, v in info["features"].items() if v.get("dtype") == "video"} != set(CAMERAS)
    ):
        raise ValueError("30Hz four-view named Sharpa state116/action58 schema required")


_LEGACY = object()


def freeze(source, *, expected_episodes=_LEGACY, expected_frames=_LEGACY, version=VERSION):
    if expected_episodes is _LEGACY:
        expected_episodes = EPISODES
    if expected_frames is _LEGACY:
        expected_frames = FRAMES
    from openpi.subtasks.binding import verify_annotation
    from openpi.subtasks.prepare import digest

    source = Path(source).resolve(strict=True)
    manifest = json.loads((source / "manifest.json").read_text())
    records, seen = [], set()
    for package in manifest["packages"]:
        root = Path(package["path"]).resolve(strict=True)
        if not root.is_relative_to(source) or package.get("category") != "normal":
            raise ValueError("Only the selected normal publication is allowed")
        info = json.loads((root / "meta/info.json").read_text())
        validate_info(info)
        roi = json.loads((root / "roi_manifest.json").read_text())
        if (
            digest(root / "roi_manifest.json") != package["manifest_sha256"]
            or roi.get("current_rgb_pixels_only") is not True
            or roi.get("state_action_unchanged") is not True
            or roi.get("future_coordinates_used") is not True
            or roi.get("semantic_review_status")
            != "sampled_coding_agent_review_complete_not_independent_per_frame_accuracy"
            or digest(root / "visual_review.json") != roi["visual_review_sha256"]
        ):
            raise ValueError("Offline ROI provenance or sampled review changed")
        entries = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line]
        if len(entries) != package["episodes"] or sum(e["length"] for e in entries) != package["frames"]:
            raise ValueError("Publication episode/frame counts changed")
        receipts = {r["episode"]: r for r in roi["roi_receipts"]}
        for e in sorted(entries, key=lambda e: e["episode_index"]):
            ep = e["episode_index"]
            path = root / f"annotations/subtasks/coarse/episode_{ep:06d}.json"
            doc = json.loads(path.read_text())
            reviewed_segments(doc, e["length"])
            if (
                doc["source"].get("episode_index") != ep
                or doc["source"].get("length") != e["length"]
                or doc["source"].get("fps") != 30
                or not e.get("source_recording_id")
                or doc["source"].get("source_recording_id") != e["source_recording_id"]
                or e["source_recording_id"] in seen
            ):
                raise ValueError("Missing, duplicate or changed recording identity")
            verify_annotation(root, doc)
            receipt = receipts[ep]
            check = receipt["checks"]
            video = info["video_path"].format(episode_chunk=ep // 1000, episode_index=ep, video_key=CAMERAS[-1])
            if (
                receipt["frames"] != e["length"]
                or receipt["output_sha256"] != doc["source"]["sha256"][video]
                or not check.get("all_boxes_contained")
                or check.get("uncovered_frames")
                or check.get("center_steps_over_3pct") != 0
                or check.get("scale_steps_over_2pct") != 0
            ):
                raise ValueError("ROI receipt does not match reviewed current video")
            seen.add(e["source_recording_id"])
            records.append(
                {
                    "episode_index": len(records),
                    "source_dataset": str(root),
                    "source_episode_index": ep,
                    "recording_group": e["source_recording_id"],
                    "length": e["length"],
                    "annotation": str(path),
                    "annotation_sha256": digest(path),
                    "source_sha256": doc["source"]["sha256"],
                    "roi_manifest_sha256": digest(root / "roi_manifest.json"),
                    "entry": e,
                }
            )
            print(f"Frozen {len(records)}", flush=True)
    frames = sum(r["length"] for r in records)
    if (
        not records
        or (expected_episodes is not None and len(records) != expected_episodes)
        or (expected_frames is not None and frames != expected_frames)
    ):
        raise ValueError("Frozen episode/frame counts differ from the selected release")
    return {
        "version": version,
        "source": str(source),
        "publication_sha256": digest(source / "manifest.json"),
        "episodes": records,
        "frames": frames,
        "profile": PROFILE,
        "data_policy": "all_sources",
        "source_files_verified": True,
        "offline_roi_uses_future_coordinates": True,
        "independent_semantic_accuracy_measured": False,
    }
