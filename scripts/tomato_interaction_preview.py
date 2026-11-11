"""Offline, manually annotated interaction crops; never called by the B policy.

Geometry follows FoveaVLA Appendix A-D (1.5x context, square, contain the
interaction region, move/shrink at edges, zero-pad only if it cannot fit).
Keyframe interpolation uses future annotations: this is a data preview, NOT
an online ROI detector or a reproduction of learned foveation actions.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def valid_box(box, width, height):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Expected finite pixel xyxy coordinates")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Box must be ordered and inside the native image")
    return values


def interaction_at(clip, frame, width, height):
    keys = clip["keyframes"]
    indices = [key["frame"] for key in keys]
    if len(indices) < 2 or any(b <= a for a, b in itertools.pairwise(indices)):
        raise ValueError("Need strictly ordered annotation keyframes")
    if indices[0] != clip["start"] or indices[-1] != clip["end"] - 1:
        raise ValueError("Annotations must cover both endpoints; no extrapolation")
    if not clip["start"] <= frame < clip["end"]:
        raise ValueError("Frame outside annotated interval")
    roles = clip["roles"]
    if not roles or len(roles) != len(set(roles)):
        raise ValueError("Need distinct nonempty semantic roles")
    boxes = []
    for role in roles:
        samples = [valid_box(k["boxes"][role], width, height) for k in keys]
        boxes.append([np.interp(frame, indices, [b[c] for b in samples]) for c in range(4)])
    boxes = np.asarray(boxes)
    return [*boxes[:, :2].min(axis=0), *boxes[:, 2:].max(axis=0)]



def annotation_available_frame(clip, frame):
    """Latest source frame used by the two keys, including interpolated key provenance."""
    import bisect

    keys = clip["keyframes"]
    indices = [key["frame"] for key in keys]
    if not indices or any(b <= a for a, b in itertools.pairwise(indices)):
        raise ValueError("Need strictly ordered annotation keyframes")
    if not indices[0] <= frame <= indices[-1]:
        raise ValueError("Frame outside annotation keyframes")
    right = bisect.bisect_left(indices, frame)
    used = [keys[right]] if indices[right] == frame else [keys[right - 1], keys[right]]
    available = []
    for key in used:
        value = key.get("available_at_frame", key["frame"])
        if not isinstance(value, int) or isinstance(value, bool) or value < key["frame"]:
            raise ValueError("Invalid keyframe availability provenance")
        available.append(value)
    return max(available)


def square_box(box, width, height, scale=1.5):
    x1, y1, x2, y2 = valid_box(box, width, height)
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("Context scale must be finite and at least one")
    required = max(math.ceil(x2) - math.floor(x1), math.ceil(y2) - math.floor(y1), 2)
    side = max(required, math.ceil(max(x2 - x1, y2 - y1) * scale))
    side = min(side, min(width, height) if required <= min(width, height) else max(width, height))

    def fit(center, low, high, length):
        start = math.floor(center - side / 2)
        lower = max(0, math.ceil(high - side)) if side <= length else length - side
        upper = min(length - side, math.floor(low)) if side <= length else 0
        if lower > upper:
            raise ValueError("Cannot contain interaction box")
        return min(max(start, lower), upper)

    left = fit((x1 + x2) / 2, x1, x2, width)
    top = fit((y1 + y2) / 2, y1, y2, height)
    return left, top, left + side, top + side


def native_crop(image, box):
    left, top, right, bottom = box
    side = right - left
    if side <= 0 or side != bottom - top:
        raise ValueError("Expected nonempty square crop")
    height, width = image.shape[:2]
    result = np.zeros((side, side, 3), dtype=np.uint8)
    x1, y1, x2, y2 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    result[y1 - top : y2 - top, x1 - left : x2 - left] = image[y1:y2, x1:x2]
    return result


def letterbox(image, size=224):
    import cv2

    height, width = image.shape[:2]
    ratio = size / max(height, width)
    w, h = round(width * ratio), round(height * ratio)
    out = np.zeros((size, size, 3), dtype=np.uint8)
    x, y = (size - w) // 2, (size - h) // 2
    out[y : y + h, x : x + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return out


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    import av
    import cv2
    from PIL import Image
    from PIL import ImageDraw

    from scripts.tomato_crop_audit import frames_at

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True, help="Directory with cam0/1/2.mp4")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.annotations.read_text())
    if not spec["clips"] or any(c["start"] < 0 or c["end"] <= c["start"] for c in spec["clips"]):
        raise ValueError("Need nonempty nonnegative clip intervals")
    names = [c["name"] for c in spec["clips"]]
    if len(names) != len(set(names)) or any(Path(n).name != n or n in (".", "..") for n in names):
        raise ValueError("Need unique simple clip names")
    wanted = {f for c in spec["clips"] for f in range(c["start"], c["end"])}
    if sum(c["end"] - c["start"] for c in spec["clips"]) != len(wanted):
        raise ValueError("Overlapping or empty preview intervals")
    views, sources = {}, {}
    for cam in ("cam0", "cam1", "cam2"):
        path = args.videos / f"{cam}.mp4"
        digest = sha256(path)
        if digest != spec["source_sha256"][cam]:
            raise ValueError(f"Wrong source video: {cam}")
        with av.open(str(path)) as stream:
            video = stream.streams.video[0]
            if float(video.average_rate) != spec["fps"]:
                raise ValueError("Video FPS differs from annotation")
        views[cam] = frames_at(path, wanted)
        sources[cam] = {"path": str(path.resolve()), "sha256": digest}
    args.output.mkdir(parents=True, exist_ok=False)
    rows, posters = [], []
    for clip in spec["clips"]:
        folder = args.output / clip["name"]
        folder.mkdir()
        video_path = folder / "preview.mp4"
        with av.open(str(video_path), "w") as container:
            stream = container.add_stream("libx264", rate=spec["fps"])
            stream.width, stream.height, stream.pix_fmt = 896, 672, "yuv420p"
            for frame in range(clip["start"], clip["end"]):
                full = views["cam0"][frame]
                height, width = full.shape[:2]
                if [width, height] != spec["source_size"]:
                    raise ValueError("Annotation dimensions differ from video")
                tight = interaction_at(clip, frame, width, height)
                square = square_box(tight, width, height)
                crop = native_crop(full, square)
                small = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                stem = f"f{frame:06d}"
                Image.fromarray(crop).save(folder / f"{stem}_crop_native.png")
                Image.fromarray(small).save(folder / f"{stem}_crop224.png")
                images = [letterbox(views[cam][frame]) for cam in ("cam0", "cam2", "cam1")]
                for cam, im in zip(("cam0", "cam2", "cam1"), images, strict=True):
                    Image.fromarray(im).save(folder / f"{stem}_{cam}_224.png")
                overlay = Image.fromarray(full)
                draw = ImageDraw.Draw(overlay)
                draw.rectangle(tuple(map(int, tight)), outline="#ffb000", width=2)
                draw.rectangle(square, outline="#00e0d0", width=2)
                panel = Image.new("RGB", (896, 672), "#151515")
                panel.paste(overlay, (0, 25))
                panel.paste(Image.fromarray(small), (660, 80))
                for i, im in enumerate([*images, small]):
                    panel.paste(Image.fromarray(im), (224 * i, 448))
                draw = ImageDraw.Draw(panel)
                draw.text(
                    (8, 5), f"{clip['name']} | f{frame} | {frame / spec['fps']:.3f}s | MANUAL offline ROI", fill="white"
                )
                draw.text((660, 40), "native crop -> 224", fill="white")
                for i, label in enumerate(("global", "left wrist", "right wrist", "extra interaction crop")):
                    draw.text((224 * i + 6, 430), label, fill="white")
                for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(panel), format="rgb24")):
                    container.mux(packet)
                rows.append(
                    {
                        "clip": clip["name"],
                        "source_frame": frame,
                        "timestamp": frame / spec["fps"],
                        "interaction_xyxy": list(map(float, tight)),
                        "crop_xyxy": square,
                        "views224": [f"{clip['name']}/{stem}_{cam}_224.png" for cam in ("cam0", "cam2", "cam1")]
                        + [f"{clip['name']}/{stem}_crop224.png"],
                        "native_crop": f"{clip['name']}/{stem}_crop_native.png",
                        "linear_magnification_vs_letterboxed_full": width / (square[2] - square[0]),
                    }
                )
                if frame in clip["show_frames"]:
                    panel.save(folder / f"{stem}_panel.jpg", quality=95)
                    posters.append(panel)
            for packet in stream.encode():
                container.mux(packet)
    poster = Image.new("RGB", (896, 672 * len(posters)))
    for i, panel in enumerate(posters):
        poster.paste(panel, (0, i * 672))
    poster.save(args.output / "comparison.jpg", quality=95)
    (args.output / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {
        "scope": "offline manually annotated/interpolated preview, not model ROI prediction or policy evaluation",
        "baseline": "B unchanged; no training; no input pipeline changes",
        "sources": sources,
        "annotation_sha256": sha256(args.annotations),
        "frames": len(rows),
        "fps": spec["fps"],
        "clips": [
            {"name": c["name"], "start": c["start"], "end_exclusive": c["end"], "frames": c["end"] - c["start"]}
            for c in spec["clips"]
        ],
        "linear_magnification_range": [
            min(r["linear_magnification_vs_letterboxed_full"] for r in rows),
            max(r["linear_magnification_vs_letterboxed_full"] for r in rows),
        ],
        "annotation_method": spec["annotation_method"],
        "paper": spec["paper"],
        "code_sha256": sha256(Path(__file__)),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
