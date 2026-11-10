"""Read-only native-resolution crop feasibility sheets; no policy performance claim."""

import argparse
import json
from pathlib import Path


def centered_box(width, height, fraction=0.5):
    if not 0 < fraction <= 1 or min(width, height) < 1:
        raise ValueError("Invalid native crop dimensions")
    w, h = max(1, round(width * fraction)), max(1, round(height * fraction))
    x, y = (width - w) // 2, (height - h) // 2
    return x, y, x + w, y + h


def frames_at(path, wanted):
    import av

    found = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                found[index] = frame.to_ndarray(format="rgb24")
            if len(found) == len(wanted):
                break
    if set(found) != set(wanted):
        raise ValueError("Video did not contain all required native frames")
    return found


def main():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 46, 89])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new crop audit directory")
    args.output.mkdir(parents=True)
    info = json.loads((args.root / "meta/info.json").read_text())
    report = {
        "scope": "Illustrative fixed center-half crops, not learned attention/ROI or evidence of better control",
        "episodes": [],
    }
    for ep in args.episodes:
        segments = json.loads((args.root / f"annotations/episode_{ep:06d}.json").read_text())["segments"]
        wanted = [(s["start_frame"] + s["end_frame"] - 1) // 2 for s in segments]
        views = {}
        for cam in ("cam0", "cam2", "cam1"):
            path = args.root / info["video_path"].format(
                episode_chunk=ep // 1000, episode_index=ep, video_key=f"observation.images.{cam}"
            )
            views[cam] = frames_at(path, set(wanted))
        fig, axes = plt.subplots(7, 5, figsize=(20, 19))
        for phase, frame in enumerate(wanted):
            for col, cam in enumerate(("cam0", "cam2", "cam1")):
                axes[phase, col].imshow(views[cam][frame])
                axes[phase, col].set_title(
                    f"phase {phase + 1} {segments[phase]['label']} / {cam} / f{frame}", fontsize=9
                )
            image = views["cam1"][frame]
            h, w = image.shape[:2]
            x1, y1, x2, y2 = centered_box(w, h)
            axes[phase, 3].imshow(image[y1:y2, x1:x2])
            axes[phase, 3].set_title("right wrist native center-half crop (illustrative)", fontsize=8)
            rx1, ry1, rx2, ry2 = round(w * 0.45), round(h / 3), w, h
            axes[phase, 4].imshow(image[ry1:ry2, rx1:rx2])
            axes[phase, 4].set_title("right-lower candidate crop (not learned ROI)", fontsize=8)
            for ax in axes[phase]:
                ax.axis("off")
            for cam in views:
                plt.imsave(args.output / f"ep{ep:03d}_phase{phase + 1}_{cam}.png", views[cam][frame])
        fig.tight_layout()
        fig.savefig(args.output / f"ep{ep:03d}_native_crop_sheet.jpg", dpi=130)
        plt.close(fig)
        report["episodes"].append(
            {
                "episode": ep,
                "frames": wanted,
                "native_shapes": {cam: list(views[cam][wanted[0]].shape) for cam in views},
                "crop_box": list(centered_box(w, h)),
                "patch_token_density_per_native_pixel_relative_to_full": 4.0,
                "limitations": "Center box may miss fingers/object; zoom cannot recover occlusion; wrist cameras already provide close-up views.",
            }
        )
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
