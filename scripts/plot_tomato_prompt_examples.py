"""Plot joint targets for predefined representative paired observations; no robot I/O."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episode", type=int, default=89)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    folder = args.root / f"shard{args.episode % 4}"
    protocol = json.loads((folder / "protocol.json").read_text())
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    joints = (0, 3, 9, 12, 29, 32, 38, 41)
    names = (
        "left arm joint1",
        "left arm joint4",
        "left thumb MCP FE",
        "left index MCP FE",
        "right arm joint1",
        "right arm joint4",
        "right thumb MCP FE",
        "right index MCP FE",
    )
    for phase in (1, 3):
        row = next(
            r
            for r in protocol["plan"]
            if r["episode"] == args.episode and r["phase"] == phase and any(t["kind"] == "midpoint" for t in r["tags"])
        )
        file = folder / f"ep{args.episode:03d}_f{row['frame']:06d}.npz"
        with np.load(file) as f:
            prediction = f["absolute_s42"]
            target = f["target_absolute"]
            state = f["state"]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        time = np.arange(50) / 30
        for ax, joint, name in zip(axes.flat, joints, names, strict=True):
            ax.plot(time, target[:, joint], "k--", lw=2, label="recorded expert target")
            ax.axhline(state[joint], color="gray", ls=":", lw=1, label="observed joint position")
            for p, (prompt, color) in enumerate(zip(protocol["prompt_order"], colors, strict=True)):
                ax.plot(time, prediction[p, :, joint], color=color, label=prompt)
            ax.set(title=name, xlabel="action horizon (s)", ylabel="absolute joint target (rad)")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            f"Same observation and noise / recording {args.episode}, frame {row['frame']}, phase {phase + 1}\nCounterfactual prompt correctness has no expert target; curves show sensitivity only"
        )
        fig.tight_layout()
        fig.savefig(args.output / f"ep{args.episode:03d}_phase{phase + 1}_actions.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
