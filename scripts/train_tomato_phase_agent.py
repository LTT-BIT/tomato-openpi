"""Extract causal V-JEPA features and train a tomato phase head on a reviewed bundle.

Adapted from supplied lemon train_lemon_vjepa_phase.py. Explicit invocation only;
no Pi05 training, robot control, automatic downloads or lemon weight reuse.
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq

from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.subtasks import phase_model as model
from openpi.subtasks.prepare import digest
from openpi.subtasks.prepare import write_json
from openpi.subtasks.tomato import PHASES
from openpi.subtasks.tomato import PROMPTS


def extract(root, episode, encoder, device):
    import cv2

    info = json.loads((root / "meta/info.json").read_text())
    values = {"episode_chunk": episode // 1000, "episode_index": episode}
    table = pq.read_table(root / info["data_path"].format(**values))
    times = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if state.shape != (len(times), 116) or not np.isfinite(state).all():
        raise ValueError("Expected full 116D converter state")
    state = model.canonical_state(state, info["features"]["observation.state"]["names"])
    views = []
    for camera in model.CAMERAS:
        path = root / info["video_path"].format(**values, video_key=f"observation.images.{camera}")
        cap = cv2.VideoCapture(str(path))
        frames = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(
                    cv2.resize(
                        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (model.SIZE, model.SIZE), interpolation=cv2.INTER_AREA
                    )
                )
        finally:
            cap.release()
        if len(frames) != len(times):
            raise ValueError("Video/state frame count mismatch")
        views.append(np.stack(frames))
    # Fixed 2 Hz decision sampling, independent of annotation transition times.
    ends = np.unique(np.searchsorted(times, np.arange(times[0], times[-1] + 1e-9, 0.5), side="right") - 1)
    labels = np.asarray(table["coarse_phase_index"].to_pylist(), dtype=np.int64)
    features = []
    for end in ends:
        history = model.history_indices(times, int(end))
        features.append(model.encode_features(encoder, np.stack([v[history] for v in views]), state[history], device))
    return np.stack(features), labels[ends]


def metrics(logits, labels):
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((7, 7), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    f1 = 2 * confusion.diagonal() / np.maximum(1, confusion.sum(axis=0) + confusion.sum(axis=1))
    return {
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": float(f1.mean()),
        "per_phase_f1": f1.tolist(),
        "confusion": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--wandb-mode", choices=("online", "offline"), required=True)
    args = parser.parse_args()
    if args.output.exists() or args.epochs < 1:
        raise ValueError("Choose a new output directory and positive epochs")
    manifest = json.loads((args.bundle / "bundle.json").read_text())
    splits = json.loads((args.bundle / "splits.json").read_text())["splits"]
    if manifest.get("review_complete") is not True or manifest.get("phases") != list(PHASES):
        raise ValueError("Prepare the reviewed tomato bundle first")
    groups = {}
    ids = []
    for split in ("train", "val", "test"):
        if not splits[split]:
            raise ValueError("All evaluation splits must be nonempty")
        for episode in splits[split]:
            group = manifest["mapping"][episode]["recording_group"]
            if group in groups and groups[group] != split:
                raise ValueError("Original recording leaked across splits")
            groups[group] = split
            ids.append(episode)
    if len(ids) != len(set(ids)) or sorted(ids) != list(range(len(manifest["mapping"]))):
        raise ValueError("Splits must cover each prepared episode exactly once")
    import torch
    import wandb

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    args.output.mkdir(parents=True)
    settings = {
        "bundle_sha256": digest(args.bundle / "bundle.json"),
        "split_sha256": digest(args.bundle / "splits.json"),
        "encoder_path": str(args.encoder.resolve()),
        "encoder_revision": model.REVISION,
        "sample_hz": 10,
        "decision_hz": 2,
        "clip": 16,
        "seed": 42,
        "labels": PHASES,
        "prompts": PROMPTS,
        "joints": ORDER,
        "normalization": "training split only",
        "epochs": args.epochs,
    }
    write_json(args.output / "training_config.json", settings)
    run = wandb.init(project="tomato-phase-agent", dir=str(args.output), mode=args.wandb_mode, config=settings)
    try:
        encoder = model.load_encoder(args.encoder, args.device)
        arrays = {}
        for split, episodes in splits.items():
            items = [extract(args.bundle / "b_coarse", i, encoder, args.device) for i in episodes]
            arrays[split] = (np.concatenate([x for x, _ in items]), np.concatenate([y for _, y in items]))
        mean, std = model.training_statistics(arrays["train"][0])
        dim = len(mean)
        tensors = {s: (torch.from_numpy((x - mean) / std), torch.from_numpy(y)) for s, (x, y) in arrays.items()}
        head = model.head_model(dim).to(args.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
        counts = np.bincount(arrays["train"][1], minlength=7)
        if len(counts) != 7 or (counts == 0).any():
            raise ValueError("Training split must contain all seven phase occurrences")
        weights = torch.tensor(1 / np.sqrt(counts), dtype=torch.float32, device=args.device)
        weights /= weights.mean()

        def predict(split):
            head.eval()
            with torch.inference_mode():
                return torch.cat([head(x.to(args.device)).cpu() for x in tensors[split][0].split(128)]).numpy()

        best, stale = -1.0, 0
        for epoch in range(args.epochs):
            head.train()
            losses = []
            x, y = tensors["train"]
            for index in torch.randperm(len(y)).split(128):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(
                    head(x[index].to(args.device)), y[index].to(args.device), weight=weights
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())
            score = metrics(predict("val"), arrays["val"][1])["macro_f1"]
            report = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": score}
            print(json.dumps(report), flush=True)
            run.log(report)
            if score > best:
                best, stale = score, 0
                saved = {
                    "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "dimension": dim,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "labels": list(PHASES),
                    "prompts": list(PROMPTS),
                    "joints": list(ORDER),
                    "cameras": list(model.CAMERAS),
                    "encoder_revision": model.REVISION,
                    "clip": 16,
                    "sample_hz": 10,
                    "bundle_sha256": settings["bundle_sha256"],
                    "epoch": epoch,
                }
                torch.save(saved, args.output / "best_head.pt")
            else:
                stale += 1
            if stale >= 15:
                break
        saved = torch.load(args.output / "best_head.pt", map_location="cpu", weights_only=True)
        head.load_state_dict(saved["head"])
        evaluation = {s: metrics(predict(s), arrays[s][1]) for s in ("val", "test")}
        write_json(args.output / "evaluation.json", evaluation)
        run.summary.update({f"{s}_macro_f1": v["macro_f1"] for s, v in evaluation.items()})
    finally:
        run.finish()


if __name__ == "__main__":
    main()
