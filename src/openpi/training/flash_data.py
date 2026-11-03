"""Read-only real LeRobot batches for FLASH offline smoke tests.

Numeric cache has no images: callers must label image fixtures as synthetic.
Only measured state positions at ticks <=0 enter history and current state.
Future commands are supervised targets only. No episode crossing or padding.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.policies.fr3_sharpa_protocol import ORDER
from openpi.shared.flash_polynomial import normalize_actions
from openpi.shared.flash_polynomial import unnormalize_actions


class ActionStatistics:
    def __init__(self, q01, q99, provenance):
        self.q01 = np.asarray(q01, np.float32)
        self.q99 = np.asarray(q99, np.float32)
        self.provenance = provenance

    def normalize(self, actions):
        return normalize_actions(actions, self.q01, self.q99)

    def denormalize(self, actions):
        return unnormalize_actions(actions, self.q01, self.q99)

    def to_dict(self):
        return {
            "q01": self.q01.tolist(),
            "q99": self.q99.tolist(),
            "epsilon": 1e-6,
            "clip": False,
            "fit_scope": "explicit_train_episodes_only",
            **self.provenance,
        }


def read_manifest(directory):
    return json.loads((Path(directory) / "manifest.json").read_text())


def load_episode(record, directory):
    path = Path(directory) / record["local_file"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != record["sha256"]:
        raise ValueError(f"Cache hash mismatch: {record['id']}")
    features = record["info"]["features"]
    names = features["action"]["names"]
    if not all(n.endswith((".position", ".command")) for n in names):
        raise ValueError("Absolute action semantics required")
    names = [n.rsplit(".", 1)[0] for n in names]
    if len(names) != 58 or set(names) != set(ORDER):
        raise ValueError("Action schema mismatch")
    state_names = features["observation.state"]["names"]
    table = pq.read_table(path)
    actions = np.asarray(table["action"].to_pylist(), np.float32)[:, [names.index(n) for n in ORDER]]
    states = np.asarray(table["observation.state"].to_pylist(), np.float32)[
        :, [state_names.index(n + ".position") for n in ORDER]
    ]
    frames = np.asarray(table["frame_index"].to_pylist()).reshape(-1)
    episode_ids = np.asarray(table["episode_index"].to_pylist()).reshape(-1)
    timestamp = np.asarray(table["timestamp"].to_pylist()).reshape(-1)
    if len(set(episode_ids)) != 1 or not np.all(np.diff(frames) == 1):
        raise ValueError("Episode/frame continuity violated")
    if not np.allclose(np.diff(timestamp), 1 / record["info"]["fps"], atol=2e-4, rtol=2e-3):
        raise ValueError("Timestamp continuity violated")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Nonfinite episode")
    return {
        "record": record,
        "actions": actions,
        "states": states,
        "frames": frames,
        "episode_index": int(episode_ids[0]),
        "canonical_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
    }


def fit_action_statistics(train_episodes):
    if not train_episodes:
        raise ValueError("Explicit nonempty train episodes required")
    actions = np.concatenate([e["actions"] for e in train_episodes])
    q01, q99 = np.quantile(actions, [0.01, 0.99], axis=0)
    state_q01, state_q99 = np.quantile(np.concatenate([e["states"] for e in train_episodes]), [0.01, 0.99], axis=0)
    provenance = {
        "train_sample_ids": [e["record"]["id"] for e in train_episodes],
        "train_cache_sha256": [e["record"]["sha256"] for e in train_episodes],
        "train_rows": len(actions),
        "joint_names": list(ORDER),
        "state_q01": state_q01.tolist(),
        "state_q99": state_q99.tolist(),
    }
    digest = hashlib.sha256(
        json.dumps({**provenance, "q01": q01.tolist(), "q99": q99.tolist()}, sort_keys=True).encode()
    ).hexdigest()
    return ActionStatistics(q01, q99, {**provenance, "statistics_sha256": digest})


def validate_split(train_episodes, eval_episodes):
    """Reject identical trajectories and same root in smoke split.

    Cross-version nonidentical resamples may still be same demonstration: use
    only a single version and disjoint dataset roots for the smoke test. This
    does not establish a benchmark-grade original-demo split.
    """
    train_kinds = {e["record"]["kind"] for e in train_episodes}
    eval_kinds = {e["record"]["kind"] for e in eval_episodes}
    if len(train_kinds | eval_kinds) != 1:
        raise ValueError("Smoke split must use one version to avoid raw/normal/speedup aliases")
    for key in ("id", "root"):
        if {e["record"][key] for e in train_episodes} & {e["record"][key] for e in eval_episodes}:
            raise ValueError(f"Overlapping {key} across train/eval")
    if {e["canonical_action_sha256"] for e in train_episodes} & {e["canonical_action_sha256"] for e in eval_episodes}:
        raise ValueError("Identical action trajectory across train/eval")


def make_batch(episodes, indices, statistics=None):
    """indices = [(cache_sample_id, current_row)], target/history physical radians.

    If statistics is passed, additionally returns normalized target/history.
    No inference callable accepts the future targets in this module.
    """
    by_id = {e["record"]["id"]: e for e in episodes}
    target, history, state, provenance = [], [], [], []
    for sample_id, start in indices:
        e = by_id[sample_id]
        if start < 7 or start + 51 >= len(e["actions"]):
            raise ValueError("Need real history [-7,0] and target context [-2,51]")
        target.append(e["actions"][start - 2 : start + 52])
        history.append(e["states"][start - 7 : start + 1])
        state.append(e["states"][start])
        provenance.append(
            {
                "sample_id": sample_id,
                "source_path": e["record"]["source_path"],
                "episode_index": e["episode_index"],
                "current_row": start,
                "current_frame": int(e["frames"][start]),
                "history_rows": [start - 7, start],
                "target_rows": [start - 2, start + 51],
                "cache_sha256": e["record"]["sha256"],
            }
        )
    batch = {
        "target_ext": np.stack(target),
        "history": np.stack(history),
        "state": np.stack(state),
        "provenance": provenance,
    }
    batch["actions"] = batch["target_ext"][:, 2:52]
    if statistics is not None:
        batch["normalized_target_ext"] = statistics.normalize(batch["target_ext"])
        batch["normalized_history"] = statistics.normalize(batch["history"])
    return batch
