"""Normal-only continuous LeRobot windows with source-level split and real RGB.

Reuses P1 canonical/history extraction. The index spans every valid current row
in every published episode. Inference conditions contain no future labels.
"""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess

import jax.numpy as jnp
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from openpi.models import model as model_lib
from openpi.training.flash_data import load_episode
from openpi.training.flash_data import make_batch

CAMERAS = {
    "base_0_rgb": "observation.images.cam0",
    "left_wrist_0_rgb": "observation.images.cam2",
    "right_wrist_0_rgb": "observation.images.cam1",
}
REMOTE_FRAME = """import av,io,json,sys
path=sys.argv[1]; target=int(sys.argv[2]); fps=float(sys.argv[3])
with av.open(path) as container:
 stream=container.streams.video[0]
 container.seek(int((target/fps)/stream.time_base),stream=stream,backward=True)
 for frame in container.decode(stream):
  i=round(float(frame.time)*fps)
  if i<target: continue
  if i!=target: raise RuntimeError("Exact current video frame missing")
  out=io.BytesIO(); frame.to_image().save(out,format="PNG")
  meta={"frame":i,"pts":frame.pts,"time_base":str(frame.time_base),"time_seconds":float(frame.time)}
  sys.stdout.buffer.write((json.dumps(meta)+"\\n").encode()+out.getvalue()); break
 else: raise RuntimeError("Requested frame not decoded")
"""


class FlashLeRobotDataset:
    def __init__(self, directory, image_cache, tokenizer=None):
        self.directory = Path(directory)
        self.image_cache = Path(image_cache)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.split = json.loads((self.directory / "split.json").read_text())
        self.normalization = json.loads((self.directory / "normalization.json").read_text())
        for label, document in (("manifest", self.manifest), ("split", self.split)):
            actual = hashlib.sha256(
                json.dumps(
                    {k: v for k, v in document.items() if k != "sha256"}, sort_keys=True, ensure_ascii=False
                ).encode()
            ).hexdigest()
            if actual != document["sha256"]:
                raise ValueError(f"{label} content fingerprint mismatch")
        self.normalization_snapshot_sha256 = hashlib.sha256(
            json.dumps(self.normalization, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        roots = {r["root"]: r for r in self.manifest["roots"]}
        self.records = {
            r["id"]: {**r, "info": roots[r["root"]]["info"], "tasks": roots[r["root"]]["tasks"]}
            for r in self.manifest["records"]
        }
        self.tokenizer = tokenizer
        self._episodes = OrderedDict()
        for r in self.records.values():
            if r["kind"] != "normal" or r["info"]["fps"] != 30:
                raise ValueError("Only normal 30Hz permitted")
        train = {
            r["source_recording_id"] for r in self.records.values() if self.split["assignments"][r["id"]] == "train"
        }
        validation = {
            r["source_recording_id"]
            for r in self.records.values()
            if self.split["assignments"][r["id"]] == "validation"
        }
        if train & validation:
            raise ValueError("Original recording crosses split")
        expected = set(self.normalization["train_sample_ids"])
        actual = {r["id"] for r in self.records.values() if self.split["assignments"][r["id"]] == "train"}
        if expected != actual:
            raise ValueError("Normalization fit scope differs from training split")

    def episode(self, ident):
        if ident not in self._episodes:
            self._episodes[ident] = load_episode(self.records[ident], self.directory)
            if not np.array_equal(self._episodes[ident]["frames"], np.arange(len(self._episodes[ident]["frames"]))):
                raise ValueError("LeRobot frame_index must start at zero for video alignment")
            if len(self._episodes) > 8:
                self._episodes.popitem(last=False)
        self._episodes.move_to_end(ident)
        return self._episodes[ident]

    def window(self, index):
        ident, start = index["id"], index["start"]
        record = self.records[ident]
        batch = make_batch([self.episode(ident)], [(ident, start)])
        batch["provenance"][0].update(
            source_recording_id=record["source_recording_id"], split=self.split["assignments"][ident]
        )
        return batch

    def training_plan(self, steps, seed):
        if not 1 <= steps <= 100:
            raise ValueError("Controlled P2 limited to 1..100 optimizer steps")
        records = [r for r in self.records.values() if self.split["assignments"][r["id"]] == "train"]
        counts = np.array([max(0, r["rows"] - 58) for r in records])
        ends = np.cumsum(counts)
        selected = np.random.default_rng(seed).choice(int(ends[-1]), steps, replace=False)
        plan = []
        for value in selected:
            episode = int(np.searchsorted(ends, value, side="right"))
            offset = int(value - (ends[episode - 1] if episode else 0))
            plan.append({"id": records[episode]["id"], "start": offset + 7})
        return plan

    def validation_plan(self):
        ident = self.split["complete_validation_episode"]
        result = [
            {"id": ident, "start": start, "role": "complete_held_out_episode"}
            for start in range(7, self.records[ident]["rows"] - 51, 50)
        ]
        for item in self.split["challenge_windows"]:
            result.extend(
                {"id": item["id"], "start": item["start"] + offset, "role": "predeclared_sharpa_transient"}
                for offset in (0, 50)
            )
        for item in result:
            if self.split["assignments"][item["id"]] != "validation":
                raise ValueError("Validation source leakage")
            self.window(item)
        return result

    def image(self, index, role):
        record = self.records[index["id"]]
        frame = index["start"]
        info = record["info"]
        folder = self.image_cache / index["id"] / str(frame)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (role + ".png")
        sidecar = folder / (role + ".json")
        if not path.exists() or not sidecar.exists():
            i = record["episode_index"]
            remote = (
                record["root"]
                + "/"
                + info["video_path"].format(
                    episode_chunk=i // info["chunks_size"], episode_index=i, video_key=CAMERAS[role]
                )
            )
            command = (
                "/mnt/data/train_platform/environments/pi05/20261014/bin/python -c "
                + shlex.quote(REMOTE_FRAME)
                + " "
                + shlex.quote(remote)
                + " "
                + str(frame)
                + " 30"
            )
            response = subprocess.check_output(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", record["host"], command]
            )
            header, payload = response.split(b"\n", 1)
            metadata = json.loads(header)
            if metadata["frame"] != frame or abs(metadata["time_seconds"] - frame / 30) > 1e-4:
                raise ValueError("Video/numeric timestamp mismatch")
            Image.open(io.BytesIO(payload)).verify()
            metadata.update(
                source_path=remote, sha256=hashlib.sha256(payload).hexdigest(), sample_id=index["id"], role=role
            )
            path.write_bytes(payload)
            sidecar.write_text(json.dumps(metadata, indent=2))
        metadata = json.loads(sidecar.read_text())
        payload = path.read_bytes()
        if metadata["sha256"] != hashlib.sha256(payload).hexdigest() or metadata["frame"] != frame:
            raise ValueError("Image cache fingerprint mismatch")
        return np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))

    def prefetch_images(self, indices):
        requests = {(i["id"], i["start"], role) for i in indices for role in CAMERAS}
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda entry: self.image({"id": entry[0], "start": entry[1]}, entry[2]), sorted(requests)))

    def observation(self, index):
        if self.tokenizer is None:
            raise ValueError("A real PaliGemma tokenizer is required")
        batch = self.window(index)
        record = self.records[index["id"]]
        lo = np.array(self.normalization["state_q01"])
        hi = np.array(self.normalization["state_q99"])
        state = (batch["state"] - lo) / (hi - lo + 1e-6) * 2 - 1
        task = int(
            pq.read_table(self.directory / record["local_file"], columns=["task_index"])["task_index"][
                index["start"]
            ].as_py()
        )
        tasks = {t["task_index"]: t["task"] for t in record["tasks"]}
        prompt = tasks[task]
        tokens, mask = self.tokenizer.tokenize(prompt, state[0])
        images = {role: jnp.asarray(self.image(index, role), jnp.float32)[None] / 127.5 - 1 for role in CAMERAS}
        obs = model_lib.Observation(
            images=images,
            image_masks={key: jnp.ones((1,), bool) for key in images},
            state=jnp.asarray(state, jnp.float32),
            tokenized_prompt=jnp.asarray(tokens[None], jnp.int32),
            tokenized_prompt_mask=jnp.asarray(mask[None]),
        )
        return model_lib.preprocess_observation(None, obs, train=False), batch, prompt
