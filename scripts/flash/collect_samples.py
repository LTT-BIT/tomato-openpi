"""Read-only SSH Parquet column extraction; never downloads video/tactile payloads.

Usage: python scripts/flash/collect_samples.py --output .cache/flash_samples
Select first/middle/last episode of each published dataset, excluding hidden and staging roots.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess

import pyarrow.parquet as pq

REMOTE_READER = """import sys,json
f=open(sys.argv[1],"rb")
for line in sys.stdin:
 p=json.loads(line); f.seek(p[0]); b=f.read(p[1]); sys.stdout.buffer.write(b); sys.stdout.buffer.flush()
"""


class RemoteFile(io.RawIOBase):
    def __init__(self, host, path, size):
        self.pos, self.size = 0, size
        self.proc = subprocess.Popen(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "python3 -u -c " + shlex.quote(REMOTE_READER) + " " + shlex.quote(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, size=-1):
        size = max(0, min(self.size - self.pos, size if size >= 0 else self.size - self.pos))
        if not size:
            return b""
        self.proc.stdin.write((json.dumps([self.pos, size]) + "\n").encode())
        self.proc.stdin.flush()
        data = bytearray()
        while len(data) < size:
            piece = self.proc.stdout.read(size - len(data))
            if not piece:
                raise OSError("Remote reader closed before requested bytes arrived")
            data.extend(piece)
        self.pos += size
        return bytes(data)

    def close(self):
        if not self.closed:
            self.proc.stdin.close()
            self.proc.wait(timeout=15)
        super().close()


def ssh_json(host, code):
    return json.loads(subprocess.check_output(["ssh", "-o", "BatchMode=yes", host, "python3 -c " + shlex.quote(code)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Use a new output path; existing samples are immutable")
    sources = [
        ("202", "/data/data-fruits/real/harvest-tomato/data/data-lerobot/sharpa", "raw"),
        ("28", "/mnt/data/train_platform/datasets/tomato_sharpa/normal", "normal"),
        ("28", "/mnt/data/train_platform/datasets/tomato_sharpa/speedup", "speedup"),
    ]
    records = []
    for host, base, kind in sources:
        code = f"""import os,json
from pathlib import Path
roots=[]
for d,dirs,files in os.walk({base!r}):
 dirs[:]=sorted(x for x in dirs if not x.startswith('.') and x not in ('production','ingest','conversion','videos'))
 if 'info.json' in files and Path(d).name=='meta':
  root=Path(d).parent; info=json.loads((root/'meta/info.json').read_text()); fs=sorted(root.glob('data/**/*.parquet'))
  if not fs: continue
  selected=sorted(set([0,len(fs)//2,len(fs)-1]))
  roots.append(dict(root=str(root),info=info,files=[dict(path=str(fs[i]),size=fs[i].stat().st_size) for i in selected]))
print(json.dumps(roots))"""
        datasets = ssh_json(host, code)
        for ds in datasets:
            for entry in ds["files"]:
                ident = hashlib.sha256((host + entry["path"]).encode()).hexdigest()[:16]
                remote = RemoteFile(host, entry["path"], entry["size"])
                try:
                    table = pq.read_table(
                        remote, columns=["action", "observation.state", "timestamp", "frame_index", "episode_index"]
                    )
                finally:
                    remote.close()
                out = args.output / (ident + ".parquet")
                pq.write_table(table, out)
                record = {
                    "id": ident,
                    "host": host,
                    "kind": kind,
                    "root": ds["root"],
                    "source_path": entry["path"],
                    "source_size": entry["size"],
                    "local_file": out.name,
                    "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                    "rows": len(table),
                    "info": ds["info"],
                }
                records.append(record)
                print(kind, ds["root"].split("/")[-1], entry["path"].split("/")[-1], len(table), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(records, indent=2))
    print(f"Saved {len(records)} real episodes")


if __name__ == "__main__":
    main()
