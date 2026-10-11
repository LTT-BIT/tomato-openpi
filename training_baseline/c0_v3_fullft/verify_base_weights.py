"""Verify the official π0.5 base checkpoint before exposing it to training.

The object listing is the public Google Cloud Storage JSON response for
``checkpoints/pi05_base/params/``. Downloads stay in ``params.partial`` until
every object's size and GCS CRC32C digest match the listing. MD5 is also
checked for the objects where GCS provides it.
"""

import base64
import hashlib
import json
import os
from pathlib import Path

import google_crc32c


def main() -> None:
    runtime = Path(os.environ.get("PI05_RUNTIME_DIR", "/home/zhiwei/pi05-runtime"))
    manifest = json.loads((runtime / "pi05-base-objects.json").read_text())
    if manifest.get("nextPageToken") or not manifest.get("items"):
        raise ValueError("Incomplete GCS object listing")
    items = manifest["items"]
    prefix = "checkpoints/pi05_base/params/"
    staging = runtime / "openpi-cache/openpi-assets/checkpoints/pi05_base/params.partial"
    destination = staging.with_name("params")
    if destination.exists():
        raise FileExistsError(f"Base checkpoint already exposed: {destination}")

    first = max(items, key=lambda item: int(item["size"]))
    first_name = Path(first["name"]).name
    probe = runtime / "base-download-probe" / first_name
    expected_paths = []
    for item in items:
        name = item["name"]
        if not name.startswith(prefix):
            raise ValueError(f"Unexpected GCS object: {name}")
        relative = Path(name.removeprefix(prefix))
        if not relative.parts or ".." in relative.parts:
            raise ValueError(f"Unsafe GCS object path: {name}")
        path = probe if item is first else staging / relative
        if path.with_name(path.name + ".aria2").exists():
            raise ValueError(f"Download is still in progress: {path}")
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise ValueError(f"Missing or incomplete GCS object: {path}")
        expected_crc32c = base64.b64decode(item["crc32c"], validate=True)
        expected_md5 = base64.b64decode(item["md5Hash"], validate=True) if item.get("md5Hash") else None
        crc32c = google_crc32c.Checksum()
        md5 = hashlib.md5() if expected_md5 is not None else None
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                crc32c.update(chunk)
                if md5 is not None:
                    md5.update(chunk)
        if crc32c.digest() != expected_crc32c or (md5 is not None and md5.digest() != expected_md5):
            raise ValueError(f"GCS checksum mismatch: {path}")
        expected_paths.append(path)
        print(f"Verified: {relative} ({item['size']} bytes)", flush=True)

    target = staging / first["name"].removeprefix(prefix)
    target.parent.mkdir(parents=True, exist_ok=True)
    probe.replace(target)
    staging.rename(destination)
    print(f"READY: {len(expected_paths)} official GCS objects verified at {destination}")


if __name__ == "__main__":
    main()
