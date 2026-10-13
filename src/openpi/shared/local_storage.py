"""Publish checkout-local artifacts without exposing partially written bundles."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


@contextmanager
def new_artifact_directory(destination: Path):
    """Lock a target and atomically publish a new directory on the same filesystem."""
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    locks = destination.parent / ".norm-locks"
    locks.mkdir(exist_ok=True)
    key = hashlib.sha256(str(destination).encode()).hexdigest()
    with (locks / f"{key}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Statistics are already being computed for {destination}") from error
        try:
            if destination.exists():
                raise FileExistsError(f"Statistics target already exists: {destination}; use a new asset_id")
            with tempfile.TemporaryDirectory(
                prefix=f".{destination.name}.building-", dir=destination.parent
            ) as temporary:
                staging = Path(temporary)
                yield staging
                if destination.exists():
                    raise FileExistsError(f"Statistics target appeared during computation: {destination}")
                staging.rename(destination)
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def atomic_write_json(path: Path, value: dict) -> None:
    """Replace a small local record only after its full JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
