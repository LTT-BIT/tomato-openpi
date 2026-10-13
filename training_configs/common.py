"""Data identity shared by experiment composition and launch preflight."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    version: str
    repo_id: str
    root: str
    assets_dir: str
    asset_id: str
    sources: tuple[str, ...]
    prompt_example: str
    expected_norm_sha256: str | None = None
    expected_norm_audit_sha256: str | None = None

    def __post_init__(self):
        if not self.sources or any(not isinstance(source, str) or not source.strip() for source in self.sources):
            raise ValueError("Record at least one nonempty data source")


def repository_root() -> Path:
    """Locate this checkout regardless of the working directory or old runtime settings."""
    return Path(__file__).resolve().parents[1]
