"""Data identity shared by experiment composition and launch preflight."""

from dataclasses import dataclass
import os
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
    allow_legacy_norm_stats: bool = False

    def __post_init__(self):
        if not self.sources or any(not isinstance(source, str) or not source.strip() for source in self.sources):
            raise ValueError("Record at least one nonempty data source")


def repository_root() -> Path:
    """Locate this checkout regardless of the working directory or old runtime settings."""
    return Path(__file__).resolve().parents[1]


def configure_environment() -> None:
    """Set checkout-local paths before training dependencies are imported."""
    repository = repository_root()
    cache = repository / ".cache"
    paths = {
        "UV_PROJECT_ENVIRONMENT": repository / ".venv",
        "UV_CACHE_DIR": cache / "uv",
        "XDG_CACHE_HOME": cache,
        "OPENPI_DATA_HOME": cache / "openpi",
        "HF_HOME": cache / "huggingface",
        "HF_HUB_CACHE": cache / "huggingface/hub",
        "HF_DATASETS_CACHE": cache / "huggingface/datasets",
        "TORCH_HOME": cache / "torch",
        "WANDB_DIR": repository,
        "WANDB_CACHE_DIR": cache / "wandb",
        "WANDB_DATA_DIR": cache / "wandb-data",
        "TMPDIR": cache / "tmp",
    }
    os.environ.pop("PI05_RUNTIME_DIR", None)
    os.environ.update({key: str(path) for key, path in paths.items()})
    os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    for directory in (
        cache / "tmp",
        *(repository / name for name in ("data", "assets", "logs", "checkpoints", "wandb")),
    ):
        directory.mkdir(parents=True, exist_ok=True)
