"""Export inference checkpoint, committed source, tokenizer and deployment notes."""

import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    repo = Path(__file__).resolve().parents[1]
    checkpoint = (
        repo / "checkpoints/pi05_fr3_wuji_1002_subtasks_detailed_30hz/tomato_1002_subtasks_detailed_lora_30hz/19999"
    )
    output = repo / "exports/pi05_1002_subtasks_19999_deploy.zip"
    output.parent.mkdir(exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    assert (checkpoint / "_CHECKPOINT_METADATA").is_file()
    assert (checkpoint / "params/_METADATA").is_file()
    assert (checkpoint / "assets/1002_subtasks_detailed/norm_stats.json").is_file()
    tokenizer = Path("/home/descfly/.cache/openpi/big_vision/paligemma_tokenizer.model")
    assert tokenizer.is_file()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    prefix = "pi05_1002_subtasks_19999"
    paths = [
        "src",
        "packages",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "README.md",
        "LICENSE",
        "LICENSE_GEMMA.txt",
        "scripts/serve_1002_bundle.py",
    ]
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD", "--", *paths], cwd=repo, text=True
    ).splitlines()
    manifest = {
        "source_commit": revision,
        "checkpoint_step": 19999,
        "config": "pi05_fr3_wuji_1002_subtasks_detailed_30hz",
        "files": {},
    }

    def add_disk(archive, path, name):
        digest = hashlib.sha256()
        with path.open("rb") as src, archive.open(prefix + "/" + name, "w", force_zip64=True) as dst:
            while block := src.read(8 * 1024 * 1024):
                digest.update(block)
                dst.write(block)
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}

    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for folder in ("params", "assets"):
            for path in sorted((checkpoint / folder).rglob("*")):
                if path.is_file():
                    add_disk(archive, path, "checkpoint/" + str(path.relative_to(checkpoint)))
        add_disk(archive, tokenizer, "cache/big_vision/paligemma_tokenizer.model")
        for path in files:
            content = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=repo)
            archive.writestr(prefix + "/code/" + path, content)
        for source, target in [
            ("docs/部署包使用说明_1002.md", "部署说明.md"),
            ("docs/prompts_1002_subtasks.json", "prompts.json"),
            ("docs/部署记录_1002子任务.md", "训练与部署记录.md"),
        ]:
            add_disk(archive, repo / source, target)
        archive.writestr(prefix + "/manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
    with output.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    output.with_suffix(".zip.sha256").write_text(f"{checksum}  {output.name}\n")
    print(
        json.dumps(
            {
                "zip": str(output),
                "bytes": output.stat().st_size,
                "sha256": checksum,
                "source_commit": revision,
                "crc_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
