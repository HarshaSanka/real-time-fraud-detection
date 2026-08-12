"""Reproducible anonymous Sparkov download and checksum verification."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from fraud_detection.exceptions import DataValidationError
from fraud_detection.utils.config import Settings, project_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(settings: Settings, force: bool = False) -> list[Path]:
    root = project_root()
    raw_dir = root / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "sparkov-fraud-detection.zip"
    expected_paths = [raw_dir / name for name in settings.data.files]
    if all(path.exists() for path in expected_paths) and not force:
        verify_source_files(settings)
        return expected_paths
    temporary = archive.with_suffix(".download")
    with (
        urllib.request.urlopen(settings.data.url, timeout=120) as response,
        temporary.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)
    if sha256_file(temporary) != settings.data.archive_sha256:
        temporary.unlink(missing_ok=True)
        raise DataValidationError("downloaded Sparkov archive checksum mismatch")
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        if set(settings.data.files) != names:
            raise DataValidationError(f"unexpected archive members: {sorted(names)}")
        for name in settings.data.files:
            destination = raw_dir / name
            with bundle.open(name) as source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
    verify_source_files(settings)
    return expected_paths


def verify_source_files(settings: Settings) -> None:
    raw_dir = project_root() / "data/raw"
    for name, expected in settings.data.files.items():
        path = raw_dir / name
        if not path.exists():
            raise DataValidationError(f"missing source file: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise DataValidationError(f"checksum mismatch for {name}: {actual}")
