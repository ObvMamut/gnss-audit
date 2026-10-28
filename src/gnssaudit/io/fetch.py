"""Download, checksum and extraction of public datasets (docs/7 section 3.2).

Files are written to a `.part` name and renamed only after the checksum holds; the raw store is read-only
(docs/6 section 8.2). Archives are extracted into a new, empty directory, and members that would land outside it
are refused.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

CHUNK = 1 << 20


class ChecksumError(ValueError):
    """A downloaded file does not match its published checksum."""


@dataclass(frozen=True)
class FetchResult:
    path: Path
    size_bytes: int
    sha256: str


def commit_part(part: Path, target: Path) -> None:
    """Make a finished `.part` file durable (fsync it, rename it, fsync the directory) so a crash never leaves a
    zero-length file under the final name."""
    fd = os.open(part, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    part.rename(target)
    dfd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _read_only(path: Path) -> None:
    path.chmod(path.stat().st_mode & 0o555)


def fetch(
    url: str,
    dest_dir: Path,
    *,
    name: str | None = None,
    md5: str | None = None,
    sha256: str | None = None,
    timeout_s: float = 60.0,
    retries: int = 3,
) -> FetchResult:
    """Download url into dest_dir, verify md5 and/or sha256 where given, and make the file read-only."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / (name or url.rstrip("/").rsplit("/", 1)[-1])
    part = target.with_name(target.name + ".part")
    h_md5, h_sha = hashlib.md5(), hashlib.sha256()
    for attempt in range(retries):
        try:
            h_md5, h_sha = hashlib.md5(), hashlib.sha256()
            with urllib.request.urlopen(url, timeout=timeout_s) as resp, part.open("wb") as out:
                while chunk := resp.read(CHUNK):
                    out.write(chunk)
                    h_md5.update(chunk)
                    h_sha.update(chunk)
            break
        except OSError:
            part.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2.0**attempt)
    got_md5, got_sha = h_md5.hexdigest(), h_sha.hexdigest()
    if (md5 and got_md5 != md5.lower()) or (sha256 and got_sha != sha256.lower()):
        part.unlink()
        raise ChecksumError(f"{url}: md5 {got_md5}, sha256 {got_sha} do not match the published checksum")
    commit_part(part, target)
    _read_only(target)
    return FetchResult(target, target.stat().st_size, got_sha)


def _inside(root: Path, member: str) -> Path:
    path = (root / member).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"archive member {member!r} would land outside {root}")
    return path


def extract(archive: Path, dest_dir: Path) -> Path:
    """Extract a tar, zip or single .gz file into dest_dir, which must not exist or be empty; output is read-only."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise FileExistsError(f"{dest_dir} is not empty")
    dest_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                _inside(dest_dir, m.name)
                if not (m.isfile() or m.isdir()):
                    raise ValueError(f"archive member {m.name!r} is not a regular file or directory")
            tar.extractall(dest_dir, filter="data")
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                _inside(dest_dir, n)
            z.extractall(dest_dir)
    elif archive.suffix == ".gz":
        with gzip.open(archive, "rb") as src, (dest_dir / archive.stem).open("wb") as out:
            shutil.copyfileobj(src, out, CHUNK)
    else:
        raise ValueError(f"{archive}: not a tar, zip or gzip file")
    for path in sorted(dest_dir.rglob("*"), reverse=True):
        _read_only(path)
    return dest_dir
