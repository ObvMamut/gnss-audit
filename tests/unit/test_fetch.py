"""Download, checksum and safe extraction of public datasets (docs/7 section 3.2)."""

import gzip
import hashlib
import io
import stat
import tarfile
from pathlib import Path

import pytest

from gnssaudit.io.fetch import ChecksumError, extract, fetch


def _source(tmp_path: Path, content: bytes = b"gnss data\n") -> Path:
    src = tmp_path / "src" / "file.txt"
    src.parent.mkdir()
    src.write_bytes(content)
    return src


def test_fetch_verifies_md5_and_makes_file_read_only(tmp_path: Path) -> None:
    src = _source(tmp_path)
    md5 = hashlib.md5(src.read_bytes()).hexdigest()
    got = fetch(src.as_uri(), tmp_path / "dest", md5=md5)
    assert got.path.read_bytes() == src.read_bytes()
    assert got.size_bytes == src.stat().st_size
    assert got.sha256 == hashlib.sha256(src.read_bytes()).hexdigest()
    assert not got.path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def test_bad_checksum_raises_and_leaves_nothing(tmp_path: Path) -> None:
    src = _source(tmp_path)
    with pytest.raises(ChecksumError):
        fetch(src.as_uri(), tmp_path / "dest", md5="0" * 32)
    assert list((tmp_path / "dest").iterdir()) == []


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def test_extract_tar_into_new_read_only_directory(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"top/x.csv": b"1,2\n"})
    out = extract(archive, tmp_path / "out")
    target = out / "top" / "x.csv"
    assert target.read_bytes() == b"1,2\n"
    assert not target.stat().st_mode & stat.S_IWUSR


def test_extract_refuses_member_outside_destination(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "evil.tar.gz", {"../escape.txt": b"x"})
    with pytest.raises(ValueError, match="outside"):
        extract(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_refuses_non_empty_destination(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "a.tar.gz", {"x": b"1"})
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old").write_text("keep")
    with pytest.raises(FileExistsError):
        extract(archive, tmp_path / "out")


def test_extract_single_gzip_file(tmp_path: Path) -> None:
    gz = tmp_path / "nav.rnx.gz"
    gz.write_bytes(gzip.compress(b"RINEX\n"))
    out = extract(gz, tmp_path / "out")
    assert (out / "nav.rnx").read_bytes() == b"RINEX\n"
