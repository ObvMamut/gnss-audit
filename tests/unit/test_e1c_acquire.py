"""Resume safety of the arm c acquisition: only valid reduced tables and raw files count as done."""

import gzip
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from gnssaudit.experiments import e1c_acquire
from gnssaudit.experiments.e1c import reduced_path
from gnssaudit.io.fetch import FetchResult, commit_part
from gnssaudit.io.rgp import REDUCED_SCHEMA


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GNSSAUDIT_DATA", str(tmp_path))
    return tmp_path


def write_valid(path: Path) -> None:
    meta = {k: b"[]" for k in e1c_acquire.REDUCED_META}
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(REDUCED_SCHEMA.empty_table().replace_schema_metadata(meta), path)


def test_zero_byte_reduced_file_is_redone_and_a_valid_one_skipped() -> None:
    bad, good, absent = (reduced_path("ptrc", 2024, d) for d in (251, 252, 253))
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"")
    write_valid(good)
    assert e1c_acquire.pending_days("ptrc", 2024, [251, 252, 253]) == [251, 253]
    assert not bad.exists() and good.exists() and not absent.exists()


def test_reduced_file_without_the_metadata_keys_is_not_done() -> None:
    path = reduced_path("abcd", 2024, 253)
    path.parent.mkdir(parents=True)
    pq.write_table(
        REDUCED_SCHEMA.empty_table().replace_schema_metadata({b"station": json.dumps("abcd").encode()}), path
    )
    assert not e1c_acquire.reduced_valid(path)


def test_raw_file_is_reused_only_if_it_expands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    dest = tmp_path / "scratch"
    dest.mkdir()
    target = dest / "file.rnx.gz"
    target.write_bytes(gzip.compress(b"hello"))
    assert e1c_acquire._get("http://x/file.rnx.gz", dest) is not None  # reused, no download
    target.write_bytes(b"truncated garbage")
    calls = []

    def fake_fetch(url: str, d: Path) -> FetchResult:
        calls.append(url)
        target.write_bytes(gzip.compress(b"fresh"))
        return FetchResult(target, target.stat().st_size, "0" * 64)

    monkeypatch.setattr(e1c_acquire, "fetch", fake_fetch)
    got = e1c_acquire._get("http://x/file.rnx.gz", dest)
    assert calls == ["http://x/file.rnx.gz"] and got is not None


def test_commit_part_renames_and_leaves_no_part(tmp_path: Path) -> None:
    part = tmp_path / "a.bin.part"
    part.write_bytes(b"data")
    commit_part(part, tmp_path / "a.bin")
    assert (tmp_path / "a.bin").read_bytes() == b"data" and not part.exists()
