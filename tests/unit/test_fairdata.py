"""Prefix streaming from Fairdata (P2b task 4): the service ignores HTTP Range, so a file is read from its start; only
its first bytes are written, the whole file is hashed and checked against the published SHA-256 (docs/6 section 8.2).
The HTTP stream is injected, so these tests never touch the network."""

import hashlib
import io
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import pytest

from gnssaudit.io import fairdata
from gnssaudit.io.fairdata import StorageGuardError, require_free, stream_prefix
from gnssaudit.io.fetch import ChecksumError

DATA = bytes(range(256)) * 4096  # 1 MiB with structure, so a misplaced byte changes the hash
SHA = hashlib.sha256(DATA).hexdigest()
PREFIX = 300_000


def opener(*payloads: bytes) -> Callable[[], BinaryIO]:
    """Successive calls return successive payloads (the last one repeats)."""
    calls = iter(payloads)
    last = [payloads[-1]]

    def open_stream() -> BinaryIO:
        last[0] = next(calls, last[0])
        return io.BytesIO(last[0])

    return open_stream


class Dying(io.BytesIO):
    """A stream whose connection drops after `limit` bytes."""

    def __init__(self, data: bytes, limit: int) -> None:
        super().__init__(data)
        self.limit = limit

    def read(self, n: int | None = -1) -> bytes:
        if self.tell() >= self.limit:
            raise ConnectionResetError("connection reset by peer")
        n = self.limit - self.tell() if n is None or n < 0 else min(n, self.limit - self.tell())
        return super().read(n)


class Counting(io.BytesIO):
    """Counts the bytes served (the stream is closed by the reader)."""

    served = 0

    def read(self, n: int | None = -1) -> bytes:
        out = super().read(n)
        self.served += len(out)
        return out


def files(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


def test_the_prefix_is_written_and_the_whole_file_verified(tmp_path: Path) -> None:
    dest = tmp_path / "f.iq"
    r = stream_prefix(opener(DATA), dest, prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, reserve=0)
    assert dest.read_bytes() == DATA[:PREFIX]
    assert r.verified and r.full_sha256 == SHA and r.prefix_sha256 == hashlib.sha256(DATA[:PREFIX]).hexdigest()
    assert not dest.stat().st_mode & 0o222  # read-only once verified
    side = json.loads((tmp_path / "f.iq.prefix.json").read_text())
    assert side["verified"] and side["prefix_bytes"] == PREFIX and side["size_bytes"] == len(DATA)
    assert side["published_sha256"] == SHA and side["full_sha256"] == SHA
    assert files(tmp_path) == {"f.iq", "f.iq.prefix.json"}


def test_a_wrong_full_hash_leaves_no_usable_file(tmp_path: Path) -> None:
    dest = tmp_path / "f.iq"
    with pytest.raises(ChecksumError):
        stream_prefix(opener(DATA), dest, prefix_bytes=PREFIX, size_bytes=len(DATA), sha256="0" * 64, reserve=0)
    assert files(tmp_path) == set()


def test_a_longer_file_than_published_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ChecksumError, match="bytes"):
        stream_prefix(
            opener(DATA + b"x"), tmp_path / "f.iq", prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, reserve=0
        )
    assert files(tmp_path) == set()


def test_a_stream_that_keeps_ending_short_raises_after_its_retries(tmp_path: Path) -> None:
    short = DATA[: len(DATA) // 2]
    with pytest.raises(OSError, match="ended after"):
        stream_prefix(opener(short), tmp_path / "f.iq", prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA,
                      retries=2, reserve=0)  # fmt: skip
    assert files(tmp_path) == set()


def test_a_drop_after_the_prefix_is_retried_and_the_prefix_kept(tmp_path: Path) -> None:
    """The prefix is committed (readable as .unverified) before the drop; the retry re-streams from the start, finds
    the same prefix bytes and does not rewrite them."""
    calls = []

    def open_stream() -> BinaryIO:
        calls.append(1)
        return Dying(DATA, 700_000) if len(calls) == 1 else io.BytesIO(DATA)

    dest = tmp_path / "f.iq"
    r = stream_prefix(open_stream, dest, prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, reserve=0)
    assert len(calls) == 2 and r.verified and dest.read_bytes() == DATA[:PREFIX]


def test_a_drop_inside_the_prefix_restarts_the_prefix(tmp_path: Path) -> None:
    calls = []

    def open_stream() -> BinaryIO:
        calls.append(1)
        return Dying(DATA, 100_000) if len(calls) == 1 else io.BytesIO(DATA)

    dest = tmp_path / "f.iq"
    stream_prefix(open_stream, dest, prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, reserve=0)
    assert dest.read_bytes() == DATA[:PREFIX]


def test_a_re_streamed_prefix_that_differs_is_refused(tmp_path: Path) -> None:
    other = b"\xff" + DATA[1:]  # DATA starts with 0x00
    calls = []

    def open_stream() -> BinaryIO:
        calls.append(1)
        return Dying(DATA, 700_000) if len(calls) == 1 else io.BytesIO(other)

    with pytest.raises(ChecksumError, match="prefix"):
        stream_prefix(open_stream, tmp_path / "f.iq", prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, reserve=0)
    assert files(tmp_path) == set()


def test_the_prefix_is_readable_before_the_whole_file_is_verified(tmp_path: Path) -> None:
    """The screen may start on the prefix while the rest of the file is still being hashed."""
    seen: list[set[str]] = []

    def progress(n: int) -> None:
        if n >= 600_000 and not seen:
            seen.append(files(tmp_path))

    stream_prefix(opener(DATA), tmp_path / "f.iq", prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA,
                  progress=progress, reserve=0)  # fmt: skip
    assert seen == [{"f.iq.unverified"}]


def test_without_full_verification_the_stream_stops_at_the_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fairdata, "CHUNK", 1 << 16)
    src = Counting(DATA)
    dest = tmp_path / "f.iq"
    r = stream_prefix(lambda: src, dest, prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, verify_full=False,
                      reserve=0)  # fmt: skip
    assert not r.verified and r.full_sha256 is None and src.served < len(DATA)
    # final review R2, I2: the final, read-only name means verified; an unchecked prefix keeps its .unverified name
    assert r.path == tmp_path / "f.iq.unverified" and not dest.exists()
    assert r.path.read_bytes() == DATA[:PREFIX]
    assert json.loads((tmp_path / "f.iq.prefix.json").read_text())["verified"] is False
    assert files(tmp_path) == {"f.iq.unverified", "f.iq.prefix.json"}


def test_an_existing_target_is_never_overwritten(tmp_path: Path) -> None:
    (tmp_path / "f.iq").write_bytes(b"old")
    with pytest.raises(FileExistsError):
        stream_prefix(opener(DATA), tmp_path / "f.iq", prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, reserve=0)
    assert (tmp_path / "f.iq").read_bytes() == b"old"


def test_the_storage_guard_keeps_the_reserve_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(fairdata.shutil, "disk_usage", lambda p: usage._replace(free=50 * 10**9))
    require_free(tmp_path, 9 * 10**9)  # 41 GB would remain
    with pytest.raises(StorageGuardError, match="40"):
        require_free(tmp_path, 11 * 10**9)  # 39 GB would remain
    with pytest.raises(StorageGuardError):
        stream_prefix(opener(DATA), tmp_path / "f.iq", prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA,
                      reserve=50 * 10**9)  # fmt: skip
    assert files(tmp_path) == set()


def test_authorize_returns_the_single_use_url(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = {}

    class Resp(io.BytesIO):
        def __enter__(self) -> Resp:
            return self

        def __exit__(self, *a: object) -> None:
            return None

    def urlopen(req: object, timeout: float) -> Resp:
        sent["data"] = json.loads(req.data)  # type: ignore[attr-defined]
        sent["url"] = req.full_url  # type: ignore[attr-defined]
        return Resp(b'{"url": "https://download.fairdata.fi:443/download?token=abc"}')

    monkeypatch.setattr(fairdata.urllib.request, "urlopen", urlopen)
    assert fairdata.authorize("cr-1", "/a/b.iq") == "https://download.fairdata.fi:443/download?token=abc"
    assert sent == {"data": {"cr_id": "cr-1", "file": "/a/b.iq"}, "url": fairdata.AUTHORIZE_URL}


def test_an_unverified_prefix_gets_its_final_name_once_a_later_stream_verifies_it(tmp_path: Path) -> None:
    """Final review R2, I2: after --no-verify, a full stream may still verify the file; the prefix then gets its final,
    read-only name and the sidecar says verified."""
    dest = tmp_path / "f.iq"
    stream_prefix(opener(DATA), dest, prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, verify_full=False,
                  reserve=0)  # fmt: skip
    r = stream_prefix(opener(DATA), dest, prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, reserve=0)
    assert r.verified and r.path == dest and dest.read_bytes() == DATA[:PREFIX] and not dest.stat().st_mode & 0o222
    assert json.loads((tmp_path / "f.iq.prefix.json").read_text())["verified"] is True
    assert files(tmp_path) == {"f.iq", "f.iq.prefix.json"}


def test_a_failed_verification_also_removes_the_unverified_prefixs_sidecar(tmp_path: Path) -> None:
    """Final review R2, I2: a stream that fails to verify an unverified prefix discards it, and the sidecar that
    described it goes with it."""
    dest = tmp_path / "f.iq"
    stream_prefix(opener(DATA), dest, prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA, verify_full=False,
                  reserve=0)  # fmt: skip
    with pytest.raises(ChecksumError):
        stream_prefix(opener(DATA), dest, prefix_bytes=PREFIX, size_bytes=len(DATA), sha256="0" * 64, reserve=0)
    assert files(tmp_path) == set()


def test_the_sidecar_is_read_only_like_the_prefix_it_vouches_for(tmp_path: Path) -> None:
    """Final review R2, minor 7."""
    for verify in (True, False):
        d = tmp_path / str(verify)
        d.mkdir()
        stream_prefix(opener(DATA), d / "f.iq", prefix_bytes=PREFIX, size_bytes=len(DATA), sha256=SHA,
                      verify_full=verify, reserve=0)  # fmt: skip
        assert not (d / "f.iq.prefix.json").stat().st_mode & 0o222
