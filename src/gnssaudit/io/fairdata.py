"""Prefix streaming from Fairdata IDA (docs/6 section 8.2; docs/7 section 2.3; P2b task 4).

The download service ignores HTTP Range, so a file can only be read from its start. `stream_prefix` writes the first
`prefix_bytes` of the stream, keeps hashing the rest without writing it, and checks the whole file against its
published size and SHA-256. The prefix is committed as `<name>.unverified` as soon as it is complete, so the admission
screen can start on it; it gets its final, read-only name only once the whole file is verified, so a prefix streamed
without full verification (`verify_full=False`, the CLI's --no-verify) keeps the `.unverified` name. A connection that
drops is re-opened from the start; a prefix already committed must then come back byte-identical. The read-only sidecar
`<name>.prefix.json` records the prefix's hash and whether the file was verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from gnssaudit.io.fetch import ChecksumError, commit_part

AUTHORIZE_URL = "https://etsin.fairdata.fi/api/download/authorize"
RESERVE_BYTES = 40 * 10**9  # kept free on the store's file system (docs/6 section 8.2, item 6)
CHUNK = 1 << 20


class StorageGuardError(RuntimeError):
    """Writing would leave less than the reserve free on the target file system."""


@dataclass(frozen=True)
class PrefixResult:
    path: Path
    prefix_bytes: int
    prefix_sha256: str
    full_sha256: str | None  # None when the stream stopped at the prefix
    verified: bool
    wall_s: float


def authorize(metax_id: str, record_path: str, *, timeout_s: float = 60.0) -> str:
    """A single-use download URL for one file of an open Fairdata dataset (the anonymous call the Etsin page makes)."""
    body = json.dumps({"cr_id": metax_id, "file": record_path}).encode()
    req = urllib.request.Request(AUTHORIZE_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return str(json.loads(resp.read())["url"])


def require_free(path: Path, extra_bytes: int, reserve: int = RESERVE_BYTES) -> None:
    """Refuse to write extra_bytes under path if less than `reserve` bytes would stay free on its file system. The check
    is per call: writers in parallel (several streams) each check their own bytes, so their total is the caller's to
    check."""
    free = shutil.disk_usage(path).free
    if free - extra_bytes < reserve:
        raise StorageGuardError(
            f"{path}: writing {extra_bytes / 1e9:.1f} GB would leave {(free - extra_bytes) / 1e9:.1f} GB free, "
            f"below the {reserve / 1e9:.0f} GB reserve"
        )


def _discard(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            p.chmod(0o644)
            p.unlink()


def stream_prefix(
    open_stream: Callable[[], BinaryIO],
    dest: Path,
    *,
    prefix_bytes: int,
    size_bytes: int,
    sha256: str,
    verify_full: bool = True,
    retries: int = 3,
    reserve: int = RESERVE_BYTES,
    progress: Callable[[int], None] | None = None,
) -> PrefixResult:
    """Write the first prefix_bytes of the stream to dest and verify the whole file (see the module docstring)."""
    if not 0 < prefix_bytes <= size_bytes:
        raise ValueError(f"prefix of {prefix_bytes} bytes for a file of {size_bytes} bytes")
    if dest.exists():
        raise FileExistsError(f"{dest} exists; the raw store is write-once")
    dest.parent.mkdir(parents=True, exist_ok=True)
    require_free(dest.parent, prefix_bytes, reserve)
    part = dest.with_name(dest.name + ".part")
    unverified = dest.with_name(dest.name + ".unverified")
    sidecar = dest.with_name(dest.name + ".prefix.json")
    t0 = time.perf_counter()
    state = _Pass()  # survives a dropped pass: a prefix committed before the drop stays committed
    try:
        for attempt in range(retries):
            try:
                full = _one_pass(open_stream, part, unverified, prefix_bytes, size_bytes, verify_full, state,
                                 progress)  # fmt: skip
                break
            except ChecksumError:
                raise
            except OSError:
                _discard(part)
                if attempt == retries - 1:
                    raise
                time.sleep(2.0**attempt)
        else:  # retries == 0
            raise ValueError("retries must be at least 1")
        if full is not None and full != sha256.lower():
            raise ChecksumError(f"{dest.name}: sha256 {full} does not match the published {sha256}")
    except BaseException:
        _discard(part, unverified, sidecar)  # the sidecar of an earlier --no-verify prefix described what is discarded
        raise
    verified = full is not None
    path = dest if verified else unverified  # the final, read-only name means verified
    if verified:
        commit_part(unverified, dest)
        dest.chmod(dest.stat().st_mode & 0o555)
    side = {
        "name": dest.name,
        "prefix_bytes": prefix_bytes,
        "prefix_sha256": state.committed,
        "size_bytes": size_bytes,
        "published_sha256": sha256.lower(),
        "full_sha256": full,
        "verified": verified,
        "utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    tmp = sidecar.with_name(sidecar.name + ".part")
    _discard(tmp)  # a crashed earlier start's, read-only
    tmp.write_text(json.dumps(side, indent=2) + "\n")
    tmp.chmod(0o444)
    commit_part(tmp, sidecar)  # a rename: it also replaces an earlier --no-verify sidecar, read-only as it is
    return PrefixResult(path, prefix_bytes, str(state.committed), full, verified, round(time.perf_counter() - t0, 3))


@dataclass
class _Pass:
    committed: str | None = None  # SHA-256 of the prefix once it is on disk


def _one_pass(
    open_stream: Callable[[], BinaryIO],
    part: Path,
    unverified: Path,
    prefix_bytes: int,
    size_bytes: int,
    verify_full: bool,
    state: _Pass,
    progress: Callable[[int], None] | None,
) -> str | None:
    """One read of the stream from its start: the full SHA-256, or None when it stops at the prefix. Raises OSError
    on a dropped or short stream (retried by the caller), ChecksumError on a longer file or a changed prefix."""
    h_full = hashlib.sha256()
    h_pre: hashlib._Hash | None = hashlib.sha256()
    total = 0
    out = None if state.committed is not None else part.open("wb")
    try:
        with open_stream() as src:
            while True:
                chunk = src.read(CHUNK)
                if not chunk:
                    break
                head = chunk[: max(0, prefix_bytes - total)]
                if head and h_pre is not None:
                    h_pre.update(head)
                    if out is not None:
                        out.write(head)
                h_full.update(chunk)
                total += len(chunk)
                if total > size_bytes:
                    raise ChecksumError(f"the stream holds more than the published {size_bytes} bytes")
                if total >= prefix_bytes and h_pre is not None:
                    pre = h_pre.hexdigest()
                    h_pre = None
                    if out is not None:
                        out.flush()
                        os.fsync(out.fileno())
                        out.close()
                        out = None
                        commit_part(part, unverified)
                        state.committed = pre
                    elif pre != state.committed:
                        raise ChecksumError(f"the re-streamed prefix hashes {pre}, not the committed {state.committed}")
                    if not verify_full:
                        return None
                if progress is not None:
                    progress(total)
    finally:
        if out is not None:
            out.close()
    if total < size_bytes:
        raise OSError(f"the stream ended after {total} of {size_bytes} bytes")
    return h_full.hexdigest()
