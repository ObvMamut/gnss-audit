"""Where a twin's samples go: a file, normally in /dev/shm (D-10; docs/8 section 6, step 6).

The stream is written to `<name>.part` and renamed only after fsync, so a crash or a failed gate never leaves a partial
twin under its final name. A FIFO sink is not offered: GNSS-SDR's FIFO source never ends its flowgraph at end of
stream (docs/9 section 10), so a FIFO run would have to be killed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from gnssaudit.io.fetch import commit_part

SHM = Path("/dev/shm")


@dataclass(frozen=True)
class SinkSpec:
    kind: Literal["shm_file", "file"]
    path: Path
    item_type: Literal["ishort"] = "ishort"  # interleaved int16 I and Q, GNSS-SDR's `ishort`

    def __post_init__(self) -> None:
        if self.kind == "shm_file" and not self.path.is_relative_to(SHM):
            raise ValueError(f"an shm_file sink must live under {SHM}, not {self.path}")


@contextmanager
def open_sink(sink: SinkSpec) -> Iterator[BinaryIO]:
    sink.path.parent.mkdir(parents=True, exist_ok=True)
    part = sink.path.with_name(sink.path.name + ".part")
    fh = part.open("wb")
    try:
        yield fh
        fh.flush()
        os.fsync(fh.fileno())
    except BaseException:
        fh.close()
        part.unlink(missing_ok=True)
        raise
    fh.close()
    commit_part(part, sink.path)
