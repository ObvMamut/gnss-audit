"""Registered IQ recordings as read-only complex sample streams (docs/8 section 6, step 1)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

_DTYPE = {"ibyte": np.int8, "ishort": np.int16}  # interleaved I and Q


@dataclass(frozen=True)
class IQSource:
    """Samples [sample_start, sample_end) of an interleaved-IQ file, as GNSS-SDR's `item_type` names it. `q_first` marks
    a file that stores Q before I (the SpoofRepo L5 files, P2b task 6); it is read as I + jQ either way."""

    path: Path
    item_type: Literal["ibyte", "ishort"]
    fs_hz: float
    sample_start: int
    sample_end: int
    q_first: bool = False

    def __post_init__(self) -> None:
        n_file = self.path.stat().st_size // (2 * np.dtype(_DTYPE[self.item_type]).itemsize)
        if not 0 <= self.sample_start < self.sample_end <= n_file:
            raise ValueError(
                f"{self.path}: samples [{self.sample_start}, {self.sample_end}) outside the file's {n_file} samples"
            )


def read_iq(source: IQSource, start: int, n: int) -> NDArray[np.complex64]:
    """n complex samples from absolute sample index `start` (memory-mapped, read-only), all inside the source range."""
    if n < 0 or not (source.sample_start <= start and start + n <= source.sample_end):
        raise ValueError(f"[{start}, {start + n}) is outside the source range")
    mm = np.memmap(source.path, dtype=_DTYPE[source.item_type], mode="r")
    iq = np.asarray(mm[2 * start : 2 * (start + n)], dtype=np.float32)
    i, q = (iq[1::2], iq[0::2]) if source.q_first else (iq[0::2], iq[1::2])
    return (i + 1j * q).astype(np.complex64)


def iter_iq(source: IQSource, block: int) -> Iterator[tuple[int, NDArray[np.complex64]]]:
    """(absolute sample index, samples) in blocks of `block` samples over the source range."""
    for start in range(source.sample_start, source.sample_end, block):
        yield start, read_iq(source, start, min(block, source.sample_end - start))
