"""Real intermediate-frequency samples to complex baseband (P2b part 2, task 8; docs/7 section 4.2).

SpoofRepo's L1 file holds real 2-bit samples (in 8-bit containers) at an intermediate frequency, with the signal at
`centre + if_hz` (high side). The conversion is x[n] exp(-j 2 pi if_hz n / fs), a low-pass FIR and a decimation,
written as interleaved little-endian int16 (I then Q), GNSS-SDR's `ishort`. A tone at if_hz + df comes out at +df.

- **Mix.** The oscillator is a table over one period of the (rational) ratio if_hz / fs, indexed by the absolute sample
  number, so a sample's phase does not depend on how the file was read.
- **DC.** The recorder's DC offset lands at -if_hz after the mix, in the stop band.
- **Chunk independence.** The filter runs on a fixed grid of GRID input samples, with the previous block's last taps - 1
  mixed samples as its history (zeros before the start), whatever `chunk` the file is read in; the output and its
  SHA-256 are bitwise the same for every chunk size.
- **Scale.** Output counts = scale x input counts (OUT_SCALE by default). For 2-bit input (|x| <= 3) and a unit-gain
  FIR the output stays under about 5 x 4096 = 20 000 counts of 32 767, while the noise (SD about 1.5 counts at the
  input) is thousands of counts: no underflow and no clipping. A clip raises.
"""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray
from scipy import signal

GRID = 1 << 18  # input samples per filter block; the decimation must divide it
OUT_SCALE = 4096.0  # int16 counts per input count
INT16_MAX = 32767
KAISER_BETA = 6.0  # about 60 dB of stop band


def _taps(taps: int, decim: int, fs_hz: float) -> NDArray[np.float64]:
    """Unit-DC-gain Kaiser low-pass whose -6 dB point is 0.9 of the output Nyquist band."""
    if (taps - 1) % decim:
        raise ValueError(f"taps - 1 = {taps - 1} must be a multiple of decim = {decim}")
    m = np.arange(taps) - (taps - 1) / 2
    h = 0.9 / decim * np.sinc(0.9 / decim * m) * np.kaiser(taps, KAISER_BETA)  # cutoff in units of fs / 2
    return np.asarray(h / h.sum(), dtype=np.float64)


def _oscillator(if_hz: float, fs_hz: float) -> NDArray[np.complex128]:
    """exp(-j 2 pi k num / den) for k = 0..den-1, where if_hz / fs_hz = num / den (one period of the oscillator)."""
    r = Fraction(if_hz / fs_hz).limit_denominator(1 << 22)
    if abs(float(r) - if_hz / fs_hz) > 1e-12:
        raise ValueError(f"if_hz / fs_hz = {if_hz / fs_hz} is not a ratio of small integers")
    k = np.arange(r.denominator, dtype=np.int64)
    return np.exp(-2j * np.pi * ((k * r.numerator) % r.denominator) / r.denominator)


def to_baseband(src: Path, dst: Path, *, fs_hz: float, if_hz: float, decim: int, taps: int, sample_end: int,
                chunk: int = 1 << 22, scale: float = OUT_SCALE) -> str:  # fmt: skip
    """Convert samples [0, sample_end) of the real int8 file `src` to int16 IQ in `dst` at fs_hz / decim, and return the
    SHA-256 of what was written. Output sample m is the filtered baseband at input sample m x decim (zero history
    before the start), so there are ceil(sample_end / decim) of them."""
    if GRID % decim:
        raise ValueError(f"decim = {decim} must divide the block of {GRID} samples")
    h = _taps(taps, decim, fs_hz)
    osc = _oscillator(if_hz, fs_hz)
    den = osc.size
    hist = taps - 1  # a multiple of decim
    prev = np.zeros(hist, dtype=np.complex128)
    digest = hashlib.sha256()
    buf = np.empty(0, dtype=np.int8)
    done = 0  # input samples already mixed and filtered

    def block(x: NDArray[np.int8], out_file: BinaryIO) -> None:
        nonlocal prev, done
        n = (done + np.arange(x.size, dtype=np.int64)) % den
        mixed = x.astype(np.float64) * osc[n]
        ext = np.concatenate([prev, mixed])
        full = signal.upfirdn(h, ext, up=1, down=decim)
        first = hist // decim
        y = full[first : first + -(-x.size // decim)] * scale
        iq = np.empty(2 * y.size, dtype="<i2")
        if np.abs(y.real).max(initial=0) > INT16_MAX or np.abs(y.imag).max(initial=0) > INT16_MAX:
            raise ValueError(f"{src}: the output would clip int16 at scale {scale} (peak {np.abs(y).max():.0f})")
        iq[0::2] = np.round(y.real)
        iq[1::2] = np.round(y.imag)
        raw = iq.tobytes()
        digest.update(raw)
        out_file.write(raw)
        prev = ext[-hist:] if hist else prev
        done += x.size

    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        read = 0
        while read < sample_end:
            part = np.fromfile(f_in, dtype=np.int8, count=min(chunk, sample_end - read))
            if part.size == 0:
                raise ValueError(f"{src}: ends at {read} samples, before sample_end = {sample_end}")
            read += part.size
            buf = np.concatenate([buf, part]) if buf.size else part
            while buf.size >= GRID:
                block(buf[:GRID], f_out)
                buf = buf[GRID:]
        if buf.size:
            block(buf, f_out)
    return digest.hexdigest()
