"""Synthetic GPS L1 C/A recordings with exactly known carrier phase (Route A, reduced: docs/3 section 5).

One satellite with random 50 bit/s data on white Gaussian noise, written as interleaved int8 I/Q. The carrier phase is
2 pi f_D t exactly and the code clock follows the carrier Doppler, so the receiver's tracking error is known without
any reference. The amplitude follows the calibration of the s17 identification runs: with sigma counts of noise per
quadrature, N0 = 2 sigma^2 / fs, so A^2 / N0 = C/N0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gnssaudit.signals import carrier_hz
from gnssaudit.sim.despread import ca_code

NOISE_COUNTS = 16.0
_CHUNK = 1 << 22


def write_ca_recording(
    path: Path,
    *,
    prn: int,
    doppler_hz: float,
    cn0_dbhz: float,
    seconds: float,
    fs_hz: float = 4.0e6,
    seed: int = 1,
    chip0: float = 211.7,
) -> None:
    """Write `seconds` of int8 IQ at fs_hz; chip0 is the code phase (chips) at sample 0."""
    rng = np.random.default_rng(seed)
    code = ca_code(prn)
    amp = np.sqrt(10 ** (cn0_dbhz / 10) / fs_hz) * NOISE_COUNTS * np.sqrt(2)
    f_code = 1.023e6 * (1 + doppler_hz / carrier_hz("G", "1C"))
    bits = np.where(rng.random(int(seconds * 50) + 2) < 0.5, -1.0, 1.0)
    n_total = int(seconds * fs_hz)
    with path.open("wb") as fh:
        for start in range(0, n_total, _CHUNK):
            t = np.arange(start, min(start + _CHUNK, n_total)) / fs_hz
            chips = chip0 + f_code * t
            s = amp * bits[(chips // (1023 * 20)).astype(np.int64)] * code[(chips // 1).astype(np.int64) % 1023]
            x = s * np.exp(2j * np.pi * doppler_hz * t)
            x = x + NOISE_COUNTS * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))
            iq = np.empty(2 * t.size, dtype=np.int8)
            iq[0::2] = np.clip(np.round(x.real), -127, 127)
            iq[1::2] = np.clip(np.round(x.imag), -127, 127)
            iq.tofile(fh)


@dataclass(frozen=True)
class SkySat:
    """One C/A signal of a synthetic sky; a spoofed duplicate is a second entry with the same PRN, another code
    phase or Doppler, and a later start."""

    prn: int
    doppler_hz: float
    cn0_dbhz: float
    chip0: float
    start_s: float = 0.0


def write_sky_recording(
    path: Path,
    sats: Sequence[SkySat],
    *,
    seconds: float,
    fs_hz: float,
    seed: int = 1,
    cw: tuple[float, float, float] | None = None,
) -> None:
    """Several C/A signals with random data on white noise, as interleaved int8 IQ (the same calibration as
    write_ca_recording); cw = (onset_s, f_hz, J/N0 dB-Hz) adds a tone from onset_s on."""
    rng = np.random.default_rng(seed)
    n0 = 2 * NOISE_COUNTS**2 / fs_hz
    n_total = int(seconds * fs_hz)
    plan = []
    for s in sats:
        bits = np.where(rng.random(int(seconds * 50) + 2) < 0.5, -1.0, 1.0)
        f_code = 1.023e6 * (1 + s.doppler_hz / carrier_hz("G", "1C"))
        plan.append((s, ca_code(s.prn), np.sqrt(10 ** (s.cn0_dbhz / 10) * n0), f_code, bits))
    with path.open("wb") as fh:
        for start in range(0, n_total, _CHUNK):
            t = np.arange(start, min(start + _CHUNK, n_total)) / fs_hz
            x = NOISE_COUNTS * (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size))
            for s, code, amp, f_code, bits in plan:
                chips = s.chip0 + f_code * t
                on = t >= s.start_s
                sig = amp * bits[(chips // (1023 * 20)).astype(np.int64)] * code[(chips // 1).astype(np.int64) % 1023]
                x = x + np.where(on, sig * np.exp(2j * np.pi * s.doppler_hz * t), 0.0)
            if cw is not None:
                onset, f_hz, jn0 = cw
                x = x + np.where(t >= onset, np.sqrt(10 ** (jn0 / 10) * n0) * np.exp(2j * np.pi * f_hz * t), 0.0)
            iq = np.empty(2 * t.size, dtype=np.int8)
            iq[0::2] = np.clip(np.round(x.real), -127, 127)
            iq[1::2] = np.clip(np.round(x.imag), -127, 127)
            iq.tofile(fh)
