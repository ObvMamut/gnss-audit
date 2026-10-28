"""Chip-level despreading of interference against a GNSS code replica (docs/8 section 3, docs/9 section 9).

The prompt correlator of block k against a unit-amplitude replica c(t) exp(j 2 pi f_D t) is

    eta_k = (1 / N_s) sum_n x[k N_s + n] c[n] exp(-j 2 pi f_D (k N_s + n) / f_s),

so a unit-amplitude signal gives eta = 1 and white sample noise of power N_0 f_s gives E|eta|^2 = N_0 / T. The
interference series eta_k is computed once against the nominal replica and fed to the closed-loop simulator in
``gnssaudit.sim.correlator``. This is the correlator-domain approximation of docs/9 section 9: exact for the
carrier, and approximate only in the small code and frequency divergence of the tracking replica.

Waveforms are complex baseband relative to the band centre, with power expressed relative to the satellite's
carrier power C = 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]

# IS-GPS-200 G2 phase-selector taps (1-based stage numbers), PRN 1..32
_G2_TAPS: tuple[tuple[int, int], ...] = (
    (2, 6), (3, 7), (4, 8), (5, 9), (1, 9), (2, 10), (1, 8), (2, 9), (3, 10), (2, 3),
    (3, 4), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (1, 4), (2, 5), (3, 6), (4, 7),
    (5, 8), (6, 9), (1, 3), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 6), (2, 7),
    (3, 8), (4, 9),
)  # fmt: skip


def ca_code(prn: int) -> FloatArray:
    """GPS L1 C/A Gold code as +-1 chips (logical 0 -> +1), 1023 chips."""
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    t1, t2 = _G2_TAPS[prn - 1]
    out = np.empty(1023, dtype=int)
    for i in range(1023):
        out[i] = g1[9] ^ g2[t1 - 1] ^ g2[t2 - 1]
        f1 = g1[2] ^ g1[9]
        f2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = np.roll(g1, 1)
        g1[0] = f1
        g2 = np.roll(g2, 1)
        g2[0] = f2
    return (1 - 2 * out).astype(float)


@dataclass
class Despreader:
    """Despread x against PRN ``prn`` at a fixed carrier Doppler; one block per code period (1 ms)."""

    prn: int
    fs_hz: float = 2.046e6
    doppler_hz: float = 0.0
    chip_rate_hz: float = 1.023e6
    code: FloatArray | None = None  # override the C/A code (e.g. a second component's code)
    ns: int = field(init=False)
    chips_sampled: FloatArray = field(init=False, repr=False)
    _template: ComplexArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        chips = ca_code(self.prn) if self.code is None else np.asarray(self.code, dtype=float)
        self.ns = round(self.fs_hz * chips.size / self.chip_rate_hz)
        idx = np.floor(np.arange(self.ns) * self.chip_rate_hz / self.fs_hz).astype(int) % chips.size
        c = chips[idx]
        self.chips_sampled = c
        n = np.arange(self.ns)
        self._template = np.asarray(c * np.exp(-2j * np.pi * self.doppler_hz * n / self.fs_hz), dtype=np.complex128)

    @property
    def t_s(self) -> float:
        return self.ns / self.fs_hz

    def replica(self, n_blocks: int) -> ComplexArray:
        """Unit-amplitude signal with this code and Doppler (for tests)."""
        n = np.arange(self.ns * n_blocks)
        return np.asarray(
            np.tile(self.chips_sampled, n_blocks) * np.exp(2j * np.pi * self.doppler_hz * n / self.fs_hz),
            dtype=np.complex128,
        )

    def despread(self, x: ComplexArray, first_block: int = 0) -> ComplexArray:
        """eta_k for consecutive blocks of x (length a multiple of ns); first_block sets the carrier origin."""
        nb = x.size // self.ns
        blocks = np.asarray(x[: nb * self.ns]).reshape(nb, self.ns)
        k = np.arange(first_block, first_block + nb)
        rot = np.exp(-2j * np.pi * self.doppler_hz * k * self.ns / self.fs_hz)
        return rot * (blocks @ self._template) / self.ns


# ----------------------------------------------------------------------------- waveforms (power relative to C)
def white_noise(rng: np.random.Generator, n: int, power: float) -> ComplexArray:
    s = np.sqrt(power / 2.0)
    return s * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def cw(n: int, fs_hz: float, f_hz: float, power: float, phase: float = 0.0, start: int = 0) -> ComplexArray:
    t = (start + np.arange(n)) / fs_hz
    return np.sqrt(power) * np.exp(1j * (2 * np.pi * f_hz * t + phase))


def swept_cw(
    n: int,
    fs_hz: float,
    f0_hz: float,
    span_hz: float,
    period_s: float,
    power: float,
    phase: float = 0.0,
    offset_s: float = 0.0,
    start: int = 0,
    triangle: bool = False,
) -> ComplexArray:
    """Sawtooth (or triangle) frequency sweep f(t) = f0 + span * frac((t + offset) / period), continuous phase.

    The phase is the exact integral of f(t), evaluated in float64 from the absolute sample index so that chunks
    concatenate bit-for-bit.
    """
    t = (start + np.arange(n)) / fs_hz + offset_s
    u = t / period_s
    cyc = np.floor(u)
    fr = u - cyc
    if triangle:
        # f = f0 + span * (1 - |2 fr - 1|); integral over one period = span * period / 2
        tri_int = np.where(fr < 0.5, fr**2, 0.5 - (1 - fr) ** 2) * span_hz * period_s
        ph = f0_hz * t + cyc * span_hz * period_s / 2 + tri_int
    else:
        ph = f0_hz * t + cyc * span_hz * period_s / 2 + 0.5 * span_hz * period_s * fr**2
    return np.sqrt(power) * np.exp(1j * (2 * np.pi * ph + phase))


def pulsed(
    base: ComplexArray, fs_hz: float, width_s: float, prf_hz: float, offset_s: float = 0.0, start: int = 0
) -> ComplexArray:
    """Gate a waveform on for width_s every 1/prf_hz seconds (power while on is that of ``base``)."""
    t = (start + np.arange(base.size)) / fs_hz + offset_s
    on = np.mod(t * prf_hz, 1.0) < width_s * prf_hz
    return base * on
