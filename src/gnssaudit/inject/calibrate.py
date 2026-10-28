"""Power calibration relative to the recording's own noise floor (docs/8 section 2; D-12).

N0 is the two-sided noise density in sample power per Hz, read off a Welch PSD as the median over the central 80 % of
the front-end band. Levels are set as J/N0 (or C/N0 for a ghost) in dB-Hz, so the recorder's absolute gain cancels.
The median ignores narrowband lines and the real satellites; a C/A correlator does not, so the density it sees (the
PSD weighted by the C/A spectrum) is reported beside it, for predicting the C/N0 a receiver reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from gnssaudit.inject.source import IQSource, read_iq

FloatArray = NDArray[np.float64]
CA_CHIP_S = 1 / 1.023e6


@dataclass(frozen=True)
class NoiseFloor:
    n0_per_hz: float  # median over segments
    spread_db: float  # largest over smallest segment estimate
    n_segments: int
    ca_weighted_per_hz: float  # the density a C/A correlator sees, median over segments

    @property
    def line_excess_db(self) -> float:
        """How far the C/A-weighted density sits above the median floor: lines, real satellites, a non-flat floor."""
        return float(10 * np.log10(self.ca_weighted_per_hz / self.n0_per_hz))


def estimate_n0(
    source: IQSource, frontend_bw_hz: float, nperseg: int = 1 << 16, segment: int = 1 << 21, max_segments: int = 16
) -> NoiseFloor:
    """Welch (Hann, 50 % overlap) on up to max_segments evenly spread segments; per segment, the median density over
    |f| <= 0.4 frontend_bw_hz, and the density weighted by the unit-area C/A spectrum over the sampled band."""
    total = source.sample_end - source.sample_start
    seg = min(segment, total)
    k = max(1, min(max_segments, total // seg))
    starts = source.sample_start + (np.arange(k) * ((total - seg) // max(k - 1, 1))).astype(np.int64)
    n0s, cas = [], []
    for s in starts:
        x = read_iq(source, int(s), seg)
        f, p = signal.welch(x, fs=source.fs_hz, window="hann", nperseg=min(nperseg, seg), return_onesided=False)
        f, p = np.fft.fftshift(f), np.fft.fftshift(p)
        n0s.append(float(np.median(p[np.abs(f) <= 0.4 * frontend_bw_hz])))
        g = ca_psd(f)
        cas.append(float(np.trapezoid(p * g, f) / np.trapezoid(g, f)))
    arr = np.array(n0s)
    return NoiseFloor(float(np.median(arr)), float(10 * np.log10(arr.max() / arr.min())), k, float(np.median(cas)))


def gain(level_db_hz: float, n0_per_hz: float) -> float:
    """Amplitude of a unit-power waveform that sits level_db_hz above N0: sqrt(10^(L/10) N0) (docs/8 2.2, 2.3)."""
    return float(np.sqrt(10 ** (level_db_hz / 10) * n0_per_hz))


def ca_psd(f_hz: FloatArray) -> FloatArray:
    """Unit-area power spectrum of a C/A-like BPSK(1) signal, T_c sinc^2(f T_c)."""
    return CA_CHIP_S * np.sinc(f_hz * CA_CHIP_S) ** 2


def ssc_kappa(f_hz: FloatArray, psd_interf: FloatArray, psd_signal: FloatArray) -> float:
    """Spectral separation coefficient [2, 37]: the integral of the product of the two spectra, each normalized to
    unit area over the given band (the receiver's band)."""
    gi = psd_interf / np.trapezoid(psd_interf, f_hz)
    gs = psd_signal / np.trapezoid(psd_signal, f_hz)
    return float(np.trapezoid(gi * gs, f_hz))
