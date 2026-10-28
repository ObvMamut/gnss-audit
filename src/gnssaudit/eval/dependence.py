"""Effective sample size and the audited / case-study rule (docs/5 sections 4.1 and 4.2)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats

FloatArray = NDArray[np.float64]


def autocovariance(x: FloatArray) -> FloatArray:
    """Biased (1/N) autocovariance at lags 0 to N - 1, by FFT."""
    y = x - x.mean()
    n = y.size
    f = np.fft.rfft(y, 2 * n)
    return np.fft.irfft(np.abs(f) ** 2)[:n] / n


def tau_int_s(x: FloatArray, t_s: float) -> float:
    """Integrated autocorrelation time T_s (1 + 2 sum rho_l) with Geyer's initial-positive-sequence truncation.

    A constant series (for example exceedance indicators that are all 0) has no measurable dependence: T_s.
    """
    g = autocovariance(x)
    if g[0] == 0:
        return t_s
    pairs = g[: 2 * (g.size // 2)].reshape(-1, 2).sum(axis=1)  # Gamma_m = gamma(2m) + gamma(2m + 1)
    stop = np.flatnonzero(pairs <= 0)
    pairs = pairs[: stop[0] if stop.size else pairs.size]
    return t_s * (-1.0 + 2.0 * pairs.sum() / g[0])


def n_eff(x: FloatArray, t_s: float) -> float:
    return x.size * t_s / tau_int_s(x, t_s)


def wilson_half_width(c: float, n: float, level: float = 0.95) -> float:
    z = stats.norm.ppf((1 + level) / 2)
    return float(z / (1 + z**2 / n) * np.sqrt(c * (1 - c) / n + z**2 / (4 * n**2)))


def audit_label(c_observed: float, n_effective: float, max_half_width: float = 0.03) -> str:
    """'audited' when the 95 % interval of c(0.95), at the observed coverage, is no wider than +-0.03 (F17)."""
    return "audited" if wilson_half_width(c_observed, n_effective) <= max_half_width else "case study"
