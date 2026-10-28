"""Calibration and proper scores for Gaussian or Student-t predictives (docs/5 sections 3.1 and 3.3).

nu = 0 encodes the Gaussian (docs/7 section 5.5); Student-t needs nu > 1 for the CRPS.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats

FloatArray = NDArray[np.float64]


def _split(nu: FloatArray) -> tuple[NDArray[np.bool_], FloatArray]:
    gauss = nu == 0
    return gauss, np.where(gauss, 1.0e6, nu)  # placeholder dof keeps the t branch finite where unused


def two_sided_quantile(p: float, nu: FloatArray) -> FloatArray:
    gauss, dof = _split(nu)
    return np.where(gauss, stats.norm.ppf((1 + p) / 2), stats.t.ppf((1 + p) / 2, dof))


def coverage(e: FloatArray, scale: FloatArray, nu: FloatArray, levels: Sequence[float]) -> FloatArray:
    """c(p) = mean 1[|e| <= q_p scale] for each level p."""
    return np.array([np.mean(np.abs(e) <= two_sided_quantile(p, nu) * scale) for p in levels])


def pit(e: FloatArray, scale: FloatArray, nu: FloatArray) -> FloatArray:
    gauss, dof = _split(nu)
    z = e / scale
    return np.where(gauss, stats.norm.cdf(z), stats.t.cdf(z, dof))


def log_score(e: FloatArray, scale: FloatArray, nu: FloatArray) -> FloatArray:
    """-log f(e) (docs/5 section 3.3)."""
    gauss, dof = _split(nu)
    z = e / scale
    return np.where(gauss, -stats.norm.logpdf(z), -stats.t.logpdf(z, dof)) + np.log(scale)


def crps(e: FloatArray, scale: FloatArray, nu: FloatArray) -> FloatArray:
    """Closed-form CRPS [30, 64] (docs/5 section 3.3)."""
    gauss, dof = _split(nu)
    if np.any(~gauss & (dof <= 1)):
        raise ValueError("the Student-t CRPS needs nu > 1")
    z = e / scale
    g = z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    t_cdf, t_pdf = stats.t.cdf(z, dof), stats.t.pdf(z, dof)
    tail = 2 * np.sqrt(dof) * special.beta(0.5, dof - 0.5) / ((dof - 1) * special.beta(0.5, dof / 2) ** 2)
    t = z * (2 * t_cdf - 1) + 2 * t_pdf * (dof + z**2) / (dof - 1) - tail
    return scale * np.where(gauss, g, t)
