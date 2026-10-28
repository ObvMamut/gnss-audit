"""Arcs and the double-difference kernels of Route C (docs/3 sections 4.1 to 4.3)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

LLI_LOSS_OF_LOCK = 0x01
LLI_HALF_CYCLE = 0x02  # half-cycle ambiguity possible: the phase of that epoch is not usable


def arc_ids(t_s: FloatArray, lli: NDArray[np.uint8], usable: NDArray[np.bool_], max_gap_s: float) -> IntArray:
    """Arc index per epoch of one satellite-signal series sorted by time; -1 where the epoch is not usable.

    A new arc starts at loss of lock (bit 0 of the RINEX LLI digit; other bits do not break an arc), after an
    unusable epoch, or after a gap longer than max_gap_s.
    """
    ids = np.full(t_s.size, -1, dtype=np.int64)
    current, previous_t = -1, -np.inf
    for k in range(t_s.size):
        if not usable[k]:
            previous_t = -np.inf
            continue
        if lli[k] & LLI_LOSS_OF_LOCK or t_s[k] - previous_t > max_gap_s:
            current += 1
        ids[k] = current
        previous_t = t_s[k]
    return ids


def dd_operator(n_sat: int, ref: int) -> FloatArray:
    """D of shape (n_sat - 1, n_sat): each row is satellite j minus the reference, j != ref."""
    others = [j for j in range(n_sat) if j != ref]
    d = np.zeros((n_sat - 1, n_sat))
    d[np.arange(n_sat - 1), others] = 1.0
    d[:, ref] = -1.0
    return d


def remove_arc_mean(v: FloatArray, arc: IntArray) -> tuple[FloatArray, FloatArray]:
    """Float-ambiguity removal: subtract each arc's mean; return the centred series and the variance inflation
    n_arc / (n_arc - 1) per epoch (docs/3 section 4.2). Epochs with arc -1 come back as NaN."""
    out = np.full(v.size, np.nan)
    inflation = np.full(v.size, np.nan)
    for a in np.unique(arc[arc >= 0]):
        sel = arc == a
        n = int(sel.sum())
        out[sel] = v[sel] - v[sel].mean()
        inflation[sel] = n / (n - 1) if n > 1 else np.nan
    return out, inflation


def whiten(v: FloatArray, cov: FloatArray) -> FloatArray:
    """u = L^-1 v with L L^T = cov (docs/3 section 4.3)."""
    return np.linalg.solve(np.linalg.cholesky(cov), v)
