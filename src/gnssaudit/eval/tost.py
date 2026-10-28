"""Equivalence of coverage with its nominal level by two one-sided tests [47] (docs/5 section 4.2, D-14)."""

from __future__ import annotations

from scipy import stats


def tost_coverage(c_hat: float, se: float, p: float, delta: float = 0.02, alpha: float = 0.05) -> bool:
    """True when both one-sided tests reject: the (1 - 2 alpha) interval of c lies inside (p - delta, p + delta).

    se is the standard error of c_hat, from the block bootstrap or, for independent indicators, at the observed
    coverage; a standard error taken at the nominal p inflates the size when the true coverage is below p.
    """
    z = float(stats.norm.ppf(1 - alpha))
    return bool(c_hat - z * se > p - delta and c_hat + z * se < p + delta)
