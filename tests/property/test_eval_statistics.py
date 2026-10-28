"""tau_int, the power rule, KA-cov, M-bootstrap and M-tost (docs/5 sections 4 and 5, docs/11 sections 3 and 7)."""

import numpy as np
import pytest
from scipy import stats

from gnssaudit.eval.bootstrap import Hierarchy, bootstrap_ci, resample
from gnssaudit.eval.dependence import audit_label, n_eff, tau_int_s, wilson_half_width
from gnssaudit.eval.tost import tost_coverage


def ar1(rng: np.random.Generator, phi: float, n: int) -> np.ndarray:
    """Unit-variance AR(1)."""
    x = np.empty(n)
    x[0] = rng.standard_normal()
    eps = rng.standard_normal(n) * np.sqrt(1 - phi**2)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + eps[k]
    return x


def test_tau_int_of_ar1() -> None:
    # tau_int = (1 + phi) / (1 - phi) samples: 19 at phi = 0.9
    x = ar1(np.random.default_rng(0), 0.9, 400_000)
    assert tau_int_s(x, 0.1) == pytest.approx(1.9, rel=0.05)
    assert n_eff(x, 0.1) == pytest.approx(400_000 / 19, rel=0.05)


def test_constant_series_has_no_dependence() -> None:
    assert tau_int_s(np.ones(500), 0.1) == 0.1
    assert n_eff(np.zeros(500), 0.1) == 500


def test_power_rule_numbers_of_docs5() -> None:
    # F17: at c = 0.85, n_eff = 200 gives +-0.049; +-0.03 needs about 544
    assert wilson_half_width(0.85, 200) == pytest.approx(0.049, abs=0.001)
    assert wilson_half_width(0.85, 560) <= 0.03 < wilson_half_width(0.85, 520)
    assert audit_label(0.95, 250) == "audited"  # Wilson at c = 0.95 needs about 215
    assert audit_label(0.85, 300) == "case study"


def arcs_of_calibrated_errors(rng: np.random.Generator, n_arcs: int, n: int, phi: float) -> np.ndarray:
    return np.stack([ar1(rng, phi, n) for _ in range(n_arcs)])


def test_resample_keeps_size_and_group_structure() -> None:
    labels = np.repeat(np.arange(4), 25)
    idx = resample(Hierarchy(levels=[labels], block_len=7), np.random.default_rng(1))
    assert idx.size == 100
    # each resampled arc is a run of 25 indices from one original arc
    assert all(len(set(labels[idx[k : k + 25]])) == 1 for k in range(0, 100, 25))


def test_calibrated_model_ci_contains_nominal_coverage() -> None:
    # KA-cov: perfectly calibrated Gaussian predictive on AR(1) errors
    rng = np.random.default_rng(2)
    z = arcs_of_calibrated_errors(rng, 40, 500, 0.9).ravel()
    inside = np.abs(z) <= stats.norm.ppf(0.975)
    h = Hierarchy(levels=[np.repeat(np.arange(40), 500)], block_len=57)
    ci = bootstrap_ci(lambda idx: float(inside[idx].mean()), h, n_boot=500, seed=3)
    assert ci.lo <= 0.95 <= ci.hi


@pytest.mark.slow
def test_bootstrap_interval_coverage() -> None:
    # M-bootstrap: 95 % intervals cover the true coverage in 93 % to 97 % of 500 replicates
    rng = np.random.default_rng(4)
    q = stats.norm.ppf(0.975)
    h = Hierarchy(levels=[np.repeat(np.arange(20), 200)], block_len=57)
    hits = 0
    for r in range(500):
        inside = np.abs(arcs_of_calibrated_errors(rng, 20, 200, 0.9).ravel()) <= q
        ci = bootstrap_ci(lambda idx, inside=inside: float(inside[idx].mean()), h, n_boot=400, seed=r)
        hits += ci.lo <= 0.95 <= ci.hi
    assert 0.93 <= hits / 500 <= 0.97


def test_tost_size_and_power() -> None:
    # M-tost: size <= alpha at the margin, power about 0.8 at zero deviation, n_eff = 1000
    rng = np.random.default_rng(5)
    n, p, delta, sims = 1000, 0.95, 0.02, 40_000

    def rate(p_true: float) -> float:
        c = rng.binomial(n, p_true, sims) / n
        se = np.sqrt(c * (1 - c) / n)
        return float(np.mean([tost_coverage(ci, si, p, delta) for ci, si in zip(c, se, strict=True)]))

    mc_se = np.sqrt(0.05 * 0.95 / sims)
    # Binomial lattice: the lower margin has size 0.057 at n = 1000 (0.055 to 0.061 up to n = 10 000), so the
    # tolerance is alpha + 1 point there (docs/11 section 7, amended); the upper margin stays below alpha.
    assert rate(p - delta) <= 0.06 + 3 * mc_se
    assert rate(p + delta) <= 0.05 + 3 * mc_se
    assert rate(p) == pytest.approx(0.8, abs=0.05)
