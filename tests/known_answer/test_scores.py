"""KA-crps and U-metrics (docs/5 section 3, docs/11 sections 3 and 4)."""

import numpy as np
import pytest
from scipy import integrate, stats

from gnssaudit.eval.scores import coverage, crps, log_score, pit


def crps_numeric(e: float, scale: float, nu: float) -> float:
    cdf = (lambda x: stats.norm.cdf(x / scale)) if nu == 0 else (lambda x: stats.t.cdf(x / scale, nu))
    below = integrate.quad(lambda x: cdf(x) ** 2, -np.inf, e, epsabs=1e-14, epsrel=1e-13, limit=500)[0]
    above = integrate.quad(lambda x: (1 - cdf(x)) ** 2, e, np.inf, epsabs=1e-14, epsrel=1e-13, limit=500)[0]
    return below + above


@pytest.mark.parametrize("nu", [0.0, 2.5, 5.0, 20.0])
@pytest.mark.parametrize("scale", [0.5, 3.0])
@pytest.mark.parametrize("e", [-4.0, 0.0, 1.3, 5.0])
def test_crps_closed_form_matches_integration(nu: float, scale: float, e: float) -> None:
    got = crps(np.array([e]), np.array([scale]), np.array([nu]))[0]
    assert got == pytest.approx(crps_numeric(e, scale, nu), rel=1e-9)


def test_crps_rejects_nu_at_most_one() -> None:
    with pytest.raises(ValueError):
        crps(np.array([0.0]), np.array([1.0]), np.array([1.0]))


def test_metrics_on_hand_computed_examples() -> None:
    e = np.array([0.5, -1.5, 2.5, -0.1])
    scale = np.ones(4)
    nu = np.zeros(4)
    # |e| <= 1.0 for 2 of 4; |e| <= 1.96 for 3 of 4
    assert coverage(e, scale, nu, [0.6827, 0.95]) == pytest.approx([0.5, 0.75], abs=1e-12)
    assert pit(np.array([0.0]), np.array([2.0]), np.array([0.0]))[0] == 0.5
    assert log_score(np.array([0.0]), np.array([1.0]), np.array([0.0]))[0] == pytest.approx(0.5 * np.log(2 * np.pi))
    assert log_score(np.array([0.0]), np.array([2.0]), np.array([0.0]))[0] == pytest.approx(
        0.5 * np.log(2 * np.pi) + np.log(2.0)
    )
    t_cov = coverage(np.array([2.5]), np.array([1.0]), np.array([5.0]), [0.95])[0]
    assert t_cov == 1.0  # t(5) 97.5 % quantile is 2.571


def test_calibrated_gaussian_has_nominal_coverage_and_uniform_pit() -> None:
    rng = np.random.default_rng(0)
    scale = rng.uniform(0.5, 2.0, 200_000)
    e = rng.standard_normal(scale.size) * scale
    nu = np.zeros(scale.size)
    assert coverage(e, scale, nu, [0.68, 0.95, 0.99]) == pytest.approx([0.68, 0.95, 0.99], abs=0.004)
    hist, _ = np.histogram(pit(e, scale, nu), bins=10, range=(0, 1))
    assert hist / e.size == pytest.approx(np.full(10, 0.1), abs=0.003)
