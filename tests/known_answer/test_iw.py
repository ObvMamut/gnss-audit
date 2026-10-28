"""KA-nu, P-iw-identity: in-phase witness building blocks (docs/4 section 3)."""

import numpy as np
import pytest
from scipy import signal

from gnssaudit.uq import iw
from gnssaudit.uq.loop import CarrierLoop


def test_window_moments_match_monte_carlo() -> None:
    loop = CarrierLoop.textbook(5.0, 1e-3)
    n_win = 1000
    gamma = iw.autocov_white_through(loop.impulse_response(60_000), n_win)
    factor, nu = iw.window_variance_moments(gamma, n_win)
    rng = np.random.default_rng(0)
    x = loop.filter(rng.standard_normal(2_000_000))[50_000:]
    blocks = x[: (x.size // n_win) * n_win].reshape(-1, n_win)
    s2 = blocks.var(axis=1)
    # about 1950 windows with relative SD sqrt(2/11): SE of the mean ratio about 0.01
    assert np.mean(s2) / gamma[0] == pytest.approx(factor, abs=0.025)
    assert 2 * np.mean(s2) ** 2 / np.var(s2) == pytest.approx(nu, rel=0.12)


def test_mean_removal_factor_is_one_minus_one_over_two_bn_w() -> None:
    # finding F5: the factor is 1 - 1/(2 B_n,eff W), not 1 - 1/(B_n,eff W) as written in docs/4 P4
    loop = CarrierLoop.textbook(5.0, 1e-3)
    for w_s in (1.0, 2.0, 5.0, 10.0):
        n_win = int(w_s / 1e-3)
        gamma = iw.autocov_white_through(loop.impulse_response(80_000), n_win)
        factor, _ = iw.window_variance_moments(gamma, n_win)
        assert factor == pytest.approx(1 - 1 / (2 * loop.bn_eff_hz() * w_s), abs=0.015)
        assert abs(factor - (1 - 1 / (loop.bn_eff_hz() * w_s))) > 0.04 / w_s  # the formula written in docs/4


def test_exact_nu_values_for_textbook_loop() -> None:
    # KA-nu, exact: 10.9 / 21.1 / 51.6 / 102.7 at W = 1 / 2 / 5 / 10 s (docs/4 Monte Carlo gave
    # 10.9 / 20.5 / 48.1 / 97.1, low by up to 7 %); the asymptote is 2 B_n,eff W
    loop = CarrierLoop.textbook(5.0, 1e-3)
    h = loop.impulse_response(80_000)
    for w_s, ref in [(1.0, 10.90), (2.0, 21.11), (5.0, 51.61), (10.0, 102.70)]:
        n_win = int(w_s / 1e-3)
        _, nu = iw.window_variance_moments(iw.autocov_white_through(h, n_win), n_win)
        assert nu == pytest.approx(ref, rel=2e-3)
        assert nu == pytest.approx(2 * loop.bn_eff_hz() * w_s, rel=0.09)


def test_witness_identity_long_run_for_simplified_loop() -> None:
    # P2 with c = 0: Var(e) = Var(d) - Var((1-H) m) + Var(H m), circular coloured noise plus phase-only process
    loop = CarrierLoop.gnsssdr(5.0, 1e-3, kappa=0.0)
    rng = np.random.default_rng(2)
    n = 2_000_000
    pole = np.exp(-2 * np.pi * 3.0 * 1e-3)  # AR(1), 3 Hz corner
    scale = 5.0
    m = scale * np.asarray(signal.lfilter([1 - pole], [1, -pole], rng.standard_normal(n))) + 0.05 * rng.standard_normal(
        n
    )
    q = scale * np.asarray(signal.lfilter([1 - pole], [1, -pole], rng.standard_normal(n))) + 0.05 * rng.standard_normal(
        n
    )
    theta = 2e-3 * np.cumsum(rng.standard_normal(n))  # phase-only random walk (oscillator-like)
    e = loop.filter(theta + q) - theta
    d = loop.filter(theta + q, "disc")
    est = iw.identity_variance(d, m, loop, skip=100_000)
    assert est == pytest.approx(np.var(e[100_000:]), rel=0.03)
