"""Regression tests for the propositions introduced by the mathematical review (docs/14).

Each test is a reduced, fast version of a review study; the full study and its numbers live in review/sNN_*.py.
"""

import numpy as np
import pytest

from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from gnssaudit.uq import iw
from gnssaudit.uq.loop import CarrierLoop
from review.s02_window_and_p5 import ar1, predictions
from review.s06_mismatch_law import CN0, predict_tone, simulate_tone

T = 1e-3


@pytest.mark.slow
def test_p5_single_projection_matches_monte_carlo() -> None:
    """F6: the identifiable fraction follows one projection weight 1 - |B_W|^2, not |1 - B_W|^4 (S02)."""
    loop = CarrierLoop.textbook(5.0, T)
    rng = np.random.default_rng(1)
    n, fc = 3_000_000, 0.1
    m = 0.05 * ar1(rng, n, fc)
    q = 0.05 * ar1(rng, n, fc)
    truth = np.var(loop.filter(q)[20_000:])
    out = iw.iw_windows(1.0 + m, np.zeros(n), loop, iw.IWConfig(window=2000, gated=False))
    phi = out.noise[5:].mean() / truth
    pred = predictions(loop, fc, 2000)
    assert phi == pytest.approx(pred["projection_once"], rel=0.15)
    assert abs(phi - pred["docs4_P5_centred"]) > 0.08


@pytest.mark.slow
def test_two_filter_mismatch_law_predicts_closed_loop() -> None:
    """N2 / P6: the closed form predicts r and reported C/N0 of the nonlinear loop for a tone (S06)."""
    sig2 = 1 / (2 * CN0 * T)
    for bn, f_b in ((5.0, 0.5), (5.0, 20.0), (35.0, 3.0)):
        sim = simulate_tone((bn, f_b, 2 * sig2, 7))
        pred = predict_tone(CarrierLoop.gnsssdr(bn, T), f_b, 2 * sig2, sim["cn0_clean_dbhz"] - 45.0)
        assert sim["r_corrected"] == pytest.approx(pred["r_corrected"], rel=0.1)
        assert sim["cn0_rep_dbhz"] == pytest.approx(pred["cn0_rep_dbhz"], abs=0.3)


@pytest.mark.slow
def test_pilot_data_witness_has_no_low_frequency_blind_band() -> None:
    """N1: d_data - d_pilot recovers the noise-driven error variance of slow circular disturbances (S07)."""
    rng = np.random.default_rng(3)
    k, burn, n_win = 400_000, 20_000, 2000
    loop = CarrierLoop.gnsssdr(5.0, T)
    sig_n = np.sqrt(1 / (2 * 10**4.5 * T))

    def colour() -> np.ndarray:
        return 2.0 * sig_n * (ar1(rng, k, 0.1) + 1j * ar1(rng, k, 0.1))[None, :] / np.sqrt(2)

    th_p, th_d = thermal_noise(rng, (1, k), 10**4.5, T), thermal_noise(rng, (1, k), 10**4.5, T)
    out = track(TrackingConfig(5.0, T), np.zeros((1, k)), th_p + colour(), eta_data=th_d + colour())
    assert out.prompt_data is not None
    dd = np.angle(out.prompt_data[0] * np.conj(out.prompt[0]))
    h = loop.filter(dd)[burn:]
    nb = h.size // n_win
    pdw = 0.5 * np.mean(h[: nb * n_win] ** 2)
    truth = np.var(2 * np.pi * out.phi[0, burn:])
    assert pdw / truth == pytest.approx(1.0, abs=0.3)
    i_prompt = np.real(out.prompt[0, burn : burn + nb * n_win])
    iw_out = iw.iw_windows(i_prompt, np.zeros(nb * n_win), loop, iw.IWConfig(window=n_win, gated=False))
    assert iw_out.noise.mean() / truth < 0.6  # the single-channel witness misses most of it


def test_single_channel_observational_equivalence() -> None:
    """N4: additive circular S vs amplitude S + phase S give the same witness outputs but different Var(e) (S13)."""
    loop = CarrierLoop.gnsssdr(35.0, T, kappa=0.0)
    rng = np.random.default_rng(13)
    n, sig = 600_000, 0.03

    def proc() -> np.ndarray:
        return 2 * sig * ar1(rng, n, 3.0)

    m0, q0 = sig * rng.standard_normal(n), sig * rng.standard_normal(n)
    m1, q1 = m0 + proc(), q0 + proc()
    r1 = iw.iw_windows(1 + m1, loop.filter(q1, "disc"), loop, iw.IWConfig(window=2000))
    a, theta = proc(), proc()
    r0 = iw.iw_windows((1 + a) * (1 + m0), loop.filter(theta + q0, "disc"), loop, iw.IWConfig(window=2000))
    e1 = loop.filter(q1)
    e0 = loop.filter(theta + q0) - theta
    assert r1.noise.mean() / r0.noise.mean() == pytest.approx(1.0, abs=0.05)
    assert np.var(e1[10_000:]) / np.var(e0[10_000:]) > 3.0


@pytest.mark.slow
def test_data_bit_form_p6prime_predicts_costas_loop() -> None:
    """P6': on a data channel the bits multiply the interference; P6' tracks the Costas loop, P6 does not (S19)."""
    from review.s19_data_bits_p6 import sim_cell

    res = sim_cell((35.0, 2.0, 55.0, 7))
    assert res["sim_r"] == pytest.approx(res["p6prime_r"], rel=0.3)
    assert res["sim_cn0_rep"] == pytest.approx(res["p6prime_cn0"], abs=0.8)
    assert res["p6_r"] > 2 * res["sim_r"]  # the pilot form overpredicts by more than 2x here
