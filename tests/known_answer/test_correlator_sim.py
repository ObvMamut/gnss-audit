"""Correlator-level closed-loop simulator (Route A reduced, docs/3 section 5)."""

import numpy as np
import pytest

from gnssaudit.sim.correlator import TrackingConfig, m2m4_cn0_dbhz, thermal_noise, track
from gnssaudit.uq.loop import CarrierLoop


def _cfg(**kw: object) -> TrackingConfig:
    base: dict[str, object] = {"bn_hz": 5.0, "t_s": 1e-3, "pilot": True}
    base.update(kw)
    return TrackingConfig(**base)  # type: ignore[arg-type]


def test_converges_to_constant_phase_without_noise() -> None:
    k = 30_000  # dominant pole 0.99937 at 5 Hz, 1 ms: time constant about 1.6 s
    theta = np.full((1, k), 0.1)  # cycles
    out = track(_cfg(), theta_avg=theta, eta=np.zeros((1, k), complex))
    assert abs(out.phi[0, -1] - 0.1) < 1e-6
    assert abs(out.d[0, -1]) < 1e-6


def test_thermal_jitter_matches_effective_bandwidth() -> None:
    cfg = _cfg()
    cn0 = 10 ** (4.5)  # 45 dB-Hz
    k, r = 40_000, 16
    rng = np.random.default_rng(3)
    eta = thermal_noise(rng, (r, k), cn0_hz=cn0, t_s=cfg.t_s)
    out = track(cfg, theta_avg=np.zeros((r, k)), eta=eta)
    e_rad = 2 * np.pi * out.phi[:, 2000:]
    loop = CarrierLoop.gnsssdr(cfg.bn_hz, cfg.t_s)
    expected = loop.bn_eff_hz() / cn0
    # ~ 16 x 38 s at tau ~ 0.1 s: relative SE about 2 %
    assert np.var(e_rad) == pytest.approx(expected, rel=0.08)


def test_linear_limit_matches_transfer_function() -> None:
    cfg = _cfg()
    k = 20_000
    rng = np.random.default_rng(5)
    eta = thermal_noise(rng, (1, k), cn0_hz=10**5.5, t_s=cfg.t_s)
    out = track(cfg, theta_avg=np.zeros((1, k)), eta=eta)
    n = eta[0].imag  # equivalent quadrature noise in rad for A = 1, theta = 0
    pred = CarrierLoop.gnsssdr(cfg.bn_hz, cfg.t_s).filter(n / (2 * np.pi))
    assert np.max(np.abs(out.phi[0] - pred)) < 2e-4


def test_m2m4_estimator_is_consistent_but_biased_on_short_buffers() -> None:
    rng = np.random.default_rng(0)
    w = thermal_noise(rng, (4000, 1000), cn0_hz=10**4.5, t_s=1e-3)
    assert np.mean(m2m4_cn0_dbhz(1 + w, 1e-3)) == pytest.approx(45.0, abs=0.03)
    # finding F21: with GNSS-SDR's 20-prompt buffer the mean-dB bias is about +0.47 dB at 45 dB-Hz
    w20 = thermal_noise(rng, (40_000, 20), cn0_hz=10**4.5, t_s=1e-3)
    assert np.mean(m2m4_cn0_dbhz(1 + w20, 1e-3)) - 45.0 == pytest.approx(0.47, abs=0.05)


def test_in_loop_cn0_carries_the_short_buffer_bias() -> None:
    cfg = _cfg()
    rng = np.random.default_rng(7)
    k = 20_000
    eta = thermal_noise(rng, (4, k), cn0_hz=10**4.5, t_s=cfg.t_s)
    out = track(cfg, theta_avg=np.zeros((4, k)), eta=eta)
    assert np.mean(out.cn0_dbhz[:, 5000:]) == pytest.approx(45.47, abs=0.1)


def test_costas_discriminator_ignores_data_bits() -> None:
    cfg = _cfg(pilot=False)
    k = 30_000
    bits = np.repeat(np.where(np.random.default_rng(1).random(k // 20) < 0.5, -1.0, 1.0), 20)[None, :]
    out = track(cfg, theta_avg=np.full((1, k), 0.05), eta=np.zeros((1, k), complex), bits=bits)
    assert abs(out.phi[0, -1] - 0.05) < 1e-6
