"""KA-loop, KA-jitter: carrier-loop transfer functions (docs/9 section 5, docs/11 section 3)."""

import numpy as np
import pytest

from gnssaudit.uq.loop import CarrierLoop


def test_textbook_noise_bandwidth_matches_design() -> None:
    # docs/4 section 3.6 and docs/9 section 5.3: 0.01006 and 5.03 / 15.3 / 5.13 / 5.73 / 24.5 Hz
    loop = CarrierLoop.textbook(bn_hz=5.0, t_s=1e-3)
    assert loop.noise_gain() == pytest.approx(0.01006, rel=1e-3)
    expected = {
        (1e-3, 5.0): 5.03,
        (1e-3, 15.0): 15.3,
        (4e-3, 5.0): 5.13,
        (20e-3, 5.0): 5.73,
        (20e-3, 15.0): 24.5,
    }
    for (t_s, bn), ref in expected.items():
        assert CarrierLoop.textbook(bn, t_s).bn_eff_hz() == pytest.approx(ref, abs=0.006 * ref)
    assert loop.crossover_hz() == pytest.approx(2.29, abs=0.01)


def test_simplified_gnsssdr_timing_reproduces_docs9_table() -> None:
    # docs/9 section 5.3 used kappa = 0 (no one-update replica lag); KA-loop values
    expected = {
        (1e-3, 5.0): 4.31,
        (1e-3, 15.0): 13.2,
        (4e-3, 5.0): 4.42,
        (20e-3, 5.0): 5.14,
        (20e-3, 15.0): 25.9,
    }
    for (t_s, bn), ref in expected.items():
        assert CarrierLoop.gnsssdr(bn, t_s, kappa=0.0).bn_eff_hz() == pytest.approx(ref, abs=0.006 * ref)


def test_closed_loop_is_type_three_and_stable() -> None:
    w = np.array([1e-4, 2e-4])
    simplified = CarrierLoop.gnsssdr(bn_hz=5.0, t_s=1e-3, kappa=0.0)
    e = np.abs(simplified.error_response(w))
    # |1 - H| = O(w^3): doubling w multiplies it by 8
    assert e[1] / e[0] == pytest.approx(8.0, rel=1e-2)
    faithful = CarrierLoop.gnsssdr(bn_hz=5.0, t_s=1e-3)
    assert np.all(np.abs(faithful.poles()) < 1.0)
    d = np.abs(faithful.freq_response(w, "disc"))
    assert d[1] / d[0] == pytest.approx(8.0, rel=1e-2)


def test_reported_phase_lags_replica_by_half_a_block() -> None:
    # GNSS-SDR reports the NCO phase at block start; the correlator averages over the block, so the
    # dynamics error referenced to the block-start true phase equals the discriminator response.
    loop = CarrierLoop.gnsssdr(bn_hz=5.0, t_s=1e-3)
    w = np.array([3e-4, 1e-3, 1e-2])
    z = np.exp(1j * w)
    dyn = loop.error_response(w) + loop.freq_response(w) * (1 - z) / 2
    assert np.allclose(np.abs(dyn) / np.abs(loop.freq_response(w, "disc")), 1.0, atol=1e-3)


def test_noise_gain_equals_impulse_energy() -> None:
    loop = CarrierLoop.gnsssdr(bn_hz=15.0, t_s=1e-3)
    h = loop.impulse_response(200_000)
    assert np.sum(h**2) == pytest.approx(loop.noise_gain(), rel=1e-6)


def test_white_noise_discriminator_identity() -> None:
    # docs/4 B5 note: (1/2pi) int |1-H|^2 = 1 + 2 B_n,eff T for a strictly proper loop
    loop = CarrierLoop.gnsssdr(bn_hz=5.0, t_s=1e-3, kappa=0.0)
    assert loop.discriminator_gain() == pytest.approx(1.0 + loop.noise_gain(), rel=1e-9)


def test_error_and_phase_responses_are_complementary() -> None:
    loop = CarrierLoop.gnsssdr(bn_hz=5.0, t_s=1e-3)
    w = np.geomspace(1e-2, np.pi, 50)
    assert np.allclose(loop.freq_response(w) + loop.error_response(w), 1.0, atol=1e-12)


def test_simulated_loop_reproduces_transfer_function() -> None:
    loop = CarrierLoop.gnsssdr(bn_hz=5.0, t_s=1e-3)
    u = np.random.default_rng(1).standard_normal(4096)
    y = loop.filter(u)
    h = loop.impulse_response(4096)
    assert np.allclose(y, np.convolve(u, h)[:4096], atol=1e-10)


def test_inverse_map_round_trip() -> None:
    loop = CarrierLoop.gnsssdr_for_bn_eff(target_hz=5.0, t_s=1e-3)
    assert loop.bn_eff_hz() == pytest.approx(5.0, rel=1e-6)


def test_dc_limits() -> None:
    loop = CarrierLoop.gnsssdr(bn_hz=5.0, t_s=1e-3)
    assert loop.freq_response(np.array([0.0]))[0] == pytest.approx(1.0)
    assert abs(loop.freq_response(np.array([0.0]), "disc")[0]) < 1e-12
