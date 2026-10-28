"""Chip-level despreading of interference (U-ca-code, docs/11 section 4; docs/8 section 3)."""

import numpy as np
import pytest

from gnssaudit.sim.despread import Despreader, ca_code, cw, swept_cw, white_noise


def test_ca_code_first_chips_match_is_gps_200() -> None:
    # IS-GPS-200 Table 3-Ia: first 10 chips in octal
    for prn, octal in [(1, "1440"), (2, "1620"), (3, "1710"), (4, "1744"), (5, "1133"), (6, "1455")]:
        chips = (ca_code(prn)[:10] < 0).astype(int)  # +1 <-> logical 0, -1 <-> logical 1
        assert int("".join(map(str, chips)), 2) == int(octal, 8)


def test_ca_autocorrelation_three_valued() -> None:
    c = ca_code(7)
    vals = {int(np.dot(c, np.roll(c, s))) for s in range(1, 1023)}
    assert vals <= {-65, -1, 63}
    assert int(np.dot(c, c)) == 1023


def test_signal_despreads_to_unit_amplitude() -> None:
    dsp = Despreader(prn=3, fs_hz=2.046e6, doppler_hz=1234.0)
    x = dsp.replica(n_blocks=5)
    eta = dsp.despread(x)
    assert np.allclose(eta, 1.0, atol=1e-9)


def test_white_jammer_despread_power() -> None:
    dsp = Despreader(prn=3, fs_hz=2.046e6, doppler_hz=500.0)
    rng = np.random.default_rng(0)
    j = 1e3  # jammer-to-signal power ratio per sample
    eta = dsp.despread(white_noise(rng, dsp.ns * 4000, power=j))
    assert np.mean(np.abs(eta) ** 2) == pytest.approx(j / dsp.ns, rel=0.05)


def test_cw_despreads_to_a_code_line_beat() -> None:
    # A CW at offset f from the replica carrier beats with code line m at f - 1000 m Hz
    dsp = Despreader(prn=5, fs_hz=2.046e6, doppler_hz=0.0)
    n_blocks = 2000
    eta = dsp.despread(cw(dsp.ns * n_blocks, dsp.fs_hz, f_hz=3000.0 + 7.0, power=1.0))
    spec = np.abs(np.fft.fft(eta)) ** 2
    freqs = np.fft.fftfreq(n_blocks, d=1e-3)
    assert freqs[np.argmax(spec)] == pytest.approx(7.0, abs=0.5)
    # its power equals the PRN's spectral line at 3 kHz
    line = np.abs(np.fft.fft(np.repeat(ca_code(5), 2)))[3] ** 2 / dsp.ns**2
    assert np.mean(np.abs(eta) ** 2) == pytest.approx(line, rel=0.02)


def test_swept_cw_is_continuous_phase() -> None:
    x = swept_cw(20_000, 2.046e6, f0_hz=-5e4, span_hz=1e5, period_s=1e-3, power=4.0)
    assert np.allclose(np.abs(x) ** 2, 4.0)
    dphi = np.angle(x[1:] * np.conj(x[:-1]))
    assert np.max(np.abs(dphi)) < 2 * np.pi * 1.01e5 / 2.046e6
