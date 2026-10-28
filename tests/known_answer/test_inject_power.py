"""Power calibration (docs/8 section 2): noise floor, gains (V2) and the spectral separation coefficient (KA-ssc)."""

from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

from gnssaudit.inject import Disturbance, IQSource, ca_psd, estimate_n0, gain, render, ssc_kappa
from gnssaudit.sim.recording import write_ca_recording

FS = 4.0e6


def test_noise_floor_of_a_recording_with_a_buried_signal(tmp_path: Path) -> None:
    """16 counts per quadrature is a complex power of 512, so N0 = 512 / fs; the C/A signal 30 dB down is invisible."""
    path = tmp_path / "r.bin"
    write_ca_recording(path, prn=4, doppler_hz=500.0, cn0_dbhz=45.0, seconds=2.0, fs_hz=FS, seed=8)
    src = IQSource(path=path, item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=int(2.0 * FS))
    nf = estimate_n0(src, frontend_bw_hz=4.0e6)
    assert 10 * np.log10(nf.n0_per_hz) == pytest.approx(10 * np.log10(2 * 16.0**2 / FS), abs=0.1)
    assert nf.n_segments >= 2 and nf.spread_db < 0.1


def test_a_line_in_the_floor_raises_the_density_the_correlator_sees_not_the_median(tmp_path: Path) -> None:
    """The median floor ignores a strong line; the C/A-weighted density counts it by the code spectrum at its offset.
    Found on the CTTC sample, whose +580 kHz line and real satellites put that density 2.3 dB above the median."""
    n, n0 = 1 << 22, 2 * 16.0**2 / FS
    rng = np.random.default_rng(5)
    f_cw, j = 5.8e5, 1e6 * n0  # a 60 dB-Hz line
    z = rng.normal(0, 16, n) + 1j * rng.normal(0, 16, n) + np.sqrt(j) * np.exp(2j * np.pi * f_cw * np.arange(n) / FS)
    path = tmp_path / "line.bin"
    np.round(np.column_stack([z.real, z.imag])).astype(np.int16).tofile(path)
    nf = estimate_n0(IQSource(path=path, item_type="ishort", fs_hz=FS, sample_start=0, sample_end=n), 4.0e6)
    f = np.linspace(-FS / 2, FS / 2, 400_001)
    excess = 1 + j * ca_psd(np.array([f_cw]))[0] / (n0 * np.trapezoid(ca_psd(f), f))
    assert 10 * np.log10(nf.n0_per_hz / n0) == pytest.approx(0.0, abs=0.1)
    assert 10 * np.log10(nf.ca_weighted_per_hz / nf.n0_per_hz) == pytest.approx(10 * np.log10(excess), abs=0.1)


def test_gain_turns_a_level_into_an_amplitude() -> None:
    assert gain(60.0, 1.28e-4) == pytest.approx(np.sqrt(1e6 * 1.28e-4))


@pytest.mark.parametrize(
    "d",
    [Disturbance("noise", {"bw_hz": 2.0e6, "f_hz": 0.0}, 0.0, 0.0, 70.0, 3),
     Disturbance("cw", {"f_hz": 2.0e5}, 0.0, 0.0, 70.0, 3),
     Disturbance("swept_cw", {"f_hz": 0.0, "span_hz": 1.0e6, "period_s": 1e-3}, 0.0, 0.0, 70.0, 3),
     Disturbance("chirp", {"f_hz": 0.0, "span_hz": 2.0e6, "period_s": 2e-5}, 0.0, 0.0, 70.0, 3)],
    ids=lambda d: d.kind,
)  # fmt: skip
def test_achieved_level_is_the_target(d: Disturbance) -> None:
    """V2: the injected component's power over N0 is the requested J/N0 within 0.2 dB."""
    n0 = 2 * 16.0**2 / FS
    x = render(d, 0, 1 << 20, FS, n0)
    assert 10 * np.log10(np.mean(np.abs(x) ** 2) / n0) == pytest.approx(70.0, abs=0.2)


def test_ssc_of_wideband_noise_is_one_over_its_bandwidth() -> None:
    """KA-ssc: flat interference over B much wider than the C/A main lobe gives kappa about 1/B [2, 37]."""
    f = np.linspace(-15e6, 15e6, 300_001)
    flat = np.where(np.abs(f) <= 10e6, 1.0, 0.0)
    assert ssc_kappa(f, flat, ca_psd(f)) == pytest.approx(1 / 20e6, rel=0.01)


def test_ssc_of_a_line_at_band_centre_is_the_code_spectrum_peak() -> None:
    """A line at f = 0 picks the normalized signal PSD there: T_c over the sinc^2 power inside the +-2 MHz band."""
    f = np.linspace(-2e6, 2e6, 400_001)
    line = np.exp(-0.5 * (f / 200.0) ** 2)  # narrower than any C/A spectral feature
    tc = 1 / 1.023e6
    in_band, _ = integrate.quad(lambda u: np.sinc(u) ** 2, -2e6 * tc, 2e6 * tc, limit=200)
    assert ssc_kappa(f, line, ca_psd(f)) == pytest.approx(tc / in_band, rel=0.01)
