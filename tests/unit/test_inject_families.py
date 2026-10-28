"""inject: IQ sources and disturbance families (docs/8 sections 1, 3, 4 and 7: V1 and the ghost)."""

from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from gnssaudit.inject import (
    Disturbance,
    GhostSpec,
    IQSource,
    ghost_code_chips,
    ghost_modulation,
    ghost_phase_rad,
    iter_iq,
    phase_process,
    tone_set,
    unit_waveform,
)
from gnssaudit.sim.despread import Despreader

FS = 16.0e6


def dist(kind: str, **params: float) -> Disturbance:
    return Disturbance(kind=kind, params=params, onset_s=0.0, ramp_s=0.0, level_db_hz=0.0, seed=11)  # type: ignore[arg-type]


def psd(x: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    f, p = signal.welch(x, fs=fs, nperseg=1 << 14, return_onesided=False)
    o = np.argsort(f)
    return f[o], p[o]


def test_iter_iq_yields_absolute_indices_and_complex_samples(tmp_path: Path) -> None:
    raw = np.arange(40, dtype=np.int16)  # I = 0, 2, 4, ...; Q = 1, 3, 5, ...
    path = tmp_path / "x.bin"
    raw.tofile(path)
    got = list(iter_iq(IQSource(path=path, item_type="ishort", fs_hz=4e6, sample_start=3, sample_end=11), block=4))
    assert [i for i, _ in got] == [3, 7]
    assert got[0][1].dtype == np.complex64
    assert got[0][1].tolist() == [6 + 7j, 8 + 9j, 10 + 11j, 12 + 13j]
    raw8 = np.array([1, -2, -3, 4], dtype=np.int8)
    raw8.tofile(tmp_path / "b.bin")
    ((_, b),) = iter_iq(
        IQSource(path=tmp_path / "b.bin", item_type="ibyte", fs_hz=4e6, sample_start=0, sample_end=2), 8
    )
    assert b.tolist() == [1 - 2j, -3 + 4j]


def test_a_range_beyond_the_file_is_refused(tmp_path: Path) -> None:
    np.zeros(20, dtype=np.int16).tofile(tmp_path / "x.bin")  # 10 complex samples
    with pytest.raises(ValueError, match="10 samples"):
        IQSource(path=tmp_path / "x.bin", item_type="ishort", fs_hz=4e6, sample_start=0, sample_end=11)


def test_cw_line_sits_at_its_frequency() -> None:
    f, p = psd(unit_waveform(dist("cw", f_hz=1.25e6), 0, 1 << 18, FS))
    assert f[np.argmax(p)] == pytest.approx(1.25e6, abs=FS / (1 << 14))


def test_band_limited_noise_has_its_centre_and_bandwidth() -> None:
    """V1: equivalent noise bandwidth (power over peak density) and power-weighted centre within 1 %."""
    x = unit_waveform(dist("noise", bw_hz=2.0e6, f_hz=1.0e6), 0, 1 << 21, FS)
    f, p = psd(x)
    df = f[1] - f[0]
    power = p.sum() * df
    enbw = power / np.median(p[np.abs(f - 1.0e6) < 0.5e6])
    assert enbw == pytest.approx(2.0e6, rel=0.01)
    assert (p * f).sum() * df / power == pytest.approx(1.0e6, abs=0.01 * 2.0e6)


@pytest.mark.parametrize(("kind", "span", "period"), [("swept_cw", 1.0e6, 1e-3), ("chirp", 10.0e6, 10e-6)])
def test_sweeps_cover_their_span(kind: str, span: float, period: float) -> None:
    f, p = psd(unit_waveform(dist(kind, f_hz=0.5e6, span_hz=span, period_s=period), 0, 1 << 20, FS))
    lo, hi = 0.5e6 - span / 2, 0.5e6 + span / 2
    inside = p[(f >= lo - 0.01 * span) & (f <= hi + 0.01 * span)].sum() / p.sum()
    core = p[(f >= lo + 0.02 * span) & (f <= hi - 0.02 * span)].sum() / p.sum()
    assert inside > 0.98 and core > 0.9


def test_unit_power_for_every_additive_family() -> None:
    for d in (
        dist("cw", f_hz=1e5),
        dist("noise", bw_hz=2e6, f_hz=0.0),
        dist("swept_cw", f_hz=0.0, span_hz=1e6, period_s=1e-3),
    ):
        x = unit_waveform(d, 0, 1 << 20, FS)
        assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=0.01), d.kind


def test_raised_cosine_onset() -> None:
    d = Disturbance(kind="cw", params={"f_hz": 0.0}, onset_s=0.001, ramp_s=0.002, level_db_hz=0.0, seed=1)
    a = np.abs(unit_waveform(d, 0, int(0.004 * FS), FS))
    n_on, n_mid = int(0.001 * FS), int(0.002 * FS)
    assert np.all(a[:n_on] == 0.0)
    assert a[n_mid] == pytest.approx(0.5, abs=1e-6)  # halfway through the ramp
    assert np.all(a[int(0.003 * FS) + 1 :] == pytest.approx(1.0))


def test_phase_tones_are_log_spaced_and_recoverable() -> None:
    d = Disturbance(kind="phase", params={"n_tones": 12, "f_min_hz": 0.05, "f_max_hz": 50.0, "a_rad": 0.05},
                    onset_s=0.0, ramp_s=0.0, level_db_hz=0.0, seed=7)  # fmt: skip
    freqs, amps, phases = tone_set(d)
    assert freqs == pytest.approx(np.geomspace(0.05, 50.0, 12)) and np.all(amps == 0.05)
    fs = 1000.0  # 1 kHz is enough to check the closed form; the injector uses the recording rate
    t = np.arange(int(40.0 * fs)) / fs
    want = sum(a * np.sin(2 * np.pi * f * t + p) for f, a, p in zip(freqs, amps, phases, strict=True))
    assert phase_process(d, 0, t.size, fs) == pytest.approx(want, abs=1e-12)


def test_ghost_code_follows_its_carrier() -> None:
    """Code Doppler is carrier Doppler / 1540: code-minus-carrier drift is zero (V7, docs/8 section 4)."""
    g = GhostSpec(prn=7, doppler_hz=-2300.0, doppler_rate_hzps=-0.5, cn0_db_hz=45.0, chip0=100.0, seed=3)
    n = np.arange(0, int(4e8), int(1e6), dtype=np.int64)  # 100 s at 4 MS/s, sparsely
    t = n / 4e6
    carrier_cycles = (ghost_phase_rad(g, n, 4e6) - ghost_phase_rad(g, n[:1], 4e6)[0]) / (2 * np.pi)
    chips = ghost_code_chips(g, n, 4e6)
    code_advance = chips - g.chip0 - 1.023e6 * t
    # chips reach 1e8, where float64 resolves 1.5e-8 chip: 1540 times that bounds the honest rounding error, while a
    # wrong sign or a missing rate term would be off by about 1e5 cycles
    assert np.max(np.abs(1540.0 * code_advance - carrier_cycles)) < 4 * 1540.0 * np.spacing(chips.max())


def test_ghost_despreads_to_its_target_cn0() -> None:
    fs = 4.0e6
    g = Disturbance(kind="ghost", params={"prn": 9, "doppler_hz": 0.0, "doppler_rate_hzps": 0.0, "chip0": 0.0},
                    onset_s=0.0, ramp_s=0.0, level_db_hz=0.0, seed=2)  # fmt: skip
    x = unit_waveform(g, 0, int(0.2 * fs), fs)  # unit amplitude: power 1
    eta = Despreader(prn=9, fs_hz=fs, doppler_hz=0.0).despread(x)
    assert np.allclose(np.abs(eta), 1.0, atol=1e-9)  # code and data aligned exactly; bits only flip the sign
    assert set(np.round(np.real(eta * np.conj(eta[0]))).astype(int)) <= {-1, 1}


def test_the_ghost_modulation_is_the_injected_waveform() -> None:
    """One source for the ghost's waveform: the injector's unit ghost and `ghost_modulation` (code, data bits,
    carrier) agree sample for sample, also across a bit block and off the block grid (B1's projection replays it, P2b
    task 11)."""
    fs = 4.0e6
    d = Disturbance(kind="ghost", params={"prn": 9, "doppler_hz": 1234.0, "doppler_rate_hzps": 0.3, "chip0": 17.3},
                    onset_s=0.0, ramp_s=0.0, level_db_hz=0.0, seed=2)  # fmt: skip
    n0, n = 3 * 65536 + 17, 200_000
    want = unit_waveform(d, n0, n, fs)
    got = ghost_modulation(GhostSpec.of(d), np.arange(n0, n0 + n, dtype=np.int64), fs)
    np.testing.assert_array_equal(got, want)
