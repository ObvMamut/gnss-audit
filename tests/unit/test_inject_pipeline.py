"""The injection pipeline (docs/8 sections 1, 5 and 6; V4, V2 at pipeline level, clipping, common-mode tones)."""

from pathlib import Path

import numpy as np
import pytest

from gnssaudit.inject import (
    ClipError,
    Disturbance,
    FrontEndSpec,
    IQSource,
    SinkSpec,
    clean_twin,
    inject,
    int16_scale,
)
from gnssaudit.sim.despread import Despreader
from gnssaudit.sim.recording import write_ca_recording

FS = 4.0e6
N0 = 2 * 16.0**2 / FS  # the synthetic recordings: 16 counts per quadrature


@pytest.fixture
def recording(tmp_path: Path) -> IQSource:
    path = tmp_path / "rec.bin"
    write_ca_recording(path, prn=5, doppler_hz=0.0, cn0_dbhz=60.0, seconds=0.5, fs_hz=FS, seed=6, chip0=0.0)
    return IQSource(path=path, item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=int(0.5 * FS))


def out(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int16).astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


NOISE = Disturbance("noise", {"bw_hz": 1.0e6, "f_hz": 0.0}, 0.1, 0.01, 75.0, 21)
TONE = Disturbance("phase", {"n_tones": 1, "f_min_hz": 4.0, "f_max_hz": 4.0, "a_rad": 0.4}, 0.0, 0.0, 0.0, 22)
GHOST = Disturbance(
    "ghost", {"prn": 30, "doppler_hz": 700.0, "doppler_rate_hzps": 0.0, "chip0": 5.0}, 0.0, 0.0, 45.0, 23
)
UNIT = FrontEndSpec(decim=1, fir_taps=1, scale=1.0)


def test_zero_gain_twin_is_the_scaled_source_bit_for_bit(recording: IQSource, tmp_path: Path) -> None:
    """V4 and D-10: the clean twin goes through the injector with every interference gain at zero."""
    fe = FrontEndSpec(1, 1, 8.0)
    reports = [
        inject(recording, clean_twin([NOISE, TONE]), fe, SinkSpec("file", tmp_path / f"c{i}.bin"), n0_per_hz=N0)
        for i in range(2)
    ]
    a, b = (tmp_path / "c0.bin").read_bytes(), (tmp_path / "c1.bin").read_bytes()
    assert a == b and reports[0].output_sha256 == reports[1].output_sha256
    raw = np.fromfile(recording.path, dtype=np.int8).astype(np.int16) * 8
    assert np.array_equal(np.frombuffer(a, dtype=np.int16), raw)


def test_clean_twin_keeps_the_ghosts_and_silences_the_interference() -> None:
    noise, tone, ghost = clean_twin([NOISE, TONE, GHOST])
    assert noise.level_db_hz == -np.inf and tone.params["a_rad"] == 0.0
    assert ghost == GHOST  # common to both twins: the ghost is part of the recording for ghost truth


def test_report_measures_levels_and_records_the_order(recording: IQSource, tmp_path: Path) -> None:
    rep = inject(recording, [GHOST, TONE, NOISE], UNIT, SinkSpec("file", tmp_path / "j.bin"), n0_per_hz=N0)
    assert rep.order == ("noise", "phase", "ghost")  # additive, then the phase process, then ghosts (docs/8 section 1)
    assert rep.achieved_db_hz[2] == pytest.approx(75.0, abs=0.2)  # index into the input list: NOISE
    assert rep.achieved_db_hz[0] == pytest.approx(45.0, abs=0.2)  # GHOST, as C/N0
    assert rep.samples_out == recording.sample_end and rep.clip_count == 0


def test_decimation_by_one_is_the_identity(recording: IQSource, tmp_path: Path) -> None:
    inject(recording, [], UNIT, SinkSpec("file", tmp_path / "id.bin"), n0_per_hz=N0)
    raw = np.fromfile(recording.path, dtype=np.int8).astype(np.int16)
    assert np.array_equal(np.fromfile(tmp_path / "id.bin", dtype=np.int16), raw)


def test_decimation_keeps_the_band_and_lowers_the_rate(recording: IQSource, tmp_path: Path) -> None:
    cw = Disturbance("cw", {"f_hz": 2.5e5}, 0.0, 0.0, 80.0, 3)
    rep = inject(recording, [cw], FrontEndSpec(decim=4, fir_taps=129, scale=1.0), SinkSpec("file", tmp_path / "d.bin"),
                 n0_per_hz=N0)  # fmt: skip
    y = out(tmp_path / "d.bin")
    assert rep.samples_out == y.size == recording.sample_end // 4
    spec = np.abs(np.fft.fft(y[2000 : 2000 + 50_000])) ** 2
    f = np.fft.fftfreq(50_000, d=4 / FS)
    assert f[np.argmax(spec)] == pytest.approx(2.5e5, abs=FS / 4 / 50_000)


def test_clipping_raises_with_the_count_and_the_first_sample(recording: IQSource, tmp_path: Path) -> None:
    with pytest.raises(ClipError) as err:
        inject(recording, [], FrontEndSpec(1, 1, 1000.0), SinkSpec("file", tmp_path / "x.bin"), n0_per_hz=N0)
    assert err.value.count > 0 and err.value.first_index >= 0
    assert not (tmp_path / "x.bin").exists()  # nothing half-written is left under the final name


def test_int16_scale_leaves_the_planned_headroom() -> None:
    """At the largest planned J/N0 the RMS sits 30 dB below full scale (docs/8 section 5, mode A)."""
    s = int16_scale(n0_per_hz=N0, fs_hz=FS, max_level_db_hz=95.0, headroom_db=30.0)
    rms_counts = s * np.sqrt(N0 * FS + 10**9.5 * N0)
    assert 20 * np.log10(32767 / rms_counts) == pytest.approx(30.0, abs=1e-9)


def test_phase_tones_move_every_satellite_together(recording: IQSource, tmp_path: Path) -> None:
    """The phase process multiplies the whole stream: the despread phase of the recorded satellite follows phi(t)."""
    rep = inject(recording, [TONE], UNIT, SinkSpec("file", tmp_path / "t.bin"), n0_per_hz=N0)
    assert rep.order == ("phase",)
    eta = Despreader(prn=5, fs_hz=FS, doppler_hz=0.0).despread(out(tmp_path / "t.bin"))
    phi_rx = np.unwrap(np.angle(eta**2)) / 2  # squaring removes the data bits
    t = (np.arange(eta.size) + 0.5) * 1e-3
    from gnssaudit.inject import tone_set

    (f,), (a,), (psi,) = tone_set(TONE)
    phi = a * np.sin(2 * np.pi * f * t + psi)
    resid = (phi_rx - phi) - np.mean(phi_rx - phi)
    assert np.std(resid) < 0.05 < np.std(phi)


def test_decimation_matches_its_definition(recording: IQSource, tmp_path: Path) -> None:
    """y[m] = sum_k h[k] x[start + m D - k], zero before the interval start: an alignment slip would be far off."""
    from dataclasses import replace

    from gnssaudit.inject import decimation_taps

    src = replace(recording, sample_start=12_345, sample_end=12_345 + 300_000)
    fe = FrontEndSpec(decim=4, fir_taps=4 * 16 + 1, scale=50.0)
    inject(src, [], fe, SinkSpec("file", tmp_path / "d.bin"), n0_per_hz=N0)
    raw = np.fromfile(recording.path, dtype=np.int8).astype(np.float64)
    x = (raw[0::2] + 1j * raw[1::2])[src.sample_start : src.sample_end]
    y = np.convolve(x, decimation_taps(fe))[: x.size][::4]
    want = np.rint(50.0 * y.real) + 1j * np.rint(50.0 * y.imag)
    got = out(tmp_path / "d.bin")
    assert got.size == want.size and np.max(np.abs(got - want)) <= np.sqrt(2)
