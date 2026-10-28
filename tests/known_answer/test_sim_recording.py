"""Synthetic GPS L1 C/A recordings with exactly known phase (Route A, docs/3 section 5): calibration and format."""

from pathlib import Path

import numpy as np
import pytest

from gnssaudit.sim.despread import Despreader
from gnssaudit.sim.recording import SkySat, write_ca_recording, write_sky_recording

FS = 4.0e6


def test_recording_is_interleaved_int8_with_the_stated_noise(tmp_path: Path) -> None:
    path = tmp_path / "r.bin"
    write_ca_recording(path, prn=3, doppler_hz=-1250.0, cn0_dbhz=45.0, seconds=0.25, fs_hz=FS, seed=2)
    raw = np.fromfile(path, dtype=np.int8)
    assert raw.size == 2 * 1_000_000
    assert np.std(raw[0::2].astype(float)) == pytest.approx(16.0, rel=0.02)  # 16 counts per quadrature
    assert np.std(raw[1::2].astype(float)) == pytest.approx(16.0, rel=0.02)


def despread(path: Path, prn: int, doppler_hz: float) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int8).astype(np.float64)
    x = (raw[0::2] + 1j * raw[1::2]) / 16.0  # noise of unit variance per quadrature
    return Despreader(prn=prn, fs_hz=FS, doppler_hz=doppler_hz).despread(x)


def test_despread_power_matches_the_requested_cn0(tmp_path: Path) -> None:
    """A / sigma per sample sets C/N0: A^2 / N0 = C/N0 with N0 = 2 sigma^2 / fs (the s17 calibration). At zero
    Doppler the code cannot slip against the despreader's fixed-rate replica."""
    path = tmp_path / "r.bin"
    write_ca_recording(path, prn=3, doppler_hz=0.0, cn0_dbhz=50.0, seconds=0.4, fs_hz=FS, seed=3, chip0=0.0)
    eta = despread(path, 3, 0.0)
    noise_var = 2.0 / 4000  # per-sample complex variance 2, averaged over 4000 samples per code period
    a2 = float(np.mean(np.abs(eta) ** 2)) - noise_var
    assert a2 == pytest.approx(10 ** (50.0 / 10) / FS * 2.0, rel=0.03)  # C/N0 = A^2 / (2 / fs)


def test_carrier_sits_exactly_at_the_stated_doppler(tmp_path: Path) -> None:
    """Over 0.1 s the code slips 0.08 chip at -1250 Hz, which costs amplitude but not phase: eta^2 (data bits
    removed) keeps one phase when the despreader's carrier matches the recording's."""
    path = tmp_path / "r.bin"
    write_ca_recording(path, prn=3, doppler_hz=-1250.0, cn0_dbhz=50.0, seconds=0.1, fs_hz=FS, seed=4, chip0=0.0)
    ph = np.angle(despread(path, 3, -1250.0) ** 2)
    assert np.std(np.angle(np.exp(1j * (ph - ph[0])))) < 0.15
    off = np.angle(despread(path, 3, -1240.0) ** 2)  # 10 Hz off: the phase walks by 2 pi 20 Hz t
    assert np.std(np.angle(np.exp(1j * (off - off[0])))) > 0.5


def test_a_sky_recording_carries_each_satellite_at_its_level_and_a_late_start(tmp_path: Path) -> None:
    """write_sky_recording uses write_ca_recording's calibration for every satellite; a satellite with a later start
    (a spoofed duplicate, P2b task 5) is absent before it."""
    path = tmp_path / "sky.bin"
    sats = [SkySat(prn=3, doppler_hz=0.0, cn0_dbhz=50.0, chip0=0.0),
            SkySat(prn=9, doppler_hz=0.0, cn0_dbhz=47.0, chip0=0.0, start_s=0.2)]  # fmt: skip
    write_sky_recording(path, sats, seconds=0.4, fs_hz=FS, seed=3)
    noise_var = 2.0 / 4000
    for prn, cn0 in ((3, 50.0), (9, 47.0)):
        eta = despread(path, prn, 0.0)
        a2 = float(np.mean(np.abs(eta[200:]) ** 2)) - noise_var  # code periods from 0.2 s on
        assert a2 == pytest.approx(10 ** (cn0 / 10) / FS * 2.0, rel=0.05)
    early = despread(path, 9, 0.0)[:200]
    assert float(np.mean(np.abs(early) ** 2)) == pytest.approx(noise_var, rel=0.2)
