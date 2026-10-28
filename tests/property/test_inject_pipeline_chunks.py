"""V5 for the whole pipeline: the output does not depend on the chunk size (docs/8 section 6)."""

from pathlib import Path

import pytest

from gnssaudit.inject import Disturbance, FrontEndSpec, IQSource, SinkSpec, inject
from gnssaudit.sim.recording import write_ca_recording

FS = 4.0e6


@pytest.mark.parametrize("decim", [1, 4])
def test_output_hash_is_independent_of_chunking(tmp_path: Path, decim: int) -> None:
    path = tmp_path / "rec.bin"
    write_ca_recording(path, prn=2, doppler_hz=1500.0, cn0_dbhz=45.0, seconds=0.6, fs_hz=FS, seed=12)
    src = IQSource(path=path, item_type="ibyte", fs_hz=FS, sample_start=1000, sample_end=int(0.6 * FS))
    ds = [
        Disturbance("noise", {"bw_hz": 2.0e6, "f_hz": 3.0e5}, 0.2, 0.05, 70.0, 31),
        Disturbance("phase", {"n_tones": 3, "f_min_hz": 1.0, "f_max_hz": 20.0, "a_rad": 0.1}, 0.0, 0.0, 0.0, 32),
        Disturbance(
            "ghost", {"prn": 30, "doppler_hz": -800.0, "doppler_rate_hzps": 0.3, "chip0": 1.0}, 0.0, 0.0, 40.0, 33
        ),
    ]
    fe = FrontEndSpec(decim=decim, fir_taps=4 * 32 + 1, scale=20.0)
    hashes = {
        inject(src, ds, fe, SinkSpec("file", tmp_path / f"o{c}.bin"), n0_per_hz=1.28e-4, chunk=c).output_sha256
        for c in (1 << 18, 1 << 20, 1 << 22)
    }
    assert len(hashes) == 1
