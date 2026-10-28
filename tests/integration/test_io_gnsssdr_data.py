"""I-parser-golden in its semantic form (docs/11 section 6, amended 13 Oct 2026) on real GNSS-SDR v0.0.22 output:
the tracking dumps of the s17 identification runs (synthetic PRN 1 at 4 MS/s, 100 s), under $GNSSAUDIT_RESULTS."""

import numpy as np
import pytest

from gnssaudit.io.gnsssdr import read_tracking_dump
from gnssaudit.paths import results_root

pytestmark = pytest.mark.data
WORK = results_root() / "review-work" / "s17"


@pytest.mark.parametrize(("name", "t_int_s"), [("default", 1e-3), ("e3_15hz_1ms", 1e-3), ("e3_5hz_20ms", 20e-3)])
def test_real_tracking_dump_parses_with_consistent_records(name: str, t_int_s: float) -> None:
    path = WORK / f"trk_{name}0.dat"
    if not path.exists():
        pytest.skip(f"{path} not present (run review.s17_gnsssdr_identification)")
    t = read_tracking_dump(path, "stock108", run_id=name, channel=0, sys="G", signal="1C", fs_hz=4.0e6, t_int_s=t_int_s)
    assert t.num_rows == path.stat().st_size // 108 > 50_000 // (20 if t_int_s > 1e-3 else 1) // 20
    assert set(t.column("prn").to_pylist()) == {1}  # the assigned satellite (Channel0.satellite=1)
    stamps = t.column("sample_count").to_numpy().astype(np.int64)
    assert np.all(np.diff(stamps) > 0)
    cn0 = t.column("cn0_dbhz").to_numpy()[t.column("t_rx_s").to_numpy() > 20.0]
    assert 53.0 < float(np.median(cn0)) < 57.5  # recorded at 55 dB-Hz; M2M4 reads about +0.4 dB high (F21)


def test_real_stock_dump_read_as_patched_is_refused() -> None:
    path = WORK / "trk_default0.dat"
    if not path.exists():
        pytest.skip(f"{path} not present")
    from gnssaudit.io.gnsssdr import DumpError

    with pytest.raises(DumpError):
        read_tracking_dump(path, "patched112", run_id="x", channel=0, sys="G", signal="1C", fs_hz=4.0e6, t_int_s=1e-3)
