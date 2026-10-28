"""I-container (docs/11 section 6): the pinned GNSS-SDR image tracks a synthetic GPS L1 C/A recording through
gnssaudit.sdr, and its tracking dump parses (I-parser-golden, semantic form)."""

from pathlib import Path

import numpy as np
import pytest

from gnssaudit.io.gnsssdr import read_tracking_dump
from gnssaudit.sdr import ReceiverSpec, SourceSpec, render_config, run_receiver
from gnssaudit.sdr.images import STOCK
from gnssaudit.sim.recording import write_ca_recording

pytestmark = pytest.mark.container
STOCK_IMAGE = (
    "carlesfernandez/docker-gnsssdr:v0.0.22@sha256:befe06ed81163068e9eaf918024e18cedf1fc374925431320d57055650b48b79"
)
FS = 4.0e6


def test_pinned_receiver_tracks_the_assigned_satellite(tmp_path: Path) -> None:
    rec = tmp_path / "in" / "ca_prn3.bin"
    rec.parent.mkdir()
    write_ca_recording(rec, prn=3, doppler_hz=-1250.0, cn0_dbhz=50.0, seconds=20.0)
    spec = ReceiverSpec(image=STOCK, source_commit="3fcb502a", patch_id=None, template="1C", overrides={})
    source = SourceSpec(path=rec, item_type="ibyte", fs_hz=FS)
    art = run_receiver(spec, render_config(spec, source, {0: ("G", "1C", 3)}), source, tmp_path / "run", timeout_s=600)
    assert art.exit_code == 0 and sorted(art.tracking_dumps) == [0]
    t = read_tracking_dump(
        art.tracking_dumps[0], "stock108", run_id="ic", channel=0, sys="G", signal="1C", fs_hz=FS, t_int_s=1e-3
    )
    assert t.num_rows > 500
    assert set(t.column("prn").to_pylist()) == {3}
    stamps = t.column("sample_count").to_numpy().astype(np.int64)
    assert np.all(np.diff(stamps) > 0)
    late = t.column("t_rx_s").to_numpy() > 10.0
    assert 48.0 < float(np.median(t.column("cn0_dbhz").to_numpy()[late])) < 52.5
    dop = float(np.median(t.column("dop_hz").to_numpy()[late]))
    assert abs(dop - (-1250.0)) < 5.0
