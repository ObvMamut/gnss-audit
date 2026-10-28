"""P2b task 9 on the binary: GNSS-SDR's observable epochs sit on its sample-counter clock, epoch i at (i + k0) x step,
with one k0 for every channel of a run, so an onset in recording samples can be placed on observable epochs."""

from pathlib import Path

import numpy as np
import pytest

from gnssaudit.experiments.twin import TWIN_LOCK_DEVIATION, receiver_spec
from gnssaudit.io.gnsssdr import read_observables_dump, read_tracking_dump
from gnssaudit.registry import dataset_file
from gnssaudit.sdr import SourceSpec, render_config, run_receiver
from gnssaudit.truth.rxtime import match_clock

pytestmark = [pytest.mark.container, pytest.mark.data, pytest.mark.slow]
FS = 4.0e6
STEP = 80_000  # 20 ms observables at 4 MS/s


def test_every_channel_of_a_run_matches_one_clock_offset(tmp_path: Path) -> None:
    try:
        path = dataset_file("cttc-2013-sample", "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat")
        path.stat()
    except FileNotFoundError:
        pytest.skip("the CTTC recording is not fetched")
    rx = receiver_spec({"receiver_overrides": TWIN_LOCK_DEVIATION["1C"]})
    channels = {i: ("G", "1C", p) for i, p in enumerate([1, 11, 17, 20, 32])}
    src = SourceSpec(path=path, item_type="ishort", fs_hz=FS)
    art = run_receiver(rx, render_config(rx, src, channels), src, tmp_path / "rx", 1800)
    assert art.observables_dump is not None
    obs = read_observables_dump(art.observables_dump, channels, source_id="cttc", receiver_id="rx")
    epoch = np.arange(obs.num_rows) // len(channels)  # one row per channel per epoch, channel order
    k0s = set()
    for ch, p in art.tracking_dumps.items():
        tr = read_tracking_dump(p, "patched112", run_id="rx", channel=ch, sys="G", signal="1C", fs_hz=FS, t_int_s=1e-3)
        prn = channels[ch][2]
        sel = (obs.column("prn").to_numpy() == prn) & obs.column("valid").to_numpy(zero_copy_only=False)
        if sel.sum() < 1000:
            continue
        stamps = tr.column("sample_count").to_numpy().astype(np.int64)
        phase = -tr.column("acc_phase_rad").to_numpy() / (2 * np.pi)
        t = tr.column("t_rx_s").to_numpy()
        late = sel & (epoch * STEP > (t[0] + 20.0) * FS)  # after pull-in and the narrow-tracking phase reset
        clock = match_clock(epoch[late], obs.column("cp_cyc").to_numpy()[late], stamps[t > t[0] + 19.0],
                            phase[t > t[0] + 19.0], step=STEP)  # fmt: skip
        k0s.add((clock.k0, clock.sign))
    assert k0s == {(1, -1.0)}, f"one clock for the run (the first tick, observables at +acc/2 pi), got {sorted(k0s)}"
