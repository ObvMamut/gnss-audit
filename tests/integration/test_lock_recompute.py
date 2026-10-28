"""D-34's condition (P2b task 8): the stock lock verdict recomputed from the patched dump must reproduce the losses of
lock of a stock-rule run before E0 and E2 report it. A stock receiver does not log the integration it drops on, so its
dump stops with a counter exactly at its limit: the replay must show that counter at the limit on the last record
before the gap, and no drop earlier. Two runs that drop by construction: a synthetic satellite pushed to about
30 dB-Hz by band-limited noise long after its pull-in (an exact check of the counters), and the CTTC sample with a
strict threshold, where the drop is decided by the end of the pull-in, whose start (the acquisition stamp) is not in
the dump: there the replay must land within LATENCY_S of the receiver's drop, and not before it."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.experiments.twin import receiver_spec
from gnssaudit.inject import Disturbance, FrontEndSpec, IQSource, SinkSpec, estimate_n0, inject, int16_scale
from gnssaudit.io.gnsssdr import read_tracking_dump
from gnssaudit.registry import dataset_file
from gnssaudit.sdr import SourceSpec, render_config, run_receiver
from gnssaudit.sdr.lock import LockRule, replay, stock_rule
from gnssaudit.sim.recording import write_ca_recording

pytestmark = [pytest.mark.container, pytest.mark.slow]
FS = 4.0e6
LATENCY_S = 0.1  # acquisition stamp to first tracking record (41 ms measured on the CTTC sample)


def first_loss(track: pa.Table, end_s: float) -> int | None:
    """Index, in time order, of the last record before the receiver dropped the channel: before the first gap, or
    the last record when the records stop more than 1 s before the recording does (no re-acquisition); else None."""
    t = np.sort(track.column("t_rx_s").to_numpy())
    gaps = np.flatnonzero(np.diff(t) > 1.5 * np.median(np.diff(t)))
    if gaps.size:
        return int(gaps[0])
    return t.size - 1 if t[-1] < end_s - 1.0 else None


def assert_replay_reaches_the_limit_at(track: pa.Table, rule: LockRule, last: int) -> None:
    r = replay(track, rule)
    drop = r.column("drop").to_numpy(zero_copy_only=False)
    at_limit = (r.column("carrier")[last].as_py() == rule.max_carrier_lock_fail
                or r.column("code")[last].as_py() == rule.max_lock_fail)  # fmt: skip
    assert at_limit and not drop[: last + 1].any(), "the replay must stand at the limit where the receiver dropped"


def receive(path: Path, fs: float, item: str, channels: dict, overrides: dict, work: Path) -> dict[int, pa.Table]:
    rx = receiver_spec({"stock_receiver": True, "receiver_overrides": overrides})
    src = SourceSpec(path=path, item_type=item, fs_hz=fs)  # type: ignore[arg-type]
    art = run_receiver(rx, render_config(rx, src, channels), src, work, 1800)
    return {ch: read_tracking_dump(p, "patched112", run_id="stock", channel=ch, sys="G", signal="1C", fs_hz=fs,
                                   t_int_s=1e-3) for ch, p in art.tracking_dumps.items()}  # fmt: skip


def test_the_replayed_rule_drops_where_the_stock_receiver_dropped(tmp_path: Path) -> None:
    base = tmp_path / "base.bin"
    write_ca_recording(base, prn=7, doppler_hz=1200.0, cn0_dbhz=45.0, seconds=40.0, fs_hz=FS, seed=3)
    src = IQSource(path=base, item_type="ibyte", fs_hz=FS, sample_start=0, sample_end=int(40 * FS))
    n0 = estimate_n0(src, frontend_bw_hz=0.8 * FS).n0_per_hz
    jam = Disturbance("noise", {"bw_hz": 3.6e6, "f_hz": 0.0}, 15.0, 0.01, 85.0, 5)  # to about 26 dB-Hz from 15 s
    out = Path("/dev/shm/gnssaudit") / tmp_path.name / "stock.bin"
    inject(src, [jam], FrontEndSpec(1, 1, int16_scale(n0, FS, 85.0)), SinkSpec("shm_file", out), n0_per_hz=n0)
    try:
        tracks = receive(out, FS, "ishort", {0: ("G", "1C", 7)}, {}, tmp_path / "rx")
    finally:
        out.unlink(missing_ok=True)
    tr = tracks[0]
    last = first_loss(tr, 40.0)
    assert last is not None, "the stock receiver should have dropped the channel after the onset"
    assert np.sort(tr.column("t_rx_s").to_numpy())[last] > 15.0
    assert_replay_reaches_the_limit_at(tr, stock_rule("1C"), last)


@pytest.mark.data
def test_the_replay_matches_a_strict_threshold_on_the_cttc_sample(tmp_path: Path) -> None:
    try:
        path = dataset_file("cttc-2013-sample", "2013_04_04_GNSS_SIGNAL_at_CTTC_SPAIN.dat")
        path.stat()
    except FileNotFoundError:
        pytest.skip("the CTTC recording is not fetched")
    strict = {"Tracking_1C.carrier_lock_th": "0.995"}  # every real channel fails it: drops after pull-in + 5 s
    channels = {i: ("G", "1C", p) for i, p in enumerate([1, 11, 17, 20, 32])}
    tracks = receive(path, FS, "ishort", channels, strict, tmp_path / "rx")
    rule = LockRule(carrier_lock_th=0.995, max_carrier_lock_fail=5000, cn0_min_dbhz=25.0, max_lock_fail=50,
                    pull_in_s=5)  # fmt: skip
    checked = 0
    for tr in tracks.values():
        last = first_loss(tr, 100.0)
        if last is None:
            continue
        r = replay(tr, rule)
        carrier = r.column("carrier")[last].as_py()
        assert not r.column("drop").to_numpy(zero_copy_only=False)[: last + 1].any()
        # counted from the first record, up to LATENCY_S after the acquisition stamp: a few integrations short
        assert rule.max_carrier_lock_fail - LATENCY_S / 1e-3 <= carrier <= rule.max_carrier_lock_fail
        checked += 1
    assert checked >= 3
