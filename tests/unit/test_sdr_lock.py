"""D-34: GNSS-SDR v0.0.22's default lock verdict, recomputed from the patched dump (P2b task 8; docs/9 sections 6
and 10). Twin runs raise the carrier-lock limit; the stock verdict is still reported as receiver behaviour."""

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.io.schemas import TRACK_STREAM, build
from gnssaudit.sdr.lock import LockRule, in_lock, replay, stock_rule, would_drop

RULE = LockRule(carrier_lock_th=0.7, max_carrier_lock_fail=50, cn0_min_dbhz=25.0, max_lock_fail=20, pull_in_s=1)


def track(lock: np.ndarray, cn0: np.ndarray, *, t0: float = 1.0, dt: float = 1e-3, prn: int = 7) -> pa.Table:
    n = lock.size
    t = t0 + dt * np.arange(n)
    nulls = {c: [None] * n for c in TRACK_STREAM.names}
    cols = {**nulls, "run_id": ["r"] * n, "channel": [3] * n, "sys": ["G"] * n, "signal": ["1C"] * n,
            "prn": [prn] * n, "k": list(range(n)), "sample_count": (t * 4e6).astype(np.uint64), "t_rx_s": t,
            "lock_test": lock.astype(np.float32), "cn0_dbhz": cn0.astype(np.float32), "acc_phase_rad": np.zeros(n),
            "t_int_s": [dt] * n}  # fmt: skip
    return build(TRACK_STREAM, cols, n)


def test_the_stock_defaults_are_those_of_v0_0_22() -> None:
    for sig in ("1C", "L5"):
        r = stock_rule(sig)
        assert (r.carrier_lock_th, r.max_carrier_lock_fail, r.cn0_min_dbhz, r.max_lock_fail, r.pull_in_s) == (
            0.7, 5000, 25.0, 50, 5)  # fmt: skip


def test_a_carrier_drop_comes_after_more_failures_than_the_limit() -> None:
    """The counter moves only after the pull-in, and the channel drops when it exceeds the limit (> not >=)."""
    lock = np.full(3000, 0.9)
    lock[2500:] = 0.1  # fails from record 2500 on, after the pull-in
    out = would_drop(track(lock, np.full(3000, 45.0)), RULE)
    assert out.column("counter")[0].as_py() == "carrier"  # the failure persists, so the drops repeat after it
    assert out.column("t_drop_s")[0].as_py() == pytest.approx(1.0 + 1e-3 * (2500 + 50))  # the 51st failure
    assert out.column("prn")[0].as_py() == 7 and out.column("channel")[0].as_py() == 3


def test_passes_pay_the_counter_back_one_at_a_time() -> None:
    """Fail, pass, fail, pass...: the counter never builds up, so nothing drops."""
    lock = np.where(np.arange(3000) % 2 == 0, 0.1, 0.9)
    assert would_drop(track(lock, np.full(3000, 45.0)), RULE).num_rows == 0


def test_nothing_counts_during_the_pull_in() -> None:
    """The pull-in ends once more than pull_in_s whole seconds have passed (GNSS-SDR counts in whole seconds,
    dll_pll_veml_tracking.cc:2320-2327): with 1 s, counting starts 2 s after the start."""
    lock = np.full(3000, 0.9)
    lock[:1990] = 0.0  # 1990 failures inside the pull-in
    lock[2010:2050] = 0.0  # then 40, under the limit of 50
    assert would_drop(track(lock, np.full(3000, 45.0)), RULE).num_rows == 0


def test_low_cn0_drops_on_its_own_counter() -> None:
    cn0 = np.full(3000, 45.0)
    cn0[2500:] = 20.0
    out = would_drop(track(np.full(3000, 0.9), cn0), RULE)
    assert out.column("counter")[0].as_py() == "cn0"
    assert out.column("t_drop_s")[0].as_py() == pytest.approx(1.0 + 1e-3 * (2500 + 20))


def test_after_a_drop_the_counters_restart_with_a_new_pull_in() -> None:
    """As the stock receiver would: re-acquisition, counters at zero, another pull-in."""
    lock = np.full(8000, 0.9)
    lock[2500:] = 0.1
    out = would_drop(track(lock, np.full(8000, 45.0)), RULE)
    t = np.array(out.column("t_drop_s").to_pylist())
    assert t[0] == pytest.approx(1.0 + 1e-3 * 2550)
    np.testing.assert_allclose(np.diff(t), 2.0 + 0.051, atol=2e-3)  # each time: a 2 s pull-in, then 51 failures
    assert t.size == 3  # 2550 + 2051 k up to the 8000th record


def test_a_record_gap_is_a_new_tracking_start() -> None:
    """A gap in the records (the receiver re-acquired) starts a new pull-in."""
    lock = np.full(2000, 0.1)
    a = track(lock[:400], np.full(400, 45.0))  # never past its pull-in
    b = track(lock[:400], np.full(400, 45.0), t0=10.0)
    assert would_drop(pa.concat_tables([a, b]), RULE).num_rows == 0


def test_the_replay_exposes_both_counters_record_by_record() -> None:
    """A stock receiver does not log the integration it drops on, so a stock dump ends with a counter exactly at its
    limit; the replay must show that state (the validation of D-34's recomputation reads it)."""
    lock = np.full(3000, 0.9)
    lock[2500:] = 0.1
    r = replay(track(lock, np.full(3000, 45.0)), RULE)
    carrier = r.column("carrier").to_numpy()
    drop = r.column("drop").to_numpy(zero_copy_only=False)
    assert carrier[2549] == 50 and not drop[2549]  # the 50th failure: at the limit, not over it
    assert drop[2550] and carrier[2551] == 0  # the 51st drops; the counter restarts
    assert r.column("code").to_numpy().max() == 0


def test_records_in_lock_skip_the_start_of_every_tracking_episode() -> None:
    """A survey on the stock receiver drops a noise track and re-acquires another (P2b task 6). Each new episode opens
    with no C/N0 estimate yet (0 dB-Hz) and a lock statistic near its initial 1, so its first seconds say nothing:
    only settled records whose lock statistic reaches the bar count. An episode ends at a gap or at a change of PRN."""
    n = 4000  # 4 s per episode at 1 ms
    opening = np.r_[np.ones(500, dtype=bool), np.zeros(n - 500, dtype=bool)]  # 0.5 s at the initial values
    noise = track(np.where(opening, 1.0, 0.01), np.where(opening, 0.0, 29.0), t0=1.0)
    real = track(np.where(opening, 1.0, 0.9), np.where(opening, 0.0, 42.0), t0=5.0, prn=9)  # no gap: a new PRN
    again = track(np.where(opening, 1.0, 0.9), np.where(opening, 0.0, 42.0), t0=10.0, prn=9)  # after a 1 s gap
    ok = in_lock(pa.concat_tables([noise, real, again]), min_lock=0.2, settle_s=2.0)
    assert not ok[:n].any()  # the noise episode, its opening included
    for e in (1, 2):  # the real episodes, once settled
        assert not ok[e * n : e * n + 2000].any() and ok[e * n + 2000 : (e + 1) * n].all()
