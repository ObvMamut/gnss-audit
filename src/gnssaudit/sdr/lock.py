"""GNSS-SDR v0.0.22's default lock verdict, recomputed from a patched tracking dump (D-34; docs/9 sections 6 and 10).

Twin runs raise `max_carrier_lock_fail` so the carrier-lock rule never drops a channel. The verdict the stock receiver
would have reached is still reported as receiver behaviour: the dump holds, at every integration, the smoothed lock
statistic and C/N0 the rule tests (`log_data`, dll_pll_veml_tracking.cc:1937-1940), so the two fail counters can be
replayed (`cn0_and_tracking_lock_status`, :1352-1408): nothing counts during the pull-in (:2343-2350), which ends
once more than `pull_in_time_s` *whole* seconds have passed since the acquisition sample stamp (integer division,
`tracking_elapsed_seconds`, :2320-2327), so 5 s means about 6 s; each failing integration adds one, each passing one
takes one back down to zero; the channel drops when a counter exceeds its limit, and both counters restart. After a
would-be drop the replay assumes the stock receiver re-acquires at once and pulls in again; a gap in the records (a
real re-acquisition) is a new start too. Valid at T = 1 ms, where every record after the pull-in is one rule
evaluation (docs/9 section 10).

Limits. The acquisition stamp is not in the dump: the replay counts from the first record, which comes some tens of
milliseconds after it (41 ms on the CTTC sample), so a drop decided by the pull-in's end is placed that much late;
once the pull-in is over, the counters match the binary record for record. Telemetry faults, which force a drop in the
stock receiver, are not in the dump.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

GAP_SPACINGS = 1.5  # a record gap longer than this many median spacings is a re-acquisition
SETTLE_S = 2.0  # skipped at the start of a tracking episode: 4 memories of the smoothed lock statistic at 1 ms


@dataclass(frozen=True)
class LockRule:
    carrier_lock_th: float
    max_carrier_lock_fail: int
    cn0_min_dbhz: float
    max_lock_fail: int  # the C/N0 ("code") counter's limit, key `max_lock_fail`
    pull_in_s: int  # whole seconds, as GNSS-SDR's uint32 `pull_in_time_s`


def stock_rule(signal: str) -> LockRule:
    """The v0.0.22 defaults, shared by GPS L1 C/A and L5 (gnss_sdr_flags.cc:44-53, dll_pll_conf.h:64; Verified)."""
    if signal not in ("1C", "L5"):
        raise ValueError(f"no verified lock defaults for signal {signal!r}")
    return LockRule(carrier_lock_th=0.7, max_carrier_lock_fail=5000, cn0_min_dbhz=25.0, max_lock_fail=50, pull_in_s=5)


def replay(track: pa.Table, rule: LockRule) -> pa.Table:
    """The stock rule replayed record by record, in time order: (channel, prn, t_rx_s, carrier, code, drop), the
    counters as they stand after each record and whether the rule drops the channel on it. A stock dump never holds
    the record it dropped on (the dropping integration is not logged), so it ends with a counter at its limit."""
    t = track.column("t_rx_s").to_numpy()
    order = np.argsort(t, kind="stable")
    t = t[order]
    lock = track.column("lock_test").to_numpy(zero_copy_only=False).astype(np.float64)[order]
    cn0 = track.column("cn0_dbhz").to_numpy(zero_copy_only=False).astype(np.float64)[order]
    n = t.size
    carrier_out = np.zeros(n, dtype=np.int64)
    code_out = np.zeros(n, dtype=np.int64)
    drop = np.zeros(n, dtype=bool)
    spacing = float(np.median(np.diff(t))) if n > 1 else 0.0
    start = t[0] if n else 0.0
    carrier = code = 0
    for i in range(n):
        if i and t[i] - t[i - 1] > GAP_SPACINGS * spacing:  # the receiver re-acquired: a new pull-in
            start, carrier, code = t[i], 0, 0
        if np.floor(t[i] - start) > rule.pull_in_s:  # whole seconds since the start
            carrier = carrier + 1 if lock[i] < rule.carrier_lock_th else max(carrier - 1, 0)
            code = code + 1 if cn0[i] < rule.cn0_min_dbhz else max(code - 1, 0)
            if carrier > rule.max_carrier_lock_fail or code > rule.max_lock_fail:
                drop[i] = True
        carrier_out[i], code_out[i] = carrier, code
        if drop[i]:
            start, carrier, code = t[i], 0, 0  # stock: counters reset, re-acquisition, a new pull-in
    return pa.table(
        {
            "channel": track.column("channel").to_numpy()[order],
            "prn": track.column("prn").to_numpy()[order],
            "t_rx_s": t,
            "carrier": carrier_out,
            "code": code_out,
            "drop": drop,
        }
    )


def would_drop(track: pa.Table, rule: LockRule) -> pa.Table:
    """Every time the stock rule would drop the channel: (channel, prn, t_drop_s, counter "carrier" or "cn0")."""
    if track.num_rows == 0:
        return pa.table({"channel": [], "prn": [], "t_drop_s": [], "counter": []})
    r = replay(track, rule)
    d = r.filter(r.column("drop"))
    counter = np.where(d.column("carrier").to_numpy() > rule.max_carrier_lock_fail, "carrier", "cn0")
    return pa.table(
        {
            "channel": d.column("channel"),
            "prn": d.column("prn"),
            "t_drop_s": d.column("t_rx_s"),
            "counter": pa.array(counter.tolist(), pa.string()),
        }
    )


def in_lock(track: pa.Table, *, min_lock: float, settle_s: float) -> NDArray[np.bool_]:
    """Per record (in the table's order): settled and in lock. A tracking episode is a run of records of one PRN on
    the channel without a re-acquisition gap; its first settle_s are skipped, since an episode opens with no C/N0
    estimate yet and a smoothed lock statistic near its initial value (a stock survey re-acquires noise tracks every
    few seconds, P2b task 6). In lock: smoothed lock statistic at least min_lock and C/N0 above 0."""
    n = track.num_rows
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(track.column("sample_count").to_numpy(), kind="stable")
    t = track.column("t_rx_s").to_numpy()[order]
    prn = track.column("prn").to_numpy()[order]
    lock = track.column("lock_test").to_numpy(zero_copy_only=False).astype(np.float64)[order]
    cn0 = track.column("cn0_dbhz").to_numpy(zero_copy_only=False).astype(np.float64)[order]
    d = np.diff(t)
    spacing = float(np.median(d)) if d.size else 0.0
    new = np.r_[True, (d > GAP_SPACINGS * spacing) | (prn[1:] != prn[:-1])]
    start = t[new][np.cumsum(new) - 1]
    ok = (t >= start + settle_s) & (lock >= min_lock) & (cn0 > 0)
    out = np.empty(n, dtype=bool)
    out[order] = ok
    return out
