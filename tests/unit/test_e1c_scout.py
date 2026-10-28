"""E1(c) scout detector: sidereal reference, flagged epochs, events, partner veto (experiments/E1c-scout.yaml)."""

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import yaml

from gnssaudit.experiments.e1c_scout import (
    ScoutRule,
    detect_events,
    find_events,
    flag_epochs,
    load_rule,
    screen_partner,
)

RULE = load_rule(Path(__file__).parents[2] / "experiments" / "E1c-scout.yaml")
S = RULE.sidereal_s
T0 = 1_000_000.0  # day 3 of the synthetic series starts at T0 + 3 S
N_SAT = 8


def make_table(
    days: int = 4,
    dt: float = 1.0,
    start: float = T0,
    span_s: float = 600.0,
    drops: dict[int, tuple[float, float, float]] | None = None,
    elev: float = 45.0,
    sys: str = "G",
    signal: str = "1C",
    n_sat: int = N_SAT,
) -> pa.Table:
    """N_SAT satellites with a repeating pattern each sidereal day; drops = {prn: (t_from, t_to, depth_db)}."""
    t1 = np.arange(0.0, span_s, dt)
    ts, prns, cn0 = [], [], []
    for day in range(days):
        for prn in range(1, n_sat + 1):
            base = 40.0 + prn + 2.0 * np.sin(2 * np.pi * t1 / 300.0 + prn)
            t = start + day * S + t1
            y = base.copy()
            if day == days - 1 and drops and prn in drops:
                a, b, d = drops[prn]
                y[(t >= a) & (t < b)] -= d
            ts.append(t)
            prns.append(np.full(t.shape, prn))
            cn0.append(y)
    t_all = np.concatenate(ts)
    return pa.table(
        {
            "t_gps_s": t_all,
            "sys": np.full(t_all.shape, sys),
            "prn": np.concatenate(prns).astype(np.int16),
            "signal": np.full(t_all.shape, signal),
            "cn0_dbhz": np.concatenate(cn0).astype(np.float32),
            "elev_deg": np.full(t_all.shape, elev, dtype=np.float32),
        }
    )


LAST = T0 + 3 * S


def drop_all(n: int, a: float = 100.0, b: float = 140.0, depth: float = 10.0) -> dict[int, tuple[float, float, float]]:
    return {prn: (LAST + a, LAST + b, depth) for prn in range(1, n + 1)}


def test_rule_values_come_from_the_frozen_yaml() -> None:
    cfg = yaml.safe_load((Path(__file__).parents[2] / "experiments" / "E1c-scout.yaml").read_text())["rule"]
    assert RULE == ScoutRule.from_config(cfg)
    assert (RULE.drop_db, RULE.min_sats, RULE.min_frac, RULE.elev_mask_deg) == (6.0, 4, 0.5, 10.0)
    assert (RULE.merge_gap_s, RULE.min_duration_s, RULE.partner_guard_s, RULE.sidereal_s) == (30.0, 10.0, 60.0, 86164.0)
    assert RULE.partner_min_coverage == 0.8
    assert RULE.lags == (1, 2, 3) and RULE.reference_signals == {"G": ("1C",)}


def test_frozen_rule_block_is_unchanged_since_the_freeze() -> None:
    """Pre-registration guard: SHA-256 of the canonical JSON of the YAML `rule` block, as frozen in 45a7f80.

    A change to any threshold needs a dated amendment in docs/10 and docs/7 and a new digest here."""
    spec = yaml.safe_load((Path(__file__).parents[2] / "experiments" / "E1c-scout.yaml").read_text())
    canonical = json.dumps(spec["rule"], sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == "e0f88bbe0390e8abebd71d997971394d14019858540de3b5cbbe077b70b9f284"


def test_injected_multi_satellite_drop_is_detected() -> None:
    events = detect_events(make_table(drops=drop_all(5)), RULE)
    assert len(events) == 1
    ev = events[0]
    assert ev.start == LAST + 100.0 and ev.end == LAST + 139.0 and ev.duration == 39.0
    assert ev.n_sats == 5
    assert abs(ev.depth_db - 10.0) < 1e-3


def test_single_satellite_fade_is_not_an_event() -> None:
    assert detect_events(make_table(drops={1: (LAST + 100, LAST + 200, 20.0)}), RULE) == []


def test_three_satellites_are_below_min_sats() -> None:
    assert detect_events(make_table(drops=drop_all(3)), RULE) == []


def test_fraction_rule_needs_half_of_the_satellites_above_the_mask() -> None:
    """4 dropped of 8 = 50 %: flagged. 4 dropped of 9 = 44 %: not."""
    assert len(detect_events(make_table(drops=drop_all(4)), RULE)) == 1
    assert detect_events(make_table(drops=drop_all(4), n_sat=9), RULE) == []


def test_drop_below_6_db_is_not_counted() -> None:
    assert detect_events(make_table(drops=drop_all(6, depth=5.9)), RULE) == []


def test_satellites_below_the_mask_or_with_nan_elevation_are_not_counted() -> None:
    low = make_table(drops=drop_all(6), elev=5.0)
    assert detect_events(low, RULE) == []
    nan = make_table(drops=drop_all(6), elev=float("nan"))
    assert detect_events(nan, RULE) == []


def test_short_spell_under_10_s_is_dropped_and_gaps_under_30_s_merge() -> None:
    assert detect_events(make_table(drops=drop_all(5, 100, 109)), RULE) == []  # 9 s: end - start = 8
    # second spell 25 s after the first ends: merged; 35 s: two events
    for gap, n in ((25.0, 1), (35.0, 2)):
        d = {prn: (LAST + 100, LAST + 120, 10.0) for prn in range(1, 6)}
        t = make_table(drops=d)
        t2 = make_table(drops={prn: (LAST + 120 + gap, LAST + 140 + gap, 10.0) for prn in range(1, 6)})
        both = np.minimum(t["cn0_dbhz"].to_numpy(), t2["cn0_dbhz"].to_numpy())
        merged = t.set_column(4, "cn0_dbhz", pa.array(both))
        assert len(detect_events(merged, RULE)) == n


def test_satellite_epoch_with_fewer_than_two_reference_lags_is_not_counted() -> None:
    """With 2 days of history only lag 1 exists: nothing has a reference, nothing is flagged."""
    assert detect_events(make_table(days=2, drops=drop_all(6) | {}), RULE) == []  # drop lands on day 2, lag 2 missing
    # three days: lags 1 and 2 exist (2 of 3)
    t = make_table(days=3, drops={prn: (T0 + 2 * S + 100, T0 + 2 * S + 140, 10.0) for prn in range(1, 6)})
    assert len(detect_events(t, RULE)) == 1


def test_reference_is_a_median_so_one_disturbed_past_day_is_ignored() -> None:
    t = make_table(drops=drop_all(5))
    cn0 = t["cn0_dbhz"].to_numpy().copy()
    tt = t["t_gps_s"].to_numpy()
    cn0[(tt >= T0 + S + 100) & (tt < T0 + S + 140)] -= 15.0  # day 2 itself dropped
    ev = detect_events(t.set_column(4, "cn0_dbhz", pa.array(cn0)), RULE)
    assert len(ev) == 1 and abs(ev[0].depth_db - 10.0) < 1e-3


def test_only_gps_1c_is_a_reference_signal() -> None:
    assert detect_events(make_table(drops=drop_all(6), sys="R", signal="1C"), RULE) == []
    assert detect_events(make_table(drops=drop_all(6), sys="E", signal="1X"), RULE) == []
    assert detect_events(make_table(drops=drop_all(6), signal="2W"), RULE) == []


def test_partner_drop_vetoes_the_event_and_clean_partner_admits_it() -> None:
    own = flag_epochs(make_table(drops=drop_all(5)), RULE)
    events = find_events(own, RULE)
    vetoed = screen_partner(events, flag_epochs(make_table(drops=drop_all(5, 150, 190)), RULE), RULE)
    assert [e.partner_state for e in vetoed] == ["vetoed"]  # partner flagged at 150..189, guard reaches 199
    far = screen_partner(events, flag_epochs(make_table(drops=drop_all(5, 300, 340)), RULE), RULE)
    assert [e.partner_state for e in far] == ["clear"]  # 300 > 139 + 60
    edge = screen_partner(events, flag_epochs(make_table(drops=drop_all(5, 199, 240)), RULE), RULE)
    assert [e.partner_state for e in edge] == ["vetoed"]  # exactly end + 60 is inside the window


def test_partner_without_data_in_the_window_is_unscreened_not_clear() -> None:
    events = find_events(flag_epochs(make_table(drops=drop_all(5)), RULE), RULE)
    gone = flag_epochs(make_table(start=T0 + 10 * S), RULE)
    assert [e.partner_state for e in screen_partner(events, gone, RULE)] == ["unscreened"]


def thin_last_day(table: pa.Table, keep) -> pa.Table:
    """Drop last-day rows whose offset from LAST is not selected by keep(offset); history days stay."""
    off = table["t_gps_s"].to_numpy() - LAST
    return table.filter(pa.array((off < 0) | keep(off)))


def test_partner_with_data_only_in_the_guard_margin_is_unscreened() -> None:
    events = find_events(flag_epochs(make_table(drops=drop_all(5)), RULE), RULE)  # event 100..139
    margin = thin_last_day(make_table(), lambda o: ((o >= 40) & (o < 100)) | ((o >= 140) & (o < 200)))
    assert [e.partner_state for e in screen_partner(events, flag_epochs(margin, RULE), RULE)] == ["unscreened"]


def test_partner_coverage_threshold_is_80_percent_of_the_event_stamps() -> None:
    events = find_events(flag_epochs(make_table(drops=drop_all(5)), RULE), RULE)  # 40 stamps, 100..139
    for covered, state in ((34, "clear"), (32, "clear"), (31, "unscreened"), (28, "unscreened")):
        partner = thin_last_day(make_table(), lambda o, c=covered: (o < 100 + c) | (o >= 140))
        got = screen_partner(events, flag_epochs(partner, RULE), RULE)
        assert got[0].partner_state == state, covered
    # stamps are counted once even when the partner table repeats rows
    dup = pa.concat_tables([thin_last_day(make_table(), lambda o: (o >= 100) & (o < 128))] * 2)
    assert screen_partner(events, flag_epochs(dup, RULE), RULE)[0].partner_state == "unscreened"


def test_a_flagged_partner_epoch_wins_over_low_coverage() -> None:
    events = find_events(flag_epochs(make_table(drops=drop_all(5)), RULE), RULE)
    partner = thin_last_day(make_table(drops=drop_all(5, 150, 190)), lambda o: o >= 150)
    assert screen_partner(events, flag_epochs(partner, RULE), RULE)[0].partner_state == "vetoed"


def test_reference_comes_from_past_days_even_with_nan_elevation_there() -> None:
    t = make_table(drops=drop_all(5))
    elev = t["elev_deg"].to_numpy().copy()
    elev[t["t_gps_s"].to_numpy() < LAST] = np.nan  # past days have no elevation; the current day does
    assert len(detect_events(t.set_column(5, "elev_deg", pa.array(elev)), RULE)) == 1


def test_denominator_is_the_reference_bearing_set() -> None:
    """4 dropped of 8 referenced satellites flags; a 9th seen only today has no reference, so is not counted."""
    t = make_table(drops=drop_all(4))
    n = 600
    new = pa.table(
        {
            "t_gps_s": LAST + np.arange(n, dtype=float),
            "sys": np.full(n, "G"),
            "prn": np.full(n, 9, dtype=np.int16),
            "signal": np.full(n, "1C"),
            "cn0_dbhz": np.full(n, 45.0, dtype=np.float32),
            "elev_deg": np.full(n, 45.0, dtype=np.float32),
        }
    )
    assert len(detect_events(pa.concat_tables([t, new]), RULE)) == 1
    assert len(detect_events(make_table(drops=drop_all(4), n_sat=9), RULE)) == 0  # with a reference it does count


def test_detect_events_with_partner_table_returns_state() -> None:
    own = make_table(drops=drop_all(5))
    ev = detect_events(own, RULE, partner=make_table(drops=drop_all(5)))
    assert [e.partner_state for e in ev] == ["vetoed"]
    ev = detect_events(own, RULE, partner=make_table())
    assert [e.partner_state for e in ev] == ["clear"]
    assert detect_events(own, RULE)[0].partner_state == "not_screened"


def test_rule_is_frozen() -> None:
    import dataclasses

    try:
        replace(RULE, drop_db=3.0).drop_db = 1.0  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("ScoutRule must be frozen")


def test_a_synthetic_day_runs_in_seconds() -> None:
    """Timing guard (a station-day at 1 s, 8 satellites): the loop-free path takes well under 10 s."""
    t = make_table(days=4, span_s=S, drops=drop_all(5))
    t0 = time.perf_counter()
    events = detect_events(t, RULE)
    assert time.perf_counter() - t0 < 10.0
    assert len(events) == 1
