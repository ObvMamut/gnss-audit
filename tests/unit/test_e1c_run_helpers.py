"""Arm c scout run helpers: station sampling, day windows, G1c totals (D-32, proposed), reduction of RGP tables."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.experiments.e1c import day_start_gps, flag_days
from gnssaudit.experiments.e1c_scout import (
    Event,
    Flagged,
    concat_flagged,
    depth_class,
    find_events,
    flag_epochs,
    g1c_totals,
    n_flagged_in,
    screen_sample,
    window,
)
from gnssaudit.geometry import Ephemerides, geodetic_to_ecef
from gnssaudit.io.kp import read_kp_json
from gnssaudit.io.rgp import REDUCED_SCHEMA, detector_table, is_metropolitan, reduce_obs
from gnssaudit.io.schemas import OBS_EPOCH_SAT, build
from tests.unit.test_e1c_scout import LAST, RULE, S, drop_all, make_table


def ecef(lat: float, lon: float) -> np.ndarray:
    return geodetic_to_ecef(np.radians(lat), np.radians(lon), 100.0)


def test_metropolitan_window_keeps_mainland_and_corsica_only() -> None:
    assert is_metropolitan(ecef(43.3, 3.5))  # Agde
    assert is_metropolitan(ecef(42.0, 9.0))  # Corsica
    assert not is_metropolitan(ecef(16.2, -61.5))  # Guadeloupe
    assert not is_metropolitan(ecef(-21.1, 55.5))  # Reunion
    assert not is_metropolitan(ecef(46.9, -56.2))  # Saint-Pierre-et-Miquelon


def _coords() -> dict[str, np.ndarray]:
    """10 metropolitan stations in pairs 10 km apart, one lone metropolitan station, two overseas within 10 km."""
    c = {}
    for i in range(5):
        c[f"a{i:03d}"] = ecef(44.0 + i, 2.0)
        c[f"b{i:03d}"] = ecef(44.0 + i, 2.0 + 10e3 / 111e3 / np.cos(np.radians(44.0 + i)))
    c["lone"] = ecef(48.0, -4.0)
    c["guad"] = ecef(16.2, -61.5)
    c["gua2"] = ecef(16.25, -61.5)
    return c


def test_screen_sample_draws_from_metropolitan_stations_with_a_partner() -> None:
    coords = _coords()
    got = screen_sample(coords, 4, 7, ["a000"], 5e3, 30e3)
    assert got == screen_sample(dict(reversed(list(coords.items()))), 4, 7, ["a000"], 5e3, 30e3)  # order-free
    drawn = got[:4]
    assert drawn == sorted(drawn) and len(set(drawn)) == 4
    assert not {"lone", "guad", "gua2"} & set(got)
    assert got.count("a000") == 1 and "a000" in got
    every = screen_sample(coords, 10, 1, [], 5e3, 30e3)
    assert every == sorted(f"{p}{i:03d}" for p in "ab" for i in range(5))
    assert screen_sample(coords, 4, 8, [], 5e3, 30e3) != screen_sample(coords, 4, 7, [], 5e3, 30e3)
    with pytest.raises(ValueError, match="candidates"):
        screen_sample(coords, 11, 1, [], 5e3, 30e3)


def test_day_windows_joined_give_the_events_of_the_whole_run() -> None:
    """An event cut by a window boundary comes back whole once the windows are joined."""
    whole = flag_epochs(make_table(drops=drop_all(5)), RULE)
    cut = LAST + 120.0
    parts = [window(whole, -np.inf, cut), window(whole, cut, np.inf)]
    assert parts[0].t_evaluated.max() < cut <= parts[1].t_evaluated.min()
    assert find_events(concat_flagged(parts), RULE) == find_events(whole, RULE)
    assert len(find_events(parts[0], RULE)) == 1 and find_events(parts[0], RULE)[0].end == LAST + 119.0
    with pytest.raises(ValueError, match="overlap"):
        concat_flagged([parts[1], parts[0]])
    assert concat_flagged([]).t_evaluated.size == 0


def test_flag_days_matches_one_pass_over_all_days() -> None:
    t0 = day_start_gps(2024, 250) + 40_000.0  # each sidereal-day block inside its calendar day
    full = make_table(start=t0, drops={p: (t0 + 3 * S + 100, t0 + 3 * S + 140, 10.0) for p in range(1, 6)})
    t = full["t_gps_s"].to_numpy()
    days = {}
    for doy in range(250, 254):
        a = day_start_gps(2024, doy)
        sel = (t >= a) & (t < a + 86400.0)
        days[doy] = full.filter(pa.array(sel)).select(["t_gps_s", "prn", "cn0_dbhz", "elev_deg"])
    got = flag_days(days, [253], 2024, RULE)
    a = day_start_gps(2024, 253)
    want = window(flag_epochs(full, RULE), a, a + 86400.0)
    np.testing.assert_array_equal(got.t_evaluated, want.t_evaluated)
    np.testing.assert_array_equal(got.flagged, want.flagged)
    assert len(find_events(got, RULE)) == 1


def test_n_flagged_in_counts_the_event_epochs_not_its_gaps() -> None:
    t = np.arange(0.0, 100.0)
    flagged = (t >= 10) & (t <= 40) & ~((t >= 20) & (t < 25))
    f = Flagged(t, flagged, np.zeros(t.size, dtype=np.int64), np.zeros(t.size))
    assert n_flagged_in(f, 10.0, 40.0) == 26
    assert n_flagged_in(f, 41.0, 99.0) == 0


def test_depth_classes_are_left_closed_and_the_last_is_open() -> None:
    edges = [6.0, 12.0, 20.0]
    assert [depth_class(d, edges) for d in (6.0, 11.99, 12.0, 19.9, 20.0, 45.0)] == [
        "6-12 dB", "6-12 dB", "12-20 dB", "12-20 dB", ">=20 dB", ">=20 dB"
    ]  # fmt: skip
    with pytest.raises(ValueError, match="below"):
        depth_class(5.9, edges)


def _kp_file(tmp_path: Path, t0_iso: str, values: list[float]) -> list[tuple[float, float]]:
    path = tmp_path / "kp.json"
    start = np.datetime64(t0_iso)
    times = [str(start + np.timedelta64(3 * i, "h")) + "Z" for i in range(len(values))]
    path.write_text(json.dumps({"Kp": values, "datetime": times}))
    return read_kp_json(path)


def test_g1c_totals_states_kp_exclusion_classes_and_threshold(tmp_path: Path) -> None:
    kp = _kp_file(tmp_path, "2024-09-01T00:00:00", [2.0, 5.0, 2.0, 2.0])  # bin 1 (03-06 UTC) is stormy
    t_bin = kp[0][0]
    t = t_bin + np.arange(0.0, 4 * 10800.0)
    fl = np.zeros(t.size, dtype=bool)
    spans = [(100, 399, "clear", 8.0), (1000, 1099, "clear", 15.0), (2000, 2199, "clear", 25.0),
             (12000, 12299, "clear", 9.0), (5000, 5099, "vetoed", 9.0), (6000, 6099, "unscreened", 9.0)]  # fmt: skip
    events = []
    for a, b, state, depth in spans:
        fl[a : b + 1] = True
        events.append(Event(t[a], t[b], t[b] - t[a], depth, 5, state))
    f = Flagged(t, fl, np.zeros(t.size, dtype=np.int64), np.zeros(t.size))
    out = g1c_totals(events, f, kp, 1.0, 10.0, 4.0, [6.0, 12.0, 20.0])
    assert out["events_by_partner_state"] == {"clear": 4, "vetoed": 1, "unscreened": 1}
    assert out["clear_kp_excluded"] == {"events": 1, "flagged_min": 5.0}  # the 300 s event in the Kp 5 bin
    assert out["admitted"] == {"events": 3, "flagged_min": pytest.approx((300 + 100 + 200) / 60.0)}
    assert out["classes"]["6-12 dB"] == {"events": 1, "flagged_min": 5.0}
    assert out["classes"]["12-20 dB"]["events"] == 1 and out["classes"][">=20 dB"]["events"] == 1
    assert out["passed_events"] is True
    assert g1c_totals(events, f, kp, 1.0, 10.01, 4.0, [6.0, 12.0, 20.0])["passed_events"] is False
    assert [r["admitted"] for r in out["events"]] == [True, True, True, False, False, False]


def _obs(rows: list[tuple[float, str, int, str, float | None]]) -> pa.Table:
    cols = {
        "source_id": ["x"] * len(rows),
        "receiver_id": ["x"] * len(rows),
        "t_gps_s": [r[0] for r in rows],
        "sys": [r[1] for r in rows],
        "prn": [r[2] for r in rows],
        "signal": [r[3] for r in rows],
        "cn0_dbhz": [r[4] for r in rows],
    }
    return build(OBS_EPOCH_SAT, {n: cols.get(n) for n in OBS_EPOCH_SAT.names}, len(rows))


def test_reduce_obs_keeps_gps_1c_with_cn0_sorted_and_deduplicated() -> None:
    obs = _obs([
        (2.0, "G", 5, "1C", 40.0), (1.0, "G", 7, "1C", 41.0), (1.0, "G", 5, "1C", 42.0), (1.0, "G", 5, "2W", 30.0),
        (1.0, "E", 5, "1X", 44.0), (1.0, "R", 5, "1C", 45.0), (3.0, "G", 9, "1C", None), (1.0, "G", 5, "1C", 99.0),
    ])  # fmt: skip
    out = reduce_obs(obs, Ephemerides([]), ecef(43.3, 3.5))
    assert out.schema.equals(REDUCED_SCHEMA)
    assert out["t_gps_s"].to_pylist() == [1.0, 1.0, 2.0]
    assert out["prn"].to_pylist() == [5, 7, 5]
    assert out["cn0_dbhz"].to_pylist() == [42.0, 41.0, 40.0]  # the first of the duplicated (1.0, G05) rows
    assert np.isnan(out["elev_deg"].to_numpy(zero_copy_only=False)).all()  # no ephemeris: NaN, never counted
    det = detector_table(out)
    assert set(det["sys"].to_pylist()) == {"G"} and set(det["signal"].to_pylist()) == {"1C"}
