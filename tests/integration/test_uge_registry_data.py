"""The registered UGE 1.6.4 intervals agree with the scenario file, the 2024 plan, MON-RF and Kp (docs/7 section 4)."""

import json

import pytest

from gnssaudit.io.kp import KP_THRESHOLD, kp_max, read_kp_json
from gnssaudit.io.rinex import gps_seconds
from gnssaudit.io.uge import UTC_OFFSET_ATTACK_LOG_S, agc_dip_window, read_mon_rf, read_scenario
from gnssaudit.paths import data_root
from gnssaudit.registry import load_intervals

pytestmark = pytest.mark.data
RAW = data_root() / "raw"
SCENARIO = RAW / "uge-jammertest2024" / "extracted/Jamming/stationary/Very High Power (≥10W)/Bands_L1_L2_L5/1.6.4"
if not SCENARIO.is_dir():
    pytest.skip("UGE data not fetched", allow_module_level=True)

IV = load_intervals()


def _plan_window() -> tuple[float, float]:
    plan = json.loads((RAW / "jammertest-plan-2024" / "plan-monday-2024-09-09.json").read_text())
    found: list[dict] = []

    def walk(o: object) -> None:
        if isinstance(o, dict):
            if o.get("test_id") == "1.6.4":
                found.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(plan)
    (entry,) = found

    def gps(stamp: str) -> float:
        d, t = stamp.split("T")
        y, m, dd = map(int, d.split("-"))
        hh, mm, ss = map(int, t.split(":"))
        return gps_seconds(y, m, dd, hh, mm, ss) - UTC_OFFSET_ATTACK_LOG_S + 18.0

    return gps(entry["start_time"]), gps(entry["end_time"])


def test_jam_interval_matches_plan_and_scenario() -> None:
    jam = IV["uge-1.6.4-jam"]
    on, off = _plan_window()
    sc = read_scenario(SCENARIO / "scenario.json")
    assert (jam["t_start"], jam["t_end"]) == (on, off)
    assert sc.t_on_gps_s == on and abs(sc.t_off_gps_s - off) <= 10.0


def test_agc_dip_inside_jam_interval_on_both_blocks() -> None:
    jam = IV["uge-1.6.4-jam"]
    rf = read_mon_rf(SCENARIO / "mon_rf.csv")
    for block in (0, 1):
        start, end = agc_dip_window(rf, block)
        assert jam["t_start"] < start < end < jam["t_end"]


def test_intervals_tile_the_log_and_pass_kp_screening() -> None:
    pre, jam, post = IV["uge-1.6.4-clean-pre"], IV["uge-1.6.4-jam"], IV["uge-1.6.4-clean-post"]
    assert pre["t_end"] == jam["t_start"] and jam["t_end"] == post["t_start"]
    series = read_kp_json(RAW / "gfz-kp" / "kp_2024-09-09.json")
    for entry in (pre, jam, post):
        kp = kp_max(series, entry["t_start"], entry["t_end"])
        assert kp == entry["kp_max"] and kp <= KP_THRESHOLD
