"""UGE Jammertest 2024 side files: scenario, MON-RF and NAV-PVT CSVs (docs/7 section 2.1).

Fixtures are synthetic and mimic the column layout; the GPL data are never copied into tests.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from gnssaudit.geometry import ecef_to_geodetic, geodetic_to_ecef
from gnssaudit.io.rinex import gps_seconds
from gnssaudit.io.uge import (
    agc_dip_window,
    check_attack_window,
    mean_position_ecef,
    read_mon_rf,
    read_nav_pvt,
    read_scenario,
)

T0 = gps_seconds(2024, 9, 9, 14, 0, 0)
MON_RF_HEADER = (
    "identity,real_time,agcCnt_01,agcCnt_02,jamInd_01,jamInd_02,jammingState_01,jammingState_02,"
    "matched_iTOW,noisePerMS_01,noisePerMS_02"
)
NAV_PVT_HEADER = "timestamp,identity,iTOW,fixType,hAcc,vAcc,height,hMSL,lat,lon,numSV,real_time"


def _clock(t_gps: float) -> str:
    seconds = t_gps - T0
    return f"2024-09-09 14:{int(seconds // 60):02d}:{seconds % 60:09.6f}"


def _itow_ms(t_gps: float) -> int:
    return round((t_gps % 604_800) * 1000)


def _scenario(tmp_path: Path, on: str = "16:25:00", off: str = "16:39:00") -> Path:
    path = tmp_path / "scenario.json"
    log = [
        {"timestamp_utc": f"2024-09-09T{on}Z", "event": "Jamming started"},
        {"timestamp_utc": f"2024-09-09T{off}Z", "event": "Jamming end"},
    ]
    path.write_text(
        json.dumps({"scenario_id": "1.6.4", "attack_parameters": {"description": "PRN"}, "attack_log": log})
    )
    return path


def test_scenario_attack_log_is_local_summer_time(tmp_path: Path) -> None:
    sc = read_scenario(_scenario(tmp_path))
    # 16:25 "Z" is CEST: 14:25 UTC, which is 14:25:18 GPST in 2024
    assert sc.scenario_id == "1.6.4"
    assert sc.t_on_gps_s == gps_seconds(2024, 9, 9, 14, 25, 18)
    assert sc.t_off_gps_s == gps_seconds(2024, 9, 9, 14, 39, 18)


def _mon_rf(tmp_path: Path, agc: list[int]) -> Path:
    path = tmp_path / "mon_rf.csv"
    rows = [MON_RF_HEADER]
    for k, a in enumerate(agc):
        t = T0 + 60.0 * k
        rows.append(f"MON-RF,{_clock(t)},{a},{a - 700},9,9,0,0,{_itow_ms(t)},96,54")
    path.write_text("\n".join(rows) + "\n")
    return path


def test_mon_rf_one_row_per_block_in_gps_time(tmp_path: Path) -> None:
    rf = read_mon_rf(_mon_rf(tmp_path, [5600, 5600, 700]))
    assert rf.num_rows == 6
    blk0 = rf.filter(np.array(rf.column("block").to_pylist()) == 0)
    assert blk0.column("t_gps_s").to_pylist() == [T0, T0 + 60.0, T0 + 120.0]
    assert blk0.column("agc_cnt").to_pylist() == [5600, 5600, 700]


def test_mon_rf_time_must_agree_with_itow(tmp_path: Path) -> None:
    path = _mon_rf(tmp_path, [5600])
    text = path.read_text().replace(str(_itow_ms(T0)), str(_itow_ms(T0) + 18_000))
    path.write_text(text)
    with pytest.raises(ValueError, match="iTOW"):
        read_mon_rf(path)


def _nav_pvt(tmp_path: Path, rows: list[tuple[float, float, float, float, int]]) -> Path:
    """rows: (t_gps, lat_deg, lon_deg, hAcc, fixType), written with the dataset's 1e-7 scaling bug."""
    path = tmp_path / "nav_pvt.csv"
    lines = [NAV_PVT_HEADER]
    for t, lat, lon, h_acc, fix in rows:
        lines.append(f",NAV-PVT,{_itow_ms(t)},{fix},{h_acc},0.26,40.97,6.67,{lat * 1e-7},{lon * 1e-7},32,{_clock(t)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_nav_pvt_undoes_the_1e7_scaling(tmp_path: Path) -> None:
    pvt = read_nav_pvt(_nav_pvt(tmp_path, [(T0, 69.2753, 15.9681, 0.148, 3)]))
    assert pvt.column("lat_deg")[0].as_py() == pytest.approx(69.2753, abs=1e-9)
    assert pvt.column("lon_deg")[0].as_py() == pytest.approx(15.9681, abs=1e-9)
    assert pvt.column("height_m")[0].as_py() == pytest.approx(40.97)
    assert pvt.column("h_acc_m")[0].as_py() == pytest.approx(0.148)


def test_nav_pvt_rejects_coordinates_already_in_degrees(tmp_path: Path) -> None:
    path = _nav_pvt(tmp_path, [(T0, 69.2753e7, 15.9681e7, 0.148, 3)])
    with pytest.raises(ValueError, match="latitude"):
        read_nav_pvt(path)


def test_geodetic_round_trip() -> None:
    lat, lon, h = np.radians(69.2753), np.radians(15.9681), 40.97
    back = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    assert back == pytest.approx((lat, lon, h), abs=1e-9)


def test_mean_position_weights_by_inverse_h_acc_squared_and_skips_bad_fixes(tmp_path: Path) -> None:
    d = 1e-5  # degrees of latitude, about 1.1 m
    rows = [
        (T0, 69.2753, 15.9681, 0.1, 3),
        (T0 + 1, 69.2753 + d, 15.9681, 0.2, 3),  # weight 1/4 of the first
        (T0 + 2, 69.0, 15.0, 0.1, 2),  # 2-D fix: ignored
        (T0 + 99, 70.0, 16.0, 0.1, 3),  # outside the window: ignored
    ]
    pvt = read_nav_pvt(_nav_pvt(tmp_path, rows))
    ecef, spread = mean_position_ecef(pvt, T0, T0 + 10)
    lat, lon, _ = ecef_to_geodetic(ecef)
    assert np.degrees(lat) == pytest.approx(69.2753 + d / 5, abs=1e-8)
    assert np.degrees(lon) == pytest.approx(15.9681, abs=1e-8)
    assert 0.3 < spread < 1.0


def test_agc_dip_window_and_attack_window_check(tmp_path: Path) -> None:
    agc = [5600] * 5 + [2000, 700, 700, 1500] + [5600] * 3  # dip at minutes 5 to 8
    rf = read_mon_rf(_mon_rf(tmp_path, agc))
    start, end = agc_dip_window(rf, block=0, baseline_s=120.0)
    assert (start, end) == (T0 + 300.0, T0 + 480.0)
    sc = read_scenario(_scenario(tmp_path, on="16:04:00", off="16:09:00"))  # 14:04:18 to 14:09:18 GPST
    check_attack_window(sc, rf, block=0, baseline_s=120.0)
    wrong = read_scenario(_scenario(tmp_path, on="14:04:00", off="14:09:00"))  # read as if it were UTC
    with pytest.raises(ValueError, match="AGC"):
        check_attack_window(wrong, rf, block=0, baseline_s=120.0)
