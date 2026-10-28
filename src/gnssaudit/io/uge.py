"""Side files of the UGE Jammertest 2024 dataset [56]: scenario.json, mon_rf.csv and nav_pvt.csv (docs/7 section 2.1).

Observations come from the UBX file through gnssaudit.io.ubx; these readers add the jamming labels, the RF
monitor and the receiver's own position. Three quirks of the data, found on 10 Oct 2026, are handled here:

- `attack_log` times carry a "Z" but are local summer time (CEST, UTC+2), as the 2024 transmission plan [58] and
  the AGC dip in MON-RF both show.
- The CSV `real_time` column is GPS time, not UTC: it equals iTOW within the GPS week. Each row is checked.
- The NAV-PVT `lat` and `lon` columns are degrees scaled by 1e-7.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.geometry import WEEK_S, geodetic_to_ecef
from gnssaudit.io.rinex import gps_seconds
from gnssaudit.io.schemas import PVT_EPOCH, RF_EPOCH, build, validate

FloatArray = NDArray[np.float64]

UTC_OFFSET_ATTACK_LOG_S = 7200.0  # CEST
GPS_UTC_S = 18.0  # leap seconds, valid from 2017-01-01 to at least 2026
LATLON_SCALE = 1e7


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    t_on_gps_s: float
    t_off_gps_s: float
    description: str


def _gps_of_clock(text: str) -> float:
    d = datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    return gps_seconds(d.year, d.month, d.day, d.hour, d.minute, d.second + d.microsecond / 1e6)


def read_scenario(path: Path) -> Scenario:
    """Attack window in GPS seconds, with the attack_log's CEST times converted to GPS time."""
    meta = json.loads(path.read_text())
    times = {e["event"].lower(): e["timestamp_utc"] for e in meta["attack_log"]}
    on = next(t for e, t in times.items() if "start" in e)
    off = next(t for e, t in times.items() if "end" in e)
    if datetime.fromisoformat(on.replace("Z", "")).year < 2017:
        raise ValueError("GPS_UTC_S = 18 holds only from 2017")
    t_on, t_off = (_gps_of_clock(x) - UTC_OFFSET_ATTACK_LOG_S + GPS_UTC_S for x in (on, off))
    params = meta.get("attack_parameters", {})
    return Scenario(str(meta["scenario_id"]), t_on, t_off, str(params.get("description", "")))


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _gps_time(row: dict[str, str], itow_key: str) -> float:
    t = _gps_of_clock(row["real_time"])
    if abs((t % WEEK_S) - int(row[itow_key]) / 1000.0) > 1e-3:
        raise ValueError(f"real_time {row['real_time']} disagrees with iTOW {row[itow_key]}: not GPS time")
    return t


def read_mon_rf(path: Path) -> pa.Table:
    """rf_epoch rows, one per epoch and RF block (block 0 is L1, block 1 is L2 or L5)."""
    cols: dict[str, list] = {n: [] for n in RF_EPOCH.names}
    for row in _rows(path):
        t = _gps_time(row, "matched_iTOW")
        blocks = sorted({k.rsplit("_", 1)[1] for k in row if k.startswith("agcCnt_")})
        for b in blocks:
            cols["t_gps_s"].append(t)
            cols["block"].append(int(b) - 1)
            cols["agc_cnt"].append(int(row[f"agcCnt_{b}"]))
            cols["jam_ind"].append(int(row[f"jamInd_{b}"]))
            cols["jamming_state"].append(int(row[f"jammingState_{b}"]))
            cols["noise_per_ms"].append(int(row[f"noisePerMS_{b}"]))
    return validate(build(RF_EPOCH, cols, len(cols["t_gps_s"])), RF_EPOCH)


def read_nav_pvt(path: Path) -> pa.Table:
    """pvt_epoch rows with lat/lon rescaled to degrees; heights are ellipsoidal (m), accuracies in m."""
    cols: dict[str, list] = {n: [] for n in PVT_EPOCH.names}
    for row in _rows(path):
        lat, lon = float(row["lat"]) * LATLON_SCALE, float(row["lon"]) * LATLON_SCALE
        if abs(lat) > 90.0 or abs(lon) > 180.0:
            raise ValueError(f"latitude {lat} or longitude {lon} out of range after the 1e7 rescaling")
        cols["t_gps_s"].append(_gps_time(row, "iTOW"))
        cols["lat_deg"].append(lat)
        cols["lon_deg"].append(lon)
        cols["height_m"].append(float(row["height"]))
        cols["h_acc_m"].append(float(row["hAcc"]))
        cols["v_acc_m"].append(float(row["vAcc"]))
        cols["fix_type"].append(int(row["fixType"]))
        cols["num_sv"].append(int(row["numSV"]))
    return validate(build(PVT_EPOCH, cols, len(cols["t_gps_s"])), PVT_EPOCH)


def mean_position_ecef(pvt: pa.Table, t0: float, t1: float) -> tuple[FloatArray, float]:
    """hAcc^-2-weighted mean ECEF of the 3-D fixes in [t0, t1], and the weighted RMS distance from it (m)."""
    pvt = validate(pvt, PVT_EPOCH)
    t = pvt.column("t_gps_s").to_numpy()
    use = (t >= t0) & (t <= t1) & (pvt.column("fix_type").to_numpy() == 3)
    if not use.any():
        raise ValueError("no 3-D fix in the interval")
    lat = np.radians(pvt.column("lat_deg").to_numpy()[use])
    lon = np.radians(pvt.column("lon_deg").to_numpy()[use])
    h = pvt.column("height_m").to_numpy()[use]
    xyz = np.array([geodetic_to_ecef(a, b, c) for a, b, c in zip(lat, lon, h, strict=True)])
    w = 1.0 / pvt.column("h_acc_m").to_numpy()[use].astype(float) ** 2
    mean = (w[:, None] * xyz).sum(axis=0) / w.sum()
    spread = float(np.sqrt((w * ((xyz - mean) ** 2).sum(axis=1)).sum() / w.sum()))
    return mean, spread


def agc_dip_window(rf: pa.Table, block: int, baseline_s: float = 120.0, drop_frac: float = 0.5) -> tuple[float, float]:
    """First and last epoch whose AGC count is below drop_frac x the median over the first baseline_s seconds."""
    rf = validate(rf, RF_EPOCH)
    sel = rf.column("block").to_numpy() == block
    t = rf.column("t_gps_s").to_numpy()[sel]
    agc = rf.column("agc_cnt").to_numpy()[sel].astype(float)
    base = np.median(agc[t < t[0] + baseline_s])
    low = np.flatnonzero(agc < drop_frac * base)
    if low.size == 0:
        raise ValueError(f"no AGC dip below {drop_frac} x {base:.0f} in block {block}")
    return float(t[low[0]]), float(t[low[-1]])


def check_attack_window(sc: Scenario, rf: pa.Table, block: int = 0, baseline_s: float = 120.0) -> None:
    """Raise unless the AGC dip lies inside the scenario's attack window."""
    try:
        start, end = agc_dip_window(rf, block, baseline_s)
    except ValueError as err:
        raise ValueError(f"AGC shows no jamming: {err}") from err
    if start < sc.t_on_gps_s or end > sc.t_off_gps_s:
        raise ValueError(
            f"AGC dip {start:.0f} to {end:.0f} lies outside the attack window "
            f"{sc.t_on_gps_s:.0f} to {sc.t_off_gps_s:.0f}"
        )
