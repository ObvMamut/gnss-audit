"""E1 arm c, scout run: the frozen detector (`e1c_scout`) on IGN RGP data, screen and one station-month (docs/10 2.2).

`gnssaudit run experiments/E1c-scout.yaml` runs two stages:
- **screen**: one target day for a seeded sample of metropolitan stations, each with its nearest partner, read from
  the reduced tables that `e1c_acquire screen` left (the raw hourly files are deleted after reduction). It ranks the
  stations by "clear" events (ties: more partners, then name) and names the month station.
- **month** (when `month.station` is set): the kept raw hourly files of the station and its partner are reduced here,
  in a process pool, then scanned day by day, each day with its 3 history days. The day windows are joined before
  events are formed, so an event crossing midnight stays one event. G1c is evaluated under D-32, which is
  *proposed, pending owner confirmation*: every verdict this module writes is labelled provisional.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gnssaudit.experiments import Outcome
from gnssaudit.experiments.e1c_scout import (
    Event,
    Flagged,
    ScoutRule,
    concat_flagged,
    find_events,
    flag_epochs,
    g1c_totals,
    screen_partner,
    screen_sample,
    window,
)
from gnssaudit.geometry import Ephemerides, FloatArray
from gnssaudit.io.kp import read_kp_json
from gnssaudit.io.rgp import REDUCED_SCHEMA, detector_table, hourly_url, partners, read_rgf93_coords, reduce_hourly
from gnssaudit.io.rinex import gps_seconds, read_rinex_nav
from gnssaudit.paths import data_root

DAY_S = 86_400.0
PROVISIONAL = "provisional: D-32 is proposed, pending owner confirmation"


# ---- data layout under data_root() ----------------------------------------------------------------------------------


def day_date(year: int, doy: int) -> date:
    return date(year, 1, 1) + timedelta(days=doy - 1)


def day_start_gps(year: int, doy: int) -> float:
    d = day_date(year, doy)
    return gps_seconds(d.year, d.month, d.day, 0, 0, 0.0)


def raw_hour_path(station: str, year: int, doy: int, hour: int, root: Path | None = None) -> Path:
    """Where a kept hourly file lives: raw/rgp/YYYY/DDD/<name>, as on the server."""
    name = hourly_url(station, year, doy, hour).rsplit("/", 1)[-1]
    return (root or data_root() / "raw" / "rgp" / f"{year}" / f"{doy:03d}") / name


def nav_name(year: int, doy: int) -> str:
    return f"BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx"


def nav_path(year: int, doy: int) -> Path:
    return data_root() / "raw" / "igs-brdc" / "extracted" / nav_name(year, doy)


def coords_path(year: int, doy: int) -> Path:
    return data_root() / "raw" / "rgp-coords" / f"coorrgf93_{year}-{doy:03d}.txt"


def reduced_path(station: str, year: int, doy: int) -> Path:
    return data_root() / "reduced" / "rgp" / f"{year}" / f"{doy:03d}" / f"{station}.parquet"


def screen_index_path(year: int, doy: int) -> Path:
    return data_root() / "reduced" / "rgp" / f"{year}" / f"{doy:03d}" / "screen-index.json"


def kp_path(year: int, month: int) -> Path:
    return data_root() / "raw" / "gfz-kp" / f"kp_{year}-{month:02d}.json"


# ---- reduction in a process pool ------------------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _nav(path: str) -> Ephemerides:  # cached per worker process
    return read_rinex_nav(Path(path))[0]


def reduce_job(job: tuple[str, str, tuple[float, float, float]]) -> pa.Table:
    raw, nav, rx = job
    return reduce_hourly(Path(raw), _nav(nav), np.array(rx))


def workers() -> int:
    return max(1, min(16, (os.cpu_count() or 2) - 2))


def reduce_files(jobs: list[tuple[Path, Path, FloatArray]], pool: ProcessPoolExecutor) -> list[pa.Table]:
    """Reduced tables of hourly files, in job order (jobs: raw file, nav file, station ECEF)."""
    args = [(str(r), str(n), (float(x[0]), float(x[1]), float(x[2]))) for r, n, x in jobs]
    return list(pool.map(reduce_job, args, chunksize=1))


def concat_reduced(tables: list[pa.Table]) -> pa.Table:
    return pa.concat_tables(tables) if tables else REDUCED_SCHEMA.empty_table()


# ---- detection over day windows -------------------------------------------------------------------------------------


def flag_days(days: dict[int, pa.Table], targets: list[int], year: int, rule: ScoutRule) -> Flagged:
    """Each target day flagged with its history days (the days before it present in `days`), cut to the target day,
    and the windows joined."""
    parts = []
    hist = max(rule.lags)
    for doy in targets:
        tables = [days[d] for d in range(doy - hist, doy + 1) if d in days]
        f = flag_epochs(detector_table(concat_reduced(tables)), rule)
        t0 = day_start_gps(year, doy)
        parts.append(window(f, t0, t0 + DAY_S))
    return concat_flagged(parts)


def epoch_interval_s(days: dict[int, pa.Table]) -> float:
    """Median spacing of a station's distinct epochs (1 s for the RGP data_1 files)."""
    t = np.unique(concat_reduced(list(days.values()))["t_gps_s"].to_numpy())
    if t.size < 2:
        raise ValueError("fewer than two epochs")
    return float(np.median(np.diff(t)))


def _state_counts(events: list[Event]) -> dict[str, int]:
    out = {"clear": 0, "vetoed": 0, "unscreened": 0}
    for e in events:
        out[e.partner_state] += 1
    return out


# ---- stages ---------------------------------------------------------------------------------------------------------


def _screen(spec: dict[str, Any], rule: ScoutRule, inputs: list[Path]) -> dict[str, Any]:
    sc, pr = spec["screen"], spec["partner"]
    year, doy = sc["year"], sc["doy"]
    min_m, max_m = pr["min_km"] * 1e3, pr["max_km"] * 1e3
    cpath, ipath = coords_path(year, doy), screen_index_path(year, doy)
    inputs += [cpath, ipath]
    coords = read_rgf93_coords(cpath)
    sample = screen_sample(coords, sc["n_stations"], sc["seed"], sc["always"], min_m, max_m)
    index = json.loads(ipath.read_text())
    if sorted(index["stations"]) != sorted(sample):
        raise ValueError("the screen index does not hold the sample drawn from the coordinate file and seed")
    days = list(range(doy - max(rule.lags), doy + 1))
    rows = []
    for station in sample:
        entry = index["stations"][station]
        tables = {}
        for who in (station, entry["partner"]):
            if who is None:
                continue
            paths = [reduced_path(who, year, d) for d in days]
            inputs += paths
            tables[who] = {d: pq.read_table(p) for d, p in zip(days, paths, strict=True)}
        own = flag_days(tables[station], [doy], year, rule)
        events = find_events(own, rule)
        if entry["partner"] is not None:
            events = screen_partner(events, flag_days(tables[entry["partner"]], [doy], year, rule), rule)
        else:
            events = [Event(e.start, e.end, e.duration, e.depth_db, e.n_sats, "unscreened") for e in events]
        rows.append(
            {"station": station, "n_partners": len(partners(station, coords, min_m, max_m)),
             "partner": entry["partner"], "partner_km": entry["partner_km"],
             "missing_hours": entry["missing_hours"], "evaluated_epochs": int(own.t_evaluated.size),
             "flagged_epochs": int(own.flagged.sum()), "events": len(events), **_state_counts(events),
             "event_list": [{"start": e.start, "end": e.end, "depth_db": e.depth_db, "n_sats": e.n_sats,
                             "partner_state": e.partner_state} for e in events]}
        )  # fmt: skip
    best = min(rows, key=lambda r: (-r["clear"], -r["n_partners"], r["station"]))
    return {"year": year, "doy": doy, "sample": sample, "rows": rows,
            "pick": {"station": best["station"], "clear": best["clear"], "n_partners": best["n_partners"]}}  # fmt: skip


def _month(spec: dict[str, Any], rule: ScoutRule, inputs: list[Path], counts: dict[str, int]) -> dict[str, Any]:
    mo, g = spec["month"], spec["g1c"]
    year, station, partner = mo["year"], mo["station"], mo["partner"]
    targets = list(range(mo["doy_first"], mo["doy_last"] + 1))
    days = list(range(mo["doy_first"] - max(rule.lags), mo["doy_last"] + 1))
    coords: dict[int, dict[str, FloatArray]] = {}
    for doy in days:
        inputs += [coords_path(year, doy), nav_path(year, doy)]
        coords[doy] = read_rgf93_coords(coords_path(year, doy))
    # published coordinates for both stations on every target day (D-32); a history day without them borrows the
    # first target day's position, which only sets elevations there, and history elevations are never counted
    coords_ok = {who: all(who in coords[d] for d in targets) for who in (station, partner)}
    jobs, keys = [], []
    for doy in days:
        for who in (station, partner):
            rx = coords[doy].get(who, coords[targets[0]].get(who))
            if rx is None:
                raise ValueError(f"no published coordinates for {who} on day {doy} or {targets[0]}")
            for hour in range(24):
                raw = raw_hour_path(who, year, doy, hour)
                if raw.exists():
                    jobs.append((raw, nav_path(year, doy), rx))
                    keys.append((who, doy))
                else:
                    counts[f"missing_hours.{who}"] = counts.get(f"missing_hours.{who}", 0) + 1
    inputs += [j[0] for j in jobs]
    with ProcessPoolExecutor(max_workers=workers()) as pool:
        reduced = reduce_files(jobs, pool)
    per: dict[str, dict[int, list[pa.Table]]] = {station: {}, partner: {}}
    for (who, doy), tab in zip(keys, reduced, strict=True):
        per[who].setdefault(doy, []).append(tab)
    tables = {who: {d: concat_reduced(ts) for d, ts in by.items()} for who, by in per.items()}
    epoch_s = epoch_interval_s(tables[station])
    own = flag_days(tables[station], targets, year, rule)
    other = flag_days(tables[partner], targets, year, rule)
    events = screen_partner(find_events(own, rule), other, rule)
    months = sorted({(day_date(year, d).year, day_date(year, d).month) for d in targets})
    kp = []
    for y, m in months:
        inputs.append(kp_path(y, m))
        kp += read_kp_json(kp_path(y, m))
    totals = g1c_totals(events, own, kp, epoch_s, g["min_clear_flagged_min"], g["kp_max"], g["depth_edges_db"])
    coordinates = {"in_every_target_day_file": coords_ok, "passed": all(coords_ok.values())}
    passed = totals["passed_events"] and coordinates["passed"]
    first = coords[targets[0]]
    distance_km = float(np.linalg.norm(first[station] - first[partner])) / 1e3
    rows = {who: int(sum(t.num_rows for t in tables[who].values())) for who in (station, partner)}
    return {
        "station": station, "partner": partner, "partner_km": distance_km, "year": year,
        "target_doys": [targets[0], targets[-1]], "history_doys": [days[0], targets[0] - 1],
        "hourly_files": len(jobs), "reduced_rows": rows, "epoch_s": epoch_s,
        "evaluated_epochs": {station: int(own.t_evaluated.size), partner: int(other.t_evaluated.size)},
        "flagged_epochs": {station: int(own.flagged.sum()), partner: int(other.flagged.sum())},
        "g1c": {"label": PROVISIONAL, "decision": "D-32", "passed": passed, "coordinates": coordinates,
                "kp_max_allowed": g["kp_max"], "depth_edges_db": g["depth_edges_db"], **totals},
    }  # fmt: skip


def run_e1c_scout(spec: dict[str, Any], run_dir: Path) -> Outcome:
    rule = ScoutRule.from_config(spec["rule"])
    if spec["screen"]["history_days"] != max(rule.lags):
        raise ValueError("screen.history_days must equal the largest sidereal lag")
    inputs: list[Path] = []
    counts: dict[str, int] = {}
    result: dict[str, Any] = {"experiment": "E1c-scout", "screen": _screen(spec, rule, inputs)}
    summary: dict[str, Any] = {"screen_pick": result["screen"]["pick"]}
    if spec["month"]["station"]:
        month = _month(spec, rule, inputs, counts)
        result["month"] = month
        g = month["g1c"]
        summary |= {"g1c_passed_provisional": g["passed"], "g1c_label": PROVISIONAL,
                    "admitted_flagged_min": g["admitted"]["flagged_min"],
                    "events_by_partner_state": g["events_by_partner_state"]}  # fmt: skip
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return Outcome(inputs=inputs, counts=counts, summary=summary)
