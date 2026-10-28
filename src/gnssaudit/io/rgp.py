"""IGN RGP reference-station access and partner pairing for arm c (docs/7 section 2.2).

Arm c scores jamming events seen at a French reference station against a partner station 5 to 30 km away that
saw no simultaneous drop. This module holds only the pure helpers: the URL of an hourly 1 s file, the reader of
the daily coordinate file, the partner search, the metropolitan-France test and the reduction of an hourly file to
the GPS 1C C/N0 and elevation the detector reads. Downloading is done by the caller through `io.fetch`.

The daily coordinate file `coorrgf93_YYYY-DDD.txt` sits in the day directory, beside `data_1/` (it is not in
the directory listing, but the URL answers). It is Latin-1 text: `#` comment lines, a two-line column header, a
dashed rule, then one fixed-width row per station: site (4 characters, upper case), DOMES, X, Y, Z in metres
(Cartesian, already ECEF) and a one-letter annual stability status. It covers the whole network, so overseas
stations appear in their own legal frames (RGAF09, RGFG95, RGR92, ...); only pairs within 30 km are used, so a
mixed-frame pair cannot arise. RGF93 differs from ITRF by under 1 m, and D-30 showed that 1 m is immaterial over
Delta = 1 s. Distance is the Euclidean ECEF chord, which is below the arc by about 3 cm at 30 km.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyarrow as pa

from gnssaudit.geometry import Ephemerides, FloatArray, annotate_elevation, ecef_to_geodetic
from gnssaudit.io.hatanaka import crx_to_rnx, decompress_z
from gnssaudit.io.rinex import read_rinex2_obs

BASE_URL = "https://rgpdata.ign.fr/pub/data"
_ROW = re.compile(r"^\s*(\S{4})\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\S)\s*$")
_TEXT = re.compile(r"^\s*(#|SITE\b|\(m\)|-{5,})")


def hourly_url(station: str, year: int, doy: int, hour: int) -> str:
    """URL of the hourly 1 s Hatanaka file of a station, e.g. `.../2024/253/data_1/agde253a.24d.Z`."""
    if len(station) != 4:
        raise ValueError(f"station must be 4 characters, got {station!r}")
    if not 0 <= hour <= 23:
        raise ValueError(f"hour must be 0 to 23, got {hour}")
    name = f"{station.lower()}{doy:03d}{chr(ord('a') + hour)}.{year % 100:02d}d.Z"
    return f"{BASE_URL}/{year}/{doy:03d}/data_1/{name}"


def coords_url(year: int, doy: int) -> str:
    """URL of the daily station-coordinate file, which sits in the day directory."""
    return f"{BASE_URL}/{year}/{doy:03d}/coorrgf93_{year}-{doy:03d}.txt"


def read_rgf93_coords(path: Path) -> dict[str, FloatArray]:
    """Station (lower case, as in the data file names) to ECEF position (m) from a `coorrgf93_*.txt` file."""
    coords: dict[str, FloatArray] = {}
    for n, line in enumerate(path.read_text(encoding="latin-1").splitlines(), 1):
        if not line.strip() or _TEXT.match(line):
            continue
        row = _ROW.match(line)
        if row is None:
            raise ValueError(f"{path.name} line {n}: not a station row: {line!r}")
        coords[row[1].lower()] = np.array([float(row[3]), float(row[4]), float(row[5])])
    if not coords:
        raise ValueError(f"{path.name}: no station rows")
    return coords


def partners(
    station: str, coords: dict[str, FloatArray], min_m: float = 5e3, max_m: float = 30e3
) -> list[tuple[str, float]]:
    """Stations whose ECEF chord distance to `station` lies in [min_m, max_m], nearest first."""
    here = coords[station.lower()]
    found = [(s, float(np.linalg.norm(x - here))) for s, x in coords.items() if s != station.lower()]
    return sorted(((s, d) for s, d in found if min_m <= d <= max_m), key=lambda sd: sd[1])


# Metropolitan France (mainland and Corsica): geodetic latitude and longitude window, degrees. Every station of the
# coordinate file inside it has a French DOMES number (1xxxx); overseas stations (97xxx, 90xxx, ...) lie outside.
METRO_LAT_DEG = (41.0, 51.5)
METRO_LON_DEG = (-5.5, 10.0)


def is_metropolitan(x: FloatArray) -> bool:
    """ECEF position inside the metropolitan-France latitude and longitude window."""
    lat, lon, _ = ecef_to_geodetic(x)
    lat_d, lon_d = float(np.degrees(lat)), float(np.degrees(lon))
    return METRO_LAT_DEG[0] <= lat_d <= METRO_LAT_DEG[1] and METRO_LON_DEG[0] <= lon_d <= METRO_LON_DEG[1]


# A reduced station table: the GPS 1C (S1) C/N0 and elevation the arm c detector reads, one row per satellite-epoch.
REDUCED_SCHEMA = pa.schema(
    [("t_gps_s", pa.float64()), ("prn", pa.int16()), ("cn0_dbhz", pa.float32()), ("elev_deg", pa.float32())]
)


def reduce_obs(obs: pa.Table, eph: Ephemerides, rx_ecef: FloatArray) -> pa.Table:
    """GPS 1C rows with a C/N0 of an obs_epoch_sat table, elevation filled from the broadcast ephemerides (NaN where
    none is valid), sorted by time then PRN, duplicate (time, PRN) rows dropped."""
    sys = np.asarray(obs["sys"].to_numpy(zero_copy_only=False))
    sig = np.asarray(obs["signal"].to_numpy(zero_copy_only=False))
    cn0 = np.asarray(obs["cn0_dbhz"].to_numpy(zero_copy_only=False), dtype=np.float64)
    keep = (sys == "G") & (sig == "1C") & np.isfinite(cn0)
    g = annotate_elevation(obs.filter(pa.array(keep)), eph, rx_ecef)
    t, prn = g["t_gps_s"].to_numpy(), g["prn"].to_numpy()
    order = np.lexsort((prn, t))
    t, prn = t[order], prn[order]
    first = np.ones(t.size, dtype=bool)
    first[1:] = (np.diff(t) != 0) | (np.diff(prn) != 0)
    idx = order[first]
    elev = g["elev_deg"].to_numpy(zero_copy_only=False)
    return pa.table(
        {
            "t_gps_s": pa.array(t[first], pa.float64()),
            "prn": pa.array(prn[first], pa.int16()),
            "cn0_dbhz": pa.array(g["cn0_dbhz"].to_numpy(zero_copy_only=False)[idx], pa.float32()),
            "elev_deg": pa.array(np.asarray(elev, dtype=np.float32)[idx], pa.float32()),
        },
        schema=REDUCED_SCHEMA,
    )


def reduce_hourly(path: Path, eph: Ephemerides, rx_ecef: FloatArray) -> pa.Table:
    """One hourly `.d.Z` file to its reduced table: expand, decode Hatanaka, read RINEX 2.11, keep GPS 1C."""
    text = crx_to_rnx(decompress_z(path).decode("ascii"))
    return reduce_obs(read_rinex2_obs(text, source_id=path.name), eph, rx_ecef)


def detector_table(reduced: pa.Table) -> pa.Table:
    """A reduced table with the constant sys and signal columns the detector (`e1c_scout.flag_epochs`) selects on."""
    n = reduced.num_rows
    return reduced.append_column("sys", pa.array(["G"] * n, pa.string())).append_column(
        "signal", pa.array(["1C"] * n, pa.string())
    )
