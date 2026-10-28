"""Broadcast-orbit geometry for GPS and Galileo (D-11): satellite position and clock, ranges, elevation.

Algorithms: IS-GPS-200 Table 20-IV (orbit) and section 20.3.3.3.3.1 (clock); the Galileo OS-ICD uses the same
Keplerian model with its own GM. Times are continuous GPS seconds since 1980-01-06. Group delays (TGD/BGD) are
omitted: they are constant and cancel in the time and double differences these functions serve (docs/3 section 4).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.io.schemas import OBS_EPOCH_SAT, validate
from gnssaudit.signals import C_M_S, carrier_hz

FloatArray = NDArray[np.float64]

WEEK_S = 604_800.0
OMEGA_E = 7.2921151467e-5  # rad/s, WGS-84 Earth rotation rate as used by GPS and Galileo
GM = {"G": 3.986005e14, "E": 3.986004418e14}  # m^3/s^2
F_REL = {"G": -4.442807633e-10, "E": -4.442807309e-10}  # s/m^(1/2)
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
MAX_AGE_S = {"G": 7_200.0, "E": 10_800.0}  # nearest-toe selection limit
GAL_INAV = 0b101  # data-source bits of I/NAV (E1-B, E5b-I); F/NAV records are not used


@dataclass(frozen=True)
class KeplerEphemeris:
    """One broadcast ephemeris record (RINEX 3 navigation field names)."""

    sys: str
    prn: int
    toc_s: float
    af0: float
    af1: float
    af2: float
    crs: float
    delta_n: float
    m0: float
    cuc: float
    e: float
    cus: float
    sqrt_a: float
    toe_sow: float
    cic: float
    omega0: float
    cis: float
    i0: float
    crc: float
    omega: float
    omega_dot: float
    idot: float
    week: int
    health: int
    data_sources: int = 0

    @property
    def toe_s(self) -> float:
        return self.week * WEEK_S + self.toe_sow


class Ephemerides:
    """Healthy records per satellite; `select` returns the record with the nearest toe. Records flagged unhealthy are
    kept apart: only `select(..., include_unhealthy=True)` falls back on them (the sky check asks whether a satellite is
    in view, which is geometry and does not depend on its health flag; P2b part 2, task 7)."""

    def __init__(self, records: Iterable[KeplerEphemeris]) -> None:
        by_sat: dict[tuple[str, int], list[KeplerEphemeris]] = defaultdict(list)
        unhealthy: dict[tuple[str, int], list[KeplerEphemeris]] = defaultdict(list)
        for r in records:
            if r.sys == "E" and not r.data_sources & GAL_INAV:
                continue
            (by_sat if r.health == 0 else unhealthy)[(r.sys, r.prn)].append(r)
        self._by_sat = {k: sorted(v, key=lambda r: r.toe_s) for k, v in by_sat.items()}
        self._unhealthy = {k: sorted(v, key=lambda r: r.toe_s) for k, v in unhealthy.items()}

    def satellites(self) -> list[tuple[str, int]]:
        return sorted(self._by_sat)

    @staticmethod
    def _nearest(records: list[KeplerEphemeris], sys: str, t_s: float) -> KeplerEphemeris | None:
        best = min(records, key=lambda r: abs(t_s - r.toe_s), default=None)
        return best if best is not None and abs(t_s - best.toe_s) <= MAX_AGE_S[sys] else None

    def select(self, sys: str, prn: int, t_s: float, include_unhealthy: bool = False) -> KeplerEphemeris:
        """Nearest-toe healthy record; KeyError if none lies within the system's validity limit. With
        include_unhealthy, a healthy record is still preferred and the nearest unhealthy one is the fallback."""
        best = self._nearest(self._by_sat.get((sys, prn), []), sys, t_s)
        if best is None and include_unhealthy:
            best = self._nearest(self._unhealthy.get((sys, prn), []), sys, t_s)
        if best is None:
            raise KeyError(f"no {'' if include_unhealthy else 'healthy '}ephemeris for {sys}{prn:02d} at t = {t_s}")
        return best


def sat_pos_clock(eph: KeplerEphemeris, t_s: float) -> tuple[FloatArray, float]:
    """ECEF position (m, Earth-fixed frame at time t) and clock offset (s, with the relativistic term) at GPS time t."""
    a = eph.sqrt_a**2
    n = np.sqrt(GM[eph.sys] / a**3) + eph.delta_n
    tk = t_s - eph.toe_s
    m = eph.m0 + n * tk
    ecc = m
    for _ in range(30):
        step = (ecc - eph.e * np.sin(ecc) - m) / (1.0 - eph.e * np.cos(ecc))
        ecc -= step
        if abs(step) < 1e-15:
            break
    nu = np.arctan2(np.sqrt(1.0 - eph.e**2) * np.sin(ecc), np.cos(ecc) - eph.e)
    phi = nu + eph.omega
    s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
    u = phi + eph.cus * s2 + eph.cuc * c2
    r = a * (1.0 - eph.e * np.cos(ecc)) + eph.crs * s2 + eph.crc * c2
    inc = eph.i0 + eph.cis * s2 + eph.cic * c2 + eph.idot * tk
    xp, yp = r * np.cos(u), r * np.sin(u)
    om = eph.omega0 + (eph.omega_dot - OMEGA_E) * tk - OMEGA_E * eph.toe_sow
    pos = np.array(
        [
            xp * np.cos(om) - yp * np.cos(inc) * np.sin(om),
            xp * np.sin(om) + yp * np.cos(inc) * np.cos(om),
            yp * np.sin(inc),
        ]
    )
    dt = t_s - eph.toc_s
    clock = eph.af0 + eph.af1 * dt + eph.af2 * dt**2 + F_REL[eph.sys] * eph.e * eph.sqrt_a * np.sin(ecc)
    return pos, float(clock)


def sat_velocity(eph: KeplerEphemeris, t_s: float, half_s: float = 0.5) -> FloatArray:
    """ECEF velocity (m/s) of the satellite by central difference of `sat_pos_clock` over t_s +- half_s."""
    ahead, _ = sat_pos_clock(eph, t_s + half_s)
    behind, _ = sat_pos_clock(eph, t_s - half_s)
    return (ahead - behind) / (2.0 * half_s)


def predicted_doppler_hz(eph: KeplerEphemeris, rx_ecef: FloatArray, t_s: float) -> float:
    """Carrier Doppler (Hz, L1) of a satellite at a static receiver, in GNSS-SDR's sign (positive when the range
    shrinks): f = -(v_sat . u) / wavelength, u the unit line of sight from the receiver. The satellite and receiver
    clock drifts are left out; the receiver's is common to all satellites."""
    pos, _ = sat_pos_clock(eph, t_s)
    los = pos - rx_ecef
    u = los / np.linalg.norm(los)
    return -float(np.dot(sat_velocity(eph, t_s), u)) * carrier_hz(eph.sys, "1C") / C_M_S


def geometric_range(eph: KeplerEphemeris, rx_ecef: FloatArray, t_rx_s: float) -> tuple[float, float]:
    """Range (m) from the transmit-time satellite position, rotated into the receive-time frame (Sagnac), and the
    satellite clock offset (s) at transmit time. t_rx_s is the receiver's time tag; its clock error is ignored."""
    tau = 0.075
    rho, clock = C_M_S * tau, 0.0
    for _ in range(4):
        pos, clock = sat_pos_clock(eph, t_rx_s - tau)
        th = OMEGA_E * tau
        rot = np.array([pos[0] * np.cos(th) + pos[1] * np.sin(th), -pos[0] * np.sin(th) + pos[1] * np.cos(th), pos[2]])
        rho = float(np.linalg.norm(rot - rx_ecef))
        tau = rho / C_M_S
    return rho, clock


def ecef_to_geodetic(x: FloatArray) -> tuple[float, float, float]:
    """WGS-84 latitude and longitude (rad) and ellipsoidal height (m)."""
    e2 = WGS84_F * (2.0 - WGS84_F)
    lon = float(np.arctan2(x[1], x[0]))
    p = float(np.hypot(x[0], x[1]))
    lat = float(np.arctan2(x[2], p * (1.0 - e2)))
    h = 0.0
    for _ in range(10):
        n = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = float(np.arctan2(x[2], p * (1.0 - e2 * n / (n + h))))
    return lat, lon, float(h)


def geodetic_to_ecef(lat_rad: float, lon_rad: float, h_m: float) -> FloatArray:
    """WGS-84 ECEF (m) of a latitude, longitude (rad) and ellipsoidal height (m)."""
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = WGS84_A / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
    return np.array(
        [
            (n + h_m) * np.cos(lat_rad) * np.cos(lon_rad),
            (n + h_m) * np.cos(lat_rad) * np.sin(lon_rad),
            (n * (1.0 - e2) + h_m) * np.sin(lat_rad),
        ]
    )


def elev_azim_deg(rx_ecef: FloatArray, sat_ecef: FloatArray) -> tuple[float, float]:
    lat, lon, _ = ecef_to_geodetic(rx_ecef)
    d = sat_ecef - rx_ecef
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    east = -so * d[0] + co * d[1]
    north = -sl * co * d[0] - sl * so * d[1] + cl * d[2]
    up = cl * co * d[0] + cl * so * d[1] + sl * d[2]
    elev = np.degrees(np.arctan2(up, np.hypot(east, north)))
    azim = np.degrees(np.arctan2(east, north)) % 360.0
    return float(elev), float(azim)


def annotate_elevation(obs: pa.Table, eph: Ephemerides, rx_ecef: FloatArray) -> pa.Table:
    """obs_epoch_sat with elev_deg and azim_deg filled; rows without a usable ephemeris stay null."""
    obs = validate(obs, OBS_EPOCH_SAT)
    t = obs.column("t_gps_s").to_numpy()
    sys = obs.column("sys").to_pylist()
    prn = obs.column("prn").to_numpy()
    elev = np.full(obs.num_rows, np.nan)
    azim = np.full(obs.num_rows, np.nan)
    for i in range(obs.num_rows):
        try:
            record = eph.select(sys[i], int(prn[i]), float(t[i]))
        except KeyError:
            continue
        pos, _ = sat_pos_clock(record, float(t[i]))
        elev[i], azim[i] = elev_azim_deg(rx_ecef, pos)
    for name, values in (("elev_deg", elev), ("azim_deg", azim)):
        idx = obs.schema.get_field_index(name)
        obs = obs.set_column(idx, name, pa.array(values, from_pandas=True).cast(pa.float32()))
    return obs
