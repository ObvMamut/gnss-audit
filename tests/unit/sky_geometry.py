"""Constructed geometry for the sky check's tests: eight equatorial satellites along an arc seen from a receiver on the
equator at Greenwich, with C/N0 rising with elevation (shared by test_screen and test_admit_helpers)."""

import numpy as np

from gnssaudit.geometry import OMEGA_E, WGS84_A, Ephemerides, KeplerEphemeris, predicted_doppler_hz, sat_pos_clock
from gnssaudit.screen import SkyRules, SkySeries

TOE_SOW = 345_600.0
RX = np.array([WGS84_A, 0.0, 0.0])  # on the equator at Greenwich; the orbits below are equatorial with the node there


def orbit(prn: int, m0: float) -> KeplerEphemeris:
    return KeplerEphemeris(
        sys="G", prn=prn, toc_s=2330 * 604_800.0 + TOE_SOW, af0=0.0, af1=0.0, af2=0.0, crs=0.0, delta_n=0.0, m0=m0,
        cuc=0.0, e=0.0, cus=0.0, sqrt_a=5153.7, toe_sow=TOE_SOW, cic=0.0, omega0=OMEGA_E * TOE_SOW, cis=0.0, i0=0.0,
        crc=0.0, omega=0.0, omega_dot=0.0, idot=0.0, week=2330, health=0,
    )  # fmt: skip


ARC = [-1.0, -0.7, -0.4, -0.1, 0.2, 0.5, 0.8, 1.05]  # eight satellites along the arc; the ends are low
EPH = Ephemerides(orbit(i + 1, m) for i, m in enumerate(ARC))
T0 = orbit(1, 0.0).toe_s
SKYRULES = SkyRules()


def sky_sats(drift_hz: float = 0.0, error_hz: dict[int, float] | None = None,
             cn0: str = "elevation") -> dict[int, SkySeries]:  # fmt: skip
    """prn -> (C/N0, GPS times, Dopplers) of one record at T0: the predicted Doppler plus a common receiver drift, and
    C/N0 rising with elevation."""
    out = {}
    for i, m in enumerate(ARC):
        prn = i + 1
        pos, _ = sat_pos_clock(orbit(prn, m), T0)
        up = float(np.dot(pos - RX, RX) / np.linalg.norm(RX) / np.linalg.norm(pos - RX))  # sine of the elevation
        c = 30.0 + 18.0 * up if cn0 == "elevation" else 42.0
        d = predicted_doppler_hz(orbit(prn, m), RX, T0) + drift_hz + (error_hz or {}).get(prn, 0.0)
        out[prn] = (c, np.array([T0]), np.array([d]))
    return out


def tracked(prn: int, t_from: float, t_to: float, drift_hz: float = 0.0) -> SkySeries:
    """A PRN's records at 1 Hz over [t_from, t_to): the Doppler its orbit gives at each instant plus a drift."""
    m = ARC[prn - 1]
    t = T0 + np.arange(t_from, t_to, 1.0)
    d = np.array([predicted_doppler_hz(orbit(prn, m), RX, float(x)) for x in t]) + drift_hz
    pos, _ = sat_pos_clock(orbit(prn, m), T0)
    up = float(np.dot(pos - RX, RX) / np.linalg.norm(RX) / np.linalg.norm(pos - RX))
    return 30.0 + 18.0 * up, t, d
