"""KA-orbit: broadcast-orbit geometry against closed-form orbits (docs/3 section 4, D-11).

The comparison with precise orbits on real data is in plan P1b (KA-dd, KA-orbit-sp3).
"""

from dataclasses import replace

import numpy as np
import pytest

from gnssaudit.geometry import (
    F_REL,
    GM,
    OMEGA_E,
    WGS84_A,
    WGS84_F,
    Ephemerides,
    KeplerEphemeris,
    ecef_to_geodetic,
    elev_azim_deg,
    geometric_range,
    predicted_doppler_hz,
    sat_pos_clock,
    sat_velocity,
)
from gnssaudit.signals import C_M_S

SQRT_A = 5153.7
WEEK = 2330
TOE_SOW = 345_600.0


def circular(**changes: float) -> KeplerEphemeris:
    """Circular equatorial orbit whose ascending node sits on the Greenwich meridian at toe."""
    base = KeplerEphemeris(
        sys="G", prn=1, toc_s=WEEK * 604_800.0 + TOE_SOW, af0=0.0, af1=0.0, af2=0.0, crs=0.0, delta_n=0.0, m0=0.0,
        cuc=0.0, e=0.0, cus=0.0, sqrt_a=SQRT_A, toe_sow=TOE_SOW, cic=0.0, omega0=OMEGA_E * TOE_SOW, cis=0.0, i0=0.0,
        crc=0.0, omega=0.0, omega_dot=0.0, idot=0.0, week=WEEK, health=0,
    )  # fmt: skip
    return replace(base, **changes)


def test_circular_orbit_position() -> None:
    eph = circular()
    a = SQRT_A**2
    n = np.sqrt(GM["G"] / a**3)
    for dt in (0.0, 900.0, -3600.0):
        pos, clock = sat_pos_clock(eph, eph.toe_s + dt)
        th = (n - OMEGA_E) * dt
        assert pos == pytest.approx([a * np.cos(th), a * np.sin(th), 0.0], abs=1e-6)
        assert clock == 0.0


def test_eccentric_orbit_radius_and_relativistic_clock() -> None:
    e, ecc_anomaly = 0.02, 1.0
    eph = circular(e=e, m0=ecc_anomaly - e * np.sin(ecc_anomaly), af0=1e-5, af1=2e-12)
    pos, clock = sat_pos_clock(eph, eph.toe_s + 0.0)
    assert np.linalg.norm(pos) == pytest.approx(SQRT_A**2 * (1 - e * np.cos(ecc_anomaly)), abs=1e-6)
    assert clock == pytest.approx(1e-5 + F_REL["G"] * e * SQRT_A * np.sin(ecc_anomaly), abs=1e-18)
    _, later = sat_pos_clock(replace(eph, e=0.0, m0=0.0), eph.toe_s + 100.0)
    assert later == pytest.approx(1e-5 + 2e-12 * 100.0, abs=1e-18)


def test_inclined_orbit_reaches_its_inclination() -> None:
    inc = np.radians(55.0)
    eph = circular(i0=inc, m0=np.pi / 2)  # argument of latitude 90 deg: the northernmost point
    pos, _ = sat_pos_clock(eph, eph.toe_s)
    assert np.arcsin(pos[2] / np.linalg.norm(pos)) == pytest.approx(inc, abs=1e-12)


def test_light_time_and_sagnac() -> None:
    eph = circular(i0=np.radians(55.0), m0=0.3)
    rx = np.array([WGS84_A, 0.0, 0.0])
    t_rx = eph.toe_s + 600.0
    rho, _ = geometric_range(eph, rx, t_rx)
    tau = rho / C_M_S
    pos, _ = sat_pos_clock(eph, t_rx - tau)
    th = OMEGA_E * tau
    rot = np.array([pos[0] * np.cos(th) + pos[1] * np.sin(th), -pos[0] * np.sin(th) + pos[1] * np.cos(th), pos[2]])
    assert np.linalg.norm(rot - rx) == pytest.approx(rho, abs=1e-6)
    sagnac = OMEGA_E / C_M_S * (pos[0] * rx[1] - pos[1] * rx[0])
    assert rho - np.linalg.norm(pos - rx) == pytest.approx(sagnac, abs=2e-3)


def geodetic_to_ecef(lat: float, lon: float, h: float) -> np.ndarray:
    e2 = WGS84_F * (2 - WGS84_F)
    n = WGS84_A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array(
        [(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon), (n * (1 - e2) + h) * np.sin(lat)]
    )


def test_geodetic_round_trip() -> None:
    lat, lon, h = np.radians(48.8566), np.radians(2.3522), 100.0
    out = ecef_to_geodetic(geodetic_to_ecef(lat, lon, h))
    assert out == pytest.approx((lat, lon, h), abs=1e-9)


def test_elevation_and_azimuth() -> None:
    rx = np.array([WGS84_A, 0.0, 0.0])
    assert elev_azim_deg(rx, rx + [2e7, 0, 0])[0] == pytest.approx(90.0)
    elev, azim = elev_azim_deg(rx, rx + [0, 0, 1e7])
    assert (elev, azim) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert elev_azim_deg(rx, rx + [0, 1e7, 0])[1] == pytest.approx(90.0)


def test_ephemeris_selection() -> None:
    near = circular(toe_sow=TOE_SOW)
    far = circular(toe_sow=TOE_SOW + 7200.0)
    sick = circular(toe_sow=TOE_SOW + 3600.0, health=1)
    fnav = replace(circular(toe_sow=TOE_SOW), sys="E", data_sources=0b010)
    eph = Ephemerides([far, near, sick, fnav])
    assert eph.select("G", 1, near.toe_s + 3000.0) is near
    assert eph.select("G", 1, near.toe_s + 4000.0) is far
    with pytest.raises(KeyError):
        eph.select("G", 1, near.toe_s - 8000.0)
    with pytest.raises(KeyError):
        eph.select("E", 1, near.toe_s)
    assert eph.satellites() == [("G", 1)]


def test_unhealthy_records_are_a_fallback_only_when_asked_for() -> None:
    """The sky check asks whether a satellite is in view, not whether it is usable: with include_unhealthy a healthy
    record is still preferred, and the nearest unhealthy one serves only where no healthy one is in range. The default
    never returns one."""
    healthy = circular(toe_sow=TOE_SOW)
    sick_near = circular(toe_sow=TOE_SOW + 3600.0, health=63)
    only_sick = circular(prn=2, toe_sow=TOE_SOW, health=63)
    eph = Ephemerides([healthy, sick_near, only_sick])
    t = healthy.toe_s + 3500.0  # the unhealthy record is the nearer one
    assert eph.select("G", 1, t) is healthy
    assert eph.select("G", 1, t, include_unhealthy=True) is healthy
    with pytest.raises(KeyError, match="healthy"):
        eph.select("G", 2, only_sick.toe_s)
    assert eph.select("G", 2, only_sick.toe_s + 60.0, include_unhealthy=True) is only_sick
    with pytest.raises(KeyError):
        eph.select("G", 2, only_sick.toe_s + 20_000.0, include_unhealthy=True)  # out of range either way
    assert eph.satellites() == [("G", 1)]


def test_predicted_doppler_is_positive_for_an_approaching_satellite() -> None:
    """GNSS-SDR's carrier Doppler is positive when the range shrinks: f = -(range rate) / wavelength, the range rate
    taken from an independent difference of the ECEF range, with a static receiver."""
    rx = np.array([WGS84_A, 0.0, 0.0])
    signs = []
    for m0 in (-0.9, 0.9):  # one rising towards the receiver's meridian, one past it
        eph = circular(m0=m0)
        t = eph.toe_s
        rate = np.linalg.norm(sat_pos_clock(eph, t + 0.5)[0] - rx) - np.linalg.norm(
            sat_pos_clock(eph, t - 0.5)[0] - rx
        )  # m/s over 1 s
        got = predicted_doppler_hz(eph, rx, t)
        assert got == pytest.approx(-rate * 1575.42e6 / C_M_S, rel=1e-3)
        assert abs(got) > 1000.0 and np.sign(got) == -np.sign(rate)  # shrinking range: positive Doppler
        signs.append(np.sign(got))
        assert np.linalg.norm(sat_velocity(eph, t)) == pytest.approx(
            np.sqrt(GM["G"] / SQRT_A**2) - OMEGA_E * SQRT_A**2, rel=1e-6
        )
    assert signs == [1.0, -1.0]  # the satellite before the meridian approaches, the one past it recedes
