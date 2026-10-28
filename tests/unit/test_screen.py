"""Clean-segment admission (P2b task 5; docs/7 section 4.2): change detectors find an onset inside a prefix, and
absolute checks refuse a prefix that is disturbed from its first second, where there is no change to detect."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gnssaudit.geometry import Ephemerides, predicted_doppler_hz
from gnssaudit.inject.source import IQSource
from gnssaudit.screen import (
    AcqPeak,
    ScreenRules,
    SkyRules,
    acquisition_scan,
    admit,
    cn0_spread_db,
    cn0_steps,
    psd_deviation_db,
    pvt_summary,
    sky_check,
)
from gnssaudit.sim.recording import SkySat, write_sky_recording
from tests.unit.sky_geometry import ARC, EPH, RX, SKYRULES, T0, orbit, sky_sats, tracked

RULES = ScreenRules()
RNG = np.random.default_rng(3)


def cn0_table(levels: list[float], *, drop_db: float = 0.0, n_drop: int = 0, t_drop: float = 40.0,
              seconds: float = 120.0) -> dict[int, tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """Per PRN, C/N0 at 1 kHz from the 5 s pull-in on; the first n_drop PRNs drop by drop_db at t_drop."""
    t = np.arange(5.0, seconds, 1e-3)
    out = {}
    for i, lv in enumerate(levels):
        v = lv + 0.2 * RNG.standard_normal(t.size)
        if i < n_drop:
            v = np.where(t >= t_drop, v - drop_db, v)
        out[i + 1] = (t, v)
    return out


SKY = [47.0, 46.0, 44.0, 43.0, 41.0, 40.0, 38.0, 37.0]  # a real sky: elevation spreads C/N0 over 10 dB


def quiet_psd(seconds: float = 150.0) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, seconds, 1.0)
    return t, 0.3 * np.abs(RNG.standard_normal(t.size))


def pvt_ok() -> dict[str, float]:
    return {"n": 1000, "sd_h_m": 3.0, "offset_m": float("nan")}


def test_a_common_drop_on_most_satellites_is_a_step() -> None:
    steps = cn0_steps(cn0_table(SKY, drop_db=3.0, n_drop=5), step_db=RULES.cn0_step_db, n_sats=RULES.cn0_step_sats)
    assert steps.size and 39.0 <= steps.min() <= 42.0


def test_a_drop_on_two_satellites_is_not_common_mode() -> None:
    steps = cn0_steps(cn0_table(SKY, drop_db=3.0, n_drop=2), step_db=RULES.cn0_step_db, n_sats=RULES.cn0_step_sats)
    assert steps.size == 0


def test_the_spread_of_cn0_across_satellites() -> None:
    assert cn0_spread_db(cn0_table(SKY), t_min_s=30.0) == pytest.approx(10.0, abs=0.1)


def test_pvt_summary_measures_spread_and_offset() -> None:
    lat0, lon0 = 60.182, 24.828
    m_per_deg = 111_320.0
    n = 1000
    pos = np.column_stack([
        lon0 + 3.0 * RNG.standard_normal(n) / (m_per_deg * np.cos(np.radians(lat0))),
        lat0 + 100.0 / m_per_deg + 3.0 * RNG.standard_normal(n) / m_per_deg,  # 100 m north of the reference
        47.0 + 5.0 * RNG.standard_normal(n),
    ])  # fmt: skip
    s = pvt_summary(pos, known=(lat0, lon0, 47.248))
    assert s["n"] == n and 3.5 <= s["sd_h_m"] <= 5.0  # horizontal: two 3 m components
    assert s["offset_m"] == pytest.approx(100.0, abs=2.0)
    drifting = pos.copy()
    drifting[:, 1] += np.arange(n) * 0.1 / m_per_deg  # 0.1 m per fix, 100 m in all
    assert pvt_summary(drifting, known=None)["sd_h_m"] > RULES.pvt_static_sd_m


def test_a_clean_prefix_is_admitted_and_trimmed_to_the_target_length() -> None:
    a = admit(duration_s=150.0, psd=quiet_psd(), peaks=[], steps=np.array([]), cn0=cn0_table(SKY, seconds=150.0),
              pvt=pvt_ok(), rules=RULES)  # fmt: skip
    assert a.window_s == (0.0, 120.0) and a.onset_s is None and not a.reasons


def test_an_onset_shortens_the_window_and_too_short_a_window_is_refused() -> None:
    t, dev = quiet_psd()
    late = admit(duration_s=150.0, psd=(t, np.where(t >= 110, 6.0, dev)), peaks=[], steps=np.array([]),
                 cn0=cn0_table(SKY, seconds=150.0), pvt=pvt_ok(), rules=RULES)  # fmt: skip
    assert late.onset_s == 110.0 and late.window_s == (0.0, 109.0) and "psd_change" in late.reasons
    early = admit(duration_s=150.0, psd=(t, np.where(t >= 70, 6.0, dev)), peaks=[], steps=np.array([]),
                  cn0=cn0_table(SKY, seconds=150.0), pvt=pvt_ok(), rules=RULES)  # fmt: skip
    assert early.onset_s == 70.0 and early.window_s is None


def test_a_spoofed_duplicate_present_from_the_start_is_refused() -> None:
    """Review Focus 1: no change to detect, but a second correlation peak for a visible PRN."""
    peaks = [AcqPeak(t_s=float(s), prn=7, code_chips=100.0, doppler_hz=1250.0, peak_ratio=30.0, second_ratio=0.8,
                     second_over_median=24.0) for s in range(150)]  # fmt: skip
    a = admit(duration_s=150.0, psd=quiet_psd(), peaks=peaks, steps=np.array([]), cn0=cn0_table(SKY, seconds=150.0),
              pvt=pvt_ok(), rules=RULES)  # fmt: skip
    assert a.window_s is None and a.onset_s == 0.0 and a.reasons["second_peak"][0] == 0.0


def test_satellites_of_equal_strength_are_refused() -> None:
    """A spoofer typically lays every PRN at one power; a real sky spreads them by elevation."""
    flat = cn0_table([45.0, 44.5, 44.0, 45.2, 44.8, 44.1, 45.1, 44.6], seconds=150.0)
    a = admit(duration_s=150.0, psd=quiet_psd(), peaks=[], steps=np.array([]), cn0=flat, pvt=pvt_ok(), rules=RULES)
    assert a.window_s is None and "cn0_spread" in a.reasons


def test_a_moving_or_displaced_position_is_refused() -> None:
    moving = admit(duration_s=150.0, psd=quiet_psd(), peaks=[], steps=np.array([]), cn0=cn0_table(SKY, seconds=150.0),
                   pvt={"n": 1000, "sd_h_m": 40.0, "offset_m": float("nan")}, rules=RULES)  # fmt: skip
    assert moving.window_s is None and "pvt_moving" in moving.reasons
    away = admit(duration_s=150.0, psd=quiet_psd(), peaks=[], steps=np.array([]), cn0=cn0_table(SKY, seconds=150.0),
                 pvt={"n": 1000, "sd_h_m": 3.0, "offset_m": 300.0}, rules=RULES)  # fmt: skip
    assert away.window_s is None and "pvt_offset" in away.reasons


def test_too_few_strong_satellites_after_the_warm_up_is_refused() -> None:
    weak = cn0_table([47.0, 45.0, 42.0, 39.0, 36.0, 33.0, 31.0, 30.0], seconds=150.0)  # 5 at or above 35 dB-Hz
    a = admit(duration_s=150.0, psd=quiet_psd(), peaks=[], steps=np.array([]), cn0=weak, pvt=pvt_ok(), rules=RULES)
    assert a.window_s is None and "few_sats" in a.reasons


FS_REC = 4.092e6  # two samples per acquisition sample: the scan resamples by exactly 1/2
SKY_SATS = [
    SkySat(prn=p, doppler_hz=d, cn0_dbhz=c, chip0=k)
    for p, d, c, k in ((3, 1200.0, 47.0, 100.0), (9, -2300.0, 44.0, 500.0), (17, 400.0, 41.0, 800.0))
]


def test_a_cw_onset_moves_the_spectrum_at_its_second(tmp_path: Path) -> None:
    path = tmp_path / "cw.bin"
    write_sky_recording(path, SKY_SATS, seconds=6.0, fs_hz=FS_REC, seed=5, cw=(3.0, 250e3, 75.0))
    src = IQSource(path=path, item_type="ibyte", fs_hz=FS_REC, sample_start=0, sample_end=int(6.0 * FS_REC))
    t, dev = psd_deviation_db(src, frontend_bw_hz=2.0e6, ref_s=2.0, seg_samples=1 << 16)
    assert dev[t < 3.0].max() < RULES.psd_dev_db <= dev[t >= 3.0].min()


def test_a_spoofed_duplicate_shows_as_a_second_peak_from_its_start(tmp_path: Path) -> None:
    path = tmp_path / "spoof.bin"
    dup = SkySat(prn=3, doppler_hz=1450.0, cn0_dbhz=47.0, chip0=400.0, start_s=2.0)  # same PRN, 300 chips away
    write_sky_recording(path, [*SKY_SATS, dup], seconds=4.0, fs_hz=FS_REC, seed=6)
    src = IQSource(path=path, item_type="ibyte", fs_hz=FS_REC, sample_start=0, sample_end=int(4.0 * FS_REC))
    peaks = acquisition_scan(src, [3, 9, 17, 25])
    by = {(p.t_s, p.prn): p for p in peaks}
    assert {prn for _, prn in by} == {3, 9, 17}  # PRN 25 is absent
    assert by[(0.0, 3)].doppler_hz == pytest.approx(1250.0, abs=250.0)
    assert by[(0.0, 3)].second_ratio < 0.2 and by[(1.0, 3)].second_ratio < 0.2
    assert by[(2.0, 3)].second_ratio >= RULES.second_peak_ratio and by[(3.0, 3)].second_ratio >= 0.5
    assert all(by[(t, 9)].second_ratio < 0.2 for t in (0.0, 1.0, 2.0, 3.0))


def test_a_survey_without_a_position_fix_is_refused() -> None:
    """No fix means no valid telemetry, hence no observables for twin truth."""
    a = admit(duration_s=150.0, psd=quiet_psd(), peaks=[], steps=np.array([]), cn0=cn0_table(SKY, seconds=150.0),
              pvt={"n": 0.0, "sd_h_m": float("nan"), "offset_m": float("nan")}, rules=RULES)  # fmt: skip
    assert a.window_s is None and "pvt_none" in a.reasons and "pvt_moving" not in a.reasons


def test_a_second_peak_at_noise_level_is_no_spoofer() -> None:
    """On JT23_4.1.5 (P2b task 6) the second peak always sat 3-4x above the grid median, the noise maximum; its ratio
    to a weak main peak (5-7x) passed 0.5. A spoofed copy is itself a detection: both conditions are required."""
    weak = [AcqPeak(t_s=float(s), prn=7, code_chips=1.0, doppler_hz=0.0, peak_ratio=7.1, second_ratio=0.52,
                    second_over_median=3.7) for s in range(150)]  # fmt: skip
    a = admit(duration_s=150.0, psd=quiet_psd(), peaks=weak, steps=np.array([]), cn0=cn0_table(SKY, seconds=150.0),
              pvt=pvt_ok(), rules=RULES)  # fmt: skip
    assert "second_peak" not in a.reasons and a.window_s == (0.0, 120.0)


def test_rules_for_a_partial_constellation_without_a_fix() -> None:
    """GPS L5 alone: two strong satellites in view, so no PVT fix; the L5 rules ask for 2 and no fix."""
    l5 = cn0_table([49.0, 40.0], seconds=150.0)
    rules = ScreenRules(min_sats=2, require_pvt=False)
    no_fix = {"n": 0.0, "sd_h_m": float("nan"), "offset_m": float("nan")}
    a = admit(duration_s=150.0, psd=quiet_psd(), peaks=[], steps=np.array([]), cn0=l5, pvt=no_fix, rules=rules)
    assert a.window_s == (0.0, 120.0) and not a.reasons


def test_satellites_and_their_spread_are_measured_before_the_first_onset() -> None:
    """P2b task 6 (JT23_16.1.1): a spoofer that arrives at 120 s adds its own PRN at 50 dB-Hz. Measured over the whole
    prefix, that PRN widened the spread past the bar and the window was admitted on it; measured over the window the
    absolute checks guard (from the warm-up to the first onset), the satellites stay those of the real sky."""
    flat = cn0_table([43.0, 42.5, 42.0, 41.0, 40.0, 39.5, 39.0, 38.5], seconds=150.0)  # a 4.5 dB spread
    t = np.arange(130.0, 150.0, 1e-3)
    spoofed = {**flat, 12: (t, 50.0 + 0.2 * RNG.standard_normal(t.size))}  # a PRN only the spoofer brings
    steps = np.array([120.0])
    a = admit(duration_s=150.0, psd=quiet_psd(), peaks=[], steps=steps, cn0=spoofed, pvt=pvt_ok(), rules=RULES)
    assert 12 not in a.sats and a.cn0_spread_db == pytest.approx(4.5, abs=0.2)
    assert a.window_s is None and "cn0_spread" in a.reasons
    wide = cn0_table(SKY, seconds=150.0)
    b = admit(duration_s=150.0, psd=quiet_psd(), peaks=[], steps=steps, cn0={**wide, 12: spoofed[12]}, pvt=pvt_ok(),
              rules=RULES)  # fmt: skip
    assert b.window_s == (0.0, 119.0) and 12 not in b.sats


# ------------------------------------------------------------------------------------------------ sky check
def test_a_sky_that_follows_the_geometry_passes_with_a_common_clock_drift() -> None:
    """A receiver clock drift shifts every Doppler alike; removing the common mean leaves the geometry (S2)."""
    chk = sky_check(sky_sats(drift_hz=1234.5), EPH, RX, SKYRULES)
    assert chk.n_sats == 8
    assert chk.clock_drift_hz == pytest.approx(1234.5, abs=1e-6)
    assert chk.doppler_rms_hz < 1e-6
    assert chk.passed == {"S1": True, "S2": True, "S3": True}
    assert chk.spearman_rho == pytest.approx(1.0) and chk.spearman_p < 0.01
    assert chk.elev_deg[1] < chk.elev_deg[4] and all(5.0 <= e <= 90.0 for e in chk.elev_deg.values())  # degrees


def test_the_doppler_sign_is_pinned_by_an_approaching_satellite() -> None:
    sats = sky_sats()
    assert (
        sats[1][2][0] > 1000.0 > -1000.0 > sats[8][2][0]
    )  # before the meridian it approaches (positive), past it recedes
    flipped = {p: (c, t, -d) for p, (c, t, d) in sats.items()}  # a sign error in the convention
    chk = sky_check(flipped, EPH, RX, SKYRULES)
    assert not chk.passed["S2"] and chk.doppler_rms_hz > 100.0


def test_thirty_hz_on_one_satellite_fails_the_doppler_rule() -> None:
    """The common mean absorbs 1/n of the error: 30 Hz on one of six leaves 30 sqrt(5) / 6 = 11.2 Hz RMS."""
    six = {p: v for p, v in sky_sats(drift_hz=500.0, error_hz={3: 30.0}).items() if p in (1, 3, 4, 5, 6, 7)}
    chk = sky_check(six, EPH, RX, SKYRULES)
    assert chk.n_sats == 6 and chk.doppler_rms_hz == pytest.approx(30.0 * np.sqrt(5) / 6, rel=1e-6)
    assert not chk.passed["S2"] and chk.passed["S1"] and chk.passed["S3"]
    ok = sky_check(sky_sats(error_hz={3: 8.0}), EPH, RX, SKYRULES)  # 8 Hz on one of eight stays under 10 Hz RMS
    assert ok.passed["S2"]


def test_a_satellite_below_five_degrees_fails_visibility() -> None:
    sats = sky_sats()
    low = orbit(9, 1.28)  # a few degrees above the horizon
    eph = Ephemerides([*(orbit(i + 1, m) for i, m in enumerate(ARC)), low])
    sats[9] = (31.0, np.array([T0]), np.array([predicted_doppler_hz(low, RX, T0)]))
    chk = sky_check(sats, eph, RX, SKYRULES)
    assert 0.0 < chk.elev_deg[9] < 5.0
    assert chk.passed["S1"] is False and chk.passed["S2"] is True
    below = Ephemerides([orbit(1, 2.5)])  # beneath the horizon
    assert sky_check({1: (40.0, np.array([T0]), np.array([0.0]))}, below, RX, SKYRULES).elev_deg[1] < 0.0


def test_a_satellite_with_no_broadcast_record_at_all_fails_visibility() -> None:
    sats = {**sky_sats(), 30: (40.0, np.array([T0]), np.array([0.0]))}
    chk = sky_check(sats, EPH, RX, SKYRULES)
    assert np.isnan(chk.elev_deg[30]) and chk.n_sats == 8 and chk.passed["S1"] is False


def test_cn0_rising_with_elevation_passes_and_equal_cn0_fails() -> None:
    assert sky_check(sky_sats(), EPH, RX, SKYRULES).passed["S3"]
    flat = sky_check(sky_sats(cn0="equal"), EPH, RX, SKYRULES)  # a spoofer lays every PRN at one power
    assert np.isnan(flat.spearman_rho) and flat.passed["S3"] is False
    falling = {p: (80.0 - c, t, d) for p, (c, t, d) in sky_sats().items()}
    assert sky_check(falling, EPH, RX, SKYRULES).spearman_rho == pytest.approx(-1.0)


def test_s3_needs_six_satellites() -> None:
    few = dict(list(sky_sats().items())[2:7])  # five satellites with a perfect rank correlation
    chk = sky_check(few, EPH, RX, SKYRULES)
    assert chk.n_sats == 5 and chk.spearman_rho == pytest.approx(1.0) and chk.passed["S3"] is False
    assert sky_check(few, EPH, RX, SkyRules(min_sats=5)).passed["S3"]


def test_an_unhealthy_flag_does_not_take_a_satellite_out_of_view() -> None:
    """Amendment of 19 Oct 2026 (Judgment): S1 asks whether a satellite is in view, which is geometry. A PRN whose only
    record is flagged unhealthy (G01 on 20 Sep 2023) is judged on that record; the old reading is reported beside S1."""
    sick = replace(orbit(9, 0.0), health=63)
    eph = Ephemerides([*(orbit(i + 1, m) for i, m in enumerate(ARC)), sick])
    sats = {**sky_sats(), 9: (46.0, np.array([T0]), np.array([predicted_doppler_hz(sick, RX, T0)]))}
    chk = sky_check(sats, eph, RX, SKYRULES)
    assert chk.unhealthy_prns == [9] and chk.elev_deg[9] > 80.0 and chk.n_sats == 9
    assert chk.passed == {"S1": True, "S2": True, "S3": True}
    assert chk.s1_healthy_only is False  # the pre-amendment reading refuses the PRN
    assert sky_check(sky_sats(), EPH, RX, SKYRULES).s1_healthy_only is True
    healthy_too = Ephemerides([sick, orbit(9, 0.0)])  # a healthy record is preferred when there is one
    assert sky_check({9: sats[9]}, healthy_too, RX, SKYRULES).unhealthy_prns == []


def test_a_satellite_tracked_only_at_the_end_of_the_window_is_judged_where_it_was_tracked() -> None:
    """Amendment of 19 Oct 2026 (a defect fix): the median Doppler of a PRN tracked for 5 s at the end of a 120 s window
    differs from the prediction at the window's centre by the Doppler change in between (here 0.23 Hz/s over 57 s).
    Compared at its own epochs, its residual is zero."""
    drift = 200.0
    sats = {p: tracked(p, -60.0, 60.0, drift) for p in (1, 2, 3, 5, 6, 7, 8)}
    sats[4] = tracked(4, 55.0, 60.0, drift)  # in lock only for the last 5 s
    centre = predicted_doppler_hz(orbit(4, ARC[3]), RX, T0)
    assert abs(float(np.median(sats[4][2])) - drift - centre) > 10.0  # the window-centre comparison would err by this
    chk = sky_check(sats, EPH, RX, SKYRULES)
    assert chk.n_sats == 8 and chk.doppler_rms_hz < 0.05 and chk.clock_drift_hz == pytest.approx(drift, abs=0.05)
    assert chk.doppler_resid_hz[4] == pytest.approx(0.0, abs=0.05)
    assert chk.passed == {"S1": True, "S2": True, "S3": True}


def test_the_sky_rules_are_the_pre_registered_ones() -> None:
    assert SkyRules() == SkyRules(min_elev_deg=5.0, max_doppler_rms_hz=10.0, min_rho=0.5, min_sats=6)


def test_the_scan_range_reaches_a_satellite_above_6_khz_only_when_asked(tmp_path: Path) -> None:
    path = tmp_path / "hi.bin"
    hi = SkySat(prn=3, doppler_hz=7000.0, cn0_dbhz=50.0, chip0=100.0)
    write_sky_recording(path, [hi], seconds=1.0, fs_hz=FS_REC, seed=8)
    src = IQSource(path=path, item_type="ibyte", fs_hz=FS_REC, sample_start=0, sample_end=int(1.0 * FS_REC))
    assert not any(p.prn == 3 and abs(p.doppler_hz - 7000.0) <= 250.0 for p in acquisition_scan(src, [3]))
    wide = acquisition_scan(src, [3], doppler_min_hz=-6000.0, doppler_max_hz=7500.0)
    assert any(p.prn == 3 and p.doppler_hz == pytest.approx(7000.0, abs=250.0) for p in wide)
