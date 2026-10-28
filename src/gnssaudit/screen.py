"""Admission of clean Route B segments from the start of a recording (docs/7 section 4.2; P2b task 5).

A file can only be read from its start (Fairdata ignores HTTP Range), so a candidate segment is the file's first
seconds. Two kinds of evidence decide whether they are clean:
- change detectors find an onset inside the prefix: the spectrum moves away from its first seconds, or the C/N0 of
  several satellites steps together;
- absolute checks refuse a prefix that is disturbed from its first second, where nothing changes: a second
  correlation peak for a visible PRN (an incoherent spoofer), satellites all of one strength (a spoofer lays every PRN
  at one power, a real sky spreads them by elevation), too few strong satellites, or a position that moves or lies
  away from a published one.
A sky-consistency check (`sky_check`; P2b part 2, task 7) tests what a spoofer lacks: that the tracked satellites'
visibility, Doppler and C/N0-against-elevation follow the geometry of the broadcast ephemerides.
Thresholds are Judgment (ScreenRules, SkyRules) and are recorded in docs/7 section 4.2.
Pure functions, apart from reading the recording; `experiments.admit` runs them with a GNSS-SDR survey under
`gnssaudit run`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats

from gnssaudit.geometry import Ephemerides, elev_azim_deg, predicted_doppler_hz, sat_pos_clock
from gnssaudit.inject.source import IQSource, read_iq
from gnssaudit.sim.despread import ca_code

FloatArray = NDArray[np.float64]
SeriesByPrn = Mapping[int, tuple[FloatArray, FloatArray]]  # prn -> (t_s, values)
ACQ_FS_HZ = 2.046e6  # two samples per chip for the acquisition scan
ACQ_DETECT_RATIO = 6.0  # main peak over the grid median for a PRN to count as present
M_PER_DEG = 111_320.0


@dataclass(frozen=True)
class ScreenRules:
    psd_dev_db: float = 2.0  # largest band deviation from the prefix's first seconds
    second_peak_ratio: float = 0.5  # second correlation peak over the main one, > 2 chips away
    cn0_step_db: float = 2.0  # common-mode C/N0 step ...
    cn0_step_sats: int = 4  # ... on at least this many satellites at once
    min_sats: int = 6  # satellites at or above min_cn0_dbhz after the warm-up
    min_cn0_dbhz: float = 35.0
    min_cn0_spread_db: float = 6.0  # strongest minus weakest tracked satellite
    pvt_static_sd_m: float = 15.0  # horizontal SD of the fixes
    pvt_max_offset_m: float = 50.0  # from a published position, where there is one
    require_pvt: bool = True  # False for a band whose constellation gives no fix on its own (GPS L5: 2 satellites)
    min_len_s: float = 90.0
    target_len_s: float = 120.0
    warmup_s: float = 30.0
    margin_s: float = 1.0  # kept clear before an onset


@dataclass(frozen=True)
class SkyRules:  # Judgment, pre-registered in the plan of 16 Oct 2026 (commit 15ef977; docs/7 section 4.2)
    min_elev_deg: float = 5.0  # S1: every in-lock PRN at or above this elevation
    max_doppler_rms_hz: float = 10.0  # S2: RMS of measured minus predicted Doppler, common drift removed
    min_rho: float = 0.5  # S3: Spearman rho of C/N0 against elevation ...
    min_sats: int = 6  # ... with at least this many satellites


@dataclass(frozen=True)
class SkyCheck:
    elev_deg: dict[int, float]  # NaN for a PRN without a healthy ephemeris
    doppler_resid_hz: dict[int, float]  # measured - predicted - common drift
    clock_drift_hz: float  # the common mean of measured - predicted (the receiver clock drift)
    doppler_rms_hz: float
    spearman_rho: float
    spearman_p: float
    n_sats: int  # satellites with an ephemeris, the ones S2 and S3 use
    passed: dict[str, bool]  # S1 visibility, S2 doppler, S3 cn0_vs_elevation
    s1_healthy_only: bool  # reported only: S1 where a PRN with no healthy record fails (the pre-amendment reading)
    unhealthy_prns: list[int]  # PRNs whose record, and so elevation, came from an unhealthy broadcast record


@dataclass(frozen=True)
class AcqPeak:
    t_s: float
    prn: int
    code_chips: float
    doppler_hz: float
    peak_ratio: float  # main peak over the median of the search grid
    second_ratio: float  # highest peak more than 2 chips from the main one, over the main one
    second_over_median: float  # that second peak over the grid median: a spoofed copy is itself a detection


@dataclass(frozen=True)
class Admission:
    window_s: tuple[float, float] | None
    onset_s: float | None
    reasons: dict[str, list[float]] = field(default_factory=dict[str, list[float]])
    sats: dict[int, float] = field(default_factory=dict[int, float])  # median C/N0 after the warm-up
    cn0_spread_db: float = float("nan")
    pvt: dict[str, float] = field(default_factory=dict[str, float])


# ------------------------------------------------------------------------------------------ change detectors
def band_psd_db(x: NDArray[np.complexfloating], fs_hz: float, frontend_bw_hz: float, n_bands: int,
                nperseg: int = 4096) -> FloatArray:  # fmt: skip
    """Welch PSD (dB) averaged into n_bands equal bands across |f| <= frontend_bw_hz / 2."""
    f, p = signal.welch(x, fs=fs_hz, window="hann", nperseg=min(nperseg, x.size), return_onesided=False)
    f, p = np.fft.fftshift(f), np.fft.fftshift(p)
    edges = np.linspace(-frontend_bw_hz / 2, frontend_bw_hz / 2, n_bands + 1)
    idx = np.digitize(f, edges) - 1
    out = np.array([p[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bands)])
    return 10 * np.log10(out)


def psd_deviation_db(source: IQSource, *, frontend_bw_hz: float, ref_s: float = 10.0, step_s: float = 1.0,
                     n_bands: int = 64, seg_samples: int = 1 << 18) -> tuple[FloatArray, FloatArray]:  # fmt: skip
    """Per step: the largest |band PSD - reference| in dB, the reference being the median band PSD of the steps
    starting before ref_s. Each step is judged on its first seg_samples samples."""
    fs = source.fs_hz
    step = int(round(step_s * fs))
    starts = np.arange(source.sample_start, source.sample_end - min(seg_samples, step) + 1, step)
    bands = np.array([band_psd_db(read_iq(source, int(s), min(seg_samples, step)), fs, frontend_bw_hz, n_bands)
                      for s in starts])  # fmt: skip
    t = (starts - source.sample_start) / fs
    ref = np.nanmedian(bands[t < ref_s], axis=0)
    return t, np.nanmax(np.abs(bands - ref), axis=1)


def cn0_steps(cn0: SeriesByPrn, *, step_db: float, n_sats: int, window_s: float = 1.0, lag: int = 2) -> FloatArray:
    """Times at which at least n_sats satellites' C/N0 moved by step_db or more in the same direction: per-window
    medians compared `lag` windows apart (the receiver's C/N0 smoother spreads a step over about 0.5 s)."""
    t_max = max(float(t.max()) for t, _ in cn0.values())
    t_min = min(float(t.min()) for t, _ in cn0.values())
    edges = np.arange(t_min, t_max + window_s, window_s)
    rows = []
    for t, v in cn0.values():
        idx = np.digitize(t, edges) - 1
        rows.append([np.median(v[idx == w]) if np.any(idx == w) else np.nan for w in range(edges.size - 1)])
    med = np.array(rows)
    d = med[:, lag:] - med[:, :-lag]
    up = np.sum(d >= step_db, axis=0)
    down = np.sum(d <= -step_db, axis=0)
    hit = (up >= n_sats) | (down >= n_sats)
    return edges[lag:-1][hit] if edges.size > lag + 1 else np.array([])


# --------------------------------------------------------------------------------------------- absolute checks
def acquisition_scan(source: IQSource, prns: Sequence[int], *, step_s: float = 1.0, noncoh: int = 10,
                     doppler_max_hz: float = 6000.0, doppler_step_hz: float = 250.0,
                     doppler_min_hz: float | None = None) -> list[AcqPeak]:  # fmt: skip
    """GPS C/A parallel code-phase search every step_s on `noncoh` non-coherently summed 1 ms blocks, resampled to two
    samples per chip, over [doppler_min_hz, doppler_max_hz] (default +-doppler_max_hz).
    Returns, per step, every PRN whose main peak clears ACQ_DETECT_RATIO, with its second peak."""
    ratio = Fraction(ACQ_FS_HZ / source.fs_hz).limit_denominator(100_000)
    n_ms = int(ACQ_FS_HZ / 1000)  # 2046 samples per code period
    need = int(np.ceil((noncoh * n_ms + 64) / float(ratio)))
    lo = -doppler_max_hz if doppler_min_hz is None else doppler_min_hz  # symmetric unless a minimum is given
    dopplers = np.arange(lo, doppler_max_hz + doppler_step_hz / 2, doppler_step_hz)
    t_ms = np.arange(n_ms) / ACQ_FS_HZ
    codes = {p: np.conj(np.fft.fft(np.repeat(ca_code(p), 2))) for p in prns}
    wipe = np.exp(-2j * np.pi * dopplers[:, None] * t_ms[None, :])
    out: list[AcqPeak] = []
    step = int(round(step_s * source.fs_hz))
    for s in range(source.sample_start, source.sample_end - need + 1, step):
        x = signal.resample_poly(read_iq(source, s, need), ratio.numerator, ratio.denominator)
        blocks = x[: noncoh * n_ms].reshape(noncoh, n_ms)
        xf = np.fft.fft(blocks[:, None, :] * wipe[None, :, :], axis=2)  # (block, doppler, freq)
        for p in prns:
            grid = np.sum(np.abs(np.fft.ifft(xf * codes[p][None, None, :], axis=2)) ** 2, axis=0)
            d0, c0 = np.unravel_index(int(np.argmax(grid)), grid.shape)
            main = float(grid[d0, c0])
            ratio_main = main / float(np.median(grid))
            if ratio_main < ACQ_DETECT_RATIO:
                continue
            dist = np.abs((np.arange(n_ms) - c0 + n_ms // 2) % n_ms - n_ms // 2)  # samples, circular
            second = float(grid[:, dist > 4].max())  # more than 2 chips away, any Doppler
            out.append(AcqPeak(t_s=(s - source.sample_start) / source.fs_hz, prn=int(p), code_chips=c0 / 2.0,
                               doppler_hz=float(dopplers[d0]), peak_ratio=ratio_main, second_ratio=second / main,
                               second_over_median=second / float(np.median(grid))))  # fmt: skip
    return out


def _medians(cn0: SeriesByPrn, t_min_s: float, t_max_s: float) -> dict[int, float]:
    out = {}
    for p, (t, v) in cn0.items():
        sel = (t >= t_min_s) & (t < t_max_s)
        if np.any(sel):
            out[p] = float(np.median(v[sel]))
    return out


def cn0_spread_db(cn0: SeriesByPrn, *, t_min_s: float, t_max_s: float = np.inf) -> float:
    """Strongest minus weakest satellite, by median C/N0 over [t_min_s, t_max_s)."""
    med = list(_medians(cn0, t_min_s, t_max_s).values())
    return float(max(med) - min(med)) if med else float("nan")


def pvt_summary(positions: FloatArray, *, known: tuple[float, float, float] | None) -> dict[str, float]:
    """Fixes as (lon, lat, h) rows: their horizontal SD (m) and the horizontal distance of their median from a known
    (lat, lon, h) position, NaN without one."""
    pos = np.asarray(positions, dtype=np.float64)
    lon0, lat0 = np.median(pos[:, 0]), np.median(pos[:, 1])
    e = (pos[:, 0] - lon0) * M_PER_DEG * np.cos(np.radians(lat0))
    n = (pos[:, 1] - lat0) * M_PER_DEG
    out = {"n": float(pos.shape[0]), "sd_h_m": float(np.sqrt(np.var(e) + np.var(n))), "offset_m": float("nan")}
    if known is not None:
        de = (lon0 - known[1]) * M_PER_DEG * np.cos(np.radians(known[0]))
        dn = (lat0 - known[0]) * M_PER_DEG
        out["offset_m"] = float(np.hypot(de, dn))
    return out


# ------------------------------------------------------------------------------------------------- decision
def admit(*, duration_s: float, psd: tuple[FloatArray, FloatArray], peaks: Sequence[AcqPeak], steps: FloatArray,
          cn0: SeriesByPrn, pvt: Mapping[str, float], rules: ScreenRules) -> Admission:  # fmt: skip
    """The admitted window [0, end) of the prefix, or None, with every reason found (times in s from the start).

    The window starts at the prefix's start (the twin's warm-up is its first rules.warmup_s) and ends rules.margin_s
    before the first onset; it is admitted when at least rules.min_len_s long, and trimmed to rules.target_len_s.
    """
    reasons: dict[str, list[float]] = {}
    t_psd, dev = psd
    if np.any(dev > rules.psd_dev_db):
        reasons["psd_change"] = [float(x) for x in t_psd[dev > rules.psd_dev_db]]
    if np.size(steps):
        reasons["cn0_step"] = [float(x) for x in np.asarray(steps)]
    # a spoofed copy is itself a detection; noise maxima sit 3-4x the grid median (JT23_4.1.5, P2b task 6)
    spoof = sorted(p.t_s for p in peaks if p.second_ratio >= rules.second_peak_ratio
                   and p.second_over_median >= ACQ_DETECT_RATIO)  # fmt: skip
    if spoof:
        reasons["second_peak"] = spoof
    # satellites at or above the bar, second by second after the warm-up
    seconds = np.arange(rules.warmup_s, duration_s, 1.0)
    strong = np.zeros(seconds.size, dtype=int)
    for t, v in cn0.values():
        idx = np.digitize(t, seconds) - 1
        ok = np.array([np.any(idx == i) and np.median(v[idx == i]) >= rules.min_cn0_dbhz for i in range(seconds.size)])
        strong += ok
    if np.any(strong < rules.min_sats):
        reasons["few_sats"] = [float(x) for x in seconds[strong < rules.min_sats]]
    onsets = [v[0] for k, v in reasons.items() if k in ("psd_change", "cn0_step", "second_peak", "few_sats")]
    onset = min(onsets) if onsets else None
    # the absolute checks guard the window: satellites and their spread from the warm-up to the first onset, so a
    # spoofer's own PRNs after its arrival do not count (JT23_16.1.1, P2b task 6)
    until = onset if onset is not None else np.inf
    sats = _medians(cn0, rules.warmup_s, until)
    spread = cn0_spread_db(cn0, t_min_s=rules.warmup_s, t_max_s=until)
    whole: list[str] = []
    if not spread >= rules.min_cn0_spread_db:
        reasons["cn0_spread"] = [spread]
        whole.append("cn0_spread")
    if rules.require_pvt:
        if not pvt.get("n", 0.0) > 0:
            reasons["pvt_none"] = [0.0]  # no fix: no valid telemetry, so no observables for twin truth
            whole.append("pvt_none")
        elif not pvt.get("sd_h_m", np.inf) <= rules.pvt_static_sd_m:
            reasons["pvt_moving"] = [float(pvt.get("sd_h_m", np.nan))]
            whole.append("pvt_moving")
        if pvt.get("offset_m", np.nan) > rules.pvt_max_offset_m:
            reasons["pvt_offset"] = [float(pvt["offset_m"])]
            whole.append("pvt_offset")
    end = min(duration_s, onset - rules.margin_s) if onset is not None else duration_s
    window = None
    if not whole and end >= rules.min_len_s:
        window = (0.0, float(min(end, rules.target_len_s)))
    return Admission(window, None if onset is None else float(onset), reasons, sats, spread, dict(pvt))


# ------------------------------------------------------------------------------------------------ sky check
SKY_MAX_RECORDS = 200  # records per PRN the Doppler is compared at: evenly spread over the window's in-lock records
SkySeries = tuple[float, FloatArray, FloatArray]  # median C/N0 (dB-Hz), GPS time (s) and Doppler (Hz) of each record


def sky_check(sats: Mapping[int, SkySeries], eph: Ephemerides, rx_ecef: FloatArray, rules: SkyRules) -> SkyCheck:
    """Do the tracked satellites sit where the broadcast ephemerides put them, with the C/N0 and Doppler the sky gives?

    `sats` maps a GPS PRN to its median C/N0 (dB-Hz) over the window and to the GPS time (s) and Doppler (Hz) of its
    in-lock records in the window; `rx_ecef` is the static receiver. A spoofer lays PRNs at one power and Dopplers of
    its own, so it fails S2 or S3. Each PRN is judged at its own epochs (amendment of 19 Oct 2026): a satellite tracked
    only part of the window is compared where it was tracked, not at the window's centre.
    - S1: every PRN has elevation >= min_elev_deg, at the median time of its records. The elevation comes from the
      broadcast record whatever its health flag (a healthy one is preferred); a PRN with no record at all fails.
    - S2: the PRN's residual is the median over its records of measured - predicted Doppler (`geometry.
      predicted_doppler_hz` at the record's own time); after removing the common mean of the residuals (the receiver
      clock drift), the RMS is <= max_doppler_rms_hz.
    - S3: Spearman rho of C/N0 against elevation >= min_rho, over at least min_sats satellites.
    Also reported, not decisive: S1 with the unhealthy records refused, and the PRNs whose record was unhealthy.
    """
    elev: dict[int, float] = {}
    raw: dict[int, float] = {}
    cn0: dict[int, float] = {}
    unhealthy: list[int] = []
    for prn in sorted(sats):
        c, t_all, d_all = sats[prn]
        t_all, d_all = np.asarray(t_all, dtype=np.float64), np.asarray(d_all, dtype=np.float64)
        if t_all.size == 0:
            continue
        pick = np.unique(np.linspace(0, t_all.size - 1, min(t_all.size, SKY_MAX_RECORDS)).round().astype(int))
        t, d = t_all[pick], d_all[pick]
        t_mid = float(np.median(t_all))
        try:
            rec = eph.select("G", prn, t_mid, include_unhealthy=True)
        except KeyError:
            elev[prn] = float("nan")
            continue
        if rec.health != 0:
            unhealthy.append(prn)
        elev[prn] = elev_azim_deg(rx_ecef, sat_pos_clock(rec, t_mid)[0])[0]
        cn0[prn] = c
        raw[prn] = float(np.median(d - np.array([predicted_doppler_hz(rec, rx_ecef, float(ti)) for ti in t])))
    known = sorted(raw)
    drift = float(np.mean([raw[p] for p in known])) if known else float("nan")
    resid = {p: raw[p] - drift for p in known}
    rms = float(np.sqrt(np.mean([r**2 for r in resid.values()]))) if len(known) >= 2 else float("nan")
    x = np.array([elev[p] for p in known])
    y = np.array([cn0[p] for p in known])
    if len(known) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:  # a constant series has no rank correlation
        res: Any = stats.spearmanr(x, y)  # scipy's stubs type the result as an unnamed tuple
        rho, p_value = float(res.statistic), float(res.pvalue)
    else:
        rho = p_value = float("nan")
    passed = {
        "S1": bool(elev) and all(e >= rules.min_elev_deg for e in elev.values()),  # NaN fails
        "S2": bool(rms <= rules.max_doppler_rms_hz),
        "S3": bool(len(known) >= rules.min_sats and rho >= rules.min_rho),
    }
    healthy_only = bool(elev) and all(e >= rules.min_elev_deg and p not in unhealthy for p, e in elev.items())
    return SkyCheck(elev, resid, drift, rms, rho, p_value, len(known), passed, healthy_only, unhealthy)
