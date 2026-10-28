"""Admit: is the start of a recording a clean Route B segment? (docs/7 section 4.2; P2b task 5)

The prefix of a published file (P2b task 4) is screened with `gnssaudit.screen`:
- the spectrum, second by second, against its first seconds;
- a C/A acquisition scan for second correlation peaks (spoofed duplicates);
- a survey: a zero-gain twin of the prefix, decimated as the twins will be, through the patched receiver with its
  stock settings on free channels with PVT on, which gives each satellite's C/N0 over time (records in lock only) and
  the position fixes.
- a sky check (P2b part 2, task 7; docs/7 section 4.2) for the windows a spec names: do the in-lock satellites' C/N0
  and Doppler follow the geometry the broadcast ephemerides give, at the survey's position and on GPS time read from
  the survey's observables?
- for a real-IF recording (P2b part 2, task 8; a spec's `source.real_if`): the prefix is first converted to complex
  baseband in /dev/shm (`inject.realif`), screened from there and deleted, and the survey reads the converted file;
  a spec's `power_windows` add the 1 ms in-band power in the windows it names.

The summary carries the decision, every reason with its times, the noise floor (with its line excess, which V6 and V7
are judged against), and a proposed `registry/intervals.yaml` row that is committed by hand. The prefix may still be
`.unverified` while the rest of its file is being hashed: it is then not hashed into the manifest, and the summary
says so; the registry row is committed only once the file is verified. The sky check's navigation file is hashed into
the manifest too.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from gnssaudit.experiments import Outcome
from gnssaudit.experiments.twin import SHM_MARGIN, SHM_ROOT, WARMUP_LOCK_MIN, keep_logs, provenance, receiver_spec
from gnssaudit.geometry import WEEK_S, Ephemerides, FloatArray, geodetic_to_ecef
from gnssaudit.inject import FrontEndSpec, IQSource, SinkSpec, estimate_n0, inject, iter_iq
from gnssaudit.inject.realif import to_baseband
from gnssaudit.io.fairdata import require_free
from gnssaudit.io.gnsssdr import read_observables_dump, read_pvt_geojson, read_pvt_nmea, read_tracking_dump
from gnssaudit.io.rinex import gps_seconds, read_rinex_nav
from gnssaudit.paths import data_root, results_root
from gnssaudit.screen import (
    M_PER_DEG,
    Admission,
    ScreenRules,
    SeriesByPrn,
    SkyRules,
    SkySeries,
    acquisition_scan,
    admit,
    cn0_steps,
    psd_deviation_db,
    pvt_summary,
    sky_check,
)
from gnssaudit.sdr import SourceSpec, render_config, run_receiver
from gnssaudit.sdr.lock import SETTLE_S, in_lock
from gnssaudit.truth.rxtime import ObsClock, epoch_samples

ITEM_BYTES = {"ibyte": 2, "ishort": 4}  # per complex sample
MAX_LISTED = 20  # reason times kept in the summary (the count is kept too)
OBS_INTERVAL_S = 0.02  # GNSS-SDR.observable_interval_ms in the templates
OBS_K0 = 1  # the first observables epoch is the sample counter's first tick (truth.rxtime, verified on v0.0.22)
SURVEY_PEAK = 16384  # int16 counts the prefix's largest component maps to in the survey twin (half of full scale)


def survey_scale(source: IQSource) -> float:
    """int16 counts per input count for the survey twin: the prefix's largest I or Q value maps to SURVEY_PEAK, so a
    zero-gain copy cannot clip whatever the recording holds (lines, a start-up transient)."""
    peak = max(float(np.max(np.abs(np.concatenate([x.real, x.imag])))) for _, x in iter_iq(source, 1 << 22))
    return SURVEY_PEAK / peak


def resolve_prefix(root: Path, local_path: str) -> tuple[Path, bool]:
    """The prefix file and whether its whole file is verified: the final name if present, else `<name>.unverified`
    (io.fairdata gives the final name only to a prefix whose whole file it verified)."""
    final = root / local_path
    if final.exists():
        return final, True
    pending = final.with_name(final.name + ".unverified")
    if pending.exists():
        return pending, False
    raise FileNotFoundError(f"neither {final} nor {pending.name} exists")


def power_1ms_db(source: IQSource, windows: Sequence[Mapping[str, Any]], baseline_s: Sequence[float]
                ) -> dict[str, Any]:  # fmt: skip
    """The in-band power (mean |x|^2 of the converted baseband, whose FIR is the band) of consecutive 1 ms blocks in dB.

    D2's number (D-38, amendment of 19 Oct 2026): per window {name, t_start_s, t_end_s}, the minimum 1 ms power relative
    to the median over `baseline_s` (post-warm-up, pre-window); the window is a dip if that is -1 dB or lower. Reported
    with it: the window's median, when the minimum is (s) and how many 1 ms blocks are 1 dB or more below the baseline.
    """
    n_ms = round(source.fs_hz * 1e-3)
    total = (source.sample_end - source.sample_start) // n_ms
    p = np.empty(total)
    per_read = 1000  # ms per read
    for start, x in iter_iq(replace(source, sample_end=source.sample_start + total * n_ms), n_ms * per_read):
        k = (start - source.sample_start) // n_ms
        blocks = x.reshape(-1, n_ms)
        p[k : k + blocks.shape[0]] = np.mean(blocks.real.astype(np.float64) ** 2 + blocks.imag.astype(np.float64) ** 2,
                                             axis=1)  # fmt: skip
    db = 10.0 * np.log10(p)
    b0, b1 = round(float(baseline_s[0]) * 1000), round(float(baseline_s[1]) * 1000)
    base = float(np.median(db[b0:b1]))
    out: dict[str, Any] = {}
    for w in windows:
        t0, t1 = float(w["t_start_s"]), float(w["t_end_s"])
        lo, hi = round(t0 * 1000), min(round(t1 * 1000), total)
        rel = db[lo:hi] - base
        out[str(w["name"])] = {
            "t_start_s": t0,
            "t_end_s": t1,
            "min_rel_db": float(rel.min()),
            "dip": bool(rel.min() <= -1.0),
            "median_rel_db": float(np.median(rel)),
            "t_min_s": float((lo + int(np.argmin(rel))) * 1e-3),
            "n_ms": int(rel.size),
            "n_ms_below_1db": int(np.count_nonzero(rel <= -1.0)),
        }
    return {
        "baseline_s": [float(baseline_s[0]), float(baseline_s[1])],
        "baseline_db": base,
        "n_ms": total,
        "windows": out,
        "note": "2-bit quantisation compresses RF power changes (about 0.6 dB of this power per dB of RF)",
    }


def check_windows(spec: Mapping[str, Any], duration_s: float) -> None:
    """The power windows, their baseline and the onset windows lie inside the prefix (checked before any conversion)."""
    spans = [(str(w["name"]), float(w["t_start_s"]), float(w["t_end_s"])) for w in spec.get("power_windows", [])]
    spans += [(f"onset_windows[{i}]", float(lo), float(hi)) for i, (lo, hi) in enumerate(spec.get("onset_windows", []))]
    if "power_windows" in spec:
        spans.append(("power_baseline_s", float(spec["power_baseline_s"][0]), float(spec["power_baseline_s"][1])))
    for name, t0, t1 in spans:
        if not 0.0 <= t0 < t1 <= duration_s:
            raise ValueError(f"{name} [{t0:g}, {t1:g}] s is not inside the {duration_s:g} s prefix")


def _lock_series(tracks: Sequence[pa.Table], column: str) -> SeriesByPrn:
    parts: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for tr in tracks:
        prn = tr.column("prn").to_numpy()
        t = tr.column("t_rx_s").to_numpy()
        c = tr.column(column).to_numpy(zero_copy_only=False).astype(np.float64)
        locked = in_lock(tr, min_lock=WARMUP_LOCK_MIN, settle_s=SETTLE_S)
        for p in np.unique(prn[(prn > 0) & locked]):
            m = (prn == p) & locked
            parts[int(p)].append((t[m], c[m]))
    out = {}
    for p, chunks in parts.items():
        t = np.concatenate([a for a, _ in chunks])
        c = np.concatenate([b for _, b in chunks])
        order = np.argsort(t, kind="stable")
        out[p] = (t[order], c[order])
    return out


def cn0_by_prn(tracks: Sequence[pa.Table]) -> SeriesByPrn:
    """Each PRN's C/N0 over receiver time, gathered across the free channels that tracked it. Idle records (PRN 0) are
    dropped, and so are records not settled in lock (`sdr.lock.in_lock`): a noise track (about 29 dB-Hz, lock
    statistic near 0) is no satellite, nor is the opening of a tracking episode (P2b task 6)."""
    return _lock_series(tracks, "cn0_dbhz")


def dop_by_prn(tracks: Sequence[pa.Table]) -> SeriesByPrn:
    """Each PRN's carrier Doppler (Hz, positive for an approaching satellite) over receiver time, on the same in-lock
    records as `cn0_by_prn`."""
    return _lock_series(tracks, "dop_hz")


def gps_time_map(obs: pa.Table, *, n_channels: int, fs_hz: float, interval_s: float, day_start_s: float
                 ) -> Callable[[Any], Any]:  # fmt: skip
    """Recording time (s from the first sample, at the survey's sample rate fs_hz) to GPS seconds (a float or an array).

    The observables epoch i sits at sample (i + k0) x step of the sample-counter clock (`truth.rxtime`; docs/9 section
    10) and carries the receiver time (TOW) of that sample, so a line through (sample time, TOW) over the valid epochs
    maps one onto the other; its slope is the receiver clock's rate. The week comes from `day_start_s` (GPS seconds at
    0 h of the recording's day, from the spec), and the TOW must put the recording inside that day.
    """
    slope, intercept = receiver_time_line(obs, n_channels=n_channels, fs_hz=fs_hz, interval_s=interval_s)
    week_start = np.floor(day_start_s / WEEK_S) * WEEK_S
    if not day_start_s <= week_start + intercept < day_start_s + 86_400.0:
        raise ValueError(f"the receiver time of the first sample (TOW {intercept:.1f} s) is not in the spec's day")
    return lambda t_s: week_start + intercept + slope * np.asarray(t_s, dtype=np.float64)  # a float or an array


def receiver_time_line(obs: pa.Table, *, n_channels: int, fs_hz: float, interval_s: float) -> tuple[float, float]:
    """(slope, intercept) of the receiver time (TOW, s) against recording time (s from the first sample): a line through
    the valid observables epochs, each at sample (i + k0) x step of the sample-counter clock (`truth.rxtime`)."""
    clock = ObsClock(k0=OBS_K0, step=round(fs_hz * interval_s))
    if obs.num_rows % n_channels:
        raise ValueError(f"{obs.num_rows} observables rows are not whole epochs of {n_channels} channels")
    epoch = np.arange(obs.num_rows) // n_channels
    tow = obs.column("t_gps_s").to_numpy(zero_copy_only=False).astype(np.float64)
    ok = obs.column("valid").to_numpy(zero_copy_only=False).astype(bool) & np.isfinite(tow)
    if not ok.any():
        raise ValueError("no valid observables epoch: no receiver time to map the recording onto")
    idx, first = np.unique(epoch[ok], return_index=True)
    t_rec = epoch_samples(idx, clock) / fs_hz
    slope, intercept = np.polyfit(t_rec, tow[ok][first], 1)
    if abs(slope - 1.0) > 1e-3:
        raise ValueError(f"receiver time runs at {slope:.6f} s per recording second: not a sample-counter clock")
    return float(slope), float(intercept)


class FixTimesError(ValueError):
    """The survey's fixes cannot be put on the recording's time axis (the PVT clause of D1 is then not observable)."""


GPS_UTC_S = 18.0  # GPS time less UTC since 1 Jan 2017; RTKLIB's NMEA time is UTC


def timed_fixes(nmea: Sequence[Path], obs: pa.Table | None, *, n_channels: int, fs_hz: float, duration_s: float
                ) -> tuple[FloatArray, FloatArray, FloatArray]:  # fmt: skip
    """The survey's fixes as (recording time in s, latitude, longitude in degrees).

    GNSS-SDR's own NMEA log gives each fix's UTC time of day; plus the 18 s GPS-UTC offset it is the receiver time (TOW)
    of the fix, and the line TOW(recording time) of the survey's observables (`receiver_time_line`, as the sky check
    uses) puts it on the recording's axis. The day number is not needed: the time of day is taken to the line's nearest
    day. Times outside [-2, duration + 2] s mean the mapping is wrong and raise FixTimesError."""
    if not nmea:
        raise FixTimesError("the survey wrote no NMEA log")
    if obs is None:
        raise FixTimesError("the survey wrote no observables, so the fixes have no time axis")
    tod, lat, lon = read_pvt_nmea(nmea[0])
    if not tod.size:
        raise FixTimesError("the NMEA log holds no fix")
    try:
        slope, intercept = receiver_time_line(obs, n_channels=n_channels, fs_hz=fs_hz, interval_s=OBS_INTERVAL_S)
    except ValueError as exc:
        raise FixTimesError(str(exc)) from exc
    delta = (tod + GPS_UTC_S - intercept + 43_200.0) % 86_400.0 - 43_200.0
    t = delta / slope
    if t.min() < -2.0 or t.max() > duration_s + 2.0:
        raise FixTimesError(f"fix times {t.min():.1f} to {t.max():.1f} s fall outside the {duration_s:g} s prefix")
    return t, lat, lon


def _dist_m(lat: float, lon: float, lat0: float, lon0: float) -> float:
    """Horizontal distance (m) between two points a few hundred metres apart (the screen's own flat-earth metric)."""
    return float(np.hypot((lon - lon0) * M_PER_DEG * np.cos(np.radians(lat0)), (lat - lat0) * M_PER_DEG))


def onset_blocks(windows: Sequence[Sequence[float]], reasons: Mapping[str, Sequence[float]],
                 fixes: tuple[FloatArray, FloatArray, FloatArray] | str, *, pre_s: tuple[float, float],
                 max_offset_m: float, known: Sequence[float] | None) -> list[dict[str, Any]]:  # fmt: skip
    """D1's evidence per window [lo, hi] (closed): every psd_change, cn0_step and second_peak time inside it, and the
    PVT clause. The PVT sign (D-38, amendment of 19 Oct 2026) is that the median of the window's timed fixes lies more
    than `max_offset_m` from the median of the timed fixes in `pre_s`; the window's median against the record's known
    position (lat, lon, h) is reported beside it and is not decisive. `fixes` is a string (the reason) when the fixes
    could not be timed: the clause is then `not observable`."""
    out = []
    for lo, hi in windows:
        block: dict[str, Any] = {"t_start_s": float(lo), "t_end_s": float(hi)}
        for kind in ("psd_change", "cn0_step", "second_peak"):
            block[kind] = [float(t) for t in reasons.get(kind, []) if lo <= t <= hi]
        if isinstance(fixes, str):
            block["pvt_offset"] = {"observable": False, "reason": fixes, "sign": None}
        else:
            t, lat, lon = fixes
            w = (t >= lo) & (t <= hi)
            pre = (t >= pre_s[0]) & (t < pre_s[1])
            pv: dict[str, Any] = {"observable": bool(w.any() and pre.any()), "n_window_fixes": int(w.sum()),
                                  "n_pre_fixes": int(pre.sum()), "max_offset_m": max_offset_m}  # fmt: skip
            if pv["observable"]:
                wl, wo = float(np.median(lat[w])), float(np.median(lon[w]))
                d = _dist_m(wl, wo, float(np.median(lat[pre])), float(np.median(lon[pre])))
                pv.update(offset_vs_pre_m=d, sign=bool(d > max_offset_m))
                if known is not None:
                    pv["offset_vs_known_m"] = _dist_m(wl, wo, float(known[0]), float(known[1]))
            else:
                pv.update(reason="no timed fix in the window or in the pre-window span", sign=None)
            block["pvt_offset"] = pv
        signs = [bool(block[k]) for k in ("psd_change", "cn0_step", "second_peak")] + [block["pvt_offset"]["sign"]]
        block["any_sign"] = any(x is True for x in signs)
        out.append(block)
    return out


def survey_ecef(positions: FloatArray) -> FloatArray:
    """The survey's position: the median of its (lon, lat, h) fixes, in ECEF."""
    lon, lat, h = np.median(np.asarray(positions, dtype=np.float64), axis=0)
    return geodetic_to_ecef(float(np.radians(lat)), float(np.radians(lon)), float(h))


def sky_blocks(windows: Sequence[Mapping[str, Any]], *, cn0: SeriesByPrn, dop: SeriesByPrn,
               to_gps: Callable[[Any], Any], eph: Ephemerides, rx_ecef: FloatArray,
               rules: SkyRules) -> dict[str, dict[str, Any]]:  # fmt: skip
    """The sky check of each window {name, t_start_s, t_end_s, expect: pass|fail}: per PRN the median C/N0 of its
    in-lock records in [t_start_s, t_end_s) and the GPS time (`to_gps`, on a float or an array) and Doppler of each,
    judged at those epochs. Each block reports S1 to S3 and, beside them, S1 with the unhealthy records refused
    (`S1_healthy_only`, not decisive) and the PRNs whose elevation came from an unhealthy record. `as_expected` is
    whether the verdict is the spec's (a control that fails, or a window that passes); the decision is the
    controller's."""
    out: dict[str, dict[str, Any]] = {}
    for w in windows:
        t0, t1 = float(w["t_start_s"]), float(w["t_end_s"])
        sats: dict[int, SkySeries] = {}
        for p, (t, d) in dop.items():
            sel = (t >= t0) & (t < t1)
            tc, c = cn0[p]
            sel_c = (tc >= t0) & (tc < t1)
            if sel.any() and sel_c.any():
                sats[p] = (float(np.median(c[sel_c])), np.asarray(to_gps(t[sel]), dtype=np.float64), d[sel])
        gps_t = float(to_gps(0.5 * (t0 + t1)))
        chk = sky_check(dict(sorted(sats.items())), eph, rx_ecef, rules)
        expect = str(w["expect"])
        ok = all(chk.passed.values())
        out[str(w["name"])] = {
            "t_start_s": t0,
            "t_end_s": t1,
            "gps_t_s": gps_t,
            "expect": expect,
            "as_expected": ok if expect == "pass" else not ok,
            "passed": dict(chk.passed),
            "S1_healthy_only": chk.s1_healthy_only,
            "unhealthy_prns": list(chk.unhealthy_prns),
            "n_sats": chk.n_sats,
            "clock_drift_hz": chk.clock_drift_hz,
            "doppler_rms_hz": chk.doppler_rms_hz,
            "spearman_rho": chk.spearman_rho,
            "spearman_p": chk.spearman_p,
            "prn": {
                str(p): {
                    "cn0_dbhz": sats[p][0],
                    "n_records": int(sats[p][1].size),
                    "t_gps_median_s": float(np.median(sats[p][1])),
                    "doppler_hz": float(np.median(sats[p][2])),
                    "elev_deg": chk.elev_deg[p],
                    "doppler_resid_hz": chk.doppler_resid_hz.get(p, float("nan")),
                }
                for p in sats
            },
        }
    return out


def sky_spec(sky: Mapping[str, Any]) -> tuple[Ephemerides, float, SkyRules]:
    """The broadcast ephemerides, the GPS second at 0 h of the recording's day, and the rules, from a spec's `sky:`."""
    eph, _ = read_rinex_nav(data_root() / str(sky["nav"]))
    y, m, d = (int(x) for x in str(sky["day"]).split("-"))
    return eph, gps_seconds(y, m, d, 0, 0, 0.0), SkyRules(**sky.get("rules", {}))


def interval_row(spec: Mapping[str, Any], adm: Admission) -> dict[str, Any]:
    """The `registry/intervals.yaml` entry this admission proposes (docs/7 section 4.1)."""
    name = Path(spec["local_path"]).name
    fs = float(spec["fs_hz"])
    w = adm.window_s
    return {
        "interval_id": f"{spec['dataset']}-{Path(name).stem}-clean",
        "dataset_id": spec["dataset"],
        "file": name,
        "t_start": None,  # receiver time is not known from the file; sample ranges are the reference
        "t_end": None,
        "sample_start": 0 if w is None else round(w[0] * fs),
        "sample_end": 0 if w is None else round(w[1] * fs),
        "state": "clean" if w is not None else "excluded",
        "class": None,
        "band": spec.get("band"),
        "power": None,
        "receivers": list(spec.get("receivers", [])),
        "static": True,
        "coordinates_source": spec.get("coordinates_source"),
        "kp_max": None,
        "reference_check": "none",
        "excluded": [] if w is not None else sorted(adm.reasons),
        "notes": None if adm.onset_s is None else f"first onset sign at {adm.onset_s:g} s",
    }


def run_sky(sky: Mapping[str, Any], tracks: Sequence[pa.Table], obs_dump: Path | None,
            channels: Mapping[int, tuple[str, str, int]], fs_hz: float, pos: FloatArray, cn0: SeriesByPrn
            ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:  # fmt: skip
    """The spec's sky windows on a survey: position from its fixes, GPS time from its observables."""
    if obs_dump is None or not pos.size:
        raise ValueError("the sky check needs the survey's observables and at least one position fix")
    eph, day_start, rules = sky_spec(sky)
    obs = read_observables_dump(obs_dump, channels, source_id="survey", receiver_id="survey")
    to_gps = gps_time_map(obs, n_channels=len(channels), fs_hz=fs_hz, interval_s=OBS_INTERVAL_S, day_start_s=day_start)
    rx = survey_ecef(pos)
    blocks = sky_blocks(sky["windows"], cn0=cn0, dop=dop_by_prn(tracks), to_gps=to_gps, eph=eph, rx_ecef=rx,
                        rules=rules)  # fmt: skip
    context = {"nav": str(sky["nav"]), "day": str(sky["day"]), "rx_ecef_m": [float(x) for x in rx],
               "gps_s_at_recording_start": float(to_gps(0.0)), "rules": asdict(rules)}  # fmt: skip
    return blocks, context


def run_admit(spec: dict[str, Any], run_dir: Path) -> Outcome:
    path, verified = resolve_prefix(data_root(), spec["local_path"])
    fs, item = float(spec["fs_hz"]), str(spec["item_type"])
    real_if = (spec.get("source") or {}).get("real_if")
    rules = ScreenRules(**spec.get("rules", {}))
    bw = float(spec["frontend_bw_hz"])
    template = str(spec.get("template", "1C"))
    rx = receiver_spec(spec, template=template)
    work = results_root() / "Admit-work" / run_dir.name
    shm = SinkSpec("shm_file", SHM_ROOT / f"admit-{run_dir.name}" / "survey.bin")
    if real_if and item != "byte":  # "ibyte" is interleaved complex bytes; a real IF file has one byte per sample
        raise ValueError(f"a real-IF source is real bytes: item_type must be 'byte', not {item!r}")
    n_file = path.stat().st_size // (1 if real_if else ITEM_BYTES[item])
    duration = n_file / fs
    check_windows(spec, duration)  # before the conversion, which takes minutes and gigabytes
    onset_windows = spec.get("onset_windows", [])
    try:  # the converted prefix and the survey's copy live in /dev/shm: gone on success and on failure
        real_sha = ""
        if real_if:  # real int8 samples at an intermediate frequency: converted to baseband ishort, screened from there
            n_in = n_file
            decim = int(real_if["decim"])
            n = -(-n_in // decim)
            require_free(Path("/dev/shm"), 4 * n, reserve=SHM_MARGIN)
            shm.path.parent.mkdir(parents=True, exist_ok=True)
            real_sha = to_baseband(path, shm.path, fs_hz=fs, if_hz=float(real_if["if_hz"]), decim=decim,
                        taps=int(real_if["taps"]), sample_end=n_in)  # fmt: skip
            source = IQSource(path=shm.path, item_type="ishort", fs_hz=fs / decim, sample_start=0, sample_end=n)
        else:
            n = n_file
            source = IQSource(path=path, item_type=spec["item_type"], fs_hz=fs, sample_start=0, sample_end=n,
                              q_first=bool(spec.get("q_first", False)))  # fmt: skip
        power = power_1ms_db(source, spec["power_windows"], spec["power_baseline_s"]) if "power_windows" in spec else {}
        floor = estimate_n0(source, frontend_bw_hz=bw)
        psd = psd_deviation_db(source, frontend_bw_hz=bw)
        d_lo, d_hi = spec.get("acquisition_doppler_hz", (None, 6000.0))  # the second-peak scan's Doppler range, Hz
        peaks = (
            acquisition_scan(source, list(range(1, 33)), doppler_min_hz=d_lo, doppler_max_hz=float(d_hi))
            if spec.get("acquisition", True)
            else []  # fmt: skip
        )
        if real_if:  # the converted file is the survey's source: its scale is fixed and cannot clip
            src = SourceSpec(path=shm.path, item_type="ishort", fs_hz=source.fs_hz)
        else:
            fe = FrontEndSpec(int(spec["decim"]), int(spec["fir_taps"]), survey_scale(source))
            inject(source, [], fe, shm, n0_per_hz=floor.n0_per_hz, workers=int(spec.get("workers", 1)))
            src = SourceSpec(path=shm.path, item_type="ishort", fs_hz=fs / fe.decim)
        channels = {ch: ("G", template, 0) for ch in range(int(spec.get("max_channels", 12)))}
        art = run_receiver(
            rx, render_config(rx, src, channels), src, work / "survey", float(spec.get("timeout_s", 3600))
        )
    finally:
        shutil.rmtree(shm.path.parent, ignore_errors=True)
    tracks = [
        read_tracking_dump(
            p, "patched112", run_id="survey", channel=ch, sys="G", signal=template, fs_hz=src.fs_hz, t_int_s=1e-3
        )  # fmt: skip
        for ch, p in art.tracking_dumps.items()
    ]
    cn0 = cn0_by_prn(tracks)
    geo = sorted((work / "survey" / "pvt").glob("*.geojson"))
    known = spec.get("known_position")
    pos = read_pvt_geojson(geo[0]) if geo else np.empty((0, 3))
    sky: dict[str, Any] = {}
    sky_context: dict[str, Any] = {}
    fixes: tuple[FloatArray, FloatArray, FloatArray] | str = "no onset_windows in the spec"
    logs = keep_logs(work, run_dir)  # first, so that a failure below still leaves the survey's log
    try:
        if onset_windows:  # the work directory is deleted below: the timed fixes are read here
            try:
                obs = (read_observables_dump(art.observables_dump, channels, source_id="survey", receiver_id="survey")
                       if art.observables_dump is not None else None)  # fmt: skip
                fixes = timed_fixes(sorted((work / "survey" / "pvt").glob("*.nmea")), obs, n_channels=len(channels),
                                    fs_hz=src.fs_hz, duration_s=duration)  # fmt: skip
            except FixTimesError as exc:
                fixes = str(exc)
        if "sky" in spec:  # the work directory is deleted below: everything the check needs is read here
            sky, sky_context = run_sky(spec["sky"], tracks, art.observables_dump, channels, src.fs_hz, pos, cn0)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if pos.size:
        pvt = pvt_summary(pos, known=None if known is None else (float(known[0]), float(known[1]), float(known[2])))
    else:
        pvt = {"n": 0.0, "sd_h_m": float("nan"), "offset_m": float("nan")}
    steps = cn0_steps(cn0, step_db=rules.cn0_step_db, n_sats=rules.cn0_step_sats) if cn0 else np.array([])
    adm = admit(duration_s=duration, psd=psd, peaks=peaks, steps=steps, cn0=cn0, pvt=pvt, rules=rules)
    timed = ("psd_change", "cn0_step", "second_peak", "few_sats")  # reason values that are times (s)
    whole = {"pvt_offset": "offset_m", "pvt_moving": "sd_h_m", "cn0_spread": "spread_db", "pvt_none": "n_fixes"}
    onsets = (
        onset_blocks(
            onset_windows,
            adm.reasons,
            fixes,
            pre_s=(rules.warmup_s, float(onset_windows[0][0])),
            max_offset_m=rules.pvt_max_offset_m,
            known=known,
        )  # fmt: skip
        if onset_windows
        else []
    )
    second = defaultdict(float)
    for pk in peaks:
        second[pk.prn] = max(second[pk.prn], pk.second_ratio)
    summary = {
        "prefix": str(path.relative_to(data_root())),
        "verified": verified,
        "duration_s": duration,
        "n0_per_hz": floor.n0_per_hz,
        "line_excess_db": floor.line_excess_db,
        "admission": {
            **{k: v for k, v in asdict(adm).items() if k != "reasons"},
            "reasons": {k: {"count": len(v), "first": v[:MAX_LISTED]} for k, v in adm.reasons.items() if k in timed},
            # checks of the whole prefix: values with a unit, not times (the old `first` read as a time in seconds)
            "whole_prefix": {k: {whole[k]: v[0]} for k, v in adm.reasons.items() if k in whole},
        },
        "onset_windows": onsets,
        "real_if": {"sha256": real_sha, "n_samples": n} if real_if else None,
        "power_1ms": power,
        "sky": sky,
        "sky_context": sky_context,
        "interval_row": interval_row(spec, adm),
        "acquisition_max_second_ratio": dict(sorted(second.items())),
        "rules": asdict(rules),
        "receivers": {**provenance(rx, {"survey": art}), "logs": logs},
    }
    inputs = [path] if verified else []  # an unverified prefix changes its name once verified: not hashed yet
    if "sky" in spec:  # the ephemerides S1-S3 were computed from
        inputs.append(data_root() / str(spec["sky"]["nav"]))
    return Outcome(inputs=inputs, counts={"reasons": len(adm.reasons)}, summary=summary)
