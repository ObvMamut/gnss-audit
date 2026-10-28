"""E0 (b-d): the whole Route B path on a synthetic recording and on registered real recordings (P2a, P2b task 10).

Variants:
- `synthetic`: a ghost on white noise (P2a development);
- `recording`: any registered interval (the CTTC sample, an FGI or SpoofRepo segment), GPS L1 C/A or L5, decimated as
  the spec says. Real satellites, an optional ghost (C/A only), three zero-gain twins, band-limited noise at several
  J/N0, and phase tones received under each loop cell (docs/10 E0 (c)).

The run exercises injection (docs/8), the patched receiver (D-08, D-27; D-34 and its amendments), the dump readers,
twin and ghost truth (docs/3) and the gates (docs/11 section 5; D-33). It reports V6 (C/N0 against Betz), V7 and the
ghost sanity check, linearity (docs/3 section 7), the loop response identified on real twins per cell (G-hz, a gate
for G3), the stock lock verdict replayed from the dumps (D-34), the integrated autocorrelation time of the exceedance
indicators, observables-epoch truth beside tracking-record truth, and the throughput of injector and receiver. Gate
failures are recorded, then raised as GateFailure once every result is written.

Times inside a run are recording time (`truth.rxtime.TwinClock`); the spec's times (warm-up, onsets) count from the
segment's start.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
from numpy.typing import ArrayLike

from gnssaudit.eval.dependence import tau_int_s
from gnssaudit.experiments import Outcome
from gnssaudit.experiments.twin import (
    SHM_ROOT,
    WARMUP_LOCK_MIN,
    LockPolicy,
    Twin,
    common_obs,
    concat_tracks,
    ghost_sanity,
    inject_twin,
    jsonable,
    keep_logs,
    median_of,
    provenance,
    receive,
    receiver_run_count,
    receiver_spec,
    run_twin,
    runs_of,
    sigma_disc_cyc,
    sigma_reported_cyc,
    throughput,
    track_as_obs,
    warmup_lock_gates,
    write_table,
)
from gnssaudit.gates import GateFailure, GateReport, identity_gate, lockin, null_gate, repeat_gate, stamp_and_verdict
from gnssaudit.inject import (
    Disturbance,
    FrontEndSpec,
    GhostSpec,
    IQSource,
    ca_psd,
    clean_twin,
    disturbance_seed,
    estimate_n0,
    int16_scale,
    ssc_kappa,
    tone_set,
)
from gnssaudit.io.gnsssdr import Channel
from gnssaudit.paths import results_root
from gnssaudit.registry import dataset_file, load_intervals
from gnssaudit.sdr import loop_overrides
from gnssaudit.sdr.lock import SETTLE_S, in_lock, stock_rule, would_drop
from gnssaudit.sim.recording import write_ca_recording
from gnssaudit.truth.ghost import ghost_truth, phase_precision_ok
from gnssaudit.truth.rxtime import ObsClock, TwinClock, onset_epoch
from gnssaudit.truth.twin import twin_difference
from gnssaudit.uq.loop import CarrierLoop

T_S = 1e-3  # GPS L1 C/A and L5 code period: the integration time of the default loop
SLIP_CYC = {"1C": 0.5, "L5": 1.0}  # Costas on C/A data; a pure PLL on the L5 pilot
TONE_PRE_S = 10.0  # pre-onset reference window of the phase-tone twin
TONE_MIN_SPAN_S = 10.0  # least post-onset span a channel needs for the lock-in
DEFAULT_BW_HZ = 35.0  # the v0.0.22 default loop (D-34)
OBS_INTERVAL_S = 0.02  # GNSS-SDR.observable_interval_ms in the templates
IN_VIEW_MIN_S = 1.0  # time in lock that puts a surveyed PRN in view


# ------------------------------------------------------------------------------------------------- helpers
def choose_ghost(
    visible: Mapping[int, float], *, doppler_hz: float, min_sep_hz: float = 50.0, prn: int | None = None
) -> tuple[int, float]:
    """The highest GPS PRN not in view (or the pinned `prn`, refused if in view), at the Doppler nearest doppler_hz
    (10 Hz steps) that stays min_sep_hz from 1 kHz multiples of every visible satellite's Doppler, where C/A
    cross-correlation is worst (docs/8 section 4)."""
    if prn is not None and prn in visible:
        raise ValueError(f"the pinned ghost PRN {prn} is in view")
    if prn is None:
        prn = next(p for p in range(32, 0, -1) if p not in visible)

    def clear(f: float) -> bool:
        beats = [(f - v) % 1000.0 for v in visible.values()]
        return all(min(b, 1000.0 - b) >= min_sep_hz for b in beats)

    for k in range(100):
        for f in (doppler_hz + 10.0 * k, doppler_hz - 10.0 * k):
            if clear(f):
                return prn, float(f)
    raise ValueError(f"no Doppler within 1 kHz of {doppler_hz} Hz keeps clear of the visible satellites' code lines")


def linearity_slope(levels_db_hz: ArrayLike, variances: ArrayLike) -> float:
    """Slope of log10 Var against level / 10: 1 when the variance scales as g^2 (docs/3 section 7)."""
    return float(np.polyfit(np.asarray(levels_db_hz) / 10, np.log10(np.asarray(variances)), 1)[0])


def betz_loss_db(jn0_db_hz: float, *, kappa_s: float) -> float:
    """C/N0 loss under interference of spectral separation kappa: 10 log10(1 + (J/N0) kappa) [2] (V6)."""
    return float(10 * np.log10(1 + 10 ** (jn0_db_hz / 10) * kappa_s))


@dataclass(frozen=True)
class LoopCell:
    """A receiver loop configuration whose response is identified on real twins (docs/10 E0 (c); D-21, D-34)."""

    name: str
    overrides: dict[str, str]
    loop: CarrierLoop  # the source-timing model the identified H is judged against
    t_s: float


def loop_cell(name: str, template: str) -> LoopCell:
    """`default` (35 Hz, order 3, T = 1 ms), `default-o2` (the same at order 2), or `bn<B>-t<T in ms>`: the inverse-map
    loop of effective bandwidth B at integration T (D-21; the E3 cells, `sdr.config.loop_overrides`)."""
    role = f"Tracking_{template}"
    if name == "default":
        return LoopCell(name, {}, CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S), T_S)
    if name == "default-o2":
        return LoopCell(name, {f"{role}.pll_filter_order": "2"}, CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S, order=2), T_S)
    if name.startswith("bn") and "-t" in name:
        bn_s, t_ms = name[2:].split("-t")
        bn, t = float(bn_s), float(t_ms) / 1000.0
        model = CarrierLoop.gnsssdr_for_bn_eff(bn, t, kappa=0.5, t_code_s=T_S if t > T_S else None)
        return LoopCell(name, loop_overrides(template, bn, t), model, t)
    raise ValueError(f"unknown loop cell {name!r}")


def lock_report(tracks: Mapping[int, pa.Table], template: str, warm_abs: float) -> dict[int, Any]:
    """D-34: the stock lock verdict replayed per channel (docs/9 section 10): the first would-be drop after the warm-up
    (recording time), and how many there would have been."""
    out: dict[int, Any] = {}
    for ch, tr in tracks.items():
        d = would_drop(tr, stock_rule(template))
        t = d.column("t_drop_s").to_numpy() if d.num_rows else np.zeros(0)
        t = t[t >= warm_abs]
        out[ch] = {"first_drop_s": float(t[0]) if t.size else None, "drops": int(t.size)}
    return out


def exceedance_tau_s(truth: pa.Table, sigma_cyc: float, after_s: float) -> float:
    """tau_int of the 95 % exceedance indicator 1[|e| > 1.96 sigma] of a twin truth series after `after_s` (the planning
    value of docs/5 section 4.1 is 1 s). NaN with fewer than 100 kept epochs."""
    t = truth.column("t_rx_s").to_numpy()
    ok = (t >= after_s) & ~truth.column("transient").to_numpy(zero_copy_only=False)
    if ok.sum() < 100:
        return float("nan")
    e = truth.column("e_cyc").to_numpy()[ok]
    dt = float(np.median(np.diff(t[ok])))
    return tau_int_s((np.abs(e) > 1.96 * sigma_cyc).astype(float), dt)


def epoch_label(obs: pa.Table, n_channels: int, clock: TwinClock, file_t: float) -> float:
    """The receiver-time label of the first observable epoch at or after recording time file_t. The dump holds one row
    per channel per epoch, and epoch i sits at output sample (i + 1) x step (docs/3 section 2.6; P2b task 9). NaN when
    that epoch has no receiver time yet."""
    step = round(clock.fs_native_hz / clock.decim * OBS_INTERVAL_S)
    m = int(np.ceil((file_t * clock.fs_native_hz - clock.sample_start + clock.delay) / clock.decim))
    i = max(onset_epoch(m, ObsClock(k0=1, step=step)), 0)
    rows = obs.column("t_gps_s").to_numpy()
    return float(rows[i * n_channels]) if i * n_channels < rows.size and rows[i * n_channels] > 0 else float("nan")


def recording_source(spec: Mapping[str, Any]) -> tuple[IQSource, TwinClock]:
    """The registered interval the spec names (`recording.interval`) as a source, and the clock of its twins (the
    interval's start, the spec's decimation and the decimation filter's delay)."""
    rec = spec["recording"]
    iv = load_intervals()[rec["interval"]]
    fs = float(rec["fs_hz"])
    s0, s1 = int(iv["sample_start"]), int(iv["sample_end"])
    source = IQSource(path=dataset_file(iv["dataset_id"], iv["file"]), item_type=rec["item_type"], fs_hz=fs,
                      sample_start=s0, sample_end=s1, q_first=bool(rec.get("q_first", False)))  # fmt: skip
    decim, taps = int(spec.get("decim", 1)), int(spec.get("fir_taps", 1))
    return source, TwinClock(fs_native_hz=fs, sample_start=s0, decim=decim, delay=(taps - 1) / 2 if decim > 1 else 0.0)


# ------------------------------------------------------------------------------------------------ the runner
def run_e0b(spec: dict[str, Any], run_dir: Path) -> Outcome:
    check_spec(spec)
    work = results_root() / "E0b-work" / run_dir.name
    shutil.rmtree(work, ignore_errors=True)  # a crashed run's dumps must never be read as this run's
    work.mkdir(parents=True)
    t0 = time.perf_counter()
    try:
        if spec["variant"] == "synthetic":
            summary, inputs, gates = _synthetic(spec, work, run_dir)
        elif spec["variant"] == "recording":
            summary, inputs, gates = _recording(spec, work, run_dir)
        else:
            raise ValueError(f"unknown E0b variant {spec['variant']!r}")
    finally:
        shutil.rmtree(SHM_ROOT / work.name, ignore_errors=True)
    summary["wall_s"] = round(time.perf_counter() - t0, 1)
    summary["receivers"]["logs"] = keep_logs(work, run_dir)
    summary["gates"] = {g.gate + suffix: {"passed": g.passed, **jsonable(g.details)} for suffix, g in gates}
    (run_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    shutil.rmtree(work, ignore_errors=True)  # raw dumps are reduced and deleted (D-18)
    failed = [g for _, g in gates if not g.passed]
    if failed and spec.get("abort_on_gate_failure", True):
        raise GateFailure(failed[0])
    counts = {"gates_failed": len(failed), "receiver_runs": int(summary.get("receiver_runs", 0))}
    return Outcome(inputs=inputs, counts=counts, summary=jsonable(summary))


def check_spec(spec: Mapping[str, Any]) -> None:
    """Refuses, before any work: a pre-onset reference window that starts inside the warm-up (the twins share no
    phase there yet), a ghost on a signal other than GPS L1 C/A (D-16), and an unknown loop cell."""
    warm = float(spec["warmup_s"])
    windows = [(float(spec["onset_s"]), float(spec["pre_window_s"]))]
    if "tones" in spec:
        windows.append((float(spec["tones"]["onset_s"]), TONE_PRE_S))
    for onset, pre in windows:
        if onset - pre < warm:
            raise ValueError(
                f"the pre-onset window [{onset - pre:g}, {onset:g}) s starts inside the {warm:g} s warm-up"
            )
    template = str(spec.get("template", "1C"))
    if "ghost" in spec and template != "1C":
        raise ValueError(f"ghosts are GPS L1 C/A only (D-16); this spec runs {template}")
    for name in spec.get("loop_cells", []):
        loop_cell(name, template)


def _fe(source: IQSource, n0: float, max_level_db_hz: float, decim: int = 1, taps: int = 1) -> FrontEndSpec:
    return FrontEndSpec(decim=decim, fir_taps=taps, scale=int16_scale(n0, source.fs_hz, max_level_db_hz))


def _ghost(spec: Mapping[str, Any], prn: int, doppler: float, seed: int) -> Disturbance:
    params = {
        "prn": prn,
        "doppler_hz": doppler,
        "doppler_rate_hzps": float(spec.get("doppler_rate_hzps", 0.0)),
        "chip0": float(spec.get("chip0", 17.3)),
    }
    return Disturbance("ghost", params, 0.0, 0.0, float(spec["cn0_db_hz"]), seed)


def _noise(spec: Mapping[str, Any], level: float, onset_s: float, seed: int) -> Disturbance:
    return Disturbance("noise", {"bw_hz": float(spec["bw_hz"]), "f_hz": 0.0}, onset_s, 0.01, level, seed)


def _ghost_checks(
    twin: Twin, g: Disturbance, ch: int, clock: TwinClock, converge_s: float, line_excess_db: float
) -> dict[str, Any]:
    """I-ghost and V7 (acquired, tracked, C/N0 within 0.5 dB of the set level) and the ghost sanity check (docs/3 7).
    The level is set against the median floor; the set C/N0 the receiver can see is that level less the floor's line
    excess (lines, real satellites), which is zero on synthetic noise."""
    track = twin.tracks.get(ch)
    if track is None or track.num_rows == 0:
        return {"tracked": False}
    truth = ghost_truth(track, GhostSpec.of(g), clock=clock, converge_s=converge_s, truth_id=f"{twin.name}-ghost")
    t0 = float(track.column("t_rx_s").to_numpy()[0])
    cn0 = median_of(track, "cn0_dbhz", t0 + converge_s)
    e = truth.column("e_cyc").to_numpy()[truth.column("valid").to_numpy(zero_copy_only=False)]
    loop = CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S)
    cn0_set = g.level_db_hz - line_excess_db
    sigma = sigma_reported_cyc(cn0_set, bn_eff_hz=loop.bn_eff_hz(), t_s=T_S)
    dt = float(np.median(np.diff(truth.column("t_rx_s").to_numpy()))) if truth.num_rows > 1 else T_S
    sanity = ghost_sanity(e, sigma_cyc=sigma, dt_s=dt)
    return {
        "tracked": True,
        "start_s": t0,
        "cn0_set_dbhz": cn0_set,
        "cn0_reported_dbhz": cn0,
        "v7_cn0_error_db": cn0 - cn0_set,
        "v7_pass": abs(cn0 - cn0_set) <= 0.5,
        "error_sd_over_b3": sanity["sd_over_b3"],
        "error_sd_se": sanity["se"],
        "sanity_pass": sanity["pass"],
        "truth": truth,
    }


def _precision_gate(tracks: Sequence[pa.Table], sigma_cyc: float) -> GateReport:
    acc_max = max(float(np.max(np.abs(t.column("acc_phase_rad").to_numpy()))) for t in tracks)
    sigma_rad = sigma_cyc * 2 * np.pi
    return GateReport(
        "G-phase-precision",
        phase_precision_ok(acc_max, np.dtype(np.float64), sigma_rad),
        {
            "max_abs_phase_rad": acc_max,
            "float32_would_pass": phase_precision_ok(acc_max, np.dtype(np.float32), sigma_rad),
        },
    )


def _synthetic(
    spec: dict[str, Any], work: Path, run_dir: Path
) -> tuple[dict[str, Any], list[Path], list[tuple[str, GateReport]]]:
    fs, seconds = float(spec["fs_hz"]), float(spec["seconds"])
    base = work / "base.bin"
    write_ca_recording(base, prn=1, doppler_hz=0.0, cn0_dbhz=-np.inf, seconds=seconds, fs_hz=fs, seed=int(spec["seed"]))
    source = IQSource(path=base, item_type="ibyte", fs_hz=fs, sample_start=0, sample_end=int(seconds * fs))
    clock = TwinClock.identity(fs)
    floor = estimate_n0(source, frontend_bw_hz=0.8 * fs)
    n0 = floor.n0_per_hz
    rx = receiver_spec(spec)
    gs, ns = spec["ghost"], spec["noise"]
    g = _ghost(gs, int(gs["prn"]), float(gs["doppler_hz"]), disturbance_seed(int(spec["seed"]), 0, 0, 0))
    channels: dict[int, Channel] = {0: ("G", "1C", int(gs["prn"]))}
    onset, warm, pre = float(spec["onset_s"]), float(spec["warmup_s"]), float(spec["pre_window_s"])
    fe = _fe(source, n0, max(ns["levels_db_hz"]))
    timeout = float(spec.get("timeout_s", 1800))
    ref_noise = _noise(ns, ns["levels_db_hz"][0], onset, disturbance_seed(int(spec["seed"]), 1, 0, 1))

    def twin(name: str, ds: Sequence[Disturbance]) -> Twin:
        return run_twin(name, source, ds, fe, channels, work, n0, timeout, rx, clock=clock, warmup_s=warm)

    zero = [twin(f"zero{i}", clean_twin([g, ref_noise])) for i in range(3)]
    jammed = {lv: twin(f"jam{lv:g}", [g, _noise(ns, lv, onset, disturbance_seed(int(spec["seed"]), 1, k, 1))])
              for k, lv in enumerate(ns["levels_db_hz"])}  # fmt: skip
    lock_gates = warmup_lock_gates([*zero, *jammed.values()])
    if not all(g_.passed for _, g_ in lock_gates):  # one channel: nothing can be paired without it
        raise GateFailure(next(g_ for _, g_ in lock_gates if not g_.passed))
    ghost = _ghost_checks(zero[0], g, 0, clock, warm / 2, floor.line_excess_db)
    if ghost.get("tracked"):
        write_table(run_dir, "ghost_truth_clean", ghost.pop("truth"))
    gates: list[tuple[str, GateReport]] = list(lock_gates)
    sd_d = sigma_disc_cyc(g.level_db_hz, t_s=T_S)
    sd_rep = sigma_reported_cyc(g.level_db_hz, bn_eff_hz=CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S).bn_eff_hz(), t_s=T_S)
    reports = [z.report for z in zero if z.report is not None]
    gates.append(("", repeat_gate(reports, [concat_tracks(z.tracks) for z in zero], warmup_s=warm, sigma_th_cyc=sd_d)))
    obs = {z.name: track_as_obs(concat_tracks(z.tracks), source_id="synthetic", receiver_id=z.name) for z in zero}

    def diff(a: pa.Table, b: pa.Table, truth_id: str) -> tuple[pa.Table, pa.Table, pa.Table]:
        return twin_difference(a, b, onset_s=onset, pre_window_s=pre, warmup_s=warm, slip_size_cyc=SLIP_CYC["1C"],
                               tau_loop_s=0.03, truth_id=truth_id)  # fmt: skip

    null, _, null_ex = diff(obs["zero0"], obs["zero1"], "null")
    write_table(run_dir, "null_truth", null)
    gates.append(("", null_gate(null, null_ex, sigma_th_cyc=sd_rep)))
    var_by_level = {}
    kappa = _kappa(float(ns["bw_hz"]), fs)
    cn0_clean = median_of(zero[0].tracks[0], "cn0_dbhz", onset + 2.0)
    v6 = {}
    for lv, tw in jammed.items():
        if 0 in tw.tracks:
            loss = cn0_clean - median_of(tw.tracks[0], "cn0_dbhz", onset + 2.0)
            v6[lv] = loss - betz_loss_db(lv - floor.line_excess_db, kappa_s=kappa)
        gates.append((f"[{lv:g}]", identity_gate(concat_tracks(zero[0].tracks), concat_tracks(tw.tracks),
                                                 onset_s=onset, warmup_s=warm, sigma_th_cyc=sd_d)))  # fmt: skip
        jo = track_as_obs(concat_tracks(tw.tracks), source_id="synthetic", receiver_id=tw.name)
        truth, _, _ = diff(obs["zero0"], jo, tw.name)
        write_table(run_dir, f"truth_{tw.name}", truth)
        post = truth.filter((truth.column("t_rx_s").to_numpy() >= onset + 2.0)
                            & ~truth.column("transient").to_numpy(zero_copy_only=False))  # fmt: skip
        var_by_level[lv] = float(np.var(post.column("e_cyc").to_numpy())) if post.num_rows else float("nan")
    gates.append(("", _precision_gate([t for z in zero for t in z.tracks.values()], sd_rep)))
    runs = runs_of([*zero, *jammed.values()])
    summary = {
        "variant": "synthetic",
        "n0_per_hz": n0,
        "floor_line_excess_db": floor.line_excess_db,
        "ghost": ghost,
        "kappa_s": kappa,
        "v6_residual_db": v6,
        "v6_pass": bool(v6) and all(abs(r) <= 0.5 for r in v6.values()),
        "jammed_var_cyc2": var_by_level,
        "null_exclusions": int(sum(null_ex.column("count").to_pylist())),
        "receiver_runs": receiver_run_count(runs),
        "receivers": provenance(rx, runs),
        "throughput": throughput([*zero, *jammed.values()], seconds),
    }
    return summary, [], gates


def survey_satellites(survey: Twin, warm_abs: float) -> tuple[dict[int, float], dict[int, float]]:
    """The PRNs in view after the warm-up, with their median Doppler and C/N0 over the records settled in lock
    (`sdr.lock.in_lock`), on the channel that tracked each best. A PRN is in view once held in lock for
    IN_VIEW_MIN_S: a noise track reads about 29 dB-Hz at a lock statistic near 0, while a weak real satellite reads
    0.36 at 30 dB-Hz (F22). Every PRN in view, however weak, stays out of the ghost's choice."""
    dop: dict[int, float] = {}
    cn0: dict[int, float] = {}
    held: dict[int, float] = {}
    for tr in survey.tracks.values():
        t = tr.column("t_rx_s").to_numpy()
        prn = tr.column("prn").to_numpy()
        locked = in_lock(tr, min_lock=WARMUP_LOCK_MIN, settle_s=SETTLE_S)
        dt = float(np.median(np.diff(t))) if t.size > 1 else 0.0
        for p in np.unique(prn[(prn > 0) & (t >= warm_abs) & locked]):
            sel = (prn == p) & (t >= warm_abs) & locked
            held[int(p)] = held.get(int(p), 0.0) + float(sel.sum()) * dt
            c = float(np.median(tr.column("cn0_dbhz").to_numpy()[sel]))
            if c > cn0.get(int(p), -np.inf):
                dop[int(p)], cn0[int(p)] = float(np.median(tr.column("dop_hz").to_numpy()[sel])), c
    keep = sorted(p for p, s in held.items() if s >= IN_VIEW_MIN_S)
    return {p: dop[p] for p in keep}, {p: cn0[p] for p in keep}


def strongest(cn0: Mapping[int, float], *, min_cn0_dbhz: float, n: int) -> list[int]:
    """The n strongest PRNs at min_cn0_dbhz or more, in PRN order."""
    keep = sorted((p for p in cn0 if cn0[p] >= min_cn0_dbhz), key=lambda p: -cn0[p])[:n]
    if not keep:
        raise ValueError(f"the survey found no satellite at {min_cn0_dbhz:g} dB-Hz or more")
    return sorted(keep)


def _recording(
    spec: dict[str, Any], work: Path, run_dir: Path
) -> tuple[dict[str, Any], list[Path], list[tuple[str, GateReport]]]:
    source, clock = recording_source(spec)
    template = str(spec.get("template", "1C"))
    fs = source.fs_hz
    seconds = (source.sample_end - source.sample_start) / fs
    t0 = clock.sample_start / fs  # the segment's start in recording time
    floor = estimate_n0(source, frontend_bw_hz=float(spec["frontend_bw_hz"]))
    n0 = floor.n0_per_hz
    timeout = float(spec.get("timeout_s", 1800))
    seed = int(spec.get("seed", 7))
    workers = int(spec.get("workers", 1))
    max_ch = int(spec.get("max_channels", 8))
    warm, pre = float(spec["warmup_s"]), float(spec["pre_window_s"])
    warm_abs, onset = t0 + warm, t0 + float(spec["onset_s"])
    slip = SLIP_CYC[template]
    ns, ts = spec["noise"], spec["tones"]
    fe = _fe(source, n0, max(ns["levels_db_hz"]), clock.decim, int(spec.get("fir_taps", 1)))
    rx = receiver_spec(spec, template=template)
    # survey: a zero-gain twin, decimated as the twins are, on free channels: which satellites are in view, how strong.
    # It is no twin, so it runs the stock receiver: D-34's raised limit would keep noise tracks and bad starts on the
    # free channels for good, and the search would stop (P2b task 6)
    free: dict[int, Channel] = {ch: ("G", template, 0) for ch in range(max_ch)}
    rx_survey = receiver_spec({"stock_receiver": True}, template=template)
    survey = run_twin("survey", source, [], fe, free, work, n0, timeout, rx_survey, clock=clock, warmup_s=warm,
                      workers=workers)  # fmt: skip
    tracked, survey_cn0 = survey_satellites(survey, warm_abs)
    sats = strongest(survey_cn0, min_cn0_dbhz=float(spec.get("min_cn0_dbhz", 35.0)),
                     n=max_ch - (1 if "ghost" in spec else 0))  # fmt: skip
    ghost_d: list[Disturbance] = []
    gprn = gdop = None
    if "ghost" in spec:
        gs = spec["ghost"]
        pin = int(gs["prn"]) if "prn" in gs else None
        gprn, gdop = choose_ghost(tracked, doppler_hz=float(gs["doppler_hz"]), prn=pin)  # clear of every PRN in view
        ghost_d = [_ghost(gs, gprn, gdop, disturbance_seed(seed, 0, 0, 0))]
    prns = [*sats, gprn] if gprn is not None else sats
    channels: dict[int, Channel] = {i: ("G", template, p) for i, p in enumerate(prns)}
    ghost_ch = len(sats) if gprn is not None else None
    ref_noise = _noise(ns, ns["levels_db_hz"][0], onset, disturbance_seed(seed, 1, 0, 1))

    # what each channel should hear at the end of the warm-up: the survey's Doppler, the ghost's set one
    expected = {
        **{p: tracked[p] for p in sats},
        **({gprn: float(gdop)} if gprn is not None and gdop is not None else {}),
    }

    # the ghost must lock; a real channel whose start fails in every run is excluded and counted (D-34, P2b task 10)
    policy = LockPolicy(frozenset({ghost_ch} if ghost_ch is not None else ()), int(spec.get("min_locked_real", 4)))

    def twin(name: str, ds: Sequence[Disturbance]) -> Twin:
        return run_twin(name, source, ds, fe, channels, work, n0, timeout, rx, clock=clock, warmup_s=warm,
                        workers=workers, expected_doppler_hz=expected, lock_policy=policy)  # fmt: skip

    zero = [twin(f"zero{i}", clean_twin([*ghost_d, ref_noise])) for i in range(3)]
    jammed = {lv: twin(f"jam{lv:g}", [*ghost_d, _noise(ns, lv, onset, disturbance_seed(seed, 1, k, 1))])
              for k, lv in enumerate(ns["levels_db_hz"])}  # fmt: skip
    tones = Disturbance("phase", {k: float(v) for k, v in ts.items() if k != "onset_s"}, t0 + float(ts["onset_s"]),
                        0.0, 0.0, disturbance_seed(seed, 2, 0, 0))  # fmt: skip
    cells = [loop_cell(c, template) for c in spec.get("loop_cells", ["default"])]
    cell_twins = _cell_twins(cells, source, ghost_d, tones, fe, channels, work, n0, timeout, spec, template, clock,
                             warm, workers, expected, policy)  # fmt: skip
    all_twins = [*zero, *jammed.values(), *[tw for pair in cell_twins.values() for tw in pair]]
    gates: list[tuple[str, GateReport]] = warmup_lock_gates(all_twins)
    clean = zero[0]
    loop = CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S)
    cn0 = {ch: median_of(clean.tracks[ch], "cn0_dbhz", warm_abs) for ch in clean.tracks}
    real = [ch for ch in sorted(clean.tracks) if ch != ghost_ch and np.isfinite(cn0[ch])]
    rec_obs = {z.name: {ch: track_as_obs(z.tracks[ch], source_id="rec", receiver_id=z.name) for ch in z.tracks}
               for z in [*zero, *jammed.values()]}  # fmt: skip

    def diff(a: pa.Table, b: pa.Table, truth_id: str) -> tuple[pa.Table, pa.Table, pa.Table]:
        return twin_difference(a, b, onset_s=onset, pre_window_s=pre, warmup_s=warm_abs, slip_size_cyc=slip,
                               tau_loop_s=0.03, truth_id=truth_id)  # fmt: skip

    null_var: dict[int, float] = {}
    for ch in real:
        sd_d = sigma_disc_cyc(cn0[ch], t_s=T_S)
        sd_rep = sigma_reported_cyc(cn0[ch], bn_eff_hz=loop.bn_eff_hz(), t_s=T_S)
        gates.append((f"[ch{ch}]", repeat_gate([z.report for z in zero if z.report is not None],
                                               [z.tracks[ch] for z in zero if ch in z.tracks], warmup_s=warm_abs,
                                               sigma_th_cyc=sd_d)))  # fmt: skip
        if ch not in rec_obs["zero1"]:  # unlocked in the null twin: reported by G-warmup-lock
            continue
        null, _, null_ex = diff(rec_obs["zero0"][ch], rec_obs["zero1"][ch], f"null-ch{ch}")
        gates.append((f"[ch{ch}]", null_gate(null, null_ex, sigma_th_cyc=sd_rep)))
        null_var.update(_post_var(null, onset))
    obs_null = _observables_null(clean, zero[1], channels, warm, slip, run_dir)
    kappa = _kappa(float(ns["bw_hz"]), fs / clock.decim)
    lin: dict[int, dict[float, float]] = {}
    v6: dict[float, dict[int, float]] = {}
    tau: dict[float, dict[int, float]] = {}
    excluded: dict[str, int] = {}
    agreement: dict[float, dict[int, float]] = {}
    for lv, tw in jammed.items():
        v6[lv], tau[lv] = {}, {}
        for ch in real:
            if ch not in tw.tracks:
                continue
            prn = channels[ch][2]
            sd_d = sigma_disc_cyc(cn0[ch], t_s=T_S)
            ident = identity_gate(clean.tracks[ch], tw.tracks[ch], onset_s=onset, warmup_s=warm_abs, sigma_th_cyc=sd_d)
            gates.append((f"[{lv:g}][ch{ch}]", ident))
            truth, events, excl = diff(rec_obs["zero0"][ch], rec_obs[tw.name][ch], f"{tw.name}-ch{ch}")
            write_table(run_dir, f"truth_{tw.name}_ch{ch}", truth)
            for reason, count in zip(excl.column("reason_code").to_pylist(), excl.column("count").to_pylist(),
                                     strict=True):  # fmt: skip
                excluded[reason] = excluded.get(reason, 0) + count
            excluded["slips"] = excluded.get("slips", 0) + events.num_rows
            for p, v in _post_var(truth, onset).items():
                lin.setdefault(p, {})[lv] = v - null_var.get(p, 0.0)
            cn0_j = median_of(tw.tracks[ch], "cn0_dbhz", onset + 2.0)
            if template == "1C":  # kappa is taken against the C/A spectrum
                loss = median_of(clean.tracks[ch], "cn0_dbhz", onset + 2.0) - cn0_j
                v6[lv][prn] = loss - betz_loss_db(lv - floor.line_excess_db, kappa_s=kappa)
            sd_j = sigma_reported_cyc(cn0_j, bn_eff_hz=loop.bn_eff_hz(), t_s=T_S)
            tau[lv][prn] = exceedance_tau_s(truth, sd_j, onset + 2.0)
        agreement[lv] = _observables_agreement(clean, tw, clock, onset, pre, warm_abs, slip, real, channels)
    slopes = {prn: linearity_slope(list(d), list(d.values())) for prn, d in lin.items()
              if len(d) >= 3 and all(v > 0 for v in d.values())}  # fmt: skip
    ghz: dict[str, dict[str, Any]] = {}
    for cell in cells:
        c_clean, c_tones = cell_twins[cell.name]
        res = _tone_identification(c_clean, c_tones, tones, warm_abs, real, cell, slip)
        ghz[cell.name] = res
        gates.append((f"[{cell.name}]", GateReport("G-hz", bool(res["passed"]),
                                                   {k: v for k, v in res.items() if k != "rows"})))  # fmt: skip
        if res["identified"]:
            (run_dir / "loop").mkdir(exist_ok=True)
            record = {"cell": cell.name, "template": template, "t_s": cell.t_s, "overrides": cell.overrides,
                      "model": cell.loop.name, **res}  # fmt: skip
            (run_dir / "loop" / f"{template}.{cell.name}.loop.json").write_text(
                json.dumps(jsonable(record), indent=2, sort_keys=True) + "\n"
            )
    ghost: dict[str, Any] = {"applicable": False, "tracked": False}
    if ghost_ch is not None:
        ghost = {"applicable": True, **_ghost_checks(clean, ghost_d[0], ghost_ch, clock, warm / 2,
                                                     floor.line_excess_db)}  # fmt: skip
        if ghost.get("tracked"):
            write_table(run_dir, "ghost_truth_clean", ghost.pop("truth"))
    sd_min = min(sigma_reported_cyc(cn0[ch], bn_eff_hz=loop.bn_eff_hz(), t_s=T_S) for ch in real)
    gates.append(("", _precision_gate(list(clean.tracks.values()), sd_min)))
    runs = {**runs_of([survey]), **runs_of(all_twins)}
    summary = {
        "variant": "recording",
        "template": template,
        "segment": {
            "interval": spec["recording"]["interval"],
            "sample_start": source.sample_start,
            "sample_end": source.sample_end,
            "decim": clock.decim,
            "delay": clock.delay,
        },  # fmt: skip
        "seconds": seconds,
        "n0_per_hz": n0,
        "floor_line_excess_db": floor.line_excess_db,
        "survey_cn0_dbhz": survey_cn0,
        "visible": {p: tracked[p] for p in sats},
        "channels": channels,
        "ghost_prn": gprn,
        "ghost_doppler_hz": gdop,
        "cn0_clean_dbhz": cn0,
        "ghost": ghost,
        "linearity_slope": slopes,
        "linearity_pass": bool(slopes) and all(abs(s - 1) <= 0.05 for s in slopes.values()),
        "v6_residual_db": v6 if template == "1C" else None,
        "v6_pass": (bool(v6) and all(abs(r) <= 0.5 for d in v6.values() for r in d.values()))
        if template == "1C"
        else None,  # fmt: skip
        "kappa_s": kappa,
        "ghz": {name: {k: v for k, v in r.items() if k != "rows"} for name, r in ghz.items()},
        "ghz_rows": {name: r.get("rows", []) for name, r in ghz.items()},
        "tau_int_95_s": tau,
        "observables_null": obs_null,
        "observables_agreement_rms_over_sigma": agreement,
        "stock_lock_verdict": lock_report(clean.tracks, template, warm_abs),
        "null_floor_var_cyc2": null_var,
        "jammed_exclusions": excluded,
        "receiver_runs": receiver_run_count(runs),
        "receivers": provenance(rx, runs),
        "survey_receiver": {"stock_receiver": True, "overrides": dict(rx_survey.overrides)},
        "throughput": throughput(all_twins, seconds),
    }
    return summary, [source.path], gates


def _cell_twins(cells: Sequence[LoopCell], source: IQSource, ghosts: Sequence[Disturbance], tones: Disturbance,
                fe: FrontEndSpec, channels: Mapping[int, Channel], work: Path, n0: float, timeout: float,
                spec: Mapping[str, Any], template: str, clock: TwinClock, warm: float, workers: int,
                expected: Mapping[int, float], policy: LockPolicy) -> dict[str, tuple[Twin, Twin]]:  # fmt: skip
    """Per loop cell, a clean and a tone twin received under that cell's loop. Each twin is injected once and its
    bytes are received under every cell before they are deleted (one stream in /dev/shm at a time)."""
    out: dict[str, list[Twin]] = {c.name: [] for c in cells}
    for name, ds in (("cleancell", clean_twin([*ghosts, tones])), ("tones", [*ghosts, tones])):
        src, report = inject_twin(name, source, ds, fe, work, n0, workers=workers)
        try:
            for cell in cells:
                rx = receiver_spec(spec, template=template, extra=cell.overrides)
                tw = receive(f"{name}-{cell.name}", src, channels, work, timeout, rx, clock=clock, warmup_s=warm,
                             t_int_s=cell.t_s, expected_doppler_hz=expected, lock_policy=policy)  # fmt: skip
                tw.report = report
                out[cell.name].append(tw)
        finally:
            src.path.unlink(missing_ok=True)
    return {k: (v[0], v[1]) for k, v in out.items()}


def _observables_null(clean: Twin, other: Twin, channels: Mapping[int, Channel], warm: float, slip: float,
                      run_dir: Path) -> dict[str, Any]:  # fmt: skip
    """The observables path on the zero-gain pair. Its labels are receiver time, so the warm-up counts from the first
    valid epoch; two zero-gain twins have no onset, so the reference window is the whole settled span."""
    if clean.obs is None or other.obs is None:
        raise ValueError("the zero-gain twins produced no observables")
    t_obs = clean.obs.filter(clean.obs.column("valid")).column("t_gps_s").to_numpy()
    if not t_obs.size:
        raise ValueError("the clean twin produced no valid observables")
    obs_warm, obs_end = float(t_obs.min()) + warm, float(t_obs.max()) + 1.0
    oa, ob = common_obs(clean, other, channels)
    assert oa is not None and ob is not None
    null, slips, excl = twin_difference(oa, ob, onset_s=obs_end, pre_window_s=obs_end - obs_warm,
                                        warmup_s=obs_warm, slip_size_cyc=slip, tau_loop_s=0.03,
                                        truth_id="null-observables")  # fmt: skip
    write_table(run_dir, "null_truth_observables", null)
    prn = null.column("prn").to_numpy()
    e = null.column("e_cyc").to_numpy()
    return {
        "rms_cyc": {int(p): float(np.sqrt(np.mean(e[prn == p] ** 2))) for p in np.unique(prn)},
        "exclusions": int(sum(excl.column("count").to_pylist())),
        "slips": slips.num_rows,
    }


def _observables_agreement(clean: Twin, jammed: Twin, clock: TwinClock, onset: float, pre: float, warm_abs: float,
                           slip: float, real: Sequence[int],
                           channels: Mapping[int, Channel]) -> dict[int, float]:  # fmt: skip
    """Twin truth on observable epochs (after the map M, docs/3 section 2.6), with the onset placed by the clock map,
    against the tracking-record truth interpolated to those epochs: per PRN, the RMS of their difference over the
    reported-phase thermal SD. The observables phase is +acc / 2 pi, so the two truths have opposite signs (P2b
    task 9). Empty when the twins produced no observables or the onset epoch has no receiver time."""
    if clean.obs is None or jammed.obs is None:
        return {}
    n_ch = len(channels)
    rx_onset = epoch_label(clean.obs, n_ch, clock, onset)
    rx_warm = epoch_label(clean.obs, n_ch, clock, warm_abs)
    if not (np.isfinite(rx_onset) and np.isfinite(rx_warm)):
        return {}
    oa, ob = common_obs(clean, jammed, channels)
    assert oa is not None and ob is not None
    obs_truth, _, _ = twin_difference(oa, ob, onset_s=rx_onset, pre_window_s=pre, warmup_s=rx_warm,
                                      slip_size_cyc=slip, tau_loop_s=0.03, truth_id=f"{jammed.name}-obs")  # fmt: skip
    labels = clean.obs.column("t_gps_s").to_numpy()[0::n_ch]  # one label per epoch
    step = round(clock.fs_native_hz / clock.decim * OBS_INTERVAL_S)
    loop = CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S)
    out: dict[int, float] = {}
    for ch in real:
        if ch not in jammed.tracks:
            continue
        prn = channels[ch][2]
        o = obs_truth.filter(obs_truth.column("prn").to_numpy() == prn)
        if o.num_rows < 100:
            continue
        a = track_as_obs(clean.tracks[ch], source_id="rec", receiver_id="c")
        b = track_as_obs(jammed.tracks[ch], source_id="rec", receiver_id="j")
        trk, _, _ = twin_difference(a, b, onset_s=onset, pre_window_s=pre, warmup_s=warm_abs, slip_size_cyc=slip,
                                    tau_loop_s=0.03, truth_id="trk")  # fmt: skip
        t_lab = o.column("t_rx_s").to_numpy()
        i = np.clip(np.searchsorted(labels, t_lab), 1, labels.size - 1)
        i -= (t_lab - labels[i - 1]) < (labels[i] - t_lab)  # the nearest epoch label
        t_file = clock.file_time((i + 1) * step)
        e_trk = np.interp(t_file, trk.column("t_rx_s").to_numpy(), trk.column("e_cyc").to_numpy())
        d = o.column("e_cyc").to_numpy() + e_trk
        d = d - np.median(d)
        sigma = sigma_reported_cyc(median_of(clean.tracks[ch], "cn0_dbhz", warm_abs), bn_eff_hz=loop.bn_eff_hz(),
                                   t_s=T_S)  # fmt: skip
        out[prn] = float(np.sqrt(np.mean(d**2)) / sigma)
    return out


def _post_var(truth: pa.Table, onset_s: float) -> dict[int, float]:
    """Per PRN: variance of the twin difference after onset (2 s settling), transients excluded."""
    t = truth.column("t_rx_s").to_numpy()
    ok = (t >= onset_s + 2.0) & ~truth.column("transient").to_numpy(zero_copy_only=False)
    prn = truth.column("prn").to_numpy()
    e = truth.column("e_cyc").to_numpy()
    return {int(p): float(np.var(e[ok & (prn == p)])) for p in np.unique(prn[ok])}


def _kappa(bw_hz: float, fs_hz: float) -> float:
    """SSC of flat noise of width bw_hz against the C/A spectrum, both inside the receiver band +-fs/2."""
    f = np.linspace(-fs_hz / 2, fs_hz / 2, 400_001)
    return ssc_kappa(f, np.where(np.abs(f) <= bw_hz / 2, 1.0, 0.0), ca_psd(f))


def integration_mismatch(a: Twin, b: Twin, *, after_s: float) -> list[int]:
    """Channels whose records after after_s come at another spacing in one twin than in the other: in a 20 ms cell a
    channel that missed bit sync stays at 1 ms, so the pair compares two loops (P2b task 10)."""

    def step(tr: pa.Table) -> float:
        t = tr.column("t_rx_s").to_numpy()
        t = t[t >= after_s]
        return float(np.median(np.diff(t))) if t.size > 1 else float("nan")

    out = []
    for ch in sorted(set(a.tracks) & set(b.tracks)):
        sa, sb = step(a.tracks[ch]), step(b.tracks[ch])
        if not (np.isfinite(sa) and np.isfinite(sb) and abs(sa - sb) <= 0.25 * min(sa, sb)):
            out.append(ch)
    return out


def _tone_identification(clean: Twin, tone_twin: Twin, tones: Disturbance, warm_abs: float, real: Sequence[int],
                         cell: LoopCell, slip: float) -> dict[str, Any]:  # fmt: skip
    """G-hz on real twins: H_phi from the tone twin minus the clean twin at every record, averaged over the real
    satellites, against the cell's source-timing model (docs/9 section 5.4). Only tones below the cell's Nyquist
    frequency 1 / (2 T) enter the lock-in; the others alias, and are listed as above_nyquist."""
    freqs, amps, phases = tone_set(tones)
    below = freqs < 0.5 / cell.t_s
    mismatched = integration_mismatch(clean, tone_twin, after_s=warm_abs)
    tone_in = -1j * amps[below] * np.exp(1j * phases[below])
    onset_s = float(tones.onset_s)
    meas, sds, n_rec = [], [], 0
    for ch in real:
        if ch not in tone_twin.tracks or ch not in clean.tracks or ch in mismatched:
            continue
        a = track_as_obs(clean.tracks[ch], source_id="rec", receiver_id="clean")
        b = track_as_obs(tone_twin.tracks[ch], source_id="rec", receiver_id="tones")
        truth, _, _ = twin_difference(a, b, onset_s=onset_s, pre_window_s=TONE_PRE_S, warmup_s=warm_abs,
                                      slip_size_cyc=slip, tau_loop_s=0.03, truth_id="tones")  # fmt: skip
        t = truth.column("t_rx_s").to_numpy()
        sel = t >= onset_s + 5.0
        if not sel.any() or t[sel][-1] - t[sel][0] < TONE_MIN_SPAN_S:
            continue
        lp = lockin(2 * np.pi * truth.column("e_cyc").to_numpy()[sel], t[sel], list(freqs[below]))
        meas.append(lp[:-1] / tone_in)
        sds.append(float(lp[-1].real))
        n_rec = int(sel.sum())
    if not meas:
        return {"identified": False, "passed": False, "integration_mismatch": mismatched}
    m = np.asarray(np.mean(meas, axis=0), dtype=np.complex128)
    sd = float(np.sqrt(np.mean(np.square(sds))) / np.sqrt(len(meas)))
    resp = cell.loop.freq_response(2 * np.pi * freqs[below] * cell.t_s)
    off_s, resid_deg, rows, passed = stamp_and_verdict(m, resp, cell.t_s, freqs[below], sd, n_rec,
                                                       tone_amp_rad=amps[below])  # fmt: skip
    above = [{"f_hz": float(f), "snr_db": float("nan"), "err_db": float("nan"), "err_deg": float("nan"),
              "status": "above_nyquist"} for f in freqs[~below]]  # fmt: skip
    return {
        "identified": True,
        "channels": len(meas),
        "stamp_offset_T": off_s / cell.t_s,
        "rms_phase_resid_deg": resid_deg,
        "passed": passed,
        "integration_mismatch": mismatched,
        "rows": [*rows, *above],
    }
