"""E2post: the paired decomposition of a finished E2 run, and gate A's parts A1 and A2 (P2b part 2, task 1; the plan's
pre-registered rules).

E2 on segment A read Var_B / Var_A of the ghosts' twin difference at 1.2-1.4, which blocked J_max. Part of mode B's
dPhi may only reshuffle the clean twin's own error; the paired excess Var(dPhi) + 2 Cov(e_c, dPhi) =
Var(e_jammed) - Var(e_clean) keeps what jamming adds and cancels that part (e2_frontend).

The source run is read, never written. Its manifest gives E2's spec; E2's own helpers rebuild the segment (the ghosts'
specs, the twin clock) from a copy of its cached survey in this run's directory (with a cache it found invalid, E2's
survey helper would redo the survey: that is refused before it is called). Every unit read must still hash as its
done.json records.

Per family, level and ghost, on epochs at or after onset + 2 s that are valid and not transient in every series,
joined on exact record times (no interpolation; the records dropped are counted):
- dPhi_A and dPhi_B: the ghost's twin difference in the grid's mode A and mode B units (one jammer draw);
- e_c(A) and e_c(B): the clean twins' absolute ghost errors, ghost truth on the tracks of the units the grid differences
  against, converged after half the warm-up as in E0b;
- Var(dPhi), 2 Cov(e_c, dPhi) and the excess per mode, and the jammer slope of B on A.
Per cell: R = sum_g excess_B / sum_g excess_A, its block-bootstrap SE and the cell verdict (reported here; J_max is
E2's to decide). Gate A1 and A2 as the plan states them, in summary.json["gate_A"].

B1's decomposition (task 3; the spec's `b1` block), per mode A grid unit and ghost, in summary.json["b1"]:
- r recomputed from the unit's saved e_cyc and pred_cyc on E2's epochs (onset + 2 s on, valid, not transient), which
  must be E2's saved r within the spec's r_tol: E2post reads what E2 computed or stops;
- n_iota (rad) per record of the clean twin, rendered again: the unit's interference alone (its disturbances from the
  run's spec and the jammer seed E2 recorded for the unit, mode A's front end; the recording is not read), despread by
  the ghost. E2 projected the jammed-minus-clean bytes instead, which differ by the twins' int16 rounding; E2's
  prediction rebuilt from this n_iota is compared with the saved pred_cyc, and reported;
- Var(n_iota) on r's records, the thermal floor 1 / (C/N0 T) at the clean ghost's median C/N0, and r against the
  formula floor + Var(n_iota) of docs/3 section 2.2;
- post hoc, reported only (summary["b1_post_hoc_note"]): r against a shift of the prediction by up to 5 records,
  where n_iota's power sits against the loop (eta_n), and r decomposed record by record with the clean twin's own
  prompts (the thermal x interference products and the second order), the evidence of task 3's chirp investigation.
"""

from __future__ import annotations

import json
import multiprocessing
import shutil
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy import signal

from gnssaudit.experiments import Outcome, e2
from gnssaudit.experiments.e0b import DEFAULT_BW_HZ, T_S
from gnssaudit.experiments.e2_frontend import (
    FloatArray,
    cell_verdict,
    front_end_cell,
    jammer_slope,
    pair_on_time,
    paired_excess,
)
from gnssaudit.experiments.twin import jsonable
from gnssaudit.inject import Disturbance, FrontEndSpec, GhostSpec, IQSource, gain
from gnssaudit.inject.pipeline import interference
from gnssaudit.paths import results_root
from gnssaudit.truth.ghost import ghost_despread, ghost_truth
from gnssaudit.truth.rxtime import TwinClock
from gnssaudit.uq.loop import CarrierLoop

Series = tuple[FloatArray, FloatArray]  # (t_rx_s, e_cyc) on the usable post-onset epochs
ComplexArray = NDArray[np.complex128]
MAX_LAG = 5  # records either way in B1's lag scan
SKIP_RECORDS = 2000  # the loop's start-up transient, left out of eta_n (2 s at T = 1 ms)
WELCH_RECORDS = 4096  # Welch segment for n_iota's spectrum (0.24 Hz bins at 1 kHz)
LOW_HZ = 50.0  # n_iota's power share below this (the loop band, B_n 35 Hz)
POST_HOC_NOTE = (
    "b1[fam][lv][ch]['post_hoc'] is post hoc (P2b part 2, task 3, after E2's numbers were seen): the lag scan, "
    "n_iota's spectrum and the record-level decomposition of r. Reported only: no gate or rule reads them."
)


def open_source(src: Path, run_dir: Path, key: str) -> tuple[e2.Context, list[Path]]:
    """The finished E2 run at `src` as E2's Context with segment `key`, and the files read. Nothing is written under
    `src`: the survey helper reads a copy of the cached survey in run_dir, made afresh at every start. A run whose
    summary records a failed gate, or no gates, is refused: with abort_on_gate_failure false, E2 finishes such a run
    with a manifest."""
    s, r = src.resolve(), run_dir.resolve()
    if s == r or s in r.parents or r in s.parents:
        raise ValueError("E2post never writes into the run it reads")
    manifest = src / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"{src} has no manifest.json: E2post reads finished E2 runs only")
    m = json.loads(manifest.read_text())
    gates = m["summary"].get("gates")
    if gates is None:
        raise ValueError(f"{src}: its summary records no gates: E2post reads finished E2 runs only")
    failed = sorted(g for g, v in gates.items() if not v["passed"])
    if failed:
        raise ValueError(f"{src}: gates failed ({', '.join(failed)}): a run whose gates failed is not read")
    spec, describe = m["spec"], str(m["git_describe"])
    ctx = e2.Context(spec, run_dir, run_dir / "work", describe, None, {}, {}, {}, {})
    name = f"survey-{key}"
    dst = ctx.unit_dir(name)
    shutil.rmtree(dst, ignore_errors=True)  # an earlier start's copy
    dst.mkdir(parents=True)
    for f in ("survey.json", "done.json"):
        shutil.copyfile(src / "units" / name / f, dst / f)
    if not e2.is_done(dst, describe):
        raise ValueError(f"{name}: the cached survey does not hash as its done.json records under {describe}")
    ctx.segments[key] = e2._segment_survey(ctx, key)
    return ctx, [manifest, src / "units" / name / "survey.json"]


def usable(t: FloatArray, e: FloatArray, valid: Any, transient: Any, onset: float) -> Series:
    ok = e2._post(t, onset) & np.asarray(valid, dtype=bool) & ~np.asarray(transient, dtype=bool)
    return t[ok], e[ok]


def gate_a(cells: Mapping[str, Mapping[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    """A1: every ghost's slope in [slope_min, slope_max] for the gate's families at the A1 levels. A2: for each gate
    family at the A2 level, pooled over ghosts, -(2Cov_B - 2Cov_A) >= cancel_share x (Var_B - Var_A)."""
    a1, a2 = gate["a1"], gate["a2"]
    slopes = {f"{fam}-{float(lv):g}-ch{ch}": float(g["slope"]) for fam in gate["families"] for lv in a1["levels_db_hz"]
              for ch, g in cells[fam][f"{float(lv):g}"]["ghosts"].items()}  # fmt: skip
    ok1 = bool(slopes) and all(float(a1["slope_min"]) <= v <= float(a1["slope_max"]) for v in slopes.values())
    pooled = {}
    for fam in gate["families"]:
        gs = list(cells[fam][f"{float(a2['level_db_hz']):g}"]["ghosts"].values())
        cancel = -(sum(g["cov2_b"] for g in gs) - sum(g["cov2_a"] for g in gs))
        extra = sum(g["var_b"] for g in gs) - sum(g["var_a"] for g in gs)
        pooled[fam] = {"cancel": cancel, "extra_var": extra, "holds": cancel >= float(a2["cancel_share"]) * extra}
    return {"A1": ok1, "A2": all(v["holds"] for v in pooled.values()), "a1_slopes": slopes, "a2": pooled}


def project(source: IQSource, ds: list[Disturbance], fe: FrontEndSpec, n0: float, block: int, ghost: GhostSpec,
            stamps: Any, clock: TwinClock, amplitude: float) -> ComplexArray:  # fmt: skip
    """iota / (A N) of one ghost in one unit, whole (its imaginary part is B1's n_iota, rad): the unit's interference
    alone, rendered from its disturbances through the front end (the recording is not read), despread by the ghost
    over each record of `stamps`."""
    return ghost_despread(interference(source, ds, fe, n0_per_hz=n0, block=block), ghost, stamps, clock, amplitude)


# ------------------------------------------------------------------- B1 post hoc (task 3; reported only)
def lag_scan(e: FloatArray, p: FloatArray, records: Any, *, max_lag: int) -> dict[str, Any]:
    """r with the prediction shifted by lag records, q_k = p_(k - lag) (+ = the prediction delayed), on the epochs given
    less max_lag at either end; the lag of the least r; corr(e, p) at lag 0 and its sign. A timing error of the
    prediction moves the minimum off 0, a sign error makes the correlation negative. `records` is each epoch's record
    index: a shift of the epochs is a shift by records only if they skip none, so a gap is refused."""
    if np.any(np.diff(np.asarray(records, dtype=np.int64)) != 1):
        raise ValueError("the lag scan needs epochs on consecutive records: these skip a record")
    lags = list(range(-max_lag, max_lag + 1))
    core = slice(max_lag, e.size - max_lag)
    rs = [float(np.var(e[core] - np.roll(p, lag)[core]) / np.var(e[core])) for lag in lags]
    c = float(np.corrcoef(e, p)[0, 1])
    return {"lags": lags, "r": rs, "argmin_lag": lags[int(np.argmin(rs))], "corr_lag0": c,
            "corr_sign_lag0": int(np.sign(c))}  # fmt: skip


def _eta(nq: FloatArray, loop: CarrierLoop) -> float:
    """Var(H n_Q) / Var(n_Q) on a record series, its mean removed and the loop's start-up left out."""
    x = nq - np.mean(nq)
    return float(np.var(loop.filter(x)[SKIP_RECORDS:]) / np.var(x[SKIP_RECORDS:]))


def n_iota_spectrum(n: ComplexArray, on: Any, loop: CarrierLoop) -> dict[str, float]:
    """Where n_iota's power sits against the loop, on the records `on` (taken as one series): eta_n = Var(H n_Q) /
    Var(n_Q) (H the loop E2's prediction uses), against eta_w, its value for white input (noise_gain); the share of
    n_Q's power below LOW_HZ; the frequency at which the complex n's spectrum peaks (Welch, two-sided)."""
    fs = 1.0 / loop.t_s
    z = n[on] - np.mean(n[on])
    f, pqq = signal.welch(z.imag, fs, nperseg=WELCH_RECORDS)
    fc, pc = signal.welch(z, fs, nperseg=WELCH_RECORDS, return_onesided=False)
    share = float(np.sum(pqq[f < LOW_HZ]) / np.sum(pqq))
    return {"eta_n": _eta(n.imag[on], loop), "eta_w": loop.noise_gain(), "share_below_50_hz": share,
            "peak_hz": float(fc[np.argmax(pc)])}  # fmt: skip


def record_prompts(prompt_i: Any, prompt_q: Any) -> ComplexArray:
    """Record k's prompt P = sum x conj(replica) over [stamp_k, stamp_k+1): GNSS-SDR dumps it with record k+1 (read off
    E2-A in task 3: aligned so, the thermal x interference prediction removes most of B1's residual; one record either
    way, or conjugated, it adds variance). The last record's is not dumped: NaN."""
    p = np.asarray(prompt_i, dtype=np.float64) + 1j * np.asarray(prompt_q, dtype=np.float64)
    return np.r_[p[1:], np.nan + 1j * np.nan]


def normalised_prompts(prompts: ComplexArray, on: Any) -> tuple[ComplexArray, float]:
    """(P~, A^): the prompts with the data bit wiped by the sign of their in-phase part, over A^, the mean in-phase
    amplitude on the records `on`: P~ = e^(j dphi) (1 + w~), the ghost at 1 and the thermal noise the twins share."""
    sg = np.where(prompts.real < 0, -1.0, 1.0)
    a_hat = float(np.mean((prompts.real * sg)[on]))
    return prompts * sg / a_hat, a_hat


def _costas(z: ComplexArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.arctan(z.imag / z.real)  # GNSS-SDR's 1C PLL discriminator: two-quadrant, atan(Q / I)


def decomposition(e: FloatArray, predict: Callable[[FloatArray], FloatArray], n: ComplexArray, prompts: ComplexArray,
                  g: float, on: Any, *, cn0_dbhz: float, loop: CarrierLoop) -> dict[str, float]:  # fmt: skip
    """B1's residual, record by record. The loop is linear, so dPhi = H delta with delta_k = atan2q(P~ + g n) -
    atan2q(P~) (atan2q the Costas atan(Q / I); P~ the clean twin's normalised prompt, holding the thermal noise the
    jammed twin shares; g n the interference at the prompt's ghost amplitude); `predict` maps a per-record
    discriminator input (rad) to dPhi on e's epochs (E2's steps). Predictions and their r:
    - b1: H n_Q (E2's first order, gain 1);
    - lin: H Im(g n / P~), first order in n, every order in the thermal noise;
    - exact: H delta, every order.
    cross_share = Var(lin - H g n_Q) / Var(dPhi) is the thermal x interference products. Their closed form: with
    P~ = 1 + w (w = w_I + j w_Q circular Gaussian, s2 = Var w_I = Var w_Q) and u = 1 / (1 + w) - 1 = -w + w^2 - ...,
    lin - g n_Q = g Im(n u) = g (n_I u_Q + n_Q u_I). For circular w, E[w^a conj(w)^b] = 0 unless a = b, so E u = 0
    and E u^2 = 0 (u is proper: E u_I^2 = E u_Q^2 = E|u|^2 / 2, E u_I u_Q = 0), and E|u|^2 = sum_m E|w|^(2m) =
    2 s2 + 8 s2^2 + 48 s2^3 + ... (E|w|^(2m) = m! (2 s2)^m): E u_I^2 = s2 (1 + 4 s2 + 24 s2^2 + ...). n is independent
    of w and the u_k are i.i.d., so the products are white, with variance E u_I^2 (Var n_I + Var n_Q), and uncorrelated
    with H g n_Q. Through H they keep eta_w of it and g n_Q keeps eta_n, so X = Var(H g Im(n u)) / Var(H g n_Q) =
    E u_I^2 (1 + Var n_I / Var n_Q) eta_w / eta_n, share X / (1 + X) of dPhi with the second order neglected. The key
    `x` is that with E u_I^2 = s2, first order (s2 = Var w_I measured, for both parts: Var w_Q holds the clean
    tracking error too). The 4 s2 term raises X by 1 + 4 s2 (10 % at FGI's s2 = 0.024-0.027); the 24 s2^2 term,
    neglected, adds 1.4-1.7 % more. A Monte Carlo of E u_I^2 (task 3, fix round 1) gives s2 x 1.116 at s2 = 0.0245,
    against 1.098 with 4 s2, 1.112 with 24 s2^2 too and 1.115 with the next term (192 s2^3). The series is
    asymptotic: E|u|^2 itself diverges at w = -1, a region of probability ~exp(-1 / (2 s2)) ~ 1e-8 here. The unit tests
    check X / (1 + X) at s2 = 0.002 and the 1 + 4 s2 form at 0.0245 on a synthetic discriminator.
    second_share = Var(exact - lin) / Var(dPhi). Var w per component is reported against 1 / (2 C/N0 T) at
    `cn0_dbhz`."""
    nz = np.nan_to_num(n)
    with np.errstate(divide="ignore", invalid="ignore"):  # the last record has no prompt (NaN): it predicts 0
        x_lin = np.imag(g * nz / prompts)
    pb1, pg, plin, pex = (
        predict(x) for x in (nz.imag, g * nz.imag, x_lin, _costas(prompts + g * nz) - _costas(prompts))
    )
    ve = float(np.var(e))
    w = prompts[on] - 1.0
    var_wi, var_wq = float(np.var(w.real)), float(np.var(w.imag))
    thermal = 0.5 / (10 ** (cn0_dbhz / 10) * loop.t_s)
    vq, vi = float(np.var(n.imag[on])), float(np.var(n.real[on]))
    x = var_wi * (1.0 + vi / vq) * loop.noise_gain() / _eta(n.imag[on], loop)
    return {"var_w_i": var_wi, "var_w_q": var_wq, "var_w_i_over_thermal": var_wi / thermal,
            "var_w_q_over_thermal": var_wq / thermal, "g": g, "x": x, "x_over_1px": x / (1.0 + x),
            "cross_share": float(np.var(plin - pg)) / ve, "second_share": float(np.var(pex - plin)) / ve,
            "r_b1": float(np.var(e - pb1)) / ve, "r_lin": float(np.var(e - plin)) / ve,
            "r_exact": float(np.var(e - pex)) / ve}  # fmt: skip


def recorded_unit(planned: e2.Unit, rec: Mapping[str, Any]) -> e2.Unit:
    """The unit as E2 recorded it in its unit.json. E2's first run keyed a cell's jammer seed on the order of the
    spec's `families` (e2._cell until P2b part 2, task 5), which the manifest's sorted JSON does not keep, so a plan
    rebuilt from the manifest can carry another cell's seed: the recorded seed is the one E2 drew. Everything else must
    be the plan's, or the run is refused."""
    u = e2.Unit(**rec)
    if replace(u, seed=planned.seed) != planned:
        raise ValueError(f"{planned.name}: E2's unit record {rec} is not the planned unit {planned}")
    return u


def b1_cells(spec: Mapping[str, Any], ctx: e2.Context, seg: e2.Segment, grid: Mapping[tuple[Any, ...], e2.Unit],
             unit_file: Callable[[str, str], Path], onset: float) -> dict[str, Any]:  # fmt: skip
    """summary["b1"]: per family, level and ghost channel of the spec's `b1` block, from the mode A grid unit."""
    b1 = spec["b1"]
    pre, tol = float(ctx.spec["pre_window_s"]), float(b1["r_tol"])
    jobs: dict[tuple[str, float, int], tuple[Any, ...]] = {}
    clean: dict[tuple[str, float, int], tuple[FloatArray, float, ComplexArray, float]] = {}
    recs: dict[tuple[str, float], dict[str, Any]] = {}
    for fam in b1["families"]:
        for lv in (float(v) for v in b1["levels_db_hz"]):
            planned = grid[fam, lv, "A"]
            recs[fam, lv] = json.loads(unit_file(planned.name, "unit.json").read_text())
            u = recorded_unit(planned, recs[fam, lv]["unit"])
            for ch, gd in zip(seg.ghost_ch, seg.ghosts, strict=True):
                track = unit_file(u.clean or "", f"track_ch{ch}.parquet")
                tr = pq.read_table(track, columns=["sample_count", "cn0_dbhz", "prompt_i", "prompt_q"])
                stamps = tr.column("sample_count").to_numpy()
                n_rec = float(np.median(np.diff(stamps.astype(np.int64))))  # output samples per record
                clean[fam, lv, ch] = (seg.clock.file_time(stamps), float(np.median(tr.column("cn0_dbhz").to_numpy())),
                                      record_prompts(*(tr.column(c).to_numpy() for c in ("prompt_i", "prompt_q"))),
                                      gain(gd.level_db_hz, seg.n0) * n_rec * seg.scale)  # fmt: skip
                jobs[fam, lv, ch] = (seg.source, e2.disturbances(ctx, u), e2.front_end(ctx, u), seg.n0,
                                     int(b1["block"]), GhostSpec.of(gd), stamps, seg.clock,
                                     gain(gd.level_db_hz, seg.n0))  # fmt: skip
    workers = int(b1.get("workers", 1))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {k: pool.submit(project, *a) for k, a in jobs.items()}
            n_iota = {k: f.result() for k, f in futures.items()}
    else:
        n_iota = {k: project(*a) for k, a in jobs.items()}
    loop = CarrierLoop.gnsssdr(DEFAULT_BW_HZ, T_S)
    out: dict[str, Any] = {}
    for (fam, lv, ch), nc in n_iota.items():
        n = np.ascontiguousarray(nc.imag)
        planned, seed = grid[fam, lv, "A"], int(recs[fam, lv]["unit"]["seed"])
        prn = int(seg.channels[ch][2])
        tr = pq.read_table(unit_file(planned.name, f"ghost_ch{ch}.parquet"),
                           columns=["t_rx_s", "e_cyc", "pred_cyc", "valid", "transient"])  # fmt: skip
        t, e, p = (tr.column(c).to_numpy() for c in ("t_rx_s", "e_cyc", "pred_cyc"))
        sel = e2._post(t, onset) & tr.column("valid").to_numpy(zero_copy_only=False)
        sel &= ~tr.column("transient").to_numpy(zero_copy_only=False)
        rec = recs[fam, lv]["residual"].get(str(prn))
        if rec is None:
            raise ValueError(f"{planned.name}: E2 saved no residual for PRN {prn}")
        r = e2.residual_ratio(e[sel], p[sel])
        if not abs(r - float(rec["r"])) <= tol:
            raise ValueError(f"{planned.name} PRN {prn}: r from the saved e_cyc and pred_cyc is {r!r}, E2 saved "
                             f"{rec['r']!r} (tolerance {tol:g})")  # fmt: skip
        t_rec, cn0, prompts, ans = clean[fam, lv, ch]
        # n_iota on r's records: the truth's records are the clean twin's, joined on the record time to 1 ns
        keys = np.rint(t_rec * 1e9).astype(np.int64)
        on = np.isin(keys, np.rint(t[sel] * 1e9).astype(np.int64)) & np.isfinite(n)
        var_n = float(np.var(n[on]))
        floor = e2.thermal_floor(cn0, T_S)
        d = e2.b1_prediction(n, t_rec, t, onset, pre)[sel] - p[sel]
        pt, a_hat = normalised_prompts(prompts, on)

        def predict(x: FloatArray, t_rec: FloatArray = t_rec, t: FloatArray = t, sel: Any = sel) -> FloatArray:
            return e2.b1_prediction(x, t_rec, t, onset, pre)[sel]

        post_hoc = {
            "lag_scan": lag_scan(
                e[sel], p[sel], np.searchsorted(keys, np.rint(t[sel] * 1e9).astype(np.int64)), max_lag=MAX_LAG
            ),
            "spectrum": n_iota_spectrum(nc, on, loop),
            "decomposition": decomposition(e[sel], predict, nc, pt, ans / a_hat, on, cn0_dbhz=cn0, loop=loop),
        }
        out.setdefault(fam, {}).setdefault(f"{lv:g}", {})[str(ch)] = {
            "prn": prn, "n": int(sel.sum()), "r": r, "r_saved": float(rec["r"]), "r_diff": abs(r - float(rec["r"])),
            "cn0_clean_dbhz": cn0, "floor": floor, "var_n_iota_rad2": var_n, "n_iota_records": int(on.sum()),
            "formula": floor + var_n, "r_minus_formula": r - (floor + var_n), "r_minus_floor": r - floor,
            "corr": float(np.corrcoef(e[sel], p[sel])[0, 1]), "corr_saved": float(rec["corr"]),
            "pred_max_abs_diff_cyc": float(np.max(np.abs(d))), "pred_rms_diff_cyc": float(np.sqrt(np.mean(d**2))),
            "pred_sd_cyc": float(np.std(p[sel])), "jammer_seed": seed, "planned_seed": planned.seed,
            "post_hoc": post_hoc,
        }  # fmt: skip
    return out


def run_e2post(spec: dict[str, Any], run_dir: Path) -> Outcome:
    src = results_root() / str(spec["source_run"])
    key = str(spec["segment"])
    ctx, inputs = open_source(src, run_dir, key)
    seg = ctx.segments[key]
    if [int(c) for c in spec["ghost_channels"]] != seg.ghost_ch:
        raise ValueError(f"the spec's ghost channels {spec['ghost_channels']} are not the run's {seg.ghost_ch}")
    onset = seg.t0 + float(ctx.spec["onset_s"])
    ghosts = {ch: GhostSpec.of(d) for ch, d in zip(seg.ghost_ch, seg.ghosts, strict=True)}
    grid = {(u.family, u.level, u.mode): u for u in e2.plan_grid(ctx.spec, key) if u.block == "grid"}
    checked: set[str] = set()

    def unit_file(name: str, f: str) -> Path:
        udir = src / "units" / name
        if name not in checked:
            if not e2.is_done(udir, ctx.describe):
                raise ValueError(f"{name}: not a finished unit of {src} (no done.json, or files that no longer hash "
                                 f"as recorded)")  # fmt: skip
            checked.add(name)
        if not (udir / f).is_file():
            raise ValueError(f"{name} has no {f} (a ghost lost in that twin)")
        inputs.append(udir / f)
        return udir / f

    cleans: dict[tuple[str, int], Series] = {}

    def clean_error(name: str, ch: int) -> Series:
        if (name, ch) not in cleans:
            track = pq.read_table(unit_file(name, f"track_ch{ch}.parquet"))
            tr = ghost_truth(track, ghosts[ch], clock=seg.clock, converge_s=ctx.warm / 2, truth_id=f"{name}-g{ch}")
            cleans[name, ch] = usable(tr.column("t_rx_s").to_numpy(), tr.column("e_cyc").to_numpy(),
                                      tr.column("valid").to_numpy(zero_copy_only=False),
                                      tr.column("transient").to_numpy(zero_copy_only=False), onset)  # fmt: skip
        return cleans[name, ch]

    def twin_diff(name: str, ch: int) -> Series:
        tr = pq.read_table(unit_file(name, f"ghost_ch{ch}.parquet"), columns=["t_rx_s", "e_cyc", "valid", "transient"])
        return usable(tr.column("t_rx_s").to_numpy(), tr.column("e_cyc").to_numpy(),
                      tr.column("valid").to_numpy(zero_copy_only=False),
                      tr.column("transient").to_numpy(zero_copy_only=False), onset)  # fmt: skip

    boot, rules = spec["bootstrap"], spec["rules"]
    block = round(float(boot["block_s"]) / T_S)
    cells: dict[str, dict[str, Any]] = {}
    dropped_pairing = dropped_aligning = 0
    for fam in spec["families"]:
        for lv in (float(v) for v in spec["levels_db_hz"]):
            ua, ub = grid[fam, lv, "A"], grid[fam, lv, "B"]
            per: dict[str, dict[str, float | int]] = {}
            paired = []
            for ch in seg.ghost_ch:
                t, (da, db, ca, cb), dropped = pair_on_time(twin_diff(ua.name, ch), twin_diff(ub.name, ch),
                                                            clean_error(ua.clean or "", ch),
                                                            clean_error(ub.clean or "", ch))  # fmt: skip
                xa, xb = paired_excess(da, ca), paired_excess(db, cb)
                per[str(ch)] = {"n": int(t.size), "dropped": dropped, "var_a": xa.var_dphi, "var_b": xb.var_dphi,
                                "cov2_a": xa.cov2, "cov2_b": xb.cov2, "excess_a": xa.excess, "excess_b": xb.excess,
                                "slope": jammer_slope(da, db)}  # fmt: skip
                paired.append((t, da, db, ca, cb))
                dropped_pairing += dropped
            c, dropped = front_end_cell(paired, block=block, tol_s=T_S / 2, n_boot=int(boot["n_boot"]),
                                        seed=int(boot["seed"]))  # fmt: skip
            dropped_aligning += dropped
            verdict = cell_verdict(c, tol=float(rules["tol"]), slope_tol=float(rules["slope_tol"]),
                                   se_max=float(rules["se_max"]),
                                   cell=f"segment {key}, {fam} at {lv:g} dB-Hz")  # fmt: skip
            cells.setdefault(fam, {})[f"{lv:g}"] = {"ghosts": per, "ratio": c.ratio, "ratio_se": c.ratio_se,
                                                    "verdict": verdict, "dropped_aligning": dropped}  # fmt: skip
    summary = {
        "source_run": str(spec["source_run"]),
        "source_describe": ctx.describe,
        "segment": key,
        "ghosts": {str(ch): {"prn": g.prn, "doppler_hz": g.doppler_hz} for ch, g in ghosts.items()},
        "onset_s": onset,
        "settle_s": e2.SETTLE_S,
        "converge_s": ctx.warm / 2,
        "block_records": block,
        "cells": cells,
        "gate_A": gate_a(cells, spec["gate_a"]),
    }
    if "b1" in spec:
        summary["b1"] = b1_cells(spec, ctx, seg, grid, unit_file, onset)
        summary["b1_post_hoc_note"] = POST_HOC_NOTE
    (run_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    counts = {"cells": sum(len(v) for v in cells.values()), "records_dropped_pairing": dropped_pairing,
              "records_dropped_aligning": dropped_aligning}  # fmt: skip
    return Outcome(inputs=list(dict.fromkeys(inputs)), counts=counts, summary=jsonable(summary))
