"""Pipeline gates (docs/11 section 5): each returns a GateReport, and `require` turns a failed one into GateFailure,
which aborts the run (docs/6 section 8.4).

GNSS-SDR v0.0.22 starts tracking at a different sample on every run of identical input (F24), so the identity,
null and repeat gates are tolerance gates (D-33): after warm-up, twin and repeat differences must stay
below a fraction of the thermal SD, while injector output must still be identical byte for byte. G-hz judges an
identified loop response against its model (docs/9 section 5.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.inject import InjectionReport
from gnssaudit.io.schemas import EXCLUSIONS, TRACK_STREAM, TRUTH_EPOCH_SAT, validate

FloatArray = NDArray[np.float64]
IDENTITY_TOL = 0.10  # pre-onset twin difference of d over its thermal SD (docs/3 section 2.1, amended F24)
SYNC_MIN = 0.9  # share of post-warm-up records the twins hold at the same stamps
GHZ_SNR_MIN_DB = 20.0  # G-hz (docs/11): tones whose response SNR is below this are not judged
GHZ_MAX_DB = 0.3
GHZ_MAX_DEG = 5.0


@dataclass(frozen=True)
class GateReport:
    gate: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict[str, Any])


class GateFailure(RuntimeError):
    """A pipeline gate failed; the run is aborted (docs/6 section 8.4)."""

    def __init__(self, report: GateReport) -> None:
        super().__init__(f"gate {report.gate} failed: {report.details}")
        self.report = report


def require(report: GateReport) -> GateReport:
    if not report.passed:
        raise GateFailure(report)
    return report


def _by_prn(track: pa.Table, t_min_s: float, t_max_s: float) -> dict[int, tuple[NDArray[np.uint64], FloatArray]]:
    track = validate(track, TRACK_STREAM)
    t = track.column("t_rx_s").to_numpy()
    sel = (t >= t_min_s) & (t < t_max_s)
    prn = track.column("prn").to_numpy()[sel]
    stamps = track.column("sample_count").to_numpy()[sel]
    d = track.column("carr_err").to_numpy(zero_copy_only=False)[sel].astype(np.float64)
    return {int(p): (stamps[prn == p], d[prn == p]) for p in np.unique(prn)}


def _agreement(a: pa.Table, b: pa.Table, t_min_s: float, t_max_s: float) -> tuple[dict[int, float], dict[int, float]]:
    """Per PRN: share of records at common stamps, and RMS of the discriminator difference there (cycles)."""
    ca, cb = _by_prn(a, t_min_s, t_max_s), _by_prn(b, t_min_s, t_max_s)
    share, rms = {}, {}
    for p in sorted(set(ca) | set(cb)):
        sa, da = ca.get(p, (np.zeros(0, np.uint64), np.zeros(0)))
        sb, db = cb.get(p, (np.zeros(0, np.uint64), np.zeros(0)))
        common, ia, ib = np.intersect1d(sa, sb, return_indices=True)
        share[p] = common.size / max(sa.size, sb.size, 1)
        rms[p] = float(np.sqrt(np.mean((db[ib] - da[ia]) ** 2))) if common.size else float("inf")
    return share, rms


def identity_gate(
    clean_track: pa.Table,
    jammed_track: pa.Table,
    *,
    onset_s: float,
    warmup_s: float,
    sigma_th_cyc: float,
    tol: float = IDENTITY_TOL,
) -> GateReport:
    """G-identity (tolerance form): between warm-up and onset (the tracks' time base, recording time in the twin
    pipeline), both twins integrate on the same samples (at least 90 % of records at common stamps) and their
    discriminators differ by at most tol thermal SDs (RMS)."""
    share, rms = _agreement(clean_track, jammed_track, warmup_s, onset_s)
    ratio = {p: v / sigma_th_cyc for p, v in rms.items()}
    passed = bool(ratio) and all(share[p] >= SYNC_MIN and ratio[p] <= tol for p in ratio)
    return GateReport("G-identity", passed, {"common_fraction": share, "rms_over_sigma": ratio, "tol": tol})


# Exclusions that mean the pair broke (a step that is no slip, a receiver-time reset, a re-acquisition): a zero-gain
# pair must have none, whatever its kept epochs show (p2a review, critical 1)
BREAKING = frozenset({"ambiguous_slip", "after_ambiguous_step", "rx_time_break", "reacquisition",
                      "no_pre_onset_reference"})  # fmt: skip
NULL_MIN_KEPT = 0.99  # share of the pair's post-warm-up epochs that must be kept


def null_gate(
    null_truth: pa.Table, exclusions: pa.Table, *, sigma_th_cyc: float, tol: float = IDENTITY_TOL
) -> GateReport:
    """G-null (tolerance form, D-33): the twin difference of a zero-gain jammed twin stays below tol thermal SDs (RMS
    over valid, non-transient epochs), holds no slip, has no breaking exclusion, and keeps at least NULL_MIN_KEPT of
    its epochs after the warm-up."""
    truth = validate(null_truth, TRUTH_EPOCH_SAT)
    excl = validate(exclusions, EXCLUSIONS)
    ok = truth.column("valid").to_numpy(zero_copy_only=False) & ~truth.column("transient").to_numpy(
        zero_copy_only=False
    )
    e = truth.column("e_cyc").to_numpy()[ok]
    slips = int(np.max(truth.column("slip_count").to_numpy(), initial=0))
    ratio = float(np.sqrt(np.mean(e**2)) / sigma_th_cyc) if e.size else float("inf")
    reasons = excl.column("reason_code").to_pylist()
    counts = excl.column("count").to_pylist()
    breaking = sum(c for r, c in zip(reasons, counts, strict=True) if r in BREAKING)
    lost = sum(c for r, c in zip(reasons, counts, strict=True) if r != "warmup")
    kept_share = truth.num_rows / (truth.num_rows + lost) if truth.num_rows + lost else 0.0
    passed = e.size > 0 and ratio <= tol and slips == 0 and breaking == 0 and kept_share >= NULL_MIN_KEPT
    return GateReport("G-null", passed, {"rms_over_sigma": ratio, "slips": slips, "epochs": int(e.size), "tol": tol,
                                         "breaking_exclusions": breaking, "kept_share": kept_share})  # fmt: skip


def repeat_gate(
    reports: Sequence[InjectionReport],
    tracks: Sequence[pa.Table],
    *,
    warmup_s: float,
    sigma_th_cyc: float,
    tol: float = IDENTITY_TOL,
) -> GateReport:
    """G-repeat (tolerance form): repeated injections are byte-identical (SHA-256), and the receiver runs on them
    agree with the first run within the identity tolerance after warm-up."""
    injector_identical = len({r.output_sha256 for r in reports}) == 1
    receiver = []
    for other in tracks[1:]:
        share, rms = _agreement(tracks[0], other, warmup_s, float("inf"))
        receiver.append({p: (share[p], rms[p] / sigma_th_cyc) for p in rms})
    receivers_ok = all(s >= SYNC_MIN and r <= tol for run in receiver for s, r in run.values())
    return GateReport("G-repeat", injector_identical and receivers_ok and len(tracks) > 1,
                      {"injector_identical": injector_identical, "receiver": receiver, "tol": tol})  # fmt: skip


# ---------------------------------------------------------------------------- G-hz (moved from review/s17, P2a)
def lockin(
    y: NDArray[np.floating[Any]], t: NDArray[np.floating[Any]], freqs: Sequence[float]
) -> NDArray[np.complex128]:
    """Least-squares complex amplitudes C of sinusoids y = Re(C exp(j 2 pi f t)) at freqs, plus a cubic trend; the
    last element is the residual SD."""
    tc = (t - t.mean()) / (t.max() - t.min())
    cols = [np.ones_like(t), tc, tc**2, tc**3]
    for f in freqs:
        cols += [np.cos(2 * np.pi * f * t), np.sin(2 * np.pi * f * t)]
    a = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    c = coef[4:].reshape(-1, 2)
    resid = y - a @ coef
    return np.asarray(np.concatenate([c[:, 0] - 1j * c[:, 1], [np.std(resid) + 0j]]), dtype=np.complex128)


class ToneRow(TypedDict):
    f_hz: float
    snr_db: float
    err_db: float
    err_deg: float
    status: str  # "pass", "fail", "excluded_low_snr" or "above_nyquist"


def ghz_verdict(
    meas: NDArray[np.complex128],
    model: NDArray[np.complex128],
    amp_se: float,
    stamp_offset_s: float,
    f_hz: FloatArray,
    t_s: float | None = None,
    *,
    tone_amp_rad: float | FloatArray,
) -> tuple[list[ToneRow], bool]:
    """G-hz (docs/11): +-0.3 dB and +-5 deg at every tone whose response SNR is at least 20 dB.

    meas and model are complex responses at f_hz (measured lock-in ratio; model including the block-average sinc).
    The model is delayed by the fitted stamp offset, model * exp(j 2 pi f stamp_offset_s). Response SNR of a tone is
    20 log10(|meas| tone_amp_rad / amp_se): the output amplitude in rad over the lock-in amplitude standard error.
    Lower-SNR tones are listed as excluded_low_snr; with t_s given, tones at or above 1/(2 t_s) as above_nyquist;
    neither decides. The verdict is False when no tone is judged.
    """
    ref = model * np.exp(2j * np.pi * f_hz * stamp_offset_s)
    ratio = meas / ref
    snr = 20 * np.log10(np.abs(meas) * tone_amp_rad / amp_se)
    err_db = 20 * np.log10(np.abs(ratio))
    err_deg = np.degrees(np.angle(ratio))
    rows: list[ToneRow] = []
    for k, f in enumerate(f_hz):
        if t_s is not None and f >= 0.5 / t_s:
            status = "above_nyquist"
        elif snr[k] < GHZ_SNR_MIN_DB:
            status = "excluded_low_snr"
        elif abs(err_db[k]) <= GHZ_MAX_DB and abs(err_deg[k]) <= GHZ_MAX_DEG:
            status = "pass"
        else:
            status = "fail"
        rows.append(ToneRow(f_hz=float(f), snr_db=float(snr[k]), err_db=float(err_db[k]),
                            err_deg=float(err_deg[k]), status=status))  # fmt: skip
    judged = [r["status"] for r in rows if r["status"] in ("pass", "fail")]
    return rows, bool(judged) and all(st == "pass" for st in judged)


def stamp_and_verdict(
    meas: NDArray[np.complex128],
    resp: NDArray[np.complex128],
    t_s: float,
    f_hz: FloatArray,
    resid_sd_rad: float,
    n_rec: int,
    *,
    tone_amp_rad: float | FloatArray,
) -> tuple[float, float, list[ToneRow], bool]:
    """Stamp-delay fit, lock-in amplitude SE and G-hz verdict for one response (H_phi or H_d).

    resp is the loop's freq_response (without the block-average sinc), meas the lock-in ratio. The delay is fitted on
    the tones with |meas| > 0.15 over +-2 T. amp_se = resid_sd_rad sqrt(2 / n_rec). Returns (offset in seconds, rms
    phase residual in degrees, per-tone rows, passed).
    """
    model = resp * np.sinc(f_hz * t_s)
    ok = np.abs(meas) > 0.15
    grid = np.linspace(-2 * t_s, 2 * t_s, 801)
    costs = [np.sum(np.angle(meas[ok] / (model[ok] * np.exp(2j * np.pi * f_hz[ok] * dl))) ** 2) for dl in grid]
    k = int(np.argmin(costs))
    amp_se = float(resid_sd_rad * np.sqrt(2.0 / n_rec))
    rows, passed = ghz_verdict(meas, model, amp_se, float(grid[k]), f_hz, t_s, tone_amp_rad=tone_amp_rad)
    return float(grid[k]), float(np.degrees(np.sqrt(costs[k] / ok.sum()))), rows, passed
