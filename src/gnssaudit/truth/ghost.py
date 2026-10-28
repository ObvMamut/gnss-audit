"""Ghost-satellite truth and the phase-precision gate (docs/3 sections 2.6 and 3.2).

The ghost's phase is known exactly, so its tracking error is the reported phase minus the true one. GNSS-SDR
accumulates the NCO phase with `-=`, so the reported phase is -acc_phase_rad, read at the PRN start sample stamp,
where the dump's accumulated phase is the NCO phase (docs/9 section 5.4). It needs the patched build's float64 phase
(D-08). The constant ambiguity (an integer, or half a cycle for a Costas loop) is the mean over the first 10 s after
convergence.

GNSS-SDR restarts the accumulated phase from zero when a channel re-acquires after a loss of lock, and sets it again
when narrow tracking begins after bit sync (`check_carrier_phase_coherent_initialization`, dll_pll_veml_tracking.cc,
v0.0.22). Each restart or reset starts a new arc with its own constant. The truth is valid on the first arc with a full
ambiguity window after convergence; later arcs have no disturbance-free window of their own, so they are not valid.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.inject import GhostSpec, ghost_modulation, ghost_phase_rad
from gnssaudit.io.schemas import TRACK_STREAM, TRUTH_EPOCH_SAT, build, validate
from gnssaudit.truth.rxtime import TwinClock

AMBIGUITY_WINDOW_S = 10.0
RESET_STEP_CYC = 0.5  # a one-record step this large is a reset, not tracking: pull-in moves under 0.3 cycle per ms
GAP_SPACINGS = 1.5  # a record gap longer than this many median spacings is a re-acquisition


class PhasePrecisionError(ValueError):
    """A phase-truth column is not float64 or not fine enough (G-phase-precision)."""


class NoAmbiguityWindow(ValueError):
    """No arc of a ghost track holds a full ambiguity window after convergence: the track has no truth. E2's check C-1
    records this case, and only this one, as not computed (P2c task 3)."""


def phase_precision_ok(max_abs_phase: float, dtype: np.dtype, sigma_th: float) -> bool:
    """G-phase-precision: the storage spacing at the largest phase is at most 1e-3 of the thermal sigma."""
    return float(np.spacing(dtype.type(max_abs_phase))) <= 1e-3 * sigma_th


def ghost_truth(track: pa.Table, ghost: GhostSpec, *, clock: TwinClock, converge_s: float, truth_id: str) -> pa.Table:
    """Truth rows (kind 'ghost', e_cyc = tracking error in cycles) for one ghost channel's tracking dump. The ghost's
    phase is evaluated at the native index each stamp stands for (`clock`: the twin's start in the recording, its
    decimation and the filter delay). Rows within `converge_s` of a tracking start are kept as transient and not
    valid; rows outside the arc that holds the ambiguity window are kept and not valid."""
    track = validate(track, TRACK_STREAM)
    acc = track.column("acc_phase_rad")
    if acc.null_count:
        raise PhasePrecisionError("ghost truth needs the float64 accumulated phase of the patched build (D-08)")
    stamps = track.column("sample_count").to_numpy().astype(np.int64)
    t = track.column("t_rx_s").to_numpy()
    native = np.rint(clock.native_index(stamps)).astype(np.int64)  # integer: odd-length FIRs delay whole samples
    e_rad = -acc.to_numpy() - ghost_phase_rad(ghost, native, clock.fs_native_hz)
    d = np.diff(stamps)
    restart = np.r_[True, d > GAP_SPACINGS * np.median(d)] if d.size else np.ones(t.size, dtype=bool)
    arc = np.cumsum(restart | np.r_[False, np.abs(np.diff(e_rad)) > 2 * np.pi * RESET_STEP_CYC]) - 1
    episode_start = t[restart][np.cumsum(restart) - 1]
    converged = t > episode_start + converge_s
    window = np.zeros(t.size, dtype=bool)
    for a in range(int(arc[-1]) + 1):
        ok = converged & (arc == a)
        if not ok.any():
            continue
        end = max(float(t[arc == a][0]), float(episode_start[ok][0]) + converge_s) + AMBIGUITY_WINDOW_S
        if t[ok][-1] > end:
            window = ok & (t <= end)
            break
    if not window.any():
        raise NoAmbiguityWindow(f"{truth_id}: no arc holds a full ambiguity window after {converge_s} s of convergence")
    chosen = arc == arc[window][0]
    e_cyc = (e_rad - np.mean(e_rad[window])) / (2 * np.pi)
    n = t.size
    return build(TRUTH_EPOCH_SAT, {
        "truth_id": [truth_id] * n, "kind": ["ghost"] * n, "sys": track.column("sys"), "prn": track.column("prn"),
        "signal": track.column("signal"), "t_rx_s": t, "e_cyc": e_cyc, "slip_count": np.zeros(n, dtype=np.int16),
        "arc_id": arc.astype(np.int32), "transient": ~converged, "valid": converged & chosen,
        "regime": np.zeros(n, dtype=np.uint8),
    }, n)  # fmt: skip


def ghost_despread(
    interference: Iterable[tuple[int, NDArray[np.complex128]]],
    ghost: GhostSpec,
    stamps: NDArray[np.integer],
    clock: TwinClock,
    amplitude: float,
) -> NDArray[np.complex128]:
    """iota_k / (A N_k) per tracking record, whole (P2b part 2, task 3): iota_k the interference despread by the ghost's
    whole modulation (data bits, code, carrier: a Costas loop sees the bit-wiped prompt) over record k's integration
    [stamp_k, stamp_k+1), in output samples. Its imaginary part is B1's n_iota (rad; `ghost_projection`), its real
    part the in-phase part the discriminator's normalisation sees. `interference` yields (first output index,
    samples) at the twin's output rate and in the injector's scale, whose ghost amplitude is `amplitude`; `clock` maps
    output samples to the native index the ghost was drawn on. A record whose integration the stream does not cover
    whole, and the last record, are NaN."""
    s = np.asarray(stamps, dtype=np.int64)
    k = s.size
    acc = np.zeros(k, dtype=np.complex128)
    cnt = np.zeros(k, dtype=np.int64)
    for m0, y in interference:
        m = np.arange(m0, m0 + y.size, dtype=np.int64)
        idx = np.searchsorted(s, m, side="right") - 1
        ok = (idx >= 0) & (idx < k - 1)
        if not ok.any():
            continue
        native = np.rint(clock.native_index(m[ok])).astype(np.int64)
        p = y[ok] * np.conj(ghost_modulation(ghost, native, clock.fs_native_hz))
        acc += np.bincount(idx[ok], weights=p.real, minlength=k) + 1j * np.bincount(
            idx[ok], weights=p.imag, minlength=k
        )
        cnt += np.bincount(idx[ok], minlength=k)
    whole = np.zeros(k, dtype=bool)
    whole[:-1] = (cnt[:-1] == np.diff(s)) & (cnt[:-1] > 0)
    out = np.full(k, np.nan + 1j * np.nan)
    parts = out.view(np.float64).reshape(k, 2)  # (real, imag): each part divided as ghost_projection always did
    parts[whole, 0] = acc[whole].real / (amplitude * cnt[whole])
    parts[whole, 1] = acc[whole].imag / (amplitude * cnt[whole])
    return out


def ghost_projection(
    interference: Iterable[tuple[int, NDArray[np.complex128]]],
    ghost: GhostSpec,
    stamps: NDArray[np.integer],
    clock: TwinClock,
    amplitude: float,
) -> NDArray[np.float64]:
    """B1's n_iota per tracking record (rad; docs/3 section 2.2): Im(iota_k) / (A N_k), the imaginary part of
    `ghost_despread` (which see). A record whose integration the stream does not cover whole, and the last record,
    are NaN."""
    return np.ascontiguousarray(ghost_despread(interference, ghost, stamps, clock, amplitude).imag)
