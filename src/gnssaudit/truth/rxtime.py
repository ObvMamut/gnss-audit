"""Observable epochs on the recording's sample axis (docs/3 section 2.6; docs/9 section 10; P2b task 9).

GNSS-SDR v0.0.22's observables block is driven by a sample-counter clock: its stamps are exact multiples of
step = round(fs x interval) (gnss_sdr_sample_counter.cc:150-151), and each epoch interpolates the tracking phase at
that sample (hybrid_observables_gs.cc:510). The dump records the receiver TOW of each epoch but not its sample, so
epoch i is placed at (i + k0) x step, and k0 is pinned by matching the observables carrier phase to the float64
tracking phase of the same channel: at the right k0 the two differ by a constant only (up to half-cycle polarity
steps, removed modulo 0.5 cycle), at a wrong one by the Doppler change over the shift. This places an onset given in
recording samples on observable epochs, so twin truth can be formed after the map M.

Verified on the v0.0.22 binary (CTTC sample, five channels): k0 = 1, the first output being the clock's first tick, and
sign = -1: the observables carrier phase is +acc/2 pi, the opposite sign of the reported phase -acc/2 pi used for the
tracking dump (`experiments.twin.track_as_obs`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MATCH_TOL_CYC = 0.01  # RMS of the matched residual. The observables are interpolated at the record's code-phase time,
# a few samples from its stamp: 0.0008 to 0.003 cycle at 5 to 10 kHz Doppler on the CTTC sample (P2b task 9)
SEPARATION = 10.0  # the next-best offset must leave at least this many times the best residual
MIN_POINTS = 100


class RxTimeError(ValueError):
    """No clock offset maps the observables onto the tracking phase."""


@dataclass(frozen=True)
class ObsClock:
    k0: int  # epoch i sits at sample (i + k0) x step
    step: int  # samples per observable interval
    sign: float = 1.0  # observables phase = sign x tracking phase + constant
    residual_cyc: float = 0.0  # RMS of the matched residual


def epoch_samples(idx: IntArray, clock: ObsClock) -> IntArray:
    return (np.asarray(idx, dtype=np.int64) + clock.k0) * clock.step


def onset_epoch(onset_sample: int, clock: ObsClock) -> int:
    """The first epoch at or after the onset sample."""
    return -(-onset_sample // clock.step) - clock.k0


def _rms(obs: FloatArray, model: FloatArray) -> float:
    r = obs - model
    r = r - np.median(r)
    r = r - 0.5 * np.round(r / 0.5)  # half-cycle polarity steps of the observables' PLL-180 frame
    return float(np.sqrt(np.mean(r**2)))


def match_clock(
    idx: IntArray,
    obs_cyc: FloatArray,
    stamps: IntArray,
    track_cyc: FloatArray,
    *,
    step: int,
    k_max: int = 200,
    tol_cyc: float = MATCH_TOL_CYC,
) -> ObsClock:
    """The clock offset k0 (and the sign) under which one channel's observables phase is its tracking phase plus a
    constant. Raises when no offset fits within tol_cyc, or when two offsets do (no Doppler change to tell them)."""
    idx = np.asarray(idx, dtype=np.int64)
    fits: list[tuple[float, int, float]] = []
    for sign in (1.0, -1.0):
        for k0 in range(k_max):
            s = (idx + k0) * step
            inside = (s >= stamps[0]) & (s <= stamps[-1])
            if inside.sum() < MIN_POINTS:
                continue
            fits.append((_rms(obs_cyc[inside], sign * np.interp(s[inside], stamps, track_cyc)), k0, sign))
    if not fits:
        raise RxTimeError("no clock offset leaves enough epochs inside the tracking record")
    fits.sort()
    best = fits[0]
    if best[0] > tol_cyc:
        raise RxTimeError(f"no clock offset fits: best residual {best[0]:.3g} cycles at k0 = {best[1]}")
    if len(fits) > 1 and fits[1][0] <= SEPARATION * best[0]:
        raise RxTimeError(f"two clock offsets fit (k0 = {best[1]} and {fits[1][1]}): too little Doppler change")
    return ObsClock(k0=best[1], step=step, sign=best[2], residual_cyc=best[0])


@dataclass(frozen=True)
class TwinClock:
    """Where a twin's output stamps sit in the recording (p2a review, important 3). The injector draws every
    disturbance, ghosts included, on the absolute native sample index; GNSS-SDR stamps the decimated twin from its own
    first sample. Output sample m stands for native index sample_start + decim m - delay, delay being the decimation
    FIR's group delay ((taps - 1) / 2 native samples; 0 without decimation)."""

    fs_native_hz: float
    sample_start: int
    decim: int
    delay: float

    @classmethod
    def identity(cls, fs_hz: float) -> TwinClock:
        """A twin that is the recording itself, from its first sample, undecimated."""
        return cls(fs_native_hz=fs_hz, sample_start=0, decim=1, delay=0.0)

    def native_index(self, stamps: NDArray[np.integer]) -> FloatArray:
        return self.sample_start + self.decim * np.asarray(stamps, dtype=np.float64) - self.delay

    def file_time(self, stamps: NDArray[np.integer]) -> FloatArray:
        """Recording time (s from the file's first sample), the time base of onsets and disturbance windows."""
        return self.native_index(stamps) / self.fs_native_hz


def retime(track: pa.Table, clock: TwinClock) -> pa.Table:
    """A tracking-dump table with t_rx_s on recording time."""
    t = clock.file_time(track.column("sample_count").to_numpy())
    return track.set_column(track.schema.get_field_index("t_rx_s"), "t_rx_s", pa.array(t, pa.float64()))
