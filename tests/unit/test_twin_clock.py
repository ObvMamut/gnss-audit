"""The map from a twin's sample stamps to the recording (p2a review, important 3; P2b task 10): the injector draws
every disturbance, the ghost included, on the absolute native sample index, while GNSS-SDR stamps the decimated twin
from its own first sample. Output sample m is native index sample_start + decim m, delayed by the decimation FIR's
group delay (taps - 1) / 2."""

import numpy as np
import pyarrow as pa

from gnssaudit.inject.families import GhostSpec, ghost_phase_rad
from gnssaudit.io.schemas import TRACK_STREAM, build
from gnssaudit.truth.ghost import ghost_truth
from gnssaudit.truth.rxtime import TwinClock, retime

FS = 30.69e6
CLOCK = TwinClock(fs_native_hz=FS, sample_start=9_207_000, decim=4, delay=64.0)
G = GhostSpec(prn=31, doppler_hz=1500.0, doppler_rate_hzps=0.5, cn0_db_hz=45.0, chip0=3.0, seed=2)


def track(stamps: np.ndarray, acc: np.ndarray) -> pa.Table:
    n = stamps.size
    nulls = {c: [None] * n for c in TRACK_STREAM.names}
    cols = {**nulls, "run_id": ["r"] * n, "channel": [0] * n, "sys": ["G"] * n, "signal": ["1C"] * n,
            "prn": [31] * n, "k": list(range(n)), "sample_count": stamps.astype(np.uint64),
            "t_rx_s": stamps / (FS / 4), "acc_phase_rad": acc, "t_int_s": [1e-3] * n}  # fmt: skip
    return build(TRACK_STREAM, cols, n)


def test_output_stamps_map_to_native_indices_and_recording_time() -> None:
    m = np.array([0, 1, 7672], dtype=np.int64)
    np.testing.assert_allclose(CLOCK.native_index(m), [9_207_000 - 64, 9_207_000 + 4 - 64, 9_207_000 + 4 * 7672 - 64])
    np.testing.assert_allclose(CLOCK.file_time(m), CLOCK.native_index(m) / FS)
    assert TwinClock(fs_native_hz=4e6, sample_start=0, decim=1, delay=0.0).file_time(np.array([4000]))[0] == 1e-3


def test_retime_puts_the_track_on_recording_time() -> None:
    stamps = np.arange(7672, 7672 * 50, 7672, dtype=np.int64)
    tr = retime(track(stamps, np.zeros(stamps.size)), CLOCK)
    np.testing.assert_allclose(tr.column("t_rx_s").to_numpy(), CLOCK.file_time(stamps))


def test_ghost_truth_follows_the_ghost_on_a_decimated_mid_file_twin() -> None:
    """A receiver that tracks the ghost perfectly has zero error only if truth evaluates theta_g at the native index
    the stamp stands for; on stamps / fs a Doppler rate of 0.5 Hz/s and a 0.3 s offset leave a drifting error."""
    stamps = np.arange(7672, 7672 * 40_000, 7672, dtype=np.int64)  # 40 s of 1 ms records at 7.6725 MS/s
    acc = -ghost_phase_rad(G, CLOCK.native_index(stamps).astype(np.int64), FS)  # GNSS-SDR's `-=`: reported = -acc
    truth = ghost_truth(retime(track(stamps, acc), CLOCK), G, clock=CLOCK, converge_s=5.0, truth_id="g")
    e = truth.column("e_cyc").to_numpy()[truth.column("valid").to_numpy(zero_copy_only=False)]
    assert e.size > 10_000 and np.max(np.abs(e)) < 1e-6
