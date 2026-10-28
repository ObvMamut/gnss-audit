"""S16: where GNSS-SDR's carrier lock test declares loss of lock (side finding; docs/9 section 6, docs/0 item 1).

The test is the single-sample cos(2 angle P) (carrier_lock_detector(&d_P_accu, 1), dll_pll_veml_tracking.cc:1369),
exponentially smoothed with alpha = 0.002, against carrier_lock_th = 0.7. Its mean under perfect tracking is
E[cos 2 angle(1 + w)] with per-integration SNR = C/N0 T, so the declared lock threshold is set by C/N0 T, not by
the loop's actual phase jitter. Computed by Monte Carlo of the statistic, and checked in the closed loop.
"""

from __future__ import annotations

import numpy as np

from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from review._common import save


def mean_lock_stat(cn0_db: float, t_s: float, rng: np.random.Generator) -> float:
    w = thermal_noise(rng, (2_000_000,), 10 ** (cn0_db / 10), t_s)
    p = 1 + w
    return float(np.mean((p.real**2 - p.imag**2) / np.abs(p) ** 2))


def main() -> None:
    rng = np.random.default_rng(16)
    rows = []
    for t_s in (1e-3, 4e-3, 20e-3):
        for cn0_db in np.arange(20, 46, 1.0):
            rows.append({"t_s": t_s, "cn0_dbhz": float(cn0_db), "mean_lock_stat": mean_lock_stat(cn0_db, t_s, rng)})
    thresholds = {}
    for t_s in (1e-3, 4e-3, 20e-3):
        xs = [(r["cn0_dbhz"], r["mean_lock_stat"]) for r in rows if r["t_s"] == t_s]
        c = np.array([x[0] for x in xs])
        v = np.array([x[1] for x in xs])
        thresholds[str(t_s)] = float(np.interp(0.7, v, c))
    # closed-loop check at T = 1 ms, pilot, 5 Hz: smoothed statistic at 30 and 36 dB-Hz
    loop_check = {}
    for cn0_db in (30.0, 33.0, 36.0, 40.0):
        cfg = TrackingConfig(bn_hz=5.0, t_s=1e-3, pilot=True)
        out = track(cfg, np.zeros((4, 40_000)), thermal_noise(rng, (4, 40_000), 10 ** (cn0_db / 10), 1e-3))
        loop_check[str(cn0_db)] = {"smoothed_lock_mean": float(np.mean(out.lock[:, 10_000:])),
                                   "phase_sd_rad": float(np.std(2 * np.pi * out.phi[:, 10_000:]))}  # fmt: skip
    print(thresholds, loop_check)
    save("s16_lock_detector", {"threshold_cn0_dbhz_for_lock_0.7": thresholds, "rows": rows,
                               "closed_loop_check": loop_check})  # fmt: skip


if __name__ == "__main__":
    main()
