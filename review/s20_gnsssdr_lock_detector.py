"""S20: does the GNSS-SDR binary drop channels that are still tracking? (F22 on the binary.)

Recordings at 50 dB-Hz for 20 s (reliable 1 ms acquisition), then stepped to 30 / 33 / 36 / 40 dB-Hz for 40 s.
Each runs twice: with defaults (does the receiver declare loss of lock after the step?) and with
`max_carrier_lock_fail` raised so that no carrier loss is declared (the tracking quality the receiver would have
kept, measured against the known true phase, and the smoothed lock statistic it dumps). The harness prediction
of the dumped statistic (`review/s16_lock_detector.py`) is reported alongside.

Usage: uv run python -m review.s20_gnsssdr_lock_detector   (needs Docker via sg)
"""

from __future__ import annotations

import numpy as np

from review._common import save
from review.s16_lock_detector import mean_lock_stat
from review.s18_gnsssdr_cn0_lock_p6 import OUT, STEP_S, generate, phase_error, run_receiver


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "home").mkdir(exist_ok=True)
    (OUT / "pvt").mkdir(exist_ok=True)
    rng = np.random.default_rng(20)
    rows = []
    for cn0 in (30.0, 33.0, 36.0, 40.0):
        rec_name = f"step{int(cn0)}.bin"
        generate(OUT / rec_name, cn0, None, 0.0, step_from_db=50.0)
        row: dict[str, object] = {"cn0_after_step_dbhz": cn0,
                                  "predicted_mean_lock_stat": mean_lock_stat(cn0, 1e-3, rng)}  # fmt: skip
        for tag, overrides in (("default", {}), ("nodrop", {"Tracking_1C.max_carrier_lock_fail": "1000000000"})):
            name = f"step{int(cn0)}{tag}"
            r = run_receiver(name, rec_name, overrides)
            log = str(r["log"])
            rec = r["rec"]
            assert isinstance(rec, np.ndarray)
            t = rec["prn_start_sample"].astype(np.float64) / 4.0e6 if rec.size else np.zeros(0)
            sel = t > STEP_S + 10.0
            res: dict[str, object] = {
                "acquired": "Tracking of GPS L1 C/A signal started" in log,
                "loss_of_lock_declared": log.count("Loss of lock"),
                "tracking_sessions": log.count("Tracking of GPS L1 C/A signal started"),
            }
            if sel.sum() > 100 and res["tracking_sessions"] == 1:
                _, e = phase_error(rec, t_min=STEP_S + 10.0)
                res |= {"phase_sd_rad_after_step": float(np.std(e)),
                        "dumped_lock_stat_median": float(np.median(rec["lock"][sel])),
                        "reported_cn0_median": float(np.median(rec["cn0_dbhz"][sel])),
                        "doppler_sd_hz": float(np.std(rec["doppler_hz"][sel]))}  # fmt: skip
            row[tag] = res
        rows.append(row)
        print(row, flush=True)
        (OUT / rec_name).unlink()
    save("s20_gnsssdr_lock_detector", {"rows": rows})


if __name__ == "__main__":
    main()
