"""S11: the M2M4 short-buffer bias and its effect on the thermal model B3 (F21; docs/4 B3, docs/2 H1b noise arm).

GNSS-SDR estimates C/N0 with M2M4 on 20 prompts and smooths the dB values (dll_pll_veml_tracking.cc:1355-1368).
The mean-dB bias b makes B3 under-predict the variance by 10^(b/10), so B3 is overconfident even on clean data, and
more so when interference lowers C/N0 (wideband noise moves C/N0 down the bias curve).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from gnssaudit.sim.correlator import TrackingConfig, m2m4_cn0_dbhz, thermal_noise, track
from gnssaudit.uq.loop import CarrierLoop
from review._common import save

T = 1e-3


def main() -> None:
    rng = np.random.default_rng(21)
    rows = []
    for cn0_db in (25, 28, 30, 33, 35, 40, 45, 50):
        cn0 = 10 ** (cn0_db / 10)
        w = thermal_noise(rng, (50_000, 20), cn0, T)
        est = m2m4_cn0_dbhz(1 + w, T)
        valid = est > -99
        b = float(np.mean(est[valid]) - cn0_db)
        z = stats.norm.ppf(0.975)
        cov = float(2 * stats.norm.cdf(z * 10 ** (-b / 20)) - 1)
        row = {"cn0_dbhz": cn0_db, "iid_mean_db_bias": b, "invalid_fraction": float(1 - valid.mean()),
               "b3_variance_ratio_true_over_pred": 10 ** (b / 10), "b3_coverage95_if_only_bias": cov}  # fmt: skip
        if cn0_db >= 30:
            for bn in (5.0, 35.0):
                cfg = TrackingConfig(bn_hz=bn, t_s=T, pilot=True)
                k = 60_000
                eta = thermal_noise(rng, (8, k), cn0, T)
                out = track(cfg, np.zeros((8, k)), eta)
                rep = np.nanmean(out.cn0_dbhz[:, 10_000:])
                e = 2 * np.pi * out.phi[:, 10_000:]
                loop = CarrierLoop.gnsssdr(bn, T)
                c_rep = 10 ** (rep / 10)
                var_b3 = loop.bn_eff_hz() / c_rep * (1 + 1 / (2 * T * c_rep))
                row[f"in_loop_bn{int(bn)}"] = {
                    "reported_minus_true_db": float(rep - cn0_db),
                    "var_true_over_b3": float(np.var(e) / var_b3),
                    "b3_coverage95": float(np.mean(np.abs(e) <= z * np.sqrt(var_b3))),
                }
        rows.append(row)
        print(row)
    save("s11_cn0_bias", {"rows": rows})


if __name__ == "__main__":
    main()
