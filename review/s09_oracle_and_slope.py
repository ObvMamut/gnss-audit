"""S09: what the ensemble oracle conditions on (F8) and regression dilution of the calibration slope (F9).

docs/3 section 2.4 randomises the start offset of sweeps and pulses across the K realizations. For interference
whose power varies with a period P comparable to or longer than W, the per-epoch ensemble variance is then the
*marginal* (time-averaged) variance, while IW, and the error itself, follow the *conditional* variance at the
jammer's actual state. Here the despread disturbance is circular white noise with power envelope
p(t) = 1 + 0.9 sin(2 pi (t + tau) / P); realizations draw tau at random (marginal oracle) or keep it fixed
(conditional oracle). docs/5 section 3.5 metrics are computed against both.

F9: the slope of log sigma2_oracle on log sigma2_hat (docs/5 section 3.5) is attenuated by the sampling noise of
sigma2_hat; the errors-in-variables slope uses the known noise variance psi'(nu/2) of log chi^2_nu / nu.
"""

from __future__ import annotations

import numpy as np
from scipy import special

from gnssaudit.uq import iw
from gnssaudit.uq.loop import CarrierLoop
from review._common import save

T = 1e-3


def run(period_s: float, k_real: int, conditional: bool, seed: int, loop: CarrierLoop, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * T
    tau0 = rng.uniform(0, period_s)
    out = np.empty((k_real, n))
    for r in range(k_real):
        tau = tau0 if conditional else rng.uniform(0, period_s)
        env = np.sqrt(1 + 0.9 * np.sin(2 * np.pi * (t + tau) / period_s))
        out[r] = loop.filter(env * rng.standard_normal(n))
    return out


def main() -> None:
    loop = CarrierLoop.gnsssdr(5.0, T)
    n_win = 2000
    n = 400_000
    burn = 10
    rows = []
    for period in (1.0, 4.0, 16.0, 64.0):
        res: dict[str, object] = {"period_s": period}
        rng = np.random.default_rng(11)
        t = np.arange(n) * T
        tau_eval = rng.uniform(0, period)
        env = np.sqrt(1 + 0.9 * np.sin(2 * np.pi * (t + tau_eval) / period))
        m = env * rng.standard_normal(n)
        out = iw.iw_windows(1.0 + 0.01 * m, np.zeros(n), loop, iw.IWConfig(window=n_win, gated=False))
        sig_hat = out.noise[burn:] / 0.01**2
        _, nu = iw.noise_term_moments(loop, n_win)
        for name, cond in (("marginal", False), ("conditional", True)):
            ens = run(period, 51, cond, 99, loop, n) if not cond else None
            if cond:
                # conditional oracle: the evaluated realization's own jammer timing, fresh noise
                rng2 = np.random.default_rng(5)
                ens = np.stack([loop.filter(env * rng2.standard_normal(n)) for _ in range(51)])
            assert ens is not None
            orc = ens.var(axis=0, ddof=1)
            nb = sig_hat.size + burn
            orc_w = orc[: nb * n_win].reshape(nb, n_win).mean(axis=1)[burn:]
            ell = np.log(sig_hat / orc_w)
            x, y = np.log(sig_hat), np.log(orc_w)
            slope_ols = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
            noise_var = special.polygamma(1, nu / 2)
            slope_eiv = np.cov(x, y)[0, 1] / (np.var(x, ddof=1) - noise_var)
            res[name] = {"bias_mean_log_ratio": float(np.mean(ell)),
                         "deconvolved_spread": float(np.sqrt(max(np.var(ell) - 2 / 50, 0.0))),
                         "slope_ols_docs5": float(slope_ols), "slope_errors_in_variables": float(slope_eiv)}  # fmt: skip
        rows.append(res)
        print(res)
    save("s09_oracle_and_slope", {"W_s": n_win * T, "rows": rows})


if __name__ == "__main__":
    main()
