"""S05: is coverage the right endpoint, and does a distribution-level witness fix it? (F4, P4, N5; docs/4 P4).

Thermal noise at 45 dB-Hz plus one despread interference class (S04 generator) scaled so that the
interference-induced error variance is four times the thermal one. Linear regime, faithful GNSS-SDR loop.
Per W = 2 s window the IW noise term gives sigma^2 (amplitude-normalised witness); the epochs of the same window
are scored (evaluation form of docs/4 section 3.7) with:
* gauss: N(0, sigma_IW^2);  * student: t_nu with the exact nu of the window;
* quantile (N5): the window's empirical quantiles of |h - mean(h)|, h = H_phi m~, rescaled by the mean-removal
  factor, used directly as interval half-widths;
* oracle_gauss: N(0, sigma^2) with the true long-run variance of e.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from gnssaudit.sim.correlator import thermal_noise
from gnssaudit.uq import iw
from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s04_structured_interference import despread_series

LEVELS = (0.68, 0.95, 0.99, 0.999)
W_S = 2.0
T = 1e-3


def cell(args: tuple[str, float, int]) -> dict[str, object]:
    name, bn, seed = args
    loop = CarrierLoop.gnsssdr(bn, T)
    eta_i = despread_series(name, seed)
    eta_i = eta_i - eta_i.mean()
    rng = np.random.default_rng(seed + 1)
    k = eta_i.size
    eta_w = thermal_noise(rng, (k,), cn0_hz=10**4.5, t_s=T)
    var_th = np.var(loop.filter(eta_w.imag)[5000:])
    var_i = np.var(loop.filter(eta_i.imag)[5000:])
    eta = eta_w + eta_i * np.sqrt(4.0 * var_th / var_i)
    m, n = eta.real, eta.imag
    e = loop.filter(n)
    n_win = int(W_S / T)
    out = iw.iw_windows(1.0 + m, np.zeros(k), loop, iw.IWConfig(window=n_win, gated=False))
    nb = out.noise.size
    burn = 5  # windows
    eb = e[: nb * n_win].reshape(nb, n_win)[burn:]
    sig = np.sqrt(out.noise)[burn:, None]
    ib = (1.0 + m)[: nb * n_win].reshape(nb, n_win)
    mt = ib / ib.mean(axis=1, keepdims=True) - 1.0
    hb = loop.filter(mt.ravel()).reshape(nb, n_win)[burn:]
    factor, nu = iw.noise_term_moments(loop, n_win)
    dev = np.abs(hb - hb.mean(axis=1, keepdims=True)) / np.sqrt(factor)
    sig_oracle = np.sqrt(np.var(e[5000:]))
    res: dict[str, dict[str, float]] = {}
    for p in LEVELS:
        zg = stats.norm.ppf(0.5 + p / 2)
        zt = stats.t.ppf(0.5 + p / 2, nu)
        qh = np.quantile(dev, p, axis=1)[:, None]
        res[str(p)] = {
            "gauss": float(np.mean(np.abs(eb) <= zg * sig)),
            "student": float(np.mean(np.abs(eb) <= zt * sig)),
            "quantile_N5": float(np.mean(np.abs(eb) <= qh)),
            "oracle_gauss": float(np.mean(np.abs(eb) <= zg * sig_oracle)),
        }
    log_ratio = np.log(out.noise[burn:] / eb.var(axis=1))
    return {"class": name, "bn_cfg_hz": bn, "nu": nu, "coverage": res,
            "mean_log_ratio_sigma2_iw_over_window_var": float(np.mean(log_ratio)),
            "half_width95_ratio_quantile_over_gauss": float(
                np.mean(np.quantile(dev, 0.95, axis=1)) / np.mean(stats.norm.ppf(0.975) * sig))}  # fmt: skip


def main() -> None:
    classes = ["wideband_noise", "cw_beat_7hz", "cw_beat_0.3hz", "swept_slow", "chirp_inband",
               "pulsed_noise_lowprf", "pulsed_cw"]  # fmt: skip
    cells = [(c, bn, 900 + 10 * i) for i, c in enumerate(classes) for bn in (5.0, 35.0)]
    with ProcessPoolExecutor(max_workers=len(cells)) as ex:
        rows = list(ex.map(cell, cells))
    for r in rows:
        cov = r["coverage"]
        assert isinstance(cov, dict)
        print(r["class"], r["bn_cfg_hz"], "nu=%.0f" % r["nu"],
              {p: {k: round(v, 3) for k, v in d.items()} for p, d in cov.items() if p in ("0.95", "0.99")})  # fmt: skip
    save("s05_predictive_shape", {"W_s": W_S, "rows": rows})


if __name__ == "__main__":
    main()
