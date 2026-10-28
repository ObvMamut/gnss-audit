"""S10: statistical-procedure checks (F15, F16, F17; docs/4 section 3.8, docs/5 sections 4-5, docs/11 P-circ-null).

F16: the circularity statistic Lambda = mean over multitaper bins of log(S_d / (|H_d|^2 S_m)) has null variance
     2 psi'(L) / n_bins only if bins are independent; DPSS tapers with half-bandwidth NW/W correlate neighbouring
     bins, and |H_d|^2 changes across the taper bandwidth. Empirical null mean, variance and test size.
F15: satellites sharing one jammer envelope give exceedance indicators that are correlated across satellites when
     the model does not follow the envelope; pooling satellites as independent arcs understates the SE.
F17: the C1 audit rule sizes n_eff with p(1-p) at nominal p; under real overconfidence the interval is wider.
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats
from scipy.signal import windows

from gnssaudit.uq.loop import CarrierLoop
from review._common import save

T = 1e-3


def lambda_stat(d: np.ndarray, m: np.ndarray, hd2: np.ndarray, sel: np.ndarray, tapers: np.ndarray) -> float:
    sd = np.mean(np.abs(np.fft.rfft(tapers * d, axis=1)) ** 2, axis=0)
    sm = np.mean(np.abs(np.fft.rfft(tapers * m, axis=1)) ** 2, axis=0)
    return float(np.mean(np.log(sd[sel] / (hd2[sel] * sm[sel]))))


def f16(bn: float, n_win: int = 2000, nw: float = 4.0, reps: int = 4000) -> dict[str, float]:
    loop = CarrierLoop.gnsssdr(bn, T)
    n_tap = int(2 * nw - 1)
    tapers = np.asarray(windows.dpss(n_win, nw, n_tap))
    f = np.fft.rfftfreq(n_win, T)
    hd2 = np.abs(loop.freq_response(2 * np.pi * f * T, "disc")) ** 2
    sel = (hd2 >= 0.5) & (f <= 50.0)
    rng = np.random.default_rng(int(bn))
    lam = np.empty(reps)
    burn = 20_000
    for i in range(reps):
        q = rng.standard_normal(n_win + burn)
        d = loop.filter(q, "disc")[burn:]
        m = rng.standard_normal(n_win)
        lam[i] = lambda_stat(d, m, hd2, sel, tapers)
    nb = int(sel.sum())
    var_doc = 2 * special.polygamma(1, n_tap) / nb
    z = lam / np.sqrt(var_doc)
    return {"bn_cfg_hz": bn, "n_bins": nb, "L_tapers": n_tap, "null_mean": float(lam.mean()),
            "null_var": float(lam.var()), "docs_var_2psi1_over_nbins": float(var_doc),
            "variance_inflation": float(lam.var() / var_doc),
            "size_at_nominal_0.05_docs_variance": float(np.mean(np.abs(z) > 1.96)),
            "size_at_nominal_0.05_centred_empirical_var": float(
                np.mean(np.abs(lam - lam.mean()) > 1.96 * lam.std()))}  # fmt: skip


def f15(n_sat: int = 8, reps: int = 400) -> dict[str, float]:
    """Ten 60 s jamming events whose power varies from event to event (log-normal, shared by all satellites).

    The model's sigma follows the long-run mean power, as a smoothed C/N0 model calibrated on pooled data would.
    """
    rng = np.random.default_rng(15)
    n = 600  # 1 Hz scoring epochs over 10 min
    pooled = np.empty(reps)
    naive_se = np.empty(reps)
    for r in range(reps):
        env = np.repeat(np.exp(0.7 * rng.standard_normal(10)), 60)  # error SD multiplier, common to all
        err = env[None, :] * rng.standard_normal((n_sat, n))
        sigma_model = 1.6  # fixed model scale (about the mean RMS of env)
        hit = np.abs(err) <= 1.96 * sigma_model
        pooled[r] = hit.mean()
        # docs/5: n_eff summed over satellites treated as independent (indicators white in time here)
        p = hit.mean()
        naive_se[r] = np.sqrt(p * (1 - p) / (n_sat * n))
    return {"true_sd_of_pooled_coverage": float(pooled.std()), "mean_naive_se": float(naive_se.mean()),
            "se_understatement_factor": float(pooled.std() / naive_se.mean()),
            "mean_pooled_coverage": float(pooled.mean())}  # fmt: skip


def f17() -> dict[str, float]:
    n_eff = 200
    out = {}
    for c in (0.95, 0.90, 0.85, 0.80):
        out[f"half_width_95ci_at_c={c}"] = float(stats.norm.ppf(0.975) * np.sqrt(c * (1 - c) / n_eff))
    out["n_eff_needed_for_pm0.03_at_c=0.85"] = float((stats.norm.ppf(0.975) / 0.03) ** 2 * 0.85 * 0.15)
    return out


def main() -> None:
    res16 = [f16(5.0), f16(35.0)]
    for r in res16:
        print(r)
    res15 = f15()
    print(res15)
    res17 = f17()
    print(res17)
    save("s10_statistics", {"f16": res16, "f15": res15, "f17": res17})


if __name__ == "__main__":
    main()
