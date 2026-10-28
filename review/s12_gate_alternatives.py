"""S12: replacing the hard gate (fix for F7; N4). Same scenarios as S08.

Estimators of Var(e) per W = 2 s window, from the noise term V and the band-limited dynamics estimate
D = Var(G d) - Var(G s) with its chi-square SE:
  gate5   V + D 1[D > 5 SE]        (docs/4 section 3.7)
  gate2   V + D 1[D > 2 SE]
  pospart V + max(D, 0)           (no test: conservative by E[max(D, 0)] when there are no dynamics)
  ungated max(V + D, V / 4)       (unbiased for the band-limited dynamics; floor against negative totals)
Scored by the mean ratio to the truth, Student-t 95 % coverage of the epochs in the window, and sharpness.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from gnssaudit.uq import iw
from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s08_gate_bias import bandpass

T = 1e-3
N_WIN = 2000


def cell(args: tuple[float, str, float | None, int]) -> dict[str, object]:
    bn, dyn, f_amp, seed = args
    loop = CarrierLoop.gnsssdr(bn, T, kappa=0.0)
    rng = np.random.default_rng(seed)
    n = 3_000_000
    sig_n = np.sqrt(1 / (2 * 10**4.5 * T))
    m = sig_n * rng.standard_normal(n)
    q = sig_n * rng.standard_normal(n)
    if dyn == "none":
        theta = np.zeros(n)
    else:
        level = float(dyn.removeprefix("inband_x"))
        theta = bandpass(rng, n, 1.5 * loop.bn_eff_hz(), 1.0)
        theta *= np.sqrt(level * np.var(loop.filter(q)[10_000:]) / np.var(theta - loop.filter(theta)))
    a = np.zeros(n) if f_amp is None else bandpass(rng, n, f_amp, 0.02)
    e = loop.filter(theta + q) - theta
    d = loop.filter(theta + q, "disc")
    out = iw.iw_windows((1 + a) * (1 + m), d, loop, iw.IWConfig(window=N_WIN, gate_z=5.0))
    burn = 5
    v, dr, se = out.noise[burn:], out.dyn_raw[burn:], out.dyn_se[burn:]
    eb = e[: out.noise.size * N_WIN].reshape(-1, N_WIN)[burn:]
    truth = float(np.mean(eb**2))
    sharp_ref = np.sqrt(truth)
    zt = stats.t.ppf(0.975, out.nu)
    ests = {
        "gate5": v + dr * (dr > 5 * se),
        "gate2": v + dr * (dr > 2 * se),
        "pospart": v + np.maximum(dr, 0.0),
        "ungated": np.maximum(v + dr, v / 4),
    }
    res: dict[str, object] = {
        "bn_cfg_hz": bn,
        "dynamics": dyn,
        "amp_band_hz": f_amp if f_amp is not None else "none",
    }
    for k, s2 in ests.items():
        res[k] = {"mean_ratio": float(np.mean(s2) / truth),
                  "coverage95_t": float(np.mean(np.abs(eb) <= zt * np.sqrt(s2)[:, None])),
                  "mean_sigma_over_true": float(np.mean(np.sqrt(s2)) / sharp_ref)}  # fmt: skip
    return res


def main() -> None:
    jobs = []
    for bn in (5.0, 35.0):
        for dyn in ("none", "inband_x1", "inband_x3", "inband_x20"):
            for fa in (None, 5.0, 20.0):
                jobs.append((bn, dyn, fa, 6100 + len(jobs)))
    with ProcessPoolExecutor(max_workers=19) as ex:
        rows = list(ex.map(cell, jobs))
    for r in rows:
        print(
            r["bn_cfg_hz"],
            r["dynamics"],
            r["amp_band_hz"],
            " ".join(
                f"{k}:{v['coverage95_t']:.3f}/{v['mean_sigma_over_true']:.2f}"  # type: ignore[index]
                for k, v in r.items()
                if isinstance(v, dict)
            ),
        )
    save("s12_gate_alternatives", {"rows": rows})


if __name__ == "__main__":
    main()
