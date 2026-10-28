"""S03: time-differenced truth is blind to slow errors (F3), and the witness predicts differenced targets (N3).

Route D (docs/3 section 4.5) and TDCP-RAIM (docs/5 section 9.1) use Var(delta e) = sigma^2(t) + sigma^2(t + Delta).
The exact value is 2 int S_e(w) (1 - cos(w Delta)) dw / 2pi = 2 Var(e) rho_Delta with rho_Delta <= 1, and the
weight (1 - cos w Delta) vanishes like w^2 exactly where interference-induced error is concentrated.

Part A: rho_Delta for thermal plus coloured interference, and the coverage an overconfident model then shows.
Part B (N3): the differenced witness, Var_W(H_phi m~ (t + Delta) - H_phi m~ (t)), against the true Var(delta e).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s02_window_and_p5 import ar1, ar1_psd

T = 1e-3


def rho_delta(loop: CarrierLoop, fc: float | None, delta: int) -> float:
    w = np.concatenate([np.geomspace(1e-8, 1e-3, 200_000), np.linspace(1e-3, np.pi, 400_001)[1:]])
    h2 = np.abs(loop.freq_response(w)) ** 2
    s = np.ones_like(w) if fc is None else ar1_psd(w, fc)
    return float(np.trapezoid(h2 * s * (1 - np.cos(w * delta)), w) / np.trapezoid(h2 * s, w))


def coverage(var_ratio: float, p: float = 0.95) -> float:
    z = stats.norm.ppf(0.5 + p / 2)
    return float(2 * stats.norm.cdf(z / np.sqrt(var_ratio)) - 1)


def part_a() -> list[dict[str, object]]:
    rows = []
    for bn in (5.0, 35.0):
        loop = CarrierLoop.gnsssdr(bn, T)
        for fc in (None, 10.0, 3.0, 1.0, 0.3, 0.1, 0.03):
            rho = rho_delta(loop, fc, 1000)
            # an overconfident model under-predicts Var(e) by r; Route D then sees variance ratio r * rho
            cov = {f"r={r}": coverage(r * rho) for r in (1, 2, 3, 5, 10, 30)}
            r_pass = 0.92
            r_max = float(stats.norm.ppf(0.5 + 0.95 / 2) ** 2 / stats.norm.ppf(0.5 + r_pass / 2) ** 2 / rho)
            rows.append({"bn_cfg_hz": bn, "corner_hz": fc if fc is not None else "white",
                         "rho_delta_1s": rho, "route_d_coverage": cov,
                         "largest_underprediction_still_passing_c95_ge_0.92": r_max})  # fmt: skip
    return rows


def part_b_cell(args: tuple[float, float | None, float, int]) -> dict[str, object]:
    bn, fc, w_s, seed = args
    loop = CarrierLoop.gnsssdr(bn, T)
    rng = np.random.default_rng(seed)
    n_total, burn, delta = 3_000_000, 20_000, 1000
    n_win = int(w_s / T)
    sig = 0.05

    def dist() -> np.ndarray:
        x = rng.standard_normal(n_total + burn) if fc is None else ar1(rng, n_total + burn, fc)
        return sig * x

    m, q = dist(), dist()
    e = loop.filter(q)[burn:]
    de = e[delta:] - e[:-delta]
    truth = float(np.var(de))
    i_prompt = 1.0 + m[burn:]
    nb = n_total // n_win
    ib = i_prompt[: nb * n_win].reshape(nb, n_win)
    mt = (ib / ib.mean(axis=1, keepdims=True) - 1.0).ravel()
    h = loop.filter(mt)
    hb = h.reshape(nb, n_win)
    dh = hb[:, delta:] - hb[:, :-delta]
    est_diff = (dh**2).mean(axis=1)  # no mean removal: differencing removes the window mean exactly
    est_naive = 2 * hb.var(axis=1) / (1 - 1 / (2 * loop.bn_eff_hz() * w_s))  # docs/3: 2 sigma^2
    return {"bn_cfg_hz": bn, "corner_hz": fc if fc is not None else "white", "W_s": w_s,
            "truth_var_delta_e": truth, "ratio_differenced_witness": float(est_diff.mean() / truth),
            "ratio_2sigma2_witness": float(est_naive.mean() / truth),
            "rel_sd_differenced": float(est_diff.std() / est_diff.mean())}  # fmt: skip


def main() -> None:
    rows_a = part_a()
    for r in rows_a:
        print(r["bn_cfg_hz"], r["corner_hz"], "rho=%.3f" % r["rho_delta_1s"],
              "rmax=%.1f" % r["largest_underprediction_still_passing_c95_ge_0.92"])  # fmt: skip
    cells = [
        (bn, fc, w, 77 + i)
        for i, (bn, fc, w) in enumerate(
            (bn, fc, w) for bn in (5.0, 35.0) for fc in (None, 3.0, 1.0, 0.3, 0.1, 0.03) for w in (5.0, 10.0)
        )
    ]
    with ProcessPoolExecutor(max_workers=18) as ex:
        rows_b = list(ex.map(part_b_cell, cells))
    for r in rows_b:
        print(r["bn_cfg_hz"], r["corner_hz"], r["W_s"], "diff=%.3f" % r["ratio_differenced_witness"],
              "2sig2=%.3f" % r["ratio_2sigma2_witness"])  # fmt: skip
    save("s03_tdcp_and_spectral_witness", {"part_a": rows_a, "part_b": rows_b})


if __name__ == "__main__":
    main()
