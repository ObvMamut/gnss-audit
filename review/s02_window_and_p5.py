"""S02: windowed witness statistics and the P5 identifiable-fraction prediction (findings F5, F6; docs/4 P4, P5).

Monte Carlo of the IW noise term (amplitude normalisation by the window mean of I, then window variance of
H_phi m~ divided by the mean-removal factor) on circular AR(1) disturbances, against:
* docs/4 P5 as written: L_W = (1 - B_W)^2, weight |L_W|^2 = |1 - B_W|^4 (centred and causal boxcar B_W);
* a projection model: both steps remove the window mean, whose expected power response is 1 - |B_W|^2.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import signal

from gnssaudit.uq import iw
from gnssaudit.uq.loop import CarrierLoop
from review._common import save

T = 1e-3


def ar1(rng: np.random.Generator, n: int, fc: float) -> np.ndarray:
    a = np.exp(-2 * np.pi * fc * T)
    x = signal.lfilter([np.sqrt(1 - a * a)], [1, -a], rng.standard_normal(n))
    return np.asarray(x)


def ar1_psd(w: np.ndarray, fc: float) -> np.ndarray:
    a = np.exp(-2 * np.pi * fc * T)
    return (1 - a * a) / np.abs(1 - a * np.exp(-1j * w)) ** 2


def boxcar(w: np.ndarray, n: int, centred: bool) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(w == 0, 1.0, np.sin(n * w / 2) / (n * np.sin(w / 2)))
    return b if centred else b * np.exp(-1j * w * (n - 1) / 2)


def predictions(loop: CarrierLoop, fc: float, n: int) -> dict[str, float]:
    w = np.concatenate([np.geomspace(1e-8, 1e-3, 200_000), np.linspace(1e-3, np.pi, 800_001)[1:]])
    h2 = np.abs(loop.freq_response(w)) ** 2
    s = ar1_psd(w, fc)
    out = {}
    for name, weight in {
        "docs4_P5_centred": np.abs(1 - boxcar(w, n, True)) ** 4,
        "docs4_P5_causal": np.abs(1 - boxcar(w, n, False)) ** 4,
        "projection_twice": (1 - np.abs(boxcar(w, n, True)) ** 2) ** 2,
        "projection_once": 1 - np.abs(boxcar(w, n, True)) ** 2,
    }.items():
        kappa = np.trapezoid(h2 * weight, w) / np.trapezoid(h2, w)
        out[name] = float(np.trapezoid(h2 * weight * s, w) / np.trapezoid(h2 * s, w) / kappa)
    return out


def run_cell(args: tuple[str, float, float, int]) -> dict[str, object]:
    loop_name, fc, w_s, seed = args
    loop = CarrierLoop.textbook(5.0, T) if loop_name == "textbook5" else CarrierLoop.gnsssdr(5.0, T)
    n = int(round(w_s / T))
    rng = np.random.default_rng(seed)
    n_total = max(4_000_000, 400 * n)
    burn = 20_000
    sig = 0.05
    m = sig * ar1(rng, n_total + burn, fc)
    q = sig * ar1(rng, n_total + burn, fc)
    truth = np.var(loop.filter(q)[burn:])
    i_prompt = 1.0 + m[burn:]
    out = iw.iw_windows(i_prompt, np.zeros(n_total), loop, iw.IWConfig(window=n, gated=False))
    unnorm = iw.iw_windows(i_prompt, np.zeros(n_total), loop, iw.IWConfig(window=n, gated=False, normalise=False))
    phi = out.noise.mean() / truth
    se = out.noise.std(ddof=1) / np.sqrt(out.noise.size) / truth
    return {
        "loop": loop_name, "corner_hz": fc, "W_s": w_s,
        "phi_mc": phi, "phi_mc_se_naive": se, "n_windows": int(out.noise.size),
        "phi_mc_without_normalisation": unnorm.noise.mean() / truth,
        "pred": predictions(loop, fc, n),
    }  # fmt: skip


def main() -> None:
    cells = [
        (lp, fc, w_s, 1000 + i)
        for i, (lp, fc, w_s) in enumerate(
            (lp, fc, w_s)
            for lp in ("textbook5", "gnsssdr5")
            for fc in (10.0, 3.0, 1.0, 0.3, 0.1, 0.03)
            for w_s in (1.0, 2.0, 5.0)
        )
    ]
    with ProcessPoolExecutor(max_workers=18) as ex:
        rows = list(ex.map(run_cell, cells))
    for r in rows:
        p = r["pred"]
        assert isinstance(p, dict)
        print(r["loop"], r["corner_hz"], r["W_s"], "mc=%.3f" % r["phi_mc"],
              " ".join(f"{k}={v:.3f}" for k, v in p.items()))  # fmt: skip
    save("s02_window_and_p5", {"rows": rows})


if __name__ == "__main__":
    main()
