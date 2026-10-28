"""S08: is the gated IW always conservative under amplitude contamination? (F7; docs/4 P3, section 3.7 step 6).

docs/4 P3 claims the gated form's bias is Var(H a) >= 0. When genuine phase dynamics make the gate fire, the
dynamics term D = Var(G d) - Var(G s) is reduced by Var(G H_d a), because s = H_d m~ contains H_d a. The net bias
Var(H a) - Var(G H_d a) is negative for amplitude processes between the |H| = |1 - H| crossover and f_G.

Simplified loop (kappa = 0) so that e = H n - (1 - H) theta exactly. Thermal 45 dB-Hz; phase dynamics theta with
Var((1 - H) theta) equal to 1, 3 or 20 times the thermal Var(H n); amplitude processes a of 2 % RMS at several corner frequencies.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import signal

from gnssaudit.uq import iw
from gnssaudit.uq.loop import CarrierLoop
from review._common import save

T = 1e-3


def bandpass(rng: np.random.Generator, n: int, f0: float, rms: float) -> np.ndarray:
    sos = signal.butter(2, [0.7 * f0 * 2 * T, 1.4 * f0 * 2 * T], btype="band", output="sos")
    x = np.asarray(signal.sosfilt(sos, rng.standard_normal(n)))
    return rms * x / x.std()


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
        target = level * np.var(loop.filter(q)[10_000:])
        theta *= np.sqrt(target / np.var(theta - loop.filter(theta)))
    a = np.zeros(n) if f_amp is None else bandpass(rng, n, f_amp, 0.02)
    e = loop.filter(theta + q) - theta
    d = loop.filter(theta + q, "disc")
    i_prompt = (1 + a) * (1 + m)
    out = iw.iw_windows(i_prompt, d, loop, iw.IWConfig(window=2000, gate_z=5.0))
    burn = 5
    nb = out.var.size
    eb = e[: nb * 2000].reshape(nb, 2000)[burn:]
    truth = (eb**2).mean(axis=1)  # zero-mean error: mean square, not the mean-removed window variance
    ratio = out.var[burn:] / truth
    return {"bn_cfg_hz": bn, "dynamics": dyn, "amp_band_hz": f_amp if f_amp is not None else "none",
            "mean_ratio": float(np.mean(out.var[burn:]) / np.mean(truth)),
            "frac_windows_under_0.8": float(np.mean(ratio < 0.8)),
            "gate_fire_rate": float(np.mean(out.gate[burn:])),
            "noise_term_ratio": float(np.mean(out.noise[burn:]) / np.mean(truth))}  # fmt: skip


def main() -> None:
    jobs = []
    for bn in (5.0, 35.0):
        for dyn in ("none", "inband_x1", "inband_x3", "inband_x20"):
            for fa in (None, 0.5, 2.0, 5.0, 10.0, 20.0, 60.0):
                jobs.append((bn, dyn, fa, 6000 + len(jobs)))
    with ProcessPoolExecutor(max_workers=19) as ex:
        rows = list(ex.map(cell, jobs))
    for r in rows:
        print(r)
    save("s08_gate_bias", {"rows": rows})


if __name__ == "__main__":
    main()
