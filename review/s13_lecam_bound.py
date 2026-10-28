"""S13: a minimax lower bound for window-limited estimation of Var(e) from one channel (N4; replaces docs/4 P1).

Two worlds that a single channel cannot tell apart within a window of N samples:
  P0: circular white noise sigma^2, plus an amplitude-only process a with spectrum S_D (fading, AGC);
  P1: circular white noise sigma^2 plus an additive circular disturbance with spectrum S_D in both m and n.
The in-phase witness m~ = m + a has the same law in both worlds, so only d can separate them, and d sees the
difference only through |H_d|^2 S_D, which vanishes at low frequency. Var(e) differs by Delta = int |H_phi|^2 S_D.
Le Cam's two-point bound: inf_est max_i E_i |est - Var_i(e)| >= (Delta / 2) (1 - TV(P0, P1)), TV <= sqrt(KL / 2).
KL between the Gaussian laws of d over the window is computed exactly from the Toeplitz covariances.

The pilot-data witness (N1) escapes the bound: its in-phase difference has a known amplitude ratio, so a does not
cancel the additive disturbance there.

Part B (exact non-identifiability). If world P0 also carries a phase-only process with S_theta = S_D, then
(m~, d) have the same joint law in both worlds at every frequency, while Var(e) differs by
int (|H_phi|^2 - |H_d|^2) S_D, the docs/4 P3 weight. Simulation: IW outputs (V, D) and the circularity ratio agree
in distribution while the true Var(e) differs.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from gnssaudit.uq.loop import CarrierLoop
from review._common import save
from review.s02_window_and_p5 import ar1_psd

T = 1e-3


def autocov_from_psd(psd: np.ndarray, n: int) -> np.ndarray:
    """psd sampled on the full FFT grid of size M >> n (two-sided, per sample); returns gamma(0..n-1)."""
    g = np.real(np.fft.ifft(psd))
    return g[:n]


def kl_gauss(s0: np.ndarray, s1: np.ndarray) -> float:
    """KL(N(0, S1) || N(0, S0))."""
    c0 = linalg.cho_factor(s0, lower=True)
    a = linalg.cho_solve(c0, s1)
    n = s0.shape[0]
    sign, logdet = np.linalg.slogdet(a)
    return float(0.5 * (np.trace(a) - n - logdet))


def main() -> None:
    rows = []
    m_fft = 1 << 20
    w = 2 * np.pi * np.fft.fftfreq(m_fft)
    for bn in (5.0, 35.0):
        loop = CarrierLoop.gnsssdr(bn, T, kappa=0.0)
        hd2 = np.abs(loop.freq_response(np.abs(w), "disc")) ** 2
        hp2 = np.abs(loop.freq_response(np.abs(w))) ** 2
        var_e0 = float(np.mean(hp2))  # sigma^2 = 1
        for fc in (3.0, 1.0, 0.3, 0.1, 0.03):
            sd = ar1_psd(np.abs(w), fc)
            scale = var_e0 / float(np.mean(hp2 * sd))  # Delta = Var(e0): the disturbance doubles Var(e)
            sd = sd * scale
            for w_s in (2.0, 10.0):
                n = int(w_s / T)
                g0 = autocov_from_psd(hd2 * 1.0, n)
                g1 = autocov_from_psd(hd2 * (1.0 + sd), n)
                s0 = linalg.toeplitz(g0)
                s1 = linalg.toeplitz(g1)
                kl = kl_gauss(s0, s1)
                tv = min(1.0, np.sqrt(kl / 2))
                rows.append({"bn_cfg_hz": bn, "corner_hz": fc, "W_s": w_s, "kl_d_channel": kl,
                             "tv_upper": tv, "minimax_rel_error_lower_bound": 0.5 * (1 - tv)})  # fmt: skip
                print(rows[-1])
    part_b = equivalence_sim()
    print(part_b)
    save("s13_lecam_bound", {"delta_over_var_e0": 1.0, "rows": rows, "part_b_equivalence": part_b})


def equivalence_sim() -> list[dict[str, float]]:
    from scipy import signal as sg

    from gnssaudit.uq import iw
    from review.s02_window_and_p5 import ar1

    out = []
    for bn, fc in ((5.0, 1.0), (5.0, 3.0), (35.0, 3.0), (35.0, 10.0)):
        loop = CarrierLoop.gnsssdr(bn, T, kappa=0.0)
        rng = np.random.default_rng(13)
        n = 2_000_000
        sig = 0.03

        def proc(rng: np.random.Generator = rng, n: int = n, fc: float = fc, sig: float = sig) -> np.ndarray:
            return 2 * sig * ar1(rng, n, fc)

        m0, q0 = sig * rng.standard_normal(n), sig * rng.standard_normal(n)
        # P1: additive circular disturbance in both quadratures
        m1, q1 = m0 + proc(), q0 + proc()
        e1 = loop.filter(q1)
        d1 = loop.filter(q1, "disc")
        r1 = iw.iw_windows(1 + m1, d1, loop, iw.IWConfig(window=2000))
        # P0: amplitude-only a and phase-only theta with the same spectrum
        a, theta = proc(), proc()
        e0 = loop.filter(theta + q0) - theta
        d0 = loop.filter(theta + q0, "disc")
        r0 = iw.iw_windows((1 + a) * (1 + m0), d0, loop, iw.IWConfig(window=2000))
        _, s_d1 = sg.welch(d1, nperseg=4096)
        _, s_d0 = sg.welch(d0, nperseg=4096)
        out.append({"bn_cfg_hz": bn, "corner_hz": fc,
                    "true_var_e_P1_over_P0": float(np.var(e1[10_000:]) / np.var(e0[10_000:])),
                    "iw_noise_P1_over_P0": float(r1.noise.mean() / r0.noise.mean()),
                    "iw_dynraw_P1_minus_P0_rel": float((r1.dyn_raw.mean() - r0.dyn_raw.mean()) / r0.noise.mean()),
                    "max_rel_diff_psd_d": float(np.max(np.abs(s_d1 / s_d0 - 1)[1:200]))})  # fmt: skip
    return out


if __name__ == "__main__":
    main()
