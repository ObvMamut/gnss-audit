"""S26: B1's residual has a thermal floor when the discriminator is atan (Costas) (P2b task 11; docs/3 section 2.2).

Two nonlinear third-order loops (GNSS-SDR's coefficients, T = 1 ms) see the same signal and the same thermal noise;
the jammed one also sees complex Gaussian interference iota. The first-order prediction is the same loop, linear,
driven by n_iota = Im(iota) / A. The atan discriminator normalises by the noisy in-phase prompt, so products of
thermal noise and interference are first order in iota, and the residual ratio r = Var(dPhi - prediction) / Var(dPhi)
does not vanish as the interference weakens: r ~ 1 / (C/N0 T) + Var(n_iota), at any loop bandwidth.

Usage: uv run python -m review.s26_b1_floor
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from review._common import save

T_S = 1e-3
N = 40_000
SETTLE = 2_000
CASES = [(45.0, 0.112, 35.0), (45.0, 0.112, 5.0), (45.0, 0.03, 35.0), (45.0, 0.3, 35.0), (55.0, 0.112, 35.0),
         (55.0, 0.03, 35.0)]  # fmt: skip


def loop_run(disc: Callable[[int, float], float], bn_hz: float) -> np.ndarray:
    """A third-order loop with GNSS-SDR's filter coefficients (w0 = Bn / 0.7845, a3 = 1.1, b3 = 2.4)."""
    w0 = bn_hz / 0.7845
    a3, b3 = 1.1, 2.4
    w = x = phase = 0.0
    out = np.empty(N)
    for k in range(N):
        out[k] = phase
        d = disc(k, phase)
        w_new = w + T_S * w0**3 * d
        x = x + T_S * (0.5 * (w_new + w) + a3 * w0**2 * d)
        w = w_new
        phase += (0.5 * x + b3 * w0 * d) * T_S
    return out


def trial(cn0_db: float, sigma_niota: float, bn_hz: float, seed: int = 1) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sw = 1.0 / np.sqrt(2 * 10 ** (cn0_db / 10) * T_S)  # thermal SD per component, signal amplitude 1
    w = sw * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    iota = sigma_niota * (rng.standard_normal(N) + 1j * rng.standard_normal(N))

    def costas(extra: np.ndarray) -> Callable[[int, float], float]:
        def d(k: int, ph: float) -> float:
            p = (1.0 + w[k] + extra[k]) * np.exp(-1j * ph)
            return float(np.arctan(p.imag / p.real))

        return d

    clean = loop_run(costas(np.zeros(N, dtype=complex)), bn_hz)
    jam = loop_run(costas(iota), bn_hz)
    n_iota = iota.imag
    lin = loop_run(lambda k, ph: float(n_iota[k] - ph), bn_hz)
    dphi, pred = (jam - clean)[SETTLE:], lin[SETTLE:]
    return {
        "cn0_dbhz": cn0_db,
        "sigma_n_iota_rad": sigma_niota,
        "bn_hz": bn_hz,
        "r": float(np.var(dphi - pred) / np.var(dphi)),
        "floor_plus_var_n_iota": float(1 / (10 ** (cn0_db / 10) * T_S) + sigma_niota**2),
        "corr": float(np.corrcoef(dphi, pred)[0, 1]),
    }


def main() -> None:
    rows = [trial(*c) for c in CASES]
    for r in rows:
        print(f"{r['cn0_dbhz']:4.0f} dB-Hz  sigma(n_iota) {r['sigma_n_iota_rad']:.3f}  Bn {r['bn_hz']:4.0f} Hz  ->  "
              f"r {r['r']:.4f}   1/(C/N0 T) + Var(n_iota) {r['floor_plus_var_n_iota']:.4f}")  # fmt: skip
    save("s26_b1_floor", {"t_s": T_S, "n": N, "settle": SETTLE, "rows": rows})


if __name__ == "__main__":
    main()
