"""S15: does first-order Tikhonov theory predict slips of GNSS-SDR's third-order Costas loop? (F18; docs/4 section 4).

docs/4 S1 uses T_slip = pi^2 rho I0(rho)^2 / (2 B_L) with rho_Costas = rho / 4 as the theory offset. Here: the
nonlinear closed-loop simulator (Costas atan(Q/I), faithful timing, data bits) at low C/N0, half-cycle slips
measured as the first-passage rate of round(2 phi) away from 0 (a half-cycle slip or loss of lock, persisting
100 ms) over 32 independent runs, against that formula with rho = 1 / sigma^2_phi from the thermal model with
squaring loss and B_L = B_n,eff.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import special, stats

from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from gnssaudit.uq.loop import CarrierLoop
from review._common import save

T = 1e-3


def tikhonov_mtbs(rho: float, bl: float) -> float:
    return float(np.pi**2 * rho * special.i0(rho) ** 2 / (2 * bl))


def first_slip_index(phi: np.ndarray, hold: int = 100) -> int | None:
    """First block where round(2 phi) leaves 0 and stays off 0 for `hold` blocks (half-cycle slip or lock loss)."""
    off = np.round(2 * phi) != 0
    if not off.any():
        return None
    run = np.convolve(off.astype(int), np.ones(hold, dtype=int), mode="valid")
    idx = np.nonzero(run == hold)[0]
    return int(idx[0]) if idx.size else None


def cell(args: tuple[float, float, int]) -> dict[str, object]:
    """First-passage rate: events / exposure over R independent runs (exponential MLE, right-censored)."""
    bn, cn0_db, seed = args
    cn0 = 10 ** (cn0_db / 10)
    rng = np.random.default_rng(seed)
    r, k, burn = 32, 300_000, 5000
    cfg = TrackingConfig(bn_hz=bn, t_s=T, pilot=False)
    bits = np.repeat(np.where(rng.random((r, k // 20)) < 0.5, -1.0, 1.0), 20, axis=1)
    out = track(cfg, np.zeros((r, k)), thermal_noise(rng, (r, k), cn0, T), bits=bits)
    events, exposure = 0, 0.0
    for j in range(r):
        i = first_slip_index(out.phi[j, burn:])
        if i is None:
            exposure += (k - burn) * T
        else:
            events += 1
            exposure += i * T
    loop = CarrierLoop.gnsssdr(bn, T)
    var_phi = loop.bn_eff_hz() / cn0 * (1 + 1 / (2 * T * cn0))
    rho_costas = 1 / var_phi / 4
    th = tikhonov_mtbs(rho_costas, loop.bn_eff_hz())
    mtbs = exposure / events if events else float("inf")
    lo = exposure / stats.chi2.ppf(0.975, 2 * events + 2) * 2
    hi = exposure / stats.chi2.ppf(0.025, 2 * events) * 2 if events else float("inf")
    return {"bn_cfg_hz": bn, "cn0_dbhz": cn0_db, "runs": r, "events": events, "exposure_s": exposure,
            "mtbs_sim_s": mtbs, "mtbs_sim_95ci": [lo, hi], "mtbs_tikhonov_s": th,
            "rho_costas": rho_costas}  # fmt: skip


def main() -> None:
    # rho_Costas = 1 / (4 sigma^2) between about 2 and 6, where a tracking loop slips but stays locked
    jobs = [(5.0, c, 11 + i) for i, c in enumerate((24.0, 26.0, 28.0, 30.0, 32.0))]
    jobs += [(35.0, c, 31 + i) for i, c in enumerate((28.0, 30.0, 32.0, 34.0, 36.0))]
    with ProcessPoolExecutor(max_workers=10) as ex:
        rows = list(ex.map(cell, jobs))
    for r in rows:
        print(r)
    save("s15_slips", {"rows": rows})


if __name__ == "__main__":
    main()
