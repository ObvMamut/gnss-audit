"""S27: is the F18 slip gap one effective loop-SNR offset? (F18; docs/4 section 4; docs/12 S8).

S15 found the nonlinear third-order Costas loop slipping 38 to 6500 times sooner than first-order Tikhonov theory.
Because the mean time between slips grows like exp(2 rho), a ratio that large can come from a few dB of loop SNR.
Here the theory's loop SNR is scaled, rho -> kappa rho:
  - per cell with events, kappa is solved from the simulated mean time between slips and from its 95 % CI ends;
  - per loop, one kappa is fitted by maximum likelihood over every cell, zero-event cells included
    (first-passage times with right-censoring give events ~ Poisson(exposure / T_slip(kappa rho, B_L))),
    and the Poisson deviance tests whether that single offset fits. Cells whose expected count is negligible add
    almost nothing to the deviance, so the p-value with df = cells - 1 is conservative.
Reads review/results/s15_slips.json; run s15 first.
"""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
from scipy import optimize, stats

from gnssaudit.uq.loop import CarrierLoop
from review._common import RESULTS, save
from review.s15_slips import T, tikhonov_mtbs


def kappa_for(mtbs_s: float, rho: float, bl: float) -> float:
    """The kappa at which first-order theory gives the mean time between slips mtbs_s."""
    root = cast(float, optimize.brentq(lambda x: np.log(tikhonov_mtbs(x, bl)) - np.log(mtbs_s), 1e-6, 5 * rho))
    return root / rho


def expected_events(kappa: float, cell: dict[str, Any]) -> float:
    return cell["exposure_s"] / tikhonov_mtbs(kappa * cell["rho_costas"], cell["bl_hz"])


def deviance(kappa: float, cells: list[dict[str, Any]]) -> float:
    d = 0.0
    for c in cells:
        e, mu = c["events"], expected_events(kappa, c)
        d += 2 * ((e * np.log(e / mu) if e else 0.0) - (e - mu))
    return d


def fit_loop(cells: list[dict[str, Any]]) -> dict[str, object]:
    """One kappa for all cells of a loop: Poisson maximum likelihood and deviance goodness of fit."""

    def nll(k: float) -> float:
        return sum(expected_events(k, c) - c["events"] * np.log(expected_events(k, c)) for c in cells)

    res = cast(optimize.OptimizeResult, optimize.minimize_scalar(nll, bounds=(0.05, 1.5), method="bounded"))
    k = float(res.x)
    dev, df = deviance(k, cells), len(cells) - 1
    return {"bn_cfg_hz": cells[0]["bn_cfg_hz"], "kappa_hat": k, "penalty_db": -10 * np.log10(k),
            "deviance": dev, "df": df, "p_value": float(stats.chi2.sf(dev, df)),
            "cells": [{"cn0_dbhz": c["cn0_dbhz"], "events": c["events"], "expected": expected_events(k, c)} for c in cells]}  # fmt: skip


def main() -> None:
    rows: list[dict[str, Any]] = json.loads((RESULTS / "s15_slips.json").read_text())["rows"]
    for r in rows:
        r["bl_hz"] = CarrierLoop.gnsssdr(r["bn_cfg_hz"], T).bn_eff_hz()
        # the B_L recomputed here must be the one s15 used for its theory column
        assert np.isclose(tikhonov_mtbs(r["rho_costas"], r["bl_hz"]), r["mtbs_tikhonov_s"], rtol=1e-9)
    per_cell = []
    for r in rows:
        if not r["events"]:
            continue
        lo, hi = r["mtbs_sim_95ci"]
        k = kappa_for(r["mtbs_sim_s"], r["rho_costas"], r["bl_hz"])
        k_ci = [kappa_for(lo, r["rho_costas"], r["bl_hz"]), kappa_for(hi, r["rho_costas"], r["bl_hz"])]
        per_cell.append({"bn_cfg_hz": r["bn_cfg_hz"], "cn0_dbhz": r["cn0_dbhz"], "events": r["events"],
                         "ratio_theory_over_sim": r["mtbs_tikhonov_s"] / r["mtbs_sim_s"],
                         "kappa": k, "kappa_95ci": k_ci, "penalty_db": -10 * np.log10(k)})  # fmt: skip
    loops = [fit_loop([r for r in rows if r["bn_cfg_hz"] == bn]) for bn in sorted({r["bn_cfg_hz"] for r in rows})]
    for c in per_cell:
        print(c)
    for f in loops:
        print(f)
    save("s27_slip_offset", {"per_cell": per_cell, "per_loop": loops})


if __name__ == "__main__":
    main()
