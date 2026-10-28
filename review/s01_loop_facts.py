"""S01: which loop GNSS-SDR actually runs, and its effective bandwidth (findings F1, F20; docs/9 section 5).

Source facts (GNSS-SDR v0.0.22, commit 3fcb502a):
* dll_pll_veml_tracking.cc:757-764 enables extended integration only if extend_correlation_symbols > 1;
  lines 2620-2639 switch to pll_bw_narrow_hz only on that path. dll_pll_conf.h:50-52,74: pll_bw_hz = 35,
  pll_bw_narrow_hz = 5, extend_correlation_symbols = 1. So the default loop is configured at 35 Hz.
* Replica timing: block k is correlated with the frequency set after block k-1 (update_tracking_vars,
  lines 1627-1657), so the correlator averages psi_k = phi_k + T f_{k-1}/2 (kappa = 1/2).
"""

from __future__ import annotations

import numpy as np

from gnssaudit.uq.loop import CarrierLoop
from review._common import save


def describe(loop: CarrierLoop) -> dict[str, float]:
    return {
        "bn_eff_hz": loop.bn_eff_hz(),
        "pole_radius_max": float(np.max(np.abs(loop.poles()))),
        "sensitivity_peak_db": loop.sensitivity_peak_db(),
        "crossover_hz": loop.crossover_hz(),
    }


def main() -> None:
    cells = [
        ("C/A default (T=1 ms, pll_bw_hz=35)", 35.0, 1e-3, None),
        ("C/A narrow value if it were used (5 Hz)", 5.0, 1e-3, None),
        ("C/A 15 Hz", 15.0, 1e-3, None),
        ("E1 default (T=4 ms, 35 Hz)", 35.0, 4e-3, None),
        ("E1 5 Hz", 5.0, 4e-3, None),
        ("C/A extended 20 ms, narrow 5 Hz", 5.0, 20e-3, 1e-3),
        ("C/A extended 20 ms, narrow 15 Hz", 15.0, 20e-3, 1e-3),
    ]
    table = []
    for label, bn, t, tc in cells:
        simplified = CarrierLoop.gnsssdr(bn, t, kappa=0.0)
        faithful = CarrierLoop.gnsssdr(bn, t, kappa=0.5, t_code_s=tc)
        table.append(
            {"cell": label, "configured_hz": bn, "t_s": t, "t_code_s": tc,
             "simplified_docs9": describe(simplified), "faithful": describe(faithful)}
        )  # fmt: skip

    # inverse map with the faithful timing (docs/9 section 5.3 table, D-21)
    inverse = []
    for t, tc in [(1e-3, None), (4e-3, None), (20e-3, 1e-3)]:
        for target in (5.0, 15.0):
            row: dict[str, object] = {"t_s": t, "target_bn_eff_hz": target}
            for name, kappa, tcode in [("simplified_docs9", 0.0, None), ("faithful", 0.5, tc)]:
                try:
                    loop = CarrierLoop.gnsssdr_for_bn_eff(target, t, kappa=kappa, t_code_s=tcode)
                    row[name] = {"configured_hz": loop.bn_hz, **describe(loop)}
                except ValueError as exc:
                    row[name] = {"error": str(exc)}
            inverse.append(row)

    # consequences of F1 for numbers used across the notes (C/A, T = 1 ms)
    cn0 = 10**4.5
    conseq = {}
    for name, bn in [("assumed_5hz", 5.0), ("actual_35hz", 35.0)]:
        loop = CarrierLoop.gnsssdr(bn, 1e-3)
        b = loop.bn_eff_hz()
        conseq[name] = {
            "bn_eff_hz": b,
            "white_model_underestimate_factor_1_over_2BT": 1 / (2 * b * 1e-3),
            "thermal_sigma_rad_45dbhz": float(np.sqrt(b / cn0 * (1 + 1 / (2 * 1e-3 * cn0)))),
            "nu_eff_W2s_asymptotic": 2 * b * 2.0,
            "W_over_loop_time_constants_at_2s": 2.0 * b,
            "crossover_hz": loop.crossover_hz(),
            "cn0_memory_in_loop_time_constants": 0.5 * b,  # 0.5 s smoother memory at T = 1 ms
        }

    # F20: identity with a single H versus the pair (H_phi, H_d) under the faithful timing
    f20 = []
    for bn, t in [(5.0, 1e-3), (35.0, 1e-3), (35.0, 4e-3), (15.0, 20e-3)]:
        loop = CarrierLoop.gnsssdr(bn, t, t_code_s=1e-3 if t > 4e-3 else None)
        w = np.linspace(1e-6, np.pi, 400_001)
        hd = loop.freq_response(w, "disc")
        e1 = loop.error_response(w)
        hp = loop.freq_response(w)
        i_hd = np.trapezoid(np.abs(hd) ** 2, w) / np.pi
        i_e1 = np.trapezoid(np.abs(e1) ** 2, w) / np.pi
        i_hp = np.trapezoid(np.abs(hp) ** 2, w) / np.pi
        f20.append({
            "configured_hz": bn, "t_s": t,
            "white_m_error_of_single_H_identity_rel_to_Var_e": (i_hd - i_e1) / i_hp,
        })  # fmt: skip

    save("s01_loop_facts", {"cells": table, "inverse_map": inverse, "f1_consequences": conseq, "f20": f20})


if __name__ == "__main__":
    main()
