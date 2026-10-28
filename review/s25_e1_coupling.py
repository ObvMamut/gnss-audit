"""S25: why the CW coupled 2.6-2.9x more weakly into the binary's E1 pilot correlator than designed (S22).

1. Measured coupling: from the S22 tracking dumps, the pilot prompt is rebuilt as abs(P) exp(j 2 pi d) (atan2
   discriminator). The in-phase variance is not loop-filtered, so 2 [Var(I)/A^2]_jammed - 2 [Var(I)/A^2]_clean
   estimates the despread interference power relative to A^2.
2. Explanation: S22 placed the CW at f_D + 1.023 MHz, the BOC(1,1) subcarrier frequency. At 8 Msps (3.91 samples
   per half-chip) the despread power of a line there depends strongly on the sub-sample alignment of the sampled
   replica. The receiver's code phase drifts, so it averages over alignments, whereas S22's design used one fixed
   alignment. Reported: the alignment sensitivity at several offsets and the alignment-averaged coupling.

Needs the S22 dumps in the scratchpad (run S22 first).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from review._common import save
from review.s17_gnsssdr_identification import DUMP_DTYPE
from review.s22_gnsssdr_e1_pilot import CN0_PILOT_DB, DOPPLER, FS, OUT, T, boc, e1_code


def _prompt(name: str, t_min: float) -> NDArray[np.complex128]:
    r = np.fromfile(OUT / f"trk_{name}0.dat", dtype=DUMP_DTYPE)
    t = r["prn_start_sample"].astype(np.float64) / FS
    s = t > t_min
    p = r["abs_p"][s].astype(float) * np.exp(2j * np.pi * r["carr_err_cyc"][s].astype(float))
    return np.asarray(p, dtype=np.complex128)


def p_unit(f_cw: float, phase: float, code: np.ndarray, blocks: int = 100) -> float:
    rate = 2 * 1.023e6
    ns = int(round(FS * code.size / rate))
    idx = np.floor(np.arange(ns) * rate / FS + phase).astype(int) % code.size
    tmpl = code[idx] * np.exp(-2j * np.pi * DOPPLER * np.arange(ns) / FS)
    x = np.exp(2j * np.pi * f_cw * np.arange(ns * blocks) / FS).reshape(blocks, ns)
    rot = np.exp(-2j * np.pi * DOPPLER * np.arange(blocks) * ns / FS)
    return float(np.mean(np.abs(rot * (x @ tmpl) / ns) ** 2))


def main() -> None:
    cn0 = 10 ** (CN0_PILOT_DB / 10)
    designed = 4 / (2 * cn0 * T)
    pc = _prompt("e1clean", 15.0)
    a_c = np.mean(pc.real)
    measured = {}
    for f_b in (2, 5):
        pj = _prompt(f"e1cw{f_b}", 30.0)
        a_j = np.mean(pj.real)
        excess = 2 * (np.var(pj.real) / a_j**2 - np.var(pc.real) / a_c**2)
        measured[f"beat_{f_b}hz"] = {"interference_over_A2": float(excess), "ratio_to_design": float(excess / designed),
                                     "amplitude_ratio": float(a_j / a_c)}  # fmt: skip
    code = boc(e1_code("C"))
    phases = np.linspace(0, 1, 201)[:-1]
    sensitivity = {}
    for off_mhz in (0.5, 0.86, 1.023, 2.046):
        v = np.array([p_unit(DOPPLER + off_mhz * 1e6 + 2.0, ph, code) for ph in phases[::10]])
        sensitivity[f"{off_mhz}MHz"] = {"max_over_min": float(v.max() / v.min())}
    averaged = {}
    for f_b in (2.0, 5.0):
        v = np.array([p_unit(DOPPLER + 1.023e6 + f_b, ph, code) for ph in phases])
        averaged[f"beat_{f_b:g}hz"] = {
            "phase0": float(v[0]),
            "mean": float(v.mean()),
            "mean_over_phase0": float(v.mean() / v[0]),
        }
    print(measured, sensitivity, averaged)
    save("s25_e1_coupling", {"measured": measured, "alignment_sensitivity": sensitivity, "alignment_average": averaged})


if __name__ == "__main__":
    main()
