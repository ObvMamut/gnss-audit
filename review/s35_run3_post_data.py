"""S35: run 3 against run 2, s29 under s34's rule, the rule's error rate, and B1 near its bar (P2c task 9).

Exploratory and post-data. Written on 28 Oct 2026, after run 3's data (9650dc00e766), after s34's verdict and after the
independent referee's report (plan workspace .superpowers/sdd/2026-10-24-p2c-g3-fallback-ghost-truth/referee-report.md).
It decides nothing: no rule of R-0 to R-8, no verdict and no range changes with it, and every number it gives is
"conditional, exploratory". It exists so that the comparisons the referee computed in a session scratchpad, which does
not persist, can be cited from a committed file (ruling T9a). Where it does not reproduce a referee number, the docs
quote its value.

Questions.
  1. Replication. Does run 3 reproduce run 2's mode B / mode A ratios R, cell by cell, and which rule (a') verdicts
     changed between the two runs?
  2. s29 under s34's rule. s29 (review/results/s29_true_analog_jmax.json, committed before run 3) is the other
     simulation of mode B. How does it fare against run 3 under the rule s34 applied to s33, and how far apart are the
     two simulations' R_B at weak jamming?
  3. The rule's error rate. How often does s34's rule (accept iff sum z^2 <= chi2_10(0.95) and max |z| < 3) reject a
     correct model, with independent and with correlated z? How robust are s34's two rejections to correlation?
  4. B1 near its bar. Which ghost cells have r - 1/(C/N0 T) within 0.005 of the 0.05 bar in run 2 or run 3, and how
     much does that quantity move from one draw to the next?

Method.
  1. Per segment, for noise and swept CW at 50-70 dB-Hz: z_c = (R_run3 - R_run2) / sqrt(SE_run3^2 + SE_run2^2), from
     segments[s].front_end[family][level].ratio and .ratio_se of the two summaries. Run 2's cells are put in s33's
     schema (cells[s][family][level].ratio_B, .ratio_B_se; `as_model`) so that s34's own `accept` computes z, sum z^2
     and max |z|. Its "accepted" field is reported as `within_s34_bounds`: it accepts nothing. Beside it: the number of
     positive z, P(chi2_n >= sum z^2) (independent z) and P(chi2_1 >= sum z^2 / n) (fully correlated z, the largest
     tail found in item 3). The verdicts are each summary's own (.verdict).
  2. s29's cells hold ratio_B and ratio_B_se at the key paths s34 reads from s33 (cells[s][family][level]), so s34's
     `accept` reads s29 unchanged; no mapping is needed for the rule. s29 differs from s33's schema elsewhere: its
     levels stop at 70 dB-Hz, and its true front end is `ratio_truth` (one stationary AGC), not s33's `ratio_TA` per
     tau, so s34's J_max_TA and conditional range do not apply to it and are not computed. No rule admits s29 (R-7
     names s33): the result is reported only. `require_cells` checks that each of the 20 cells is present and finite
     in every source before any z is computed. Run 2 against s29 is reported beside it (in sample for s29). Run 3
     against s33 is recomputed the same way and checked against s34's JSON. R_B(s29) - R_B(s33) is given per cell; its
     range at 50-55 dB-Hz is the summary.
  3. Monte Carlo of the rule under a correct model: z ~ N(0, C(rho)), C(rho) the n = 10 equicorrelation matrix, drawn
     as z_i = sqrt(rho) u + sqrt(1 - rho) e_i, for rho in RHOS, N_SIZE draws per rho, numpy default_rng(SEED), the rhos
     in that order on one generator. The rule rejects iff sum z^2 > chi2_10(0.95) or max |z| >= 3 (s34's `chi2_bound`
     and `MAX_ABS_Z`; `rule_rejects`, tested against s34's `accept`). Exact values beside the simulated ones: at
     rho = 1 every z equals one N(0, 1) draw u, so P(reject) = P(10 u^2 > chi2_10(0.95)) (sqrt(1.8307) < 3) and
     P(sum z^2 >= t) = P(chi2_1 >= t / 10); at rho = 0, P(sum z^2 >= t) = P(chi2_10 >= t).
     Robustness of s34's rejections: P(sum z^2 >= t) at s34's own sum z^2 (read from s34's JSON), under each rho, on
     the same draws. Beyond equicorrelation: for a correlation matrix with eigenvalues lambda (lambda >= 0,
     sum lambda = n), sum z^2 = sum_j lambda_j w_j^2 with w iid N(0, 1), so its law depends on lambda alone; and every
     such lambda belongs to some correlation matrix (Schur-Horn: the unit diagonal is majorised by any lambda summing to
     n). The tail is evaluated (a) exactly on the k-equal family, lambda = n/k on k coordinates, P = P(chi2_k >= t k/n),
     k = 1..10; (b) by Monte Carlo, with common random numbers (N_EIG draws of w, default_rng(SEED + 1)), on the
     two-point family lambda = n (a, 1 - a) and on N_DIRICHLET flat-Dirichlet lambda drawn first from the same
     generator. The worst case found is reported with where it was found. A numerical check, not a proof.
  4. Per ghost, r - floor in mode A (segments[s].residual[family][level][PRN].r - .floor) in both runs, for the PRNs
     both runs hold. Listed: every cell within NEAR of the bar in either run. The spread: the run 3 - run 2 differences
     over noise and swept CW at 50-60 dB-Hz, both segments and both ghosts (24 cells): their SD (ddof 1), RMS, mean
     |difference| and range, and SD / sqrt(2) as the per-draw SD. A two-draw estimate: each cell has one draw per run,
     and pooling the cells assumes one per-draw spread for all of them.

Inputs, read-only, pinned by SHA-256 (the study stops on a mismatch): E2 run 3 (9650dc00e766) and run 2 (6214cf52489f)
summaries; s29, s33 and s34's JSON.

The referee's values this study was written to reproduce (referee report, appendix A1 to A3):
  1. run 3 - run 2: A sum z^2 7.22, max |z| 1.54, 5 of 10 z positive; B 4.09, 0.97, 4 of 10. A's 55 dB-Hz verdicts flip:
     noise fail (0.868) to pass (0.919), z +0.98; swept CW pass (0.908) to fail (0.880), z -0.70.
  2. run 3 - s29: A 15.39, 2.04, 5 of 10; B 59.05, 5.44, 0 of 10. run 2 - s29: A 16.96, 2.86; B 51.88, 4.67.
     s29 - s33 at 50-55 dB-Hz: 0.11 to 0.28.
  3. P(reject | correct model): 0.061 (rho 0), 0.077 (0.2), 0.105 (0.4), 0.133 (0.6), 0.157 (0.8), 0.176 (1).
     At rho 1: P(sum z^2 >= 64.92) = 0.0110 and P(sum z^2 >= 42.79) = 0.0383.
  4. 24 cells: SD 0.0039, mean |d| 0.0029, per-draw SD about 0.003; at 50 dB-Hz the 8 differences run from -0.0106 to
     +0.0048; run 3's noise margins at 50 dB-Hz 0.0021 (A, PRN 31) and 0.0014 (B, PRN 29).

Result (review/results/s35_run3_post_data.json; deterministic: every draw is seeded). Every referee value above
reproduces to its printed digits except where noted.
  1. run 3 - run 2 (`replication`): A sum z^2 7.22, max |z| 1.54 (swept CW 70), 5 of 10 positive; B 4.09, 0.97 (noise
     60), 4 of 10; P(chi2_1 >= sum z^2 / 10) 0.396 and 0.522. Three verdicts flip, not two: A noise 55 fail -> pass
     (0.868 -> 0.919, z +0.98), A swept CW 55 pass -> fail (0.908 -> 0.880, z -0.70), and B noise 60 fail -> pass
     (0.898 -> 0.921, z +0.97), which the referee did not list. B fails at 55 dB-Hz for both families in both runs.
  2. run 3 - s29: A 15.39, 2.04 (swept CW 60), 5 of 10 positive, within s34's bounds; B 59.05, 5.44 (swept CW 65), 0 of
     10, outside them. run 2 - s29: A 16.96, 2.86; B 51.88, 4.67. run 3 - s33 reproduces s34's z exactly. R_B(s29) -
     R_B(s33) at 50-55 dB-Hz is 0.107 to 0.276 over the 8 cells (s29 above s33 at each); the referee printed 0.11 to
     0.28 over the five cells it listed.
  3. P(reject | correct model), +- MC SE 0.0004 to 0.0006: 0.0611 (rho 0), 0.0765 (0.2), 0.1042 (0.4), 0.1194 (0.5),
     0.1317 (0.6), 0.1563 (0.8), 0.1764 (1; exact 0.1760). The referee's simulation, on other draws, printed 0.105
     (0.4), 0.133 (0.6) and 0.157 (0.8); the differences, up to about 0.0013, are within about 2.5 combined MC SE. At
     rho 1, exactly: P(sum z^2 >= 64.92) = 0.0108 (A) and P(sum z^2 >= 42.79) = 0.0386 (B); simulated here 0.0109 and
     0.0386 (the referee's simulation: 0.0110 and 0.0383). Over the eigenvalue vectors checked, the largest tail is at
     one eigenvalue n, which is rho 1: the k-equal family's worst is k = 1, the two-point family's a = 1, and the
     flat-Dirichlet worst is 0.0031 (A) and 0.0178 (B).
  4. Within 0.005 of the bar in either run: 11 ghost cells, all noise or swept CW at 50, 55 or 65 dB-Hz (`b1.near_bar`).
     The 24 cells: SD 0.0039, RMS 0.0039, mean |d| 0.0029, range -0.0106 to +0.0061; per-draw SD 0.0028. At 50 dB-Hz
     the 8 differences run from -0.0106 to +0.0048. Run 3's 50 dB-Hz margins (larger ghost): noise A 0.0021 (PRN 31),
     B 0.0014 (PRN 29); swept CW A 0.0011 (PRN 31), B 0.0009 (PRN 30).

Usage: uv run python -m review.s35_run3_post_data (reads the pinned files read-only; under a minute).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chi2, norm

from review._common import RESULTS, save
from review.s30_k_rule_null import SourceRefused
from review.s32_g3_rerun_forecast import RUN2_ID, RUN2_SUMMARY, RUN2_SUMMARY_SHA256, S29_JSON, S29_SHA256, pinned_json
from review.s33_front_end_prediction import DECIDING, SEGMENTS, level_key
from review.s34_front_end_acceptance import (
    ACCEPTED_LEVELS,
    MAX_ABS_Z,
    RUN3_DIR,
    RUN3_ID,
    RUN3_SUMMARY_SHA256,
    S33_JSON,
    S33_SHA256,
    accept,
    chi2_bound,
)

FloatArray = NDArray[np.float64]

S34_JSON = RESULTS / "s34_front_end_acceptance.json"
S34_SHA256 = "b0a194d5163c10cb66e4bd46ef1ab9b2d6a0431750c0eb140bab14643374f1e9"  # commit 2c809bd

N_CELLS = len(DECIDING) * len(ACCEPTED_LEVELS)  # 10 per segment
SEED = 20261028
RHOS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0)
N_SIZE = 400_000  # draws per rho
N_EIG = 200_000  # draws of w for the eigenvalue check
N_DIRICHLET = 1_000
TWO_POINT = tuple(round(0.5 + 0.05 * i, 2) for i in range(11))  # a = 0.5, 0.55, ..., 1.0
B1_BAR = 0.05  # E2-v3.yaml rules.residual_max: B1 passes iff r - floor < 0.05
NEAR = 0.005  # item 4: cells within this of the bar
SPREAD_LEVELS = ("50.0", "55.0", "60.0")
B1_FAMILIES = ("noise", "swept_cw", "chirp")
B1_LEVELS = tuple(level_key(lv) for lv in ACCEPTED_LEVELS)


# ------------------------------------------------------------------------------------------------ item 1 and item 2
def as_model(summary: Mapping[str, Any]) -> dict[str, Any]:
    """An E2 summary's front-end cells in s33's schema, cells[s][family][level].{ratio_B, ratio_B_se}, so that s34's
    `accept` can compare two runs. Only the noise and swept CW cells at 50-70 dB-Hz are carried."""
    cells: dict[str, Any] = {}
    for s in SEGMENTS:
        fe = summary["segments"][s]["front_end"]
        cells[s] = {f: {level_key(lv): {"ratio_B": fe[f][level_key(lv)]["ratio"],
                                         "ratio_B_se": fe[f][level_key(lv)]["ratio_se"]}
                        for lv in ACCEPTED_LEVELS} for f in DECIDING}  # fmt: skip
    return {"cells": cells}


def _number(x: object) -> float | None:
    """x as a float if it is a finite real number (not a bool), else None."""
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    return float(x) if math.isfinite(x) else None


def require_cells(model: Mapping[str, Any], name: str) -> None:
    """Refuse a model (s33's schema) unless each of the 20 noise and swept CW cells at 50-70 dB-Hz has a finite ratio_B
    and a finite, positive ratio_B_se: s34's `accept` would otherwise leave the cell out without a word."""
    for s in SEGMENTS:
        for f in DECIDING:
            for lv in ACCEPTED_LEVELS:
                c: object = model.get("cells")
                for key in (s, f, level_key(lv)):
                    c = c.get(key) if isinstance(c, Mapping) else None
                r = _number(c.get("ratio_B")) if isinstance(c, Mapping) else None
                se = _number(c.get("ratio_B_se")) if isinstance(c, Mapping) else None
                if r is None or se is None or se <= 0.0:
                    raise SourceRefused(f"{name}: cell {s}/{f}/{level_key(lv)} has no finite ratio_B and ratio_B_se")


def compare(run: Mapping[str, Any], model: Mapping[str, Any], segment: str) -> dict[str, Any]:
    """s34's `accept` of `run` (an E2 summary) against `model` (s33's schema) on one segment, with the positive z,
    P(chi2_n >= sum z^2) (independent z) and P(chi2_1 >= sum z^2 / n) (fully correlated z) beside it."""
    a = accept(run, model, segment)
    n, chi_sq = a["n"], a["chi2"]
    if n != N_CELLS:
        raise SourceRefused(f"segment {segment}: {n} cells present, not {N_CELLS}")
    return {
        "z": a["z"],
        "chi2": chi_sq,
        "n": n,
        "max_abs_z": a["max_abs_z"],
        "max_abs_z_cell": max(a["z"], key=lambda k: abs(a["z"][k])),
        "positive": sum(v > 0 for v in a["z"].values()),
        "within_s34_bounds": a["accepted"],
        "p_independent": float(chi2.sf(chi_sq, n)),
        "p_full_correlation": float(chi2.sf(chi_sq / n, 1)),
    }


def verdicts(run2: Mapping[str, Any], run3: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Each noise and swept CW cell's rule (a') verdict in both runs (the summaries' own), and the cells whose verdict
    differs, with R +- SE in each run."""
    table: dict[str, Any] = {}
    flips: list[dict[str, Any]] = []
    for s in SEGMENTS:
        for f in DECIDING:
            for lv in ACCEPTED_LEVELS:
                k = level_key(lv)
                c2, c3 = run2["segments"][s]["front_end"][f][k], run3["segments"][s]["front_end"][f][k]
                row = {"run2": c2["verdict"], "run3": c3["verdict"]}
                table.setdefault(s, {}).setdefault(f, {})[k] = row
                if row["run2"] != row["run3"]:
                    flips.append({"segment": s, "family": f, "level": k, **row,
                                  "run2_ratio": c2["ratio"], "run2_se": c2["ratio_se"],
                                  "run3_ratio": c3["ratio"], "run3_se": c3["ratio_se"]})  # fmt: skip
    return table, flips


def rb_difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """R_B(a) - R_B(b) per segment, family and level (both in s33's schema)."""
    return {s: {f: {level_key(lv): float(a["cells"][s][f][level_key(lv)]["ratio_B"])
                                   - float(b["cells"][s][f][level_key(lv)]["ratio_B"])
                    for lv in ACCEPTED_LEVELS} for f in DECIDING} for s in SEGMENTS}  # fmt: skip


def cell_table(run2: Mapping[str, Any], run3: Mapping[str, Any], s29: Mapping[str, Any], s33: Mapping[str, Any]) -> Any:
    """Per noise and swept CW cell: R +- SE in run 2 and run 3, R_B +- SE in s29 and s33."""
    out: dict[str, Any] = {}
    for s in SEGMENTS:
        for f in DECIDING:
            for lv in ACCEPTED_LEVELS:
                k = level_key(lv)
                c2, c3 = run2["segments"][s]["front_end"][f][k], run3["segments"][s]["front_end"][f][k]
                m29, m33 = s29["cells"][s][f][k], s33["cells"][s][f][k]
                out.setdefault(s, {}).setdefault(f, {})[k] = {
                    "run2": [c2["ratio"], c2["ratio_se"]], "run3": [c3["ratio"], c3["ratio_se"]],
                    "s29": [m29["ratio_B"], m29["ratio_B_se"]], "s33": [m33["ratio_B"], m33["ratio_B_se"]],
                }  # fmt: skip
    return out


# ------------------------------------------------------------------------------------------------ item 3
def rule_rejects(z: FloatArray) -> NDArray[np.bool_]:
    """s34's rule on rows of z (one row per draw, all cells present): reject iff sum z^2 > chi2_n(0.95) or
    max |z| >= 3."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    return (np.sum(z * z, axis=1) > chi2_bound(z.shape[1])) | (np.max(np.abs(z), axis=1) >= MAX_ABS_Z)


def equicorrelated(rho: float, n_draws: int, n: int, rng: np.random.Generator) -> FloatArray:
    """n_draws rows of n standard normals with pairwise correlation rho: sqrt(rho) u + sqrt(1 - rho) e."""
    u = rng.standard_normal((n_draws, 1))
    e = rng.standard_normal((n_draws, n))
    return math.sqrt(rho) * u + math.sqrt(1.0 - rho) * e


def exact_reject_full_correlation(n: int = N_CELLS) -> float:
    """P(reject) at rho = 1: every z is one N(0, 1) draw u, so the rule rejects iff n u^2 > chi2_n(0.95) or |u| >= 3."""
    return float(2.0 * norm.sf(min(math.sqrt(chi2_bound(n) / n), MAX_ABS_Z)))


def size_and_tails(tails: Mapping[str, float], *, rng: np.random.Generator, n_draws: int = N_SIZE,
                   rhos: Sequence[float] = RHOS, n: int = N_CELLS) -> dict[str, Any]:  # fmt: skip
    """Per rho (on one generator, in order): P(reject | correct model) and P(sum z^2 >= t) for each named t."""
    out: dict[str, Any] = {}
    for rho in rhos:
        z = equicorrelated(rho, n_draws, n, rng)
        q = np.sum(z * z, axis=1)
        p = float(rule_rejects(z).mean())
        out[f"{rho:g}"] = {
            "p_reject": p,
            "mc_se": math.sqrt(p * (1.0 - p) / n_draws),
            "p_sum_z2_at_least": {k: float(np.mean(q >= t)) for k, t in tails.items()},
        }
    return out


def k_equal_tail(t: float, k: int, n: int = N_CELLS) -> float:
    """P(sum z^2 >= t) when the correlation matrix has k eigenvalues n/k and the rest 0: (n/k) chi2_k >= t."""
    return float(chi2.sf(t * k / n, k))


def eigen_check(tails: Mapping[str, float], *, rng: np.random.Generator, n_draws: int = N_EIG,
                n_dirichlet: int = N_DIRICHLET, n: int = N_CELLS) -> dict[str, Any]:  # fmt: skip
    """P(sum z^2 >= t) over eigenvalue vectors lambda (lambda >= 0, sum n): exact on the k-equal family; by Monte Carlo
    with common random numbers on the two-point family and on flat-Dirichlet lambda (drawn first). Reports the largest
    tail of each family and where it lies."""
    lam_d = rng.dirichlet(np.ones(n), size=n_dirichlet) * n
    w2 = rng.standard_normal((n_draws, n)) ** 2
    lam_2 = np.zeros((len(TWO_POINT), n))
    lam_2[:, 0] = [n * a for a in TWO_POINT]
    lam_2[:, 1] = [n * (1.0 - a) for a in TWO_POINT]
    out: dict[str, Any] = {}
    for name, t in tails.items():
        k_eq = {str(k): k_equal_tail(t, k, n) for k in range(1, n + 1)}
        p2 = {f"{a:g}": float(np.mean(w2 @ lam >= t)) for a, lam in zip(TWO_POINT, lam_2, strict=True)}
        pd = np.concatenate([np.mean(w2 @ lam_d[i : i + 100].T >= t, axis=0) for i in range(0, n_dirichlet, 100)])
        worst_k = max(k_eq, key=lambda k: k_eq[k])
        worst_a = max(p2, key=lambda a: p2[a])
        i_d = int(np.argmax(pd))
        out[name] = {
            "t": t,
            "k_equal_exact": k_eq,
            "k_equal_worst": {"k": int(worst_k), "p": k_eq[worst_k]},
            "two_point_mc": p2,
            "two_point_worst": {"a": float(worst_a), "p": p2[worst_a]},
            "dirichlet_mc_worst": {
                "p": float(pd[i_d]),
                "lambda": lam_d[i_d].tolist(),
                "largest_lambda_share": float(lam_d[i_d].max() / n),
            },  # fmt: skip
            "dirichlet_mc_median": float(np.median(pd)),
            "mc_se_at_rho1": math.sqrt(k_eq["1"] * (1.0 - k_eq["1"]) / n_draws),
        }
    return out


# ------------------------------------------------------------------------------------------------ item 4
def b1_values(summary: Mapping[str, Any]) -> dict[tuple[str, str, str, str], float]:
    """r - floor per (segment, family, level, PRN), mode A, from an E2 summary's `residual`."""
    out: dict[tuple[str, str, str, str], float] = {}
    for s in SEGMENTS:
        res = summary["segments"][s]["residual"]
        for f in B1_FAMILIES:
            for k in B1_LEVELS:
                for prn, c in (res.get(f, {}).get(k) or {}).items():
                    out[(s, f, k, str(prn))] = float(c["r"]) - float(c["floor"])
    return out


def near_bar(b2: Mapping[tuple[str, str, str, str], float], b3: Mapping[tuple[str, str, str, str], float],
             near: float = NEAR) -> list[dict[str, Any]]:  # fmt: skip
    """Cells both runs hold whose r - floor lies within `near` of the bar in either run, sorted by run 3's margin."""
    rows = []
    for key in sorted(set(b2) & set(b3)):
        d2, d3 = b2[key], b3[key]
        if min(abs(d2 - B1_BAR), abs(d3 - B1_BAR)) <= near:
            s, f, k, prn = key
            rows.append({"segment": s, "family": f, "level": k, "prn": prn, "run2": d2, "run3": d3,
                         "run2_margin": B1_BAR - d2, "run3_margin": B1_BAR - d3,
                         "run2_passes": d2 < B1_BAR, "run3_passes": d3 < B1_BAR})  # fmt: skip
    return sorted(rows, key=lambda r: abs(r["run3_margin"]))


def spread(b2: Mapping[tuple[str, str, str, str], float], b3: Mapping[tuple[str, str, str, str], float],
           families: Sequence[str] = DECIDING, levels: Sequence[str] = SPREAD_LEVELS) -> dict[str, Any]:  # fmt: skip
    """The run 3 - run 2 differences of r - floor over the cells both runs hold in `families` x `levels`: SD (ddof 1),
    RMS, mean |d|, range, and SD / sqrt(2) as the per-draw SD (a two-draw estimate)."""
    keys = sorted(k for k in set(b2) & set(b3) if k[1] in families and k[2] in levels)
    if len(keys) < 2:
        raise ValueError("the spread needs at least two cells held by both runs")
    d = np.array([b3[k] - b2[k] for k in keys])
    sd = float(np.std(d, ddof=1))
    return {
        "cells": len(keys),
        "sd": sd,
        "rms": float(np.sqrt(np.mean(d * d))),
        "mean_abs": float(np.mean(np.abs(d))),
        "min": float(d.min()),
        "max": float(d.max()),
        "per_draw_sd": sd / math.sqrt(2.0),
        "differences": {"/".join(k): float(v) for k, v in zip(keys, d, strict=True)},
    }


def margins(b: Mapping[tuple[str, str, str, str], float], level: str = "50.0") -> dict[str, Any]:
    """Per segment and family at one level: the larger ghost's r - floor, its PRN and its margin to the bar."""
    out: dict[str, Any] = {}
    for s in SEGMENTS:
        for f in DECIDING:
            cells = {k[3]: v for k, v in b.items() if k[0] == s and k[1] == f and k[2] == level}
            prn = max(cells, key=lambda p: cells[p])
            out.setdefault(s, {})[f] = {"prn": prn, "value": cells[prn], "margin": B1_BAR - cells[prn]}
    return out


# ------------------------------------------------------------------------------------------------ the study
def read_inputs() -> dict[str, Any]:
    if RUN3_SUMMARY_SHA256 is None or S33_SHA256 is None:
        raise SourceRefused("s34's pins are not set")
    return {
        "run3": pinned_json(RUN3_DIR / "summary.json", RUN3_SUMMARY_SHA256, "E2 run 3 summary"),
        "run2": pinned_json(RUN2_SUMMARY, RUN2_SUMMARY_SHA256, "E2 run 2 summary"),
        "s29": pinned_json(S29_JSON, S29_SHA256, "s29 result"),
        "s33": pinned_json(S33_JSON, S33_SHA256, "s33 result"),
        "s34": pinned_json(S34_JSON, S34_SHA256, "s34 result"),
    }


def main() -> None:
    t0 = time.time()
    inp = read_inputs()
    run2, run3, s29, s33, s34 = (inp[k] for k in ("run2", "run3", "s29", "s33", "s34"))
    run2_model = as_model(run2)
    for model, name in ((run2_model, "run 2"), (as_model(run3), "run 3"), (s29, "s29"), (s33, "s33")):
        require_cells(model, name)

    # 1. Replication
    replication = {s: compare(run3, run2_model, s) for s in SEGMENTS}
    verdict_table, flips = verdicts(run2, run3)

    # 2. s29 under s34's rule
    run3_s29 = {s: compare(run3, s29, s) for s in SEGMENTS}
    run2_s29 = {s: compare(run2, s29, s) for s in SEGMENTS}
    run3_s33 = {s: compare(run3, s33, s) for s in SEGMENTS}
    dz = max(abs(run3_s33[s]["z"][k] - s34["segments"][s]["z"][k]) for s in SEGMENTS for k in s34["segments"][s]["z"])
    if dz > 1e-12:
        raise SourceRefused(f"run 3 against s33 does not reproduce s34's z (max |difference| {dz})")
    diff = rb_difference(s29, s33)
    weak = [{"segment": s, "family": f, "level": k, "s29": s29["cells"][s][f][k]["ratio_B"],
             "s33": s33["cells"][s][f][k]["ratio_B"], "difference": diff[s][f][k]}
            for s in SEGMENTS for f in DECIDING for k in ("50.0", "55.0")]  # fmt: skip

    # 3. The rule's error rate
    tails = {s: float(s34["segments"][s]["chi2"]) for s in SEGMENTS}
    size = size_and_tails(tails, rng=np.random.default_rng(SEED))
    exact = {
        "p_reject_rho1": exact_reject_full_correlation(),
        "p_sum_z2_at_least_rho1": {s: k_equal_tail(t, 1) for s, t in tails.items()},
        "p_sum_z2_at_least_rho0": {s: float(chi2.sf(t, N_CELLS)) for s, t in tails.items()},
    }
    eigen = eigen_check(tails, rng=np.random.default_rng(SEED + 1))

    # 4. B1 near its bar
    b2, b3 = b1_values(run2), b1_values(run3)
    near = near_bar(b2, b3)
    sp = spread(b2, b3)
    sp50 = spread(b2, b3, levels=("50.0",))

    for name, res in (("run 3 - run 2", replication), ("run 3 - s29", run3_s29), ("run 2 - s29", run2_s29),
                      ("run 3 - s33", run3_s33)):  # fmt: skip
        for s, r in res.items():
            print(f"{name} {s}: sum z^2 {r['chi2']:.2f}, max|z| {r['max_abs_z']:.2f} ({r['max_abs_z_cell']}), "
                  f"positive {r['positive']}/{r['n']}, within s34's bounds {r['within_s34_bounds']}, "
                  f"P(chi2_1 >= sum/n) {r['p_full_correlation']:.3f}")  # fmt: skip
    for fl in flips:
        z = replication[fl["segment"]]["z"][f"{fl['family']}/{fl['level']}"]
        print(f"flip {fl['segment']} {fl['family']} {fl['level']}: run 2 {fl['run2']} {fl['run2_ratio']:.3f} "
              f"+- {fl['run2_se']:.3f}, run 3 {fl['run3']} {fl['run3_ratio']:.3f} +- {fl['run3_se']:.3f}, z {z:+.2f}")  # fmt: skip
    ds = [w["difference"] for w in weak]
    print(f"R_B(s29) - R_B(s33) at 50-55 dB-Hz: {min(ds):.3f} to {max(ds):.3f}; run 3 - s33 = s34's z (max |dz| {dz})")
    for rho, r in size.items():
        print(f"rho {rho}: P(reject) {r['p_reject']:.4f} +- {r['mc_se']:.4f}; P(sum z^2 >= t) "
              + ", ".join(f"{s} {p:.4f}" for s, p in r["p_sum_z2_at_least"].items()))  # fmt: skip
    print(f"exact: {exact}")
    for s, e in eigen.items():
        print(f"eigen {s}: k-equal worst {e['k_equal_worst']}, two-point worst {e['two_point_worst']}, "
              f"Dirichlet worst {e['dirichlet_mc_worst']['p']:.4f} (largest share "
              f"{e['dirichlet_mc_worst']['largest_lambda_share']:.2f})")  # fmt: skip
    for r in near:
        print(f"B1 near bar {r['segment']} {r['family']} {r['level']} PRN {r['prn']}: run 2 {r['run2']:.4f}, "
              f"run 3 {r['run3']:.4f}")  # fmt: skip
    print(f"B1 spread ({sp['cells']} cells): SD {sp['sd']:.4f}, RMS {sp['rms']:.4f}, mean|d| {sp['mean_abs']:.4f}, "
          f"range {sp['min']:+.4f} to {sp['max']:+.4f}, per-draw SD {sp['per_draw_sd']:.4f}; at 50 dB-Hz "
          f"{sp50['min']:+.4f} to {sp50['max']:+.4f}")  # fmt: skip

    save(
        "s35_run3_post_data",
        {
            "label": "exploratory, post-data: written on 28 Oct 2026 after run 3's data, s34's verdict and the "
            "referee's report; conditional, exploratory",
            "decides": "nothing: no rule of R-0 to R-8, no verdict and no range changes with it",
            "question": "run 3 against run 2; s29 under s34's rule; the rule's error rate under a correct model; B1 "
            "near its bar (referee report of 28 Oct 2026, appendix A1 to A3)",
            "design": {
                "cells": f"{', '.join(DECIDING)} at {ACCEPTED_LEVELS[0]:g}-{ACCEPTED_LEVELS[-1]:g} dB-Hz, per segment",
                "z": "(x - y) / sqrt(SE_x^2 + SE_y^2), by s34's accept; run 2 put in s33's schema by as_model",
                "s29_schema": "s29's cells hold ratio_B and ratio_B_se at the key paths s34 reads from s33 "
                "(cells[s][family][level]), so accept reads s29 unchanged. s29 stops at 70 dB-Hz and has "
                "ratio_truth (one stationary AGC) where s33 has ratio_TA per tau, so s34's J_max_TA and "
                "conditional range do not apply to it and are not computed; no rule admits s29 (R-7 names s33)",
                "within_s34_bounds": "s34's accept on this pair: sum z^2 <= chi2_10(0.95) and max |z| < 3; it "
                "accepts nothing here",
                "p_full_correlation": "P(chi2_1 >= sum z^2 / n), the tail at rho = 1",
                "size": {
                    "model": "z ~ N(0, C(rho)), equicorrelation, n = 10: sqrt(rho) u + sqrt(1 - rho) e",
                    "rule": "reject iff sum z^2 > chi2_10(0.95) or max |z| >= 3 (s34's chi2_bound, MAX_ABS_Z)",
                    "seed": SEED,
                    "draws_per_rho": N_SIZE,
                    "rhos": list(RHOS),
                    "order": "the rhos in this order on one default_rng(seed); u then e per rho",
                },
                "eigen_check": {
                    "argument": "sum z^2 = sum_j lambda_j w_j^2 with lambda the correlation matrix's eigenvalues "
                    "(lambda >= 0, sum n) and w iid N(0, 1); every such lambda belongs to some correlation matrix "
                    "(Schur-Horn)",
                    "k_equal": "lambda = n/k on k coordinates: P(chi2_k >= t k / n), exact, k = 1..10",
                    "two_point": "lambda = n (a, 1 - a), a = 0.5 to 1.0 by 0.05, Monte Carlo",
                    "dirichlet": f"{N_DIRICHLET} flat-Dirichlet lambda, Monte Carlo",
                    "seed": SEED + 1,
                    "draws": N_EIG,
                    "order": "Dirichlet lambda first, then w, on one default_rng(seed); common random numbers",
                    "status": "a numerical check, not a proof",
                },
                "b1": {
                    "value": "r - floor per ghost, mode A (segments[s].residual[family][level][PRN])",
                    "bar": f"passes iff r - floor < {B1_BAR} (rules.residual_max)",
                    "near": NEAR,
                    "spread_cells": f"{', '.join(DECIDING)} at {', '.join(SPREAD_LEVELS)} dB-Hz, both segments, both "
                    "ghosts",
                    "per_draw_sd": "SD(run 3 - run 2) / sqrt(2): a two-draw estimate; pooling the cells assumes one "
                    "per-draw spread for all of them",
                },
            },
            "inputs": {
                "run3_summary": {
                    "path": str(RUN3_DIR / "summary.json"),
                    "sha256": RUN3_SUMMARY_SHA256,
                    "run_id": RUN3_ID,
                },
                "run2_summary": {"path": str(RUN2_SUMMARY), "sha256": RUN2_SUMMARY_SHA256, "run_id": RUN2_ID},
                "s29": {"path": str(S29_JSON), "sha256": S29_SHA256},
                "s33": {"path": str(S33_JSON), "sha256": S33_SHA256},
                "s34": {"path": str(S34_JSON), "sha256": S34_SHA256},
            },  # fmt: skip
            "cells": cell_table(run2, run3, s29, s33),
            "replication": {"segments": replication, "verdicts": verdict_table, "flips": flips},
            "s29_under_s34_rule": {
                "run3_against_s29": run3_s29,
                "run2_against_s29": {"label": "in sample for s29", "segments": run2_s29},
                "run3_against_s33": {
                    "label": "s34's own comparison, recomputed",
                    "segments": run3_s33,
                    "max_abs_difference_from_s34_z": dz,
                },  # fmt: skip
                "rb_s29_minus_s33": diff,
                "rb_s29_minus_s33_at_50_55": {"cells": weak, "min": min(ds), "max": max(ds)},
            },
            "rule_error_rate": {"simulated": size, "exact": exact, "eigen_check": eigen, "observed_sum_z2": tails},
            "b1": {
                "near_bar": near,
                "spread_50_60": sp,
                "spread_50": sp50,
                "margins_50": {"run3": margins(b3), "run2": margins(b2)},
            },
            "run_time_s": round(time.time() - t0, 1),
        },
    )


if __name__ == "__main__":
    main()
