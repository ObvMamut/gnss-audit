"""S32: the G3 re-run's forecast, reproduced (P2c task 1). Written after run 2 and E2K, before run 3; it decides nothing.

Question. P2c's plan (5612ebf) states, as R-4, how likely the G3 re-run is to give the front end a J_max, to start the
K-study, to find the thermal block clean, and to pass G3. Can every number in R-4 be reproduced from data already on
disk?

Method. Per (family, basis, segment) the five cells' ratios R (50, 55, 60, 65, 70 dB-Hz) are drawn independently:
  - basis "run2": N(run 2's R, 2 SE^2), the variance doubled because the re-run's R is a new estimate beside run 2's;
  - basis "s29":  N(s29's mode-B ratio, SE^2), the ratio a simulated true analog front end gives.
Each cell keeps run 2's SE. The verdict is rule (a') without slopes (all of run 2's slopes lie in 0.926-1.012): a cell is
unresolved if SE > 0.05, else a fail if |R - 1| >= 0.10, else a pass (gnssaudit.experiments.e2_frontend.cell_verdict's
order and ties). J_max is the largest level that passes with no fail at or below it. R-4's rows use the event "J_max >=
level i" as the plan's script computed it: no fail at levels 0..i and a pass at level i. The clause-relevant event is
J_max is not None (some level passes with no fail at or below it), computed from the same draws (ruling P1), so the
"J_max >= 60" rows are unchanged. default_rng(20261024), N = 200 000, one (N, 5) draw per (family, basis, segment) in the
order family (noise, swept_cw) x basis (run2, s29) x segment (A, B). "Both" is the mean of the two segments' events on
the same draws. Between families, and between segments where a product is taken, events are treated as independent;
this ignores the families' shared clean twin and dither.
The thermal block: E2K failed G-warmup-lock on 7 of its 102 thermal-block units. P(a re-run block of 102 has none) is
((102 - 7) / 102)^102 (plug-in) or B(8, 102 - 7 + 1 + 102) / B(8, 102 - 7 + 1) (Beta(1, 1) predictive). G3's upper bound
is the strict both-family P(J_max not None) times each.

Inputs, read-only and pinned by SHA-256 (the study stops on a mismatch): E2 run 2 (6214cf52489f) summary.json
(segments[s].front_end[family][level].ratio and ratio_se), s29 (cells[s][family][level].ratio_B), and s31's
q1_unlocks (thermal block: units, lock_failed).

Reading, stated before this study ran (R-4 as committed in the plan):
  | Clause | Forecast (run 2 basis / s29 basis) |
  | K-study runs (either family has a ghost ceiling) | P = 0.037 / 0.880 |
  | 1, if the K-study runs: P(thermal block clean) | 0.071 % (plug-in) to 0.353 % (Beta(1, 1) predictive) |
  | 3, noise: P(J_max >= 60) on A, B, both | A 0.279 / 0.427; B 0.021 / 0.471; both 0.006 / 0.202 |
  | 3, swept CW: same | A 0.574 / 0.918; B 0.054 / 0.925; both 0.031 / 0.850 |
  | 3, both families (strict; independent) | about 0.0002 / about 0.17 |
  | G3 | < 0.1 % on either basis |

Result (review/results/s32_g3_rerun_forecast.json; deterministic: every draw is seeded).
  - Every printed digit of R-4 reproduces (run 2 basis / s29 basis, event J_max >= 60): noise A 0.279 / 0.427, B 0.021 /
    0.471, both 0.006 / 0.202; swept CW A 0.574 / 0.918, B 0.054 / 0.925, both 0.031 / 0.850; strict about 0.0002 / 0.171;
    K-study runs 0.037 / 0.880; thermal block clean 0.071 % (plug-in) to 0.353 % (predictive).
  - The events "J_max >= 60", ">= 65" and ">= 70" have the same probability on every row (a pass at 55 and at 60 fixes
    the rest: the cells from 60 up have SE 0.0172 or less), and taken literally they equal the plan's. 50 dB-Hz is
    unresolved in every draw (SE 0.0597 or more), so J_max cannot be 50.
  - Clause-relevant (event: J_max not None; ruling P1; a dated correction to R-4's strict and K-study rows, which used
    J_max >= 60): noise both 0.013 / 0.209, swept CW both 0.038 / 0.850; strict 0.0005 / 0.178; K-study runs 0.050 /
    0.881. The difference is a 55 dB-Hz pass with a fail at 60.
  - G3's upper bound, strict not-None times the thermal block's chance: 3.4e-7 (plug-in) to 1.7e-6 (predictive) on the
    run 2 basis; 1.3e-4 to 6.3e-4 on the s29 basis. Both are below 0.1 %, as R-4 says. Families and segments are taken as
    independent, which ignores their shared clean twin and dither.
  - Written after run 2 and E2K and before run 3. It decides nothing.
Usage: uv run python -m review.s32_g3_rerun_forecast (reads run 2's summary, s29 and s31 read-only; seconds).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from gnssaudit.paths import results_root
from review._common import RESULTS, save
from review.s30_k_rule_null import SourceRefused

FloatArray = NDArray[np.float64]

RUN2_ID = "6214cf52489f"
RUN2_SUMMARY = results_root() / "E2" / RUN2_ID / "summary.json"  # read only
RUN2_SUMMARY_SHA256 = "4419fd70188279102a2ab862fda3ef17acdbb81018b8793918d165b11e962c99"
S29_JSON = RESULTS / "s29_true_analog_jmax.json"
S29_SHA256 = "efd04c8d0cc39d2d373fb7fae6454793f80c2f8d88648226b99d7562a70e6fc2"
S31_JSON = RESULTS / "s31_e2k_warmup_unlocks.json"
S31_SHA256 = "815db39bdb9b631c716526e001b0d06170a524768ea070541bb337f9c0541bed"

LEVELS = ("50.0", "55.0", "60.0", "65.0", "70.0")
FAMILIES = ("noise", "swept_cw")
BASES = ("run2", "s29")
SEGMENTS = ("A", "B")
CLAUSE_LEVELS = (2, 3, 4)  # 60, 65, 70 dB-Hz
TOL = 0.10  # rule (a'): a cell fails at |R - 1| >= TOL
SE_MAX = 0.05  # a cell with SE(R) above this is unresolved
SEED = 20261024
N = 200_000


# ------------------------------------------------------------------------------------------------ the model
def cell_verdicts(r: FloatArray, se: FloatArray) -> NDArray[np.str_]:
    """Rule (a') without slopes, elementwise: "u" (unresolved) if SE > 0.05, else "f" (fail) if |R - 1| >= 0.10, else
    "p" (pass). The order and the ties are e2_frontend.cell_verdict's."""
    r, se = np.broadcast_arrays(np.asarray(r, dtype=float), np.asarray(se, dtype=float))
    return np.where(se > SE_MAX, "u", np.where(np.abs(r - 1.0) >= TOL, "f", "p"))


def draw_verdicts(
    mu: FloatArray, sd: FloatArray, se: FloatArray, *, rng: np.random.Generator, n: int
) -> NDArray[np.str_]:
    """One (n, levels) draw of R = mu + sd * z, judged against `se`."""
    return cell_verdicts(mu + sd * rng.standard_normal((n, mu.size)), se)


def reaches(verdicts: NDArray[np.str_]) -> NDArray[np.bool_]:
    """Per draw and level: a pass with no fail at that level or below."""
    return np.logical_and.accumulate(verdicts != "f", axis=1) & (verdicts == "p")


def at_least(verdicts: NDArray[np.str_], level_idx: int) -> NDArray[np.bool_]:
    """R-4's event "J_max >= level i" as the plan's script computed it: no fail at levels 0..i and a pass at i."""
    return reaches(verdicts)[:, level_idx]


def not_none(verdicts: NDArray[np.str_]) -> NDArray[np.bool_]:
    """J_max is not None: some level passes with no fail at or below it."""
    return reaches(verdicts).any(axis=1)


def jmax_at_least(verdicts: NDArray[np.str_], level_idx: int) -> NDArray[np.bool_]:
    """J_max >= level i taken literally: some level j >= i passes with no fail at or below it."""
    return reaches(verdicts)[:, level_idx:].any(axis=1)


def p_jmax_at_least(
    mu: FloatArray, sd: FloatArray, se: FloatArray, level_idx: int, *, rng: np.random.Generator, n: int
) -> float:
    return float(at_least(draw_verdicts(mu, sd, se, rng=rng, n=n), level_idx).mean())


def p_jmax_not_none(mu: FloatArray, sd: FloatArray, se: FloatArray, *, rng: np.random.Generator, n: int) -> float:
    return float(not_none(draw_verdicts(mu, sd, se, rng=rng, n=n)).mean())


def p_thermal_clean(failed: int = 7, units: int = 102) -> tuple[float, float]:
    """P(a block of `units` has no lock failure) given `failed` of `units` in the last one: the plug-in
    ((units - failed) / units)^units, and the Beta(1, 1) predictive B(failed + 1, units - failed + 1 + units) /
    B(failed + 1, units - failed + 1)."""

    def log_beta(a: float, b: float) -> float:
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    a, b = failed + 1, units - failed + 1
    return ((units - failed) / units) ** units, math.exp(log_beta(a, b + units) - log_beta(a, b))


# ------------------------------------------------------------------------------------------------ sources
def pinned_json(path: Path, sha256: str, what: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise SourceRefused(f"{path}: not the pinned {what} (SHA-256 {sha256[:8]}...)")
    return json.loads(raw)


def read_inputs() -> dict[str, Any]:
    run2 = pinned_json(RUN2_SUMMARY, RUN2_SUMMARY_SHA256, "E2 run 2 summary")
    s29 = pinned_json(S29_JSON, S29_SHA256, "s29 result")
    s31 = pinned_json(S31_JSON, S31_SHA256, "s31 result")
    cells: dict[str, dict[str, dict[str, FloatArray]]] = {}
    for fam in FAMILIES:
        cells[fam] = {}
        for seg in SEGMENTS:
            fe = run2["segments"][seg]["front_end"][fam]
            cells[fam][seg] = {
                "run2_r": np.array([fe[lv]["ratio"] for lv in LEVELS], dtype=float),
                "se": np.array([fe[lv]["ratio_se"] for lv in LEVELS], dtype=float),
                "s29_r": np.array([s29["cells"][seg][fam][lv]["ratio_B"] for lv in LEVELS], dtype=float),
            }
    thermal = s31["q1_unlocks"]["groups"]["thermal block"]
    return {"cells": cells, "failed": len(thermal["lock_failed"]), "units": int(thermal["units"])}


# ------------------------------------------------------------------------------------------------ the study
def simulate(cells: dict[str, dict[str, dict[str, FloatArray]]]) -> dict[str, Any]:
    """Draw in the order family x basis x segment, one (N, 5) draw each; every event comes from the same draw."""
    rng = np.random.default_rng(SEED)
    out: dict[str, Any] = {}
    for fam in FAMILIES:
        out[fam] = {}
        for basis in BASES:
            ev: dict[str, dict[str, NDArray[np.bool_]]] = {}
            for seg in SEGMENTS:
                c = cells[fam][seg]
                mu = c["run2_r"] if basis == "run2" else c["s29_r"]
                sd = c["se"] * math.sqrt(2.0) if basis == "run2" else c["se"]
                v = draw_verdicts(mu, sd, c["se"], rng=rng, n=N)
                ev[seg] = {f"ge{int(float(LEVELS[i]))}": at_least(v, i) for i in CLAUSE_LEVELS}
                ev[seg] |= {f"literal_ge{int(float(LEVELS[i]))}": jmax_at_least(v, i) for i in CLAUSE_LEVELS}
                ev[seg]["not_none"] = not_none(v)
            out[fam][basis] = {
                key: {"A": float(ev["A"][key].mean()), "B": float(ev["B"][key].mean()),
                      "both": float((ev["A"][key] & ev["B"][key]).mean())}
                for key in ev["A"]
            }  # fmt: skip
    return out


def combine(p: dict[str, Any], key: str) -> dict[str, dict[str, float]]:
    """Per basis: the strict both-family product and P(K-study runs) (either family has a ghost ceiling), with the
    two families and the two segments treated as independent."""
    out = {}
    for basis in BASES:
        n, s = p["noise"][basis][key]["both"], p["swept_cw"][basis][key]["both"]
        out[basis] = {"strict_both_families": n * s, "k_study_runs": 1.0 - (1.0 - n) * (1.0 - s)}
    return out


def main() -> None:
    t0 = time.time()
    inp = read_inputs()
    p = simulate(inp["cells"])
    plug_in, predictive = p_thermal_clean(inp["failed"], inp["units"])
    as_printed = combine(p, "ge60")
    clause = combine(p, "not_none")
    g3 = {b: {"plug_in": clause[b]["strict_both_families"] * plug_in,
              "predictive": clause[b]["strict_both_families"] * predictive} for b in BASES}  # fmt: skip

    print("as printed in R-4 (event: J_max >= 60), run 2 basis / s29 basis")
    for fam in FAMILIES:
        for seg in ("A", "B", "both"):
            a, b = (p[fam][basis]["ge60"][seg] for basis in BASES)
            print(f"  {fam:9s} {seg:4s} {a:.3f} / {b:.3f}")
    print(f"  strict both families {as_printed['run2']['strict_both_families']:.4f} / "
          f"{as_printed['s29']['strict_both_families']:.4f}")  # fmt: skip
    print(f"  K-study runs {as_printed['run2']['k_study_runs']:.3f} / {as_printed['s29']['k_study_runs']:.3f}")
    print("clause-relevant (event: J_max not None), run 2 basis / s29 basis")
    for fam in FAMILIES:
        for seg in ("A", "B", "both"):
            a, b = (p[fam][basis]["not_none"][seg] for basis in BASES)
            print(f"  {fam:9s} {seg:4s} {a:.3f} / {b:.3f}")
    print(f"  strict both families {clause['run2']['strict_both_families']:.4f} / "
          f"{clause['s29']['strict_both_families']:.4f}")  # fmt: skip
    print(f"  K-study runs {clause['run2']['k_study_runs']:.3f} / {clause['s29']['k_study_runs']:.3f}")
    print(f"thermal block clean ({inp['failed']} of {inp['units']}): plug-in {100 * plug_in:.3f} %, "
          f"Beta(1,1) predictive {100 * predictive:.3f} %")  # fmt: skip
    print(f"G3 upper bound (strict not-None x thermal): plug-in {g3['run2']['plug_in']:.2e} / "
          f"{g3['s29']['plug_in']:.2e}; predictive {g3['run2']['predictive']:.2e} / {g3['s29']['predictive']:.2e}")  # fmt: skip
    save(
        "s32_g3_rerun_forecast",
        {
            "question": "R-4's forecast of the G3 re-run (P2c plan 5612ebf), reproduced from data on disk",
            "design": {
                "seed": SEED,
                "n": N,
                "levels_db_hz": [float(lv) for lv in LEVELS],
                "rule": "(a') without slopes: unresolved if SE > 0.05, else fail if |R - 1| >= 0.10, else pass",
                "bases": {"run2": "R ~ N(run 2's R, 2 SE^2)", "s29": "R ~ N(s29's ratio_B, SE^2)"},
                "se": "run 2's SE in both bases",
                "draw_order": "family (noise, swept_cw) x basis (run2, s29) x segment (A, B); one (N, 5) draw each",
                "events": {
                    "ge60/65/70": "R-4's event as the plan's script computed it: no fail at levels 0..i, pass at i",
                    "literal_ge60/65/70": "J_max >= level taken literally: any level j >= i passes with no fail at or below",
                    "not_none": "J_max is not None: some level passes with no fail at or below it (ruling P1)",
                },
                "independence": "families, and segments where a product is taken, are treated as independent; "
                "this ignores their shared clean twin and dither",
            },
            "inputs": {
                "run2_summary": {"path": str(RUN2_SUMMARY), "sha256": RUN2_SUMMARY_SHA256},
                "s29": {"path": str(S29_JSON), "sha256": S29_SHA256},
                "s31": {"path": str(S31_JSON), "sha256": S31_SHA256},
                "cells": {
                    fam: {seg: {k: v.tolist() for k, v in c.items()} for seg, c in by_seg.items()}
                    for fam, by_seg in inp["cells"].items()
                },
            },
            "probabilities": p,
            "as_printed_in_r4": {
                "note": "event: J_max >= 60; the strict and K-study rows are R-4's",
                **as_printed,
            },
            "clause_relevant": {
                "note": "event: J_max not None; a dated correction to R-4's strict and K-study rows",
                **clause,
            },
            "thermal_block": {
                "failed": inp["failed"],
                "units": inp["units"],
                "p_clean_plug_in": plug_in,
                "p_clean_beta11_predictive": predictive,
            },
            "g3_upper_bound": {
                "note": "strict both-family P(J_max not None) x P(thermal block clean)",
                **g3,
            },
            "run_time_s": round(time.time() - t0, 1),
        },
    )


if __name__ == "__main__":
    main()
