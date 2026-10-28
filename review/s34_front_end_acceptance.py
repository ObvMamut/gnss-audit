"""S34: the front-end acceptance rule (P2c task 5; R-7). Written before run 3 exists; it runs only after run 3, once both
pins below are set (task 8). It decides only the R9 re-scope's conditional range of jamming levels, never G3.

Question. Does s33's model of the true analog front end reproduce the mode B / mode A ratios that run 3 measured? If it
does, what range of jamming levels does s33's true front end support, conditional on that?

Method (R-7, as registered; nothing here is calibrated). Per segment s:
  - For each noise and swept CW cell c at 50-70 dB-Hz (ten cells):
        z_c = (R_run3 - R_B,s33) / sqrt(SE_run3^2 + SE_s33^2),
    R_run3 and SE_run3 from run 3's summary (segments[s].front_end[family][level].ratio and .ratio_se), R_B,s33 and its SE
    from s33 (cells[s][family][level].ratio_B and .ratio_B_se). A cell whose R or SE is missing, None or non-finite, in
    either source, is not present (n excludes it).
  - Accept s iff sum z_c^2 <= chi2_n(0.95), n the number of cells present, and max |z_c| < 3. A segment with fewer than
    8 cells present is not accepted.
  - If s is accepted: J_max_TA,s33(s, f) = gnssaudit.experiments.e2_frontend.front_end_jmax over s33's levels 50-85, each
    level judged on the tau (1, 10, 100 ms) with the largest |R_TA - 1|, its SE and its slope (tolerances 0.10 / 0.10 /
    0.05: ratio, slope, SE). The conditional range is e2_frontend.ghost_ceiling over run 3's grid, the
    expected levels 50, 55, ..., 70 (so capped at 70): the largest grid level up to J_max_TA,s33 at which run 3's truth
    is kept at every grid level up to it. A level whose truth_kept is False, None or absent (key missing, family
    missing) is not kept and stops the range; the grid is not read off the keys present. If J_max_TA,s33 is None, or s is rejected, the range is None.
  - Combined over segments by e2_frontend.combine_segments: None if either segment's range is None. This is the owner's
    principle of 19 Oct: no range from one segment alone.
  - Refused: an s33 commit that is not strictly earlier than run 3's started_utc (check_order), and any run 3 summary or
    s33 JSON whose SHA-256 is not the pinned one. Run 2 against s33 is reported beside, as in-sample.

Correlated z (ruling T5a). Each z_c is taken as N(0, 1) under a correct model, and the chi2 bound assumes independent
cells. They are not: within a segment s33's cells share their synthetic noise, ghost and dither, and run 3's cells share
the recording and the clean twins. Positive correlation leaves E[sum z^2] = n but widens its spread, so under a correct
model the test rejects more often than 5 %. That makes it conservative for the claim it gates: a rejection only removes
the conditional support. The threshold is the registered one and is not calibrated.

Reading, stated before run 3 exists.
  - Accepted on both segments: s33's model reproduces run 3's measured ratios to within its own noise on these ten
    cells per segment, and the conditional range (s33's true-front-end J_max, gated by run 3's truth) may be quoted as
    conditional on that model. It says nothing about G3.
  - Rejected on either segment: the model does not reproduce run 3, the combined range is None, and the R9 re-scope has
    no conditional range from s33.
  - The run 2 line is in-sample (s33 was built on run 2's N0 and analog SD): it is reported, not used.

Usage: uv run python -m review.s34_front_end_acceptance (after run 3; refuses while RUN3_SUMMARY_SHA256 or S33_SHA256
is None).
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from scipy.stats import chi2

from gnssaudit.experiments.e2_frontend import FrontEndCell, combine_segments, front_end_jmax, ghost_ceiling
from gnssaudit.paths import results_root
from review._common import RESULTS, save
from review.s30_k_rule_null import SourceRefused
from review.s32_g3_rerun_forecast import RUN2_SUMMARY, RUN2_SUMMARY_SHA256, pinned_json
from review.s33_front_end_prediction import ACCEPTANCE_MAX_DB_HZ, DECIDING, LEVELS, SEGMENTS, level_key

REPO = Path(__file__).resolve().parent.parent

RUN3_ID = "9650dc00e766"  # the hash of E2-v3.yaml with workers_b: 6 (task 7a, ruling R3a)
# 1203d37425c6: the aborted start of 27 Oct, stopped before any unit (ruling R3a)
RUN3_DIR = results_root() / "E2" / RUN3_ID
RUN3_SUMMARY_SHA256: str | None = "f38389b07f8c025114d980d4c0cef1952f6f102b2f5d493e1eb19f717be2a88b"  # task 8, 28 Oct
S33_JSON = RESULTS / "s33_front_end_prediction.json"
S33_SHA256: str | None = "cee73739dcc8b2c207ab5b2b9683ddecb92412e583d6c75e80a21b141e8579ac"  # a907825 (task 8)

MIN_CELLS = 8  # a segment with fewer cells present is not accepted
MAX_ABS_Z = 3.0
CHI2_LEVEL = 0.95
TOL, SLOPE_TOL, SE_MAX = 0.10, 0.10, 0.05  # front_end_jmax's ratio, slope and SE tolerances
RANGE_CAP_DB_HZ = ACCEPTANCE_MAX_DB_HZ  # 70, the top of run 3's grid
ACCEPTED_LEVELS = tuple(lv for lv in LEVELS if lv <= ACCEPTANCE_MAX_DB_HZ)  # 50, 55, ..., 70

ASSUMPTIONS = [
    "each z_c is taken as N(0, 1) under a correct model, and the chi2 bound assumes independent cells; within a segment "
    "s33's cells share their synthetic noise, ghost and dither, and run 3's cells share the recording and the clean "
    "twins, so the z_c are positively correlated",
    "positive correlation leaves E[sum z^2] = n but widens its spread, so under a correct model the test rejects more "
    "often than 5 %: conservative for the claim it gates (a rejection only removes the conditional support)",
    "the threshold is R-7's as registered; it is not calibrated",
    "the run 2 line is in-sample: s33 was built on run 2's N0 and analog SD",
]


def _finite(x: object) -> float | None:
    """x as a float if it is a finite real number (not a bool), else None."""
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    return float(x) if math.isfinite(x) else None


def _at(m: object, *keys: str) -> object:
    """m[k0][k1]... or None where a step is missing or not a mapping."""
    for k in keys:
        if not isinstance(m, Mapping):
            return None
        m = m.get(k)
    return m


# ------------------------------------------------------------------------------------------------ the acceptance rule
def chi2_bound(n: int) -> float:
    """chi2_n(0.95): the bound on sum z^2 for n cells present."""
    return float(chi2.ppf(CHI2_LEVEL, n))


def accept(run3: Mapping[str, Any], s33: Mapping[str, Any], segment: str) -> dict[str, Any]:
    """R-7's judgement of one segment. run3 is an E2 summary (segments[s].front_end[family][level]); s33 is its JSON
    (cells[s][family][level]). z is keyed "family/level"; max_abs_z is None when no cell is present."""
    z: dict[str, float] = {}
    for fam in DECIDING:
        for lv in ACCEPTED_LEVELS:
            key = level_key(lv)
            r3 = _finite(_at(run3, "segments", segment, "front_end", fam, key, "ratio"))
            se3 = _finite(_at(run3, "segments", segment, "front_end", fam, key, "ratio_se"))
            rb = _finite(_at(s33, "cells", segment, fam, key, "ratio_B"))
            seb = _finite(_at(s33, "cells", segment, fam, key, "ratio_B_se"))
            if r3 is None or se3 is None or rb is None or seb is None:
                continue
            var = se3 * se3 + seb * seb
            if var > 0.0 and math.isfinite(var):
                z[f"{fam}/{key}"] = (r3 - rb) / math.sqrt(var)
    n = len(z)
    chi_sq = float(sum(v * v for v in z.values()))
    max_abs = max((abs(v) for v in z.values()), default=None)
    ok = n >= MIN_CELLS and chi_sq <= chi2_bound(n) and max_abs is not None and max_abs < MAX_ABS_Z
    return {"z": z, "chi2": chi_sq, "n": n, "max_abs_z": max_abs, "accepted": bool(ok)}


# ------------------------------------------------------------------------------------------------ the conditional range
def jmax_ta(s33: Mapping[str, Any], segment: str, family: str) -> float | None:
    """J_max_TA,s33(segment, family): front_end_jmax over s33's levels, each judged on the tau with the largest
    |R_TA - 1| (its ratio, SE and slope). A non-finite value is refused (ValueError), not skipped."""
    by_level = _at(s33, "cells", segment, family)
    if not isinstance(by_level, Mapping) or not by_level:
        raise ValueError(f"s33 has no cells for segment {segment}, family {family}")
    cells: dict[float, FrontEndCell] = {}
    for key, c in by_level.items():
        ratios, ses, slopes = c["ratio_TA"], c["ratio_TA_se"], c["slope_TA"]
        taus = list(ratios)
        if not taus or any(_finite(ratios[t]) is None for t in taus):
            raise ValueError(f"s33 {segment}/{family}/{key}: a missing or non-finite R_TA has no verdict")
        worst = max(taus, key=lambda t: abs(ratios[t] - 1.0))
        cells[float(key)] = FrontEndCell(slopes=(slopes[worst],), ratio=ratios[worst], ratio_se=ses[worst])
    return front_end_jmax(cells, tol=TOL, slope_tol=SLOPE_TOL, se_max=SE_MAX, cell=f"s33 {segment}/{family} TA")


def conditional_range(s33: Mapping[str, Any], run3: Mapping[str, Any], segment: str, family: str) -> float | None:
    """The conditional range for (segment, family): None if the segment is not accepted or J_max_TA,s33 is None;
    otherwise the largest level of run 3's grid (ACCEPTED_LEVELS, 50-70 by 5) up to J_max_TA,s33 with run 3's truth kept
    at every grid level up to it. The grid is the expected levels, not the keys present: a level whose truth_kept is
    False, None or absent is not kept, and stops the range."""
    if not accept(run3, s33, segment)["accepted"]:
        return None
    kept = _at(run3, "segments", segment, "truth_kept", family)
    kept = kept if isinstance(kept, Mapping) else {}
    truth_ok = {lv: kept.get(level_key(lv)) is True for lv in ACCEPTED_LEVELS}
    return ghost_ceiling(ACCEPTED_LEVELS, jmax_ta(s33, segment, family), truth_ok)


def combined_range(
    s33: Mapping[str, Any], run3: Mapping[str, Any], segments: tuple[str, ...] = SEGMENTS
) -> dict[str, float | None]:
    """Per family, the segments' conditional ranges combined: None if any segment's is None."""
    return {f: combine_segments([conditional_range(s33, run3, s, f) for s in segments]) for f in DECIDING}


# ------------------------------------------------------------------------------------------------ order and pins
def _aware(stamp: str, what: str) -> datetime:
    t = datetime.fromisoformat(stamp)
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"{what} {stamp!r} has no UTC offset: the order of two times needs one")
    return t


def check_order(s33_commit_utc: str, run3_started_utc: str) -> None:
    """Refuse unless s33's commit is strictly earlier than run 3's start (both ISO-8601 with an offset)."""
    c, s = _aware(s33_commit_utc, "s33's commit time"), _aware(run3_started_utc, "run 3's started_utc")
    if c >= s:
        raise SourceRefused(
            f"s33 was committed at {s33_commit_utc}, not before run 3 started ({run3_started_utc}): later"
        )


def s33_commit_time() -> str:
    """The committer date (ISO-8601, offset included) of the commit that holds S33_JSON; refused if the file is not
    committed or differs from HEAD."""
    rel = str(S33_JSON.relative_to(REPO))
    out = subprocess.run(
        ["git", "log", "-1", "--format=%cI", "--", rel], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    if not out:
        raise SourceRefused(f"{rel} has no commit: s33's JSON must be committed before run 3")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO, check=False).returncode != 0:
        raise SourceRefused(f"{rel} differs from HEAD: the commit time would not be this file's")
    return out


def run3_started_utc() -> str:
    return str(json.loads((RUN3_DIR / "started.json").read_text())["started_utc"])


def require_pins() -> tuple[str, str]:
    """(run 3 summary SHA-256, s33 JSON SHA-256); refused while either is None."""
    if RUN3_SUMMARY_SHA256 is None:
        raise SourceRefused("RUN3_SUMMARY_SHA256 is None: it is pinned in task 8, when run 3's summary is first read")
    if S33_SHA256 is None:
        raise SourceRefused("S33_SHA256 is None: it is pinned in task 8, when s33's committed JSON is first read")
    return RUN3_SUMMARY_SHA256, S33_SHA256


# ------------------------------------------------------------------------------------------------ the study
def judge(s33: Mapping[str, Any], run3: Mapping[str, Any]) -> dict[str, Any]:
    segments: dict[str, Any] = {}
    for s in SEGMENTS:
        a = accept(run3, s33, s)
        segments[s] = {
            **a,
            "chi2_bound": chi2_bound(a["n"]) if a["n"] else None,
            "j_max_ta_s33": {f: jmax_ta(s33, s, f) for f in DECIDING},
            "conditional_range": {f: conditional_range(s33, run3, s, f) for f in DECIDING},
        }
    return {"segments": segments, "combined_range": combined_range(s33, run3)}


def main() -> None:
    run3_sha, s33_sha = require_pins()
    run3 = pinned_json(RUN3_DIR / "summary.json", run3_sha, "E2 run 3 summary")
    s33 = pinned_json(S33_JSON, s33_sha, "s33 result")
    commit, started = s33_commit_time(), run3_started_utc()
    check_order(commit, started)
    run2 = pinned_json(RUN2_SUMMARY, RUN2_SUMMARY_SHA256, "E2 run 2 summary")
    out = judge(s33, run3)
    in_sample = {s: accept(run2, s33, s) for s in SEGMENTS}
    for s, r in out["segments"].items():
        print(f"segment {s}: n {r['n']}, chi2 {r['chi2']:.2f} (bound {r['chi2_bound']}), max|z| {r['max_abs_z']}, "
              f"accepted {r['accepted']}; ranges {r['conditional_range']}")  # fmt: skip
    print(f"combined conditional range: {out['combined_range']}")
    for s, r in in_sample.items():
        print(f"run 2 against s33 (in-sample, reported) segment {s}: n {r['n']}, chi2 {r['chi2']:.2f}, "
              f"accepted {r['accepted']}")  # fmt: skip
    save(
        "s34_front_end_acceptance",
        {
            "question": "does s33's front-end model reproduce run 3's measured mode B / mode A ratios, and if so what "
            "conditional range does its true front end support (R-7)",
            "decides": "the R9 re-scope's conditional range only; never G3",
            "rule": {
                "cells": f"{', '.join(DECIDING)} at {ACCEPTED_LEVELS[0]:g}-{ACCEPTED_LEVELS[-1]:g} dB-Hz",
                "accept": f"sum z^2 <= chi2_n({CHI2_LEVEL}) and max |z| < {MAX_ABS_Z:g}, n >= {MIN_CELLS} cells present",
                "tolerances": {"ratio": TOL, "slope": SLOPE_TOL, "se_max": SE_MAX},
                "cap_db_hz": RANGE_CAP_DB_HZ,
            },
            "assumptions": ASSUMPTIONS,
            "order": {"s33_commit": commit, "run3_started_utc": started},
            "inputs": {
                "run3_summary": {"path": str(RUN3_DIR / "summary.json"), "sha256": run3_sha, "run_id": RUN3_ID},
                "s33": {"path": str(S33_JSON), "sha256": s33_sha},
                "run2_summary": {"path": str(RUN2_SUMMARY), "sha256": RUN2_SUMMARY_SHA256},
            },
            **out,
            "run2_against_s33": {"label": "in-sample, reported", "segments": in_sample},
        },
    )


if __name__ == "__main__":
    main()
