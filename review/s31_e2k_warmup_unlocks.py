"""S31: E2K's warm-up unlocks, and its thermal comparison without them (P2b part 2, task 11c fix round 1; post hoc).

Written after E2K's result. It makes a review study of E2K's post-run analysis of 22 Oct 2026 (the plan workspace's
q1_unlocks.py and q2_thermal_sensitivity.py, re-implemented here). Nothing in it is pre-registered and nothing in it
decides: K = 51 and the gate failure stand as E2K recorded them (docs/10 section 2.3.2; D-42).

Questions.
  Q1. E2K failed G-warmup-lock on 7 of its 102 thermal-block units and on none of its 152 K-study units. What
      separates them?
  Q2. E2K's thermal comparison (summary.json decisions.thermal_vs_fixed, 0.340, reported only): how does it move
      without the pairs whose twins had a ghost false lock, and what else is in it?

Sources, read-only and pinned.
  - E2K run 0443516ca48c, which has no manifest (it failed a gate): s30's checks (the directory is that run,
    started.json names code 3eaed0b7eeb6, summary.json has its pinned SHA-256), and every unit.json and parquet read
    must match the SHA-256 in its unit's done.json. The receivers' logs (<name>.gnss-sdr.log for the first receiver
    run, <name>~k.gnss-sdr.log for run k + 1) are not hashed by the run; they are read as they are.
  - E2 run 2 (6214cf52489f; its manifest's run id and its summary's SHA-256 are checked): the clean twin A-clean-A's
    tracks (hash-checked), segment A's channel map and line excess, and the spec's warm-up length.
  - The ghosts' PRNs and set Dopplers: E2K's copy of run 2's survey (source/units/survey-A/survey.json, hash-checked).

Method, Q1.
  - Per unit (unit.json): block, receiver runs (attempts), lock_ok, unlocked channels.
  - Per receiver run, from its log: whether each ghost channel reached bit synchronisation by the end of the warm-up
    (30 s), and when its tracking started. Jammed units keep no tracks and earlier runs' dumps are deleted, so "no bit
    sync by 30 s" stands for a ghost false lock. The study stops unless that proxy matches every final run's recorded
    unlock; using it on earlier runs is an assumption.
  - Per clean twin with tracks (E2K's 51 thermal-clean twins; run 2's A-clean-A, which every K-study twin equals byte
    for byte before the onset): the median C/N0, smoothed lock statistic and Doppler offset from the set Doppler over
    the warm-up check's window, the last 5 s of the warm-up.
  - The C/N0 the floor costs, predicted: adding N0 to a C/A-weighted density line_excess dB above N0 costs
    10 log10(1 + 10^(-line_excess / 10)) dB (docs/8 section 2.1).
  - Retries: units that missed the lock policy by receiver run, and Fisher's exact test of two things: the first run
    against the second, and whether a pair's two twins (which share the floor draw, so their warm-up bytes) miss
    their first run together.
Method, Q2. s30's reconstruction of E2K's tables (hash-checked units, e2.k_study_decision's construction). The study
stops unless the rebuilt thermal table equals E2K's oracle_A_thermal.parquet and the comparison equals summary.json's
thermal_vs_fixed. Whole pairs are dropped, since a pair is one realization (the jammed twin's truth is formed
against its own clean twin): (a) pairs with a lock-failed twin (lock_ok); (b) pairs with a ghost false lock in any
receiver run of either twin (Q1's proxy). Each is compared with the base of 51 and with the K = 101 block's first K of
the same size, and split at the onset (before it, both twins of a pair hold the same bytes). Also: per satellite, the
median variance ratio to the base from onset + 2 s, with k-noise as the reference; the median sigma^2 before and
after the onset; and the realizations per satellite.

Reading, stated before this study ran (from the post-run analysis, whose numbers it should reproduce): Q1, the floor
lowers the ghosts' C/N0 by about 2 dB and raises their rate of start-up false locks, which D-34's raised carrier-lock
limit keeps through the warm-up; Q2, leaving out the affected pairs does not bring the comparison under 0.2, and the
rows before the onset inflate it.

Result (review/results/s31_e2k_warmup_unlocks.json; it reproduces the post-run analysis's numbers exactly).
  - Q1. The bit-sync proxy matches every final run: 8 ghost channels recorded unlocked, none with bit sync; 526
    locked, all with it. Ghost false locks: 4 of 312 ghost-runs (1.3 %) in the K-study blocks against 44 of 270
    (16.3 %) in the thermal block, 12.7 times the rate; PRN 32 33 of 135 and PRN 31 11 of 135 there. Every ghost
    tracking start in the K-study blocks is at 1 s; 63 of 270 in the thermal block are at 2 to 4 s.
  - Under the floor (thermal-clean twins against run 2's A-clean-A, last 5 s of the warm-up), the ghosts' C/N0 is 41.04
    and 41.20 dB-Hz against 43.10 and 43.38: 2.06 and 2.18 dB lower, against 2.09 dB predicted from the line excess of
    2.08 dB; the real satellites lose 1.36 to 1.82 dB. A ghost that locks under the floor tracks cleanly: lock
    statistic 0.906 and 0.907 (median; the bar is 0.2), Doppler within 0.11 Hz of its set value (the bar is 150 Hz).
  - Retries. Thermal block: 22 of 102 units missed the policy on their first run, 11 of 22 on their second and 7 of 11
    on their third; 15 of the 22 passed on a later run on the same bytes. K-study blocks: 4 of 152, all passed on their
    second run. First against second run: Fisher p = 0.014. A pair's two twins miss their first run together in 6
    pairs, against 2.4 expected (p = 0.0065). repeat-A-thermal-clean-r19 (the same bytes as A-thermal-clean-r19, which
    failed all 3 runs) passed on its second. Channel 3 (PRN 9) is excluded in every one of the 267 units.
  - The reading holds for Q1: the floor's ~2 dB lowers the ghosts' C/N0, their start-up false locks become about 13
    times as frequent, and a false lock without bit sync is kept through the warm-up. Retries on the same bytes are not
    deterministic, but the failure rate depends on the bytes.
  - Q2. Rebuilt as E2K computed it: 0.3399 (17.37 on the 7 500 rows before the onset, 0.2224 on the 18 798 from onset
    + 2 s). (a) Without the 6 pairs with a lock-failed twin (r 7, 17, 19, 40, 47, 48): 0.3484 against the base of 51,
    0.3568 against the first 45. (b) Without the 16 pairs with a ghost false lock in any run: 0.3713 and 0.3995 (base
    35). None is at or below 0.2. From onset + 2 s the thermal variance is 2 to 17 % above the base's per satellite
    (k-noise: -0.7 to +1.4 %). The reading holds for Q2.

Usage: uv run python -m review.s31_e2k_warmup_unlocks (reads E2K and E2 run 2 read-only; about 3 min).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import fisher_exact

from gnssaudit.experiments.e2 import SETTLE_S
from gnssaudit.experiments.twin import RX_ATTEMPTS, WARMUP_DOPPLER_TOL_HZ, WARMUP_LOCK_MIN, WARMUP_LOCK_WINDOW_S
from gnssaudit.paths import results_root
from gnssaudit.truth.oracle import k_convergence
from review._common import save
from review.s30_k_rule_null import (
    E2K_DIR,
    K_SMALL,
    SEG,
    SourceRefused,
    e2k_provenance,
    moments_of,
    r_index,
    ratios,
    realizations,
    same_table,
    split,
    unit_record,
    unit_table,
)

RUN2_ID = "6214cf52489f"
RUN2_DIR = results_root() / "E2" / RUN2_ID  # read only
RUN2_SUMMARY_SHA256 = "4419fd70188279102a2ab862fda3ef17acdbb81018b8793918d165b11e962c99"
GROUPS = {"K-study blocks": ("A-k-cond", "A-k-noise"), "thermal block": ("A-thermal", "A-thermal-clean")}
START = re.compile(r"Tracking of GPS L1 C/A signal started on channel (\d+) for satellite GPS PRN (\d+)")
SYNC = re.compile(r"GPS L1 C/A (?:histogram|tracking) bit synchronization locked in channel (\d+)")
CLOCK = re.compile(r"Current receiver time: (?:(\d+) min )?(\d+) s")


def fisher_p(table: Sequence[Sequence[int]]) -> float:
    """Fisher's exact test, two-sided p (scipy returns (statistic, pvalue))."""
    return float(np.asarray(fisher_exact(table), dtype=float)[1])


def block_of(name: str) -> str:
    return re.sub(r"-r\d+$", "", name)


# ------------------------------------------------------------------------------------------------ sources
def read_run2(run: Path = RUN2_DIR) -> dict[str, Any]:
    """Run 2's manifest must name 6214cf52489f and its summary must be the pinned one."""
    found = json.loads((run / "manifest.json").read_text())
    if found.get("run_id") != RUN2_ID:
        raise SourceRefused(f"{run}: the manifest names run {found.get('run_id')!r}, not E2 run 2 ({RUN2_ID})")
    raw = (run / "summary.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != RUN2_SUMMARY_SHA256:
        raise SourceRefused(f"{run}: summary.json is not run 2's pinned summary")
    seg = json.loads(raw)["segments"][SEG]
    return {"run_dir": str(run), "warmup_s": float(found["spec"]["warmup_s"]),
            "channels": {int(ch): int(v[2]) for ch, v in seg["channels"].items()},
            "line_excess_db": float(seg["line_excess_db"])}  # fmt: skip


def hashed_json(unit: Path, name: str) -> dict[str, Any]:
    """A JSON file of a unit, refused unless its SHA-256 is the one the unit's done.json recorded."""
    want = json.loads((unit / "done.json").read_text())["files"].get(name)
    data = (unit / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != want:
        raise SourceRefused(f"{unit / name}: not the file done.json hashed")
    return json.loads(data)


# ------------------------------------------------------------------------------------------------ Q1
def parse_log(text: str) -> dict[int, dict[str, list[float]]]:
    """Per channel: the receiver times (s) at which tracking started and at which bit synchronisation locked."""
    now = 0.0
    out: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"starts": [], "sync": []})
    for line in text.splitlines():
        if m := CLOCK.match(line):
            now = 60.0 * float(m.group(1) or 0) + float(m.group(2))
        elif m := START.match(line):
            out[int(m.group(1))]["starts"].append(now)
        elif m := SYNC.match(line):
            out[int(m.group(1))]["sync"].append(now)
    return out


def attempt_logs(unit: Path) -> list[Path]:
    """The unit's receiver logs in run order: <name>.gnss-sdr.log, then <name>~1, <name>~2."""
    rest = sorted(unit.glob(f"{unit.name}~*.gnss-sdr.log"), key=lambda p: int(p.name.split("~")[1].split(".")[0]))
    return [unit / f"{unit.name}.gnss-sdr.log", *rest]


def ghost_runs(units: Mapping[str, Mapping[str, Any]], ghosts: Mapping[int, int], warmup_s: float) -> list[dict]:
    """One row per unit, receiver run and ghost channel: tracking start, bit sync by the end of the warm-up, and, for
    the final run, whether unit.json records the channel as unlocked."""
    rows = []
    for name, rec in units.items():
        logs = attempt_logs(E2K_DIR / "units" / name)
        if len(logs) != rec["attempts"] or not all(p.is_file() for p in logs):
            raise SourceRefused(f"{name}: {len(logs)} receiver logs for {rec['attempts']} receiver runs")
        for k, path in enumerate(logs, start=1):
            parsed = parse_log(path.read_text(errors="replace"))
            final = k == len(logs)
            for ch, prn in ghosts.items():
                c = parsed.get(ch, {"starts": [], "sync": []})
                rows.append({"unit": name, "block": block_of(name), "run": k, "final": final, "ch": ch, "prn": prn,
                             "first_start_s": c["starts"][0] if c["starts"] else None,
                             "bit_sync": any(t <= warmup_s for t in c["sync"]),
                             "unlocked_final": (ch in rec["unlocked"]) if final else None})  # fmt: skip
    return rows


def window_stats(track: pa.Table, set_doppler: float | None, window: tuple[float, float]) -> dict[str, float]:
    t = track.column("t_rx_s").to_numpy()
    w = (t >= window[0]) & (t < window[1])

    def med(col: str) -> float:
        return float(np.median(track.column(col).to_numpy(zero_copy_only=False).astype(float)[w]))

    dop = med("dop_hz")
    return {"cn0_dbhz": med("cn0_dbhz"), "lock": med("lock_test"),
            "doppler_offset_hz": dop - set_doppler if set_doppler is not None else math.nan}  # fmt: skip


def clean_twin_window(units: Sequence[Path], channels: Mapping[int, int], set_dop: Mapping[int, float],
                      window: tuple[float, float]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """Per channel, over the clean twins' tracks that hold it: the window's medians, summarised over the twins."""
    per: dict[int, list[dict[str, float]]] = defaultdict(list)
    for d in units:
        files = json.loads((d / "done.json").read_text())["files"]
        for ch, prn in channels.items():
            name = f"track_ch{ch}.parquet"
            if name in files:
                per[ch].append(window_stats(unit_table(d, name), set_dop.get(prn), window))
    out = {}
    for ch, rows in sorted(per.items()):
        cn0 = np.array([r["cn0_dbhz"] for r in rows])
        lk = np.array([r["lock"] for r in rows])
        dd = np.array([r["doppler_offset_hz"] for r in rows])
        out[str(ch)] = {"prn": channels[ch], "twins": len(rows), "cn0_dbhz_median": float(np.median(cn0)),
                        "cn0_dbhz_min": float(cn0.min()), "cn0_dbhz_max": float(cn0.max()),
                        "lock_median": float(np.median(lk)), "lock_min": float(lk.min()),
                        "doppler_offset_hz_max_abs": float(np.nanmax(np.abs(dd))) if np.isfinite(dd).any() else None}  # fmt: skip
    return out


def by_run(units: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> list[dict[str, int]]:
    """Per receiver run k: the units that reached it and those that missed the lock policy on it."""
    out = []
    for k in range(1, RX_ATTEMPTS + 1):
        reached = [n for n in names if units[n]["attempts"] >= k]
        missed = [n for n in reached if units[n]["attempts"] > k or not units[n]["lock_ok"]]
        out.append({"run": k, "reached": len(reached), "missed": len(missed)})
    return out


def unlocks(units: Mapping[str, Mapping[str, Any]], run2: Mapping[str, Any]) -> dict[str, Any]:
    """Q1."""
    ghost_list = hashed_json(E2K_DIR / "source" / "units" / "survey-A", "survey.json")["ghosts"]
    set_dop = {int(g["prn"]): float(g["doppler_hz"]) for g in ghost_list}
    ghosts = {ch: prn for ch, prn in run2["channels"].items() if prn in set_dop}
    rows = ghost_runs(units, ghosts, run2["warmup_s"])
    final = Counter((r["unlocked_final"], r["bit_sync"]) for r in rows if r["final"])
    if final[(True, True)] or final[(False, False)]:
        raise SourceRefused(f"the bit-sync proxy disagrees with unit.json on final runs: {dict(final)}")
    groups: dict[str, Any] = {}
    for label, blocks in GROUPS.items():
        g = [r for r in rows if r["block"] in blocks]
        names = [n for n in units if block_of(n) in blocks]
        per_prn = {str(prn): {"ghost_runs": sum(r["prn"] == prn for r in g),
                              "false_locks": sum(r["prn"] == prn and not r["bit_sync"] for r in g)}
                   for prn in ghosts.values()}  # fmt: skip
        starts = Counter(r["first_start_s"] for r in g)
        groups[label] = {
            "blocks": list(blocks),
            "units": len(names),
            "lock_failed": sorted(n for n in names if not units[n]["lock_ok"]),
            "ghost_unlocked_final": {
                n: [c for c in units[n]["unlocked"] if c in ghosts]
                for n in sorted(names)
                if set(units[n]["unlocked"]) & set(ghosts)
            },
            "attempts": dict(Counter(units[n]["attempts"] for n in names)),
            "by_run": by_run(units, names),
            "missed_first_run": sum(units[n]["attempts"] > 1 for n in names),
            "missed_first_run_then_passed": sum(units[n]["attempts"] > 1 and units[n]["lock_ok"] for n in names),
            "ghost_runs": len(g),
            "ghost_false_locks": sum(not r["bit_sync"] for r in g),
            "ghost_false_locks_by_prn": per_prn,
            "ghost_tracking_first_start_s": {str(k): v for k, v in sorted(starts.items(), key=lambda kv: kv[0] or -1)},
            "unlocked_channel_counts": {
                str(c): v for c, v in sorted(Counter(c for n in names for c in units[n]["unlocked"]).items())
            },
        }
    k_rate = groups["K-study blocks"]["ghost_false_locks"] / groups["K-study blocks"]["ghost_runs"]
    t_rate = groups["thermal block"]["ghost_false_locks"] / groups["thermal block"]["ghost_runs"]
    th = [n for n in units if block_of(n) in GROUPS["thermal block"]]
    first = sum(units[n]["attempts"] > 1 for n in th)
    second = [n for n in th if units[n]["attempts"] >= 2]
    second_missed = sum(units[n]["attempts"] > 2 or not units[n]["lock_ok"] for n in second)
    tab_runs = [[first, len(th) - first], [second_missed, len(second) - second_missed]]
    pairs = Counter((units[f"A-thermal-r{r}"]["attempts"] > 1, units[f"A-thermal-clean-r{r}"]["attempts"] > 1)
                    for r in range(len(th) // 2))  # fmt: skip
    tab_pairs = [[pairs[(True, True)], pairs[(True, False)]], [pairs[(False, True)], pairs[(False, False)]]]
    n_pairs = sum(sum(row) for row in tab_pairs)
    expected_both = (tab_pairs[0][0] + tab_pairs[0][1]) * (tab_pairs[0][0] + tab_pairs[1][0]) / n_pairs
    repeat = {}
    for n in units:
        if n.startswith("repeat-"):
            orig = n.removeprefix("repeat-")
            repeat[n] = {"original": orig, "same_bytes": units[n]["output_sha256"] == units[orig]["output_sha256"],
                         "attempts": units[n]["attempts"], "lock_ok": units[n]["lock_ok"],
                         "original_attempts": units[orig]["attempts"], "original_lock_ok": units[orig]["lock_ok"]}  # fmt: skip
    clean_dirs = sorted((E2K_DIR / "units").glob(f"{SEG}-thermal-clean-r*"), key=r_index)
    window = (run2["warmup_s"] - WARMUP_LOCK_WINDOW_S, run2["warmup_s"])
    windows = {
        "run 2 A-clean-A (no floor; every K-study twin equals it before the onset)": clean_twin_window(
            [Path(run2["run_dir"]) / "units" / "A-clean-A"], run2["channels"], set_dop, window
        ),
        "E2K thermal-clean, final runs (floor at N0)": clean_twin_window(clean_dirs, run2["channels"], set_dop, window),
    }
    no_floor, floor = windows.values()
    drop = {str(prn): no_floor[str(ch)]["cn0_dbhz_median"] - floor[str(ch)]["cn0_dbhz_median"]
            for ch, prn in run2["channels"].items() if str(ch) in no_floor and str(ch) in floor}  # fmt: skip
    return {
        "ghosts": {str(ch): {"prn": prn, "set_doppler_hz": set_dop[prn]} for ch, prn in ghosts.items()},
        "warmup_s": run2["warmup_s"],
        "check": {
            "bars": {
                "lock_min": WARMUP_LOCK_MIN,
                "doppler_tol_hz": WARMUP_DOPPLER_TOL_HZ,
                "window_s": list(window),
                "receiver_runs_max": RX_ATTEMPTS,
            },
            "bit_sync_proxy_on_final_runs": {
                "unlocked_without_bit_sync": final[(True, False)],
                "locked_with_bit_sync": final[(False, True)],
            },
        },
        "groups": groups,
        "ghost_false_lock_rate": {"K-study blocks": k_rate, "thermal block": t_rate, "ratio": t_rate / k_rate},
        "fisher_first_vs_second_run": {"table": tab_runs, "p_two_sided": fisher_p(tab_runs)},
        "fisher_pairs_first_run": {
            "table_TT_TF_FT_FF": tab_pairs,
            "expected_both": expected_both,
            "p_two_sided": fisher_p(tab_pairs),
        },
        "repeats": repeat,
        "ch3_excluded_in_every_unit": all(3 in u["unlocked"] for u in units.values()),
        "window": windows,
        "cn0_drop_db_by_prn": drop,
        "cn0_drop_predicted_db": 10 * math.log10(1 + 10 ** (-run2["line_excess_db"] / 10)),
        "line_excess_db": run2["line_excess_db"],
        "ghost_runs_rows": rows,
    }


# ------------------------------------------------------------------------------------------------ Q2
def variance_ratio_by_prn(small: pa.Table, large: pa.Table, onset: float) -> dict[str, float]:
    """Per satellite, the median var_large / var_small over common rows from onset + SETTLE_S."""
    r = ratios(small, large)
    post = r["t"] >= onset + SETTLE_S
    return {str(p): float(np.median(r["ratio"][post & (r["prn"] == p)])) for p in sorted(set(r["prn"].tolist()))}


def affected_pairs(units: Mapping[str, Mapping[str, Any]], jammed: Sequence[str],
                   false_locked: set[str]) -> tuple[list[int], list[int]]:  # fmt: skip
    """The thermal pairs (r) to leave out: (a) those with a twin that failed the lock policy (lock_ok false in the
    jammed twin or in its clean twin, unit.json unit.clean); (b) those with a ghost false lock in any receiver run of
    either twin (false_locked: the units with one)."""
    twins = {r_index(n): (n, units[n]["unit"]["clean"]) for n in jammed}
    lock_failed = sorted(r for r, tw in twins.items() if not all(units[t]["lock_ok"] for t in tw))
    any_run = sorted(r for r, tw in twins.items() if false_locked & set(tw))
    return lock_failed, any_run


def thermal(units: Mapping[str, Mapping[str, Any]], summary: Mapping[str, Any], rows: Sequence[dict]) -> dict:
    """Q2: E2K's thermal comparison rebuilt, then without the affected pairs (post hoc)."""
    dirs = {b: realizations(E2K_DIR, b) for b in ("k-cond", "k-noise", "thermal")}
    truths = {b: [unit_table(d, "truth_obs.parquet") for d in ds] for b, ds in dirs.items()}
    slips = {b: [unit_table(d, "slips_obs.parquet") for d in ds] for b, ds in dirs.items()}
    onsets = {units[d.name]["obs_onset_label"] for ds in dirs.values() for d in ds}
    if len(onsets) != 1:
        raise SourceRefused(f"E2K's oracle units disagree on the onset label: {sorted(onsets)}")
    onset = onsets.pop()
    full = {b: moments_of(b, truths[b], slips[b]) for b in dirs}
    if not same_table(full["thermal"], pq.read_table(E2K_DIR / f"oracle_{SEG}_thermal.parquet")):
        raise SourceRefused("oracle_A_thermal.parquet is not the table its units rebuild to")
    base51 = moments_of("k-cond", truths["k-cond"][:K_SMALL], slips["k-cond"][:K_SMALL])
    as_e2k = k_convergence(base51, full["thermal"])
    if as_e2k != summary["decisions"]["thermal_vs_fixed"]:
        raise SourceRefused("the rebuilt comparison is not summary.json's decisions.thermal_vs_fixed")
    pair_of = {d.name: r_index(d) for d in dirs["thermal"]}
    lock_failed, any_run = affected_pairs(units, list(pair_of), {r["unit"] for r in rows if not r["bit_sync"]})
    cases: dict[str, Any] = {
        "as E2K (51 pairs)": {"dropped": [], **as_e2k, "split": split(base51, full["thermal"], onset)}
    }
    for label, drop in (("(a) without pairs with a lock-failed twin", lock_failed),
                        ("(b) without pairs with a ghost false lock in any receiver run", any_run)):  # fmt: skip
        keep = [i for i, d in enumerate(dirs["thermal"]) if pair_of[d.name] not in drop]
        kept = moments_of("thermal", [truths["thermal"][i] for i in keep], [slips["thermal"][i] for i in keep])
        same_k = moments_of("k-cond", truths["k-cond"][: len(keep)], slips["k-cond"][: len(keep)])
        cases[label] = {"dropped": drop, "pairs": len(keep),
                        "base_51": {**k_convergence(base51, kept), "split": split(base51, kept, onset)},
                        "base_same_k": {"k": len(keep), **k_convergence(same_k, kept),
                                        "split": split(same_k, kept, onset)}}  # fmt: skip
    tables = {"k-cond first 51": base51, "k-noise 51": full["k-noise"], "thermal 51": full["thermal"]}
    pre_post = {}
    for label, t in tables.items():
        tt, v = t.column("t_rx_s").to_numpy(), t.column("var_cyc2").to_numpy(zero_copy_only=False)
        pre_post[label] = {"pre_onset": float(np.nanmedian(v[tt < onset])),
                           "post_onset": float(np.nanmedian(v[tt >= onset + SETTLE_S]))}  # fmt: skip
    prn = full["thermal"].column("prn").to_numpy()
    nv = full["thermal"].column("n_valid").to_numpy()
    return {
        "onset_label": onset,
        "cases": cases,
        "variance_ratio_post_onset_by_prn": {
            "thermal 51 / k-cond first 51": variance_ratio_by_prn(base51, full["thermal"], onset),
            "k-noise 51 / k-cond first 51 (reference)": variance_ratio_by_prn(base51, full["k-noise"], onset),
        },
        "median_var_cyc2_pre_and_post_onset": pre_post,
        "thermal_realizations_by_prn": {str(p): int(nv[prn == p].max()) for p in sorted(set(prn.tolist()))},
    }


def main() -> None:
    t0 = time.time()
    prov = e2k_provenance()
    summary = prov.pop("summary")
    run2 = read_run2()
    units = {d.name: unit_record(d) for d in sorted((E2K_DIR / "units").iterdir())}
    q1 = unlocks(units, run2)
    rows = q1.pop("ghost_runs_rows")
    for label, g in q1["groups"].items():
        print(f"{label}: units {g['units']}, lock-failed {len(g['lock_failed'])}, missed the first run "
              f"{g['missed_first_run']} (then passed {g['missed_first_run_then_passed']}), ghost false locks "
              f"{g['ghost_false_locks']}/{g['ghost_runs']} {g['ghost_false_locks_by_prn']}")  # fmt: skip
    print(f"proxy on final runs {q1['check']['bit_sync_proxy_on_final_runs']}; rate ratio "
          f"{q1['ghost_false_lock_rate']['ratio']:.1f}; C/N0 drop {q1['cn0_drop_db_by_prn']} (predicted "
          f"{q1['cn0_drop_predicted_db']:.2f} dB); Fisher runs p {q1['fisher_first_vs_second_run']['p_two_sided']:.3g}, "
          f"pairs p {q1['fisher_pairs_first_run']['p_two_sided']:.3g}")  # fmt: skip
    q2 = thermal(units, summary, rows)
    for label, c in q2["cases"].items():
        if "base_51" in c:
            print(f"{label}: {c['pairs']} pairs, base 51 {c['base_51']['median_rel_change']:.4f}, base "
                  f"{c['base_same_k']['k']} {c['base_same_k']['median_rel_change']:.4f}")  # fmt: skip
        else:
            s = c["split"]
            print(f"{label}: {c['median_rel_change']:.4f}; before the onset {s['pre_onset']:.2f} ({s['n_pre']} rows), "
                  f"from onset + {SETTLE_S:g} s {s['post_onset']:.4f} ({s['n_post']})")  # fmt: skip
    save(
        "s31_e2k_warmup_unlocks",
        {
            "label": "post hoc, read-only, not deciding (not pre-registered)",
            "e2k": {
                **prov,
                "note": "gate-failed run (G-warmup-lock), so no manifest.json; pinned by started.json's "
                "describe and summary.json's SHA-256 (s30's checks); unit.json and parquet files checked against "
                "each unit's done.json; receiver logs are not hashed by the run",
            },
            "run2": {"run_id": RUN2_ID, "summary_sha256": RUN2_SUMMARY_SHA256, **run2},
            "q1_unlocks": q1,
            "q2_thermal": q2,
            "run_time_s": round(time.time() - t0, 1),
        },
    )


if __name__ == "__main__":
    main()
