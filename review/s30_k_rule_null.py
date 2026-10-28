"""S30: the K rule's null value (P2b part 2, task 11c). Written after E2K's result: it reports, it decides nothing.

Question. docs/5 section 4.4 sets K = 51 if the median over epochs of |sigma^2_101 / sigma^2_51 - 1| is at most 0.2,
else K = 101; 0.2 is sqrt(2 / (51 - 1)), the relative SE of sigma^2_oracle at K = 51. E2K (run 0443516ca48c, code
3eaed0b) computed it at 0.0979 on 26 898 epoch-satellite rows and chose K = 51. What does the statistic give under a
stationary Gaussian ensemble with E2K's epoch count, where nothing but sampling varies?

Method. The oracle's own estimator: gnssaudit.truth.oracle.oracle_moments takes per epoch and satellite the sample
variance over the realizations (numpy var, ddof 1), and k_convergence the median over common epochs of
|var_large / var_small - 1|. The statistic is scale-free, so unit-variance draws stand for any sigma^2. Each replication
draws 26 898 rows (E2K: 4 483 epochs x 6 satellites, all common) of 101 standard normal realizations and an
independent 51, and takes three designs:
  - nested: the 101 against their first 51 (K's rule as E2K ran it);
  - independent 51 against the 101 (E2K's k-noise block against its K = 101 block);
  - 51 against 51, both independent (the design of E2K's thermal comparison).
400 replications, one SeedSequence(20261022).spawn stream each, so the result is deterministic. The same again with an
AR(1) along the epochs at rho = 0.55: E2K's truth errors after the onset have lag-1 autocorrelation 0.31 to 0.59, so
this over-states their dependence. Known answer, checked first (the study stops if it fails): on a 300-row replication
the vectorised statistic equals oracle_moments + k_convergence to 1e-12, nested and independent.

E2K, read-only. The run failed G-warmup-lock after writing every result, so it has no manifest.json. It is pinned by
started.json's describe (3eaed0b7eeb6) and summary.json's SHA-256, and every unit file read must match the SHA-256 in
that unit's done.json (unit.json included); a directory other than 0443516ca48c is refused. From the units, the
oracle tables are rebuilt as e2.k_study_decision builds them (realizations with truth, in r order; config "A-<block>",
form marginal, slip window 1 s). The study stops unless the rebuilt K = 101 and k-noise tables equal E2K's
oracle_A_*.parquet and the rebuilt statistic equals summary.json's decisions.k_convergence. It then records:
  - the observed statistic in each design, on all common epochs and split at the jammer onset (before it, a jammed
    twin and its clean twin hold the same bytes, so those rows measure receiver jitter; after it, from onset + 2 s);
  - the nested null again with the K = 101 block's own validity pattern (a satellite can lack truth in a realization,
    so a row's K is below 101 and 51), 400 replications, and how many of them reach the observed value;
  - the errors' pooled excess kurtosis after the onset, against its spread over the same 400 Gaussian replications
    with that pattern. The sampling variance of s^2 is sigma^4 (2 / (n - 1) + kappa / n), so heavy tails or
    non-stationarity across realizations, not sigma^2, move the statistic;
  - per satellite, the realizations holding truth (n_valid) in each block, and the median sigma^2_oracle from onset
    + 2 s at K = 51 (nested), 101 and the independent 51; and the rise of that median from K = 51 to K = 101 against
    its spread over the same Gaussian replications (the median of s^2 rises with n, since s^2's distribution is
    skewed);
  - the per-epoch distribution of |sigma^2_101 / sigma^2_51 - 1|, on all rows and on the rows before and from the
    onset, and of one Gaussian replication with E2K's pattern (figure f10-5).

E2K's thermal comparison (summary.json thermal_vs_fixed), reported only, is examined post hoc in s31, with the
warm-up unlocks (task 11c fix round 1 moved it there, so that this study answers one question).

Reading, stated before this study ran (E2K's post-run analysis of 22 Oct 2026 had estimated the nested null at about
0.095, with this method): if E2K's 0.0979 lies within the Gaussian null's spread and the errors' kurtosis is a
Gaussian's, the rule passed because the errors sample like a Gaussian ensemble. It does not show that sigma^2_oracle
was tested hard for convergence: at this epoch count the rule passes for any stationary Gaussian ensemble.

Result (review/results/s30_k_rule_null.json; the same with any number of runs: every draw is seeded).
  - The check holds. The null, nested: median 0.0950 (5-95 %: 0.0938-0.0962; min 0.0930, max 0.0970), and 0 of 400
    replications exceed 0.2. An independent 51 against the 101: 0.1648 (0.1627-0.1668). Two independent 51s: 0.1900
    (0.1880-0.1919), only 0.01 below the bar. With AR(1) at rho = 0.55 every median moves by less than 0.0002.
  - E2K. The rebuilt tables equal E2K's and the statistics equal summary.json's. K's rule: 0.0979 on 26 898 rows;
    0.0963 on the 7 500 before the onset and 0.0985 on the 18 798 from onset + 2 s. With E2K's validity pattern
    (n_valid 94 to 101 over the 101, 47 to 51 over the first 51) the Gaussian null is 0.0965 (0.0953-0.0975, max
    0.0985), and 8 of 400 replications reach 0.0979. Excess kurtosis after the onset: -0.1127; over the same 400
    Gaussian replications -0.1190 (5-95 %: -0.1243 to -0.1135), and 10 of 400 reach -0.1127. From K = 51 to K = 101
    the median s^2 after the onset rises by +0.26 to +1.44 % per satellite; for a Gaussian with that pattern by +0.66
    to +0.74 % (medians over replications; positive in 94 to 96 % of them), and each observed rise lies inside its
    satellite's 5-95 % range. k-noise against the K = 101 block: 0.1665 (null 0.1648). k-noise against the K = 101
    block's first 51: 0.1804 on all rows and 0.1931 from onset + 2 s (null 0.1900).
  - The reading holds, with a qualification. E2K's value lies close to the Gaussian null, in its upper ~2 %, and at
    half the bar. The errors' kurtosis lies at the upper edge of a Gaussian's spread (upper ~2.5 %), so their tails
    are slightly heavier than a Gaussian's. K = 51 says the errors are close to a Gaussian ensemble, not that K was
    tested hard.

Usage: uv run python -m review.s30_k_rule_null (reads E2K run 0443516ca48c read-only; about 4 min).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

from gnssaudit.experiments.e2 import SETTLE_S
from gnssaudit.io.schemas import EVENTS, TRUTH_EPOCH_SAT, build
from gnssaudit.paths import results_root
from gnssaudit.truth.oracle import KEY, k_convergence, oracle_moments
from review._common import save

E2K_ID = "0443516ca48c"
E2K_DIR = results_root() / "E2K" / E2K_ID  # read only
E2K_DESCRIBE = "3eaed0b7eeb6"  # started.json: the code E2K ran (commit 3eaed0b)
E2K_SUMMARY_SHA256 = "802582093b0c79c0e5df1c9fcda8f7faae70ce2ebdd81c7c27c78b2b34aa51ed"
SEG, FORM, SLIP_WINDOW_S = "A", "marginal", 1.0  # E2K's blocks ran on the noise fallback: no fixed timing
BLOCKS = ("k-cond", "k-noise")

N_ROWS = 26_898  # E2K: 4 483 observable epochs x 6 satellites, every one common to both ensembles
K_LARGE, K_SMALL = 101, 51
BAR = 0.2  # docs/5 section 4.4: sqrt(2 / (K_SMALL - 1))
REPS, SEED, RHO = 400, 20261022, 0.55
PATTERN_REPS, PATTERN_SEED = 400, 7
CHECK_ROWS = 300
DESIGNS = ("nested", "indep51_vs_101", "indep51_vs_51")
HIST_EDGES = tuple(round(0.01 * i, 2) for i in range(61))  # per-epoch |ratio - 1|; above 0.6 counted apart


class SourceRefused(ValueError):
    """The E2K directory is not run 0443516ca48c as pinned, or a unit file is not the one its done.json hashed."""


# ------------------------------------------------------------------------------------------------ the null
def draws(rng: np.random.Generator, k: int, n: int, rho: float) -> np.ndarray:
    """(k, n) unit-variance realizations, independent across realizations; AR(1) along the n rows when rho > 0."""
    z = rng.standard_normal((k, n))
    if rho == 0.0:
        return z
    start = rng.standard_normal((k, 1))  # x[-1] ~ N(0, 1): the process is stationary from its first row
    x, _ = lfilter([np.sqrt(1.0 - rho * rho)], [1.0, -rho], z, axis=1, zi=rho * start)
    return x


def statistics(x101: np.ndarray, y51: np.ndarray) -> dict[str, float]:
    """The three designs' statistics on one replication: per row sample variances (ddof 1), median |ratio - 1|."""
    v101 = x101.var(axis=0, ddof=1)
    v51 = x101[:K_SMALL].var(axis=0, ddof=1)
    w51 = y51.var(axis=0, ddof=1)
    return {
        "nested": float(np.median(np.abs(v101 / v51 - 1.0))),
        "indep51_vs_101": float(np.median(np.abs(v101 / w51 - 1.0))),
        "indep51_vs_51": float(np.median(np.abs(w51 / v51 - 1.0))),
    }


def simulate(reps: int, seed: int, rho: float, rows: int = N_ROWS) -> dict[str, np.ndarray]:
    """Each design's statistic over `reps` replications, one SeedSequence child each (reproducible in any order)."""
    out: dict[str, list[float]] = {d: [] for d in DESIGNS}
    for ss in np.random.SeedSequence(seed).spawn(reps):
        rng = np.random.default_rng(ss)
        s = statistics(draws(rng, K_LARGE, rows, rho), draws(rng, K_SMALL, rows, rho))
        for d in DESIGNS:
            out[d].append(s[d])
    return {d: np.asarray(v) for d, v in out.items()}


def summarise(v: np.ndarray) -> dict[str, float | int]:
    return {
        "median": float(np.median(v)),
        "p05": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)),
        "n_exceeds_bar": int(np.sum(v > BAR)),
        "reps": int(v.size),
    }


# ------------------------------------------------------------------------------------- the known answer
def truth_table(e: np.ndarray) -> pa.Table:
    """A TRUTH_EPOCH_SAT table holding one realization's errors e on len(e) epochs of one satellite, all valid."""
    n = len(e)
    cols = {"truth_id": ["sim"] * n, "kind": ["twin"] * n, "sys": ["G"] * n, "prn": np.full(n, 1), "signal": ["1C"] * n,
            "t_rx_s": np.arange(n) * 0.02, "e_cyc": e, "slip_count": np.zeros(n), "arc_id": np.zeros(n),
            "transient": np.zeros(n, bool), "valid": np.ones(n, bool), "regime": np.zeros(n)}  # fmt: skip
    return build(TRUTH_EPOCH_SAT, cols, n)


def check_against_project(seed: int = SEED, rows: int = CHECK_ROWS) -> dict[str, Any]:
    """The vectorised statistics against gnssaudit's oracle_moments + k_convergence on one small replication."""
    rng = np.random.default_rng(seed)
    x, y = draws(rng, K_LARGE, rows, 0.0), draws(rng, K_SMALL, rows, 0.0)
    no_slips = build(EVENTS, {c: [] for c in EVENTS.names}, 0)

    def moments(e: np.ndarray) -> pa.Table:
        return oracle_moments([truth_table(r) for r in e], [no_slips] * len(e), config_id="sim", form=FORM,
                              slip_window_s=SLIP_WINDOW_S)  # fmt: skip

    large, small, other = moments(x), moments(x[:K_SMALL]), moments(y)
    project = {
        "nested": k_convergence(small, large)["median_rel_change"],
        "indep51_vs_101": k_convergence(other, large)["median_rel_change"],
        "indep51_vs_51": k_convergence(small, other)["median_rel_change"],
    }
    mine = statistics(x, y)
    for d in DESIGNS:  # rtol: a column of a 2-D var against a 1-D one sums in another order
        if not np.isclose(project[d], mine[d], rtol=1e-12, atol=0.0):
            raise AssertionError(f"{d}: vectorised statistic {mine[d]!r} differs from gnssaudit's {project[d]!r}")
    return {"rows": rows, "gnssaudit": project, "vectorised": mine}


# ------------------------------------------------------------------------------------- E2K (read only)
def e2k_provenance(run: Path = E2K_DIR) -> dict[str, Any]:
    """The directory must be run 0443516ca48c, started.json must name the code E2K ran, and summary.json must be the
    pinned one (no manifest: the run failed a gate)."""
    if run.name != E2K_ID:
        raise SourceRefused(f"{run}: not E2K run {E2K_ID}")
    describe = json.loads((run / "started.json").read_text()).get("git_describe")
    if describe != E2K_DESCRIBE:
        raise SourceRefused(f"{run}: started.json names code {describe!r}, not E2K's {E2K_DESCRIBE}")
    raw = (run / "summary.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != E2K_SUMMARY_SHA256:
        raise SourceRefused(f"{run}: summary.json SHA-256 {digest}, not E2K's {E2K_SUMMARY_SHA256}")
    return {"run_id": E2K_ID, "run_dir": str(run), "describe": describe, "summary_sha256": digest,
            "summary": json.loads(raw)}  # fmt: skip


def unit_table(unit: Path, name: str) -> pa.Table:
    """One parquet file of a unit, refused unless its SHA-256 is the one the unit's done.json recorded."""
    want = json.loads((unit / "done.json").read_text())["files"].get(name)
    data = (unit / name).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise SourceRefused(f"{unit / name}: SHA-256 {got}, done.json records {want}")
    return pq.read_table(pa.BufferReader(data))


def unit_record(unit: Path) -> dict[str, Any]:
    """A unit's unit.json, refused unless its SHA-256 is the one the unit's done.json recorded."""
    want = json.loads((unit / "done.json").read_text())["files"].get("unit.json")
    data = (unit / "unit.json").read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise SourceRefused(f"{unit / 'unit.json'}: SHA-256 {got}, done.json records {want}")
    return json.loads(data)


def r_index(unit: Path | str) -> int:
    return int(str(unit).rsplit("-r", 1)[1])


def realizations(run: Path, block: str) -> list[Path]:
    """A block's units that hold truth, in realization order (r2 before r10): e2._realizations at 3eaed0b."""
    dirs = sorted((run / "units").glob(f"{SEG}-{block}-r*"), key=r_index)
    return [d for d in dirs if (d / "truth_obs.parquet").exists()]


def moments_of(block: str, truths: Sequence[pa.Table], slips: Sequence[pa.Table]) -> pa.Table:
    return oracle_moments(truths, slips, config_id=f"{SEG}-{block}", form=FORM, slip_window_s=SLIP_WINDOW_S)


def same_table(a: pa.Table, b: pa.Table) -> bool:
    if a.schema.names != b.schema.names or a.num_rows != b.num_rows:
        return False
    for c in a.schema.names:
        x, y = a.column(c).to_numpy(zero_copy_only=False), b.column(c).to_numpy(zero_copy_only=False)
        floats = np.issubdtype(x.dtype, np.floating)
        if not np.array_equal(x, y, equal_nan=floats):
            return False
    return True


def ratios(small: pa.Table, large: pa.Table) -> dict[str, np.ndarray]:
    """var_large / var_small on k_convergence's common rows (both variances non-zero and finite), with t and PRN."""
    a = dict(zip(zip(*(small.column(c).to_pylist() for c in KEY), strict=True), small.column("var_cyc2").to_pylist(),
                 strict=True))  # fmt: skip
    b = dict(zip(zip(*(large.column(c).to_pylist() for c in KEY), strict=True), large.column("var_cyc2").to_pylist(),
                 strict=True))  # fmt: skip
    common = [k for k in a if k in b and a[k] and b[k] and np.isfinite(a[k]) and np.isfinite(b[k])]
    return {"t": np.array([k[3] for k in common]), "prn": np.array([k[1] for k in common]),
            "ratio": np.array([b[k] / a[k] for k in common])}  # fmt: skip


def split(small: pa.Table, large: pa.Table, onset: float) -> dict[str, Any]:
    """k_convergence's statistic on its common rows, on those before the onset, and on those from onset + SETTLE_S."""
    r = ratios(small, large)
    c = np.abs(r["ratio"] - 1.0)
    pre, post = r["t"] < onset, r["t"] >= onset + SETTLE_S
    return {"all": float(np.median(c)), "n_all": int(c.size), "pre_onset": float(np.median(c[pre])),
            "n_pre": int(pre.sum()), "post_onset": float(np.median(c[post])), "n_post": int(post.sum())}  # fmt: skip


def histogram(values: np.ndarray) -> dict[str, Any]:
    counts, _ = np.histogram(values, bins=np.asarray(HIST_EDGES))
    return {"counts": counts.tolist(), "above": int(np.sum(values > HIST_EDGES[-1])), "n": int(values.size),
            "median": float(np.median(values))}  # fmt: skip


def per_prn(tables: Mapping[str, pa.Table], onset: float) -> dict[str, dict[str, Any]]:
    """Per satellite and table: n_valid (min, max over epochs) and the median sigma^2 on all rows and from onset + 2 s."""
    out: dict[str, dict[str, Any]] = {}
    for label, t in tables.items():
        prn = t.column("prn").to_numpy()
        tt = t.column("t_rx_s").to_numpy()
        nv = t.column("n_valid").to_numpy()
        v = t.column("var_cyc2").to_numpy(zero_copy_only=False)
        for p in sorted(set(prn.tolist())):
            m = prn == p
            post = m & (tt >= onset + SETTLE_S)
            out.setdefault(str(p), {})[label] = {
                "n_valid_min": int(nv[m].min()),
                "n_valid_max": int(nv[m].max()),
                "median_var_cyc2_all": float(np.nanmedian(v[m])),
                "median_var_cyc2_post_onset": float(np.nanmedian(v[post])),
            }
    return out


def error_matrix(truths: Sequence[pa.Table], rows: pa.Table) -> np.ndarray:
    """(realizations, rows) truth errors on the oracle table's rows; NaN where a realization's row is invalid or
    transient (oracle_moments leaves those out)."""
    index = {key: i for i, key in enumerate(zip(*(rows.column(c).to_pylist() for c in KEY), strict=True))}
    e = np.full((len(truths), rows.num_rows), np.nan)
    for r, tr in enumerate(truths):
        ok = tr.column("valid").to_numpy(zero_copy_only=False) & ~tr.column("transient").to_numpy(zero_copy_only=False)
        keys = zip(*(tr.column(c).to_pylist() for c in KEY), strict=True)
        for key, v, good in zip(keys, tr.column("e_cyc").to_pylist(), ok, strict=True):
            if good:
                e[r, index[key]] = v
    return e


def nested_rows(x: np.ndarray) -> np.ndarray:
    """|var_101 / var_51 - 1| per row of a (realizations, rows) matrix with NaN for missing realizations."""
    v101, v51 = np.nanvar(x, axis=0, ddof=1), np.nanvar(x[:K_SMALL], axis=0, ddof=1)
    ok = np.isfinite(v101) & np.isfinite(v51) & (v101 != 0) & (v51 != 0)
    return np.abs(v101[ok] / v51[ok] - 1.0)


def excess_kurtosis(x: np.ndarray) -> float:
    """Pooled excess kurtosis of the errors standardised per row by the row's own mean and SD (ddof 1)."""
    z = (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0, ddof=1)
    return float(np.nanmean(z**4) - 3.0)


def pattern_null(e: np.ndarray, post: np.ndarray, prn: np.ndarray, observed: float) -> dict[str, Any]:
    """The nested statistic's Gaussian null with E2K's validity pattern (NaN where e is NaN); over the same Gaussian
    replications, the pooled excess kurtosis after the onset and, per satellite, the rise of the median s^2 after the
    onset from the first 51 to the 101; the observed values of both; and one replication's per-row statistic."""
    mask = np.isfinite(e)
    sats = sorted(set(prn.tolist()))
    rows_of = {p: post & (prn == p) for p in sats}

    def rises(x: np.ndarray) -> dict[int, float]:
        v101, v51 = np.nanvar(x, axis=0, ddof=1), np.nanvar(x[:K_SMALL], axis=0, ddof=1)
        return {p: float(np.median(v101[m]) / np.median(v51[m]) - 1.0) for p, m in rows_of.items()}

    stats, kurt, rise, first = [], [], {p: [] for p in sats}, np.empty(0)
    for i, ss in enumerate(np.random.SeedSequence(PATTERN_SEED).spawn(PATTERN_REPS)):
        x = np.random.default_rng(ss).standard_normal(e.shape)
        x[~mask] = np.nan
        rows = nested_rows(x)
        stats.append(float(np.median(rows)))
        kurt.append(excess_kurtosis(x[:, post]))
        for p, r in rises(x).items():
            rise[p].append(r)
        if i == 0:
            first = rows
    ref = np.random.default_rng(np.random.SeedSequence(PATTERN_SEED + 1)).standard_normal(e.shape)
    ref[~mask] = np.nan
    v, k = np.asarray(stats), np.asarray(kurt)
    seen, kurt_seen = rises(e), excess_kurtosis(e[:, post])
    return {
        "n_valid_101_min_median": [int(mask.sum(0).min()), float(np.median(mask.sum(0)))],
        "n_valid_first51_min_median": [int(mask[:K_SMALL].sum(0).min()), float(np.median(mask[:K_SMALL].sum(0)))],
        "null": summarise(v),
        "reps_at_or_above_observed": int(np.sum(v >= observed)),
        "excess_kurtosis_post_onset": {
            "observed": kurt_seen,
            "reps_at_or_above_observed": int(np.sum(k >= kurt_seen)),
            "gaussian_same_pattern": excess_kurtosis(ref[:, post]),  # one reference draw (seed 8), as first run
            "gaussian_same_pattern_reps": {
                "median": float(np.median(k)),
                "p05": float(np.percentile(k, 5)),
                "p95": float(np.percentile(k, 95)),
                "min": float(k.min()),
                "max": float(k.max()),
                "sd": float(k.std(ddof=1)),
                "reps": int(k.size),
            },
        },
        "median_s2_rise_51_to_101_post_onset_by_prn": {
            str(p): {
                "observed": seen[p],
                "gaussian_same_pattern": {
                    "median": float(np.median(rise[p])),
                    "p05": float(np.percentile(rise[p], 5)),
                    "p95": float(np.percentile(rise[p], 95)),
                    "share_positive": float(np.mean(np.asarray(rise[p]) > 0)),
                    "reps": len(rise[p]),
                },
            }
            for p in sats
        },
        "first_replication_rows": first,
    }


def observed(run: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Everything s30 reads from E2K, each table rebuilt from hash-checked unit files and checked against E2K's own."""
    dirs = {b: realizations(run, b) for b in BLOCKS}
    truths = {b: [unit_table(d, "truth_obs.parquet") for d in ds] for b, ds in dirs.items()}
    slips = {b: [unit_table(d, "slips_obs.parquet") for d in ds] for b, ds in dirs.items()}
    onsets = {unit_record(d)["obs_onset_label"] for ds in dirs.values() for d in ds}
    if len(onsets) != 1:
        raise SourceRefused(f"E2K's oracle units disagree on the onset label: {sorted(onsets)}")
    onset = onsets.pop()
    full = {b: moments_of(b, truths[b], slips[b]) for b in BLOCKS}
    stored = {b: pq.read_table(run / f"oracle_{SEG}_{b}.parquet") for b in BLOCKS}
    for b in BLOCKS:
        if not same_table(full[b], stored[b]):
            raise SourceRefused(f"oracle_{SEG}_{b}.parquet is not the table its units rebuild to")
    cond51 = moments_of("k-cond", truths["k-cond"][:K_SMALL], slips["k-cond"][:K_SMALL])
    k_rule = k_convergence(cond51, full["k-cond"])
    if k_rule != summary["decisions"]["k_convergence"]:
        raise SourceRefused("the rebuilt statistic is not summary.json's decisions.k_convergence")
    if k_rule["epochs"] != N_ROWS:
        raise SourceRefused(f"E2K's K rule ran on {k_rule['epochs']} rows, the null on {N_ROWS}")
    designs = {
        "nested": (cond51, full["k-cond"], "k-cond 101 against its first 51 (K's rule)"),
        "indep51_vs_101": (full["k-noise"], full["k-cond"], "k-cond 101 against k-noise 51"),
        "indep51_vs_51": (cond51, full["k-noise"], "k-noise 51 against k-cond's first 51"),
    }
    obs = {
        d: {"label": lab, **k_convergence(s, lg), "split": split(s, lg, onset)} for d, (s, lg, lab) in designs.items()
    }

    e = error_matrix(truths["k-cond"], full["k-cond"])
    rebuilt = np.nanvar(e, axis=0, ddof=1)
    if not np.allclose(rebuilt, full["k-cond"].column("var_cyc2").to_numpy(zero_copy_only=False), rtol=1e-9, atol=0.0):
        raise SourceRefused("the k-cond error matrix does not reproduce oracle_A_k-cond's var_cyc2")
    post = full["k-cond"].column("t_rx_s").to_numpy() >= onset + SETTLE_S
    pattern = pattern_null(e, post, full["k-cond"].column("prn").to_numpy(), obs["nested"]["median_rel_change"])
    r = ratios(cond51, full["k-cond"])
    nested, before = np.abs(r["ratio"] - 1.0), r["t"] < onset
    hist = {"edges": list(HIST_EDGES), "observed_nested": histogram(nested),
            "observed_nested_before_onset": histogram(nested[before]),
            "observed_nested_from_onset": histogram(nested[~before]),
            "gaussian_same_pattern": histogram(pattern.pop("first_replication_rows"))}  # fmt: skip
    return {
        "onset_label": onset,
        "settle_s": SETTLE_S,
        "units_read": {b: [d.name for d in ds] for b, ds in dirs.items()},
        "oracle_sha256": {
            b: hashlib.sha256((run / f"oracle_{SEG}_{b}.parquet").read_bytes()).hexdigest() for b in BLOCKS
        },  # fmt: skip
        "rebuilt_equals_stored": {b: True for b in BLOCKS},
        "k_column": {b: sorted(set(stored[b].column("K").to_pylist())) for b in BLOCKS},
        "designs": obs,
        "realized_pattern": pattern,
        "per_prn": per_prn(
            {
                "K = 51 (k-cond first 51)": cond51,
                "K = 101 (k-cond)": full["k-cond"],
                "independent 51 (k-noise)": full["k-noise"],
            },
            onset,
        ),
        "histogram": hist,
    }


def main() -> None:
    t0 = time.time()
    check = check_against_project()
    print(f"check: gnssaudit {check['gnssaudit']} = vectorised {check['vectorised']} ({check['rows']} rows)")
    null = {}
    for rho, seed in ((0.0, SEED), (RHO, SEED + 1)):
        sims = simulate(REPS, seed, rho)
        null[f"rho={rho:g}"] = {d: summarise(v) for d, v in sims.items()}
        for d, s in null[f"rho={rho:g}"].items():
            print(f"rho={rho:<4g} {d:15s} median {s['median']:.4f}  5-95 % [{s['p05']:.4f}, {s['p95']:.4f}]  "
                  f"min {s['min']:.4f} max {s['max']:.4f}  > {BAR}: {s['n_exceeds_bar']}/{s['reps']}")  # fmt: skip
    prov = e2k_provenance()
    summary = prov.pop("summary")
    obs = observed(E2K_DIR, summary)
    pat = obs["realized_pattern"]
    print(f"E2K {E2K_ID}: rebuilt oracle tables equal E2K's; K rule {obs['designs']['nested']['median_rel_change']:.4f} "
          f"= summary.json; Gaussian null with E2K's pattern median {pat['null']['median']:.4f} "
          f"[{pat['null']['p05']:.4f}, {pat['null']['p95']:.4f}], max {pat['null']['max']:.4f}, "
          f"{pat['reps_at_or_above_observed']}/{pat['null']['reps']} at or above the observed value")  # fmt: skip
    kt = pat["excess_kurtosis_post_onset"]
    kr = kt["gaussian_same_pattern_reps"]
    print(f"excess kurtosis after the onset: observed {kt['observed']:.4f}; Gaussian with E2K's pattern median "
          f"{kr['median']:.4f} [{kr['p05']:.4f}, {kr['p95']:.4f}], min {kr['min']:.4f} max {kr['max']:.4f}; "
          f"{kt['reps_at_or_above_observed']}/{kr['reps']} at or above the observed value")  # fmt: skip
    for p_, d_ in pat["median_s2_rise_51_to_101_post_onset_by_prn"].items():
        g = d_["gaussian_same_pattern"]
        print(f"PRN {p_:>2s}: median s^2 rise 51 -> 101 after the onset {100 * d_['observed']:+.2f} %, Gaussian "
              f"{100 * g['median']:+.2f} % [{100 * g['p05']:+.2f}, {100 * g['p95']:+.2f}], positive in "
              f"{100 * g['share_positive']:.0f} %")  # fmt: skip
    for d, r in obs["designs"].items():
        s = r["split"]
        print(f"observed {d:15s} {r['median_rel_change']:.4f} on {s['n_all']} rows; before the onset {s['pre_onset']:.4f} "
              f"({s['n_pre']}), from onset + {SETTLE_S:g} s {s['post_onset']:.4f} ({s['n_post']})")  # fmt: skip
    save(
        "s30_k_rule_null",
        {
            "question": "the K rule's statistic, median over epochs of |sigma^2_101 / sigma^2_51 - 1| (docs/5 section 4.4), "
            "under a stationary Gaussian ensemble with E2K's epoch count, against E2K's observed value",
            "design": {
                "n_rows": N_ROWS,
                "k_large": K_LARGE,
                "k_small": K_SMALL,
                "bar": BAR,
                "reps": REPS,
                "seed": SEED,
                "rho": [0.0, RHO],
                "seed_rho": [SEED, SEED + 1],
                "pattern_reps": PATTERN_REPS,
                "pattern_seed": PATTERN_SEED,
                "estimator": "numpy var ddof=1 per row over the realizations; median "
                "over common rows of |var_large / var_small - 1| (gnssaudit.truth.oracle)",
                "designs": {
                    "nested": "101 against their first 51 (K's rule)",
                    "indep51_vs_101": "an independent 51 against the 101 (k-noise against k-cond)",
                    "indep51_vs_51": "two independent 51s (the thermal comparison's design)",
                },
            },
            "check": check,
            "null": null,
            "e2k": {
                **prov,
                "note": "gate-failed run (G-warmup-lock), so no manifest.json; pinned by started.json's "
                "describe and summary.json's SHA-256; unit files checked against each unit's done.json",
                **obs,
            },
            "run_time_s": round(time.time() - t0, 1),
        },
    )


if __name__ == "__main__":
    main()
