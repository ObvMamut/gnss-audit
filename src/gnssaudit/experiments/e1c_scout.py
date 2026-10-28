"""E1 arm c, scout: the pre-specified detector of jamming events in reference-station C/N0 (docs/10 2.2).

The rule is frozen in experiments/E1c-scout.yaml and read into `ScoutRule`; nothing here is tuned on data.

The reference of a satellite-epoch is the median of its C/N0 at t - k * 86164 s for k = 1, 2, 3, each matched to the
nearest epoch of the same (sys, prn, signal) within +-0.5 s. At least 2 of the 3 lags must be present, otherwise the
satellite-epoch has no reference and is not counted (neither as dropped nor as available). Only GPS 1C (band-1 C/N0, S1)
is used: one sidereal day repeats the GPS ground track, but GLONASS repeats after 8 days and Galileo after 10, so the
1-day lag would leave their multipath pattern in the drop.

A flagged epoch has at least `min_sats` satellites, and at least `min_frac` of the reference-bearing ones above the
elevation mask, each >= `drop_db` below its reference. Flagged epochs closer than `merge_gap_s` merge into one event, an
event lasts at least `min_duration_s` (end - start, both flagged epoch times), and the partner screen is a separate step
that runs the same `flag_epochs` on the partner's table.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import yaml
from numpy.typing import NDArray

from gnssaudit.io.kp import kp_max
from gnssaudit.io.rgp import is_metropolitan, partners

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ScoutRule:
    """The frozen detection rule; values come from the `rule` block of experiments/E1c-scout.yaml."""

    reference_signals: Mapping[str, tuple[str, ...]]  # sys -> RINEX signal codes used
    sidereal_s: float
    lags: tuple[int, ...]
    lag_tolerance_s: float
    min_lags: int
    drop_db: float
    elev_mask_deg: float
    min_sats: int
    min_frac: float
    merge_gap_s: float
    min_duration_s: float
    partner_guard_s: float
    partner_min_coverage: float

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> ScoutRule:
        return cls(
            reference_signals={s: tuple(codes) for s, codes in cfg["reference_signals"].items()},
            sidereal_s=float(cfg["sidereal_s"]),
            lags=tuple(int(k) for k in cfg["lags"]),
            lag_tolerance_s=float(cfg["lag_tolerance_s"]),
            min_lags=int(cfg["min_lags"]),
            drop_db=float(cfg["drop_db"]),
            elev_mask_deg=float(cfg["elev_mask_deg"]),
            min_sats=int(cfg["min_sats"]),
            min_frac=float(cfg["min_frac"]),
            merge_gap_s=float(cfg["merge_gap_s"]),
            min_duration_s=float(cfg["min_duration_s"]),
            partner_guard_s=float(cfg["partner_guard_s"]),
            partner_min_coverage=float(cfg["partner_min_coverage"]),
        )


def load_rule(path: Path) -> ScoutRule:
    return ScoutRule.from_config(yaml.safe_load(path.read_text())["rule"])


@dataclass(frozen=True)
class Flagged:
    """Per-epoch result of `flag_epochs`, one entry per distinct epoch time, sorted.

    evaluated: epochs with at least `min_sats` reference-bearing satellites above the mask, i.e. epochs at which the
    rule could have fired. flagged: the epochs where it did. n_dropped and depth_db are defined at evaluated epochs.
    """

    t_evaluated: FloatArray
    flagged: NDArray[np.bool_]
    n_dropped: IntArray
    depth_db: FloatArray  # median drop over the dropped satellites; NaN where none dropped

    @property
    def t_flagged(self) -> FloatArray:
        return self.t_evaluated[self.flagged]


@dataclass(frozen=True)
class Event:
    start: float  # first flagged epoch, GPS seconds
    end: float  # last flagged epoch
    duration: float  # end - start
    depth_db: float  # median over the event's flagged epochs of the per-epoch median drop of the dropped satellites
    n_sats: int  # largest number of simultaneously dropped satellites at a flagged epoch
    partner_state: str = "not_screened"  # "clear", "vetoed", "unscreened" (partner had no evaluable epoch) or this


def sidereal_reference(t: FloatArray, key: IntArray, cn0: FloatArray, rule: ScoutRule) -> FloatArray:
    """Median C/N0 at t - k * sidereal_s (k in rule.lags) per row; NaN where fewer than rule.min_lags lags exist.

    key identifies the (sys, prn, signal) track. Vectorised: one sorted composite key and a searchsorted per lag."""
    n = t.size
    ref = np.full(n, np.nan)
    if n == 0:
        return ref
    t_rel = t - t.min()
    stride = float(t_rel.max()) + (max(rule.lags) + 1) * rule.sidereal_s + 10.0 * rule.lag_tolerance_s + 10.0
    comp = key.astype(np.float64) * stride + t_rel
    order = np.argsort(comp, kind="stable")
    comp_s, cn0_s = comp[order], cn0[order]
    lagged = np.full((n, len(rule.lags)), np.nan)
    for j, k in enumerate(rule.lags):
        target = comp - k * rule.sidereal_s
        hi = np.clip(np.searchsorted(comp_s, target), 1, n - 1) if n > 1 else np.zeros(n, dtype=np.int64)
        lo = hi - 1 if n > 1 else hi
        best = np.where(np.abs(comp_s[lo] - target) <= np.abs(comp_s[hi] - target), lo, hi)
        ok = np.abs(comp_s[best] - target) <= rule.lag_tolerance_s
        lagged[:, j] = np.where(ok, cn0_s[best], np.nan)
    enough = np.isfinite(lagged).sum(axis=1) >= rule.min_lags
    ref[enough] = np.nanmedian(lagged[enough], axis=1)
    return ref


def _select(table: pa.Table, rule: ScoutRule) -> tuple[FloatArray, IntArray, FloatArray, FloatArray]:
    """Rows of the reference signals with finite C/N0: t, track key, cn0, elevation (NaN allowed:
    the reference needs only C/N0, the mask is applied later)."""
    sys = np.asarray(table["sys"].to_numpy(zero_copy_only=False))
    sig = np.asarray(table["signal"].to_numpy(zero_copy_only=False))
    wanted = np.zeros(table.num_rows, dtype=bool)
    for s, codes in rule.reference_signals.items():
        wanted |= (sys == s) & np.isin(sig, codes)
    t = table["t_gps_s"].to_numpy()[wanted].astype(np.float64)
    cn0 = table["cn0_dbhz"].to_numpy(zero_copy_only=False)[wanted].astype(np.float64)
    elev = table["elev_deg"].to_numpy(zero_copy_only=False)[wanted].astype(np.float64)
    prn = table["prn"].to_numpy()[wanted].astype(np.int64)
    sys_w, sig_w = sys[wanted], sig[wanted]
    ids = {
        (s, c): i
        for i, (s, c) in enumerate(sorted({(s, c) for s, codes in rule.reference_signals.items() for c in codes}))
    }
    sig_id = np.zeros(t.size, dtype=np.int64)
    for (s, c), i in ids.items():
        sig_id[(sys_w == s) & (sig_w == c)] = i
    key = sig_id * 1000 + prn
    keep = np.isfinite(cn0) & np.isfinite(t)
    return t[keep], key[keep], cn0[keep], elev[keep]


def flag_epochs(table: pa.Table, rule: ScoutRule) -> Flagged:
    """Flag the epochs of one station. Needs columns t_gps_s, sys, prn, signal, cn0_dbhz and elev_deg (elevation filled;
    rows with NaN C/N0 are dropped; rows with NaN elevation still supply
    references to later days but are never counted themselves, since NaN >= mask is false).

    Per epoch: n_dropped counts the satellites above the mask whose reference exists and is >= drop_db above their C/N0;
    n_ref counts the satellites above the mask with a reference. The epoch is evaluated when n_ref >= min_sats, flagged
    when n_dropped >= min_sats and n_dropped >= min_frac * n_ref. depth_db is the median of (reference - C/N0) over the
    dropped satellites of the epoch."""
    t, key, cn0, elev = _select(table, rule)
    ref = sidereal_reference(t, key, cn0, rule)
    usable = (elev >= rule.elev_mask_deg) & np.isfinite(ref)
    t, drop = t[usable], (ref - cn0)[usable]
    if t.size == 0:
        z = np.empty(0)
        return Flagged(z, np.empty(0, dtype=bool), np.empty(0, dtype=np.int64), z)
    epochs, inv = np.unique(t, return_inverse=True)
    n_ref = np.bincount(inv, minlength=epochs.size)
    hit = drop >= rule.drop_db
    n_drop = np.bincount(inv[hit], minlength=epochs.size)
    depth = _grouped_median(inv[hit], drop[hit], epochs.size)
    flagged = (n_drop >= rule.min_sats) & (n_drop >= rule.min_frac * n_ref)
    evaluated = n_ref >= rule.min_sats
    return Flagged(epochs[evaluated], flagged[evaluated], n_drop[evaluated], depth[evaluated])


def _grouped_median(group: IntArray, value: FloatArray, n_groups: int) -> FloatArray:
    """Median of value per group index (NaN for empty groups), without a Python loop over groups."""
    out = np.full(n_groups, np.nan)
    if group.size == 0:
        return out
    order = np.lexsort((value, group))
    g, v = group[order], value[order]
    start = np.searchsorted(g, np.arange(n_groups), side="left")
    end = np.searchsorted(g, np.arange(n_groups), side="right")
    has = end > start
    lo = start[has] + (end[has] - start[has] - 1) // 2
    hi = start[has] + (end[has] - start[has]) // 2
    out[has] = 0.5 * (v[lo] + v[hi])
    return out


def find_events(flagged: Flagged, rule: ScoutRule) -> list[Event]:
    """Merge flagged epochs separated by gaps < merge_gap_s and keep spells with end - start >= min_duration_s.

    depth_db is the median, over the event's flagged epochs, of the per-epoch median drop of that epoch's dropped
    satellites; n_sats is the maximum n_dropped over those epochs."""
    t, n, d = flagged.t_flagged, flagged.n_dropped[flagged.flagged], flagged.depth_db[flagged.flagged]
    if t.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(t) >= rule.merge_gap_s) + 1
    events = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, t.size], strict=True):  # loop over events, not over rows
        start, end = float(t[a]), float(t[b - 1])
        if end - start >= rule.min_duration_s:
            events.append(Event(start, end, end - start, float(np.median(d[a:b])), int(n[a:b].max())))
    return events


def screen_partner(events: list[Event], partner: Flagged, rule: ScoutRule) -> list[Event]:
    """Set each event's partner_state from the partner's `flag_epochs` result.

    "vetoed": a partner epoch is flagged in [start - partner_guard_s, end + partner_guard_s] (inclusive); this wins.
    "clear": not vetoed, and the partner is evaluated at >= partner_min_coverage of the event's 1 s stamps, i.e. the
    unique evaluated partner epochs within +-0.5 s of [start, end] number at least that fraction of
    round(end - start) + 1.
    An evaluated epoch has n_ref >= min_sats (at least min_sats GPS 1C satellites above the mask with a sidereal
    reference), the bar a station epoch needs to be flaggable at all.
    "unscreened": not vetoed and coverage below the minimum, so absence of a drop proves nothing.
    Only "clear" events are admitted."""
    out = []
    flagged, seen = partner.t_flagged, partner.t_evaluated
    for ev in events:
        lo, hi = ev.start - rule.partner_guard_s, ev.end + rule.partner_guard_s
        if np.searchsorted(flagged, hi, side="right") > np.searchsorted(flagged, lo, side="left"):
            state = "vetoed"
        else:
            n_seen = np.searchsorted(seen, ev.end + 0.5, side="right") - np.searchsorted(
                seen, ev.start - 0.5, side="left"
            )
            covered = n_seen >= rule.partner_min_coverage * (round(ev.end - ev.start) + 1)
            state = "clear" if covered else "unscreened"
        out.append(replace(ev, partner_state=state))
    return out


def detect_events(table: pa.Table, rule: ScoutRule, partner: pa.Table | None = None) -> list[Event]:
    """Events of one station; with a partner table, each carries its partner state (all states are returned,
    the caller admits "clear")."""
    events = find_events(flag_epochs(table, rule), rule)
    return events if partner is None else screen_partner(events, flag_epochs(partner, rule), rule)


# ---- Screen and month (P1c task 8): station sampling, day windows, G1c totals (D-32, proposed) --------------------


def screen_sample(
    coords: Mapping[str, FloatArray], n: int, seed: int, always: list[str], min_m: float, max_m: float
) -> list[str]:
    """The screen's stations: a seeded draw of n from the metropolitan stations with at least one partner in
    [min_m, max_m], plus each `always` station not drawn. Candidates are sorted by name before the draw, so the
    sample depends only on the coordinate file and the seed. Returned sorted, `always` extras last."""
    coords = dict(coords)
    candidates = sorted(s for s, x in coords.items() if is_metropolitan(x) and partners(s, coords, min_m, max_m))
    if n > len(candidates):
        raise ValueError(f"asked for {n} stations, only {len(candidates)} candidates")
    pick = np.random.default_rng(seed).choice(len(candidates), size=n, replace=False)
    drawn = sorted(candidates[i] for i in pick)
    return drawn + [s for s in always if s not in drawn]


def window(flagged: Flagged, t0: float, t1: float) -> Flagged:
    """The epochs of a `Flagged` with t0 <= t < t1 (one target day of a run that carries its history days)."""
    keep = (flagged.t_evaluated >= t0) & (flagged.t_evaluated < t1)
    return Flagged(flagged.t_evaluated[keep], flagged.flagged[keep], flagged.n_dropped[keep], flagged.depth_db[keep])


def concat_flagged(parts: list[Flagged]) -> Flagged:
    """Day windows joined in time order; they must not overlap, so events can be found across midnight."""
    if not parts:
        z = np.empty(0)
        return Flagged(z, np.empty(0, dtype=bool), np.empty(0, dtype=np.int64), z)
    t = np.concatenate([p.t_evaluated for p in parts])
    if np.any(np.diff(t) <= 0):
        raise ValueError("day windows overlap or are out of order")
    return Flagged(
        t,
        np.concatenate([p.flagged for p in parts]),
        np.concatenate([p.n_dropped for p in parts]),
        np.concatenate([p.depth_db for p in parts]),
    )


def n_flagged_in(flagged: Flagged, start: float, end: float) -> int:
    """Flagged epochs in [start, end] (an event's own epochs; the gaps inside it are not counted)."""
    t = flagged.t_flagged
    return int(np.searchsorted(t, end, side="right") - np.searchsorted(t, start, side="left"))


def depth_class(depth_db: float, edges: list[float]) -> str:
    """Drop-depth class: [e_i, e_i+1) dB, the last one open (">= e_last dB"); below the first edge raises."""
    if depth_db < edges[0]:
        raise ValueError(f"depth {depth_db} dB is below the first class edge {edges[0]}")
    for lo, hi in itertools.pairwise(edges):
        if lo <= depth_db < hi:
            return f"{lo:g}-{hi:g} dB"
    return f">={edges[-1]:g} dB"


def g1c_totals(
    events: list[Event],
    flagged: Flagged,
    kp: list[tuple[float, float]],
    epoch_s: float,
    min_clear_flagged_min: float,
    kp_limit: float,
    edges: list[float],
) -> dict[str, Any]:
    """G1c under D-32 (proposed): events by partner state; among "clear" events, those whose Kp (largest 3 h value
    over the event) exceeds kp_limit are counted apart; the admitted rest are summed in flagged-epoch minutes,
    overall and per drop-depth class. `passed_events` is admitted minutes >= min_clear_flagged_min; the coordinate
    criterion is the caller's."""
    states = {"clear": 0, "vetoed": 0, "unscreened": 0}
    classes = {depth_class(e, edges): {"events": 0, "flagged_min": 0.0} for e in edges}
    rows = []
    admitted_min, kp_excluded, kp_excluded_min = 0.0, 0, 0.0
    for ev in events:
        states[ev.partner_state] = states.get(ev.partner_state, 0) + 1
        minutes = n_flagged_in(flagged, ev.start, ev.end) * epoch_s / 60.0
        k = kp_max(kp, ev.start, ev.end)
        cls = depth_class(ev.depth_db, edges)
        admitted = ev.partner_state == "clear" and k <= kp_limit
        if ev.partner_state == "clear" and not admitted:
            kp_excluded += 1
            kp_excluded_min += minutes
        if admitted:
            admitted_min += minutes
            classes[cls]["events"] += 1
            classes[cls]["flagged_min"] += minutes
        rows.append(
            {"start": ev.start, "end": ev.end, "duration_s": ev.duration, "depth_db": ev.depth_db, "n_sats": ev.n_sats,
             "partner_state": ev.partner_state, "flagged_min": minutes, "kp_max": k, "depth_class": cls,
             "admitted": admitted}
        )  # fmt: skip
    return {
        "events_by_partner_state": states,
        "clear_kp_excluded": {"events": kp_excluded, "flagged_min": kp_excluded_min},
        "admitted": {"events": sum(r["admitted"] for r in rows), "flagged_min": admitted_min},
        "classes": classes,
        "min_clear_flagged_min": min_clear_flagged_min,
        "passed_events": admitted_min >= min_clear_flagged_min,
        "events": rows,
    }
