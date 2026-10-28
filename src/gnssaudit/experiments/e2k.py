"""E2K: the K-study at the Route B ceiling, given J_max_TA (P2b part 2 addendum of 19 Oct 2026, section "Route B
ceiling, K-study, G3"; task 13; docs/5 section 4.4).

E2 run 2 passed every gate but gave no J_max under the pre-registered rule (a'), so no ceiling and no K-study. The
owner's addendum measures J_max against a simulated true analog front end (J_max_TA, review s29) and puts it in J_max's
place in the pre-registered Route B ceiling; everything else stays as pre-registered:
- J_max_TA(f) from the committed s29 result, read by path and refused unless it has the spec's SHA-256: per segment,
  and combined over the segments by combine_segments, which must give s29's own combined value;
- L_B(s, f): the largest grid level at or below the combined J_max_TA(f) (the addendum's "largest l <= J_max_TA(f)")
  such that at every level up to it, in mode A and for both ghosts, B1's bar r - 1/(C/N0 T) < residual_max holds and
  at most lost_truth_max_s of ghost truth is lost. These are E2's own checks (e2.route_b_checks) on the per-level
  records run 2's summary keeps (residual, lost_ghost_s), which must give the residual_pass and truth_kept E2 decided;
  L_B(f) is the minimum over the segments, None if any is None;
- the oracle units as E2 plans them (e2.plan_oracle: L_K = min(80, L_B), the K = 101 and thermal blocks on noise when
  swept CW has no ceiling, seeds keyed on sorted names as in commit eaee4df), and none at all when L_B(noise) is None:
  then no K-study runs and G3 fails on K, a result rather than an error;
- E2's unit runner (done files, hash validation, resume, /dev/shm and the D-34 receiver as in E2) into this run's
  directory, with a seeded share of the units again for G-repeat. The K-study units are differenced against run 2's
  mode A clean twin, read in place: they do not project B1, so its tracks and observables are all they need. E2's
  null gate on that twin (against run 2's second mode A clean twin) and its identity gate on every pair apply;
- K by e2.choose_k (0.2, as pre-registered) and the thermal comparison, as E2 reports them (e2.k_study_decision).

Run 2 is never written: its surveys are read from copies in this run's source/ directory (e2post.open_source), and
every file of it read is hashed into the manifest. E2K's run context is run 2's spec (its manifest's: segments,
families, seed, receiver overrides), whose kstudy, thermal and rules E2K's spec must repeat unchanged. The receiver is
the current image (p2; C/A unchanged against run 2's p1, docs/9 and D-40), recorded beside run 2's. g3_clauses states
G3's clauses as far as E2 run 2 and E2K's data decide them.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gnssaudit.experiments import Outcome, e2, e2post
from gnssaudit.experiments.e2_frontend import combine_segments, route_b_ceiling
from gnssaudit.experiments.twin import Twin, jsonable, receiver_spec
from gnssaudit.gates import GateFailure, GateReport
from gnssaudit.paths import results_root
from gnssaudit.provenance import git_describe, sha256_file

PRE_REGISTERED = ("kstudy", "thermal", "rules")  # E2K's spec repeats run 2's, unchanged
STUDY_FAMILY = "noise"  # without its ceiling no K-study runs (the addendum: Route B for band-limited noise)
NO_K_STUDY = "no K-study; G3 fails on K"
GATES_TEXT = (
    "Gates on every E0 and E2 twin pair: identity, null, repeat, phase precision. E2: run 2's and E2K's "
    "own. E0: reported in the G3 report."
)
OWNER_EXCLUDED = ("swept_cw", "chirp")  # the owner's decision of 19 Oct 2026 (the addendum), with EXCLUSION's text
EXCLUSION = (
    "The pre-registered B1 bar stands. A family without a ceiling is excluded from Route B and reported: swept "
    "CW and the chirp (risk R12). No correction of the B1 floor and no segment-only ceiling."
)


@dataclass(frozen=True)
class Plan:
    """What E2K decides before any twin runs: E2 run 2 opened read-only, J_max_TA from s29, L_B per segment and
    combined, and the oracle units."""

    src: Path
    spec: dict[str, Any]  # run 2's, as its manifest keeps it
    describe: str  # run 2's code version
    summary: dict[str, Any]  # run 2's summary.json
    segments: dict[str, e2.Segment]
    front: dict[str, dict[str, Any]]  # per segment: j_max_ta_db_hz, residual_pass, truth_kept, ceiling_db_hz
    j_max_ta: dict[str, float | None]  # combined over the segments
    decides: dict[str, bool]  # s29's: noise and swept CW decide, the chirp is reported
    ceiling: dict[str, float | None]  # L_B, combined over the segments
    study: dict[str, float | None]  # the ceilings the K-study is planned at: L_B, or none without L_B(noise)
    units: list[e2.Unit]
    s29: Path
    inputs: list[Path]


def _level(v: Any) -> float | None:
    return None if v is None else float(v)


def read_s29(
    spec: Mapping[str, Any], segments: Sequence[str], families: Sequence[str]
) -> tuple[dict[str, dict[str, float | None]], dict[str, float | None], dict[str, bool], Path]:
    """(J_max_TA per segment and family, combined per family, whether each family decides, the file) from the s29 result
    the spec names, refused unless it has the spec's SHA-256. The combination is combine_segments' (None if any segment
    is None, else the minimum) and must be s29's own combined value."""
    path = e2.REPO / str(spec["s29_result"])
    digest = sha256_file(path)
    if digest != str(spec["s29_sha256"]):
        raise ValueError(f"{path} has SHA-256 {digest}, not the spec's s29_sha256 {spec['s29_sha256']}: refused")
    fams = json.loads(path.read_text())["decisions"]["families"]
    per = {key: {f: _level(fams[f]["j_max_ta_db_hz"][key]) for f in families} for key in segments}
    both = {f: combine_segments([per[key][f] for key in segments]) for f in families}
    for f in families:
        if both[f] != _level(fams[f]["combined_db_hz"]):
            raise ValueError(
                f"s29, {f}: the segments' J_max_TA combine to {both[f]}, s29 says {fams[f]['combined_db_hz']}"
            )
    return per, both, {f: bool(fams[f]["decides"]) for f in families}, path


def _decided(levels: Mapping[str, bool] | None) -> dict[float, bool] | None:
    return None if levels is None else {float(lv): bool(v) for lv, v in levels.items()}


def route_b_segment(
    spec: Mapping[str, Any], seg: e2.Segment, run_seg: Mapping[str, Any], jmax: Mapping[str, float | None]
) -> dict[str, Any]:
    """Segment s's residual_pass, truth_kept and L_B(s, f) per family, at most jmax[f], from its part of run 2's
    summary: each projected mode A grid unit's record as E2 kept it (residual by level, lost_ghost_s by unit; a unit
    without a record is not in lost_ghost_s, a level without a residual fails B1, as in E2) through e2.route_b_checks.
    Refused where that does not give the residual_pass or truth_kept E2 decided."""
    levels = sorted(float(v) for v in spec["levels_db_hz"])
    grid = {(u.family, u.level): u for u in e2.plan_grid(spec, seg.key) if u.block == "grid" and u.mode == "A"}
    prns = [int(seg.channels[ch][2]) for ch in seg.ghost_ch]
    res, lost = run_seg["residual"], run_seg["lost_ghost_s"]
    out: dict[str, dict[str, Any]] = {"residual_pass": {}, "truth_kept": {}, "ceiling_db_hz": {}}
    for fam in spec["families"]:
        units = {lv: grid[fam, lv] for lv in levels}
        recs = {lv: {"residual": res.get(fam, {}).get(str(lv)), "lost_ghost_s": lost[u.name]}
                for lv, u in units.items() if u.project and u.name in lost}  # fmt: skip
        b1, kept = e2.route_b_checks(recs, prns, seg.ghost_ch, spec["rules"])
        for field, mine in (("residual_pass", b1), ("truth_kept", kept)):
            if (mine or None) != _decided(run_seg[field][fam]):
                raise ValueError(f"segment {seg.key}, {fam}: {field} from run 2's records is {mine}, run 2 decided "
                                 f"{run_seg[field][fam]}")  # fmt: skip
            out[field][fam] = mine or None
        out["ceiling_db_hz"][fam] = route_b_ceiling(levels, jmax[fam], b1, kept)
    return out


def decide(spec: Mapping[str, Any], run_dir: Path) -> Plan:
    """Open E2 run 2 read-only (its surveys copied into run_dir/source, made afresh at every start), read J_max_TA,
    and decide L_B and the oracle units. Nothing is written outside run_dir/source."""
    src = results_root() / str(spec["source_run"])
    base = run_dir / "source"
    shutil.rmtree(base, ignore_errors=True)  # an earlier start's copies
    ctx, inputs = e2post.open_source(src, base, str(spec["kstudy"]["segment"]))
    for key in ctx.spec["segments"]:
        if key not in ctx.segments:
            other, read = e2post.open_source(src, base, key)
            ctx.segments[key] = other.segments[key]
            inputs += read
    for k in PRE_REGISTERED:
        if spec[k] != ctx.spec[k]:
            raise ValueError(
                f"E2K's {k} {spec[k]} is not run 2's {ctx.spec[k]}: the K-study's rules are pre-registered"
            )
    summary = json.loads((src / "summary.json").read_text())
    if summary != json.loads((src / "manifest.json").read_text())["summary"]:  # E2 keeps the summary there, not a hash
        raise ValueError(f"{src}/summary.json is not the summary its manifest.json records: refused")
    inputs.append(src / "summary.json")
    keys, fams = list(ctx.spec["segments"]), list(ctx.spec["families"])
    per, jmax, decides, s29 = read_s29(spec, keys, fams)
    front = {key: {"j_max_ta_db_hz": per[key], **route_b_segment(ctx.spec, ctx.segments[key],
                                                                 summary["segments"][key], jmax)}
             for key in keys}  # fmt: skip
    ceiling = e2.combined(front, "ceiling_db_hz", fams)
    study = dict(ceiling) if ceiling[STUDY_FAMILY] is not None else dict.fromkeys(ceiling)
    units = [u for key in keys for u in e2.plan_oracle(ctx.spec, key, study)]
    return Plan(src, ctx.spec, ctx.describe, summary, dict(ctx.segments), front, jmax, decides, ceiling, study, units,
                s29, inputs)  # fmt: skip


def source_twin(plan: Plan, name: str) -> tuple[Twin, dict[str, Any], list[Path]]:
    """Run 2's unit `name` (a clean twin), read in place, its lock record (lock_ok, unlocked) and its files: refused
    unless it is finished, every file still hashes as its done.json records, and it met its lock policy (E2K's own
    G-warmup-lock reads only E2K's run directory; run 2's gate held these twins)."""
    udir = plan.src / "units" / name
    if not e2.is_done(udir, plan.describe):
        raise ValueError(f"{name}: not a finished unit of {plan.src} (no done.json, or files that no longer hash as "
                         f"recorded)")  # fmt: skip
    rec = json.loads((udir / "unit.json").read_text())
    if rec.get("lock_ok") is not True:
        raise ValueError(f"{name}: run 2's unit records lock_ok {rec.get('lock_ok')!r}, not true (its lock policy was "
                         f"not met): refused")  # fmt: skip
    files = json.loads((udir / "done.json").read_text())["files"]
    lock = {"lock_ok": True, "unlocked": list(rec["unlocked"])}
    return e2._load_twin(udir, name), lock, [udir / f for f in sorted(files)]


def g3_clauses(plan: Plan, gates: Mapping[str, Mapping[str, Any]], d: Mapping[str, Any]) -> dict[str, Any]:
    """G3's clauses as far as E2 run 2 and E2K decide them (the addendum): the gates of run 2 and of E2K's own twins
    (`passed` None when E2K ran none; E0's are reported in the G3 report), J_max as pre-registered (run 2's) against
    J_max_TA, L_B, K, and the owner's exclusion of swept CW and the chirp, each with its L_B as computed. Noise is never
    listed there: without its ceiling no K-study runs and G3 fails on K (the K clause)."""

    def part(g: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        failed = sorted(n for n, v in g.items() if not v["passed"])
        return {"passed": not failed if g else None, "gates": len(g), "failed": failed}

    pre, k = plan.summary["decisions"]["j_max_db_hz"], d["k"]
    text = NO_K_STUDY if not plan.units else (f"K = {k}" if k is not None else "K not chosen; G3 fails on K")
    return {
        "gates": {"text": GATES_TEXT, "e2_run2": part(plan.summary["gates"]), "e2k": part(gates),
                  "e0": "reported in the G3 report"},
        "j_max": {f: {"pre_registered_db_hz": pre[f], "j_max_ta_db_hz": plan.j_max_ta[f], "decides": plan.decides[f]}
                  for f in plan.j_max_ta},
        "ceiling_db_hz": dict(plan.ceiling),
        "k": {"k": k, "chosen": k is not None, "text": text},
        "exclusion": {"families": {f: plan.ceiling[f] for f in OWNER_EXCLUDED}, "text": EXCLUSION},
    }  # fmt: skip


def run_e2k(spec: dict[str, Any], run_dir: Path) -> Outcome:
    started = run_dir / "started.json"
    describe = json.loads(started.read_text())["git_describe"] if started.exists() else git_describe(e2.REPO)
    plan = decide(spec, run_dir)
    work = results_root() / "E2K-work" / run_dir.name
    work.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(e2.SHM_ROOT / work.name, ignore_errors=True)  # twins a killed start left (a SIGKILL skips `finally`)
    rx = receiver_spec(plan.spec, template=str(plan.spec.get("template", "1C")))
    ran = {key: plan.segments[key] for key in sorted({u.segment for u in plan.units})}
    ctx = e2.Context(plan.spec, run_dir, work, describe, rx, ran, {}, {}, {})
    inputs = [*plan.inputs, plan.s29]
    gates: list[tuple[str, GateReport]] = []
    records: dict[str, dict[str, Any]] = {}
    from_source: dict[str, dict[str, Any]] = {}  # the source run's clean twins read in place, with their lock record
    try:
        if plan.units:
            for key in ran:  # the clean twin the K-study units pair with, and its null partner, as run 2 ran them
                for u in e2.plan_grid(plan.spec, key):
                    if u.block == "clean" and u.mode == "A":
                        ctx.cleans[u.name], from_source[u.name], read = source_twin(plan, u.name)
                        inputs += read
            e2._parallel(ctx, e2.run_clean, [u for u in plan.units if u.block == "thermal-clean"])
            e2._parallel(ctx, e2.run_jammed, [u for u in plan.units if u.block != "thermal-clean"])
            everything = {u.name: u for u in plan.units}
            reps = e2.pick_repeats(list(everything), fraction=float(plan.spec["repeat_fraction"]),
                                   seed=int(plan.spec["seed"]))  # fmt: skip
            e2._parallel(ctx, e2.run_repeat, [everything[n] for n in reps])
            for key in ran:
                records[key], seg_gates = e2.twin_gates(ctx, key)
                gates += seg_gates
            gates.append(("[repeat]", e2.repeat_report(ctx)))
        study = e2.k_study_decision(ctx, plan.study)
    finally:
        shutil.rmtree(e2.SHM_ROOT / work.name, ignore_errors=True)
    cap = float(plan.spec["kstudy"]["level_cap_db_hz"])
    d = {"j_max_ta_db_hz": plan.j_max_ta, "ceiling_db_hz": plan.ceiling,
         "k_level_db_hz": {f: e2.k_level(cap, v) for f, v in plan.ceiling.items()}, **study}  # fmt: skip
    if not plan.units:
        d["k_study"]["note"] = NO_K_STUDY
    gate_rec = {g.gate + sfx: {"passed": g.passed, **jsonable(g.details)} for sfx, g in gates}
    summary = {
        "source_run": str(spec["source_run"]),
        "source_describe": plan.describe,
        "s29_result": str(spec["s29_result"]),
        "s29_sha256": str(spec["s29_sha256"]),
        "source_clean_twins": from_source,
        "segments": {key: {**plan.front[key], **records.get(key, {})} for key in plan.front},
        "decisions": d,
        "g3_clauses": g3_clauses(plan, gate_rec, d),
        "units": len(list((run_dir / "units").glob("*/done.json"))),
        "receivers": {"overrides": dict(rx.overrides), "image": rx.image, "patch_id": rx.patch_id},
        "source_receivers": plan.summary["receivers"],
        "gates": gate_rec,
    }
    (run_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n")
    shutil.rmtree(work, ignore_errors=True)  # dumps go (D-18); units stay
    failed = [g for _, g in gates if not g.passed]
    if failed and plan.spec.get("abort_on_gate_failure", True):
        raise GateFailure(failed[0])
    inputs += [seg.source.path for seg in ran.values()]
    return Outcome(inputs=list(dict.fromkeys(inputs)),
                   counts={"units": int(summary["units"]), "gates_failed": len(failed)},
                   summary=jsonable(summary))  # fmt: skip
