"""s36: ruling R3a checked on run 3 itself (referee of 28 Oct 2026, minor 1). Exploratory, post-data; decides nothing.

Run 3 injected mode B on six cores (perf-inject, `workers_b: 6`). This study re-injects one of run 3's deciding mode B
units, `A-grid-swept_cw-55-B10` (a failing clause 3 cell), on ONE core with the registered code from before perf-inject
(a907825, the commit s33 was committed at), through E2's own `_inject_receive` with the receiver stubbed out. It compares
the twin with run 3's `unit.json`: output_sha256 (the unit's own jammed twin), achieved_db_hz and clip_count.

Not in run_all: it needs a worktree of a907825 and about 20 min of one core. Usage, from the repository:

    git worktree add --detach <wt> a907825
    (cd <wt> && uv run python <repo>/review/s36_r3a_onecore_check.py <wt>)
    git worktree remove --force <wt>

Spawn re-imports this file in the injector's workers: hence the guard."""

import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

RUN3 = Path.home() / "all/private/GNSS-results/E2/9650dc00e766"  # read only
REGISTERED = "a907825566bab52db7fe880e7486e03b82c3d7fd"
UNIT = "A-grid-swept_cw-55-B10"
OUT = Path(__file__).resolve().parent / "results" / "s36_r3a_onecore_check.json"


def main(wt: Path) -> int:
    import numpy as np
    import scipy

    from gnssaudit.experiments import e2
    from gnssaudit.experiments.twin import SHM_ROOT
    from gnssaudit.inject import pipeline

    head = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(wt), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()  # fmt: skip
    print(f"HEAD {head}; tracked changes: {dirty or 'none'}\npipeline: {pipeline.__file__}", flush=True)
    assert head == REGISTERED and not dirty and Path(pipeline.__file__).is_relative_to(wt / "src")
    assert not hasattr(e2, "inject_workers"), "this is not the pre-perf-inject code"

    spec = yaml.safe_load((wt / "experiments" / "E2-v3.yaml").read_text())  # a907825's: no workers_b
    manifest = json.loads((RUN3 / "manifest.json").read_text())
    assert "workers_b" not in spec and manifest["spec"] == {**spec, "workers_b": 6}, "specs differ beyond workers_b"
    describe = json.loads((RUN3 / "started.json").read_text())["git_describe"]

    tmp = Path(tempfile.mkdtemp(prefix="s36-"))
    run_dir, work = tmp / "run3-copy", tmp / "work" / "s36"
    shutil.copytree(RUN3 / "units" / "survey-A", run_dir / "units" / "survey-A")
    assert e2.is_done(run_dir / "units" / "survey-A", describe), "the survey copy is not done: it would be re-run"
    work.mkdir(parents=True)
    ctx = e2.Context(spec, run_dir, work, describe, None, {}, {}, {}, {})
    ctx.segments["A"] = e2._segment_survey(ctx, "A")

    def no_receiver(name: str, src: object, *args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(report=None, unlocked=[], tracks={})

    e2.receive = no_receiver  # type: ignore[assignment]  # the injection only: no GNSS-SDR
    plan = {u.name: u for u in e2.plan_grid(spec, "A")}
    rec = json.loads((RUN3 / "units" / UNIT / "unit.json").read_text())
    u = plan[UNIT]
    assert e2.Unit(**rec["unit"]) == u and u.mode == "B", "the planned unit is not run 3's mode B unit"
    try:
        t0 = time.perf_counter()
        tw, src = e2._inject_receive(ctx, u)
        wall = time.perf_counter() - t0
        assert src is not None
        src.path.unlink()
    finally:
        shutil.rmtree(SHM_ROOT / work.name, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
    r = tw.report
    assert r is not None  # added after the committed JSON's run, for pyright; it changes no value
    got = {str(k): v for k, v in r.achieved_db_hz.items()}
    ok = r.output_sha256 == rec["output_sha256"] and got == rec["achieved_db_hz"] and r.clip_count == rec["clip_count"]
    out = {
        "meta": {"study": "s36_r3a_onecore_check", "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                 "scipy": scipy.__version__, "code": head, "label": "exploratory, post-data; decides nothing"},
        "inputs": {"run3": {"run_id": "9650dc00e766", "unit_json": str(RUN3 / "units" / UNIT / "unit.json"),
                            "describe": describe}},
        "unit": UNIT, "workers": 1, "wall_s": round(wall, 1),
        "sha_run3": rec["output_sha256"], "sha_onecore": r.output_sha256,
        "achieved_run3": rec["achieved_db_hz"], "achieved_onecore": got,
        "clip_run3": rec["clip_count"], "clip_onecore": r.clip_count,
        "all_equal": ok,
    }  # fmt: skip
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    print("ALL EQUAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
