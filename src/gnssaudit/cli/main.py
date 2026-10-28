"""`gnssaudit run <experiment.yaml> [--confirmatory] [--resume]` (docs/6 section 7, D-07), and `gnssaudit
stream-prefix` (the first seconds of a published Fairdata file, P2b task 4).

A run records its code version and start in `started.json` before the runner starts. `--resume` reuses the run
directory of an interrupted run (one without `manifest.json`) under the same code version; the runner finds its
finished units there (E2, P2b task 11).

Exit codes: 0 done; 2 confirmatory run on a dirty tree (G-dirty-tree); 3 the run directory exists (write-once), or the
run has finished; 4 a resume under another code version, or of a run directory without a readable `started.json`;
5 the spec names another receiver build (`receiver_patch`).

A spec may name the patched receiver build it needs (`receiver_patch`, D-40). The run is refused with exit code 5,
before any work, unless the key equals `gnssaudit.sdr.images.PATCH_ID`; a spec without the key runs as before. The key
is part of the spec, so it also gives the run its own run id.

`stream-prefix` keeps fairdata's 40 GB reserve (a hard constraint, docs/6 section 8.2): `--reserve-gb` may raise it and
lowers it only with `--allow-lower-reserve`. The reserve is checked per stream, so parallel streams' total is the
operator's to check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import yaml

from gnssaudit import experiments
from gnssaudit.io import fairdata
from gnssaudit.paths import data_root, results_root
from gnssaudit.provenance import create_run_dir, git_describe, is_dirty, run_id, write_manifest
from gnssaudit.registry import published_file
from gnssaudit.sdr.images import PATCH_ID

REPO = Path(__file__).resolve().parents[3]


STARTED = "started.json"


def receiver_patch_refusal(spec: dict[str, object]) -> str | None:
    """Why a spec cannot run on this build (None: it can). A spec that names `receiver_patch` runs only on that build:
    the key must equal PATCH_ID exactly."""
    if "receiver_patch" in spec and spec["receiver_patch"] != PATCH_ID:
        return f"the spec needs receiver_patch {spec['receiver_patch']!r}, but this tree runs {PATCH_ID!r}"
    return None


def read_started(run_dir: Path) -> dict[str, str] | None:
    """started.json's code version and start time; None when it is missing or unreadable (a crash between the run
    directory and the file, or a directory from before --resume)."""
    try:
        first = json.loads((run_dir / STARTED).read_text())
    except FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError:
        return None
    if not isinstance(first, dict) or not {"git_describe", "started_utc"} <= first.keys():
        return None
    return {"git_describe": str(first["git_describe"]), "started_utc": str(first["started_utc"])}


def _run(spec_path: Path, confirmatory: bool, repo: Path, resume: bool = False) -> int:
    spec = yaml.safe_load(spec_path.read_text())
    if (refusal := receiver_patch_refusal(spec)) is not None:
        print(refusal, file=sys.stderr)
        return 5
    confirmatory = confirmatory or bool(spec.get("confirmatory", False))
    if confirmatory and is_dirty(repo):
        print("refusing a confirmatory run on a dirty tree (G-dirty-tree)", file=sys.stderr)
        return 2
    runner = experiments.REGISTRY[spec["id"]]
    rid = run_id(spec)
    describe = git_describe(repo)
    run_dir = results_root() / spec["id"] / rid
    if resume and run_dir.is_dir():
        if (run_dir / "manifest.json").exists():
            print(f"{spec['id']}/{rid} has finished; results are write-once", file=sys.stderr)
            return 3
        first = read_started(run_dir)
        if first is None:
            print(f"{spec['id']}/{rid} has no readable {STARTED}: its code version is unknown, no resume",
                  file=sys.stderr)  # fmt: skip
            return 4
        if first["git_describe"] != describe:
            print(f"{spec['id']}/{rid} started under {first['git_describe']}, not {describe}: no resume",
                  file=sys.stderr)  # fmt: skip
            return 4
        started = first["started_utc"]
    else:
        try:
            run_dir = create_run_dir(results_root(), spec["id"], rid)
        except FileExistsError:
            print(f"{spec['id']}/{rid} exists; results are write-once", file=sys.stderr)
            return 3
        started = datetime.now(UTC).isoformat(timespec="seconds")
        (run_dir / STARTED).write_text(json.dumps({"git_describe": describe, "started_utc": started}) + "\n")
    t0 = time.perf_counter()
    outcome = runner(spec, run_dir)
    timings = {"total": round(time.perf_counter() - t0, 3)}
    write_manifest(run_dir, spec, describe, outcome.inputs, timings, outcome.counts, started, outcome.summary)
    print(run_dir)
    return 0


def _stream_prefix(args: argparse.Namespace) -> int:
    """Stream one published file from its start into raw/<dataset>/<name>, keep its first `seconds`, verify the whole
    file (unless --no-verify: the prefix then keeps its `.unverified` name) and print the registry `files:` block."""
    reserve = int(args.reserve_gb * 1e9)
    if reserve < fairdata.RESERVE_BYTES and not args.allow_lower_reserve:
        raise SystemExit(f"a reserve of {args.reserve_gb:g} GB is below the {fairdata.RESERVE_BYTES / 1e9:g} GB hard "
                         f"constraint: refused without --allow-lower-reserve")  # fmt: skip
    pub = published_file(args.dataset, args.file)
    prefix = args.seconds * args.bytes_per_second
    if prefix != int(prefix) or int(prefix) % args.sample_bytes:
        raise SystemExit(f"a prefix of {prefix} bytes does not hold whole samples of {args.sample_bytes} bytes")
    rel = Path("raw") / pub.dataset_id / pub.name
    dest = data_root() / rel

    def open_stream() -> BinaryIO:  # a fresh single-use URL for every (re)start
        url = fairdata.authorize(pub.metax_id, pub.record_path)
        return urllib.request.urlopen(url, timeout=300)

    r = fairdata.stream_prefix(open_stream, dest, prefix_bytes=int(prefix), size_bytes=pub.size_bytes,
                               sha256=pub.sha256, verify_full=not args.no_verify,
                               reserve=reserve)  # fmt: skip
    block = {
        "name": pub.name,
        "size_bytes": r.prefix_bytes,
        "published_checksum": f"sha256:{pub.sha256}",
        "sha256": r.prefix_sha256,
        "local_path": str(rel),
        "extracted_ranges": [[0, r.prefix_bytes]],
        "verified": r.verified,
    }
    print(yaml.safe_dump([block], sort_keys=False), end="")
    return 0


def main(argv: list[str] | None = None, repo: Path = REPO) -> int:
    parser = argparse.ArgumentParser(prog="gnssaudit")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one experiment manifest")
    run.add_argument("spec", type=Path)
    run.add_argument("--confirmatory", action="store_true", help="refuse to run on a dirty tree")
    run.add_argument("--resume", action="store_true", help="continue an interrupted run in its own directory")
    sp = sub.add_parser("stream-prefix", help="keep the first seconds of a published Fairdata file")
    sp.add_argument("dataset")
    sp.add_argument("file")
    sp.add_argument("--seconds", type=float, required=True)
    sp.add_argument("--bytes-per-second", type=float, required=True)
    sp.add_argument("--sample-bytes", type=int, default=2, help="bytes per sample (2 for interleaved int8 I/Q)")
    sp.add_argument(
        "--no-verify",
        action="store_true",
        help="stop at the prefix; the published checksum stays unchecked and the prefix keeps its .unverified name",
    )
    sp.add_argument("--reserve-gb", type=float, default=fairdata.RESERVE_BYTES / 1e9,
                    help="free space kept on the data disk, per stream (at least the default)")  # fmt: skip
    sp.add_argument("--allow-lower-reserve", action="store_true", help="accept a --reserve-gb below the default")
    args = parser.parse_args(argv)
    if args.command == "stream-prefix":
        return _stream_prefix(args)
    return _run(args.spec, args.confirmatory, repo, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
