"""Run GNSS-SDR in its pinned container and collect the artefacts (docs/9 section 1; docs/6 section 8.4).

The run directory is mounted at /data and the recording's directory read-only at /in. A non-zero exit raises with the
receiver's stderr. A timeout kills the named container, since killing the docker client alone would leave it running.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from gnssaudit.sdr.config import ReceiverSpec, SourceSpec

CONFIG_NAME = "receiver.conf"
_TRK = re.compile(r"^trk_(\d+)\.dat$")


class ReceiverError(RuntimeError):
    """GNSS-SDR failed or timed out; the message carries the tail of its stderr."""


@dataclass(frozen=True)
class RunArtifacts:
    workdir: Path
    config_sha256: str
    log_path: Path
    exit_code: int
    tracking_dumps: dict[int, Path]  # channel -> trk_<channel>.dat
    observables_dump: Path | None  # absent when no epoch was ever produced (GNSS-SDR deletes an empty dump)
    wall_s: float


def docker_argv(spec: ReceiverSpec, source: SourceSpec, workdir: Path, *, name: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", name, "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/data/home", "-v", f"{workdir}:/data", "-v", f"{source.path.parent}:/in:ro", "-w", "/data",
        spec.image, "gnss-sdr", f"--config_file=/data/{CONFIG_NAME}",
    ]  # fmt: skip


def _tail(text: str, n: int = 20) -> str:
    return "\n".join(text.splitlines()[-n:])


def run_receiver(
    spec: ReceiverSpec, config_text: str, source: SourceSpec, workdir: Path, timeout_s: float
) -> RunArtifacts:
    """Run one receiver pass over `source` in `workdir` (created; the image's HOME is not writable, so it gets one).
    The directory must be new or empty: GNSS-SDR writes trk_N.dat only for channels that track, so a file left from
    an earlier run would be read as this run's."""
    if workdir.exists() and (left := sorted(p.name for p in workdir.iterdir())):
        raise FileExistsError(f"{workdir} is not empty ({', '.join(left[:5])}); a receiver run needs a fresh directory")
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "home").mkdir(exist_ok=True)
    (workdir / "pvt").mkdir(exist_ok=True)
    data = config_text.encode()
    (workdir / CONFIG_NAME).write_bytes(data)
    name = f"gnssaudit-{uuid.uuid4().hex[:12]}"
    argv = docker_argv(spec, source, workdir, name=name)
    log_path = workdir / "gnss-sdr.log"
    t0 = time.perf_counter()
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "kill", name], capture_output=True, text=True, timeout=60, check=False)
        log_path.write_text(f"{exc.stdout or ''}\n--- stderr\n{exc.stderr or ''}")
        raise ReceiverError(f"GNSS-SDR timed out after {timeout_s} s in {workdir}; container {name} killed") from exc
    wall = time.perf_counter() - t0
    log_path.write_text(res.stdout + "\n--- stderr\n" + res.stderr)
    if res.returncode != 0:
        raise ReceiverError(f"GNSS-SDR exited {res.returncode} in {workdir}:\n{_tail(res.stderr)}")
    dumps = {int(m.group(1)): p for p in sorted(workdir.glob("trk_*.dat")) if (m := _TRK.match(p.name))}
    obs = workdir / "observables.dat"
    return RunArtifacts(
        workdir=workdir,
        config_sha256=hashlib.sha256(data).hexdigest(),
        log_path=log_path,
        exit_code=res.returncode,
        tracking_dumps=dumps,
        observables_dump=obs if obs.exists() else None,
        wall_s=wall,
    )
