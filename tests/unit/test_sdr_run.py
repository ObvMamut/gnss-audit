"""gnssaudit.sdr.run_receiver with Docker replaced at the process boundary (docs/6 section 8.4)."""

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from gnssaudit.sdr import ReceiverError, ReceiverSpec, SourceSpec, docker_argv, run_receiver

PINNED = "carlesfernandez/docker-gnsssdr:v0.0.22@sha256:" + "be" * 32
SPEC = ReceiverSpec(image=PINNED, source_commit="3fcb502a", patch_id=None, template="1C", overrides={})


def source(tmp_path: Path) -> SourceSpec:
    rec = tmp_path / "in" / "twin.bin"
    rec.parent.mkdir()
    rec.write_bytes(b"\0" * 16)
    return SourceSpec(path=rec, item_type="ishort", fs_hz=4.0e6)


def test_docker_argv_pins_the_image_and_mounts_the_input_read_only(tmp_path: Path) -> None:
    src = source(tmp_path)
    argv = docker_argv(SPEC, src, tmp_path / "run", name="gnssaudit-abc")
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--name") + 1] == "gnssaudit-abc"
    assert f"{src.path.parent}:/in:ro" in argv and f"{tmp_path / 'run'}:/data" in argv
    assert argv[-3:] == [PINNED, "gnss-sdr", "--config_file=/data/receiver.conf"]


def fake_docker(writes: dict[str, bytes], code: int = 0, stderr: str = "") -> Any:
    calls: list[list[str]] = []

    def run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        workdir = Path(argv[argv.index("-v") + 1].split(":")[0])
        for name, data in writes.items():
            (workdir / name).write_bytes(data)
        return subprocess.CompletedProcess(argv, code, stdout="Tracking of GPS L1 C/A signal started", stderr=stderr)

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_run_writes_the_configuration_and_collects_dumps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = fake_docker({"trk_0.dat": b"x" * 108, "trk_1.dat": b"y" * 108, "observables.dat": b"z" * 112})
    monkeypatch.setattr(subprocess, "run", fake)
    art = run_receiver(SPEC, "[GNSS-SDR]\nA.b=1\n", source(tmp_path), tmp_path / "run", timeout_s=60.0)
    assert sorted(art.tracking_dumps) == [0, 1] and art.tracking_dumps[1].name == "trk_1.dat"
    assert art.observables_dump == tmp_path / "run" / "observables.dat"
    assert (tmp_path / "run" / "receiver.conf").read_text() == "[GNSS-SDR]\nA.b=1\n"
    assert art.config_sha256 == hashlib.sha256(b"[GNSS-SDR]\nA.b=1\n").hexdigest() and art.exit_code == 0
    assert "Tracking of GPS L1 C/A signal started" in art.log_path.read_text()


def test_a_failing_receiver_raises_with_its_stderr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", fake_docker({}, code=1, stderr="line\nterminate called: bad config"))
    with pytest.raises(ReceiverError, match="bad config"):
        run_receiver(SPEC, "[GNSS-SDR]\n", source(tmp_path), tmp_path / "run", timeout_s=60.0)


def test_a_timeout_kills_the_container_and_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def run(argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout", 0))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ReceiverError, match="timed out"):
        run_receiver(SPEC, "[GNSS-SDR]\n", source(tmp_path), tmp_path / "run", timeout_s=5.0)
    name = calls[0][calls[0].index("--name") + 1]
    assert ["docker", "kill", name] in calls


def test_a_work_directory_holding_files_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review finding (p2a, important 7): GNSS-SDR writes trk_N.dat only for channels that track, so an old dump left
    in a reused directory would be read as this run's. A run needs an empty (or new) directory."""
    calls: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    work = tmp_path / "run"
    work.mkdir()
    (work / "trk_5.dat").write_bytes(b"old" * 36)
    with pytest.raises(FileExistsError, match="trk_5.dat"):
        run_receiver(SPEC, "[GNSS-SDR]\n", source(tmp_path), work, timeout_s=60.0)
    assert calls == []  # refused before the receiver ran
