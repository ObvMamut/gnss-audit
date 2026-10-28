"""`gnssaudit stream-prefix` (P2b task 4): one published file streamed from its start, its prefix written under the
data root, and the registry `files:` block printed for a hand edit (the YAML keeps its comments)."""

import hashlib
import io
from pathlib import Path

import pytest
import yaml

from gnssaudit.cli import main as cli
from gnssaudit.experiments.admit import resolve_prefix
from gnssaudit.io import fairdata
from gnssaudit.registry import PublishedFile

DATA = bytes(range(256)) * 400  # 102 400 bytes


def test_stream_prefix_writes_the_prefix_and_prints_the_registry_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pub = PublishedFile("fgi-x", "cr-1", "/X/a.iq", "a.iq", len(DATA), hashlib.sha256(DATA).hexdigest())
    monkeypatch.setattr(cli, "published_file", lambda ds, name: pub)
    asked = []
    monkeypatch.setattr(fairdata, "authorize", lambda cr, path: asked.append((cr, path)) or "https://dl/x")
    monkeypatch.setattr(fairdata.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(DATA))
    monkeypatch.setenv("GNSSAUDIT_DATA", str(tmp_path))
    argv = ["stream-prefix", "fgi-x", "a.iq", "--seconds", "2", "--bytes-per-second", "20000", "--reserve-gb", "0",
            "--allow-lower-reserve"]  # fmt: skip
    assert cli.main(argv) == 0
    target = tmp_path / "raw" / "fgi-x" / "a.iq"
    assert target.read_bytes() == DATA[:40_000] and asked == [("cr-1", "/X/a.iq")]
    (block,) = yaml.safe_load(capsys.readouterr().out)
    assert block["local_path"] == "raw/fgi-x/a.iq" and block["extracted_ranges"] == [[0, 40_000]]
    assert block["size_bytes"] == 40_000 and block["sha256"] == hashlib.sha256(DATA[:40_000]).hexdigest()
    assert block["published_checksum"] == f"sha256:{pub.sha256}" and block["verified"] is True


def test_a_prefix_that_splits_a_sample_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pub = PublishedFile("fgi-x", "cr-1", "/X/a.iq", "a.iq", len(DATA), hashlib.sha256(DATA).hexdigest())
    monkeypatch.setattr(cli, "published_file", lambda ds, name: pub)
    monkeypatch.setenv("GNSSAUDIT_DATA", str(tmp_path))
    argv = ["stream-prefix", "fgi-x", "a.iq", "--seconds", "1.5", "--bytes-per-second", "3", "--sample-bytes", "2"]
    with pytest.raises(SystemExit, match="whole samples"):
        cli.main(argv)


def test_a_no_verify_prefix_is_not_read_as_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Final review R2, I2: --no-verify leaves the prefix under its .unverified name, so Admit's resolve_prefix (which
    reads verification from the name) reports it unverified, and E0b and E2, which open the registered final name, do
    not find it."""
    pub = PublishedFile("fgi-x", "cr-1", "/X/a.iq", "a.iq", len(DATA), hashlib.sha256(DATA).hexdigest())
    monkeypatch.setattr(cli, "published_file", lambda ds, name: pub)
    monkeypatch.setattr(fairdata, "authorize", lambda cr, path: "https://dl/x")
    monkeypatch.setattr(fairdata.urllib.request, "urlopen", lambda url, timeout: io.BytesIO(DATA))
    monkeypatch.setenv("GNSSAUDIT_DATA", str(tmp_path))
    argv = ["stream-prefix", "fgi-x", "a.iq", "--seconds", "2", "--bytes-per-second", "20000", "--reserve-gb", "0",
            "--allow-lower-reserve", "--no-verify"]  # fmt: skip
    assert cli.main(argv) == 0
    (block,) = yaml.safe_load(capsys.readouterr().out)
    assert block["verified"] is False
    assert resolve_prefix(tmp_path, block["local_path"]) == (tmp_path / "raw" / "fgi-x" / "a.iq.unverified", False)


def test_a_reserve_below_the_hard_constraint_needs_its_own_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Final review R2, minor 6: the 40 GB reserve is a hard constraint; --reserve-gb may raise it, and lowers it only
    with --allow-lower-reserve. Refused before anything is streamed."""
    pub = PublishedFile("fgi-x", "cr-1", "/X/a.iq", "a.iq", len(DATA), hashlib.sha256(DATA).hexdigest())
    monkeypatch.setattr(cli, "published_file", lambda ds, name: pub)

    def authorize(cr: str, path: str) -> str:
        raise AssertionError("streamed despite the refusal")

    monkeypatch.setattr(fairdata, "authorize", authorize)
    monkeypatch.setenv("GNSSAUDIT_DATA", str(tmp_path))
    argv = ["stream-prefix", "fgi-x", "a.iq", "--seconds", "2", "--bytes-per-second", "20000", "--reserve-gb", "39.9"]
    with pytest.raises(SystemExit, match="--allow-lower-reserve"):
        cli.main(argv)
    assert not (tmp_path / "raw").exists()
