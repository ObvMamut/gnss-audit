"""Data and results roots live outside git and are set by GNSSAUDIT_DATA and GNSSAUDIT_RESULTS (docs/6 section 5)."""

from pathlib import Path

import pytest

from gnssaudit.paths import data_root, results_root


def test_env_overrides_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GNSSAUDIT_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("GNSSAUDIT_RESULTS", str(tmp_path / "r"))
    assert data_root() == tmp_path / "d"
    assert results_root() == tmp_path / "r"


def test_defaults_sit_beside_the_repository(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GNSSAUDIT_DATA", raising=False)
    monkeypatch.delenv("GNSSAUDIT_RESULTS", raising=False)
    assert data_root() == Path.home() / "all" / "private" / "GNSS-data"
    assert results_root() == Path.home() / "all" / "private" / "GNSS-results"
