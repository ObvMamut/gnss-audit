"""Dataset and interval registries (docs/7 sections 3.1 and 4.1)."""

from pathlib import Path

import pytest

from gnssaudit.registry import PublishedFile, dataset_file, load_intervals, published_file


def test_intervals_by_id(tmp_path: Path) -> None:
    path = tmp_path / "intervals.yaml"
    path.write_text(
        "intervals:\n  - {interval_id: a, t_start: 1.0, t_end: 2.0}\n  - {interval_id: b, t_start: 2.0, t_end: 3.5}\n"
    )
    got = load_intervals(path)
    assert got["b"]["t_end"] == 3.5


def test_duplicate_interval_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "intervals.yaml"
    path.write_text(
        "intervals:\n  - {interval_id: a, t_start: 1.0, t_end: 2.0}\n  - {interval_id: a, t_start: 2.0, t_end: 3.0}\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_intervals(path)


def test_dataset_file_resolves_under_the_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNSSAUDIT_DATA", str(tmp_path / "data"))
    path = tmp_path / "datasets.yaml"
    path.write_text("datasets:\n  - id: d\n    files:\n      - {name: f.gz, local_path: raw/d/f.gz}\n")
    assert dataset_file("d", "f.gz", path) == tmp_path / "data" / "raw" / "d" / "f.gz"
    with pytest.raises(KeyError):
        dataset_file("d", "missing", path)


def test_a_published_file_carries_its_record_and_checksum(tmp_path: Path) -> None:
    reg = tmp_path / "datasets.yaml"
    reg.write_text(
        "datasets:\n"
        "  - id: fgi-x\n"
        "    metax_id: cr-42\n"
        "    published_files:\n"
        '      - {name: a.iq, record_path: "/X/a.iq", size_bytes: 1000, published_checksum: "sha256:ABCD"}\n'
    )
    f = published_file("fgi-x", "a.iq", reg)
    assert f == PublishedFile("fgi-x", "cr-42", "/X/a.iq", "a.iq", 1000, "abcd")
    with pytest.raises(KeyError):
        published_file("fgi-x", "b.iq", reg)


def test_both_fgi_records_name_their_metax_dataset() -> None:
    f = published_file("fgi-jt2023", "JT23_4.1.5_L1_E1.iq")
    assert f.metax_id == "2eee8cfd-db00-4e46-bd62-d7632c09c700" and f.size_bytes == 24_552_000_000
    assert published_file("fgi-spoofrepo", "TGS_L5_E5a.dat").metax_id == "367379a8-7d78-4b08-91f0-8027ce7a621b"
