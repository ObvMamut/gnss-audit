"""RGP URL builder, coordinate-file reader and partner pairing (arm c, docs/7 section 2.2)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gnssaudit.io.hatanaka import crx_to_rnx, decompress_z
from gnssaudit.io.rgp import coords_url, hourly_url, partners, read_rgf93_coords, reduce_hourly
from gnssaudit.io.rinex import read_rinex2_obs, read_rinex_nav
from gnssaudit.paths import data_root

# Mirrors coorrgf93_2024-253.txt: Latin-1 comment header, a column header, a dashed rule, fixed-width rows.
HEADER = (
    "# \n# Coordonn\xe9es des stations valides \xe0 la date du 09/09/2024.\n# \n"
    "# - statut de stabilit\xe9 : [     %1s]  Le statut\n#\n\n"
    " SITE   DOMES         X            Y            Z      statut de\n"
    "                     (m)          (m)          (m)     stabilit\xe9\n"
    "-----------------------------------------------------------------\n"
)


def _write(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "coorrgf93_2024-253.txt"
    path.write_bytes((HEADER + "".join(r + "\n" for r in rows)).encode("latin-1"))
    return path


ROWS = [
    " 1MEL 10051M004  4048056.321   217703.834  4907701.381     N",
    " AGDE 19976M001  4640976.559   281124.425  4351569.579     A",
    " NEAR 19000M001  4640976.559   286124.425  4351569.579     B",  # 5 km east of AGDE
    " MID_ 19001M001  4640976.559   301124.425  4351569.579     C",  # 20 km
    " FAR_ 19002M001  4640976.559   341124.425  4351569.579     N",  # 60 km
]


def test_hourly_url_pattern() -> None:
    assert hourly_url("AGDE", 2024, 253, 0) == "https://rgpdata.ign.fr/pub/data/2024/253/data_1/agde253a.24d.Z"
    assert hourly_url("agde", 2024, 5, 23).endswith("/2024/005/data_1/agde005x.24d.Z")


@pytest.mark.parametrize(("station", "hour"), [("agd", 0), ("agdee", 0), ("agde", 24), ("agde", -1)])
def test_hourly_url_rejects_bad_arguments(station: str, hour: int) -> None:
    with pytest.raises(ValueError, match=r"station|hour"):
        hourly_url(station, 2024, 253, hour)


def test_coords_url_is_in_the_day_directory() -> None:
    assert coords_url(2024, 253) == "https://rgpdata.ign.fr/pub/data/2024/253/coorrgf93_2024-253.txt"


def test_reader_gives_lowercase_ecef(tmp_path: Path) -> None:
    coords = read_rgf93_coords(_write(tmp_path, ROWS))
    assert set(coords) == {"1mel", "agde", "near", "mid_", "far_"}
    np.testing.assert_array_equal(coords["agde"], [4640976.559, 281124.425, 4351569.579])


def test_reader_rejects_a_malformed_row(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="line"):
        read_rgf93_coords(_write(tmp_path, [*ROWS, " BAD0 19003M001  not numbers"]))


def test_reader_rejects_a_file_without_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no station"):
        read_rgf93_coords(_write(tmp_path, []))


def test_partners_window_and_distances(tmp_path: Path) -> None:
    coords = read_rgf93_coords(_write(tmp_path, ROWS))
    got = partners("agde", coords)
    assert [s for s, _ in got] == ["near", "mid_"]  # 1mel is ~540 km away, far_ is 60 km, itself excluded
    assert got[0][1] == pytest.approx(5000.0, abs=1e-6)
    assert got[1][1] == pytest.approx(20000.0, abs=1e-6)


def test_partners_sorted_by_distance_and_window_is_inclusive(tmp_path: Path) -> None:
    coords = read_rgf93_coords(_write(tmp_path, ROWS))
    got = partners("agde", coords, min_m=5000.0, max_m=20000.0)
    assert [s for s, _ in got] == ["near", "mid_"]
    assert partners("agde", coords, min_m=6e3, max_m=19e3) == []
    assert [s for s, _ in partners("agde", coords, min_m=0.0, max_m=70e3)] == ["near", "mid_", "far_"]


def test_partners_unknown_station(tmp_path: Path) -> None:
    coords = read_rgf93_coords(_write(tmp_path, ROWS))
    with pytest.raises(KeyError):
        partners("zzzz", coords)


@pytest.mark.data
def test_agde_hour_reads_and_reduces() -> None:
    """The real RGP hour agde253a (2024-09-09, registry rgp-2024-agde-253-test-hour) through read_rinex2_obs and the
    GPS 1C reduction e1c_acquire runs (`reduce_hourly`): the row counts and the RINEX 2 code-use metadata."""
    root = data_root() / "raw"
    hour = root / "rgp" / "2024" / "253" / "agde253a.24d.Z"
    coords = root / "rgp-coords" / "coorrgf93_2024-253.txt"
    nav = root / "igs-brdc" / "extracted" / "BRDC00IGS_R_20242530000_01D_MN.rnx"
    for need in (hour, coords, nav):
        if not need.is_file():
            pytest.skip(f"{need} not fetched")
    sha = hashlib.sha256(hour.read_bytes()).hexdigest()
    assert sha == "6e882eecf24ba615bbf2e80a0a4aa346b083ae8546a8a3841e55f9af89f6c513"
    obs = read_rinex2_obs(crx_to_rnx(decompress_z(hour).decode("ascii")), source_id=hour.name)
    assert obs.num_rows == 264_224
    md = obs.schema.metadata
    assert md[b"skipped_epochs"] == b"0"
    used = json.loads(md[b"rinex2_code_used"])
    assert used["G:1C:C1"] == 38_931
    assert "G:1C:P1" not in used  # the C/A code on L1 came from C1 alone
    reduced = reduce_hourly(hour, read_rinex_nav(nav)[0], read_rgf93_coords(coords)["agde"])
    assert reduced.num_rows == used["G:1C:C1"]  # one GPS 1C row per C1 observation, none duplicated or dropped
    assert reduced.schema.names == ["t_gps_s", "prn", "cn0_dbhz", "elev_deg"]
    assert not np.isnan(reduced["elev_deg"].to_numpy()).any()
