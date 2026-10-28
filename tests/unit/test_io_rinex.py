"""U-io-rinex: RINEX 3 observation and navigation reading on hand-built golden files (docs/11 section 4)."""

import json
from pathlib import Path

import numpy as np
import pytest

from gnssaudit.io.hatanaka import crx_to_rnx, decompress_z
from gnssaudit.io.rinex import gps_seconds, read_rinex2_obs, read_rinex_nav, read_rinex_obs
from gnssaudit.io.schemas import OBS_EPOCH_SAT, validate
from gnssaudit.paths import data_root


def hdr(text: str, label: str) -> str:
    return f"{text:<60}{label:<20}"


def obs(value: float | None, lli: int | None = None) -> str:
    if value is None:
        return " " * 16
    return f"{value:14.3f}{'' if lli is None else lli:1}{'':1}"


OBS_HEADER = [
    hdr("     3.04           OBSERVATION DATA    M", "RINEX VERSION / TYPE"),
    hdr("UGE1", "MARKER NAME"),
    hdr("G    4 C1C L1C D1C S1C", "SYS / # / OBS TYPES"),
    hdr("E    8 C1C L1C D1C S1C C5Q L5Q D5Q S5Q", "SYS / # / OBS TYPES"),
    hdr("  2024     9    11    15    13   30.0000000     GPS", "TIME OF FIRST OBS"),
    hdr("", "END OF HEADER"),
]


def write_obs(tmp_path: Path) -> Path:
    body = [
        "> 2024 09 11 15 13 30.0000000  0  2",
        "G05" + obs(21000000.123) + obs(110000000.456, 0) + obs(-1234.567) + obs(45.0),
        "E11" + obs(23000000.0) + obs(120000000.0, 1) + obs(None) + obs(40.0) + obs(23000001.0) + obs(None)
        + obs(None) + obs(35.0),
        "> 2024 09 11 15 13 31.0000000  3  1",
        hdr("A HEADER LINE INSIDE AN EVENT", "COMMENT"),
        "> 2024 09 11 15 13 31.0000000  0  1",
        "G05" + obs(21000100.0) + obs(110000525.0, 0),
    ]  # fmt: skip
    path = tmp_path / "obs.24o"
    path.write_text("\n".join(OBS_HEADER + body) + "\n")
    return path


def test_obs_reader(tmp_path: Path) -> None:
    table = read_rinex_obs(write_obs(tmp_path), source_id="uge")
    assert validate(table, OBS_EPOCH_SAT).num_rows == 4
    assert table.schema.metadata[b"skipped_epochs"] == b"1"
    rows = table.to_pylist()
    t0 = gps_seconds(2024, 9, 11, 15, 13, 30.0)
    assert [(r["t_gps_s"] - t0, r["sys"], r["prn"], r["signal"]) for r in rows] == [
        (0.0, "G", 5, "1C"),
        (0.0, "E", 11, "1C"),
        (0.0, "E", 11, "5Q"),
        (1.0, "G", 5, "1C"),
    ]
    assert rows[0]["pr_m"] == 21000000.123 and rows[0]["cp_cyc"] == 110000000.456
    assert rows[0]["dop_hz"] == -1234.567 and rows[0]["cn0_dbhz"] == 45.0
    assert rows[1]["lli"] == 1 and rows[1]["dop_hz"] is None
    assert rows[2]["cp_cyc"] is None and rows[2]["valid"] is False and rows[2]["cn0_dbhz"] == 35.0
    assert rows[0]["receiver_id"] == "UGE1"


def test_gps_seconds() -> None:
    # 2024-09-11 is in GPS week 2331 (Wednesday, day 3 of the week)
    assert gps_seconds(2024, 9, 11, 0, 0, 0.0) == 2331 * 604800 + 3 * 86400


def test_rejects_rinex_4_and_other_time_systems(tmp_path: Path) -> None:
    new = tmp_path / "new.24o"
    new.write_text("\n".join([hdr("     4.00           OBSERVATION DATA    M", "RINEX VERSION / TYPE")]) + "\n")
    with pytest.raises(ValueError, match="RINEX 3"):
        read_rinex_obs(new)
    glo = tmp_path / "glo.24o"
    glo.write_text("\n".join([*OBS_HEADER[:4], hdr("  2024     9    11    15    13   30.0000000     GLO",
                                                         "TIME OF FIRST OBS"), OBS_HEADER[5]]) + "\n")  # fmt: skip
    with pytest.raises(ValueError, match="time system"):
        read_rinex_obs(glo)


def d19(v: float) -> str:
    return f"{v:19.12E}".replace("E", "D")


def nav_record(sys_prn: str, epoch: str, values: list[float]) -> list[str]:
    lines = [f"{sys_prn} {epoch}" + "".join(d19(v) for v in values[:3])]
    rest = values[3:]
    for k in range(0, len(rest), 4):
        lines.append("    " + "".join(d19(v) for v in rest[k : k + 4]))
    return lines


def test_nav_reader(tmp_path: Path) -> None:
    gps = [
        1e-4,
        2e-12,
        0.0,  # af0 af1 af2
        12.0,
        50.5,
        4.5e-9,
        1.1,  # IODE Crs dn M0
        1e-6,
        0.01,
        2e-6,
        5153.7,  # Cuc e Cus sqrtA
        345600.0,
        1e-7,
        -2.0,
        2e-7,  # Toe Cic OMEGA0 Cis
        0.96,
        200.0,
        0.5,
        -8e-9,  # i0 Crc omega OMEGADOT
        1e-10,
        1.0,
        2331.0,
        0.0,  # IDOT codes week L2P
        2.0,
        0.0,
        -1e-8,
        12.0,  # accuracy health TGD IODC
        345000.0,
        4.0,
    ]  # tx time, fit interval
    gal = list(gps)
    gal[3 + 17] = 517.0  # data sources: I/NAV E1-B
    glo = [1e-5, 0.0, 0.0] + [0.0] * 12
    lines = [
        hdr("     3.04           N: GNSS NAV DATA    M", "RINEX VERSION / TYPE"),
        hdr("", "END OF HEADER"),
        *nav_record("G05", "2024 09 11 00 00 00", gps),
        *nav_record("R03", "2024 09 11 00 15 00", glo),
        *nav_record("E11", "2024 09 11 00 00 00", gal),
    ]  # fmt: skip
    path = tmp_path / "brdc.24p"
    path.write_text("\n".join(lines) + "\n")
    eph, skipped = read_rinex_nav(path)
    assert skipped == {"R": 1}
    assert eph.satellites() == [("E", 11), ("G", 5)]
    g = eph.select("G", 5, 2331 * 604800.0 + 345600.0)
    assert (g.af0, g.af1, g.crs, g.delta_n, g.m0) == (1e-4, 2e-12, 50.5, 4.5e-9, 1.1)
    assert (g.e, g.sqrt_a, g.toe_sow, g.omega0, g.i0, g.omega, g.omega_dot, g.idot) == (
        0.01, 5153.7, 345600.0, -2.0, 0.96, 0.5, -8e-9, 1e-10,
    )  # fmt: skip
    assert (g.cuc, g.cus, g.cic, g.cis, g.crc, g.week, g.health) == (1e-6, 2e-6, 1e-7, 2e-7, 200.0, 2331, 0)
    assert g.toc_s == gps_seconds(2024, 9, 11, 0, 0, 0.0)
    assert eph.select("E", 11, g.toe_s).data_sources == 517
    assert np.isfinite(g.toe_s)


# ---- RINEX 2.11 ----


def o2(*values: float | tuple[float, int, int] | None) -> str:
    """Observation records of RINEX 2: five 16-column fields (F14.3, LLI, SS) per 80-column line."""
    fields = []
    for v in values:
        if v is None:
            fields.append(" " * 16)
        elif isinstance(v, tuple):
            fields.append(f"{v[0]:14.3f}{v[1] or '':1}{v[2] or '':1}")
        else:
            fields.append(f"{v:14.3f}  ")
    return "\n".join("".join(fields[k : k + 5]).rstrip() for k in range(0, len(fields), 5))


def epoch2(yy: int, sec: float, flag: int, sats: list[str], clock: float | None = None) -> str:
    head = f" {yy:02d}  9 11 15 13{sec:11.7f}  {flag}{len(sats):3d}"
    lines = []
    for k in range(0, len(sats), 12):
        prefix = head if k == 0 else " " * 32
        lines.append(prefix + "".join(sats[k : k + 12]))
    if clock is not None:
        lines[0] = f"{lines[0]:<68}{clock:12.9f}"
    return "\n".join(lines)


def header2(types: list[str], time_system: str = "GPS") -> list[str]:
    lines = [
        hdr("     2.11           OBSERVATION DATA    M (MIXED)", "RINEX VERSION / TYPE"),
        hdr("AGDE", "MARKER NAME"),
    ]
    for k in range(0, len(types), 9):
        count = f"{len(types):6d}" if k == 0 else " " * 6
        lines.append(hdr(count + "".join(f"    {t:>2}" for t in types[k : k + 9]), "# / TYPES OF OBSERV"))
    lines.append(hdr(f"  2024     9    11    15    13   30.0000000     {time_system}", "TIME OF FIRST OBS"))
    lines.append(hdr("", "END OF HEADER"))
    return lines


def test_rinex2_mixed_epoch_and_code_map() -> None:
    types = ["C1", "P1", "L1", "L2", "P2", "S1", "S2", "C5", "L5", "S5"]  # 10 types: continuation line
    text = "\n".join([
        *header2(types),
        epoch2(24, 30.0, 0, ["G05", "E11", "R03"], clock=0.000123456),
        o2(21000000.123, 21000000.9, (110000000.456, 1, 7), (85000000.5, 0, 5), 21000010.0, 45.0, 38.0,
           None, None, None),
        o2(23000000.0, None, 120000000.0, None, None, 41.0, None, 23000001.0, 90000000.0, 36.0),
        o2(19000000.0, None, 100000000.0, 77000000.0, 19000005.0, 30.0, 28.0, None, None, None),
    ]) + "\n"  # fmt: skip
    table = read_rinex2_obs(text, source_id="agde")
    assert validate(table, OBS_EPOCH_SAT).num_rows > 0
    t0 = gps_seconds(2024, 9, 11, 15, 13, 30.0)
    rows = {(r["sys"], r["prn"], r["signal"]): r for r in table.to_pylist()}
    assert all(r["t_gps_s"] == t0 and r["source_id"] == "agde" and r["receiver_id"] == "AGDE" for r in rows.values())
    # GPS: P1 ignored because C1 exists; L2/P2/S2 are 2W
    assert set(rows) == {("G", 5, "1C"), ("G", 5, "2W"), ("E", 11, "1X"), ("E", 11, "5X"), ("R", 3, "1C"),
                         ("R", 3, "2P")}  # fmt: skip
    g1 = rows[("G", 5, "1C")]
    assert g1["pr_m"] == 21000000.123 and g1["cp_cyc"] == 110000000.456 and g1["lli"] == 1
    assert g1["cn0_dbhz"] == 45.0
    g2 = rows[("G", 5, "2W")]
    assert g2["pr_m"] == 21000010.0 and g2["cp_cyc"] == 85000000.5 and g2["cn0_dbhz"] == 38.0
    assert rows[("E", 11, "5X")]["pr_m"] == 23000001.0 and rows[("E", 11, "5X")]["cn0_dbhz"] == 36.0
    assert rows[("E", 11, "1X")]["pr_m"] == 23000000.0 and rows[("E", 11, "1X")]["valid"] is True
    assert rows[("R", 3, "2P")]["cp_cyc"] == 77000000.0 and rows[("R", 3, "2P")]["pr_m"] == 19000005.0
    used = json.loads(table.schema.metadata[b"rinex2_code_used"])
    assert used["G:1C:C1"] == 1 and used["G:2W:P2"] == 1 and used["R:2P:P2"] == 1 and "G:1C:P1" not in used
    code_map = json.loads(table.schema.metadata[b"rinex2_code_map"])
    assert code_map["G"]["L2"] == "2W" and code_map["E"]["L5"] == "5X" and code_map["E"]["L7"] == "7X"
    assert table.schema.metadata[b"skipped_epochs"] == b"0"


def one_sat(types: list[str], values: list[float | None], sat: str = "G07") -> list[dict]:
    text = "\n".join([*header2(types), epoch2(24, 30.0, 0, [sat]), o2(*values)]) + "\n"
    return read_rinex2_obs(text).to_pylist()


def test_rinex2_one_signal_per_band() -> None:
    # GPS C2 + P2 + L2: one 2W row, P2 preferred as the pseudorange
    (row,) = one_sat(["C2", "P2", "L2"], [22000010.0, 22000020.0, 2.0])
    assert (row["signal"], row["pr_m"], row["cp_cyc"]) == ("2W", 22000020.0, 2.0)
    # GPS C2 + L2 without P2: one 2W row with C2
    (row,) = one_sat(["C2", "L2"], [22000010.0, 2.0])
    assert (row["signal"], row["pr_m"], row["cp_cyc"]) == ("2W", 22000010.0, 2.0)
    # GLONASS P2 + L2: one 2P row with both
    (row,) = one_sat(["P2", "L2"], [22000030.0, 3.0], sat="R03")
    assert (row["signal"], row["pr_m"], row["cp_cyc"]) == ("2P", 22000030.0, 3.0)
    # GPS P1 + L1 without C1: one 1C row; C1 wins when both exist
    (row,) = one_sat(["P1", "L1"], [22000000.0, 1.0])
    assert (row["signal"], row["pr_m"], row["cp_cyc"]) == ("1C", 22000000.0, 1.0)
    (row,) = one_sat(["P1", "C1", "L1"], [22000000.0, 22000001.0, 1.0])
    assert (row["signal"], row["pr_m"]) == ("1C", 22000001.0)


def test_rinex2_blank_system_is_gps_and_two_digit_years(tmp_path: Path) -> None:
    body = [epoch2(99, 0.0, 0, [" 05"]), o2(21000000.0, 1.0), epoch2(24, 30.0, 0, ["G05"]), o2(21000001.0, 2.0)]
    path = tmp_path / "x.99o"
    path.write_text("\n".join([*header2(["C1", "L1"]), *body]) + "\n")
    rows = read_rinex_obs(path).to_pylist()  # dispatches on the version line
    assert [(r["sys"], r["prn"]) for r in rows] == [("G", 5), ("G", 5)]
    assert rows[0]["t_gps_s"] == gps_seconds(1999, 9, 11, 15, 13, 0.0)
    assert rows[1]["t_gps_s"] == gps_seconds(2024, 9, 11, 15, 13, 30.0)
    assert rows[0]["source_id"] == "x"


def test_rinex2_more_than_12_satellites_and_obs_continuation() -> None:
    sats = [f"G{k:02d}" for k in range(1, 15)]
    types = ["C1", "L1", "S1", "P2", "L2", "S2"]  # 6 types: two lines per satellite
    obs_lines = [o2(20000000.0 + k, 1e8 + k, 40.0, 20000100.0 + k, 8e7 + k, 33.0) for k in range(14)]
    text = "\n".join([*header2(types), epoch2(24, 30.0, 0, sats), *obs_lines]) + "\n"
    rows = read_rinex2_obs(text).to_pylist()
    assert len(rows) == 28
    last = [r for r in rows if r["prn"] == 14]
    assert {r["signal"] for r in last} == {"1C", "2W"}
    assert [r["pr_m"] for r in sorted(last, key=lambda r: r["signal"])] == [20000013.0, 20000113.0]


def test_rinex2_event_epochs_skipped_and_power_failure_kept() -> None:
    text = "\n".join([
        *header2(["C1"]),
        epoch2(24, 30.0, 0, ["G05"]), o2(21000000.0),
        epoch2(24, 31.0, 4, ["", ""]),
        hdr("A HEADER LINE INSIDE AN EVENT", "COMMENT"),
        hdr("ANOTHER", "COMMENT"),
        epoch2(24, 32.0, 1, ["G05"]), o2(21000002.0),
    ]) + "\n"  # fmt: skip
    table = read_rinex2_obs(text)
    assert table.num_rows == 2 and table.schema.metadata[b"skipped_epochs"] == b"1"


def test_rinex2_cycle_slip_epoch_flag_6_skipped() -> None:
    sats = [f"G{k:02d}" for k in range(1, 15)]  # 14 satellites: a continuation line
    slip = [epoch2(24, 31.0, 6, sats), *[o2(21000000.0 + k, 1.0) for k in range(14)]]
    text = "\n".join([
        *header2(["C1", "L1"]),
        epoch2(24, 30.0, 0, ["G05"]), o2(21000000.0, 1.0),
        *slip,
        epoch2(24, 32.0, 0, ["G05"]), o2(21000002.0, 3.0),
    ]) + "\n"  # fmt: skip
    table = read_rinex2_obs(text)
    assert table.schema.metadata[b"skipped_epochs"] == b"1"
    assert [r["pr_m"] for r in table.to_pylist()] == [21000000.0, 21000002.0]


def test_rinex2_truncated_epoch_raises_with_context() -> None:
    sats = [f"G{k:02d}" for k in range(1, 15)]
    text = "\n".join([*header2(["C1"]), epoch2(24, 30.0, 0, sats)[:60]]) + "\n"  # continuation line missing
    with pytest.raises(ValueError, match="truncated"):
        read_rinex2_obs(text)


def test_rinex2_rejects_unsupported_time_system() -> None:
    with pytest.raises(ValueError, match="time system"):
        read_rinex2_obs("\n".join(header2(["C1"], time_system="GLO")) + "\n")


@pytest.mark.data
def test_rinex2_rgp_hour() -> None:
    path = data_root() / "raw" / "rgp" / "2024" / "253" / "agde253a.24d.Z"
    if not path.exists():
        pytest.skip("RGP AGDE hour not fetched")
    text = crx_to_rnx(decompress_z(path).decode("latin-1"))
    table = validate(read_rinex2_obs(text, source_id="agde"), OBS_EPOCH_SAT)
    assert table.num_rows > 1000
    assert {"G"} <= set(table.column("sys").to_pylist())
    cn0 = [v for v in table.column("cn0_dbhz").to_pylist() if v is not None]
    assert cn0 and 20 <= np.percentile(cn0, 5) and np.percentile(cn0, 95) <= 60
