"""GNSS-SDR v0.0.22 dump readers (docs/9 sections 4.1 and 4.2; docs/7 section 5.2).

Expected bytes are packed with `struct` from the field table of docs/9 section 4.1 (Verified [51]), independently of
the reader's dtypes: 7 float32, uint64 stamp, 12 float32 (field 9 is float64 in the patched build), float64 sample
count, uint32 PRN, uint64 TOW, uint32 week.
"""

import struct
from pathlib import Path

import numpy as np
import pytest

from gnssaudit.io.gnsssdr import DumpError, read_observables_dump, read_tracking_dump

STOCK_FMT = "<7fQ12fdIQI"  # 108 bytes
PATCHED_FMT = "<7fQd11fdIQI"  # 112 bytes, field 9 (accumulated phase) as float64
FS = 4.0e6


def stock_record(stamp: int, acc: float, prn: int = 7, cn0: float = 45.5) -> bytes:
    #                 VE   E    P    L    VL   I     Q
    head = (0.0, 2.0, 3.5, 2.1, 0.0, 1.25, -0.5)
    tail = (300.0, -0.25, 1.023e6, 0.01, 0.002, 0.3, 0.001, 0.0002, cn0, 0.93, 12.5)  # fields 10-20
    return struct.pack(STOCK_FMT, *head, stamp, acc, *tail, float(stamp), prn, 123_456, 2330)


def patched_record(stamp: int, acc: float, prn: int = 7) -> bytes:
    head = (0.0, 2.0, 3.5, 2.1, 0.0, 1.25, -0.5)
    tail = (300.0, -0.25, 1.023e6, 0.01, 0.002, 0.3, 0.001, 0.0002, 45.5, 0.93, 12.5)
    return struct.pack(PATCHED_FMT, *head, stamp, acc, *tail, float(stamp), prn, 123_456, 2330)


def read(path: Path, layout: str) -> object:
    return read_tracking_dump(path, layout, run_id="r1", channel=3, sys="G", signal="1C", fs_hz=FS, t_int_s=1e-3)  # type: ignore[arg-type]


def test_stock_dump_maps_every_field_to_track_stream(tmp_path: Path) -> None:
    f = tmp_path / "trk0.dat"
    f.write_bytes(stock_record(4_000, -12.5) + stock_record(8_000, -25.0, cn0=44.0))
    t = read(f, "stock108")
    rows = t.to_pylist()  # type: ignore[attr-defined]
    assert [r["k"] for r in rows] == [0, 1]
    assert [r["sample_count"] for r in rows] == [4_000, 8_000]
    assert rows[0]["t_rx_s"] == pytest.approx(0.001, abs=0.0)
    r = rows[0]
    assert (r["run_id"], r["channel"], r["sys"], r["prn"], r["signal"]) == ("r1", 3, "G", 7, "1C")
    assert (r["abs_e"], r["abs_p"], r["abs_l"], r["prompt_i"], r["prompt_q"]) == (
        2.0,
        3.5,
        pytest.approx(2.1),
        1.25,
        -0.5,
    )
    assert r["acc_phase_rad_f32"] == -12.5 and r["acc_phase_rad"] is None
    assert r["dop_hz"] == 300.0 and r["dop_rate_hzps"] == -0.25 and r["code_freq_chips"] == 1.023e6
    assert r["carr_err"] == pytest.approx(0.002) and r["code_err_chips"] == pytest.approx(0.001)
    assert (r["cn0_dbhz"], rows[1]["cn0_dbhz"]) == (45.5, 44.0)
    assert r["lock_test"] == pytest.approx(0.93) and (r["tow_ms"], r["wn"]) == (123_456, 2330)
    assert r["t_int_s"] == pytest.approx(1e-3)


def test_patched_dump_keeps_the_float64_phase(tmp_path: Path) -> None:
    acc = -3.8e5 - 0.012345678  # float32 spacing here is 0.031 rad (docs/3 section 2.6)
    f = tmp_path / "trk0.dat"
    f.write_bytes(patched_record(4_000, acc) + patched_record(8_000, acc - 1.0))
    rows = read(f, "patched112").to_pylist()  # type: ignore[attr-defined]
    assert rows[0]["acc_phase_rad"] == acc and rows[1]["acc_phase_rad"] == acc - 1.0
    assert rows[0]["acc_phase_rad_f32"] is None


def test_partial_last_record_raises(tmp_path: Path) -> None:
    f = tmp_path / "trk0.dat"
    f.write_bytes(stock_record(4_000, 0.0) + stock_record(8_000, 0.0)[:50])
    with pytest.raises(DumpError, match="not a multiple of 108"):
        read(f, "stock108")


def test_wrong_layout_is_detected_not_misread(tmp_path: Path) -> None:
    # 27 patched records are 3024 bytes, a whole number (28) of stock records: only the content can tell
    f = tmp_path / "trk0.dat"
    f.write_bytes(b"".join(patched_record(4_000 * (i + 1), -float(i)) for i in range(27)))
    with pytest.raises(DumpError, match="sample count"):
        read(f, "stock108")


OBS_FMT = "<7d"  # RX time, TOW, Doppler, carrier phase (cycles), pseudorange, PRN, valid flag (docs/9 section 4.2)


def obs_epoch(rx: float, chans: list[tuple[float, float, float, int, int]]) -> bytes:
    return b"".join(
        struct.pack(OBS_FMT, rx, rx - 0.07, dop, cp, pr, float(prn), float(ok)) for dop, cp, pr, prn, ok in chans
    )


def test_observables_rows_per_channel_and_epoch(tmp_path: Path) -> None:
    f = tmp_path / "observables.dat"
    f.write_bytes(
        obs_epoch(100.00, [(300.0, 1.5e6 + 0.25, 2.1e7, 5, 1), (-1200.0, -2.0e6, 2.3e7, 12, 1)])
        + obs_epoch(100.02, [(300.1, 1.5e6 + 6.25, 2.1e7 + 1.1, 5, 1), (0.0, 0.0, 0.0, 0, 0)])
    )
    t = read_observables_dump(f, {0: ("G", "1C", 5), 1: ("G", "1C", 12)}, source_id="cttc", receiver_id="twin-clean")
    rows = t.to_pylist()
    assert [(r["t_gps_s"], r["prn"], r["valid"]) for r in rows] == [
        (100.00, 5, True), (100.00, 12, True), (100.02, 5, True), (100.02, 12, False),
    ]  # fmt: skip
    assert (
        rows[2]["cp_cyc"] == 1.5e6 + 6.25
        and rows[2]["pr_m"] == 2.1e7 + 1.1
        and rows[2]["dop_hz"] == pytest.approx(300.1)
    )
    assert {(r["source_id"], r["receiver_id"], r["sys"], r["signal"]) for r in rows} == {
        ("cttc", "twin-clean", "G", "1C")
    }
    assert rows[0]["cn0_dbhz"] is None and rows[0]["cp_std_code"] is None


def test_observables_prn_that_contradicts_the_channel_map_raises(tmp_path: Path) -> None:
    f = tmp_path / "observables.dat"
    f.write_bytes(obs_epoch(100.0, [(300.0, 1.0, 2.1e7, 9, 1)]))
    with pytest.raises(DumpError, match="channel 0 reports PRN 9"):
        read_observables_dump(f, {0: ("G", "1C", 5)}, source_id="s", receiver_id="r")


def test_observables_size_must_match_the_channel_count(tmp_path: Path) -> None:
    f = tmp_path / "observables.dat"
    f.write_bytes(obs_epoch(100.0, [(300.0, 1.0, 2.1e7, 5, 1)] * 3))  # 3 channels written
    with pytest.raises(DumpError, match="not a multiple of 112"):
        read_observables_dump(f, {0: ("G", "1C", 5), 1: ("G", "1C", 6)}, source_id="s", receiver_id="r")


def test_observables_channel_map_must_number_every_channel(tmp_path: Path) -> None:
    f = tmp_path / "observables.dat"
    f.write_bytes(obs_epoch(100.0, [(300.0, 1.0, 2.1e7, 5, 1)] * 2))
    with pytest.raises(ValueError, match="numbered 0 to 1"):
        read_observables_dump(f, {0: ("G", "1C", 5), 2: ("G", "1C", 6)}, source_id="s", receiver_id="r")


def test_float64_phase_resolution_survives_parsing(tmp_path: Path) -> None:
    """The observables carrier phase is float64 end to end: 1e-6 cycle steps at 1.5e6 cycles are kept."""
    f = tmp_path / "observables.dat"
    f.write_bytes(b"".join(obs_epoch(100.0 + 0.02 * i, [(300.0, 1.5e6 + 1e-6 * i, 2.1e7, 5, 1)]) for i in range(4)))
    cp = read_observables_dump(f, {0: ("G", "1C", 5)}, source_id="s", receiver_id="r").column("cp_cyc").to_numpy()
    assert np.all(np.diff(cp) > 0)


def test_observables_of_a_free_channel_take_the_prn_it_reports(tmp_path: Path) -> None:
    f = tmp_path / "observables.dat"
    f.write_bytes(obs_epoch(100.0, [(300.0, 1.0, 2.1e7, 9, 1), (0.0, 0.0, 0.0, 0, 0)]))
    rows = read_observables_dump(f, {0: ("G", "1C", 0), 1: ("G", "1C", 0)}, source_id="s", receiver_id="r").to_pylist()
    assert [(r["prn"], r["valid"]) for r in rows] == [(9, True), (0, False)]
