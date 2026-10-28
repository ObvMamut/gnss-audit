"""U-io-ubx: RXM-RAWX decoding on synthetic golden bytes (docs/11 section 4). UGE data are GPL: no real fixtures."""

import struct
from pathlib import Path

import numpy as np
import pytest

from gnssaudit.io.schemas import OBS_EPOCH_SAT, validate
from gnssaudit.io.ubx import checksum, read_ubx_rawx, scan_frames


def frame(msg_class: int, msg_id: int, payload: bytes) -> bytes:
    body = bytes((msg_class, msg_id)) + struct.pack("<H", len(payload)) + payload
    return b"\xb5\x62" + body + checksum(body)


def meas(pr: float, cp: float, gnss: int, sv: int, sig: int, lock_ms: int, cno: int, pr_std: int, cp_std: int,
         trk: int) -> bytes:  # fmt: skip
    return struct.pack("<ddfBBBBHBBBBBB", pr, cp, -1234.5, gnss, sv, sig, 0, lock_ms, cno, pr_std, cp_std, 2, trk, 0)


def rawx(rcv_tow: float, week: int, records: list[bytes]) -> bytes:
    return struct.pack("<dHbBBB2x", rcv_tow, week, 18, len(records), 1, 1) + b"".join(records)


def test_checksum_matches_published_mon_ver_poll() -> None:
    # UBX-MON-VER poll: B5 62 0A 04 00 00 0E 34
    assert checksum(bytes((0x0A, 0x04, 0x00, 0x00))) == bytes((0x0E, 0x34))


def test_scanner_skips_junk_and_counts_corrupt_frames() -> None:
    good = frame(0x0A, 0x04, b"")
    corrupt = bytearray(frame(0x02, 0x15, rawx(1.0, 2300, [])))
    corrupt[-3] ^= 0xFF
    frames, bad = scan_frames(b"\x00\xb5junk" + good + bytes(corrupt) + good)
    assert [(f.msg_class, f.msg_id) for f in frames] == [(0x0A, 0x04), (0x0A, 0x04)]
    assert bad == 1


def test_rawx_decoding(tmp_path: Path) -> None:
    trk_ok = 0x01 | 0x02 | 0x04
    epoch1 = rawx(
        345600.5,
        2330,
        [
            meas(2.1e7, 1.1e8, 0, 5, 0, 5000, 45, 3, 3, trk_ok),  # GPS L1 C/A, cpStdev code 3
            meas(2.3e7, 1.2e8, 2, 11, 0, 800, 38, 5, 14, trk_ok),  # Galileo E1C, code 14: censored
            meas(2.4e7, 1.3e8, 0, 7, 0, 300, 30, 5, 15, 0x01),  # code 15: invalid, cp not valid
        ],
    )
    epoch2 = rawx(345601.5, 2330, [meas(2.1e7, 1.1e8 + 5.0, 0, 5, 0, 120, 44, 3, 4, trk_ok)])  # lock time fell
    path = tmp_path / "log.ubx"
    path.write_bytes(frame(0x01, 0x07, bytes(92)) + frame(0x02, 0x15, epoch1) + frame(0x02, 0x15, epoch2))

    table = read_ubx_rawx(path, source_id="s", receiver_id="r")
    assert validate(table, OBS_EPOCH_SAT).num_rows == 4
    assert table.schema.metadata[b"bad_frames"] == b"0"
    rows = table.to_pylist()
    assert rows[0]["t_gps_s"] == 2330 * 604800 + 345600.5
    assert (rows[0]["sys"], rows[0]["prn"], rows[0]["signal"]) == ("G", 5, "1C")
    assert (rows[1]["sys"], rows[1]["prn"], rows[1]["signal"]) == ("E", 11, "1C")
    assert rows[0]["cp_std_cyc"] == pytest.approx(0.012)
    assert rows[0]["censored"] is False
    assert rows[1]["cp_std_cyc"] == pytest.approx(0.056)
    assert rows[1]["censored"] is True
    assert rows[2]["cp_std_code"] == 15 and rows[2]["cp_std_cyc"] is None
    assert [r["valid"] for r in rows] == [True, True, False, True]
    assert rows[0]["pr_std_m"] == pytest.approx(0.08)
    assert rows[0]["lock_time_s"] == pytest.approx(5.0)
    assert [r["lli"] for r in rows] == [0, 0, 0, 1]
    assert np.isnan(table.column("elev_deg").to_numpy(zero_copy_only=False)).all()


def test_log_cut_mid_frame_keeps_complete_epochs(tmp_path: Path) -> None:
    epoch = frame(0x02, 0x15, rawx(100.0, 2330, [meas(2.1e7, 1.1e8, 0, 5, 0, 5000, 45, 3, 3, 0x07)]))
    path = tmp_path / "cut.ubx"
    path.write_bytes(epoch + epoch[:20])
    table = read_ubx_rawx(path)
    assert table.num_rows == 1
    assert table.schema.metadata[b"bad_frames"] == b"1"


def test_sub_half_cycle_change_is_a_slip(tmp_path: Path) -> None:
    # trkStat bit 3 (half cycle subtracted) changing inside an arc is a 0.5-cycle phase jump (review finding 6)
    epochs = [
        frame(0x02, 0x15, rawx(100.0 + k, 2330, [meas(2.1e7, 1.1e8, 0, 5, 0, 5000 + 1000 * k, 45, 3, 3, trk)]))
        for k, trk in enumerate([0x07, 0x07, 0x0F, 0x0F])
    ]
    path = tmp_path / "half.ubx"
    path.write_bytes(b"".join(epochs))
    assert read_ubx_rawx(path).column("lli").to_pylist() == [0, 0, 1, 0]
