"""u-blox UBX reader: frame scanner and RXM-RAWX decoder (docs/7 section 5.1; U-io-ubx, docs/11 section 4).

In-house rather than pyubx2 (D-29): one fixed-layout message, decoded with numpy. Layout from the u-blox F9
interface description, UBX-RXM-RAWX (class 0x02, id 0x15): a 16-byte header, then 32 bytes per measurement.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from gnssaudit.io.schemas import OBS_EPOCH_SAT, build

SYNC = b"\xb5\x62"
RXM_RAWX = (0x02, 0x15)
WEEK_S = 604_800.0

GNSS_SYS = {0: "G", 1: "S", 2: "E", 3: "C", 5: "J", 6: "R", 7: "I"}
# (gnssId, sigId) -> RINEX 3 signal code; other pairs are kept as "u<sigId>"
SIGNAL = {
    (0, 0): "1C",
    (0, 3): "2L",
    (0, 4): "2S",
    (0, 6): "5I",
    (0, 7): "5Q",
    (2, 0): "1C",
    (2, 1): "1B",
    (2, 3): "5I",
    (2, 4): "5Q",
    (2, 5): "7I",
    (2, 6): "7Q",
}

MEAS_DTYPE = np.dtype(
    [
        ("pr_mes", "<f8"),
        ("cp_mes", "<f8"),
        ("do_mes", "<f4"),
        ("gnss_id", "u1"),
        ("sv_id", "u1"),
        ("sig_id", "u1"),
        ("freq_id", "u1"),
        ("locktime_ms", "<u2"),
        ("cno", "u1"),
        ("pr_stdev", "u1"),
        ("cp_stdev", "u1"),
        ("do_stdev", "u1"),
        ("trk_stat", "u1"),
        ("reserved", "u1"),
    ]
)

TRK_PR_VALID = 0x01
TRK_CP_VALID = 0x02
TRK_HALF_CYC = 0x04
TRK_SUB_HALF_CYC = 0x08
CP_STD_STEP_CYC = 0.004
CP_STD_CENSORED = 14
CP_STD_INVALID = 15


def checksum(body: bytes) -> bytes:
    """8-bit Fletcher checksum over class, id, length and payload."""
    x = np.frombuffer(body, dtype=np.uint8).astype(np.int64)
    ck_a = int(x.sum()) & 0xFF
    ck_b = int(((x.size - np.arange(x.size)) * x).sum()) & 0xFF
    return bytes((ck_a, ck_b))


@dataclass(frozen=True)
class Frame:
    msg_class: int
    msg_id: int
    payload: bytes


def scan_frames(data: bytes) -> tuple[list[Frame], int]:
    """All frames with a valid checksum, and the number of sync words that did not start one."""
    frames: list[Frame] = []
    bad = 0
    i = data.find(SYNC)
    while i >= 0:
        if i + 8 > len(data):
            bad += 1
            break
        length = int.from_bytes(data[i + 4 : i + 6], "little")
        end = i + 8 + length
        if end <= len(data) and checksum(data[i + 2 : end - 2]) == data[end - 2 : end]:
            frames.append(Frame(data[i + 2], data[i + 3], data[i + 6 : end - 2]))
            i = data.find(SYNC, end)
        else:
            bad += 1
            i = data.find(SYNC, i + 1)
    return frames, bad


def decode_rawx(payload: bytes) -> tuple[float, np.ndarray]:
    """GPS time of the epoch (continuous seconds) and the measurement records of one RXM-RAWX payload."""
    rcv_tow, week, _leap_s, num_meas = struct.unpack_from("<dHbB", payload, 0)
    if len(payload) != 16 + 32 * num_meas:
        raise ValueError(f"RXM-RAWX payload of {len(payload)} bytes for {num_meas} measurements")
    meas = np.frombuffer(payload, dtype=MEAS_DTYPE, count=num_meas, offset=16)
    return week * WEEK_S + rcv_tow, meas


def _loss_of_lock(key: np.ndarray, t: np.ndarray, lock_s: np.ndarray, trk: np.ndarray) -> np.ndarray:
    """1 where the lock time of a signal fell since its previous epoch (the tracking restarted), or where its
    half-cycle-subtracted bit changed (a 0.5-cycle jump in cpMes)."""
    lli = np.zeros(t.size, dtype=np.uint8)
    order = np.lexsort((t, key))
    k, lk, sub = key[order], lock_s[order], trk[order] & TRK_SUB_HALF_CYC
    restart = (k[1:] == k[:-1]) & ((lk[1:] < lk[:-1]) | (sub[1:] != sub[:-1]))
    lli[order[1:][restart]] = 1
    return lli


def read_ubx_rawx(path: Path, source_id: str | None = None, receiver_id: str = "ubx") -> pa.Table:
    """obs_epoch_sat rows from every RXM-RAWX message in a UBX log.

    The number of sync words that did not start a valid frame is stored in the table metadata
    (`bad_frames`), so corruption is counted, not hidden. Elevation and azimuth are left null; the
    geometry stage fills them.
    """
    frames, bad = scan_frames(path.read_bytes())
    epochs = [decode_rawx(f.payload) for f in frames if (f.msg_class, f.msg_id) == RXM_RAWX]
    if epochs:
        meas = np.concatenate([m for _, m in epochs])
        t = np.concatenate([np.full(m.size, t0) for t0, m in epochs])
    else:
        meas, t = np.zeros(0, dtype=MEAS_DTYPE), np.zeros(0)
    n = meas.size
    code = meas["cp_stdev"] & 0x0F
    cp_std = np.where(code == CP_STD_INVALID, np.nan, CP_STD_STEP_CYC * code)
    trk = meas["trk_stat"]
    lock_s = meas["locktime_ms"] / 1000.0
    sys = [GNSS_SYS.get(int(g), "?") for g in meas["gnss_id"]]
    sig = [SIGNAL.get((int(g), int(s)), f"u{int(s)}") for g, s in zip(meas["gnss_id"], meas["sig_id"], strict=True)]
    key = meas["gnss_id"].astype(np.int64) * 65536 + meas["sv_id"].astype(np.int64) * 256 + meas["sig_id"]
    table = build(
        OBS_EPOCH_SAT,
        {
            "source_id": [source_id or path.stem] * n,
            "receiver_id": [receiver_id] * n,
            "t_gps_s": t,
            "sys": sys,
            "prn": meas["sv_id"].astype(np.int16),
            "signal": sig,
            "pr_m": meas["pr_mes"],
            "cp_cyc": meas["cp_mes"],
            "dop_hz": meas["do_mes"].astype(np.float64),
            "cn0_dbhz": meas["cno"].astype(np.float32),
            "lock_time_s": lock_s,
            "lli": _loss_of_lock(key, t, lock_s, trk),
            "trk_stat": trk,
            "cp_std_code": code,
            "cp_std_cyc": cp_std,
            "censored": code == CP_STD_CENSORED,
            "pr_std_m": 0.01 * 2.0 ** (meas["pr_stdev"] & 0x0F),
            "elev_deg": None,
            "azim_deg": None,
            "valid": ((trk & TRK_CP_VALID) != 0) & ((trk & TRK_HALF_CYC) != 0),
        },
        n,
    )
    return table.replace_schema_metadata({**(table.schema.metadata or {}), b"bad_frames": str(bad).encode()})
