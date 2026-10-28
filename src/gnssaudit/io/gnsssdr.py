"""GNSS-SDR v0.0.22 tracking and observables dump readers (docs/9 sections 4.1 and 4.2; docs/7 section 5.2).

The tracking dump has 24 fields and 108 bytes per record (`log_data()` in `dll_pll_veml_tracking.cc`, Verified [51]
at commit 3fcb502a). The patched build of D-08 writes field 9, the accumulated carrier phase, as float64 (112 bytes).
The observables dump holds, for every epoch and every output channel in channel order, seven float64 values
(`hybrid_observables_gs.cc:1442-1458`).

Both readers raise DumpError rather than drop data: on a size that is not a whole number of records (a run killed
mid-write), and on tracking records whose float64 sample count (field 21) differs from the uint64 stamp (field 8),
which is how a file read with the wrong layout shows (I-parser-golden, docs/11 section 6). RTKLIB_PVT's GeoJSON
position log is read for the admission screen (P2b task 5).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from gnssaudit.io.schemas import OBS_EPOCH_SAT, TRACK_STREAM, build

DumpLayout = Literal["stock108", "patched112"]
Channel = tuple[str, str, int]  # (sys, signal, prn): one entry per receiver channel, shared with gnssaudit.sdr

_HEAD = [
    ("abs_ve", "<f4"), ("abs_e", "<f4"), ("abs_p", "<f4"), ("abs_l", "<f4"), ("abs_vl", "<f4"),
    ("prompt_i", "<f4"), ("prompt_q", "<f4"), ("prn_start_sample", "<u8"),
]  # fmt: skip
_TAIL = [
    ("doppler_hz", "<f4"), ("doppler_rate", "<f4"), ("code_freq", "<f4"), ("code_rate", "<f4"),
    ("carr_err_cyc", "<f4"), ("carr_err_filt_hz", "<f4"), ("code_err", "<f4"), ("code_err_filt", "<f4"),
    ("cn0_dbhz", "<f4"), ("lock", "<f4"), ("rem_code_phase", "<f4"), ("sample_count", "<f8"),
    ("prn", "<u4"), ("tow", "<u8"), ("wn", "<u4"),
]  # fmt: skip
STOCK_108 = np.dtype([*_HEAD, ("acc_phase_rad", "<f4"), *_TAIL])
PATCHED_112 = np.dtype([*_HEAD, ("acc_phase_rad", "<f8"), *_TAIL])
_LAYOUTS: dict[str, np.dtype] = {"stock108": STOCK_108, "patched112": PATCHED_112}

# per epoch and channel: RX time (receiver TOW), TOW at the current symbol, Doppler, carrier phase in cycles,
# pseudorange, PRN and the pseudorange-valid flag, all float64 (docs/9 section 4.2)
OBSERVABLES_56 = np.dtype(
    [("rx_time_s", "<f8"), ("tow_s", "<f8"), ("dop_hz", "<f8"), ("cp_cyc", "<f8"), ("pr_m", "<f8"),
     ("prn", "<f8"), ("valid", "<f8")]
)  # fmt: skip


class DumpError(ValueError):
    """A GNSS-SDR dump is truncated, or does not have the layout it is read with."""


def _whole_records(path: Path, record_bytes: int, what: str) -> None:
    size = path.stat().st_size
    if size % record_bytes:
        raise DumpError(f"{path}: {size} bytes is not a multiple of {record_bytes} ({what})")


def read_tracking_dump(
    path: Path, layout: DumpLayout, *, run_id: str, channel: int, sys: str, signal: str, fs_hz: float, t_int_s: float
) -> pa.Table:
    """One channel's tracking dump as a `track_stream` table; k is the record index.

    t_int_s is the coherent integration time of the configuration, which the dump does not carry. The float32
    accumulated phase of the stock dump goes to `acc_phase_rad_f32` (not truth, D-08); the float64 one of the patched
    build goes to `acc_phase_rad`. The other phase column is null.
    """
    dt = _LAYOUTS[layout]
    _whole_records(path, dt.itemsize, layout)
    rec = np.fromfile(path, dtype=dt)
    stamp = rec["prn_start_sample"]
    off = rec["sample_count"] != stamp.astype(np.float64)
    if off.any():
        i = int(np.flatnonzero(off)[0])
        raise DumpError(
            f"{path}: record {i}: float64 sample count {rec['sample_count'][i]!r} differs from the stamp {stamp[i]} "
            f"(is the file really {layout}?)"
        )
    n = rec.size
    patched = layout == "patched112"
    cols: dict[str, object] = {
        "run_id": [run_id] * n,
        "channel": np.full(n, channel),
        "sys": [sys] * n,
        "prn": rec["prn"],
        "signal": [signal] * n,
        "k": np.arange(n),
        "sample_count": stamp,
        "t_rx_s": stamp.astype(np.float64) / fs_hz,
        "abs_ve": rec["abs_ve"],
        "abs_e": rec["abs_e"],
        "abs_p": rec["abs_p"],
        "abs_l": rec["abs_l"],
        "abs_vl": rec["abs_vl"],
        "prompt_i": rec["prompt_i"],
        "prompt_q": rec["prompt_q"],
        "acc_phase_rad_f32": None if patched else rec["acc_phase_rad"],
        "acc_phase_rad": rec["acc_phase_rad"] if patched else None,
        "dop_hz": rec["doppler_hz"],
        "dop_rate_hzps": rec["doppler_rate"],
        "code_freq_chips": rec["code_freq"],
        "code_rate_chipsps2": rec["code_rate"],
        "carr_err": rec["carr_err_cyc"],
        "carr_err_filt": rec["carr_err_filt_hz"],
        "code_err_chips": rec["code_err"],
        "code_err_filt": rec["code_err_filt"],
        "cn0_dbhz": rec["cn0_dbhz"],
        "lock_test": rec["lock"],
        "tow_ms": rec["tow"],
        "wn": rec["wn"],
        "t_int_s": np.full(n, t_int_s),
    }
    return build(TRACK_STREAM, cols, n)


def read_observables_dump(path: Path, channels: Mapping[int, Channel], *, source_id: str, receiver_id: str) -> pa.Table:
    """The observables dump as `obs_epoch_sat` rows, one per epoch and channel, in file order.

    `channels` names every receiver channel, numbered from 0; PRN 0 marks a free channel, whose rows take the PRN it
    reports. A channel that reports PRN 0 is idle at that epoch: its row is kept with `valid` false and null
    measurements. A channel that reports another PRN than assigned raises.
    `t_gps_s` holds the receiver time (RX time) of the epoch, as docs/7 section 5.1 allows for GNSS-SDR twins.
    """
    n_ch = len(channels)
    if sorted(channels) != list(range(n_ch)):
        raise ValueError(f"channels must be numbered 0 to {n_ch - 1}, got {sorted(channels)}")
    _whole_records(path, OBSERVABLES_56.itemsize * n_ch, f"56 B x {n_ch} channels")
    rec = np.fromfile(path, dtype=OBSERVABLES_56).reshape(-1, n_ch)
    for ch in range(n_ch):
        got = rec[:, ch]["prn"]
        wrong = (got != 0) & (got != channels[ch][2]) & (channels[ch][2] != 0)
        if wrong.any():
            raise DumpError(f"{path}: channel {ch} reports PRN {int(got[wrong][0])}, assigned PRN {channels[ch][2]}")
    flat = rec.reshape(-1)
    chan = np.tile(np.arange(n_ch), rec.shape[0])
    idle = flat["prn"] == 0
    n = flat.size
    assigned = np.array([channels[int(c)][2] for c in chan])
    prn = np.where(assigned == 0, flat["prn"].astype(np.int64), assigned)  # a free channel (PRN 0) reports its own

    def measured(name: str) -> np.ndarray:
        return np.where(idle, np.nan, flat[name])

    cols: dict[str, object] = {
        "source_id": [source_id] * n,
        "receiver_id": [receiver_id] * n,
        "t_gps_s": flat["rx_time_s"],
        "sys": [channels[int(c)][0] for c in chan],
        "prn": prn,
        "signal": [channels[int(c)][1] for c in chan],
        "pr_m": measured("pr_m"),
        "cp_cyc": measured("cp_cyc"),
        "dop_hz": measured("dop_hz"),
        "cn0_dbhz": None,
        "lock_time_s": None,
        "lli": None,
        "trk_stat": None,
        "cp_std_code": None,
        "cp_std_cyc": None,
        "censored": None,
        "pr_std_m": None,
        "elev_deg": None,
        "azim_deg": None,
        "valid": (flat["valid"] != 0) & ~idle,
    }
    return build(OBS_EPOCH_SAT, cols, n)


def read_pvt_geojson(path: Path) -> NDArray[np.float64]:
    """GNSS-SDR's position fixes (RTKLIB_PVT's GeoJSON MultiPoint) as (lon, lat, h) rows, in output order."""
    coords = json.loads(path.read_text())["geometry"]["coordinates"]
    return np.asarray(coords, dtype=np.float64).reshape(-1, 3)


def read_pvt_nmea(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """GNSS-SDR's timed fixes from its NMEA log (RTKLIB_PVT, `$xxGGA` sentences with a fix): the UTC time of day (s),
    the latitude and the longitude (degrees), in output order. The time is the fix's receiver time less the leap
    seconds (RTKLIB's `gpst2utc`); the geojson log carries positions only."""
    tod: list[float] = []
    lat: list[float] = []
    lon: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        f = line.split("*")[0].split(",")
        if len(f) < 7 or not f[0].endswith("GGA") or not f[1] or not f[2] or not f[4] or f[6] in ("", "0"):
            continue
        hh, mm, ss = int(f[1][0:2]), int(f[1][2:4]), float(f[1][4:])
        la = int(f[2][:2]) + float(f[2][2:]) / 60.0
        lo = int(f[4][:3]) + float(f[4][3:]) / 60.0
        tod.append(3600.0 * hh + 60.0 * mm + ss)
        lat.append(-la if f[3] == "S" else la)
        lon.append(-lo if f[5] == "W" else lo)
    return np.array(tod), np.array(lat), np.array(lon)
