"""RINEX 2.11 and 3 observation readers and the RINEX 3 navigation reader (docs/7 section 5.1; U-io-rinex, docs/11
section 4).

In-house rather than georinex (D-29): georinex is unmaintained since 2023 and docs/6 section 6 names this
reader as its fallback. Observation files of RINEX 2.11 and 3.0x are accepted (RINEX 2 is the format of the IGN
RGP reference stations, read after Hatanaka decoding); RINEX 4 raises, and so does RINEX 2 navigation.
Navigation records are kept for GPS and Galileo, the systems the audit uses; records of other systems are counted
in the table metadata.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from gnssaudit.geometry import Ephemerides, KeplerEphemeris
from gnssaudit.io.schemas import OBS_EPOCH_SAT, build

GPS_EPOCH = datetime(1980, 1, 6)
OBS_FIELD = 16  # F14.3, LLI digit, signal-strength digit
TIME_SYSTEMS = {"", "GPS", "GAL"}  # Galileo system time is GPS time to within nanoseconds


def gps_seconds(year: int, month: int, day: int, hour: int, minute: int, second: float) -> float:
    return (datetime(year, month, day, hour, minute) - GPS_EPOCH).total_seconds() + second


def _label(line: str) -> str:
    return line[60:80].strip()


def _check_version(line: str, kind: str) -> None:
    version = float(line[0:9])
    if not 3.0 <= version < 4.0 or line[20] != kind:
        raise ValueError(f"expected a RINEX 3 file of type {kind!r}, found version {version} type {line[20]!r}")


def _header(lines: list[str], kind: str) -> tuple[dict[str, list[str]], int]:
    """Header records by label, and the index of the first body line."""
    _check_version(lines[0], kind)
    records: dict[str, list[str]] = {}
    for i, line in enumerate(lines):
        label = _label(line)
        if label == "END OF HEADER":
            return records, i + 1
        records.setdefault(label, []).append(line)
    raise ValueError("no END OF HEADER")


def _obs_types(records: list[str]) -> dict[str, list[str]]:
    types: dict[str, list[str]] = {}
    current = ""
    for line in records:
        if line[0] != " ":
            current = line[0]
            types[current] = []
        types[current].extend(line[7:58].split())
    return types


def _value(field: str) -> float:
    text = field[:14].strip()
    return float(text) if text else np.nan


def _obs_table(
    rows: dict[str, list], source_id: str, receiver: str, skipped: int, extra: dict[bytes, bytes]
) -> pa.Table:
    """The obs_epoch_sat table of collected rows (keys t, sys, prn, signal, C, L, D, S, lli)."""
    n = len(rows["t"])
    cp = np.array(rows["L"], dtype=float)
    table = build(
        OBS_EPOCH_SAT,
        {
            "source_id": [source_id] * n,
            "receiver_id": [receiver] * n,
            "t_gps_s": np.array(rows["t"], dtype=float),
            "sys": rows["sys"],
            "prn": np.array(rows["prn"], dtype=np.int16),
            "signal": rows["signal"],
            "pr_m": np.array(rows["C"], dtype=float),
            "cp_cyc": cp,
            "dop_hz": np.array(rows["D"], dtype=float),
            "cn0_dbhz": np.array(rows["S"], dtype=float),
            "lock_time_s": None,
            "lli": np.array(rows["lli"], dtype=np.uint8),
            "trk_stat": None,
            "cp_std_code": None,
            "cp_std_cyc": None,
            "censored": None,
            "pr_std_m": None,
            "elev_deg": None,
            "azim_deg": None,
            "valid": ~np.isnan(cp),
        },
        n,
    )
    metadata = {**(table.schema.metadata or {}), b"skipped_epochs": str(skipped).encode(), **extra}
    return table.replace_schema_metadata(metadata)


def _new_rows() -> dict[str, list]:
    return {k: [] for k in ("t", "sys", "prn", "signal", "C", "L", "D", "S", "lli")}


def _append_signals(
    rows: dict[str, list], t: float, sys: str, prn: int, per_signal: dict[str, dict[str, float]]
) -> None:
    for signal, entry in per_signal.items():
        if "C" not in entry and "L" not in entry:
            continue
        rows["t"].append(t)
        rows["sys"].append(sys)
        rows["prn"].append(prn)
        rows["signal"].append(signal)
        for obs in ("C", "L", "D", "S"):
            rows[obs].append(entry.get(obs, np.nan))
        rows["lli"].append(entry.get("lli", 0))


def read_rinex_obs(path: Path, source_id: str | None = None) -> pa.Table:
    """obs_epoch_sat rows, one per satellite and signal (RINEX 3 code such as 1C) that has a code or phase value.

    Dispatches on the version line: 2.x goes to `read_rinex2_obs`, 3.0x is read here, anything else (RINEX 4)
    raises. lli is the loss-of-lock digit of the phase observation. Epochs flagged 2 to 6 (events and cycle-slip
    records) are skipped and counted in the table metadata (`skipped_epochs`).
    """
    lines = path.read_text().splitlines()
    if lines and 2.0 <= float(lines[0][0:9]) < 3.0:
        return read_rinex2_obs(path, source_id)
    header, i = _header(lines, "O")
    first_obs = header.get("TIME OF FIRST OBS", [""])[0]
    time_system = first_obs[48:51].strip()
    if time_system not in TIME_SYSTEMS:
        raise ValueError(f"time system {time_system!r} is not supported")
    receiver = header.get("MARKER NAME", ["unknown"])[0][:60].strip()
    types = _obs_types(header["SYS / # / OBS TYPES"])
    rows = _new_rows()
    skipped = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(">"):
            raise ValueError(f"{path.name}:{i + 1}: expected an epoch record")
        flag, n_sat = int(line[31]), int(line[32:35])
        i += 1
        if flag > 1:
            skipped += 1
            i += n_sat
            continue
        t = gps_seconds(int(line[2:6]), int(line[7:9]), int(line[10:12]), int(line[13:15]), int(line[16:18]),
                        float(line[18:29]))  # fmt: skip
        for sat_line in lines[i : i + n_sat]:
            sys, prn = sat_line[0], int(sat_line[1:3])
            per_signal: dict[str, dict[str, float]] = {}
            for k, code in enumerate(types.get(sys, [])):
                field = sat_line[3 + OBS_FIELD * k : 3 + OBS_FIELD * (k + 1)]
                value = _value(field)
                if np.isnan(value):
                    continue
                entry = per_signal.setdefault(code[1:3], {})
                entry[code[0]] = value
                if code[0] == "L":
                    entry["lli"] = int(field[14]) if field[14:15].strip() else 0
            _append_signals(rows, t, sys, prn, per_signal)
        i += n_sat
    return _obs_table(rows, source_id or path.stem, receiver, skipped, {})


# One RINEX 3 signal per (system, band): RINEX 2 names a band, not a tracking mode, so the mode is a judgment.
_RINEX2_BANDS = {
    "G": {"1": "1C", "2": "2W", "5": "5X"},
    "E": {"1": "1X", "5": "5X", "7": "7X", "8": "8X", "6": "6X"},
    "R": {"1": "1C", "2": "2P"},
}
# Pseudorange types of a band in order of preference; the others are dropped.
_RINEX2_PSEUDORANGE = {
    "G": {"1": ("C1", "P1"), "2": ("P2", "C2"), "5": ("C5",)},
    "R": {"1": ("C1", "P1"), "2": ("P2", "C2")},
}


def _rinex2_code_map() -> dict[str, dict[str, str]]:
    """RINEX 2 observation type (such as L2) to RINEX 3 signal, per system, stored as JSON in the table metadata.

    Every type of a band (C, P, L, D, S) maps to the band's one signal: GPS 1C, 2W, 5X; Galileo 1X, 5X, 7X, 8X, 6X;
    GLONASS 1C, 2P. The pseudorange of a band is the first present of C1, P1 (GPS and GLONASS band 1), P2, C2
    (their band 2) and C5; the other code types of the band are dropped.
    """
    code_map: dict[str, dict[str, str]] = {}
    for sys, bands in _RINEX2_BANDS.items():
        code_map[sys] = {f"{obs}{band}": signal for band, signal in bands.items() for obs in "CPLDS"}
    return code_map


RINEX2_CODE_MAP = _rinex2_code_map()
_RINEX2_FIELDS = 5  # observations per 80-column line
_RINEX2_SAT_COLUMNS = slice(32, 68)  # 12 satellites of 3 columns on the epoch line and its continuations


def _rinex2_types(records: list[str]) -> list[str]:
    count = int(records[0][0:6])
    types = [t for line in records for t in line[6:60].split()]
    if len(types) != count:
        raise ValueError(f"# / TYPES OF OBSERV announces {count} types, lists {len(types)}")
    return types


def _rinex2_signals(
    sys: str, types: list[str], sat_lines: list[str], used: Counter[str]
) -> dict[str, dict[str, float]]:
    """Values of one satellite by RINEX 3 signal; `used` counts the pseudorange type chosen per system and signal."""
    padded = [line.ljust(_RINEX2_FIELDS * OBS_FIELD) for line in sat_lines]
    fields: dict[str, str] = {}
    for k, kind in enumerate(types):
        line = padded[k // _RINEX2_FIELDS]
        start = OBS_FIELD * (k % _RINEX2_FIELDS)
        fields[kind] = line[start : start + OBS_FIELD]
    per_signal: dict[str, dict[str, float]] = {}
    for band, signal in _RINEX2_BANDS.get(sys, {}).items():
        entry: dict[str, float] = {}
        for kind in _RINEX2_PSEUDORANGE.get(sys, {}).get(band, (f"C{band}",)):
            value = _value(fields.get(kind, ""))
            if not np.isnan(value):
                entry["C"] = value
                used[f"{sys}:{signal}:{kind}"] += 1
                break
        for obs in "LDS":
            field = fields.get(f"{obs}{band}", "")
            value = _value(field)
            if np.isnan(value):
                continue
            entry[obs] = value
            if obs == "L":
                entry["lli"] = int(field[14]) if field[14:15].strip() else 0
        per_signal[signal] = entry
    return per_signal


def read_rinex2_obs(source: Path | str, source_id: str | None = None) -> pa.Table:
    """obs_epoch_sat rows from a RINEX 2.11 observation file, with the schema and conventions of `read_rinex_obs`.

    `source` is a file path, or a str holding the RINEX text (as `hatanaka.crx_to_rnx` returns it); source_id
    defaults to the file stem, or to the marker name for text. RINEX 2 observation types are mapped to RINEX 3
    signal codes by `RINEX2_CODE_MAP`, stored as JSON in the table metadata (`rinex2_code_map`): one signal per
    system and band (GPS 1C, 2W, 5X; Galileo 1X, 5X, 7X, 8X, 6X; GLONASS 1C, 2P) carrying that band's code, phase,
    Doppler and signal strength. The pseudorange is chosen by preference (C1 then P1; P2 then C2; C5) and the code
    type actually used is counted in the metadata (`rinex2_code_used`, JSON "sys:signal:type" to rows). A blank
    system letter is GPS, two-digit years 80 to 99 are 19xx and 00 to 79 are 20xx. Event epochs (flags 2 to 5, and
    flag 6 with its satellite list and observation lines) are skipped and counted in `skipped_epochs`. The receiver
    clock offset is not read.
    """
    text = source.read_text() if isinstance(source, Path) else source
    lines = text.splitlines()
    label = source.name if isinstance(source, Path) else "RINEX text"
    if not lines or not 2.0 <= float(lines[0][0:9]) < 3.0 or lines[0][20] != "O":
        raise ValueError(f"{label}: expected a RINEX 2 observation file")
    records: dict[str, list[str]] = {}
    for i, line in enumerate(lines):
        if _label(line) == "END OF HEADER":
            i += 1
            break
        records.setdefault(_label(line), []).append(line)
    else:
        raise ValueError(f"{label}: no END OF HEADER")
    time_system = records.get("TIME OF FIRST OBS", [""])[0][48:51].strip()
    if time_system not in TIME_SYSTEMS:
        raise ValueError(f"time system {time_system!r} is not supported")
    receiver = records.get("MARKER NAME", ["unknown"])[0][:60].strip()
    types = _rinex2_types(records["# / TYPES OF OBSERV"])
    lines_per_sat = math.ceil(len(types) / _RINEX2_FIELDS)
    rows = _new_rows()
    used: Counter[str] = Counter()
    skipped = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        try:
            flag, n_sat = int(line[28:29]), int(line[29:32])
            yy = int(line[0:3])
            t = gps_seconds(1900 + yy if yy >= 80 else 2000 + yy, int(line[3:6]), int(line[6:9]), int(line[9:12]),
                            int(line[12:15]), float(line[15:26]))  # fmt: skip
        except ValueError as error:
            raise ValueError(f"{label}:{i + 1}: expected an epoch record") from error
        if 1 < flag < 6:  # n_sat counts the special header lines that follow
            skipped += 1
            i += 1 + n_sat
            continue
        extra = math.ceil(n_sat / 12) - 1
        n_obs_lines = n_sat * lines_per_sat
        if i + 1 + extra + (n_obs_lines if flag <= 1 or flag == 6 else 0) > len(lines):
            raise ValueError(f"{label}:{i + 1}: truncated epoch")
        if flag == 6:  # cycle-slip records: a satellite list and observation lines, as for flag 0
            skipped += 1
            i += 1 + extra + n_obs_lines
            continue
        sat_text = line[_RINEX2_SAT_COLUMNS].ljust(36)
        for cont in lines[i + 1 : i + 1 + extra]:
            sat_text += cont[_RINEX2_SAT_COLUMNS].ljust(36)
        i += 1 + extra
        for k in range(n_sat):
            sat = sat_text[3 * k : 3 * k + 3]
            sys = sat[0] if sat[0].isalpha() else "G"
            prn = int(sat[1:3])
            sat_lines = lines[i : i + lines_per_sat]
            i += lines_per_sat
            _append_signals(rows, t, sys, prn, _rinex2_signals(sys, types, sat_lines, used))
    extra_meta = {
        b"rinex2_code_map": json.dumps(RINEX2_CODE_MAP, sort_keys=True).encode(),
        b"rinex2_code_used": json.dumps(dict(sorted(used.items()))).encode(),
    }
    default_id = source.stem if isinstance(source, Path) else receiver
    return _obs_table(rows, source_id or default_id, receiver, skipped, extra_meta)


def _num(text: str) -> float:
    text = text.strip().replace("D", "E").replace("d", "e")
    return float(text) if text else 0.0


def _record(lines: list[str]) -> KeplerEphemeris:
    first = lines[0]
    sys, prn = first[0], int(first[1:3])
    toc = gps_seconds(int(first[4:8]), int(first[9:11]), int(first[12:14]), int(first[15:17]), int(first[18:20]),
                      float(first[21:23]))  # fmt: skip
    clock = [_num(first[c : c + 19]) for c in (23, 42, 61)]
    o = [_num(line[c : c + 19]) for line in lines[1:] for c in (4, 23, 42, 61)]
    o += [0.0] * (28 - len(o))
    return KeplerEphemeris(
        sys=sys, prn=prn, toc_s=toc, af0=clock[0], af1=clock[1], af2=clock[2],
        crs=o[1], delta_n=o[2], m0=o[3], cuc=o[4], e=o[5], cus=o[6], sqrt_a=o[7], toe_sow=o[8], cic=o[9],
        omega0=o[10], cis=o[11], i0=o[12], crc=o[13], omega=o[14], omega_dot=o[15], idot=o[16],
        week=int(o[18]), health=int(o[21]), data_sources=int(o[17]) if sys == "E" else 0,
    )  # fmt: skip


def read_rinex_nav(path: Path) -> tuple[Ephemerides, dict[str, int]]:
    """GPS and Galileo broadcast ephemerides, and the number of records skipped per other system."""
    lines = path.read_text().splitlines()
    _, i = _header(lines, "N")
    groups: list[list[str]] = []
    for line in lines[i:]:
        if not line.strip():
            continue
        if line[0] != " ":
            groups.append([line])
        else:
            groups[-1].append(line)
    records: list[KeplerEphemeris] = []
    skipped: dict[str, int] = {}
    for group in groups:
        sys = group[0][0]
        if sys in ("G", "E"):
            records.append(_record(group))
        else:
            skipped[sys] = skipped.get(sys, 0) + 1
    return Ephemerides(records), skipped
