"""Compact RINEX 1.0 (Hatanaka) decoding and `.Z` expansion (docs/7 section 2.2; docs/6 section 6, D-31).

IGN RGP publishes its 1 s data as hourly Hatanaka-compressed RINEX 2.11 files (`.d.Z`), the input of arm c. The
format is Hatanaka (2008), Bulletin of the Geospatial Information Authority of Japan 55, 21-30; the reference
implementation is `crx2rnx.c` of RNXCMP 4.2.0 (GSI). In-house rather than the `hatanaka` package or a bundled
`crx2rnx` (D-31). `crx_to_rnx` reproduces `crx2rnx` output byte for byte, including its conventions: header lines
lose trailing blanks, observations print without a leading zero (`.250`), and a blank observation blanks its flags.

Deliberate differences from `crx2rnx`, none of which arises in a well-formed file except the first:
- a negative receiver clock offset at an exact multiple of 0.1 s prints at its true value (`crx2rnx` 4.2.0 prints
  -0.1 s as `-.000000000`);
- input that `crx2rnx` would skip, truncate or reject raises ValueError with the line number;
- arcs of difference order 6 to 9, which the format allows and `crx2rnx` rejects, are decoded.

Compact RINEX 3 (RINEX 3 and 4) raises. Values are exact integers in units of 0.001 (observations) and 1e-9 s
(clock), so no rounding enters.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

FIELD_WIDTH = 14  # F14.3 observation
CLOCK_WIDTH = 12  # F12.9 receiver clock offset
SATS_PER_LINE = 12
FIELDS_PER_LINE = 5
BLANK_FIELD = " " * 16
DIGITS = "0123456789"
_INT = re.compile(r"\s*([+-]?\d+)")


def decompress_z(path: Path) -> bytes:
    """The content of a Unix-compress (`.Z`) or gzip file, expanded by the system `gzip -dc`."""
    gzip = shutil.which("gzip")
    if gzip is None:
        raise FileNotFoundError("gzip is not on PATH; it expands the .Z files (D-31)")
    done = subprocess.run([gzip, "-dc", "--", str(path)], capture_output=True, check=False)
    if done.returncode != 0:
        raise ValueError(f"{path.name}: gzip -dc failed ({done.returncode}): {done.stderr.decode(errors='replace')}")
    return done.stdout


def _atoi(text: str) -> int:
    """C atoi: leading blanks, a sign, the digits that follow; 0 if none."""
    m = _INT.match(text)
    return int(m.group(1)) if m else 0


def _chop(line: str) -> str:
    """Trailing blanks removed, but a non-empty line keeps its first character (crx2rnx CHOP_BLANK)."""
    return line.rstrip(" ") or line[:1]


def _repair(old: str, diff: str) -> str:
    """Apply a text difference: blank keeps, `&` blanks, anything else replaces; the excess of diff is appended."""
    if not diff:
        return old
    head = "".join(o if d == " " else (" " if d == "&" else d) for o, d in zip(old, diff, strict=False))
    if len(diff) > len(old):
        return head + diff[len(old) :].replace("&", " ")
    return head + old[len(diff) :]


def _number(text: str, line_no: int, what: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"line {line_no}: bad {what} {text!r}") from None


def _arc_start(text: str, line_no: int, what: str) -> tuple[int, int] | None:
    """(arc order, value) of an `n&value` field, or None for a plain difference."""
    if len(text) < 2 or text[1] != "&":
        return None
    if text[0] not in DIGITS:
        raise ValueError(f"line {line_no}: bad arc order in {what} {text!r}")
    return int(text[0]), _number(text[2:], line_no, what)


def _observation(v: int, flags: str, line_no: int) -> str:
    a = -v if v < 0 else v
    whole, frac = divmod(a, 1000)
    text = f"{'-' if v < 0 else ''}{whole or ''}.{frac:03d}"
    if len(text) > FIELD_WIDTH:
        raise ValueError(f"line {line_no}: observation {text} is out of the F14.3 range")
    return f"{text:>14}{flags}"


def _clock(v: int, line_no: int) -> str:
    a = -v if v < 0 else v
    whole, frac = divmod(a, 10**9)
    text = f"{'-' if v < 0 else ''}{whole or ''}.{frac:09d}"
    if len(text) > CLOCK_WIDTH:
        raise ValueError(f"line {line_no}: clock offset {text} is out of the F12.9 range")
    return f"{text:>12}"


def _types(line: str, ntype: int) -> int:
    """The number of observation types, updated by the first `# / TYPES OF OBSERV` line of a set."""
    if line[60:79] == "# / TYPES OF OBSERV" and line[5:6] != " ":
        return _atoi(line)
    return ntype


class _Lines:
    def __init__(self, text: str) -> None:
        if "\r" in text:
            text = text.replace("\r\n", "\n")
        self.lines = text.split("\n")
        if self.lines[-1] == "":
            self.lines.pop()
        self.pos = 0

    def next(self, what: str) -> tuple[str, int]:
        """The next line and its 1-based number; a missing line means the file is truncated."""
        if self.pos >= len(self.lines):
            raise ValueError(f"line {self.pos + 1}: file truncated, expected {what}")
        self.pos += 1
        return self.lines[self.pos - 1], self.pos


def _header(src: _Lines, out: list[str]) -> int:
    """Check the CRINEX and RINEX version lines, copy the RINEX header, return the number of observation types."""
    first, _ = src.next("the CRINEX VERS / TYPE line")
    if first[60:79] != "CRINEX VERS   / TYP":
        raise ValueError("line 1: not a Compact RINEX file (no CRINEX VERS / TYPE)")
    if first[:3] != "1.0":
        raise ValueError(f"line 1: Compact RINEX {first[:3]!r} is not supported; only 1.0 (RINEX 2)")
    src.next("the CRINEX PROG / DATE line")
    version, line_no = src.next("the RINEX VERSION / TYPE line")
    version = _chop(version)
    if version[60:80] != "RINEX VERSION / TYPE" or version[5:6] != "2":
        raise ValueError(f"line {line_no}: Compact RINEX 1.0 must hold a RINEX 2 file")
    out.append(version)
    ntype = 0
    while True:
        line, line_no = src.next("END OF HEADER")
        line = _chop(line)
        out.append(line)
        ntype = _types(line, ntype)
        if line[60:73] == "END OF HEADER":
            break
    if ntype <= 0:
        raise ValueError(f"line {line_no}: header has no # / TYPES OF OBSERV")
    return ntype


def _event(src: _Lines, dline: str, out: list[str], ntype: int) -> int:
    """Copy an event record (epoch flag above 1) and the records it announces; return the updated type count."""
    line = _chop(" " + dline[1:])
    out.append(line)
    if len(line) > 29:
        for _ in range(_atoi(line[29:])):
            record, _ = src.next("an event record")
            record = _chop(record)
            out.append(record)
            ntype = _types(record, ntype)
    if src.pos < len(src.lines):
        nxt = src.lines[src.pos]
        if nxt[:1] != "&" or len(nxt) < 29 or nxt[28] not in DIGITS:
            raise ValueError(f"line {src.pos + 1}: an epoch after an event must start afresh with '&'")
    return ntype


def crx_to_rnx(text: str) -> str:
    """RINEX 2 observation text from Compact RINEX 1.0 text, as RNXCMP `crx2rnx` writes it."""
    src = _Lines(text)
    out: list[str] = []
    ntype = _header(src, out)
    epoch = ""
    prev_index: dict[str, int] = {}
    prev_arcs: list[list[list[int] | None]] = []  # per satellite, per type: [arc order, y0, y1, ..., y_order]
    prev_flags: list[str] = []
    clk1 = [0] * 10  # clk1[k]: (order - k)-th difference of the clock, as crx2rnx indexes it
    clk0 = clk1[:]
    clk_order = clk_arc = 0
    while src.pos < len(src.lines):
        dline, epoch_no = src.next("an epoch record")
        if dline[:1] == "\x1a":  # DOS end of file
            break
        if dline[:1] == "&":
            if len(dline) < 29 or dline[28] not in DIGITS:
                raise ValueError(f"line {epoch_no}: bad epoch record {dline!r}")
            if dline[28] not in "01":
                ntype = _event(src, dline, out, ntype)
                continue
            epoch, prev_index = "", {}
            dline = " " + dline[1:]
        epoch = _repair(epoch, dline)
        if epoch[:1] != " " or len(epoch) < 29 or epoch[26:28] != "  " or epoch[28] not in DIGITS:
            raise ValueError(f"line {epoch_no}: not an epoch record after differencing: {epoch!r}")
        epoch = _chop(epoch)
        nsat = _atoi(epoch[29:])
        if len(epoch) < 32 + 3 * nsat:
            raise ValueError(f"line {epoch_no}: epoch record lists fewer than {nsat} satellites")
        sats = [epoch[32 + 3 * k : 35 + 3 * k] for k in range(nsat)]

        clock, clock_no = src.next("the clock offset record")
        have_clock = clock != ""
        if have_clock:
            start = _arc_start(clock, clock_no, "clock offset")
            if start is None:
                clk1[0] = _number(clock, clock_no, "clock offset")
            else:
                clk_arc, clk1[0] = start
                clk_order = -1
        else:
            clk_order = -1

        n_flags = 2 * ntype
        sat_lines: list[str] = []
        arcs_now: list[list[list[int] | None]] = []
        flags_now: list[str] = []
        for sat in sats:
            dl, line_no = src.next(f"the record of {sat}")
            parts = dl.split(" ", ntype)
            if len(parts) > ntype:
                dflag = parts.pop()
            else:
                dflag = ""
                parts += [""] * (ntype - len(parts))
            i0 = prev_index.get(sat, -1)
            if i0 < 0:
                old_arcs = None
                flags = dflag.replace("&", " ").ljust(n_flags)
            else:
                old_arcs = prev_arcs[i0]
                flags = _repair(prev_flags[i0], dflag).ljust(n_flags)
            arcs: list[list[int] | None] = []
            fields: list[str] = []
            blanks: list[int] = []
            for j, field in enumerate(parts):
                if not field:
                    arcs.append(None)
                    fields.append(BLANK_FIELD)
                    blanks.append(j)
                    continue
                if len(field) > 1 and field[1] == "&":
                    if field[0] not in DIGITS:
                        raise ValueError(f"line {line_no}: bad arc order in field {field!r}")
                    y = [int(field[0]), _number(field[2:], line_no, "field")]
                else:
                    if old_arcs is None:
                        raise ValueError(f"line {line_no}: {sat} is new to the epoch but its arc is not initialised")
                    old = old_arcs[j]
                    if old is None:
                        raise ValueError(f"line {line_no}: {sat} type {j + 1} was blank, but its arc is not restarted")
                    d = _number(field, line_no, "field")
                    order = len(old) - 2
                    if order < old[0]:
                        order += 1
                    y = old[: order + 2]
                    if len(y) < order + 2:
                        y.append(0)
                    y[order + 1] = d
                    for k in range(order, 0, -1):
                        y[k] = y[k + 1] + old[k]
                arcs.append(y)
                fields.append(_observation(y[1], flags[2 * j : 2 * j + 2], line_no))
            if blanks:
                fl = list(flags)
                for j in blanks:
                    fl[2 * j] = fl[2 * j + 1] = " "
                flags = "".join(fl)
            arcs_now.append(arcs)
            flags_now.append(flags)
            for s in range(0, ntype, FIELDS_PER_LINE):
                sat_lines.append("".join(fields[s : s + FIELDS_PER_LINE]).rstrip(" "))

        if have_clock:
            if clk_order < clk_arc:
                clk_order += 1
                for i in range(clk_order):
                    clk1[i + 1] = clk1[i] + clk0[i]
            else:
                for i in range(clk_order):
                    clk1[i + 1] = clk1[i] + clk0[i + 1]
        if clk_order >= 0:
            out.append(f"{epoch[:68]:<68}{_clock(clk1[clk_order], clock_no)}")
        else:
            out.append(epoch[:68])
        for k in range(68, 32 + 3 * nsat, 3 * SATS_PER_LINE):
            out.append(" " * 32 + epoch[k : k + 3 * SATS_PER_LINE])
        out += sat_lines

        clk0 = clk1[:]
        prev_index = {}
        for k, sat in enumerate(sats):
            prev_index.setdefault(sat, k)
        prev_arcs, prev_flags = arcs_now, flags_now
    return "\n".join(out) + "\n"
