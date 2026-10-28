"""U-io-hatanaka: Compact RINEX 1.0 (Hatanaka) decoding (docs/6 section 6, D-31; docs/7 section 2.2).

Format, as the fixture below uses it (Hatanaka 2008; RNXCMP `crx2rnx.c`):

- Two CRINEX header lines, then the RINEX 2 header; header lines come out without trailing blanks.
- Epoch record: one text line, the whole satellite list on it. A leading `&` starts afresh (the line is taken as
  written, with `&` read as a blank); otherwise the line is a text difference against the previous epoch line: a
  blank keeps the old character, `&` blanks it, anything else replaces it, and extra characters are appended.
- Next line: the receiver clock offset in units of 1e-9 s, empty when absent, written like an observation.
- One line per satellite: the fields of the `# / TYPES OF OBSERV` types separated by single blanks, then the LLI and
  signal-strength flag text (differenced like the epoch line; a satellite new to the epoch gives its flags in full).
  A field is empty for a blank observation. `n&value` (re-)starts an arc of difference order n with the value in
  units of 0.001; a bare number is the next difference of the arc, whose order grows by one per epoch up to n.
- A blank observation also blanks its flags. A satellite absent from the previous epoch must start all its arcs.
- An epoch flag above 1 is an event: its line and the records it announces pass through as text.
"""

import hashlib
import os
import random
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from gnssaudit.io.hatanaka import crx_to_rnx, decompress_z
from gnssaudit.paths import data_root

CRX_HEAD = [
    "1.0                 COMPACT RINEX FORMAT                    CRINEX VERS   / TYPE",
    "RNX2CRX ver.4.2.0                       10-Oct-26 12:00     CRINEX PROG / DATE",
]


def hdr(text: str, label: str) -> str:
    return f"{text:<60}{label:<20}"


def f(value: str, flags: str = "  ") -> str:
    """One RINEX 2 observation field: F14.3 as crx2rnx prints it (no leading zero), LLI, signal strength."""
    return f"{value:>14}{flags}"


B = " " * 16  # blank observation


def obs_line(*fields: str) -> str:
    return "".join(fields).rstrip(" ")


def crx(*lines: str) -> str:
    return "\n".join(lines) + "\n"


HEADER_6 = [
    hdr("     2.11           OBSERVATION DATA    G (GPS)", "RINEX VERSION / TYPE"),
    hdr("TEST", "MARKER NAME"),
    hdr("     6    L1    L2    C1    P2    S1    S2", "# / TYPES OF OBSERV"),
    hdr("", "END OF HEADER"),
]

# Five epochs, types L1 L2 C1 P2 S1 S2, receiver clock with a second-order arc.
#   G05 L1: third-order arc 110000000.123, .246+500, ... (differences 500123, 131, 15, -76)
#   G05 L2: re-initialised at epoch 3 (LLI 1 set there, cleared at epoch 4)
#   G05 C1: blank at epoch 2, so re-initialised at epoch 3; P2 always blank; S2 blank at epoch 5
#   G12 leaves after epoch 3; G07 appears at epoch 3; the satellite order changes at epochs 3 and 5
FIXTURE_CRX = crx(
    *CRX_HEAD,
    *HEADER_6,
    "&26 10 10  0  0  0.0000000  0  2G05G12",
    "2&123456",
    "3&110000000123 3&85000000456 3&21000000789  3&45250 3&40000  7 6",
    "3&120000000000 3&-1234567 3&23000000000 3&23000000500 3&750  15",
    "                 1",
    "100",
    "500123 389333   0 1000",
    "600000 300 5 7 -750 1&38500 &",
    "                 2             3  7 05G12",
    "10",
    "3&99999999999  3&19999999999  3&33000 3&-5  4",
    "131 3&85100000000 3&21000100000  0 -500   1",
    "100 3&-250 -3 0 1500 -500",
    "                 3             2      &&&",
    "10",
    "1  2  0 5",
    "15 389500 100100  0 0   &",
    "                 4                5  7",
    "",
    "-76 100 50  0   8",
    "1  0  0 5 1",
)

FIXTURE_RNX = crx(
    *(h.rstrip() for h in HEADER_6),
    f"{' 26 10 10  0  0  0.0000000  0  2G05G12':<68}  .000123456",
    obs_line(f("110000000.123", " 7"), f("85000000.456", " 6"), f("21000000.789"), B, f("45.250")),
    obs_line(f("40.000")),
    obs_line(f("120000000.000", "15"), f("-1234.567"), f("23000000.000"), f("23000000.500"), f(".750")),
    "",
    f"{' 26 10 10  0  0  1.0000000  0  2G05G12':<68}  .000123556",
    obs_line(f("110000500.246", " 7"), f("85000389.789", " 6"), B, B, f("45.250")),
    obs_line(f("41.000")),
    obs_line(f("120000600.000", " 5"), f("-1234.267"), f("23000000.005"), f("23000000.507"), f(".000")),
    obs_line(f("38.500")),
    f"{' 26 10 10  0  0  2.0000000  0  3G07G05G12':<68}  .000123666",
    obs_line(f("99999999.999", " 4"), B, f("19999999.999"), B, f("33.000")),
    obs_line(f("-.005")),
    obs_line(f("110001000.500", " 7"), f("85100000.000", "16"), f("21000100.000"), B, f("45.250")),
    obs_line(f("41.500")),
    obs_line(f("120001200.100", " 5"), f("-.250"), f("23000000.007"), f("23000000.514"), f(".750")),
    obs_line(f("38.000")),
    f"{' 26 10 10  0  0  3.0000000  0  2G07G05':<68}  .000123786",
    obs_line(f("100000000.000", " 4"), B, f("20000000.001"), B, f("33.000")),
    obs_line(f(".000")),
    obs_line(f("110001500.900", " 7"), f("85100389.500", " 6"), f("21000200.100"), B, f("45.250")),
    obs_line(f("41.500")),
    " 26 10 10  0  0  4.0000000  0  2G05G07",
    obs_line(f("110002001.370", " 8"), f("85100779.100", " 6"), f("21000300.250"), B, f("45.250")),
    "",
    obs_line(f("100000000.002", "14"), B, f("20000000.003"), B, f("33.000")),
    obs_line(f(".010")),
)

HEADER_1 = [
    hdr("     2.11           OBSERVATION DATA    G (GPS)", "RINEX VERSION / TYPE"),
    hdr("     1    L1", "# / TYPES OF OBSERV"),
    hdr("", "END OF HEADER"),
]


def test_fixture_decodes() -> None:
    assert crx_to_rnx(FIXTURE_CRX) == FIXTURE_RNX


def test_crlf_line_ends() -> None:
    assert crx_to_rnx(FIXTURE_CRX.replace("\n", "\r\n")) == FIXTURE_RNX


def test_more_than_twelve_satellites_continue_on_the_next_line() -> None:
    sats = "".join(f"G{k:02d}" for k in range(1, 15))
    text = crx(*CRX_HEAD, *HEADER_1, f"&26 10 10  0  0  0.0000000  0 14{sats}", "1&-5", *(["3&1000"] * 14))
    first = f" 26 10 10  0  0  0.0000000  0 14{sats[:36]}"
    expected = crx(
        *(h.rstrip() for h in HEADER_1), f"{first:<68} -.000000005", " " * 32 + sats[36:], *([f("1.000").rstrip()] * 14)
    )
    assert crx_to_rnx(text) == expected


def test_event_records_pass_through_and_may_change_the_types() -> None:
    text = crx(
        *CRX_HEAD,
        *HEADER_1,
        "&26 10 10  0  0  0.0000000  0  1G01",
        "",
        "3&1000",
        "&26 10 10  0  0  1.0000000  4  2",
        hdr("NEW TYPES FOLLOW", "COMMENT"),
        hdr("     2    L1    S1", "# / TYPES OF OBSERV"),
        "&26 10 10  0  0  1.0000000  0  1G01",
        "",
        "3&2000 3&45000",
    )
    expected = crx(
        *(h.rstrip() for h in HEADER_1),
        " 26 10 10  0  0  0.0000000  0  1G01",
        f("1.000").rstrip(),
        " 26 10 10  0  0  1.0000000  4  2",
        hdr("NEW TYPES FOLLOW", "COMMENT").rstrip(),
        hdr("     2    L1    S1", "# / TYPES OF OBSERV").rstrip(),
        " 26 10 10  0  0  1.0000000  0  1G01",
        f("2.000") + f("45.000").rstrip(),
    )
    assert crx_to_rnx(text) == expected


def test_a_blank_observation_clears_its_flags() -> None:
    text = crx(*EPOCH_1, "3&1000 15", "                 1", "", "", "                 2", "", "3&2000")
    out = crx_to_rnx(text).splitlines()
    assert out[4] == f("1.000", "15")
    assert out[6] == ""
    assert out[8] == f("2.000").rstrip()


def test_negative_clock_at_a_multiple_of_a_tenth_keeps_its_value() -> None:
    """crx2rnx 4.2.0 prints -0.1 s as -.000000000 and -1.2 s as -1.100000000; the decoder prints the value."""
    text = crx(
        *CRX_HEAD,
        *HEADER_1,
        "&26 10 10  0  0  0.0000000  0  1G01",
        "1&-100000000",
        "3&1000",
        "                 1",
        "-1100000000",
        "0",
    )
    out = crx_to_rnx(text).splitlines()
    assert out[3].endswith("  -.100000000")
    assert out[5].endswith(" -1.200000000")


EPOCH_1 = [*CRX_HEAD, *HEADER_1, "&26 10 10  0  0  0.0000000  0  1G01", ""]  # header, epoch line 6, clock line 7


@pytest.mark.parametrize(
    ("lines", "line_no", "match"),
    [
        (["3.0                 COMPACT RINEX FORMAT                    CRINEX VERS   / TYPE"], 1, "3.0"),
        (["not a compact rinex file"], 1, "Compact RINEX"),
        ([*CRX_HEAD, hdr("     3.04           OBSERVATION DATA    M", "RINEX VERSION / TYPE")], 3, "RINEX 2"),
        ([*EPOCH_1, "1000"], 8, "not initialised"),
        ([*CRX_HEAD, *HEADER_1, "&26 10 10  0  0  0.0000000  0  2G01G02", "", "3&1000"], 9, "truncated"),
        ([*EPOCH_1, "3&1000", "", "", "3&1000", "XYZ"], 12, "epoch"),
        ([*EPOCH_1, "3&1000", " " * 28 + "&", "", "1"], 9, "epoch"),
        ([*EPOCH_1, "3&1x00"], 8, "field"),
        ([*EPOCH_1, "", "", "", "1"], 11, "blank"),
        ([*EPOCH_1, "3&99999999999999"], 8, "range"),
        ([*EPOCH_1, "3&1000", "&26 10 10  0  0  1.0000000  4  0", "                 1"], 10, "afresh"),
    ],
)
def test_bad_input_raises_with_the_line_number(lines: list[str], line_no: int, match: str) -> None:
    with pytest.raises(ValueError, match=match) as err:
        crx_to_rnx(crx(*lines))
    assert f"line {line_no}:" in str(err.value)


# --- .Z expansion through the system gzip ----------------------------------------------------------------------


def lzw_literals(data: bytes) -> bytes:
    """A valid Unix-compress (.Z) stream of 9-bit literal codes only; short inputs keep the code width at 9."""
    assert len(data) < 250
    bits = 0
    nbits = 0
    out = bytearray(b"\x1f\x9d\x90")  # magic, block mode, 16-bit maximum
    for byte in data:
        bits |= byte << nbits
        nbits += 9
        while nbits >= 8:
            out.append(bits & 0xFF)
            bits >>= 8
            nbits -= 8
    if nbits:
        out.append(bits & 0xFF)
    return bytes(out)


def test_decompress_z_expands_a_compress_stream(tmp_path: Path) -> None:
    payload = b"1.0 COMPACT RINEX\nline two\n"
    path = tmp_path / "x.24d.Z"
    path.write_bytes(lzw_literals(payload))
    assert decompress_z(path) == payload


def test_decompress_z_raises_on_a_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.24d.Z"
    path.write_bytes(b"\x1f\x9dnot lzw at all")
    with pytest.raises(ValueError, match="bad.24d.Z"):
        decompress_z(path)


def test_decompress_z_raises_without_gzip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "x.24d.Z"
    path.write_bytes(lzw_literals(b"abc"))
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="gzip"):
        decompress_z(path)


# --- round trip through a minimal encoder ----------------------------------------------------------------------


def text_diff(old: str, new: str) -> str:
    out = []
    for k in range(max(len(old), len(new))):
        n = new[k] if k < len(new) else " "
        if k >= len(old):
            out.append(n)
        elif n == old[k]:
            out.append(" ")
        else:
            out.append("&" if n == " " else n)
    return "".join(out).rstrip(" ")


def nth_difference(values: list[int]) -> int:
    while len(values) > 1:
        values = [b - a for a, b in zip(values, values[1:], strict=False)]
    return values[0]


def f14(v: int) -> str:
    s = format(Decimal(v).scaleb(-3), "14.3f")
    return s.replace(" 0.", "  .").replace("-0.", " -.") if abs(v) < 1000 else s


def f12(v: int) -> str:
    s = format(Decimal(v).scaleb(-9), "12.9f")
    return s.replace(" 0.", "  .").replace("-0.", " -.") if abs(v) < 10**9 else s


class Encoder:
    """Compact RINEX 1.0 from epochs; arcs start on new data and, at random, again mid-arc."""

    def __init__(self, ntype: int, rng: random.Random) -> None:
        self.ntype, self.rng = ntype, rng
        self.line = ""
        self.clock: list[int] = []
        self.clock_arc = 0
        self.arcs: dict[tuple[str, int], tuple[int, list[int]]] = {}
        self.flags: dict[str, str] = {}

    def value(self, key: tuple[str, int] | None, v: int) -> str:
        arc, hist = self.arcs.get(key, (0, [])) if key else (0, [])
        if not hist or self.rng.random() < 0.1:
            arc = self.rng.randint(0, 3)
            if key:
                self.arcs[key] = (arc, [v])
            return f"{arc}&{v}"
        order = min(len(hist), arc)
        hist.append(v)
        return str(nth_difference(hist[len(hist) - 1 - order :]))

    def epoch(self, line: str, clock: int | None, sats: dict[str, list[tuple[int | None, str]]]) -> list[str]:
        out = ["&" + line[1:] if not self.line else text_diff(self.line, line)]
        if self.line == "":
            self.arcs, self.flags = {}, {}
        self.line = line
        if clock is None:
            self.clock = []
            out.append("")
        elif not self.clock:
            self.clock_arc = self.rng.randint(0, 3)
            self.clock = [clock]
            out.append(f"{self.clock_arc}&{clock}")
        else:
            order = min(len(self.clock), self.clock_arc)
            self.clock.append(clock)
            out.append(str(nth_difference(self.clock[len(self.clock) - 1 - order :])))
        arcs, flags = {}, {}
        for sat, obs in sats.items():
            fields = []
            for j, (v, _) in enumerate(obs):
                if v is None:
                    fields.append("")
                    continue
                key = (sat, j)
                if key in self.arcs:
                    arcs[key] = self.arcs[key]
                fields.append(self.value(key, v))
                arcs[key] = self.arcs[key]
            flags[sat] = "".join(fl for _, fl in obs)
            dflag = text_diff(self.flags[sat], flags[sat]) if sat in self.flags else flags[sat].rstrip(" ")
            out.append(" ".join(fields) + (" " + dflag if dflag else ""))
        self.arcs, self.flags = arcs, flags
        return out


def rinex_epoch(line: str, clock: int | None, nsat: int) -> list[str]:
    first = line[:68]
    out = [f"{first:<68}{f12(clock)}" if clock is not None else first]
    out += [" " * 32 + line[68 + 36 * k : 104 + 36 * k] for k in range((nsat - 1) // 12)]
    return out


values = st.one_of(st.integers(-2000, 2000), st.integers(-999_999_999_999, 9_999_999_999_999))
flag_char = st.sampled_from(" 0123456789")


@st.composite
def observation_files(draw: st.DrawFn) -> tuple[str, str]:
    ntype = draw(st.integers(1, 7))
    rng = random.Random(draw(st.integers(0, 2**32)))
    enc = Encoder(ntype, rng)
    types = "".join(f"{t:>6}" for t in ["L1", "L2", "C1", "P2", "S1", "S2", "D1"][:ntype])
    header = [
        hdr("     2.11           OBSERVATION DATA    G (GPS)", "RINEX VERSION / TYPE"),
        hdr(f"{ntype:6d}{types}", "# / TYPES OF OBSERV"),
        hdr("", "END OF HEADER"),
    ]
    crx_lines = [*CRX_HEAD, *header]
    rnx_lines = [h.rstrip() for h in header]
    pool = [f"G{k:02d}" for k in range(1, 16)]
    for k in range(draw(st.integers(1, 6))):
        sats = draw(st.lists(st.sampled_from(pool), max_size=15, unique=True))
        clock = draw(st.none() | st.integers(-9_999_999_999, 99_999_999_999))
        obs: dict[str, list[tuple[int | None, str]]] = {}
        for sat in sats:
            row = []
            for _ in range(ntype):
                v = draw(st.none() | values)
                row.append((v, "  " if v is None else draw(flag_char) + draw(flag_char)))
            obs[sat] = row
        flag = draw(st.sampled_from("01"))
        line = f" 26 10 10  0  0{float(k):11.7f}  {flag}{len(sats):3d}" + "".join(sats)
        if draw(st.booleans()) and k:
            enc.line = ""  # an epoch that starts every arc afresh
        crx_lines += enc.epoch(line, clock, obs)
        rnx_lines += rinex_epoch(line, clock, len(sats))
        for sat in sats:
            fields = [B if v is None else f14(v) + fl for v, fl in obs[sat]]
            rnx_lines += ["".join(fields[s : s + 5]).rstrip(" ") for s in range(0, ntype, 5)]
    return crx(*crx_lines), crx(*rnx_lines)


@settings(max_examples=200, deadline=None)
@given(observation_files())
def test_round_trip_through_a_minimal_encoder(case: tuple[str, str]) -> None:
    text, expected = case
    assert crx_to_rnx(text) == expected


# --- one RGP hour against crx2rnx --------------------------------------------------------------------------------

RGP_HOUR = data_root() / "raw" / "rgp" / "2024" / "253" / "agde253a.24d.Z"
RGP_SHA256 = "6e882eecf24ba615bbf2e80a0a4aa346b083ae8546a8a3841e55f9af89f6c513"


RGP_RNX_SHA256 = "c9e325b2f4c80da891b77efacd8171fe87f45c326f86e11fbbc151017e19517f"  # 22,844,247 bytes


def _decoded_rgp_hour() -> bytes:
    if not RGP_HOUR.is_file():
        pytest.skip(f"{RGP_HOUR} not fetched")
    assert hashlib.sha256(RGP_HOUR.read_bytes()).hexdigest() == RGP_SHA256
    return crx_to_rnx(decompress_z(RGP_HOUR).decode("ascii")).encode("ascii")


@pytest.mark.data
def test_rgp_hour_decodes_to_the_pinned_rinex() -> None:
    """Golden digest of crx_to_rnx(decompress_z(agde253a.24d.Z)), so the real-data check runs without crx2rnx.

    `hatanaka.py` is unchanged since bd8f433, where this output matched crx2rnx byte for byte (the optional test below
    repeats that comparison when a crx2rnx binary is at hand). Any later change to the decoder that alters one byte of
    this hour fails here; regenerate the digest only after rechecking against crx2rnx.
    """
    decoded = _decoded_rgp_hour()
    assert len(decoded) == 22_844_247
    assert hashlib.sha256(decoded).hexdigest() == RGP_RNX_SHA256


@pytest.mark.data
def test_rgp_hour_matches_crx2rnx_byte_for_byte() -> None:
    """GNSSAUDIT_CRX2RNX names a crx2rnx built from the RNXCMP source (GSI); it is never a project dependency."""
    tool = os.environ.get("GNSSAUDIT_CRX2RNX")
    if not tool or not Path(tool).is_file():
        pytest.skip("GNSSAUDIT_CRX2RNX does not name a crx2rnx binary")
    if not RGP_HOUR.is_file():
        pytest.skip(f"{RGP_HOUR} not fetched")
    assert hashlib.sha256(RGP_HOUR.read_bytes()).hexdigest() == RGP_SHA256
    crx_bytes = decompress_z(RGP_HOUR)
    reference = subprocess.run([tool, "-"], input=crx_bytes, capture_output=True, check=True).stdout
    decoded = crx_to_rnx(crx_bytes.decode("ascii")).encode("ascii")
    if decoded != reference:
        ours, theirs = decoded.split(b"\n"), reference.split(b"\n")
        first = next(k for k, (a, b) in enumerate(zip(ours, theirs, strict=False)) if a != b)
        pytest.fail(f"first difference at output line {first + 1}: {ours[first]!r} != {theirs[first]!r}")
