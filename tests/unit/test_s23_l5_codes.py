"""Review s23's GPS L5 codes against IS-GPS-705J alone, chip for chip, for PRN 1's I5 and Q5 (P2b part 2, task 11a).

The reference is task 9b's ICD-only generator (check P-d, which matched the patched GNSS-SDR generator on 100 of 100
codes; D-40). XA starts from all ones with polynomial 1 + x9 + x10 + x12 + x13 and restarts after 8190 chips. XB is
loaded with Table 3-Ia's initial state, not the advance that s23 uses, with polynomial
1 + x + x3 + x4 + x6 + x7 + x8 + x12 + x13, and runs its natural course. Each chip is XA xor XB, the stage-13 outputs.
Code vectors follow section 3.3.2.2 (b): stage 13 is at the right, so character k is stage k + 1.
"""

import numpy as np
import pytest

from review.s23_gnsssdr_l5_pilot import N_CHIPS, XB_ADVANCE, l5_code

# IS-GPS-705J (01-AUG-2022), Table 3-Ia sheet 1 of 2, printed p. 6: PRN 1's initial XB code states
ICD_XB_INITIAL_STATE = {"I": "0101011100100", "Q": "1001011001100"}
XA_TAPS = (9, 10, 12, 13)
XB_TAPS = (1, 3, 4, 6, 7, 8, 12, 13)


def icd_register(vector: str, taps: tuple[int, ...], n: int, restart_after: int | None = None) -> np.ndarray:
    """n stage-13 outputs of a 13-stage register loaded with a section 3.3.2.2 code vector; feedback into stage 1."""
    s0 = [int(ch) for ch in vector]
    s = list(s0)
    out = np.empty(n, dtype=np.uint8)
    k = 0
    for i in range(n):
        out[i] = s[12]
        fb = 0
        for t in taps:
            fb ^= s[t - 1]
        s = [fb, *s[:12]]
        k += 1
        if restart_after is not None and k == restart_after:
            s, k = list(s0), 0
    return out


def icd_chips(kind: str) -> np.ndarray:
    xa = icd_register("1" * 13, XA_TAPS, N_CHIPS, restart_after=8190)
    xb = icd_register(ICD_XB_INITIAL_STATE[kind], XB_TAPS, N_CHIPS)
    return xa ^ xb


def as_bits(code: np.ndarray) -> np.ndarray:
    """s23's codes are 1 - 2 c; back to the chips c."""
    return ((1.0 - code) / 2.0).astype(np.uint8)


@pytest.mark.parametrize("kind", ["I", "Q"])
def test_s23_codes_equal_the_icd_reference_chip_for_chip(kind: str) -> None:
    code = l5_code(kind)
    assert code.shape == (N_CHIPS,) and set(np.unique(code)) == {-1.0, 1.0}
    assert np.array_equal(as_bits(code), icd_chips(kind))


@pytest.mark.parametrize("kind", ["I", "Q"])
def test_first_13_chips_are_the_table_footnote(kind: str) -> None:
    """Table 3-Ia's footnote: the rightmost bit of the initial state is the first XB chip out, and since XA starts at
    all ones the first 13 code chips are the complement of those XB chips."""
    first_xb = ICD_XB_INITIAL_STATE[kind][::-1]
    assert "".join(str(1 - b) for b in as_bits(l5_code(kind))[:13]) == first_xb


@pytest.mark.parametrize("kind", ["I", "Q"])
def test_the_reference_sees_the_stock_xb_wrap(kind: str) -> None:
    """XB read modulo 10230 from its advance (stock v0.0.22, and s23's first run) agrees with the ICD before chip
    10230 - advance and not after it, so the reference above would have caught the first run's codes."""
    xa = icd_register("1" * 13, XA_TAPS, N_CHIPS, restart_after=8190)
    xb_from_ones = icd_register("1" * 13, XB_TAPS, N_CHIPS)
    adv = XB_ADVANCE[kind]
    stock = xa ^ xb_from_ones[(adv + np.arange(N_CHIPS)) % N_CHIPS]
    differ = np.flatnonzero(stock != icd_chips(kind))
    assert differ.size > 0 and differ.min() >= N_CHIPS - adv
