"""Review s23's CW calibration against a receiver-like correlator (P2b; s23 diagnosis, 21-22 Oct 2026).

GNSS-SDR cuts each 1 ms block on the sample grid at the code epoch (dll_pll_veml_tracking.cc,
`update_tracking_vars`). At s23's 300 Hz Doppler a code period is 14999.996 samples, so the epoch's sub-sample
offset drifts through a whole sample every 261 blocks, and the receiver sees the CW's despread power averaged over
code-sampling alignments. s23 must scale its CW so that this average, not the value at one alignment, is the
designed 4 sigma_n^2 (the s25 lesson, docs/14 section 6.1). At one alignment the IS-GPS-705J Q5 code gives 0.64 of
the average at f_D + 1 MHz, which made the redo of 20 Oct 2026 jam at 1.56 times the design.
"""

import numpy as np
import pytest

from review.s23_gnsssdr_l5_pilot import CHIP_RATE, CN0_PILOT_DB, DOPPLER, F_L5, FS, N_CHIPS, T, cw_rel_power, l5_code

CHIP0 = 517.3  # generate()'s code phase at sample 0, in chips
F_CHIP = CHIP_RATE * (1 + DOPPLER / F_L5)  # the recording's chip rate, code Doppler included
DRIFT_CYCLE_BLOCKS = 261  # 1 / (15000 - FS * N_CHIPS / F_CHIP): blocks per whole-sample drift of the epoch


def receiver_despread_power(code: np.ndarray, f_cw: float, n_blocks: int) -> float:
    """Mean squared prompt of a unit CW at f_cw over n_blocks code periods of the recording. Each block runs from the
    first sample at or after a code epoch to the last sample before the next one, and is correlated with the code
    at the recording's chip phase after carrier wipe-off at the Doppler."""
    p = 0.0
    for k in range(1, n_blocks + 1):
        start = int(np.ceil((k * N_CHIPS - CHIP0) / F_CHIP * FS))
        stop = int(np.ceil(((k + 1) * N_CHIPS - CHIP0) / F_CHIP * FS))
        t = np.arange(start, stop) / FS
        chips = code[np.floor(CHIP0 + F_CHIP * t).astype(np.int64) % N_CHIPS]
        p += abs(np.mean(chips * np.exp(2j * np.pi * (f_cw - DOPPLER) * t))) ** 2
    return p / n_blocks


@pytest.mark.parametrize("f_b", [2.0, 5.0])
def test_the_cw_has_the_designed_despread_power_in_the_receiver(f_b: float) -> None:
    code = l5_code("Q")
    f_cw = DOPPLER + 1.0e6 + f_b
    sigma_n2 = 1 / (2 * 10 ** (CN0_PILOT_DB / 10) * T)
    p_rx = receiver_despread_power(code, f_cw, 3 * DRIFT_CYCLE_BLOCKS)
    assert cw_rel_power(code, f_cw) * p_rx == pytest.approx(4 * sigma_n2, rel=0.01)
