"""P-b1 (docs/11 section 4): the twin identity B1 on the nonlinear Costas loop (docs/3 section 2.2).

Pairing cancels thermal noise to first order: the paired twin difference is the loop's linear response to the
interference alone, H n_iota, up to a second-order residual. An unpaired difference keeps both thermal noises.
"""

import numpy as np
import pytest

from gnssaudit.sim.correlator import TrackingConfig, thermal_noise, track
from gnssaudit.uq.loop import CarrierLoop

pytestmark = pytest.mark.slow  # about 9 s: three 60 s closed-loop runs
CN0 = 10**4.5  # 45 dB-Hz
CFG = TrackingConfig(bn_hz=5.0, t_s=1e-3, pilot=False)  # Costas, third order, as the scratch check of 8 Oct 2026


def interference(rng: np.random.Generator, r: int, k: int) -> np.ndarray:
    """Moderate RFI: an in-band beat at 3 Hz plus coloured noise with a 2 Hz corner (correlator units, A = 1)."""
    t = np.arange(k) * CFG.t_s
    beat = 0.04 * np.exp(1j * (2 * np.pi * 3.0 * t + rng.uniform(0, 2 * np.pi, (r, 1))))
    rho = np.exp(-2 * np.pi * 2.0 * CFG.t_s)
    drive = (rng.standard_normal((r, k)) + 1j * rng.standard_normal((r, k))) * 0.05 * np.sqrt((1 - rho**2) / 2)
    col = np.zeros((r, k), complex)
    for i in range(1, k):
        col[:, i] = rho * col[:, i - 1] + drive[:, i]
    return beat + col


def test_pairing_cancels_thermal_noise_to_first_order() -> None:
    rng = np.random.default_rng(11)
    r, k, skip = 4, 60_000, 5_000
    zero = np.zeros((r, k))
    w, w_other = thermal_noise(rng, (r, k), CN0, CFG.t_s), thermal_noise(rng, (r, k), CN0, CFG.t_s)
    iota = interference(rng, r, k)
    clean, jammed, unpaired = track(CFG, zero, w), track(CFG, zero, w + iota), track(CFG, zero, w_other + iota)
    loop = CarrierLoop.gnsssdr(CFG.bn_hz, CFG.t_s)
    linear = np.stack([loop.filter(iota[i].imag / (2 * np.pi)) for i in range(r)])  # H n_iota, cycles
    d_paired = (jammed.phi - clean.phi)[:, skip:]
    resid = d_paired - linear[:, skip:]
    assert np.var(resid) / np.var(d_paired) < 0.01
    thermal = loop.bn_eff_hz() / CN0 * (1 + 1 / (2 * CN0 * CFG.t_s)) / (2 * np.pi) ** 2  # Costas squaring loss
    extra = np.var((unpaired.phi - clean.phi)[:, skip:] - linear[:, skip:])
    assert extra == pytest.approx(2 * thermal, rel=0.15)
