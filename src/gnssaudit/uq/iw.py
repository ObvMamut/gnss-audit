"""In-phase witness (IW) estimator and its sampling theory (docs/4 section 3; corrections in docs/14).

Shadow filters use the loop's two transfers: H_phi (equivalent input -> reported phase), which carries noise
into the error, and H_d (equivalent input -> discriminator), which carries dynamics into the discriminator and
into the error referenced to the block-start true phase. For the simplified loop (kappa = 0) H_d = 1 - H_phi and
the identity reduces to docs/4 P2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from gnssaudit.uq.loop import CarrierLoop

FloatArray = NDArray[np.float64]


# ----------------------------------------------------------------------------- exact window statistics
def autocov_white_through(h: FloatArray, n_lags: int) -> FloatArray:
    """gamma(l) = sum_i h_i h_{i+l}, l = 0..n_lags-1: autocovariance of unit white noise filtered by h."""
    n = h.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    hf = np.fft.rfft(h, nfft)
    g = np.fft.irfft(np.abs(hf) ** 2, nfft)[:n_lags]
    out = np.zeros(n_lags)
    out[: g.size] = g
    return out


def window_variance_moments(gamma: FloatArray, n: int) -> tuple[float, float]:
    """Exact moments of the window variance s^2 = (1/N) sum (x_i - xbar)^2 of a zero-mean Gaussian process.

    Returns (E[s^2] / gamma_0, nu) with nu = 2 E[s^2]^2 / Var(s^2), the Satterthwaite degrees of freedom.
    With Q = I - 11'/N and covariance S: E = tr(QS)/N, Var = 2 tr(QSQS)/N^2.
    """
    g = np.asarray(gamma[:n], dtype=float)
    lags = np.arange(1, n)
    one_s_one = n * g[0] + 2.0 * np.sum((n - lags) * g[1:])
    tr_ss = n * g[0] ** 2 + 2.0 * np.sum((n - lags) * g[1:] ** 2)
    kernel = np.concatenate([g[:0:-1], g])  # gamma(-(N-1)) .. gamma(N-1)
    v = signal.fftconvolve(np.ones(n), kernel)[n - 1 : 2 * n - 1]  # S 1
    tr_qsqs = tr_ss - 2.0 / n * float(v @ v) + one_s_one**2 / n**2
    mean = (n * g[0] - one_s_one / n) / n
    var = 2.0 * tr_qsqs / n**2
    return float(mean / g[0]), float(2.0 * mean**2 / var)


def noise_term_moments(loop: CarrierLoop, n_window: int, n_impulse: int = 200_000) -> tuple[float, float]:
    """(mean-removal factor, nu) of the windowed witness term Var_W(H_phi m) for white m."""
    gamma = autocov_white_through(loop.impulse_response(n_impulse), n_window)
    return window_variance_moments(gamma, n_window)


# ----------------------------------------------------------------------------- identities
def identity_variance(d: FloatArray, m: FloatArray, loop: CarrierLoop, skip: int = 0) -> float:
    """Long-run P2 (c = 0), general timing: Var(e) = Var(d) - Var(H_d m) + Var(H_phi m)."""
    s = loop.filter(m, "disc")
    h = loop.filter(m, "phase")
    sl = slice(skip, None)
    return float(np.var(d[sl]) - np.var(s[sl]) + np.var(h[sl]))


# ----------------------------------------------------------------------------- windowed estimator
@dataclass(frozen=True)
class IWConfig:
    window: int  # samples per window (W / T)
    gate_z: float = 5.0
    gate_hz: float | None = None  # G low-pass corner; default 4 B_n,eff
    normalise: bool = True  # amplitude normalisation by the window mean of I (docs/4 section 3.7 step 2)
    gated: bool = True


@dataclass(frozen=True)
class IWOutput:
    var: FloatArray  # per-window sigma^2 estimate of the error, same units as m^2
    noise: FloatArray  # noise term V_W (corrected for mean removal)
    dyn: FloatArray  # dynamics term D_W
    gate: NDArray[np.bool_]
    nu: float  # degrees of freedom of the noise term
    dyn_raw: FloatArray  # ungated band-limited dynamics estimate Var(G d) - Var(G s) (may be negative)
    dyn_se: FloatArray  # its chi-square standard error


def _blocks(x: FloatArray, n: int) -> FloatArray:
    nb = x.size // n
    return x[: nb * n].reshape(nb, n)


def iw_windows(
    i_prompt: FloatArray,
    d: FloatArray,
    loop: CarrierLoop,
    cfg: IWConfig,
    m_override: FloatArray | None = None,
) -> IWOutput:
    """Windowed IW on non-overlapping boxcar windows (evaluation form of docs/4 section 3.7).

    ``i_prompt`` is the bit-wiped in-phase prompt (amplitude about 1), ``d`` the discriminator (radians). Pass
    ``m_override`` to supply an already-normalised witness.
    """
    n = cfg.window
    if m_override is not None:
        m = m_override
    elif cfg.normalise:
        ib = _blocks(i_prompt, n)
        m = (ib / ib.mean(axis=1, keepdims=True) - 1.0).ravel()
    else:
        m = i_prompt - 1.0
    h = loop.filter(m, "phase")
    s = loop.filter(m, "disc")
    factor, nu = noise_term_moments(loop, n)
    noise = _blocks(h, n).var(axis=1) / factor
    nb = noise.size
    dyn = np.zeros(nb)
    dyn_raw = np.zeros(nb)
    se = np.full(nb, np.nan)
    gate = np.zeros(nb, dtype=bool)
    if cfg.gated:
        fg = cfg.gate_hz if cfg.gate_hz is not None else 4.0 * loop.bn_eff_hz()
        sos = signal.butter(4, fg * 2 * loop.t_s, output="sos")
        gd = np.asarray(signal.sosfilt(sos, d[: nb * n]), dtype=np.float64)
        gs = np.asarray(signal.sosfilt(sos, s[: nb * n]), dtype=np.float64)
        vd = _blocks(gd, n).var(axis=1)
        vs = _blocks(gs, n).var(axis=1)
        dyn_raw = vd - vs
        # chi-square SE of G-filtered noise (white-m reading, as docs/4 step 6)
        imp = np.zeros(50_000)
        imp[0] = 1.0
        hg = np.asarray(signal.sosfilt(sos, loop.filter(imp, "disc")), dtype=np.float64)
        _, nu_g = window_variance_moments(autocov_white_through(hg, n), n)
        se = np.sqrt(2.0 * (vd**2 + vs**2) / nu_g)
        gate = dyn_raw > cfg.gate_z * se
        dyn = np.where(gate, dyn_raw, 0.0)
    return IWOutput(var=noise + dyn, noise=noise, dyn=dyn, gate=gate, nu=nu, dyn_raw=dyn_raw, dyn_se=se)
