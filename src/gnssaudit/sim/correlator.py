"""Closed-loop, nonlinear, correlator-level carrier tracking simulator (Route A reduced, docs/3 section 5).

The prompt correlator of block k, in the frame of the nominal carrier (phases in cycles, A = 1 nominally), is

    P_k = D_k A_k exp(j 2 pi (theta_k - psi_k)) + eta_k exp(-j 2 pi psi_k),

where theta_k is the block-average true phase, psi_k the replica phase the correlator averages over the block,
D_k the data symbol (1 for a pilot), and eta_k the despread additive disturbance (thermal plus interference)
in the nominal frame. The update order follows GNSS-SDR v0.0.22 ``dll_pll_veml_tracking.cc``: discriminator,
loop filter (tracking_FLL_PLL_filter.cc), NCO advance phi_{k+1} = phi_k + T_c f_k + (T - T_c) f_{k-1}, with the
replica of block k ramping at f_{k-1} from phi_k (psi_k = phi_k + T f_{k-1} / 2). C/N0 is the M2M4 estimator
on a sliding buffer of ``cn0_samples`` prompts (lines 1355-1366), smoothed exponentially; the lock statistic is
the single-sample cos(2 angle P), smoothed (line 1369). A data-component correlator slaved to the same NCO is
produced when ``eta_data`` is given (track_pilot = true; dumped as Prompt I/Q, lines 1869-1876).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrackingConfig:
    bn_hz: float
    t_s: float
    pilot: bool = True  # atan2 discriminator (pilot) or Costas atan(Q/I)
    order: int = 3
    kappa: float = 0.5  # replica timing; 0 reproduces the simplified model of docs/9 section 5.3
    t_code_s: float | None = None  # code period under extended integration
    cn0_samples: int = 20
    cn0_alpha: float = 0.002
    cn0_init_samples: int = 200
    lock_alpha: float = 0.002
    lock_init_samples: int = 25


@dataclass(frozen=True)
class TrackingOutput:
    phi: FloatArray  # reported NCO phase at block start, cycles (relative to the nominal carrier)
    psi: FloatArray  # replica phase averaged by the correlator, cycles
    d: FloatArray  # discriminator output, cycles
    prompt: ComplexArray  # tracked-component prompt
    prompt_data: ComplexArray | None  # data-component prompt (slaved), if requested
    freq_hz: FloatArray  # NCO frequency after the update of block k
    cn0_dbhz: FloatArray  # smoothed M2M4 C/N0 (nan until available)
    cn0_raw_dbhz: FloatArray  # unsmoothed M2M4 C/N0
    lock: FloatArray  # smoothed lock statistic


def thermal_noise(rng: np.random.Generator, shape: tuple[int, ...], cn0_hz: float, t_s: float) -> ComplexArray:
    """Circular white Gaussian correlator noise for A = 1: A^2 / E|w|^2 = C/N0 T."""
    s = np.sqrt(0.5 / (cn0_hz * t_s))
    return s * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def m2m4_cn0_dbhz(buf: ComplexArray, t_s: float) -> FloatArray:
    """cn0_m2m4_estimator (tracking_discriminators.cc), vectorised over the leading axis."""
    p2 = buf.real**2 + buf.imag**2
    m2 = p2.mean(axis=-1)
    m4 = (p2 * p2).mean(axis=-1)
    psig = np.abs(buf.real).mean(axis=-1) ** 2
    rad = 2.0 * m2 * m2 - m4
    with np.errstate(invalid="ignore", divide="ignore"):
        aux = np.sqrt(np.where(rad >= 0, rad, np.nan))
        snr = np.where(np.isnan(aux), psig / (m2 - psig), aux / (m2 - aux))
        out = 10 * np.log10(snr) - 10 * np.log10(t_s)
    return np.where(snr > 0, out, -100.0)


def _filter_coeffs(cfg: TrackingConfig) -> tuple[float, float, float]:
    if cfg.order == 3:
        w0 = cfg.bn_hz / 0.7845
        return w0, 1.1, 2.4
    w0 = cfg.bn_hz / 0.53
    return w0, 1.414, 0.0


def track(
    cfg: TrackingConfig,
    theta_avg: FloatArray,
    eta: ComplexArray,
    bits: FloatArray | None = None,
    amplitude: FloatArray | None = None,
    eta_data: ComplexArray | None = None,
    bits_data: FloatArray | None = None,
    amplitude_data: FloatArray | None = None,
) -> TrackingOutput:
    """Run R independent channels (axis 0) for K blocks (axis 1)."""
    r, k = theta_avg.shape
    t = cfg.t_s
    t_c = t if cfg.t_code_s is None else cfg.t_code_s
    w0, a, b = _filter_coeffs(cfg)
    bits = np.ones((r, k)) if bits is None else np.broadcast_to(bits, (r, k))
    amp = np.ones((r, k)) if amplitude is None else np.broadcast_to(amplitude, (r, k))
    want_data = eta_data is not None
    bits_d = np.ones((r, k)) if bits_data is None else np.broadcast_to(bits_data, (r, k))
    amp_d = np.ones((r, k)) if amplitude_data is None else np.broadcast_to(amplitude_data, (r, k))

    phi_o = np.empty((r, k))
    psi_o = np.empty((r, k))
    d_o = np.empty((r, k))
    f_o = np.empty((r, k))
    p_o = np.empty((r, k), complex)
    pd_o = np.empty((r, k), complex) if want_data else None
    cn0_o = np.full((r, k), np.nan)
    cn0r_o = np.full((r, k), np.nan)
    lock_o = np.empty((r, k))

    phi = np.zeros(r)
    f_prev = np.zeros(r)
    w = np.zeros(r)
    x = np.zeros(r)
    buf = np.zeros((r, cfg.cn0_samples), complex)
    cn0_s = np.zeros(r)
    cn0_acc = np.zeros(r)
    n_cn0 = 0
    lock_s = np.zeros(r)
    lock_acc = np.zeros(r)
    two_pi = 2.0 * np.pi
    for i in range(k):
        psi = phi + cfg.kappa * t * f_prev
        rot = np.exp(-1j * two_pi * psi)
        p = bits[:, i] * amp[:, i] * np.exp(1j * two_pi * theta_avg[:, i]) * rot + eta[:, i] * rot
        if cfg.pilot:
            d = np.arctan2(p.imag, p.real) / two_pi
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.arctan(p.imag / p.real) / two_pi
            d = np.nan_to_num(d)
        if cfg.order == 3:
            w = w + t * w0**3 * d
            x = x + t * (0.5 * w + a * w0**2 * d)
            f = 0.5 * x + b * w0 * d
        else:
            w_new = w + d * w0**2 * t
            f = 0.5 * (w_new + w) + a * w0 * d
            w = w_new
        phi_o[:, i] = phi
        psi_o[:, i] = psi
        d_o[:, i] = d
        f_o[:, i] = f
        p_o[:, i] = p
        if want_data:
            assert pd_o is not None and eta_data is not None
            pd_o[:, i] = bits_d[:, i] * amp_d[:, i] * np.exp(1j * two_pi * theta_avg[:, i]) * rot + eta_data[:, i] * rot
        # C/N0: fill, then sliding buffer, estimate every integration (lines 1355-1366)
        buf[:, i % cfg.cn0_samples] = p
        if i + 1 >= cfg.cn0_samples:
            raw = m2m4_cn0_dbhz(buf, t)
            cn0r_o[:, i] = raw
            if n_cn0 < cfg.cn0_init_samples:
                cn0_acc += raw
                n_cn0 += 1
                cn0_s = cn0_acc / n_cn0
            else:
                cn0_s = cfg.cn0_alpha * raw + (1 - cfg.cn0_alpha) * cn0_s
            cn0_o[:, i] = cn0_s
        p2 = p.real**2 + p.imag**2
        lk = np.where(p2 > 0, (p.real**2 - p.imag**2) / np.where(p2 > 0, p2, 1.0), 0.0)
        if i < cfg.lock_init_samples:
            lock_acc += lk
            lock_s = lock_acc / (i + 1)
        else:
            lock_s = cfg.lock_alpha * lk + (1 - cfg.lock_alpha) * lock_s
        lock_o[:, i] = lock_s
        phi = phi + t_c * f + (t - t_c) * f_prev
        f_prev = f
    return TrackingOutput(phi_o, psi_o, d_o, p_o, pd_o, f_o, cn0_o, cn0r_o, lock_o)
