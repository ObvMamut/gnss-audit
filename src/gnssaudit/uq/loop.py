"""Linear models of carrier tracking loops (docs/9 section 5, docs/4 section 3).

All phases are in cycles and frequencies in Hz. One step k is one coherent integration of length T.

The loop is a linear state-space system driven by the equivalent input phase u_k, the block-average true
phase plus the discriminator's additive noise. Two outputs matter:

* ``phase``: the NCO phase phi_k at the start of block k. GNSS-SDR accumulates exactly this
  (``d_acc_carrier_phase_rad``, dll_pll_veml_tracking.cc:1657), so it is the reported carrier phase
  (c = 0 in docs/4 section 3.1).
* ``disc``: the linearised discriminator d_k = u_k - psi_k, where psi_k is the replica phase the correlator
  actually averages over block k.

In GNSS-SDR the replica of block k ramps with the frequency set after block k-1, so
psi_k = phi_k + kappa * T * f_{k-1} with kappa = 1/2, while phi_{k+1} = phi_k + T f_k
(dll_pll_veml_tracking.cc:1627-1657). The textbook model has kappa = 0, and then d = (1 - H) u exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from scipy import linalg, optimize, signal

Output = Literal["phase", "disc"]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CarrierLoop:
    """Carrier loop. Open loop: s' = M s + N d, replica psi = C_psi s, reported phase phi = C_phi s.

    Closing d = u - psi gives s' = A s + B u with A = M - N C_psi and B = N.
    """

    a: FloatArray
    b: FloatArray
    m_open: FloatArray
    c_phase: FloatArray
    c_psi: FloatArray
    t_s: float
    bn_hz: float
    name: str

    # ------------------------------------------------------------------ constructors
    @classmethod
    def _assemble(
        cls,
        fz: FloatArray,
        fd: FloatArray,
        gz: FloatArray,
        gd: float,
        t_s: float,
        bn_hz: float,
        kappa: float,
        name: str,
        t_code_s: float | None = None,
    ) -> CarrierLoop:
        """Close a loop filter (z' = Fz z + Fd d, f = Gz z + Gd d) around the NCO.

        State s = [z, phi, f_prev]; d = u - phi - kappa T f_prev, and
        phi' = phi + T_c f + (T - T_c) f_prev, where T_c is the code period. With extended
        integration (T = N T_c) GNSS-SDR advances the NCO with the old frequency for N - 1 code
        periods and with the new one only for the last (dll_pll_veml_tracking.cc states 3 and 4).
        """
        t_c = t_s if t_code_s is None else t_code_s
        nz = fz.shape[0]
        n = nz + 2
        ip, iff = nz, nz + 1
        m = np.zeros((n, n))
        nvec = np.zeros(n)
        m[:nz, :nz] = fz
        nvec[:nz] = fd
        # phi' = phi + T_c f + (T - T_c) f_prev, with f = Gz z + Gd d
        m[ip, :nz] = t_c * gz
        m[ip, ip] = 1.0
        m[ip, iff] = t_s - t_c
        nvec[ip] = t_c * gd
        # f_prev' = f
        m[iff, :nz] = gz
        nvec[iff] = gd
        c_d = np.zeros(n)
        c_d[ip] = 1.0
        c_d[iff] = kappa * t_s
        a = m - np.outer(nvec, c_d)
        c_phase = np.zeros(n)
        c_phase[ip] = 1.0
        return cls(a=a, b=nvec, m_open=m, c_phase=c_phase, c_psi=c_d, t_s=t_s, bn_hz=bn_hz, name=name)

    @classmethod
    def gnsssdr(
        cls, bn_hz: float, t_s: float, order: int = 3, kappa: float = 0.5, t_code_s: float | None = None
    ) -> CarrierLoop:
        """GNSS-SDR's Tracking_FLL_PLL_filter with the FLL off (tracking_FLL_PLL_filter.cc).

        kappa = 0.5 with t_code_s = None (or the code period under extended integration) is the timing
        of the source; kappa = 0 with t_code_s = None is the simplified model behind docs/9 section 5.3.
        """
        if order == 3:
            w0 = bn_hz / 0.7845
            a3, b3 = 1.1, 2.4
            # w' = w + T w0^3 d ; x' = x + T (0.5 w' + a3 w0^2 d) ; f = 0.5 x' + b3 w0 d
            fz = np.array([[1.0, 0.0], [0.5 * t_s, 1.0]])
            fd = np.array([t_s * w0**3, 0.5 * t_s**2 * w0**3 + t_s * a3 * w0**2])
            gz = np.array([0.25 * t_s, 0.5])
            gd = 0.25 * t_s**2 * w0**3 + 0.5 * t_s * a3 * w0**2 + b3 * w0
        elif order == 2:
            w0 = bn_hz / 0.53
            a2 = 1.414
            # w' = w + w0^2 T d ; f = 0.5 (w' + w) + a2 w0 d
            fz = np.array([[1.0]])
            fd = np.array([w0**2 * t_s])
            gz = np.array([1.0])
            gd = 0.5 * w0**2 * t_s + a2 * w0
        else:
            raise ValueError(f"unsupported loop order {order}")
        return cls._assemble(
            fz, fd, gz, gd, t_s, bn_hz, kappa, f"gnsssdr{order}(kappa={kappa},Tc={t_code_s})", t_code_s
        )

    @classmethod
    def textbook(cls, bn_hz: float, t_s: float, order: int = 3, kappa: float = 0.0) -> CarrierLoop:
        """Kaplan-Hegarty design, forward-Euler integrators (reproduces docs/4 section 3.6, docs/9 section 5.3)."""
        if order == 3:
            w0 = bn_hz / 0.7845
            a3, b3 = 1.1, 2.4
            # forward Euler, every update uses the old state:
            # w' = w + T w0^3 d ; x' = x + T (w + a3 w0^2 d) ; f = x + b3 w0 d
            fz = np.array([[1.0, 0.0], [t_s, 1.0]])
            fd = np.array([t_s * w0**3, t_s * a3 * w0**2])
            gz = np.array([0.0, 1.0])
            gd = b3 * w0
        elif order == 2:
            w0 = bn_hz / 0.53
            a2 = 1.414
            # forward Euler: w' = w + w0^2 T d ; f = w + a2 w0 d
            fz = np.array([[1.0]])
            fd = np.array([w0**2 * t_s])
            gz = np.array([1.0])
            gd = a2 * w0
        else:
            raise ValueError(f"unsupported loop order {order}")
        return cls._assemble(fz, fd, gz, gd, t_s, bn_hz, kappa, f"textbook{order}(kappa={kappa})")

    @classmethod
    def gnsssdr_for_bn_eff(
        cls, target_hz: float, t_s: float, order: int = 3, kappa: float = 0.5, t_code_s: float | None = None
    ) -> CarrierLoop:
        """Inverse map (D-21): the configured bandwidth whose loop has the target B_n,eff.

        Searches upward from 0.2 x target and stops at the first root on a stable loop.
        """

        def gap(bn: float) -> float:
            loop = cls.gnsssdr(bn, t_s, order, kappa, t_code_s)
            if np.max(np.abs(loop.poles())) >= 1.0:
                return np.inf
            return loop.bn_eff_hz() - target_hz

        grid = np.geomspace(0.2 * target_hz, 5.0 * target_hz, 400)
        vals = np.array([gap(b) for b in grid])
        idx = np.nonzero((vals[:-1] < 0) & (vals[1:] >= 0) & np.isfinite(vals[1:]))[0]
        if idx.size == 0:
            raise ValueError(f"no stable configuration reaches B_n,eff = {target_hz} Hz at T = {t_s}")
        i = int(idx[0])
        bn = cast(float, optimize.brentq(gap, grid[i], grid[i + 1], xtol=1e-12))
        return cls.gnsssdr(bn, t_s, order, kappa, t_code_s)

    # ------------------------------------------------------------------ analysis
    def poles(self) -> NDArray[np.complex128]:
        return np.asarray(np.linalg.eigvals(self.a), dtype=np.complex128)

    def _c(self, output: Output) -> FloatArray:
        return self.c_phase if output == "phase" else -self.c_psi

    def _d(self, output: Output) -> float:
        return 0.0 if output == "phase" else 1.0

    def _open_loop(self, w: FloatArray) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Open-loop gains from d to psi (G) and to psi - phi (G - G_phi), formed without cancellation."""
        w = np.atleast_1d(np.asarray(w, dtype=float))
        # the open loop has its integrator poles at z = 1; evaluate DC as the limit
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        n = self.m_open.shape[0]
        eye = np.eye(n)
        g = np.empty(w.shape, dtype=complex)
        g_diff = np.empty(w.shape, dtype=complex)
        c_diff = self.c_psi - self.c_phase
        flat_w = w.ravel()
        flat_g = g.reshape(-1)
        flat_gd = g_diff.reshape(-1)
        for start in range(0, flat_w.size, 65_536):
            z = np.exp(1j * flat_w[start : start + 65_536])
            mats = z[:, None, None] * eye - self.m_open
            rhs = np.broadcast_to(self.b.astype(complex), (z.size, n))[..., None]
            x = np.linalg.solve(mats, rhs)[..., 0]
            flat_g[start : start + z.size] = x @ self.c_psi
            flat_gd[start : start + z.size] = x @ c_diff
        return g, g_diff

    def freq_response(self, w: FloatArray, output: Output = "phase") -> NDArray[np.complex128]:
        """H(e^{jw}) from u to the chosen output (w in rad/sample): phi/u for "phase", d/u for "disc"."""
        g, g_diff = self._open_loop(w)
        out = (g - g_diff) / (1.0 + g) if output == "phase" else 1.0 / (1.0 + g)
        return np.asarray(out, dtype=np.complex128)

    def error_response(self, w: FloatArray) -> NDArray[np.complex128]:
        """1 - H for the reported phase, computed without cancellation at low frequency."""
        g, g_diff = self._open_loop(w)
        return (1.0 + g_diff) / (1.0 + g)

    def transfer_function(self, output: Output = "phase") -> tuple[FloatArray, FloatArray]:
        """Numerator and denominator in z^-1 (scipy.signal convention)."""
        num, den = signal.ss2tf(self.a, self.b[:, None], self._c(output)[None, :], [[self._d(output)]])
        return np.asarray(num[0], dtype=float), np.asarray(den, dtype=float)

    def impulse_response(self, n: int, output: Output = "phase") -> FloatArray:
        u = np.zeros(n)
        u[0] = 1.0
        return self.filter(u, output)

    def filter(self, u: FloatArray, output: Output = "phase") -> FloatArray:
        """Run the linear loop on an input sequence, zero initial state."""
        num, den = self.transfer_function(output)
        return np.asarray(signal.lfilter(num, den, np.asarray(u, dtype=float)), dtype=float)

    def _gain(self, output: Output) -> float:
        p = linalg.solve_discrete_lyapunov(self.a, np.outer(self.b, self.b))
        c = self._c(output)
        return float(c @ p @ c + self._d(output) ** 2)

    def noise_gain(self) -> float:
        """(1/2pi) int |H|^2 dw for the reported phase: Var(H n) / Var(n) for white n."""
        return self._gain("phase")

    def discriminator_gain(self) -> float:
        """(1/2pi) int |H_d|^2 dw: Var(d) / Var(n) for white n."""
        return self._gain("disc")

    def bn_eff_hz(self) -> float:
        """Effective one-sided noise bandwidth (1/2T) (1/2pi) int |H|^2 (notation table, D-21)."""
        return self.noise_gain() / (2.0 * self.t_s)

    def crossover_hz(self) -> float:
        """Lowest frequency where |H| = |1 - H| (docs/4 P3)."""
        f = np.geomspace(1e-3, 0.5 / self.t_s, 20000)
        w = 2 * np.pi * f * self.t_s
        g = np.abs(self.freq_response(w)) ** 2 - np.abs(self.error_response(w)) ** 2
        i = int(np.argmax(g < 0))
        return float(f[i])

    def sensitivity_peak_db(self) -> float:
        f = np.geomspace(1e-3, 0.5 / self.t_s, 20000)
        return float(20 * np.log10(np.max(np.abs(self.error_response(2 * np.pi * f * self.t_s)))))
