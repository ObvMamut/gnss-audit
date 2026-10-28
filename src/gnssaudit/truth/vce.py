"""Least-squares variance component estimation with diagonal cofactors (docs/3 section 4.4; [12, 13]).

Model per block: y = A x + e, Cov(e) = sum_k sigma_k^2 Q_k, with Q_k the diagonal indicator of the observations
assigned to component k. With W = Cov(e)^-1 and M = W - W A (A^T W A)^-1 A^T W (M = W when A is None):

    N_kl = 1/2 tr(Q_k M Q_l M),   l_k = 1/2 (M y)^T Q_k (M y),   sigma = N^-1 l,

iterated to convergence; Cov(sigma_hat) = N^-1 at convergence under normality. Non-negativity uses an active set:
a component that turns negative is fixed at zero and the rest re-estimated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ZERO_FLOOR = 1e-12  # relative variance used for weighting observations of a component fixed at zero


@dataclass(frozen=True)
class VceBlock:
    y: FloatArray  # observations, shape (n,)
    component: IntArray  # component index of each observation, shape (n,)
    a: FloatArray | None = None  # nuisance design, shape (n, p), for example a clock column; None for zero-mean y


@dataclass(frozen=True)
class VceResult:
    sigma2: FloatArray  # component variances
    cov: FloatArray  # their covariance N^-1 (zero rows and columns for components fixed at zero)
    fixed_zero: NDArray[np.bool_]
    n_iter: int


def _normal_equations(blocks: Sequence[VceBlock], sigma2: FloatArray) -> tuple[FloatArray, FloatArray]:
    k = sigma2.size
    floor = ZERO_FLOOR * float(np.max(sigma2))
    n_mat = np.zeros((k, k))
    l_vec = np.zeros(k)
    for b in blocks:
        w = 1.0 / np.maximum(sigma2[b.component], floor)
        m = np.diag(w)
        if b.a is not None:
            wa = b.a * w[:, None]
            m = m - wa @ np.linalg.solve(b.a.T @ wa, wa.T)
        my = m @ b.y
        e = np.zeros((b.y.size, k))
        e[np.arange(b.y.size), b.component] = 1.0
        n_mat += 0.5 * e.T @ (m * m.T) @ e
        l_vec += 0.5 * e.T @ (my**2)
    return n_mat, l_vec


def ls_vce(blocks: Sequence[VceBlock], n_comp: int, max_iter: int = 100, tol: float = 1e-10) -> VceResult:
    pooled = np.concatenate([b.y for b in blocks])
    sigma2 = np.full(n_comp, float(np.var(pooled)) or 1.0)
    active = np.ones(n_comp, dtype=bool)
    n_mat = np.eye(n_comp)
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        n_mat, l_vec = _normal_equations(blocks, sigma2)
        new = np.zeros(n_comp)
        new[active] = np.linalg.solve(n_mat[np.ix_(active, active)], l_vec[active])
        negative = active & (new < 0)
        if negative.any():
            active &= ~negative
            new[~active] = 0.0
        converged = np.max(np.abs(new - sigma2)) <= tol * max(float(np.max(new)), 1e-300) and not negative.any()
        sigma2 = new
        if converged:
            break
    else:
        raise RuntimeError(f"LS-VCE did not converge in {max_iter} iterations")
    cov = np.zeros((n_comp, n_comp))
    cov[np.ix_(active, active)] = np.linalg.inv(n_mat[np.ix_(active, active)])
    return VceResult(sigma2=sigma2, cov=cov, fixed_zero=~active, n_iter=n_iter)
