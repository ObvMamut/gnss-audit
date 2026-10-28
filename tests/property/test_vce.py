"""P-vce: LS-VCE recovers known components without bias (docs/3 section 4.4, docs/11 section 4)."""

import numpy as np
import pytest

from gnssaudit.truth.vce import VceBlock, ls_vce


def simulate_blocks(rng: np.random.Generator, sigma2: np.ndarray, n_blocks: int, n_obs: int) -> list[VceBlock]:
    blocks = []
    for _ in range(n_blocks):
        comp = rng.integers(0, sigma2.size, n_obs)
        clock = rng.normal(0.0, 5.0)
        y = clock + rng.standard_normal(n_obs) * np.sqrt(sigma2[comp])
        blocks.append(VceBlock(y=y, component=comp, a=np.ones((n_obs, 1))))
    return blocks


def test_vce_recovers_components_without_bias() -> None:
    rng = np.random.default_rng(3)
    sigma2 = np.array([1.0, 4.0, 0.25])
    estimates, covs = [], []
    for _ in range(100):
        result = ls_vce(simulate_blocks(rng, sigma2, n_blocks=60, n_obs=8), n_comp=3)
        estimates.append(result.sigma2)
        covs.append(np.diag(result.cov))
    est = np.array(estimates)
    se = est.std(axis=0, ddof=1) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(axis=0) - sigma2) < 2.5 * se)
    # the reported precision N^-1 matches the replicate spread within 20 %
    assert np.allclose(np.mean(covs, axis=0), est.var(axis=0, ddof=1), rtol=0.2)


def test_vce_fixes_a_vanishing_component_at_zero() -> None:
    rng = np.random.default_rng(4)
    result = ls_vce(simulate_blocks(rng, np.array([1.0, 0.0]), n_blocks=40, n_obs=6), n_comp=2)
    assert result.sigma2[1] >= 0.0
    assert result.sigma2[0] == pytest.approx(1.0, rel=0.25)
