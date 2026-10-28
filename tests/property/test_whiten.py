"""P-whiten: whitening of DD residuals (docs/3 section 4.3, docs/11 section 4)."""

import numpy as np
import pytest

from gnssaudit.truth.arcs import whiten


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_whitening_gives_identity_covariance(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 6
    a = rng.standard_normal((n, n))
    cov = a @ a.T + n * np.eye(n)
    x = rng.multivariate_normal(np.zeros(n), cov, size=20_000)
    u = np.array([whiten(row, cov) for row in x])
    # entries of a sample covariance of 20 000 unit-variance draws have SE about 0.007 to 0.01
    assert np.allclose(np.cov(u.T), np.eye(n), atol=0.04)
