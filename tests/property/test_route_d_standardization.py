"""Route D standardisation is standard normal under the model (docs/3 section 4.5, docs/5 section 3.1)."""

import numpy as np
import pytest

from gnssaudit.eval.route_d import standardize_epoch


def test_route_d_standardisation_is_standard_normal_under_the_model() -> None:
    rng = np.random.default_rng(5)
    n_sat, n_epochs = 8, 4000
    var_t = rng.uniform(1e-6, 2e-5, n_sat)
    var_next = rng.uniform(1e-6, 2e-5, n_sat)
    w_all, chi2 = [], []
    for _ in range(n_epochs):
        inc = rng.standard_normal(n_sat) * np.sqrt(var_t + var_next) + rng.normal(0, 3.0)  # common clock increment
        out = standardize_epoch(inc - inc.mean(), var_t, var_next)
        w_all.append(out.w)
        chi2.append(out.chi2)
    w = np.concatenate(w_all)
    assert abs(w.mean()) < 4 / np.sqrt(w.size)
    assert w.var() == pytest.approx(1.0, abs=4 * np.sqrt(2 / w.size))
    assert np.mean(chi2) == pytest.approx(n_sat - 1, abs=4 * np.sqrt(2 * (n_sat - 1) / n_epochs))
