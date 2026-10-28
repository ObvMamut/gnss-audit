"""KA-loop-inv: the inverse B_n,eff map (D-21; docs/9 section 5.3, docs/14 section 3.1, docs/11 section 3)."""

import pytest

from gnssaudit.uq.loop import CarrierLoop

# (T, target B_n,eff) -> configured bandwidth; simplified timing (docs/11) and source timing (docs/14 section 3.1)
SIMPLIFIED = {(1e-3, 5.0): 5.79, (1e-3, 15.0): 17.04, (4e-3, 5.0): 5.63, (4e-3, 15.0): 15.70,
              (20e-3, 5.0): 4.89, (20e-3, 15.0): 11.04}  # fmt: skip
FAITHFUL = {(1e-3, 5.0): 5.84, (1e-3, 15.0): 17.47, (4e-3, 5.0): 5.81, (4e-3, 15.0): 16.97,
            (20e-3, 5.0): 4.50, (20e-3, 15.0): 9.66}  # fmt: skip


def code_period(t_s: float) -> float | None:
    return 1e-3 if t_s > 4e-3 else None  # extended integration at 20 ms keeps the 1 ms C/A code period


@pytest.mark.parametrize(("cell", "configured"), SIMPLIFIED.items())
def test_inverse_map_simplified_timing(cell: tuple[float, float], configured: float) -> None:
    t_s, target = cell
    loop = CarrierLoop.gnsssdr_for_bn_eff(target, t_s, kappa=0.0)
    assert loop.bn_hz == pytest.approx(configured, rel=0.01)
    assert loop.bn_eff_hz() == pytest.approx(target, rel=1e-6)


@pytest.mark.parametrize(("cell", "configured"), FAITHFUL.items())
def test_inverse_map_source_timing(cell: tuple[float, float], configured: float) -> None:
    t_s, target = cell
    loop = CarrierLoop.gnsssdr_for_bn_eff(target, t_s, kappa=0.5, t_code_s=code_period(t_s))
    assert loop.bn_hz == pytest.approx(configured, rel=0.01)


def test_sensitivity_peaks_of_the_20_ms_15_hz_cell() -> None:
    simplified = CarrierLoop.gnsssdr_for_bn_eff(15.0, 20e-3, kappa=0.0)
    faithful = CarrierLoop.gnsssdr_for_bn_eff(15.0, 20e-3, kappa=0.5, t_code_s=1e-3)
    assert simplified.sensitivity_peak_db() == pytest.approx(3.73, abs=0.1)
    assert faithful.sensitivity_peak_db() == pytest.approx(5.72, abs=0.1)  # inside the 6 dB rule of D-21
