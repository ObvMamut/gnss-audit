"""Carrier wavelengths of docs/0 section 3.2 (L1 0.1903 m, L5 0.2548 m)."""

import pytest

from gnssaudit.signals import carrier_hz, wavelength_m


def test_wavelengths() -> None:
    assert wavelength_m("G", "1C") == pytest.approx(0.190293673, abs=1e-9)
    assert wavelength_m("E", "1B") == wavelength_m("G", "1C")
    assert wavelength_m("G", "5Q") == pytest.approx(0.254828049, abs=1e-9)
    assert carrier_hz("E", "7Q") == 1207.14e6


def test_unknown_signal_raises() -> None:
    with pytest.raises(KeyError, match="no carrier frequency"):
        wavelength_m("R", "1C")
