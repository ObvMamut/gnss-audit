"""Carrier frequencies and wavelengths by system and RINEX 3 signal code (docs/0 section 3.2)."""

from __future__ import annotations

C_M_S = 299_792_458.0

# (system letter, RINEX band digit) -> carrier frequency in Hz
_CARRIER_HZ = {
    ("G", "1"): 1575.42e6,
    ("G", "2"): 1227.60e6,
    ("G", "5"): 1176.45e6,
    ("E", "1"): 1575.42e6,
    ("E", "5"): 1176.45e6,
    ("E", "7"): 1207.14e6,
    ("E", "8"): 1191.795e6,
}


def carrier_hz(sys: str, signal: str) -> float:
    """Carrier frequency of a RINEX 3 signal code such as ("G", "1C") or ("E", "5Q")."""
    try:
        return _CARRIER_HZ[(sys, signal[:1])]
    except KeyError:
        raise KeyError(f"no carrier frequency for system {sys!r}, signal {signal!r}") from None


def wavelength_m(sys: str, signal: str) -> float:
    return C_M_S / carrier_hz(sys, signal)
