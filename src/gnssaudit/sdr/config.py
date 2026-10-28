"""GNSS-SDR configuration from a per-signal template plus the run's source, channels and overrides (docs/9 section 2).

Every rendered key must be one the receiver reads (`gnssaudit.sdr.keys`): GNSS-SDR silently ignores the others, so a
misspelt override would otherwise give a plausible default run. Loops are named by their effective bandwidth and
configured through the inverse map (D-21, docs/9 section 5.3).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from gnssaudit.io.gnsssdr import Channel
from gnssaudit.sdr.keys import unverified
from gnssaudit.uq.loop import CarrierLoop

TEMPLATES = Path(__file__).resolve().parents[3] / "configs" / "gnss-sdr"
CODE_PERIOD_S = {"1C": 1e-3, "1B": 4e-3, "L5": 1e-3}  # primary code period of each template's signal
_PINNED = re.compile(r"(^|@)sha256:[0-9a-f]{64}$")
_ADAPTER = {"ibyte": "Ibyte_To_Complex", "ishort": "Ishort_To_Complex", "gr_complex": "Pass_Through"}
_SIGNAL_ROLE = re.compile(r"^(Acquisition|Tracking|TelemetryDecoder|Channels)_(\w+)\.")  # a block of one signal

__all__ = ["Channel", "ReceiverSpec", "SourceSpec", "UnverifiedKeyError", "loop_overrides", "render_config"]


class UnverifiedKeyError(ValueError):
    """A configuration key that GNSS-SDR v0.0.22 does not read."""


@dataclass(frozen=True)
class ReceiverSpec:
    image: str  # repo@sha256:<digest>, or sha256:<image id> for a locally built image
    source_commit: str
    patch_id: str | None  # None for the stock build
    template: str  # "1C", "L5" or "1B": configs/gnss-sdr/<template>.conf
    overrides: Mapping[str, str] = field(default_factory=dict[str, str])

    def __post_init__(self) -> None:
        if not _PINNED.search(self.image):
            raise ValueError(f"image {self.image!r} is not pinned by a sha256 digest (D-04)")


@dataclass(frozen=True)
class SourceSpec:
    path: Path  # on the host; mounted read-only at /in in the container
    item_type: Literal["ibyte", "ishort", "gr_complex"]
    fs_hz: float
    implementation: Literal["File_Signal_Source", "Fifo_Signal_Source"] = "File_Signal_Source"


def _template(name: str, templates: Path) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in (templates / f"{name}.conf").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith((";", "#", "[")):
            continue
        key, _, value = line.partition("=")
        kv[key.strip()] = value.strip()
    return kv


def render_config(
    spec: ReceiverSpec, source: SourceSpec, channels: Mapping[int, Channel], templates: Path = TEMPLATES
) -> str:
    """The configuration text for one run; channels are {channel: (sys, signal, prn)}, all of the template's signal,
    and PRN 0 assigns no satellite.

    Order is fixed (template, source, channels, then overrides in key order, a template key being replaced in place),
    so equal inputs give equal text and equal hashes.
    """
    if sorted(channels) != list(range(len(channels))) or not channels:
        raise ValueError(f"channels must be numbered 0 to {len(channels) - 1}, got {sorted(channels)}")
    foreign = sorted({sig for _, sig, _ in channels.values()} - {spec.template})
    if foreign:
        raise ValueError(f"template {spec.template} cannot track signal(s) {foreign}")
    kv = _template(spec.template, templates)
    fs = str(round(source.fs_hz))
    kv["GNSS-SDR.internal_fs_sps"] = fs
    kv["SignalSource.implementation"] = source.implementation
    kv["SignalSource.filename"] = f"/in/{source.path.name}"
    if source.implementation == "Fifo_Signal_Source":
        kv["SignalSource.sample_type"] = source.item_type  # the FIFO reader's key; it always delivers gr_complex
        kv["DataTypeAdapter.implementation"] = "Pass_Through"
    else:
        kv["SignalSource.item_type"] = source.item_type
        kv["SignalSource.sampling_frequency"] = fs
        kv["SignalSource.samples"] = "0"
        kv["SignalSource.repeat"] = "false"
        kv["SignalSource.enable_throttle_control"] = "false"
        kv["DataTypeAdapter.implementation"] = _ADAPTER[source.item_type]
    kv[f"Channels_{spec.template}.count"] = str(len(channels))
    kv["Channels.in_acquisition"] = str(len(channels))
    for ch in sorted(channels):
        if channels[ch][2]:  # PRN 0 leaves the channel to the receiver's own search
            kv[f"Channel{ch}.satellite"] = str(channels[ch][2])
    for key in sorted(spec.overrides):
        kv[key] = spec.overrides[key]
    bad = unverified(kv)
    if bad:
        raise UnverifiedKeyError(f"GNSS-SDR v0.0.22 does not read: {', '.join(bad)}")
    foreign_roles = sorted(k for k in kv if (m := _SIGNAL_ROLE.match(k)) and m.group(2) != spec.template)
    if foreign_roles:  # the family's key is real, but this configuration has no block of that signal to read it
        raise UnverifiedKeyError(f"a {spec.template} configuration has no block for: {', '.join(foreign_roles)}")
    return "[GNSS-SDR]\n" + "".join(f"{k}={v}\n" for k, v in kv.items())


def loop_overrides(signal: str, target_bn_eff_hz: float, t_s: float) -> dict[str, str]:
    """Tracking keys that give the loop the target effective bandwidth at integration time t_s (D-21).

    At t_s equal to the code period the loop runs at `pll_bw_hz`. At a multiple N of it, correlation is extended over
    N symbols and `pll_bw_narrow_hz` applies (docs/9 section 2). The configured value comes from the inverse map of
    GNSS-SDR's recursion with its source timing (kappa = 1/2), which G-hz confirmed on the binary.
    """
    tc = CODE_PERIOD_S[signal]
    n = round(t_s / tc)
    if n < 1 or abs(n * tc - t_s) > 1e-9:
        raise ValueError(f"t_s = {t_s} s is not a multiple of the {signal} code period {tc} s")
    role = f"Tracking_{signal}"
    if n == 1:
        bn = CarrierLoop.gnsssdr_for_bn_eff(target_bn_eff_hz, t_s, kappa=0.5).bn_hz
        return {f"{role}.pll_bw_hz": f"{bn:.4f}"}
    bn = CarrierLoop.gnsssdr_for_bn_eff(target_bn_eff_hz, t_s, kappa=0.5, t_code_s=tc).bn_hz
    return {f"{role}.extend_correlation_symbols": str(n), f"{role}.pll_bw_narrow_hz": f"{bn:.4f}"}
