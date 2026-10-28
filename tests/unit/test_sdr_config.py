"""gnssaudit.sdr: configuration rendering, verified keys and the inverse map (docs/9 sections 2, 3 and 5.3)."""

from dataclasses import replace
from pathlib import Path

import pytest

from gnssaudit.sdr import ReceiverSpec, SourceSpec, UnverifiedKeyError, loop_overrides, render_config
from gnssaudit.sdr.keys import unverified

PINNED = "carlesfernandez/docker-gnsssdr:v0.0.22@sha256:" + "be" * 32
STOCK = ReceiverSpec(image=PINNED, source_commit="3fcb502a", patch_id=None, template="1C", overrides={})
TWIN = SourceSpec(path=Path("/dev/shm/gnssaudit/twin_a.bin"), item_type="ishort", fs_hz=4.0e6)
TWO = {0: ("G", "1C", 5), 1: ("G", "1C", 12)}


def parse(text: str) -> dict[str, str]:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith((";", "#", "["))]
    kv = dict(ln.split("=", 1) for ln in lines)
    assert len(kv) == len(lines), "a key is rendered twice"
    return kv


def test_rendered_configuration_points_the_receiver_at_the_twin() -> None:
    kv = parse(render_config(STOCK, TWIN, TWO))
    assert kv["SignalSource.implementation"] == "File_Signal_Source"
    assert kv["SignalSource.filename"] == "/in/twin_a.bin"
    assert (kv["SignalSource.item_type"], kv["DataTypeAdapter.implementation"]) == ("ishort", "Ishort_To_Complex")
    assert kv["SignalSource.sampling_frequency"] == "4000000" == kv["GNSS-SDR.internal_fs_sps"]
    assert (kv["SignalSource.repeat"], kv["SignalSource.samples"]) == ("false", "0")
    assert (kv["Channels_1C.count"], kv["Channel0.satellite"], kv["Channel1.satellite"]) == ("2", "5", "12")
    assert kv["Tracking_1C.dump"] == "true" and kv["Observables.dump"] == "true"
    assert kv["Tracking_1C.dump_mat"] == "false" and kv["Observables.dump_mat"] == "false"
    # the receiver time never moves once set (docs/9 section 3, item 5)
    assert kv["PVT.enable_rx_clock_correction"] == "false" and int(kv["PVT.max_clock_offset_ms"]) >= 10**6


@pytest.mark.parametrize(
    ("template", "channel"), [("1C", ("G", "1C", 3)), ("L5", ("G", "L5", 3)), ("1B", ("E", "1B", 11))]
)
def test_every_rendered_key_is_one_the_receiver_reads(template: str, channel: tuple[str, str, int]) -> None:
    text = render_config(replace(STOCK, template=template), TWIN, {0: channel})
    assert unverified(parse(text)) == []


def test_overrides_replace_values_and_render_deterministically() -> None:
    ov = {"Tracking_1C.pll_bw_hz": "5.8400", "Tracking_1C.cn0_samples": "100", "Acquisition_1C.blocking": "true"}
    a = render_config(replace(STOCK, overrides=ov), TWIN, TWO)
    b = render_config(replace(STOCK, overrides=dict(reversed(list(ov.items())))), TWIN, TWO)
    assert a == b
    kv = parse(a)
    assert (kv["Tracking_1C.pll_bw_hz"], kv["Tracking_1C.cn0_samples"], kv["Acquisition_1C.blocking"]) == (
        "5.8400",
        "100",
        "true",
    )
    ov2 = {"Acquisition_1C.doppler_max": "2000"}  # a template key: replaced, not duplicated (parse checks)
    assert parse(render_config(replace(STOCK, overrides=ov2), TWIN, TWO))["Acquisition_1C.doppler_max"] == "2000"


def test_a_misspelt_override_is_refused() -> None:
    with pytest.raises(UnverifiedKeyError, match="Tracking_1C.pll_bw_hzz"):
        render_config(replace(STOCK, overrides={"Tracking_1C.pll_bw_hzz": "5"}), TWIN, TWO)


def test_a_key_the_receiver_does_not_read_is_refused() -> None:
    with pytest.raises(UnverifiedKeyError, match="Channel.signal"):
        render_config(replace(STOCK, overrides={"Channel.signal": "1C"}), TWIN, TWO)


def test_a_channel_of_another_signal_is_refused() -> None:
    with pytest.raises(ValueError, match="1B"):
        render_config(STOCK, TWIN, {0: ("E", "1B", 11)})


def test_fifo_source_takes_sample_type_and_delivers_complex() -> None:
    fifo = SourceSpec(
        path=Path("/dev/shm/gnssaudit/twin.fifo"), item_type="ishort", fs_hz=4.0e6, implementation="Fifo_Signal_Source"
    )
    kv = parse(render_config(STOCK, fifo, TWO))
    assert kv["SignalSource.sample_type"] == "ishort" and "SignalSource.item_type" not in kv
    assert kv["DataTypeAdapter.implementation"] == "Pass_Through"


def test_an_unpinned_image_is_refused() -> None:
    with pytest.raises(ValueError, match="digest"):
        ReceiverSpec(
            image="carlesfernandez/docker-gnsssdr:v0.0.22",
            source_commit="3fcb502a",
            patch_id=None,
            template="1C",
            overrides={},
        )


@pytest.mark.parametrize(
    ("target", "t_s", "want"),
    [(5.0, 1e-3, {"pll_bw_hz": 5.84}), (15.0, 1e-3, {"pll_bw_hz": 17.47}),
     (5.0, 20e-3, {"pll_bw_narrow_hz": 4.50}), (15.0, 20e-3, {"pll_bw_narrow_hz": 9.66})],
)  # fmt: skip
def test_loop_overrides_follow_the_faithful_inverse_map(target: float, t_s: float, want: dict[str, float]) -> None:
    """docs/9 section 5.3, faithful map confirmed by G-hz on the binary (11 Oct 2026)."""
    ov = loop_overrides("1C", target, t_s)
    ((key, value),) = want.items()
    assert float(ov[f"Tracking_1C.{key}"]) == pytest.approx(value, rel=0.01)
    assert ov.get("Tracking_1C.extend_correlation_symbols", "1") == ("20" if t_s > 1e-3 else "1")
    assert unverified(ov) == []


def test_a_free_channel_has_no_satellite_assigned() -> None:
    """PRN 0 leaves the channel to the receiver's own search (used to survey which satellites are in view)."""
    kv = parse(render_config(STOCK, TWIN, {0: ("G", "1C", 0), 1: ("G", "1C", 0)}))
    assert "Channel0.satellite" not in kv and "Channel1.satellite" not in kv
    assert kv["Channels_1C.count"] == "2"


@pytest.mark.parametrize("key", ["Tracking_L5.pll_bw_hz", "Tracking_1B.pll_bw_hz", "Acquisition_5X.pfa",
                                 "TelemetryDecoder_L5.dump", "Channels_1B.count"])  # fmt: skip
def test_an_override_for_another_signal_role_is_refused(key: str) -> None:
    """Review finding (p2a, important 4): GNSS-SDR reads the key family for any signal role, but a 1C configuration
    has no L5 or E1 blocks, so such an override is ignored and gives a plausible default run."""
    with pytest.raises(UnverifiedKeyError, match=key.split(".")[0]):
        render_config(replace(STOCK, overrides={key: "1"}), TWIN, TWO)
