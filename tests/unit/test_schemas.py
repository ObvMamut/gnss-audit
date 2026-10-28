"""G-schema: tables are validated at every boundary and never coerced (docs/6 section 4.3, docs/7 section 5)."""

import numpy as np
import pyarrow as pa
import pytest

from gnssaudit.io.schemas import EXCLUSIONS, PREDICTION, SchemaError, build, validate


def prediction_columns(n: int) -> dict[str, object]:
    return {
        "model_id": ["B0"] * n,
        "model_version": ["1"] * n,
        "run_id": ["r"] * n,
        "sys": ["G"] * n,
        "prn": np.arange(n),
        "signal": ["1C"] * n,
        "t_rx_s": np.linspace(0.0, 1.0, n),
        "scale_cyc": np.full(n, 0.01),
        "nu": np.zeros(n),
        "p_slip": None,
        "valid": np.ones(n, dtype=bool),
        "flags": np.zeros(n, dtype=np.int64),
    }


def test_build_casts_to_schema_types_and_fills_nulls() -> None:
    table = build(PREDICTION, prediction_columns(3), 3)
    assert table.schema.equals(PREDICTION, check_metadata=False)
    assert table.column("p_slip").null_count == 3
    assert table.column("prn").type == pa.int16()


def test_build_turns_nan_into_null() -> None:
    cols = prediction_columns(2)
    cols["scale_cyc"] = np.array([0.01, np.nan])
    assert build(PREDICTION, cols, 2).column("scale_cyc").null_count == 1


def test_build_rejects_missing_unexpected_and_wrong_length() -> None:
    cols = prediction_columns(2)
    with pytest.raises(SchemaError, match="missing"):
        build(PREDICTION, {k: v for k, v in cols.items() if k != "nu"}, 2)
    with pytest.raises(SchemaError, match="unexpected"):
        build(PREDICTION, {**cols, "extra": [1, 2]}, 2)
    with pytest.raises(SchemaError, match="rows"):
        build(PREDICTION, cols, 3)


def test_validate_reorders_and_rejects_mismatches() -> None:
    table = build(PREDICTION, prediction_columns(2), 2)
    shuffled = table.select(list(reversed(table.column_names)))
    assert validate(shuffled, PREDICTION).column_names == PREDICTION.names
    with pytest.raises(SchemaError, match="missing scale_cyc"):
        validate(table.drop_columns(["scale_cyc"]), PREDICTION)
    with pytest.raises(SchemaError, match="unexpected extra"):
        validate(table.append_column("extra", pa.array([1, 2])), PREDICTION)
    wrong = table.set_column(table.schema.get_field_index("prn"), "prn", pa.array([1, 2], type=pa.int64()))
    with pytest.raises(SchemaError, match="prn is int64, expected int16"):
        validate(wrong, PREDICTION)
    with pytest.raises(SchemaError, match="^exclusions"):
        validate(table, EXCLUSIONS)


# docs/7 sections 5.2, 5.4 and 5.6, transcribed by hand: (name, type) in the documented order
TRACK_STREAM_SPEC = [
    ("run_id", pa.string()), ("channel", pa.int16()), ("sys", pa.string()), ("prn", pa.int16()),
    ("signal", pa.string()), ("k", pa.int64()), ("sample_count", pa.uint64()), ("t_rx_s", pa.float64()),
    ("abs_ve", pa.float32()), ("abs_e", pa.float32()), ("abs_p", pa.float32()), ("abs_l", pa.float32()),
    ("abs_vl", pa.float32()), ("prompt_i", pa.float32()), ("prompt_q", pa.float32()),
    ("acc_phase_rad_f32", pa.float32()), ("acc_phase_rad", pa.float64()), ("dop_hz", pa.float32()),
    ("dop_rate_hzps", pa.float32()), ("code_freq_chips", pa.float32()), ("code_rate_chipsps2", pa.float32()),
    ("carr_err", pa.float32()), ("carr_err_filt", pa.float32()), ("code_err_chips", pa.float32()),
    ("code_err_filt", pa.float32()), ("cn0_dbhz", pa.float32()), ("lock_test", pa.float32()),
    ("tow_ms", pa.uint64()), ("wn", pa.uint32()), ("t_int_s", pa.float32()),
]  # fmt: skip
ORACLE_MOMENTS_SPEC = [
    ("config_id", pa.string()), ("form", pa.string()), ("sys", pa.string()), ("prn", pa.int16()),
    ("signal", pa.string()), ("t_rx_s", pa.float64()), ("K", pa.int16()), ("n_valid", pa.int16()),
    ("mu_cyc", pa.float64()), ("var_cyc2", pa.float64()), ("p_slip", pa.float32()), ("slip_window_s", pa.float32()),
]  # fmt: skip
EVENTS_SPEC = [
    ("event_id", pa.string()), ("source_id", pa.string()), ("kind", pa.string()), ("sys", pa.string()),
    ("prn", pa.int16()), ("signal", pa.string()), ("t_start_s", pa.float64()), ("t_end_s", pa.float64()),
    ("size_cyc", pa.float32()), ("confidence", pa.float32()), ("origin", pa.string()),
]  # fmt: skip


def _literal_table(spec: list[tuple[str, pa.DataType]], n: int) -> pa.Table:
    """A table typed exactly as the note specifies, with n rows of nulls (the types are what is checked)."""
    return pa.table({name: pa.nulls(n, type=typ) for name, typ in spec})


@pytest.mark.parametrize("name", ["TRACK_STREAM", "ORACLE_MOMENTS", "EVENTS"])
def test_documented_twin_path_tables_validate(name: str) -> None:
    from gnssaudit.io import schemas

    spec = {"TRACK_STREAM": TRACK_STREAM_SPEC, "ORACLE_MOMENTS": ORACLE_MOMENTS_SPEC, "EVENTS": EVENTS_SPEC}[name]
    schema = getattr(schemas, name)
    shuffled = _literal_table(list(reversed(spec)), 2)
    assert validate(shuffled, schema).column_names == [n for n, _ in spec]


def test_float32_accumulated_phase_cannot_enter_a_track_stream() -> None:
    """G-phase-precision (docs/3 section 2.6): the truth-grade phase column is float64 by construction."""
    from gnssaudit.io.schemas import TRACK_STREAM

    spec = [(n, pa.float32() if n == "acc_phase_rad" else t) for n, t in TRACK_STREAM_SPEC]
    with pytest.raises(SchemaError, match="acc_phase_rad is float, expected double"):
        validate(_literal_table(spec, 2), TRACK_STREAM)
