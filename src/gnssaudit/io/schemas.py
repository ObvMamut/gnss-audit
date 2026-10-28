"""Canonical table schemas and their validation (docs/7 section 5; gate G-schema, docs/11 section 5).

Tables are checked at every boundary. A mismatch raises SchemaError; nothing is coerced (docs/6 section 4.3).
"""

from __future__ import annotations

from collections.abc import Mapping

import pyarrow as pa


class SchemaError(ValueError):
    """A table does not match its canonical schema."""


def _schema(name: str, fields: list[tuple[str, pa.DataType]]) -> pa.Schema:
    return pa.schema(fields, metadata={"table": name})


OBS_EPOCH_SAT = _schema(
    "obs_epoch_sat",
    [
        ("source_id", pa.string()),
        ("receiver_id", pa.string()),
        ("t_gps_s", pa.float64()),
        ("sys", pa.string()),
        ("prn", pa.int16()),
        ("signal", pa.string()),
        ("pr_m", pa.float64()),
        ("cp_cyc", pa.float64()),
        ("dop_hz", pa.float64()),
        ("cn0_dbhz", pa.float32()),
        ("lock_time_s", pa.float32()),
        ("lli", pa.uint8()),
        ("trk_stat", pa.uint8()),
        ("cp_std_code", pa.uint8()),
        ("cp_std_cyc", pa.float32()),
        ("censored", pa.bool_()),
        ("pr_std_m", pa.float32()),
        ("elev_deg", pa.float32()),
        ("azim_deg", pa.float32()),
        ("valid", pa.bool_()),
    ],
)

TRUTH_EPOCH_SAT = _schema(
    "truth_epoch_sat",
    [
        ("truth_id", pa.string()),
        ("kind", pa.string()),
        ("sys", pa.string()),
        ("prn", pa.int16()),
        ("signal", pa.string()),
        ("t_rx_s", pa.float64()),
        ("e_cyc", pa.float64()),
        ("slip_count", pa.int16()),
        ("arc_id", pa.int32()),
        ("transient", pa.bool_()),
        ("valid", pa.bool_()),
        ("regime", pa.uint8()),
    ],
)

TRACK_STREAM = _schema(
    "track_stream",
    [
        ("run_id", pa.string()),
        ("channel", pa.int16()),
        ("sys", pa.string()),
        ("prn", pa.int16()),
        ("signal", pa.string()),
        ("k", pa.int64()),
        ("sample_count", pa.uint64()),
        ("t_rx_s", pa.float64()),
        ("abs_ve", pa.float32()),
        ("abs_e", pa.float32()),
        ("abs_p", pa.float32()),
        ("abs_l", pa.float32()),
        ("abs_vl", pa.float32()),
        ("prompt_i", pa.float32()),
        ("prompt_q", pa.float32()),
        ("acc_phase_rad_f32", pa.float32()),  # stock dump: not usable as truth (D-08)
        ("acc_phase_rad", pa.float64()),  # patched build only, else null (G-phase-precision)
        ("dop_hz", pa.float32()),
        ("dop_rate_hzps", pa.float32()),
        ("code_freq_chips", pa.float32()),
        ("code_rate_chipsps2", pa.float32()),
        ("carr_err", pa.float32()),
        ("carr_err_filt", pa.float32()),
        ("code_err_chips", pa.float32()),
        ("code_err_filt", pa.float32()),
        ("cn0_dbhz", pa.float32()),
        ("lock_test", pa.float32()),
        ("tow_ms", pa.uint64()),
        ("wn", pa.uint32()),
        ("t_int_s", pa.float32()),
    ],
)

ORACLE_MOMENTS = _schema(
    "oracle_moments",
    [
        ("config_id", pa.string()),
        ("form", pa.string()),  # conditional (jammer timing fixed) | marginal (timing drawn), docs/3 section 2.4 (F8)
        ("sys", pa.string()),
        ("prn", pa.int16()),
        ("signal", pa.string()),
        ("t_rx_s", pa.float64()),
        ("K", pa.int16()),
        ("n_valid", pa.int16()),
        ("mu_cyc", pa.float64()),
        ("var_cyc2", pa.float64()),
        ("p_slip", pa.float32()),
        ("slip_window_s", pa.float32()),
    ],
)

EVENTS = _schema(
    "events",
    [
        ("event_id", pa.string()),
        ("source_id", pa.string()),
        ("kind", pa.string()),  # slip | onset | offset | interference_window | gate_fail
        ("sys", pa.string()),  # sys, prn and signal are null for receiver-wide events
        ("prn", pa.int16()),
        ("signal", pa.string()),
        ("t_start_s", pa.float64()),
        ("t_end_s", pa.float64()),
        ("size_cyc", pa.float32()),
        ("confidence", pa.float32()),
        ("origin", pa.string()),  # truth | plan | injection | receiver_flag
    ],
)

PREDICTION = _schema(
    "prediction",
    [
        ("model_id", pa.string()),
        ("model_version", pa.string()),
        ("run_id", pa.string()),
        ("sys", pa.string()),
        ("prn", pa.int16()),
        ("signal", pa.string()),
        ("t_rx_s", pa.float64()),
        ("scale_cyc", pa.float64()),
        ("nu", pa.float32()),
        ("p_slip", pa.float32()),
        ("valid", pa.bool_()),
        ("flags", pa.uint32()),
    ],
)

EXCLUSIONS = _schema(
    "exclusions",
    [
        ("stage", pa.string()),
        ("source_id", pa.string()),
        ("sys", pa.string()),
        ("prn", pa.int16()),
        ("signal", pa.string()),
        ("t_start_s", pa.float64()),
        ("t_end_s", pa.float64()),
        ("reason_code", pa.string()),
        ("count", pa.int64()),
        ("detail", pa.string()),
    ],
)

RF_EPOCH = _schema(
    "rf_epoch",
    [
        ("t_gps_s", pa.float64()),
        ("block", pa.uint8()),
        ("agc_cnt", pa.int32()),
        ("jam_ind", pa.uint8()),
        ("jamming_state", pa.uint8()),
        ("noise_per_ms", pa.int32()),
    ],
)

PVT_EPOCH = _schema(
    "pvt_epoch",
    [
        ("t_gps_s", pa.float64()),
        ("lat_deg", pa.float64()),
        ("lon_deg", pa.float64()),
        ("height_m", pa.float64()),
        ("h_acc_m", pa.float32()),
        ("v_acc_m", pa.float32()),
        ("fix_type", pa.uint8()),
        ("num_sv", pa.uint8()),
    ],
)


def _name(schema: pa.Schema) -> str:
    meta = schema.metadata or {}
    return meta.get(b"table", b"?").decode()


def validate(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Return `table` with its columns in schema order; raise SchemaError on a missing, extra or mistyped column."""
    have = {f.name: f.type for f in table.schema}
    want = {f.name: f.type for f in schema}
    problems = [f"missing {n}" for n in want if n not in have]
    problems += [f"unexpected {n}" for n in have if n not in want]
    problems += [f"{n} is {have[n]}, expected {want[n]}" for n in want if n in have and have[n] != want[n]]
    if problems:
        raise SchemaError(f"{_name(schema)}: " + "; ".join(problems))
    return pa.Table.from_arrays([table.column(n) for n in schema.names], schema=schema)


def build(schema: pa.Schema, columns: Mapping[str, object], n_rows: int) -> pa.Table:
    """Build a table of `schema` from array-likes, one per column; None makes an all-null column.

    Inputs are cast to the column type without range checks (float64 to float32 rounds), and NaN becomes null.
    """
    unexpected = sorted(set(columns) - set(schema.names))
    missing = [n for n in schema.names if n not in columns]
    if unexpected or missing:
        raise SchemaError(f"{_name(schema)}: missing {missing}, unexpected {unexpected}")
    arrays = []
    for field in schema:
        value = columns[field.name]
        if value is None:
            arrays.append(pa.nulls(n_rows, type=field.type))
            continue
        arr = pa.array(value, from_pandas=True)  # from_pandas: NaN becomes null
        if arr.type != field.type:
            arr = arr.cast(field.type, safe=False)
        if len(arr) != n_rows:
            raise SchemaError(f"{_name(schema)}: column {field.name} has {len(arr)} rows, expected {n_rows}")
        arrays.append(arr)
    return pa.Table.from_arrays(arrays, schema=schema)
