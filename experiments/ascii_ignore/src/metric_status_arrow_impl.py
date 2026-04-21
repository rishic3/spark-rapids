"""pyarrow.compute implementation of metric_status for use with @arrow_udf.

Mirrors the pandas implementation in ``metric_status_impl.py``. The UDF
receives the 20-field struct column as a ``pa.StructArray`` and returns a
``pa.Array`` of status strings ("normal" / "warning" / "critical").
"""

import pyarrow as pa
import pyarrow.compute as pc

from metric_status_impl import (
    NUMERIC_COLUMNS,
    NUMERIC_CRITICAL,
    NUMERIC_WARNING,
)


def _field(struct_arr: pa.StructArray, name: str):
    """Return the child array for ``name`` if present, else None."""
    try:
        return struct_arr.field(name)
    except (KeyError, ValueError):
        return None


def metric_status_arrow_impl(struct_arr: pa.StructArray) -> pa.Array:
    n = len(struct_arr)
    any_warning = pa.array([False] * n, type=pa.bool_())
    any_critical = pa.array([False] * n, type=pa.bool_())

    for col in NUMERIC_COLUMNS:
        vals = _field(struct_arr, col)
        if vals is None:
            continue
        any_warning = pc.or_(any_warning, pc.greater(vals, NUMERIC_WARNING[col]))
        any_critical = pc.or_(any_critical, pc.greater(vals, NUMERIC_CRITICAL[col]))

    em = _field(struct_arr, "error_message")
    if em is not None:
        em_filled = pc.fill_null(em, "")
        any_warning = pc.or_(any_warning, pc.greater(pc.utf8_length(em_filled), 0))
        any_critical = pc.or_(any_critical, pc.match_substring(em_filled, "FATAL"))

    hn = _field(struct_arr, "hostname")
    if hn is not None:
        hn_filled = pc.fill_null(hn, "")
        any_critical = pc.or_(any_critical, pc.equal(pc.utf8_length(hn_filled), 0))

    sid = _field(struct_arr, "session_id")
    if sid is not None:
        sid_filled = pc.fill_null(sid, "")
        any_critical = pc.or_(any_critical, pc.equal(pc.utf8_length(sid_filled), 0))

    cip = _field(struct_arr, "client_ip")
    if cip is not None:
        cip_filled = pc.fill_null(cip, "")
        any_warning = pc.or_(any_warning, pc.equal(pc.utf8_length(cip_filled), 0))

    # case_when evaluates conditions in field order: critical wins over warning.
    cond = pc.make_struct(any_critical, any_warning, field_names=["c", "w"])
    return pc.case_when(
        cond,
        pa.array(["critical"] * n),
        pa.array(["warning"] * n),
        pa.array(["normal"] * n),
    )
