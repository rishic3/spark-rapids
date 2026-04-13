"""cuDF (GPU) implementation for multi-column threshold status classification.

Uses vectorised cuDF comparisons across 10 numeric and 10 string columns.
The compute itself is near-instantaneous on GPU, making this UDF ideal for
demonstrating transfer-dominated overhead.
"""

import numpy as np

import cudf

from metric_status_impl import (
    NUMERIC_COLUMNS,
    NUMERIC_CRITICAL,
    NUMERIC_WARNING,
)


def metric_status_cudf_impl(df: cudf.DataFrame) -> cudf.Series:
    """Classify each row as normal / warning / critical."""
    n = len(df)
    any_warning = cudf.Series(np.zeros(n, dtype=bool))
    any_critical = cudf.Series(np.zeros(n, dtype=bool))

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            any_warning = any_warning | (df[col] > NUMERIC_WARNING[col])
            any_critical = any_critical | (df[col] > NUMERIC_CRITICAL[col])

    if "error_message" in df.columns:
        em = df["error_message"].fillna("")
        any_warning = any_warning | (em.str.len() > 0)
        any_critical = any_critical | em.str.contains("FATAL")

    if "hostname" in df.columns:
        any_critical = any_critical | (df["hostname"].fillna("").str.len() == 0)

    if "session_id" in df.columns:
        any_critical = any_critical | (df["session_id"].fillna("").str.len() == 0)

    if "client_ip" in df.columns:
        any_warning = any_warning | (df["client_ip"].fillna("").str.len() == 0)

    result = cudf.Series(["normal"]).repeat(n).reset_index(drop=True)
    result = result.where(~any_warning, "warning")
    result = result.where(~any_critical, "critical")
    return result
