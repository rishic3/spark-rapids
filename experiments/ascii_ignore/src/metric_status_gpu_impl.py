"""cuDF (GPU) implementation for multi-column threshold status classification.

Uses vectorised cuDF comparisons across all 20 metric columns.  The compute
itself is near-instantaneous on GPU, making this UDF ideal for demonstrating
transfer-dominated overhead.
"""

import numpy as np

import cudf

from metric_status_impl import (
    CRITICAL_THRESHOLDS,
    METRIC_COLUMNS,
    WARNING_THRESHOLDS,
)


def metric_status_cudf_impl(df: cudf.DataFrame) -> cudf.Series:
    """Classify each row as normal / warning / critical."""
    n = len(df)
    any_warning = cudf.Series(np.zeros(n, dtype=bool))
    any_critical = cudf.Series(np.zeros(n, dtype=bool))

    for col in METRIC_COLUMNS:
        if col in df.columns:
            any_warning = any_warning | (df[col] > WARNING_THRESHOLDS[col])
            any_critical = any_critical | (df[col] > CRITICAL_THRESHOLDS[col])

    result = cudf.Series(["normal"]).repeat(n).reset_index(drop=True)
    result = result.where(~any_warning, "warning")
    result = result.where(~any_critical, "critical")
    return result
