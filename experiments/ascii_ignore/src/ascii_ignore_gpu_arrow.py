"""GPU Arrow UDF for ascii_ignore.

Uses ``@arrow_udf`` (PySpark >= 4.0). Input/output is ``pyarrow.Array``, which
we convert directly to/from ``cudf.Series`` via Arrow's zero-copy bridge —
skipping the pandas round-trip used by the ``@pandas_udf`` variant.

Reuses ``ascii_ignore_cudf_impl`` for the actual string transformation.
"""

import time

import cudf
import pyarrow as pa
from pyspark.sql.functions import arrow_udf
from pyspark.sql.types import StringType

from ascii_ignore_gpu_impl import ascii_ignore_cudf_impl as _ascii_ignore_cudf_impl


@arrow_udf(StringType())
def ascii_ignore_gpu_arrow(arr: pa.Array) -> pa.Array:
    cudf_x = cudf.Series.from_arrow(arr)
    cudf_output = _ascii_ignore_cudf_impl(cudf_x)
    return cudf_output.to_arrow()


def make_timed_ascii_ignore_gpu_arrow(timing_acc):
    """Timed GPU Arrow UDF that appends H2D / compute / D2H metrics to timing_acc."""
    _warmed_up = [False]

    @arrow_udf(StringType())
    def timed_ascii_ignore_gpu_arrow(arr: pa.Array) -> pa.Array:
        is_cold = not _warmed_up[0]

        t0 = time.perf_counter()
        cudf_x = cudf.Series.from_arrow(arr)

        t1 = time.perf_counter()
        cudf_output = _ascii_ignore_cudf_impl(cudf_x)

        t2 = time.perf_counter()
        result = cudf_output.to_arrow()

        t3 = time.perf_counter()
        if is_cold:
            _warmed_up[0] = True
            timing_acc.add({
                "cold_init_plus_first_h2d_s": t1 - t0,
                "warm_h2d_s": 0.0,
                "warm_d2h_s": 0.0,
                "total_d2h_s": t3 - t2,
                "total_compute_s": t2 - t1,
                "batches": 1,
                "cold_batches": 1,
                "rows": len(arr),
            })
        else:
            timing_acc.add({
                "cold_init_plus_first_h2d_s": 0.0,
                "warm_h2d_s": t1 - t0,
                "warm_d2h_s": t3 - t2,
                "total_d2h_s": t3 - t2,
                "total_compute_s": t2 - t1,
                "batches": 1,
                "cold_batches": 0,
                "rows": len(arr),
            })
        return result

    return timed_ascii_ignore_gpu_arrow
