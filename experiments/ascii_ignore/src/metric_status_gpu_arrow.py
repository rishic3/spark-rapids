"""GPU Arrow UDF for metric_status.

Uses ``@arrow_udf`` (PySpark >= 4.1). Input is a ``pyarrow.StructArray`` with
20 fields (the concrete runtime type; the parameter must be annotated as the
base ``pa.Array`` so PySpark's arrow_udf type-hint inference accepts it).
We convert directly to ``cudf.DataFrame`` via Arrow (bypassing pandas), run
the existing cuDF implementation, then return the result as a ``pa.Array``
via ``cudf.Series.to_arrow()``.
"""

import time

import cudf
import pyarrow as pa
from pyspark.sql.functions import arrow_udf
from pyspark.sql.types import StringType

from metric_status_gpu_impl import metric_status_cudf_impl as _metric_status_cudf_impl


def _struct_array_to_cudf_df(struct_arr: pa.StructArray) -> cudf.DataFrame:
    """Convert a ``pa.StructArray`` into a ``cudf.DataFrame`` without pandas."""
    struct_type = struct_arr.type
    field_names = [struct_type.field(i).name for i in range(struct_type.num_fields)]
    child_arrays = [struct_arr.field(i) for i in range(struct_type.num_fields)]
    tbl = pa.Table.from_arrays(child_arrays, names=field_names)
    return cudf.DataFrame.from_arrow(tbl)


@arrow_udf(StringType())
def metric_status_gpu_arrow(metrics: pa.Array) -> pa.Array:
    cudf_df = _struct_array_to_cudf_df(metrics)
    cudf_output = _metric_status_cudf_impl(cudf_df)
    return cudf_output.to_arrow()


def make_timed_metric_status_gpu_arrow(timing_acc):
    """Timed GPU Arrow UDF that appends H2D / compute / D2H metrics to timing_acc."""
    _warmed_up = [False]

    @arrow_udf(StringType())
    def timed_metric_status_gpu_arrow(metrics: pa.Array) -> pa.Array:
        is_cold = not _warmed_up[0]

        t0 = time.perf_counter()
        cudf_df = _struct_array_to_cudf_df(metrics)

        t1 = time.perf_counter()
        cudf_output = _metric_status_cudf_impl(cudf_df)

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
                "rows": len(metrics),
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
                "rows": len(metrics),
            })
        return result

    return timed_metric_status_gpu_arrow
