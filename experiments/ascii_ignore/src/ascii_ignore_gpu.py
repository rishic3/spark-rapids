import time

import cudf
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

from ascii_ignore_gpu_impl import ascii_ignore_cudf_impl as _ascii_ignore_cudf_impl


@pandas_udf(StringType())
def ascii_ignore_gpu(x: pd.Series) -> pd.Series:
    cudf_x = cudf.Series(x)
    cudf_output = _ascii_ignore_cudf_impl(cudf_x)
    return cudf_output.to_pandas()


def make_timed_ascii_ignore_gpu(timing_acc):
    """Timed version that appends to timing_acc Spark accumulator"""
    _warmed_up = [False]

    @pandas_udf(StringType())
    def timed_ascii_ignore_gpu(x: pd.Series) -> pd.Series:
        # Separating out first cuDF operation which does
        # cuInit context initialization and also resource/allocator setup
        is_cold = not _warmed_up[0]

        t0 = time.perf_counter()
        cudf_x = cudf.Series(x)

        t1 = time.perf_counter()
        cudf_output = _ascii_ignore_cudf_impl(cudf_x)

        t2 = time.perf_counter()
        result = cudf_output.to_pandas()

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
                "rows": len(x),
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
                "rows": len(x),
            })
        return result
    return timed_ascii_ignore_gpu
