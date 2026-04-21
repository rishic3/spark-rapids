"""CPU Arrow UDF for ascii_ignore.

Uses ``@arrow_udf`` (PySpark >= 4.0) so Spark delivers a ``pyarrow.Array``
directly to the worker function, avoiding the pandas round-trip that
``@pandas_udf`` imposes. Compute is done with ``pyarrow.compute``.
"""

import time

import pyarrow as pa
from pyspark.sql.functions import arrow_udf
from pyspark.sql.types import StringType

from ascii_ignore_arrow_impl import ascii_ignore_arrow_impl as _ascii_ignore_arrow_impl


@arrow_udf(StringType())
def ascii_ignore_arrow(arr: pa.Array) -> pa.Array:
    return _ascii_ignore_arrow_impl(arr)


def make_timed_ascii_ignore_arrow(timing_acc):
    """Timed CPU Arrow UDF that appends compute/counter metrics to timing_acc."""

    @arrow_udf(StringType())
    def timed_ascii_ignore_arrow(arr: pa.Array) -> pa.Array:
        t0 = time.perf_counter()
        result = _ascii_ignore_arrow_impl(arr)
        t1 = time.perf_counter()
        timing_acc.add({
            "total_compute_s": t1 - t0,
            "batches": 1,
            "rows": len(arr),
        })
        return result

    return timed_ascii_ignore_arrow
