"""CPU Arrow UDF for metric_status.

Uses ``@arrow_udf`` (PySpark >= 4.1). The 20-column struct arrives as a
``pyarrow.StructArray`` (a subclass of ``pa.Array``) and is processed with
``pyarrow.compute``. Note: the parameter annotation must be exactly
``pa.Array`` — PySpark's arrow_udf type-hint inference does not accept
``pa.StructArray`` even though that's the concrete runtime type.
"""

import time

import pyarrow as pa
from pyspark.sql.functions import arrow_udf
from pyspark.sql.types import StringType

from metric_status_arrow_impl import metric_status_arrow_impl as _metric_status_arrow_impl


@arrow_udf(StringType())
def metric_status_arrow(metrics: pa.Array) -> pa.Array:
    return _metric_status_arrow_impl(metrics)


def make_timed_metric_status_arrow(timing_acc):
    """Timed CPU Arrow UDF that appends compute/counter metrics to timing_acc."""

    @arrow_udf(StringType())
    def timed_metric_status_arrow(metrics: pa.Array) -> pa.Array:
        t0 = time.perf_counter()
        result = _metric_status_arrow_impl(metrics)
        t1 = time.perf_counter()
        timing_acc.add({
            "total_compute_s": t1 - t0,
            "batches": 1,
            "rows": len(metrics),
        })
        return result

    return timed_metric_status_arrow
