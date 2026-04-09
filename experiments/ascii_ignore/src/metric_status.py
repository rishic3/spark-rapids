import time

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

from metric_status_impl import metric_status_pandas_impl as _metric_status_pandas_impl


@pandas_udf(StringType())
def metric_status_pandas(metrics: pd.DataFrame) -> pd.Series:
    return _metric_status_pandas_impl(metrics)


def make_timed_metric_status_pandas(timing_acc):
    """Timed CPU pandas UDF that appends compute/counter metrics to timing_acc."""

    @pandas_udf(StringType())
    def timed_metric_status_pandas(metrics: pd.DataFrame) -> pd.Series:
        t0 = time.perf_counter()
        result = _metric_status_pandas_impl(metrics)
        t1 = time.perf_counter()
        timing_acc.add({
            "total_compute_s": t1 - t0,
            "batches": 1,
            "rows": len(metrics),
        })
        return result

    return timed_metric_status_pandas
