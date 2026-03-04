import time

import pandas as pd
from pyspark.sql.functions import pandas_udf, udf
from pyspark.sql.types import StringType

from ascii_ignore_impl import ascii_ignore_pandas_impl as _ascii_ignore_pandas_impl
from ascii_ignore_impl import ascii_ignore_impl as _ascii_ignore_impl

@udf(returnType=StringType())
def ascii_ignore(x):
    return _ascii_ignore_impl(x)


@pandas_udf(StringType())
def ascii_ignore_pandas(s: pd.Series) -> pd.Series:
    return _ascii_ignore_pandas_impl(s)


def make_timed_ascii_ignore_pandas(timing_acc):
    """Timed CPU pandas UDF that appends compute/counter metrics to timing_acc."""

    @pandas_udf(StringType())
    def timed_ascii_ignore_pandas(s: pd.Series) -> pd.Series:
        t0 = time.perf_counter()
        result = _ascii_ignore_pandas_impl(s)
        t1 = time.perf_counter()
        timing_acc.add({
            "total_compute_s": t1 - t0,
            "batches": 1,
            "rows": len(s),
        })
        return result

    return timed_ascii_ignore_pandas
