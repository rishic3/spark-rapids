import time

import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

from redact_pii_impl import redact_pii_pandas_impl as _redact_pii_pandas_impl


@pandas_udf(StringType())
def redact_pii_pandas(s: pd.Series) -> pd.Series:
    return _redact_pii_pandas_impl(s)


def make_timed_redact_pii_pandas(timing_acc):
    """Timed CPU pandas UDF that appends compute/counter metrics to timing_acc."""

    @pandas_udf(StringType())
    def timed_redact_pii_pandas(s: pd.Series) -> pd.Series:
        t0 = time.perf_counter()
        result = _redact_pii_pandas_impl(s)
        t1 = time.perf_counter()
        timing_acc.add({
            "total_compute_s": t1 - t0,
            "batches": 1,
            "rows": len(s),
        })
        return result

    return timed_redact_pii_pandas
