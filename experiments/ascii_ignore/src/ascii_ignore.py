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
