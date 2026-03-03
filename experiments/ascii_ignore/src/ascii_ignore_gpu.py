import cudf
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType


def gpu_udf_impl(cudf_x: cudf.Series) -> cudf.Series:
    # Mark original nulls and empty strings (falsy in Python)
    is_falsy = cudf_x.isna() | (cudf_x.str.len() == 0)

    # Fill nulls with empty string so we can process uniformly
    s = cudf_x.fillna('')

    # Combined: remove non-ASCII chars AND specified special chars in one pass.
    # Instead of matching chars to remove, define chars to KEEP and remove everything else.
    # Characters to KEEP (ASCII chars NOT in the special removal set):
    #   tab(\t), newline(\n), carriage return(\r), space, !, -, ., 0-9, A-Z,
    #   ^, _, `, a-z, |
    # Note: - is placed at end of character class to be literal.
    # Note: ^ after [^ is negation; a second ^ is literal.
    # This removes both non-ASCII (>127) and the specified special ASCII chars.
    remove_pattern = r'[^\t\n\r 0-9A-Za-z!._`|^-]'
    s = s.str.replace(remove_pattern, '', regex=True)

    # Strip leading/trailing whitespace
    s = s.str.strip()

    # Set originally falsy values back to null
    s = s.where(~is_falsy, other=None)

    return s

def gpu_udf_impl_optimized(cudf_x: cudf.Series) -> cudf.Series:
    # Convert empty strings to null to match CPU `if x:` guard (empty string is falsy)
    cudf_x = cudf_x.where(cudf_x.str.len() > 0)
    # Remove non-ASCII characters (keep only code points 0-127)
    cudf_x = cudf_x.str.filter_characters({chr(0): chr(127)}, keep=True)
    # Strip leading/trailing whitespace
    cudf_x = cudf_x.str.strip()
    # Remove 25 special characters in a single regex replace
    cudf_x = cudf_x.str.replace(r"""[]?)(%=:&*"'><}{$@+/\[;~#,\\]""", "", regex=True)
    return cudf_x


@pandas_udf(StringType())
def ascii_ignore_gpu(x: pd.Series) -> pd.Series:
    cudf_x = cudf.Series(x)
    # cudf_output = gpu_udf_impl(cudf_x)
    cudf_output = gpu_udf_impl_optimized(cudf_x)
    return cudf_output.to_pandas()
