from typing import cast

import cudf


def ascii_ignore_cudf_impl(cudf_x: cudf.Series) -> cudf.Series:
    # Convert empty strings to null to match CPU `if x:` guard (empty string is falsy)
    cudf_x = cast(cudf.Series, cudf_x.where(cudf_x.str.len() > 0))
    # Remove non-ASCII characters (keep only code points 0-127)
    cudf_x = cast(cudf.Series, cudf_x.str.filter_characters({chr(0): chr(127)}, keep=True))
    # Strip leading/trailing whitespace
    cudf_x = cast(cudf.Series, cudf_x.str.strip())
    # Remove 25 special characters in a single regex replace
    cudf_x = cast(
        cudf.Series,
        cudf_x.str.replace(r"""[]?)(%=:&*"'><}{$@+/\[;~#,\\]""", "", regex=True),
    )
    return cudf_x
