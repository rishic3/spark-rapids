"""pyarrow.compute implementation of ascii_ignore for use with @arrow_udf.

Mirrors the pandas implementation in ``ascii_ignore_impl.py`` but operates on
``pyarrow.Array`` directly, so we can feed a CPU Arrow UDF without bouncing
through pandas.
"""

import pyarrow as pa
import pyarrow.compute as pc


NON_ASCII_PATTERN = r"[^\x00-\x7f]"
STRIP_PATTERN = r"""[?()\-%=:&*"'><\\}{$@+/\]\[;~#,]"""


def ascii_ignore_arrow_impl(arr: pa.Array) -> pa.Array:
    arr = pc.replace_substring_regex(arr, pattern=NON_ASCII_PATTERN, replacement="")
    arr = pc.utf8_trim_whitespace(arr)
    arr = pc.replace_substring_regex(arr, pattern=STRIP_PATTERN, replacement="")
    return arr
