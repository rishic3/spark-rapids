"""cuDF (GPU) implementation for PII redaction.

Uses the same regex patterns as the CPU implementation so that the two
produce identical output.  cuDF's GPU regex engine makes each replacement
extremely fast, making the Arrow transfer overhead the dominant cost.
"""

import cudf

from redact_pii_impl import (
    CC_PATTERN,
    EMAIL_PATTERN,
    IPV4_PATTERN,
    PHONE_PATTERN,
    SSN_PATTERN,
)

_PATTERNS = [
    (EMAIL_PATTERN, "[EMAIL]"),
    (CC_PATTERN, "[CC]"),
    (SSN_PATTERN, "[SSN]"),
    (PHONE_PATTERN, "[PHONE]"),
    (IPV4_PATTERN, "[IP]"),
]


def redact_pii_cudf_impl(s: cudf.Series) -> cudf.Series:
    for pattern, replacement in _PATTERNS:
        s = s.str.replace(pattern, replacement, regex=True)
    return s
