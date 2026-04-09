"""CPU (pandas) implementation for PII redaction.

Replaces email addresses, IPv4 addresses, phone numbers, SSNs, and credit
card numbers with placeholder tokens.  Patterns are kept simple so they work
identically on both the Python ``re`` engine and the cuDF regex engine.
"""

import re

import pandas as pd

EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
IPV4_PATTERN = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
PHONE_PATTERN = r"\d{3}-\d{3}-\d{4}"
SSN_PATTERN = r"\d{3}-\d{2}-\d{4}"
CC_PATTERN = r"\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}"

# Order matters: more-specific patterns first so they aren't partially
# consumed by a less-specific one (e.g. SSN before phone).
_PATTERNS = [
    (re.compile(EMAIL_PATTERN), "[EMAIL]"),
    (re.compile(CC_PATTERN), "[CC]"),
    (re.compile(SSN_PATTERN), "[SSN]"),
    (re.compile(PHONE_PATTERN), "[PHONE]"),
    (re.compile(IPV4_PATTERN), "[IP]"),
]


def redact_pii_pandas_impl(s: pd.Series) -> pd.Series:
    result = s
    for pattern, replacement in _PATTERNS:
        result = result.str.replace(pattern, replacement, regex=True)
    return result
