"""CPU (pandas) implementation for multi-column threshold status classification.

Inspired by the customer's ``get_status`` UDF (views.py) which scans many
telemetry fields against per-field thresholds and returns a status string.
With 10 float64 columns and 10 variable-length string columns, the Arrow
serialization cost is high, while the actual comparison logic is trivially
fast — a textbook case where transfer overhead dominates after a cuDF rewrite.
"""

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "cpu_pct", "gpu_pct", "mem_pct", "frame_time_ms", "rtt_ms",
    "packet_loss_pct", "gpu_temp_c", "gpu_mem_pct", "queue_depth",
    "thread_count",
]

STRING_COLUMNS = [
    "hostname", "region", "session_id", "device_model", "gpu_driver_ver",
    "os_info", "app_version", "error_message", "client_ip", "user_agent",
]

ALL_COLUMNS = NUMERIC_COLUMNS + STRING_COLUMNS

NUMERIC_WARNING = {
    "cpu_pct": 80.0, "gpu_pct": 85.0, "mem_pct": 80.0,
    "frame_time_ms": 33.0, "rtt_ms": 100.0,
    "packet_loss_pct": 1.0, "gpu_temp_c": 80.0, "gpu_mem_pct": 80.0,
    "queue_depth": 100.0, "thread_count": 200.0,
}

NUMERIC_CRITICAL = {
    "cpu_pct": 95.0, "gpu_pct": 98.0, "mem_pct": 95.0,
    "frame_time_ms": 66.0, "rtt_ms": 200.0,
    "packet_loss_pct": 5.0, "gpu_temp_c": 95.0, "gpu_mem_pct": 95.0,
    "queue_depth": 500.0, "thread_count": 500.0,
}


def metric_status_pandas_impl(df: pd.DataFrame) -> pd.Series:
    """Classify each row as normal / warning / critical."""
    n = len(df)
    any_warning = np.zeros(n, dtype=bool)
    any_critical = np.zeros(n, dtype=bool)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            vals = df[col].to_numpy()
            any_warning |= vals > NUMERIC_WARNING[col]
            any_critical |= vals > NUMERIC_CRITICAL[col]

    if "error_message" in df.columns:
        em = df["error_message"].fillna("")
        any_warning |= (em.str.len() > 0).to_numpy()
        any_critical |= em.str.contains("FATAL", na=False).to_numpy()

    if "hostname" in df.columns:
        any_critical |= (df["hostname"].fillna("").str.len() == 0).to_numpy()

    if "session_id" in df.columns:
        any_critical |= (df["session_id"].fillna("").str.len() == 0).to_numpy()

    if "client_ip" in df.columns:
        any_warning |= (df["client_ip"].fillna("").str.len() == 0).to_numpy()

    result = np.full(n, "normal", dtype=object)
    result[any_warning] = "warning"
    result[any_critical] = "critical"
    return pd.Series(result, index=df.index)
