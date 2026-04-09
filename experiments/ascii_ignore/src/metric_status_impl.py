"""CPU (pandas) implementation for multi-column threshold status classification.

Inspired by the customer's ``get_status`` UDF (views.py) which scans many
telemetry fields against per-field thresholds and returns a status string.
With 20 float64 columns the Arrow serialization cost is high, while the
actual comparison logic is trivially fast — a textbook case where transfer
overhead dominates after a cuDF rewrite.
"""

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "cpu_pct", "gpu_pct", "mem_pct", "disk_io_pct", "net_bw_pct",
    "frame_time_ms", "encode_latency_ms", "decode_latency_ms", "rtt_ms",
    "jitter_ms", "packet_loss_pct", "input_latency_ms",
    "gpu_temp_c", "gpu_mem_pct", "cpu_temp_c", "swap_pct",
    "queue_depth", "error_rate_pct", "retry_pct", "thread_count",
]

WARNING_THRESHOLDS = {
    "cpu_pct": 80.0, "gpu_pct": 85.0, "mem_pct": 80.0,
    "disk_io_pct": 80.0, "net_bw_pct": 80.0,
    "frame_time_ms": 33.0, "encode_latency_ms": 20.0,
    "decode_latency_ms": 20.0, "rtt_ms": 100.0,
    "jitter_ms": 20.0, "packet_loss_pct": 1.0,
    "input_latency_ms": 50.0,
    "gpu_temp_c": 80.0, "gpu_mem_pct": 80.0, "cpu_temp_c": 80.0,
    "swap_pct": 50.0, "queue_depth": 100.0, "error_rate_pct": 1.0,
    "retry_pct": 5.0, "thread_count": 200.0,
}

CRITICAL_THRESHOLDS = {
    "cpu_pct": 95.0, "gpu_pct": 98.0, "mem_pct": 95.0,
    "disk_io_pct": 95.0, "net_bw_pct": 95.0,
    "frame_time_ms": 66.0, "encode_latency_ms": 50.0,
    "decode_latency_ms": 50.0, "rtt_ms": 200.0,
    "jitter_ms": 50.0, "packet_loss_pct": 5.0,
    "input_latency_ms": 100.0,
    "gpu_temp_c": 95.0, "gpu_mem_pct": 95.0, "cpu_temp_c": 95.0,
    "swap_pct": 80.0, "queue_depth": 500.0, "error_rate_pct": 5.0,
    "retry_pct": 15.0, "thread_count": 500.0,
}


def metric_status_pandas_impl(df: pd.DataFrame) -> pd.Series:
    """Classify each row as normal / warning / critical."""
    n = len(df)
    any_warning = np.zeros(n, dtype=bool)
    any_critical = np.zeros(n, dtype=bool)

    for col in METRIC_COLUMNS:
        if col in df.columns:
            vals = df[col].to_numpy()
            any_warning |= vals > WARNING_THRESHOLDS[col]
            any_critical |= vals > CRITICAL_THRESHOLDS[col]

    result = np.full(n, "normal", dtype=object)
    result[any_warning] = "warning"
    result[any_critical] = "critical"
    return pd.Series(result, index=df.index)
