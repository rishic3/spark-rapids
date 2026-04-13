"""Synthetic data generation for the ``metric_status`` benchmark.

Generates 10 float64 telemetry columns and 10 variable-length string columns
with realistic distributions.  The mixed schema is intentionally wide so that
Arrow serialization (especially of strings) dominates over the trivial
threshold-comparison compute.
"""

import random
import string
import uuid
from typing import Callable, Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, IntegerType, StringType

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

_NUMERIC_DISTRIBUTIONS = {
    "cpu_pct":         (55.0, 25.0, 0.0, 100.0),
    "gpu_pct":         (60.0, 25.0, 0.0, 100.0),
    "mem_pct":         (50.0, 20.0, 0.0, 100.0),
    "frame_time_ms":   (16.0, 15.0, 1.0, 200.0),
    "rtt_ms":          (50.0, 60.0, 1.0, 500.0),
    "packet_loss_pct": (0.2, 1.5, 0.0, 30.0),
    "gpu_temp_c":      (65.0, 15.0, 30.0, 105.0),
    "gpu_mem_pct":     (50.0, 25.0, 0.0, 100.0),
    "queue_depth":     (20.0, 80.0, 0.0, 1000.0),
    "thread_count":    (80.0, 60.0, 1.0, 1000.0),
}

_HOSTNAMES = [
    "gpu-worker-{id:03d}.us-east-1.compute.internal",
    "ml-node-{id:03d}.eu-west-2.gpu-cluster.local",
    "render-{id:03d}.asia-southeast-1.prod.nvidia.com",
    "infer-server-{id:03d}.us-west-2.sagemaker.aws",
    "gfn-edge-{id:03d}.eu-central-1.streaming.local",
]

_REGIONS = [
    "us-east-1", "us-west-2", "eu-west-1", "eu-west-2", "eu-central-1",
    "ap-southeast-1", "ap-northeast-1", "ap-south-1", "sa-east-1",
    "us-east-2", "ca-central-1", "eu-north-1",
]

_DEVICE_MODELS = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 4080 SUPER",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA L40S",
    "NVIDIA Tesla T4",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA A10G",
    "NVIDIA H100 PCIe",
]

_DRIVER_VERSIONS = [
    "535.129.03", "535.161.07", "545.23.08", "550.54.14",
    "550.90.07", "555.42.02", "555.58.02", "560.35.03",
]

_OS_VERSIONS = [
    "Ubuntu 22.04.3 LTS x86_64",
    "Ubuntu 22.04.4 LTS x86_64",
    "Ubuntu 20.04.6 LTS x86_64",
    "Rocky Linux 9.3 x86_64",
    "Amazon Linux 2023.4 x86_64",
    "Debian GNU/Linux 12 (bookworm) x86_64",
    "RHEL 8.9 x86_64",
]

_APP_VERSIONS = [
    "2.14.3-beta.1", "2.14.2", "2.13.8", "2.13.7-rc.2",
    "2.12.5", "2.12.4", "2.11.9", "3.0.0-alpha.4",
]

_ERROR_MESSAGES = [
    "",  # most rows have no error
    "Connection timeout after 30000ms waiting for response from streaming server",
    "GPU memory allocation failed: out of memory (requested 2048 MiB, available 156 MiB)",
    "Frame decode error: corrupted NAL unit at offset 0x7f3a, dropping frame",
    "Audio sync drift exceeded 200ms threshold, resynchronizing pipeline",
    "Network jitter spike: 450ms measured vs 50ms SLA, triggering rebuffer",
    "Session keepalive missed 3 consecutive heartbeats, marking unhealthy",
    "FATAL: GPU ECC uncorrectable error detected on device 0, halting compute",
    "FATAL: kernel panic in nvidia-uvm module, initiating graceful shutdown",
    "FATAL: unrecoverable PCIe link error, device 0000:3b:00.0 disconnected",
    "Encoder pipeline stall: H.264 hardware encoder queue full for 500ms",
    "Input event backpressure: 1247 events queued, dropping oldest 500",
    "TLS handshake failed: certificate expired for streaming-gateway.prod.local",
    "DNS resolution timeout for session-registry.internal after 5000ms",
    "WebSocket connection reset by peer during active streaming session",
]

_USER_AGENTS = [
    "GFNClient/2.0.58.123 (Windows NT 10.0; x64) Chrome/120.0.6099.130",
    "GFNClient/2.0.57.98 (Windows NT 11.0; x64) Chrome/120.0.6099.71",
    "GFNClient/2.0.58.123 (macOS 14.2.1; arm64) Safari/17.2.1",
    "GFNClient/2.0.56.45 (Ubuntu 22.04; x86_64) Chromium/119.0.6045.199",
    "GFNClient/2.0.58.100 (ChromeOS 120.0.6099.114; x86_64)",
    "GFNMobile/1.5.12.8 (Android 14; Pixel 8 Pro) WebView/120.0.6099.43",
    "GFNMobile/1.5.11.3 (iOS 17.2.1; iPhone15,3) WebKit/605.1.15",
    "SHIELD/9.1.0 (Android TV 13; SHIELD Android TV Pro)",
    "GFNClient/2.0.55.200 (Windows NT 10.0; x64) Edge/120.0.2210.91",
    "GFNClient/2.0.58.123 (Fedora 39; x86_64) Firefox/121.0",
]


def create_gen_data_udfs() -> Dict[str, Callable]:
    """Create helper UDFs to generate synthetic data for each column."""

    def generate_id(seed: int) -> int:
        return seed

    def _make_numeric_gen(col_name: str):
        mean, std, lo, hi = _NUMERIC_DISTRIBUTIONS[col_name]

        def _generate(seed: int) -> float:
            if seed is None:
                return None
            rng = random.Random(seed + hash(col_name))
            return float(max(lo, min(hi, rng.gauss(mean, std))))

        return _generate

    def _gen_hostname(seed: int) -> str:
        rng = random.Random(seed + hash("hostname"))
        template = rng.choice(_HOSTNAMES)
        return template.format(id=rng.randint(1, 999))

    def _gen_region(seed: int) -> str:
        rng = random.Random(seed + hash("region"))
        return rng.choice(_REGIONS)

    def _gen_session_id(seed: int) -> str:
        rng = random.Random(seed + hash("session_id"))
        return f"sess-{uuid.UUID(int=rng.getrandbits(128))}"

    def _gen_device_model(seed: int) -> str:
        rng = random.Random(seed + hash("device_model"))
        return rng.choice(_DEVICE_MODELS)

    def _gen_gpu_driver_ver(seed: int) -> str:
        rng = random.Random(seed + hash("gpu_driver_ver"))
        return rng.choice(_DRIVER_VERSIONS)

    def _gen_os_info(seed: int) -> str:
        rng = random.Random(seed + hash("os_info"))
        return rng.choice(_OS_VERSIONS)

    def _gen_app_version(seed: int) -> str:
        rng = random.Random(seed + hash("app_version"))
        return rng.choice(_APP_VERSIONS)

    def _gen_error_message(seed: int) -> str:
        rng = random.Random(seed + hash("error_message"))
        r = rng.random()
        if r < 0.85:
            return ""
        return rng.choice(_ERROR_MESSAGES[1:])  # skip the empty one

    def _gen_client_ip(seed: int) -> str:
        rng = random.Random(seed + hash("client_ip"))
        return f"{rng.randint(10, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"

    def _gen_user_agent(seed: int) -> str:
        rng = random.Random(seed + hash("user_agent"))
        return rng.choice(_USER_AGENTS)

    udfs: Dict[str, Callable] = {
        "id": udf(generate_id, returnType=IntegerType()),
    }
    for col_name in NUMERIC_COLUMNS:
        udfs[col_name] = udf(_make_numeric_gen(col_name), returnType=DoubleType())

    string_generators = {
        "hostname": _gen_hostname,
        "region": _gen_region,
        "session_id": _gen_session_id,
        "device_model": _gen_device_model,
        "gpu_driver_ver": _gen_gpu_driver_ver,
        "os_info": _gen_os_info,
        "app_version": _gen_app_version,
        "error_message": _gen_error_message,
        "client_ip": _gen_client_ip,
        "user_agent": _gen_user_agent,
    }
    for col_name in STRING_COLUMNS:
        udfs[col_name] = udf(string_generators[col_name], returnType=StringType())

    return udfs


def generate_synthetic_data(
    spark: SparkSession, num_rows: int, num_partitions: int
) -> DataFrame:
    """Generate synthetic data.  Schema: id, 10 float64 + 10 string columns."""
    gen_udfs = create_gen_data_udfs()
    base_df = spark.range(0, num_rows, numPartitions=num_partitions)

    select_exprs = [gen_udfs["id"](col("id")).alias("id")]
    for col_name in ALL_COLUMNS:
        select_exprs.append(gen_udfs[col_name](col("id")).alias(col_name))

    return base_df.select(*select_exprs)


def execute_udf(spark: SparkSession, udf_name: str, df: DataFrame) -> DataFrame:
    """Apply the named UDF to the benchmark DataFrame."""
    df.createOrReplaceTempView("bench_table")
    all_cols = ", ".join(ALL_COLUMNS)
    return spark.sql(
        f"SELECT *, {udf_name}(struct({all_cols})) as result FROM bench_table"
    )
