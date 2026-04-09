"""Synthetic data generation for the ``redact_pii`` benchmark.

Generates long text strings (500-2000 chars) with embedded PII patterns
(emails, IPv4 addresses, phone numbers, SSNs, credit card numbers) at
realistic density, simulating log / telemetry ingestion pipelines.
"""

import random
from typing import Callable, Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, StringType

_COMMON_WORDS = [
    "the", "server", "connection", "user", "session", "request", "response",
    "error", "warning", "timeout", "processing", "network", "client", "data",
    "transfer", "completed", "failed", "retrying", "authenticated", "logged",
    "received", "sent", "buffer", "stream", "packet", "protocol", "device",
    "system", "application", "service", "endpoint", "gateway", "proxy",
    "firewall", "router", "switch", "interface", "configuration", "database",
    "query", "transaction", "commit", "rollback", "index", "partition",
    "cluster", "node", "replica", "shard", "cache", "memory", "disk",
    "launched", "terminated", "restarted", "upgraded", "deployed", "monitor",
    "alert", "metric", "dashboard", "pipeline", "workflow", "scheduler",
]

_NAMES = [
    "john.doe", "jane.smith", "admin", "support", "test.user",
    "dev.ops", "monitor.bot", "ci.runner", "backup.svc", "alert.mgr",
]

_DOMAINS = [
    "example.com", "company.org", "internal.net", "cloud.io",
    "services.dev", "infra.local", "corp.biz", "platform.ai",
]


def _gen_email(rng: random.Random) -> str:
    return f"{rng.choice(_NAMES)}@{rng.choice(_DOMAINS)}"


def _gen_ipv4(rng: random.Random) -> str:
    return (
        f"{rng.randint(1, 255)}.{rng.randint(0, 255)}"
        f".{rng.randint(0, 255)}.{rng.randint(0, 255)}"
    )


def _gen_phone(rng: random.Random) -> str:
    return f"{rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"


def _gen_ssn(rng: random.Random) -> str:
    return f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"


def _gen_cc(rng: random.Random) -> str:
    sep = rng.choice(["-", " "])
    parts = [str(rng.randint(1000, 9999)) for _ in range(4)]
    return sep.join(parts)


_PII_GENERATORS = [_gen_email, _gen_ipv4, _gen_phone, _gen_ssn, _gen_cc]


def create_gen_data_udfs() -> Dict[str, Callable]:
    """Create helper UDFs to generate synthetic data for each column."""

    def generate_id(seed: int) -> int:
        return seed

    def generate_input_text(seed: int) -> str:
        if seed is None:
            return None
        rng = random.Random(seed)

        if rng.random() < 0.03:
            return None

        segments = []
        num_segments = rng.randint(8, 20)

        for _ in range(num_segments):
            word_count = rng.randint(5, 15)
            words = [rng.choice(_COMMON_WORDS) for _ in range(word_count)]
            segments.append(" ".join(words))

            if rng.random() < 0.6:
                gen = rng.choice(_PII_GENERATORS)
                segments.append(gen(rng))

        return " ".join(segments)

    return {
        "id": udf(generate_id, returnType=IntegerType()),
        "input_text": udf(generate_input_text, returnType=StringType()),
    }


def generate_synthetic_data(
    spark: SparkSession, num_rows: int, num_partitions: int
) -> DataFrame:
    """Generate synthetic data.  Schema: id (int), input_text (string)."""
    gen_udfs = create_gen_data_udfs()
    base_df = spark.range(0, num_rows, numPartitions=num_partitions)
    return base_df.select(
        gen_udfs["id"](col("id")).alias("id"),
        gen_udfs["input_text"](col("id")).alias("input_text"),
    )


def execute_udf(spark: SparkSession, udf_name: str, df: DataFrame) -> DataFrame:
    """Apply the named UDF to the benchmark DataFrame."""
    df.createOrReplaceTempView("bench_table")
    return spark.sql(
        f"SELECT *, {udf_name}(input_text) as result FROM bench_table"
    )
