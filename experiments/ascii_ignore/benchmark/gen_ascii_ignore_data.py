import builtins  # To access built-in functions (min, sum, etc.) if there are naming conflicts
import random
from typing import Any, Callable, Dict

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *


def create_gen_data_udfs() -> Dict[str, Callable]:
    """Create helper UDFs to generate synthetic data for each column"""

    def generate_id(seed: int) -> int:
        """Generate sequential id"""
        return seed

    def generate_input_str(seed: int) -> str:
        """Generate random string data with mix of ASCII, Unicode, and special characters"""
        if seed is None:
            return None
        rng = random.Random(seed)

        # ~5% chance of None
        if rng.random() < 0.05:
            return None

        # ~3% chance of empty string
        if rng.random() < 0.03:
            return ""

        # ~3% chance of whitespace-only string
        if rng.random() < 0.03:
            return "   "

        # Build a string of reasonable enterprise-scale length (50-500 chars)
        length = rng.randint(50, 500)

        # Character pools
        ascii_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        digits = "0123456789"
        spaces = " \t\n"
        special_chars = "?()%=:&*\"'><\\}{$@+/][];~#,"
        unicode_chars = "àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ你好世界カタカナ日本語😀🎉🚀"
        normal_punctuation = ".-_!"

        # Weighted character pools for realistic data
        pools = [
            (ascii_letters, 0.45),
            (digits, 0.10),
            (spaces, 0.10),
            (special_chars, 0.15),
            (unicode_chars, 0.10),
            (normal_punctuation, 0.10),
        ]

        chars = []
        for _ in range(length):
            r = rng.random()
            cumulative = 0.0
            for pool, weight in pools:
                cumulative += weight
                if r < cumulative:
                    chars.append(rng.choice(pool))
                    break

        # Sometimes add leading/trailing spaces
        result = "".join(chars)
        if rng.random() < 0.2:
            result = "  " + result + "  "

        return result

    return {
        "id": udf(generate_id, returnType=IntegerType()),
        "input_str": udf(generate_input_str, returnType=StringType()),
    }


def generate_synthetic_data(
    spark: SparkSession, num_rows: int, num_partitions: int
) -> DataFrame:
    """
    Generate synthetic data using Spark.
    Output schema matches: id (IntegerType), input_str (StringType)
    """

    gen_udfs = create_gen_data_udfs()
    base_df = spark.range(0, num_rows, numPartitions=num_partitions)

    final_df = base_df.select(
        gen_udfs["id"](col("id")).alias("id"),
        gen_udfs["input_str"](col("id")).alias("input_str"),
    )

    return final_df


def execute_udf(spark: SparkSession, udf_name: str, df: DataFrame) -> DataFrame:
    """
    Execute the UDF once on the full benchmark DataFrame and return results.
    The UDF is registered under "udf_name" in the Spark session.
    """
    df.createOrReplaceTempView("bench_table")

    udf_result_df = spark.sql(
        f"SELECT *, {udf_name}(input_str) as result FROM bench_table"
    )
    return udf_result_df
