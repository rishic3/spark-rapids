# Successfully executed test
# Generated on: 2026-03-01 17:59:34
# ===================================================
import argparse
import logging
from pathlib import Path
from typing import Any, Callable, List, Union

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from ascii_ignore import ascii_ignore  # Assume the UDF is imported

logging.basicConfig(format="%(message)s")
logger = logging.getLogger(__name__)


def create_test_data(spark: SparkSession) -> DataFrame:
    """Create test data"""
    test_data = [
        (1, "hello world"),
        (2, None),
        (3, ""),
        (4, "café résumé"),
        (5, "test?string(with)special%chars=here"),
        (6, "a:b&c*d\"e'f>g<h"),
        (7, "\\path}to{file$name@place"),
        (8, "a+b/c]d[e;f~g#h,i"),
        (9, "  leading and trailing spaces  "),
        (10, "?()%=:&*\"'><\\}{$@+/][];~#,"),
        (11, "normal_text-with.dots!and_underscores"),
        (12, "Unicode: 你好世界"),
        (13, "MiXeD CaSe 123"),
        (14, "line\nnewline\ttab"),
        (15, "price: $100 (50% off)"),
        (16, "   "),
        (17, "a"),
        (18, "emoji: 😀🎉"),
        (19, "Ñoño àccéntèd"),
        (20, "key=value&other=thing"),
    ]
    schema = StructType([
        StructField("id", IntegerType()),
        StructField("input_str", StringType()),
    ])

    return spark.createDataFrame(test_data, schema)

def register_udf(
    spark: SparkSession,
    udf_name: str,
    udf_func: Callable,
) -> None:
    if hasattr(udf_func, 'returnType'):
        spark.udf.register(udf_name, udf_func)
    else:
        spark.udf.register(udf_name, udf_func, StringType())

def execute_udf(
    spark: SparkSession, udf_name: str, test_df: DataFrame
) -> DataFrame:
    test_df.createOrReplaceTempView("test_table")

    udf_result_df = spark.sql(f"SELECT id, input_str, {udf_name}(input_str) as result FROM test_table")
    return udf_result_df

def verify_udf_results(udf_result_df: DataFrame, test_df: DataFrame) -> None:
    """Verify the correctness of the UDF results"""
    results = {row['id']: row['result'] for row in udf_result_df.collect()}

    # 1: plain ASCII string, no special chars
    assert results[1] == "hello world", f"Expected 'hello world', got '{results[1]}'"

    # 2: None input -> None output
    assert results[2] is None, f"Expected None, got '{results[2]}'"

    # 3: empty string -> None (falsy in Python)
    assert results[3] is None, f"Expected None for empty string, got '{results[3]}'"

    # 4: "café résumé" -> non-ASCII chars (é, é) removed -> "caf rsum"
    assert results[4] == "caf rsum", f"Expected 'caf rsum', got '{results[4]}'"

    # 5: "test?string(with)special%chars=here" -> remove ? ( ) % =
    assert results[5] == "teststringwithspecialcharshere", f"Expected 'teststringwithspecialcharshere', got '{results[5]}'"

    # 6: 'a:b&c*d"e\'f>g<h' -> remove : & * " ' > <
    assert results[6] == "abcdefgh", f"Expected 'abcdefgh', got '{results[6]}'"

    # 7: "\\path}to{file$name@place" -> remove \ } { $ @
    assert results[7] == "pathtofilenameplace", f"Expected 'pathtofilenameplace', got '{results[7]}'"

    # 8: "a+b/c]d[e;f~g#h,i" -> remove + / ] [ ; ~ # ,
    assert results[8] == "abcdefghi", f"Expected 'abcdefghi', got '{results[8]}'"

    # 9: "  leading and trailing spaces  " -> strip -> "leading and trailing spaces"
    assert results[9] == "leading and trailing spaces", f"Expected 'leading and trailing spaces', got '{results[9]}'"

    # 10: string of only special chars -> all removed -> empty string
    assert results[10] == "", f"Expected empty string, got '{results[10]}'"

    # 11: "normal_text-with.dots!and_underscores" -> no special chars to remove
    assert results[11] == "normal_text-with.dots!and_underscores", f"Expected 'normal_text-with.dots!and_underscores', got '{results[11]}'"

    # 12: "Unicode: 你好世界" -> non-ASCII removed, : removed -> "Unicode"
    assert results[12] == "Unicode", f"Expected 'Unicode', got '{results[12]}'"

    # 13: "MiXeD CaSe 123" -> unchanged
    assert results[13] == "MiXeD CaSe 123", f"Expected 'MiXeD CaSe 123', got '{results[13]}'"

    # 14: "line\nnewline\ttab" -> contains \n and \t which are ASCII -> kept
    assert results[14] == "line\nnewline\ttab", f"Expected 'line\\nnewline\\ttab', got '{results[14]}'"

    # 15: "price: $100 (50% off)" -> remove : $ ( % )
    assert results[15] == "price 100 50 off", f"Expected 'price 100 50 off', got '{results[15]}'"

    # 16: "   " -> strip -> empty string ""
    assert results[16] == "", f"Expected empty string, got '{results[16]}'"

    # 17: "a" -> unchanged
    assert results[17] == "a", f"Expected 'a', got '{results[17]}'"

    # 18: "emoji: 😀🎉" -> non-ASCII removed, : removed -> "emoji"
    assert results[18] == "emoji", f"Expected 'emoji', got '{results[18]}'"

    # 19: "Ñoño àccéntèd" -> non-ASCII (Ñ,ñ,à,é,è) removed -> "oo ccntd"
    assert results[19] == "oo ccntd", f"Expected 'oo ccntd', got '{results[19]}'"

    # 20: "key=value&other=thing" -> remove = &
    assert results[20] == "keyvalueotherthing", f"Expected 'keyvalueotherthing', got '{results[20]}'"

    print("All assertions passed!")


def run_test(spark: SparkSession) -> None:
    """Run the comparison test"""
    test_df = create_test_data(spark)
    assert isinstance(test_df, DataFrame), "Output of create_test_data must be a DataFrame"
    test_df = test_df.repartition(1)
    
    register_udf(spark, udf_name="ascii_ignore", udf_func=ascii_ignore)
    udf_result_df = execute_udf(spark, udf_name="ascii_ignore", test_df=test_df)

    print("UDF result:")
    udf_result_df.show(truncate=False)

    verify_udf_results(udf_result_df=udf_result_df, test_df=test_df)


if __name__ == "__main__":
    """Run the comparison test"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapids-jar-path", required=True)
    args = parser.parse_args()

    
    spark = (
        SparkSession.builder.appName("UDF Unit Test: ascii_ignore")
        .master("local[*]")
        .config("spark.jars", args.rapids_jar_path)
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
        .config("spark.rapids.skipGpuArchitectureCheck", "true")
        .config("spark.rapids.sql.mode", "explainOnly")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )

    try:
        run_test(spark)
    except AssertionError as e:
        logger.error("Assertion failed")
        raise e
    except Exception as e:
        logger.error(f"Test code failed: {type(e).__name__}")
        raise e
    finally:
        spark.stop()
