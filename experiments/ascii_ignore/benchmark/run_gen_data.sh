#!/bin/bash
# Helper script to generate benchmark data
# Usage:
#   ./run_gen_data.sh --rows NUM [--validate {udf|sql}] [--sql-file-path PATH]
#
# Required arguments:
#   --rows NUM              Number of rows to generate
#
# Optional arguments:
#   --validate {udf|sql}   Run in validation mode to test the generated data
#                            with the UDF ('udf') or SQL expression ('sql')
#   --sql-file-path PATH     Path to SQL file (required when --validate sql)

set -e

ROWS=""
VALIDATE_MODE=""
SQL_FILE_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rows)
            ROWS="$2"
            shift 2
            ;;
        --validate)
            VALIDATE_MODE="$2"
            if [[ "$VALIDATE_MODE" != "udf" && "$VALIDATE_MODE" != "sql" ]]; then
                echo "Error: --validate requires 'udf' or 'sql' argument"
                echo "Usage: $0 --rows NUM [--validate {udf|sql}] [--sql-file-path PATH]"
                exit 1
            fi
            shift 2
            ;;
        --sql-file-path)
            SQL_FILE_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --rows NUM [--validate {udf|sql}] [--sql-file-path PATH]"
            exit 1
            ;;
    esac
done

if [ -z "$ROWS" ]; then
    echo "Error: --rows is required"
    echo "Usage: $0 --rows NUM [--validate {udf|sql}] [--sql-file-path PATH]"
    exit 1
fi

if [[ "$VALIDATE_MODE" == "sql" && -z "$SQL_FILE_PATH" ]]; then
    echo "Error: --sql-file-path is required when --validate sql"
    echo "Usage: $0 --rows NUM [--validate {udf|sql}] [--sql-file-path PATH]"
    exit 1
fi

# Paths
GEN_UDF_DATA_MODULE_PATH="/home/rishic/.cache/cuaether-assistant/ascii_ignore_20260301_175803/benchmark/gen_ascii_ignore_data.py"
TEST_MODULE_PATH="/home/rishic/.cache/cuaether-assistant/ascii_ignore_20260301_175803/test/test_ascii_ignore.py"
UDF_NAME="ascii_ignore"
UDF_CLASS_PATH="com.udf.ascii_ignore"

# Spark configs
SPARK_CONF_ARGS=(
    --spark-conf spark.master="local[*]"
    --spark-conf spark.driver.memory="16g"
    --spark-conf spark.rapids.sql.enabled="true"
    --spark-conf spark.plugins="com.nvidia.spark.SQLPlugin"
    --spark-conf spark.locality.wait="0s"
    --spark-conf spark.sql.cache.serializer="com.nvidia.spark.ParquetCachedBatchSerializer"
    --spark-conf spark.rapids.sql.format.parquet.reader.type="MULTITHREADED"
    --spark-conf spark.rapids.sql.reader.batchSizeBytes="1000MB"
    --spark-conf spark.sql.files.maxPartitionBytes="512MB"
    --spark-conf spark.rapids.sql.metrics.level="DEBUG"
    --spark-conf spark.rapids.memory.gpu.allocFraction="0.5"
    --spark-conf spark.jars="/home/rishic/.cache/cuaether-assistant/jars/rapids-4-spark_2.12-25.12.0.jar"
)

# Build optional SQL file path arg
SQL_FILE_PATH_ARGS=()
if [ -n "$SQL_FILE_PATH" ]; then
    SQL_FILE_PATH_ARGS=(--sql-file-path "$SQL_FILE_PATH")
fi

if [ -n "$VALIDATE_MODE" ]; then
    echo "Running gen_data.py in validation mode ($VALIDATE_MODE) with $ROWS rows..."
    python gen_data.py \
        --validate-mode "$VALIDATE_MODE" \
        --udf-name "$UDF_NAME" \
        --udf-class-path "$UDF_CLASS_PATH" \
        --test-module-path "$TEST_MODULE_PATH" \
        --gen-udf-data-module-path "$GEN_UDF_DATA_MODULE_PATH" \
        --rows "$ROWS" \
        --partitions 64 \
        "${SQL_FILE_PATH_ARGS[@]}" \
        "${SPARK_CONF_ARGS[@]}"
else
    echo "Running gen_data.py to generate $ROWS rows..."
    python gen_data.py \
        --gen-udf-data-module-path "$GEN_UDF_DATA_MODULE_PATH" \
        --rows "$ROWS" \
        --partitions 64 \
        "${SPARK_CONF_ARGS[@]}"
fi
