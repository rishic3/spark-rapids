#!/bin/bash
# Generate Parquet benchmark datasets via DBGen + spark-shell.
#
# Self-contained: this script IS the reproducible recipe. Paste it into the PR
# description (or commit it alongside the change) — anyone with Spark 3.5.7 and
# the spark-rapids datagen jar can run it.
#
# Required env vars:
#   SPARK_HOME    Spark 3.5.7 install
#   DATAGEN_JAR   absolute path to datagen_2.12-<ver>-spark357.jar
#                 (build with `mvn package -pl datagen -Dbuildver=357 -DskipTests`
#                  from the spark-rapids repo root if missing)
#
# Optional env vars:
#   BENCH_ROWS       rows per case          (default 10000000)
#   BENCH_DATA_DIR   parquet output root    (default ./data, matches MicroBenchRunner)
#                    For PR repro, override with an absolute path the reviewer
#                    can write to, e.g. BENCH_DATA_DIR=/tmp/<snake_name>_bench.

set -e

: "${SPARK_HOME:?SPARK_HOME must point at a Spark 3.5.7 install}"
: "${DATAGEN_JAR:?DATAGEN_JAR must point at datagen_2.12-<ver>-spark357.jar}"

export BENCH_DATA_DIR="${BENCH_DATA_DIR:-./data}"
export BENCH_ROWS="${BENCH_ROWS:-10000000}"
mkdir -p "$BENCH_DATA_DIR"

echo "[gen_data] SPARK_HOME=$SPARK_HOME"
echo "[gen_data] DATAGEN_JAR=$DATAGEN_JAR"
echo "[gen_data] BENCH_ROWS=$BENCH_ROWS  BENCH_DATA_DIR=$BENCH_DATA_DIR"

# Heredoc is single-quoted ('EOF') — no shell expansion inside. The Spark
# session reads BENCH_DATA_DIR / BENCH_ROWS from the environment via sys.env.
cat << 'EOF' | "$SPARK_HOME/bin/spark-shell" \
    --jars "$DATAGEN_JAR" \
    --master "local[*]" \
    --conf spark.driver.memory=16g
import org.apache.spark.sql.tests.datagen._

// -----------------------------------------------------------------------------
// TODO(operator-benchmark Step 2): one (label, ddl) entry per type the
// operator's ExprChecks advertise (captured in operator-gen-test Step 1.3).
// For multi-arg operators, the DDL must describe ALL input columns —
// e.g. binary `nvl(a, b)` over `long` becomes "a long, b long".
//
// Use a stable, filename-safe label (lowercase, underscores). The Parquet
// directory will be `$BENCH_DATA_DIR/<label>/`. Labels MUST match the case
// list in bench/spark_bench.sh.
//
// Example for unary upper(string):
//   Seq(("string", "a string"))
//
// Example for nvl(a, b) over the type surface advertised by GpuOverrides:
//   Seq(
//     ("long",       "a long, b long"),
//     ("string",     "a string, b string"),
//     ("decimal_38", "a DECIMAL(38,10), b DECIMAL(38,10)"),
//     ("array_long", "a ARRAY<long>, b ARRAY<long>"),
//     ("struct_2",   "a STRUCT<x:long, y:string>, b STRUCT<x:long, y:string>")
//   )
// -----------------------------------------------------------------------------
val cases: Seq[(String, String)] = ???

val numRows = sys.env.getOrElse("BENCH_ROWS", "10000000").toLong
val outRoot = sys.env("BENCH_DATA_DIR")

cases.foreach { case (label, ddl) =>
  val path = s"$outRoot/$label"
  println(s"[gen_data] $label  ddl=[$ddl]  rows=$numRows  -> $path")
  DBGen().addTable("data", ddl, numRows)
    .toDF(spark)
    .write.mode("overwrite").parquet(path)
}

println(s"[gen_data] generated ${cases.size} dataset(s) under $outRoot/")
:quit
EOF
