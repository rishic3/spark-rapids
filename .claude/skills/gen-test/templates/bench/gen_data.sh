#!/bin/bash
# Generate Parquet benchmark datasets via DBGen + spark-shell.
#
# Self-contained reproducible recipe — paste into a PR description as-is.
#
# -----------------------------------------------------------------------------
# Before running:
#
#   1. Set env vars (required):
#        SPARK_HOME       Spark 3.5.7 install
#        DATAGEN_JAR      absolute path to datagen_2.12-<ver>-spark357.jar
#                         build with: mvn package -pl datagen -Dbuildver=357 -DskipTests
#
#   2. Optional env vars:
#        BENCH_ROWS       rows per case                 (default 10000000)
#        BENCH_NULL_PROB  per-column null rate 0..1     (default 0.25, NDS-like;
#                         set to 0.0 to reproduce the no-nulls short-circuit path)
#        BENCH_DATA_DIR   parquet output root           (default ./data)
#                         For PR repro, use an absolute path the reviewer can
#                         write to, e.g. /tmp/<snake_name>_bench.
#
#   3. Fill in the `cases` list in the heredoc below — one (label, ddl) per
#      type in the operator's advertised ExprChecks surface. The DDL must
#      describe ALL input columns (e.g. binary `nvl(a,b)` over long =>
#      "a long, b long"). Labels become Parquet subdir names and MUST match
#      the `cases` list in spark_bench.sh.
#
#      Example cases for nvl(a, b) over the ExprChecks surface:
#        Seq(
#          ("long",       "a long, b long"),
#          ("string",     "a string, b string"),
#          ("decimal_38", "a DECIMAL(38,10), b DECIMAL(38,10)"),
#          ("array_long", "a ARRAY<long>, b ARRAY<long>"),
#          ("struct_2",   "a STRUCT<x:long, y:string>, b STRUCT<x:long, y:string>")
#        )
# -----------------------------------------------------------------------------

set -e

: "${SPARK_HOME:?SPARK_HOME must point at a Spark 3.5.7 install}"
: "${DATAGEN_JAR:?DATAGEN_JAR must point at datagen_2.12-<ver>-spark357.jar}"

export BENCH_DATA_DIR="${BENCH_DATA_DIR:-./data}"
export BENCH_ROWS="${BENCH_ROWS:-10000000}"
export BENCH_NULL_PROB="${BENCH_NULL_PROB:-0.25}"
mkdir -p "$BENCH_DATA_DIR"

echo "[gen_data] SPARK_HOME=$SPARK_HOME"
echo "[gen_data] DATAGEN_JAR=$DATAGEN_JAR"
echo "[gen_data] BENCH_ROWS=$BENCH_ROWS  BENCH_DATA_DIR=$BENCH_DATA_DIR  BENCH_NULL_PROB=$BENCH_NULL_PROB"

# Single-quoted heredoc ('EOF') — no bash expansion; Scala reads env vars via sys.env.
cat << 'EOF' | "$SPARK_HOME/bin/spark-shell" \
    --jars "$DATAGEN_JAR" \
    --master "local[*]" \
    --conf spark.driver.memory=16g
import org.apache.spark.sql.tests.datagen._

// TODO: one (label, ddl) per advertised type (see the guidance at the top of this file).
val cases: Seq[(String, String)] = ???

val numRows  = sys.env.getOrElse("BENCH_ROWS",      "10000000").toLong
val nullProb = sys.env.getOrElse("BENCH_NULL_PROB", "0.25").toDouble
val outRoot  = sys.env("BENCH_DATA_DIR")

cases.foreach { case (label, ddl) =>
  val path = s"$outRoot/$label"
  println(s"[gen_data] $label  ddl=[$ddl]  rows=$numRows  nullProb=$nullProb  -> $path")
  val dataTable = DBGen().addTable("data", ddl, numRows)
  // DBGen defaults to no nulls; inject per-column null rate so coalesce-like
  // operators see realistic input (NDS post-outer-join resembles ~25% nulls).
  // If the operator has more than two input columns, extend this block.
  dataTable("a").setNullProbability(nullProb)
  dataTable("b").setNullProbability(nullProb)
  dataTable.toDF(spark).write.mode("overwrite").parquet(path)
}

println(s"[gen_data] generated ${cases.size} dataset(s) under $outRoot/")
:quit
EOF
