#!/bin/bash
# A/B Spark-level benchmark for the operator-backport skill (Step 6).
#
# Self-contained: this script IS the reproducible recipe. Paste it into the PR
# description (or commit it alongside the change) — anyone with Spark 3.5.7,
# the two plugin jars, and the parquet datasets from bench/gen_data.sh can run it.
#
# Runs the same SQL workload twice — once with the baseline plugin jar and once
# with the optimized jar — both with the RAPIDS plugin enabled. The CPU baseline
# is owned by the gen-test correctness suite and is not re-measured here. The
# output of interest is the optimized-vs-baseline delta per case.
#
# Usage:
#   ./bench/spark_bench.sh <baseline.jar> <optimized.jar>
#
# Required env vars:
#   SPARK_HOME    Spark 3.5.7 install (must match the buildver of both jars)
#
# Optional env vars:
#   BENCH_DATA_DIR   parquet root from bench/gen_data.sh  (default ./data)
#                    For PR repro, override with an absolute path the reviewer
#                    can read from, e.g. BENCH_DATA_DIR=/tmp/<snake_name>_bench.
#   BENCH_WARMUP     untimed runs per case                (default 2)
#   BENCH_MEASURED   timed runs per case                  (default 3)

set -e

: "${SPARK_HOME:?SPARK_HOME must point at a Spark 3.5.7 install}"

BASELINE_JAR="${1:?usage: $0 <baseline.jar> <optimized.jar>}"
OPTIMIZED_JAR="${2:?usage: $0 <baseline.jar> <optimized.jar>}"

for f in "$BASELINE_JAR" "$OPTIMIZED_JAR"; do
    [ -f "$f" ] || { echo "Error: jar not found: $f"; exit 1; }
done

export BENCH_DATA_DIR="${BENCH_DATA_DIR:-./data}"
export BENCH_WARMUP="${BENCH_WARMUP:-2}"
export BENCH_MEASURED="${BENCH_MEASURED:-3}"

run_with_jar() {
    local label="$1" jar="$2"
    echo
    echo "============================================================"
    echo "[$label] jar=$jar"
    echo "============================================================"
    # Heredoc is unquoted (EOF) so $jar / $label expand at bash time.
    # Inner Scala interpolations must be escaped with \$ so bash leaves them
    # for Scala to expand at runtime.
    BENCH_LABEL="$label" "$SPARK_HOME/bin/spark-shell" \
        --jars "$jar" \
        --master "local[*]" \
        --conf spark.driver.memory=16g \
        --conf spark.plugins=com.nvidia.spark.SQLPlugin \
        --conf spark.sql.files.maxPartitionBytes=256MB << EOF
// -----------------------------------------------------------------------------
// TODO(operator-benchmark Step 2 / backport Step 6): one (label, sqlFragment)
// per case from bench/gen_data.sh — labels MUST match the parquet subdir names.
// The fragment is the projection list inside \`SELECT <fragment> FROM bench\`.
// Drive enough work per row to amortize Spark overhead — wrapping the operator
// in COUNT(...) is the standard pattern.
//
// Example (matches the nvl gen_data.sh example):
//   Seq(
//     ("long",       "COUNT(coalesce(a, 0L)), COUNT(coalesce(a, b))"),
//     ("string",     "COUNT(coalesce(a, 'x')), COUNT(coalesce(a, b))"),
//     ("decimal_38", "COUNT(coalesce(a, CAST(0 AS DECIMAL(38,10)))), COUNT(coalesce(a, b))"),
//     ("array_long", "COUNT(coalesce(a, b))"),
//     ("struct_2",   "COUNT(coalesce(a, b))")
//   )
// -----------------------------------------------------------------------------
val cases: Seq[(String, String)] = ???

val dataRoot = sys.env("BENCH_DATA_DIR")
val warmup   = sys.env.getOrElse("BENCH_WARMUP",   "2").toInt
val measured = sys.env.getOrElse("BENCH_MEASURED", "3").toInt
val tag      = sys.env.getOrElse("BENCH_LABEL",    "run")

cases.foreach { case (caseLabel, sqlFrag) =>
  val df = spark.read.parquet(s"\$dataRoot/\$caseLabel")
  df.createOrReplaceTempView("bench")
  val q = s"SELECT \$sqlFrag FROM bench"
  println(s"[\$tag] === \$caseLabel ===  \$q")
  for (_ <- 0 until warmup) spark.sql(q).collect()
  for (i <- 1 to measured) {
    print(s"[\$tag] \$caseLabel run\$i: ")
    spark.time(spark.sql(q).collect())
  }
}
:quit
EOF
}

run_with_jar baseline  "$BASELINE_JAR"
run_with_jar optimized "$OPTIMIZED_JAR"
