#!/bin/bash
# A/B Spark-level benchmark: baseline plugin jar vs. optimized plugin jar.
#
# Self-contained reproducible recipe — paste into a PR description as-is.
# Both runs execute on GPU (spark.plugins=com.nvidia.spark.SQLPlugin); the
# only variable is which plugin jar is loaded. The CPU baseline is owned by
# the gen-test correctness suite and is not re-measured here.
#
# -----------------------------------------------------------------------------
# Before running:
#
#   1. Set env vars:
#        SPARK_HOME       Spark 3.5.7 install (must match both jars' buildver)
#
#   2. Optional env vars:
#        BENCH_DATA_DIR   parquet root from gen_data.sh   (default ./data;
#                         for PR repro use the same absolute path gen_data.sh did)
#        BENCH_WARMUP     untimed runs per case           (default 2)
#        BENCH_MEASURED   timed runs per case             (default 3; consider 10
#                         for PR-quality numbers if results are noisy)
#
#   3. Pass the two jars as args (use absolute paths in PR snippets):
#        ./spark_bench.sh <baseline.jar> <optimized.jar>
#
#   4. Fill in the `cases` list in the heredoc below — labels MUST match the
#      Parquet subdir names produced by gen_data.sh. Each sqlFragment is the
#      projection list inside `SELECT <fragment> FROM bench`. Drive real work
#      per row (SUM/arithmetic, not just COUNT) so Spark overhead doesn't drown
#      the operator signal. Chain multiple operator invocations per row if the
#      target workload does (e.g. NDS `coalesce(a,0) + coalesce(b,0)`).
#
#      Example cases for nvl over the same ExprChecks surface as gen_data.sh:
#        Seq(
#          ("long",       "SUM(a - coalesce(b, 0L)) + SUM(coalesce(a, 0L) + coalesce(b, 0L))"),
#          ("decimal_38", "SUM(coalesce(a, CAST(0 AS DECIMAL(38,10))) + coalesce(b, CAST(0 AS DECIMAL(38,10))))"),
#          ("string",     "COUNT(coalesce(a, 'x')) + COUNT(coalesce(b, 'y')) + COUNT(coalesce(a, b))"),
#          ("array_long", "COUNT(coalesce(a, b))"),
#          ("struct_2",   "COUNT(coalesce(a, b))")
#        )
# -----------------------------------------------------------------------------

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

# One spark-shell invocation per jar. Unquoted `EOF` lets bash expand $jar and
# $label; Scala interpolations must be escaped as \$... so bash leaves them
# for the Scala interpreter at runtime.
for entry in "baseline:$BASELINE_JAR" "optimized:$OPTIMIZED_JAR"; do
    label="${entry%%:*}"
    jar="${entry#*:}"
    echo
    echo "============================================================"
    echo "[$label] jar=$jar"
    echo "============================================================"
    cat << EOF | BENCH_LABEL="$label" "$SPARK_HOME/bin/spark-shell" \
        --jars "$jar" \
        --master "local[*]" \
        --conf spark.driver.memory=16g \
        --conf spark.plugins=com.nvidia.spark.SQLPlugin \
        --conf spark.sql.files.maxPartitionBytes=256MB
// TODO: one (label, sqlFragment) per case from gen_data.sh (see the guidance
// at the top of this file). Labels MUST match gen_data.sh's subdir names.
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
done
