/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 */
package com.udf.bench

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

/**
 * Benchmark utilities for an extracted Spark RAPIDS operator.
 *
 *   - generateSyntheticData: Create benchmark data matching the test schema.
 *   - executeCpu: Run the target SQL expression with the plugin disabled
 *     (`spark.rapids.sql.enabled=false`), exercising Spark's pure-CPU path.
 *   - executeGpu: Run the same logical query with the plugin enabled, invoking
 *     the extracted RapidsUDF (registered here) under `spark.rapids.sql.enabled=true`.
 *
 * Both executeCpu and executeGpu must share the SparkSession — the plugin
 * toggle is a runtime `spark.conf.set`, not a session property.
 */
object BenchUtils {

  // ---------------------------------------------------------------------------
  // Data generation
  // ---------------------------------------------------------------------------

  /**
   * TODO: Generate a synthetic DataFrame matching the test data schema.
   *
   * Use `spark.range(0, numRows, 1, numPartitions)` as the base, then apply
   * randomized column generators to produce data matching the operator's
   * expected input types.
   *
   * Requirements:
   *   - Column names and types MUST match the schema from `createTestData` in
   *     [[com.udf.SqlOperatorComparisonTest]].
   *   - Data should be realistic and varied (different lengths, magnitudes,
   *     edge values, a healthy null fraction).
   *   - For variable-length inputs, generate sizable rows representative of
   *     enterprise-scale data — the GPU win typically grows with row size.
   *
   * Example (single string column):
   * {{{
   *   val baseDF = spark.range(0, numRows, 1, numPartitions)
   *   baseDF.select(
   *     col("id").cast(IntegerType).alias("id"),
   *     when(rand() < 0.05, lit(null))
   *       .otherwise(concat(lit("prefix-"), (rand() * 1e6).cast(LongType).cast(StringType)))
   *       .alias("s")
   *   )
   * }}}
   *
   * @param spark         active SparkSession
   * @param numRows       number of rows to generate
   * @param numPartitions number of output partitions
   * @return DataFrame with the same schema as the test data
   */
  def generateSyntheticData(
      spark: SparkSession,
      numRows: Long,
      numPartitions: Int
  ): DataFrame = ???

  // ---------------------------------------------------------------------------
  // Execution
  // ---------------------------------------------------------------------------

  /** Name used to register the extracted RapidsUDF in the GPU path. */
  val RapidsUdfName: String = "extracted_rapids_udf"

  /**
   * TODO: Run the target SQL expression on the CPU (plugin disabled).
   *
   *   1. Set `spark.rapids.sql.enabled=false` on the SparkSession's runtime conf.
   *   2. Register `df` as a temp view named `bench_table`.
   *   3. Run the same CPU SQL used in the comparison test, but over `bench_table`
   *      and projecting only the inputs plus the operator result (SELECT id, <op>).
   *      Avoid `SELECT *` so unrelated columns don't skew the benchmark.
   *
   * Example:
   * {{{
   *   spark.conf.set("spark.rapids.sql.enabled", "false")
   *   df.createOrReplaceTempView("bench_table")
   *   spark.sql("SELECT id, upper(s) AS result FROM bench_table")
   * }}}
   *
   * @param spark active SparkSession (shared with [[executeGpu]])
   * @param df    input benchmark DataFrame
   * @return result DataFrame from the CPU SQL execution
   */
  def executeCpu(spark: SparkSession, df: DataFrame): DataFrame = ???

  /**
   * TODO: Run the equivalent query on the GPU via the extracted RapidsUDF.
   *
   *   1. Set `spark.rapids.sql.enabled=true` on the SparkSession's runtime conf.
   *   2. Register the extracted RapidsUDF (e.g. `new com.udf.GpuUpperRapidsUDF`)
   *      under [[RapidsUdfName]], passing the correct Spark return type.
   *   3. Register `df` as `bench_table` and run the mirror SQL that invokes the
   *      UDF, with the same output schema as [[executeCpu]].
   *
   * Example:
   * {{{
   *   spark.conf.set("spark.rapids.sql.enabled", "true")
   *   spark.udf.register(RapidsUdfName, new com.udf.GpuUpperRapidsUDF(), StringType)
   *   df.createOrReplaceTempView("bench_table")
   *   spark.sql(s"SELECT id, $RapidsUdfName(s) AS result FROM bench_table")
   * }}}
   *
   * @param spark active SparkSession (shared with [[executeCpu]])
   * @param df    input benchmark DataFrame
   * @return result DataFrame from the GPU execution
   */
  def executeGpu(spark: SparkSession, df: DataFrame): DataFrame = ???
}
