/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 */
package com.udf

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll

/**
 * Compares the CPU Spark SQL operator against the extracted RapidsUDF
 * reproduction of its cuDF implementation.
 *
 *   - CPU path: run the target SQL expression with `spark.rapids.sql.enabled=false`
 *     so the plugin is inert and Spark's own CPU implementation executes.
 *   - GPU path: register the extracted `<OperatorName>RapidsUDF` as a Spark UDF
 *     and invoke it in an equivalent query, with `spark.rapids.sql.enabled=true`.
 *
 * Results must match exactly (null-aware), and the toggle is done at runtime via
 * `spark.conf.set`, so a single SparkSession is reused for both runs.
 */
class SqlOperatorComparisonTest extends AnyFunSuite with BeforeAndAfterAll {

  // Eager init: `createTestData()` is called at class-construction time (when
  // ScalaTest registers the per-case `test(...)` blocks below), which is BEFORE
  // `beforeAll`. Lazy-init via `var spark = _` would NPE inside createTestData.
  val spark: SparkSession = SparkSession.builder()
    .appName("SQL Operator vs. RapidsUDF Comparison Test")
    .master("local[*]")
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
    .config("spark.rapids.memory.gpu.pool", "NONE")
    .config("spark.rapids.sql.explain", "NONE")
    .config("spark.rapids.skipGpuArchitectureCheck", "true")
    .config("spark.sql.adaptive.enabled", "false")
    .getOrCreate()

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
  }

  /**
   * TODO: Return one DataFrame per type the operator's `ExprChecks` advertises.
   *
   * Each DataFrame is run through the CPU-vs-GPU comparison independently, so
   * the returned list is the structural enforcement of "test every advertised
   * type" (see SKILL.md, Step 1.3 / Step 3). Skipping a type here is the same
   * gap that lets primitive-only optimizations silently break on nested types.
   *
   * Guidance:
   *   - All DataFrames MUST share the same column NAMES (the queries in
   *     [[cpuSqlQuery]] / [[gpuSqlQuery]] are reused across cases). Only the
   *     column TYPES differ.
   *   - Include `id: Int` as the first column for stable ordering.
   *   - The robustness checklist (nulls, empties, boundaries, unicode,
   *     realistic shapes, all reachable cuDF branches) applies PER DataFrame —
   *     not aggregated across the list.
   *   - First check the repo for existing coverage (see SKILL.md, Step 3):
   *       * `integration_tests/src/main/python/` for pytest cases (e.g.
   *         `string_test.py`, `cast_test.py`) touching this SQL function.
   *       * `tests/src/test/scala/` for Scala tests referencing the `Gpu<Name>` class.
   *
   * Example for `nvl(a, b)` whose ExprChecks advertise long, string, and
   * array<long>:
   * {{{
   *   def longCase: DataFrame = spark.createDataFrame(
   *     spark.sparkContext.parallelize(Seq(
   *       Row(0, 1L,         2L),
   *       Row(1, null,       3L),
   *       Row(2, Long.MaxValue, null),
   *       Row(3, null,       null))),
   *     StructType(Seq(
   *       StructField("id", IntegerType, false),
   *       StructField("a",  LongType,    true),
   *       StructField("b",  LongType,    true))))
   *
   *   def stringCase: DataFrame = ... // same column names, StringType for a/b
   *   def arrayLongCase: DataFrame = ... // same column names, ArrayType(LongType)
   *
   *   Seq(longCase, stringCase, arrayLongCase)
   * }}}
   */
  def createTestData(): Seq[DataFrame] = ???

  /**
   * TODO: Return the SQL expression that invokes the target operator on the
   * test DataFrame's columns.
   *
   * This is the SAME fragment used for both CPU and GPU runs, except the GPU
   * run substitutes the UDF name (see [[runComparison]]).
   *
   * Example (for a unary string op tested as `upper(s)`):
   * {{{
   *   s"SELECT id, upper(s) AS result FROM test_table"
   * }}}
   *
   * @return a complete SQL query string referencing `test_table`
   */
  def cpuSqlQuery(): String = ???

  /**
   * TODO: Return the SQL query that invokes the extracted RapidsUDF, registered
   * under `udfName`. The result DataFrame's schema MUST match [[cpuSqlQuery]].
   *
   * Example:
   * {{{
   *   s"SELECT id, $udfName(s) AS result FROM test_table"
   * }}}
   */
  def gpuSqlQuery(udfName: String): String = ???

  /**
   * TODO: Register the extracted RapidsUDF with Spark under `udfName`.
   *
   * Because each Java `UDF<N>[T1, ..., R]` subclass binds a single type triple
   * (see SKILL.md Step 4a), dispatch on the per-case DataFrame's input column
   * type to pick the right subclass and matching return `DataType`.
   *
   * Example for a binary operator whose return type tracks input type
   * (same shape used in `opt/GpuNvl2/`):
   * {{{
   *   df.schema("a").dataType match {
   *     case LongType        => spark.udf.register(udfName, new com.udf.<Op>LongUDF(),        LongType)
   *     case StringType      => spark.udf.register(udfName, new com.udf.<Op>StringUDF(),      StringType)
   *     case dt: DecimalType => spark.udf.register(udfName, new com.udf.<Op>DecimalUDF(),     dt)
   *     case at: ArrayType   => spark.udf.register(udfName, new com.udf.<Op>ArrayLongUDF(),   at)
   *     case st: StructType  => spark.udf.register(udfName, new com.udf.<Op>StructUDF(),      st)
   *   }
   * }}}
   *
   * For operators with a fixed return type (e.g. `upper` always returns
   * `StringType`), the match still dispatches to the right input-typed
   * subclass but the return `DataType` is a constant.
   */
  def registerRapidsUDF(udfName: String, df: DataFrame): Unit = ???

  // One test per advertised type — see [[createTestData]]. The body is
  // identical across cases; only the input DataFrame's schema differs.
  createTestData().zipWithIndex.foreach { case (df, i) =>
    test(s"CPU SQL operator vs. extracted RapidsUDF — case $i") {
      df.repartition(1).createOrReplaceTempView("test_table")

      // --- CPU path: plugin disabled, pure Spark SQL ---
      spark.conf.set("spark.rapids.sql.enabled", "false")
      val cpuDF = spark.sql(cpuSqlQuery())
      val cpuPlan = cpuDF.queryExecution.executedPlan.toString
      assert(!cpuPlan.contains("Gpu"),
        s"Expected CPU plan when spark.rapids.sql.enabled=false, got:\n$cpuPlan")

      // --- GPU path: plugin enabled, extracted RapidsUDF ---
      spark.conf.set("spark.rapids.sql.enabled", "true")
      val udfName = s"extracted_rapids_udf_$i"
      registerRapidsUDF(udfName, df)
      val gpuDF = spark.sql(gpuSqlQuery(udfName))
      val gpuPlan = gpuDF.queryExecution.executedPlan.toString
      assert(gpuPlan.contains("Gpu"),
        s"Expected GPU plan when spark.rapids.sql.enabled=true, got:\n$gpuPlan")

      // Ensure GPU and CPU results are equal
      TestUtils.assertDataFrameEquals(actual = gpuDF, expected = cpuDF)
    }
  }
}
