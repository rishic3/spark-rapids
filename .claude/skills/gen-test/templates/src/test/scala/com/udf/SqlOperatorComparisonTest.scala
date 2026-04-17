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

  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .appName("SQL Operator vs. RapidsUDF Comparison Test")
      .master("local[*]")
      .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
      .config("spark.rapids.memory.gpu.pool", "NONE")
      .config("spark.rapids.sql.explain", "NONE")
      .config("spark.rapids.skipGpuArchitectureCheck", "true")
      .config("spark.sql.adaptive.enabled", "false")
      .getOrCreate()
  }

  override def afterAll(): Unit = {
    if (spark != null) spark.stop()
  }

  /**
   * TODO: Create a test DataFrame that exercises the operator branch under test.
   *
   * Guidance:
   *   - First check the repo for existing coverage (see SKILL.md, Step 3):
   *       * `integration_tests/src/main/python/` for pytest cases (e.g. `string_test.py`,
   *         `cast_test.py`) touching this SQL function. Reuse inputs/edge cases if found.
   *       * `tests/src/test/scala/` for Scala tests referencing the `Gpu<Name>` class.
   *   - If nothing specific exists, build a robust dataset by hand. Must cover:
   *       * nulls, empties, and boundary values for every input column
   *       * the exact input types handled by the cuDF branch you extracted
   *         (do not test branches the extracted code cannot reach)
   *       * representative "real-world" row shapes (realistic lengths / magnitudes)
   *   - Include `id: Int` as the first column for stable ordering.
   *
   * Example (for an operator taking a single StringType input):
   * {{{
   *   val schema = StructType(Seq(
   *     StructField("id", IntegerType, nullable = false),
   *     StructField("s", StringType, nullable = true)
   *   ))
   *   val rows = Seq(
   *     Row(0, "hello"),
   *     Row(1, ""),
   *     Row(2, null),
   *     Row(3, "MiXeD-CasE 123"),
   *     Row(4, "\u00e9\u00e8 non-ascii")
   *   )
   *   spark.createDataFrame(spark.sparkContext.parallelize(rows), schema)
   * }}}
   */
  def createTestData(): DataFrame = ???

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
   * The return type must match the Spark type of the original operator's output.
   *
   * Example:
   * {{{
   *   spark.udf.register(udfName, new com.udf.GpuUpperRapidsUDF(), StringType)
   * }}}
   */
  def registerRapidsUDF(udfName: String): Unit = ???

  test("CPU SQL operator vs. extracted RapidsUDF") {
    val testDF = createTestData().repartition(1)
    testDF.createOrReplaceTempView("test_table")

    // --- CPU path: plugin disabled, pure Spark SQL ---
    spark.conf.set("spark.rapids.sql.enabled", "false")
    val cpuDF = spark.sql(cpuSqlQuery())
    val cpuPlan = cpuDF.queryExecution.executedPlan.toString
    assert(!cpuPlan.contains("Gpu"),
      s"Expected CPU plan when spark.rapids.sql.enabled=false, got:\n$cpuPlan")

    // --- GPU path: plugin enabled, extracted RapidsUDF ---
    spark.conf.set("spark.rapids.sql.enabled", "true")
    val udfName = "extracted_rapids_udf"
    registerRapidsUDF(udfName)
    val gpuDF = spark.sql(gpuSqlQuery(udfName))
    val gpuPlan = gpuDF.queryExecution.executedPlan.toString
    assert(gpuPlan.contains("Gpu"),
      s"Expected GPU plan when spark.rapids.sql.enabled=true, got:\n$gpuPlan")

    // Ensure GPU and CPU results are equal
    TestUtils.assertDataFrameEquals(actual = gpuDF, expected = cpuDF)
  }
}
