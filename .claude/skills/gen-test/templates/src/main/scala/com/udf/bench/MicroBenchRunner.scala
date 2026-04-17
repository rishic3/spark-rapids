/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 */
package com.udf.bench

import java.io.File

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{
  ColumnVector,
  Cuda,
  Rmm,
  RmmAllocationMode,
  Table
}
import com.udf.Arm.{closeAll, withResource}

/**
 * GPU-only microbenchmark runner for an extracted Spark RAPIDS operator.
 *
 * Use the paired Spark benchmark (SparkBenchRunner + executeCpu) for CPU vs. GPU
 * end-to-end comparisons; use this microbenchmark for tight optimization loops
 * where you want to isolate libcudf costs from Spark overhead.
 *
 *   Read Parquet (via cuDF Table.readParquet) once
 *   -> warmup `evaluateColumnar` runs
 *   -> measured `evaluateColumnar` runs
 *   -> report min / median ms
 *
 * Usage:
 *   mvn exec:java -Dexec.mainClass=com.udf.bench.MicroBenchRunner \
 *     -Dexec.args="--data-path data/bench_data --rows 1000000"
 */
object MicroBenchRunner {

  private val DefaultWarmup = 2
  private val DefaultMeasured = 4
  private val DefaultRmmAllocFraction = 0.9f

  /**
   * TODO: Execute the extracted RapidsUDF via `evaluateColumnar`.
   *
   * Index the input columns of `table` in the same order as the test data
   * schema (skip the `id` column if present — it is not an operator input).
   *
   * Example (unary string op, schema = [id, s]):
   * {{{
   *   val udf = new com.udf.GpuUpperRapidsUDF()
   *   udf.evaluateColumnar(numRows, table.getColumn(1))
   * }}}
   *
   * Example (binary op, schema = [id, a, b]):
   * {{{
   *   val udf = new com.udf.GpuSomeBinaryRapidsUDF()
   *   udf.evaluateColumnar(numRows, table.getColumn(1), table.getColumn(2))
   * }}}
   *
   * @param table   dataset loaded on GPU
   * @param numRows number of rows in the dataset
   * @return result ColumnVector — caller closes it
   */
  def executeGpu(table: Table, numRows: Int): ColumnVector = ???

  def main(args: Array[String]): Unit = {
    val parsed = parseArgs(args)

    val dataPath = parsed.getOrElse("data-path",
      throw new IllegalArgumentException("--data-path is required"))
    val maxRows = parsed.getOrElse("rows", "-1").toInt
    val rmmAllocFraction = parsed.getOrElse("pool-fraction", DefaultRmmAllocFraction.toString).toFloat
    val warmup = parsed.getOrElse("warmup", DefaultWarmup.toString).toInt
    val measured = parsed.getOrElse("measured", DefaultMeasured.toString).toInt
    val profile = parsed.contains("profile")

    // Initialize RMM pool
    if (!Rmm.isInitialized()) {
      val memInfo = Cuda.memGetInfo()
      val poolSize = (memInfo.free * rmmAllocFraction).toLong & ~255L
      Rmm.initialize(RmmAllocationMode.POOL, null, poolSize)
    }

    // Read Parquet data into cuDF table
    withResource(readParquetData(dataPath, maxRows)) { table =>
      val numRows = table.getRowCount.toInt
      val numCols = table.getNumberOfColumns
      val mb = getTableSizeMB(table)
      println(f"Loaded $numRows%,d rows x $numCols columns ($mb%.1f MB) from: $dataPath")
      println(s"Microbenchmark (GPU only): warmup=$warmup, measured=$measured")

      try {
        val times = runBenchmark(warmup, measured, profile = profile) {
          withResource(executeGpu(table, numRows)) { _ => }
        }
        val medianMs = times(times.length / 2) / 1e6
        val minMs = times(0) / 1e6
        println(
          f"   GPU  | $numRows%,14d rows | median $medianMs%10.3f ms | min $minMs%10.3f ms")
      } catch {
        case e: Exception =>
          System.err.println(s"GPU microbenchmark failed: ${e.getMessage}")
          e.printStackTrace(System.err)
          sys.exit(1)
      }
    }

    System.exit(0)
  }

  /**
   * Run warmup + measured iterations. Profile the measured iterations if enabled.
   * @return sorted array of measured elapsed times in nanoseconds
   */
  private def runBenchmark(warmup: Int, measured: Int, profile: Boolean = false)
      (block: => Unit): Array[Long] = {
    for (_ <- 0 until warmup) block
    (0 until measured).map { i =>
      if (profile) Cuda.profilerStart()
      val start = System.nanoTime()
      block
      val elapsed = System.nanoTime() - start
      if (profile) Cuda.profilerStop()
      elapsed
    }.toArray.sorted
  }

  /**
   * Read Parquet partition files from a directory into a cuDF Table.
   * Reads files in sorted order, stopping once maxRows is reached.
   * @param maxRows stop after accumulating this many rows; -1 means read all.
   */
  private def readParquetData(dataPath: String, maxRows: Int): Table = {
    val partFiles = new File(dataPath).listFiles((_, name) => name.endsWith(".parquet"))
    if (partFiles == null || partFiles.isEmpty) {
      throw new IllegalArgumentException(s"No .parquet files found in: $dataPath")
    }

    val tables = ArrayBuffer.empty[Table]
    var totalRows = 0L
    try {
      for (f <- partFiles.sorted if maxRows <= 0 || totalRows < maxRows) {
        val t = Table.readParquet(f)
        tables += t
        totalRows += t.getRowCount
      }
      val combined = if (tables.length == 1) tables(0)
        else Table.concatenate(tables.toArray: _*)
      withResource(combined) { src => limitTable(src, maxRows) }
    } finally {
      if (tables.length > 1) closeAll(tables.toArray)
    }
  }

  /** Return a new Table with at most numRows rows. */
  private def limitTable(table: Table, numRows: Int): Table = {
    val n = if (numRows <= 0) table.getRowCount.toInt
      else Math.min(numRows, table.getRowCount).toInt
    val cols = new Array[ColumnVector](table.getNumberOfColumns)
    try {
      for (i <- cols.indices) {
        cols(i) = table.getColumn(i).subVector(0, n)
      }
      new Table(cols: _*)
    } finally {
      closeAll(cols)
    }
  }

  /** Get the size of the table in MB. */
  private def getTableSizeMB(table: Table): Double = {
    (0 until table.getNumberOfColumns)
      .map(i => table.getColumn(i).getDeviceMemorySize)
      .sum / (1024.0 * 1024.0)
  }

  /** Parse CLI arguments. */
  private def parseArgs(args: Array[String]): Map[String, String] = {
    var map = Map.empty[String, String]
    var i = 0
    while (i < args.length) {
      args(i) match {
        case "--data-path"     => map += ("data-path" -> args(i + 1)); i += 2
        case "--warmup"        => map += ("warmup" -> args(i + 1)); i += 2
        case "--measured"      => map += ("measured" -> args(i + 1)); i += 2
        case "--rows"          => map += ("rows" -> args(i + 1)); i += 2
        case "--pool-fraction" => map += ("pool-fraction" -> args(i + 1)); i += 2
        case "--profile"       => map += ("profile" -> "true"); i += 1
        case other =>
          throw new IllegalArgumentException(s"Unknown argument: $other")
      }
    }
    map
  }
}
