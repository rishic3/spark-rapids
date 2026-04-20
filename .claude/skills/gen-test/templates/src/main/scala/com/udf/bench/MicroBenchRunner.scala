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
 * This is the only runner used by the optimize loop — it isolates the
 * `evaluateColumnar` libcudf work from Spark overhead. Spark-level A/B (with
 * the actual plugin jar) is owned by `bench/spark_bench.sh` and only runs
 * during the operator-backport skill, never inside the optimize loop.
 *
 * Two modes (single JVM, single RMM pool, single cuDF native load):
 *   --data-root DIR   sweep every subdirectory of DIR (one Parquet dataset
 *                     per advertised type — produced by bench/gen_data.sh).
 *                     Prints one min/median row per case and a final table.
 *   --data-path DIR   single Parquet dataset (used with --profile so an nsys
 *                     report contains exactly one case).
 *
 * Per case:
 *   Read Parquet (via cuDF Table.readParquet) once, off-clock
 *   -> warmup `evaluateColumnar` runs
 *   -> measured `evaluateColumnar` runs
 *   -> report min / median ms
 *
 * Usage:
 *   mvn exec:java -Dexec.mainClass=com.udf.bench.MicroBenchRunner \
 *     -Dexec.args="--data-root data --rows 10000000"
 *
 *   mvn exec:java -Dexec.mainClass=com.udf.bench.MicroBenchRunner \
 *     -Dexec.args="--data-path data/long --rows 10000000 --profile"
 */
object MicroBenchRunner {

  private val DefaultWarmup = 2
  private val DefaultMeasured = 4
  private val DefaultRmmAllocFraction = 0.9f

  /**
   * TODO: Execute the extracted RapidsUDF via `evaluateColumnar`.
   *
   * `bench/gen_data.sh` writes only the operator's input columns (no synthetic
   * `id`), so the columns in `table` align 1:1 with `args: ColumnVector*`.
   *
   * Example (unary string op, schema = [s]):
   * {{{
   *   val udf = new com.udf.<Op>StringUDF()
   *   udf.evaluateColumnar(numRows, table.getColumn(0))
   * }}}
   *
   * Example (binary op, schema = [a, b]):
   * {{{
   *   val udf = new com.udf.<Op>LongUDF()
   *   udf.evaluateColumnar(numRows, table.getColumn(0), table.getColumn(1))
   * }}}
   *
   * @param table   dataset loaded on GPU
   * @param numRows number of rows in the dataset
   * @return result ColumnVector — caller closes it
   */
  def executeGpu(table: Table, numRows: Int): ColumnVector = ???

  def main(args: Array[String]): Unit = {
    val parsed = parseArgs(args)

    val dataPathOpt = parsed.get("data-path")
    val dataRootOpt = parsed.get("data-root")
    val maxRows = parsed.getOrElse("rows", "-1").toInt
    val rmmAllocFraction = parsed.getOrElse("pool-fraction", DefaultRmmAllocFraction.toString).toFloat
    val warmup = parsed.getOrElse("warmup", DefaultWarmup.toString).toInt
    val measured = parsed.getOrElse("measured", DefaultMeasured.toString).toInt
    val profile = parsed.contains("profile")

    val cases: Seq[(String, String)] = (dataPathOpt, dataRootOpt) match {
      case (Some(p), None) =>
        Seq(new File(p).getName -> p)
      case (None, Some(root)) =>
        if (profile) throw new IllegalArgumentException(
          "--profile requires --data-path (single case); --data-root would mix cases in one nsys report")
        val rootFile = new File(root)
        val subdirs = Option(rootFile.listFiles()).getOrElse(Array.empty)
          .filter(_.isDirectory).sortBy(_.getName)
        if (subdirs.isEmpty) throw new IllegalArgumentException(
          s"No case subdirectories found under --data-root $root")
        subdirs.toSeq.map(d => d.getName -> d.getAbsolutePath)
      case _ =>
        throw new IllegalArgumentException("Pass exactly one of --data-path or --data-root")
    }

    if (!Rmm.isInitialized()) {
      val memInfo = Cuda.memGetInfo()
      val poolSize = (memInfo.free * rmmAllocFraction).toLong & ~255L
      Rmm.initialize(RmmAllocationMode.POOL, null, poolSize)
    }

    println(s"Microbenchmark (GPU only): warmup=$warmup, measured=$measured, cases=${cases.size}")

    val results = cases.map { case (label, path) =>
      withResource(readParquetData(path, maxRows)) { table =>
        val numRows = table.getRowCount.toInt
        val numCols = table.getNumberOfColumns
        val mb = getTableSizeMB(table)
        println(f"\n[$label] loaded $numRows%,d rows x $numCols cols ($mb%.1f MB) from $path")
        try {
          val times = runBenchmark(warmup, measured, profile = profile) {
            withResource(executeGpu(table, numRows)) { _ => }
          }
          val medianMs = times(times.length / 2) / 1e6
          val minMs = times(0) / 1e6
          println(
            f"[$label] GPU  | $numRows%,14d rows | median $medianMs%10.3f ms | min $minMs%10.3f ms")
          (label, numRows, medianMs, minMs, None: Option[String])
        } catch {
          case e: Exception =>
            System.err.println(s"[$label] GPU microbenchmark failed: ${e.getMessage}")
            e.printStackTrace(System.err)
            (label, 0, Double.NaN, Double.NaN, Some(e.getClass.getSimpleName))
        }
      }
    }

    if (cases.size > 1) {
      println("\n=== Microbenchmark summary (median ms, lower is better) ===")
      println(f"${"case"}%-20s ${"rows"}%14s ${"median_ms"}%12s ${"min_ms"}%12s  status")
      results.foreach { case (label, n, med, min, err) =>
        val status = err.getOrElse("ok")
        println(f"$label%-20s $n%,14d $med%12.3f $min%12.3f  $status")
      }
    }

    if (results.exists(_._5.isDefined)) sys.exit(1)
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
        case "--data-root"     => map += ("data-root" -> args(i + 1)); i += 2
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
