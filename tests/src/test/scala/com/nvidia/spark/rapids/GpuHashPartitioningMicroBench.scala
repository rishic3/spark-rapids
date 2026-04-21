/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import java.util.function.Consumer

import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{ColumnVector => CudfColumnVector, Cuda, DType, HostColumnVector, PartitionedTable, Rmm, RmmAllocationMode, Table}
import com.nvidia.spark.rapids.Arm.withResource

import org.apache.spark.sql.catalyst.expressions.ExprId
import org.apache.spark.sql.rapids.{GpuHashExpression, GpuMurmur3Hash}
import org.apache.spark.sql.types.{DataType, DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

object GpuHashPartitioningMicroBench {
  private val DefaultRows = 2000000
  private val DefaultWarmup = 2
  private val DefaultMeasured = 5
  private val DefaultPoolFraction = 0.8f
  private val NumPartitions = 200
  private val NegativeNaN = java.lang.Double.longBitsToDouble(0xfff0000000000001L)
  private val DoublePattern = Array[Double](
    0.0d,
    -0.0d,
    Double.NaN,
    NegativeNaN,
    Double.PositiveInfinity,
    Double.NegativeInfinity,
    1.25d,
    -9.5d,
    42.0d,
    -42.0d,
    512.5d,
    -2048.125d)
  private val nestedStructType = StructType(Array(
    StructField("id", IntegerType, nullable = true),
    StructField("value", DoubleType, nullable = true)))

  private case class VariantResult(label: String, medianMs: Double, minMs: Double)

  private case class BenchmarkSpec(
      label: String,
      buildSourceTable: Int => Table,
      outputTypes: Array[DataType],
      hashExpr: GpuHashExpression,
      rowIdOrdinal: Int)

  private case class BenchmarkHashPartitioner(override protected val hashFunc: GpuHashExpression)
      extends GpuHashPartitioner {
    def partition(batch: ColumnarBatch, numPartitions: Int): PartitionedTable = {
      hashPartitionAndClose(batch, numPartitions, NvtxRegistry.CALCULATE_PART)
    }
  }

  def main(args: Array[String]): Unit = {
    val parsed = parseArgs(args)
    val rows = parsed.getOrElse("rows", DefaultRows.toString).toInt
    val warmup = parsed.getOrElse("warmup", DefaultWarmup.toString).toInt
    val measured = parsed.getOrElse("measured", DefaultMeasured.toString).toInt
    val poolFraction = parsed.getOrElse("pool-fraction", DefaultPoolFraction.toString).toFloat
    val caseFilter = parsed.getOrElse("case", "all")
    val profile = parsed.contains("profile")

    if (!Rmm.isInitialized()) {
      val memInfo = Cuda.memGetInfo()
      val poolSize = (memInfo.free * poolFraction).toLong & ~255L
      Rmm.initialize(RmmAllocationMode.POOL, null, poolSize)
    }

    val allCases = Seq(
      BenchmarkSpec(
        label = "double_string_nulls",
        buildSourceTable = rows => buildDoubleStringTable(rows, doubleNullPeriod = 17, stringNullPeriod = 23),
        outputTypes = Array[DataType](DoubleType, StringType, IntegerType),
        hashExpr = GpuMurmur3Hash(Seq(boundRef(0, DoubleType), boundRef(1, StringType)),
          GpuHashPartitioningBase.DEFAULT_HASH_SEED),
        rowIdOrdinal = 2),
      BenchmarkSpec(
        label = "double_string_no_nulls",
        buildSourceTable = rows => buildDoubleStringTable(rows, doubleNullPeriod = 0, stringNullPeriod = 0),
        outputTypes = Array[DataType](DoubleType, StringType, IntegerType),
        hashExpr = GpuMurmur3Hash(Seq(boundRef(0, DoubleType), boundRef(1, StringType)),
          GpuHashPartitioningBase.DEFAULT_HASH_SEED),
        rowIdOrdinal = 2),
      BenchmarkSpec(
        label = "int_string_seed100",
        buildSourceTable = rows => buildIntStringTable(rows, intNullPeriod = 19, stringNullPeriod = 29),
        outputTypes = Array[DataType](IntegerType, StringType, IntegerType),
        hashExpr = GpuMurmur3Hash(Seq(boundRef(0, IntegerType), boundRef(1, StringType)), 100),
        rowIdOrdinal = 2),
      BenchmarkSpec(
        label = "struct_key",
        buildSourceTable = rows => buildStructTable(rows, intNullPeriod = 13, doubleNullPeriod = 17),
        outputTypes = Array[DataType](nestedStructType, IntegerType),
        hashExpr = GpuMurmur3Hash(Seq(boundRef(0, nestedStructType)),
          GpuHashPartitioningBase.DEFAULT_HASH_SEED),
        rowIdOrdinal = 1),
      BenchmarkSpec(
        label = "three_keys",
        buildSourceTable = rows => buildThreeKeyTable(
          rows, intNullPeriod = 19, doubleNullPeriod = 17, stringNullPeriod = 23),
        outputTypes = Array[DataType](IntegerType, DoubleType, StringType, IntegerType),
        hashExpr = GpuMurmur3Hash(Seq(
          boundRef(0, IntegerType),
          boundRef(1, DoubleType),
          boundRef(2, StringType)), GpuHashPartitioningBase.DEFAULT_HASH_SEED),
        rowIdOrdinal = 3)
    )

    val cases = caseFilter match {
      case "all" => allCases
      case one => allCases.filter(_.label == one)
    }
    require(cases.nonEmpty, s"Unknown --case value: $caseFilter")

    println(s"Hash partitioning microbenchmark: rows=$rows warmup=$warmup measured=$measured cases=${cases.size}")
    println(s"numPartitions=$NumPartitions")

    val summaries = ArrayBuffer.empty[(String, VariantResult, VariantResult)]
    cases.foreach { spec =>
      withResource(spec.buildSourceTable(rows)) { sourceTable =>
        assertEquivalent(sourceTable, spec)
        println(s"[${spec.label}] output check passed: rows=${sourceTable.getRowCount}")

        val baseline = runVariant(spec.label, "legacy", warmup, measured, profile) {
          withResource(legacyHashPartitionAndClose(
            batchFromSource(sourceTable, spec.outputTypes), spec.hashExpr, NumPartitions)) { _ => }
        }
        val optimized = runVariant(spec.label, "fused", warmup, measured, profile) {
          withResource(BenchmarkHashPartitioner(spec.hashExpr).partition(
            batchFromSource(sourceTable, spec.outputTypes), NumPartitions)) { _ => }
        }
        val speedup = baseline.medianMs / optimized.medianMs
        println(f"[${spec.label}] speedup fused-vs-legacy: $speedup%.4fx")
        summaries += ((spec.label, baseline, optimized))
      }
    }

    println("\n=== Summary (lower is better) ===")
    println(f"${"case"}%-24s ${"legacy_med"}%14s ${"fused_med"}%14s ${"speedup"}%10s")
    summaries.foreach { case (label, legacy, fused) =>
      val speedup = legacy.medianMs / fused.medianMs
      println(f"$label%-24s ${legacy.medianMs}%14.3f ${fused.medianMs}%14.3f $speedup%10.4f")
    }
  }

  private def runVariant(caseLabel: String,
      variantLabel: String,
      warmup: Int,
      measured: Int,
      profile: Boolean)(block: => Unit): VariantResult = {
    val times = runBenchmark(warmup, measured, profile)(block)
    val medianMs = times(times.length / 2) / 1e6
    val minMs = times(0) / 1e6
    println(f"[$caseLabel] $variantLabel%-7s | median $medianMs%10.3f ms | min $minMs%10.3f ms")
    VariantResult(variantLabel, medianMs, minMs)
  }

  private def runBenchmark(warmup: Int, measured: Int, profile: Boolean)(block: => Unit): Array[Long] = {
    for (_ <- 0 until warmup) {
      block
    }
    (0 until measured).map { _ =>
      if (profile) {
        Cuda.profilerStart()
      }
      val start = System.nanoTime()
      block
      val elapsed = System.nanoTime() - start
      if (profile) {
        Cuda.profilerStop()
      }
      elapsed
    }.toArray.sorted
  }

  private def assertEquivalent(sourceTable: Table, spec: BenchmarkSpec): Unit = {
    withResource(legacyHashPartitionAndClose(batchFromSource(sourceTable, spec.outputTypes),
        spec.hashExpr, NumPartitions)) { legacy =>
      withResource(BenchmarkHashPartitioner(spec.hashExpr).partition(
          batchFromSource(sourceTable, spec.outputTypes), NumPartitions)) { fused =>
        val (legacyParts, legacyRows) = partitionedRowIds(legacy, spec.rowIdOrdinal)
        val (fusedParts, fusedRows) = partitionedRowIds(fused, spec.rowIdOrdinal)
        require(legacyParts.sameElements(fusedParts),
          s"Partition offsets differ: legacy=${legacyParts.mkString("[", ",", "]")}, " +
            s"fused=${fusedParts.mkString("[", ",", "]")}")
        require(legacyRows == fusedRows, s"Partition membership differs for ${spec.label}")
      }
    }
  }

  private def legacyHashPartitionAndClose(batch: ColumnarBatch,
      hashExpr: GpuHashExpression,
      numPartitions: Int): PartitionedTable = {
    withResource(batch) { cb =>
      val parts = withResource(hashExpr.columnarEval(cb)) { hash =>
        withResource(GpuScalar.from(numPartitions, IntegerType)) { partsLit =>
          hash.getBase.pmod(partsLit, DType.INT32)
        }
      }
      withResource(parts) { parts =>
        withResource(GpuColumnVector.from(cb)) { table =>
          table.partition(parts, numPartitions)
        }
      }
    }
  }

  private def partitionedRowIds(partitionedTable: PartitionedTable,
      rowIdOrdinal: Int): (Array[Int], Seq[Seq[Int]]) = {
    withResource(partitionedTable.getColumn(rowIdOrdinal).copyToHost()) { rowIds =>
      val partitions = partitionedTable.getPartitions.clone()
      val grouped = partitions.indices.dropRight(1).map { partIndex =>
        val start = partitions(partIndex)
        val end = partitions(partIndex + 1)
        val values = ArrayBuffer.empty[Int]
        var rowIndex = start
        while (rowIndex < end) {
          values += rowIds.getInt(rowIndex)
          rowIndex += 1
        }
        values.sorted.toSeq
      }
      (partitions, grouped)
    }
  }

  private def batchFromSource(sourceTable: Table, outputTypes: Array[DataType]): ColumnarBatch = {
    val columns: Array[org.apache.spark.sql.vectorized.ColumnVector] =
      outputTypes.indices.map { index =>
        GpuColumnVector.from(sourceTable.getColumn(index).incRefCount(), outputTypes(index)):
          org.apache.spark.sql.vectorized.ColumnVector
      }.toArray
    new ColumnarBatch(columns, sourceTable.getRowCount.toInt)
  }

  private def boundRef(ordinal: Int, dataType: DataType): GpuBoundReference = {
    GpuBoundReference(ordinal, dataType, nullable = true)(ExprId(ordinal.toLong), s"c$ordinal")
  }

  private def buildDoubleStringTable(numRows: Int,
      doubleNullPeriod: Int,
      stringNullPeriod: Int): Table = {
    withResource(buildDoubleColumn(numRows, doubleNullPeriod)) { doubleKey =>
      withResource(buildStringColumn(numRows, stringNullPeriod, "key-", 4096)) { stringKey =>
        withResource(buildRowIdColumn(numRows)) { rowId =>
          new Table(doubleKey, stringKey, rowId)
        }
      }
    }
  }

  private def buildIntStringTable(numRows: Int,
      intNullPeriod: Int,
      stringNullPeriod: Int): Table = {
    withResource(buildIntColumn(numRows, intNullPeriod, modulus = 4096, offset = 2048)) { intKey =>
      withResource(buildStringColumn(numRows, stringNullPeriod, "join-", 8192)) { stringKey =>
        withResource(buildRowIdColumn(numRows)) { rowId =>
          new Table(intKey, stringKey, rowId)
        }
      }
    }
  }

  private def buildStructTable(numRows: Int,
      intNullPeriod: Int,
      doubleNullPeriod: Int): Table = {
    withResource(buildIntColumn(numRows, intNullPeriod, modulus = 1024, offset = 512)) { childInt =>
      withResource(buildDoubleColumn(numRows, doubleNullPeriod)) { childDouble =>
        withResource(CudfColumnVector.makeStruct(childInt, childDouble)) { structKey =>
          withResource(buildRowIdColumn(numRows)) { rowId =>
            new Table(structKey, rowId)
          }
        }
      }
    }
  }

  private def buildThreeKeyTable(numRows: Int,
      intNullPeriod: Int,
      doubleNullPeriod: Int,
      stringNullPeriod: Int): Table = {
    withResource(buildIntColumn(numRows, intNullPeriod, modulus = 4096, offset = 2048)) { intKey =>
      withResource(buildDoubleColumn(numRows, doubleNullPeriod)) { doubleKey =>
        withResource(buildStringColumn(numRows, stringNullPeriod, "mix-", 4096)) { stringKey =>
          withResource(buildRowIdColumn(numRows)) { rowId =>
            new Table(intKey, doubleKey, stringKey, rowId)
          }
        }
      }
    }
  }

  private def buildRowIdColumn(numRows: Int): CudfColumnVector = {
    buildIntColumn(numRows, nullPeriod = 0, modulus = Int.MaxValue, offset = 0)
  }

  private def buildIntColumn(numRows: Int,
      nullPeriod: Int,
      modulus: Int,
      offset: Int): CudfColumnVector = {
    CudfColumnVector.build(DType.INT32, numRows, new Consumer[HostColumnVector.Builder] {
      override def accept(builder: HostColumnVector.Builder): Unit = {
        var rowIndex = 0
        while (rowIndex < numRows) {
          if (nullPeriod > 0 && rowIndex % nullPeriod == 0) {
            builder.appendNull()
          } else {
            builder.append((rowIndex % modulus) - offset)
          }
          rowIndex += 1
        }
      }
    })
  }

  private def buildDoubleColumn(numRows: Int, nullPeriod: Int): CudfColumnVector = {
    CudfColumnVector.build(DType.FLOAT64, numRows, new Consumer[HostColumnVector.Builder] {
      override def accept(builder: HostColumnVector.Builder): Unit = {
        var rowIndex = 0
        while (rowIndex < numRows) {
          if (nullPeriod > 0 && rowIndex % nullPeriod == 0) {
            builder.appendNull()
          } else {
            val slot = rowIndex % 16
            if (slot < DoublePattern.length) {
              builder.append(DoublePattern(slot))
            } else {
              builder.append(((rowIndex % 131071).toDouble - 65535.0d) / 16.0d)
            }
          }
          rowIndex += 1
        }
      }
    })
  }

  private def buildStringColumn(numRows: Int,
      nullPeriod: Int,
      prefix: String,
      modulus: Int): CudfColumnVector = {
    val estimatedStringBytes = numRows.toLong * math.max(prefix.length + 8, 16)
    CudfColumnVector.build(numRows, estimatedStringBytes, new Consumer[HostColumnVector.Builder] {
      override def accept(builder: HostColumnVector.Builder): Unit = {
        var rowIndex = 0
        while (rowIndex < numRows) {
          if (nullPeriod > 0 && rowIndex % nullPeriod == 0) {
            builder.appendNull()
          } else {
            builder.append(prefix + (rowIndex % modulus))
          }
          rowIndex += 1
        }
      }
    })
  }

  private def parseArgs(args: Array[String]): Map[String, String] = {
    var map = Map.empty[String, String]
    var index = 0
    while (index < args.length) {
      args(index) match {
        case "--rows" =>
          map += ("rows" -> args(index + 1))
          index += 2
        case "--warmup" =>
          map += ("warmup" -> args(index + 1))
          index += 2
        case "--measured" =>
          map += ("measured" -> args(index + 1))
          index += 2
        case "--pool-fraction" =>
          map += ("pool-fraction" -> args(index + 1))
          index += 2
        case "--case" =>
          map += ("case" -> args(index + 1))
          index += 2
        case "--profile" =>
          map += ("profile" -> "true")
          index += 1
        case other =>
          throw new IllegalArgumentException(s"Unknown argument: $other")
      }
    }
    map
  }
}
