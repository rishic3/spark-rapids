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

import ai.rapids.cudf.{DType, PartitionedTable, Table}
import ai.rapids.cudf.HostColumnVector.{BasicType, StructData, StructType => HostStructType}
import com.nvidia.spark.rapids.Arm.withResource
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.ExprId
import org.apache.spark.sql.rapids.{GpuAdd, GpuHashExpression, GpuMurmur3Hash}
import org.apache.spark.sql.types.{DataType, DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class GpuHashPartitioningSuite extends AnyFunSuite {
  private val numPartitions = 4
  private val primitiveTypes = Array[DataType](DoubleType, StringType, IntegerType)
  private val integerTypes = Array[DataType](IntegerType, StringType, IntegerType)
  private val nestedStructType = StructType(Array(
    StructField("id", IntegerType, nullable = true),
    StructField("value", DoubleType, nullable = true)))
  private val nestedTypes = Array[DataType](nestedStructType, StringType, IntegerType)
  private val negativeNaN = java.lang.Double.longBitsToDouble(0xfff0000000000001L)

  private case class TestHashPartitioner(override protected val hashFunc: GpuHashExpression)
      extends GpuHashPartitioner {
    def partitionInternalAndClose(batch: ColumnarBatch,
        outputTypes: Array[DataType],
        numPartitions: Int): (Array[Int], Array[GpuColumnVector]) = {
      val partedTable = hashPartitionAndClose(batch, numPartitions, NvtxRegistry.CALCULATE_PART)
      extractPartitionColumns(partedTable, outputTypes)
    }
  }

  private def boundRef(ordinal: Int, dataType: DataType,
      nullable: Boolean = true): GpuBoundReference = {
    GpuBoundReference(ordinal, dataType, nullable)(ExprId(ordinal.toLong), s"c$ordinal")
  }

  private def buildPrimitiveKeyBatch(): ColumnarBatch = {
    withResource(new Table.TestBuilder()
      .column(
        0.0.asInstanceOf[java.lang.Double],
        (-0.0).asInstanceOf[java.lang.Double],
        Double.NaN.asInstanceOf[java.lang.Double],
        negativeNaN.asInstanceOf[java.lang.Double],
        null.asInstanceOf[java.lang.Double],
        1.25.asInstanceOf[java.lang.Double],
        (-9.5).asInstanceOf[java.lang.Double],
        Double.PositiveInfinity.asInstanceOf[java.lang.Double],
        Double.NegativeInfinity.asInstanceOf[java.lang.Double],
        (-0.0).asInstanceOf[java.lang.Double])
      .column("zero", "neg-zero", "nan", null, "null", "one", "neg", "inf", "-inf", "repeat")
      .column(Int.box(0), Int.box(1), Int.box(2), Int.box(3), Int.box(4),
        Int.box(5), Int.box(6), Int.box(7), Int.box(8), Int.box(9))
      .build()) { table =>
      GpuColumnVector.from(table, primitiveTypes)
    }
  }

  private def buildIntegerKeyBatch(): ColumnarBatch = {
    withResource(new Table.TestBuilder()
      .column(
        10.asInstanceOf[java.lang.Integer],
        null.asInstanceOf[java.lang.Integer],
        30.asInstanceOf[java.lang.Integer],
        (-40).asInstanceOf[java.lang.Integer],
        50.asInstanceOf[java.lang.Integer],
        10.asInstanceOf[java.lang.Integer],
        70.asInstanceOf[java.lang.Integer],
        (-80).asInstanceOf[java.lang.Integer],
        90.asInstanceOf[java.lang.Integer],
        100.asInstanceOf[java.lang.Integer])
      .column("ten", "null", "thirty", "minus-forty", "fifty",
        "repeat", "seventy", "minus-eighty", "ninety", "hundred")
      .column(Int.box(0), Int.box(1), Int.box(2), Int.box(3), Int.box(4),
        Int.box(5), Int.box(6), Int.box(7), Int.box(8), Int.box(9))
      .build()) { table =>
      GpuColumnVector.from(table, integerTypes)
    }
  }

  private def buildNestedKeyBatch(): ColumnarBatch = {
    def struct(values: Object*): StructData = new StructData(values: _*)

    val hostStructType = new HostStructType(true,
      new BasicType(true, DType.INT32),
      new BasicType(true, DType.FLOAT64))

    withResource(new Table.TestBuilder()
      .column(hostStructType,
        struct(Int.box(1), Double.box(0.0)),
        struct(Int.box(1), Double.box(-0.0)),
        struct(Int.box(2), Double.box(Double.NaN)),
        struct(Int.box(2), Double.box(negativeNaN)),
        struct(null, Double.box(5.5)),
        null,
        struct(Int.box(3), null),
        struct(Int.box(3), Double.box(Double.PositiveInfinity)))
      .column("a", "b", "c", "d", "e", "f", "g", "h")
      .column(Int.box(0), Int.box(1), Int.box(2), Int.box(3),
        Int.box(4), Int.box(5), Int.box(6), Int.box(7))
      .build()) { table =>
      GpuColumnVector.from(table, nestedTypes)
    }
  }

  private def extractPartitionColumns(partedTable: PartitionedTable,
      outputTypes: Array[DataType]): (Array[Int], Array[GpuColumnVector]) = {
    withResource(partedTable) { pt =>
      val partitions = pt.getPartitions.dropRight(1)
      val table = pt.getTable
      val columns = (0 until pt.getNumberOfColumns.toInt).zip(outputTypes).map {
        case (index, sparkType) =>
          GpuColumnVector.from(table.getColumn(index).incRefCount(), sparkType)
      }
      (partitions, columns.toArray)
    }
  }

  private def legacyPartitionInternalAndClose(batch: ColumnarBatch,
      hashExpr: GpuHashExpression,
      outputTypes: Array[DataType],
      numPartitions: Int): (Array[Int], Array[GpuColumnVector]) = {
    withResource(batch) { cb =>
      val parts = withResource(hashExpr.columnarEval(cb)) { hash =>
        withResource(GpuScalar.from(numPartitions, IntegerType)) { partsLit =>
          hash.getBase.pmod(partsLit, DType.INT32)
        }
      }
      withResource(parts) { parts =>
        withResource(GpuColumnVector.from(cb)) { table =>
          extractPartitionColumns(table.partition(parts, numPartitions), outputTypes)
        }
      }
    }
  }

  private def rowIdsByPartition(result: (Array[Int], Array[GpuColumnVector]),
      rowIdOrdinal: Int,
      numRows: Int): Seq[Seq[Int]] = {
    withResource(result._2(rowIdOrdinal).getBase.copyToHost()) { rowIds =>
      result._1.indices.map { partIndex =>
        val start = result._1(partIndex)
        val end = if (partIndex + 1 < result._1.length) {
          result._1(partIndex + 1)
        } else {
          numRows
        }
        (start until end).map(i => rowIds.getInt(i)).sorted
      }
    }
  }

  private def closePartitionResult(result: (Array[Int], Array[GpuColumnVector])): Unit = {
    result._2.foreach(_.close())
  }

  private def canUseFusedPath(hashExpr: GpuHashExpression): Boolean = hashExpr match {
    case GpuMurmur3Hash(children, _) =>
      GpuProjectExec.extractSingleBoundIndex(children).nonEmpty &&
        GpuProjectExec.extractSingleBoundIndex(children).forall(_.isDefined)
    case _ => false
  }

  private def assertMatchesLegacy(
      batchBuilder: () => ColumnarBatch,
      hashExpr: GpuHashExpression,
      outputTypes: Array[DataType],
      rowIdOrdinal: Int,
      numRows: Int): Unit = {
    val partitioner = TestHashPartitioner(hashExpr)
    val actual = partitioner.partitionInternalAndClose(batchBuilder(), outputTypes, numPartitions)
    val expected = try {
      legacyPartitionInternalAndClose(batchBuilder(), hashExpr, outputTypes, numPartitions)
    } catch {
      case t: Throwable =>
        closePartitionResult(actual)
        throw t
    }
    try {
      assertResult(expected._1.toSeq)(actual._1.toSeq)
      assertResult(rowIdsByPartition(expected, rowIdOrdinal, numRows)) {
        rowIdsByPartition(actual, rowIdOrdinal, numRows)
      }
    } finally {
      closePartitionResult(actual)
      closePartitionResult(expected)
    }
  }

  test("fused Murmur3 hash partition matches legacy path for direct bound keys") {
    val hashExpr = GpuMurmur3Hash(Seq(
      boundRef(0, DoubleType),
      boundRef(1, StringType)), GpuHashPartitioningBase.DEFAULT_HASH_SEED)
    assert(canUseFusedPath(hashExpr))
    assertMatchesLegacy(() => buildPrimitiveKeyBatch(), hashExpr, primitiveTypes,
      rowIdOrdinal = 2, numRows = 10)
  }

  test("fused Murmur3 hash partition matches legacy path for nested bound keys") {
    val hashExpr = GpuMurmur3Hash(Seq(boundRef(0, nestedStructType)),
      GpuHashPartitioningBase.DEFAULT_HASH_SEED)
    assert(canUseFusedPath(hashExpr))
    assertMatchesLegacy(() => buildNestedKeyBatch(), hashExpr, nestedTypes,
      rowIdOrdinal = 2, numRows = 8)
  }

  test("fused Murmur3 hash partition preserves non-default seeds") {
    val hashExpr = GpuMurmur3Hash(Seq(boundRef(0, IntegerType), boundRef(1, StringType)), 100)
    assert(canUseFusedPath(hashExpr))
    assertMatchesLegacy(() => buildIntegerKeyBatch(), hashExpr, integerTypes,
      rowIdOrdinal = 2, numRows = 10)
  }

  test("computed Murmur3 hash keys stay on the legacy partition path") {
    val hashExpr = GpuMurmur3Hash(Seq(
      GpuAdd(boundRef(0, IntegerType), GpuLiteral(1, IntegerType), failOnError = false)()),
      GpuHashPartitioningBase.DEFAULT_HASH_SEED)
    assert(!canUseFusedPath(hashExpr))
    assertMatchesLegacy(() => buildIntegerKeyBatch(), hashExpr, integerTypes,
      rowIdOrdinal = 2, numRows = 10)
  }
}
