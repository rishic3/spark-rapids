/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.bench;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import ai.rapids.cudf.ArrowIPCOptions;
import ai.rapids.cudf.ArrowIPCWriterOptions;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.CudaMemInfo;
import ai.rapids.cudf.HostBufferConsumer;
import ai.rapids.cudf.HostBufferProvider;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.StreamedTableReader;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.TableWriter;

/**
 * JVM-side microbenchmark decomposing Pandas UDF transfer overhead into the exact
 * JNI calls used by GpuArrowWriter / GpuArrowReader in the Spark RAPIDS plugin.
 *
 * Write path (GpuArrowWriter.write → ArrowIPCTableWriter.write):
 *   convertCudfToArrowTable   — D2H: GPU cudf table → host Arrow table
 *   writeArrowIPCArrowChunk   — IPC serialize: host Arrow table → IPC bytes
 *
 * Read path (GpuArrowReader.readNext → StreamedTableReader.getNextIfAvailable):
 *   readArrowIPCChunkToArrowTable — IPC deserialize: IPC bytes → host Arrow table
 *   convertArrowTableToCudf       — H2D: host Arrow table → GPU cudf table
 *
 * The split between phases is measured via the DoneOnGpu / NeedGpu callbacks that
 * fire between the two JNI calls in each direction.
 *
 * Usage:
 *   mvn compile exec:java -Dexec.mainClass=com.nvidia.bench.TransferOverheadBench \
 *     -Dexec.args="--data-path /path/to/parquet --rows 1000000 --batch-size 100000"
 */
public class TransferOverheadBench {

    private static final int DEFAULT_WARMUP = 3;
    private static final int DEFAULT_MEASURED = 5;
    private static final float DEFAULT_RMM_FRACTION = 0.5f;

    // ─── In-memory Arrow IPC helpers ────────────────────────────────────────

    /**
     * Captures Arrow IPC bytes into a byte[] instead of writing to a socket.
     * Mirrors BufferToStreamWriter in GpuArrowWriter.scala.
     */
    static class InMemoryBufferConsumer implements HostBufferConsumer {
        private final ByteArrayOutputStream baos = new ByteArrayOutputStream();

        @Override
        public void handleBuffer(HostMemoryBuffer buffer, long length) {
            try {
                byte[] temp = new byte[(int) Math.min(128 * 1024, length)];
                long offset = 0;
                while (offset < length) {
                    int toRead = (int) Math.min(temp.length, length - offset);
                    buffer.getBytes(temp, 0, offset, toRead);
                    baos.write(temp, 0, toRead);
                    offset += toRead;
                }
            } finally {
                buffer.close();
            }
        }

        byte[] toByteArray() { return baos.toByteArray(); }
    }

    /**
     * Reads Arrow IPC bytes from a byte[] instead of a socket.
     * Mirrors StreamToBufferProvider in GpuArrowReader.scala.
     */
    static class InMemoryBufferProvider implements HostBufferProvider {
        private final byte[] data;
        private int pos = 0;

        InMemoryBufferProvider(byte[] data) { this.data = data; }

        @Override
        public long readInto(HostMemoryBuffer buffer, long length) {
            int remaining = data.length - pos;
            if (remaining <= 0) return 0;
            int toRead = (int) Math.min(length, remaining);
            buffer.setBytes(0, data, pos, toRead);
            pos += toRead;
            return toRead;
        }
    }

    // ─── Benchmark results ──────────────────────────────────────────────────

    /** Holds per-iteration timings for a two-phase operation (sorted independently). */
    static class SplitResult {
        final long[] totalNs;
        final long[] phase1Ns;
        final long[] phase2Ns;

        SplitResult(long[] totalNs, long[] phase1Ns, long[] phase2Ns) {
            this.totalNs = totalNs;
            this.phase1Ns = phase1Ns;
            this.phase2Ns = phase2Ns;
            Arrays.sort(this.totalNs);
            Arrays.sort(this.phase1Ns);
            Arrays.sort(this.phase2Ns);
        }

        double medianS(long[] a) { return a[a.length / 2] / 1e9; }
        double bestS(long[] a)   { return a[0] / 1e9; }
    }

    // ─── Benchmark operations ───────────────────────────────────────────────

    /**
     * Arrow IPC write with split timing.
     *
     * Inside ArrowIPCTableWriter.write(table):
     *   phase1: convertCudfToArrowTable  (D2H — GPU cudf → host Arrow)
     *   --- DoneOnGpu callback fires here ---
     *   phase2: writeArrowIPCArrowChunk  (IPC serialize — host Arrow → IPC bytes)
     */
    static SplitResult benchArrowWrite(Table[] batches, int numCols, int warmup, int measured) {
        final long[] splitNanos = new long[1];

        ArrowIPCWriterOptions.Builder builder = ArrowIPCWriterOptions.builder();
        builder.withCallback(table -> splitNanos[0] = System.nanoTime());
        for (int i = 0; i < numCols; i++) builder.withColumnNames("col" + i);
        ArrowIPCWriterOptions opts = builder.build();

        for (int w = 0; w < warmup; w++) {
            InMemoryBufferConsumer consumer = new InMemoryBufferConsumer();
            try (TableWriter writer = Table.writeArrowIPCChunked(opts, consumer)) {
                for (Table batch : batches) writer.write(batch);
            }
            syncGpu();
        }

        long[] totalNs = new long[measured];
        long[] phase1Ns = new long[measured]; // convertCudfToArrowTable
        long[] phase2Ns = new long[measured]; // writeArrowIPCArrowChunk

        for (int m = 0; m < measured; m++) {
            syncGpu();
            long p1Accum = 0, p2Accum = 0;
            InMemoryBufferConsumer consumer = new InMemoryBufferConsumer();
            try (TableWriter writer = Table.writeArrowIPCChunked(opts, consumer)) {
                for (Table batch : batches) {
                    long t0 = System.nanoTime();
                    writer.write(batch);
                    long t2 = System.nanoTime();
                    p1Accum += splitNanos[0] - t0;
                    p2Accum += t2 - splitNanos[0];
                }
            }
            syncGpu();
            phase1Ns[m] = p1Accum;
            phase2Ns[m] = p2Accum;
            totalNs[m] = p1Accum + p2Accum;
        }
        return new SplitResult(totalNs, phase1Ns, phase2Ns);
    }

    /**
     * Arrow IPC read with split timing.
     *
     * Inside StreamedTableReader.getNextIfAvailable(rowTarget):
     *   phase1: readArrowIPCChunkToArrowTable  (IPC deserialize — IPC bytes → host Arrow)
     *   --- NeedGpu callback fires here ---
     *   phase2: convertArrowTableToCudf        (H2D — host Arrow → GPU cudf)
     */
    static SplitResult benchArrowRead(byte[] ipcBytes, int batchSize, int warmup, int measured) {
        final long[] splitNanos = new long[1];

        ArrowIPCOptions opts = ArrowIPCOptions.builder()
            .withCallback(() -> splitNanos[0] = System.nanoTime())
            .build();

        for (int w = 0; w < warmup; w++) {
            InMemoryBufferProvider provider = new InMemoryBufferProvider(ipcBytes);
            try (StreamedTableReader reader = Table.readArrowIPCChunked(opts, provider)) {
                Table t;
                while ((t = reader.getNextIfAvailable(batchSize)) != null) t.close();
            }
            syncGpu();
        }

        long[] totalNs = new long[measured];
        long[] phase1Ns = new long[measured]; // readArrowIPCChunkToArrowTable
        long[] phase2Ns = new long[measured]; // convertArrowTableToCudf

        for (int m = 0; m < measured; m++) {
            syncGpu();
            long p1Accum = 0, p2Accum = 0;
            InMemoryBufferProvider provider = new InMemoryBufferProvider(ipcBytes);
            try (StreamedTableReader reader = Table.readArrowIPCChunked(opts, provider)) {
                while (true) {
                    long t0 = System.nanoTime();
                    Table t = reader.getNextIfAvailable(batchSize);
                    long t2 = System.nanoTime();
                    if (t == null) break;
                    t.close();
                    p1Accum += splitNanos[0] - t0;
                    p2Accum += t2 - splitNanos[0];
                }
            }
            syncGpu();
            phase1Ns[m] = p1Accum;
            phase2Ns[m] = p2Accum;
            totalNs[m] = p1Accum + p2Accum;
        }
        return new SplitResult(totalNs, phase1Ns, phase2Ns);
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    static byte[] writeToIpcBytes(Table[] batches) {
        int numCols = batches[0].getNumberOfColumns();
        ArrowIPCWriterOptions.Builder builder = ArrowIPCWriterOptions.builder();
        for (int i = 0; i < numCols; i++) builder.withColumnNames("col" + i);
        InMemoryBufferConsumer consumer = new InMemoryBufferConsumer();
        try (TableWriter writer = Table.writeArrowIPCChunked(builder.build(), consumer)) {
            for (Table batch : batches) writer.write(batch);
        }
        return consumer.toByteArray();
    }

    static Table[] splitIntoBatches(Table table, int batchSize) {
        int totalRows = (int) table.getRowCount();
        int numCols = table.getNumberOfColumns();
        if (batchSize <= 0 || batchSize >= totalRows) {
            return new Table[] { table };
        }
        int numBatches = (totalRows + batchSize - 1) / batchSize;
        Table[] batches = new Table[numBatches];
        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;
            int end = Math.min(start + batchSize, totalRows);
            ColumnVector[] cols = new ColumnVector[numCols];
            try {
                for (int c = 0; c < numCols; c++) {
                    cols[c] = table.getColumn(c).subVector(start, end);
                }
                batches[b] = new Table(cols);
            } finally {
                closeAll(cols);
            }
        }
        return batches;
    }

    static double getDataSizeMB(Table[] batches) {
        long bytes = 0;
        for (Table batch : batches) {
            for (int i = 0; i < batch.getNumberOfColumns(); i++) {
                bytes += batch.getColumn(i).getDeviceMemorySize();
            }
        }
        return bytes / (1024.0 * 1024.0);
    }

    static Table readParquetData(String dataPath, int maxRows) {
        File path = new File(dataPath);
        File[] partFiles;
        if (path.isFile()) {
            partFiles = new File[] { path };
        } else if (path.isDirectory()) {
            partFiles = path.listFiles((dir, name) -> name.endsWith(".parquet"));
            if (partFiles == null || partFiles.length == 0) {
                throw new IllegalArgumentException("No .parquet files found in: " + dataPath);
            }
            Arrays.sort(partFiles);
        } else {
            throw new IllegalArgumentException("Not a file or directory: " + dataPath);
        }

        Table[] tables = new Table[partFiles.length];
        int count = 0;
        long totalRows = 0;
        try {
            for (int i = 0; i < partFiles.length; i++) {
                tables[i] = Table.readParquet(partFiles[i]);
                count++;
                totalRows += tables[i].getRowCount();
                if (maxRows > 0 && totalRows >= maxRows) break;
            }
            Table combined = (count == 1) ? tables[0] : Table.concatenate(Arrays.copyOf(tables, count));
            try (Table src = combined) {
                return limitTable(src, maxRows);
            }
        } finally {
            if (count > 1) {
                for (int i = 0; i < count; i++) {
                    if (tables[i] != null) {
                        try { tables[i].close(); } catch (Exception ignore) {}
                    }
                }
            }
        }
    }

    private static Table limitTable(Table table, int maxRows) {
        int n = (maxRows <= 0)
            ? (int) table.getRowCount()
            : (int) Math.min(maxRows, table.getRowCount());
        ColumnVector[] cols = new ColumnVector[table.getNumberOfColumns()];
        try {
            for (int i = 0; i < cols.length; i++) {
                cols[i] = table.getColumn(i).subVector(0, n);
            }
            return new Table(cols);
        } finally {
            closeAll(cols);
        }
    }

    private static void syncGpu() {
        Cuda.deviceSynchronize();
    }

    private static void closeAll(AutoCloseable[] resources) {
        if (resources == null) return;
        for (AutoCloseable r : resources) {
            if (r != null) {
                try { r.close(); } catch (Exception ignore) {}
            }
        }
    }

    // ─── Output formatting ──────────────────────────────────────────────────

    private static void printTimes(String label, long[] timesNs, int nBatches) {
        double medianS = timesNs[timesNs.length / 2] / 1e9;
        double bestS = timesNs[0] / 1e9;
        double perBatch = medianS / nBatches;
        System.out.printf("  %-42s  median %8.4fs  best %8.4fs  (%.6fs/batch)%n",
            label, medianS, bestS, perBatch);
    }

    private static void printSummary(
            SplitResult write, SplitResult read,
            int nBatches, int totalRows, double dataMB, double ipcMB) {
        double convertCudf2Arrow = write.medianS(write.phase1Ns);
        double writeArrowIPC     = write.medianS(write.phase2Ns);
        double writeTotal        = write.medianS(write.totalNs);

        double readArrowIPC      = read.medianS(read.phase1Ns);
        double convertArrow2Cudf = read.medianS(read.phase2Ns);
        double readTotal         = read.medianS(read.totalNs);

        double d2h = convertCudf2Arrow;
        double h2d = convertArrow2Cudf;
        double ipcSer   = writeArrowIPC;
        double ipcDeser = readArrowIPC;

        double d2hBW = (d2h > 0) ? dataMB / 1024.0 / d2h : 0;
        double h2dBW = (h2d > 0) ? dataMB / 1024.0 / h2d : 0;

        System.out.println();
        System.out.println("-".repeat(70));
        System.out.println("OVERHEAD DECOMPOSITION  (median times)");
        System.out.println("-".repeat(70));
        System.out.printf("  convertCudfToArrowTable  (D2H):          %8.4fs  (%.1f GB/s)%n",
            d2h, d2hBW);
        System.out.printf("  writeArrowIPCArrowChunk  (IPC ser):      %8.4fs%n", ipcSer);
        System.out.printf("  readArrowIPCChunkToArrow (IPC deser):    %8.4fs%n", ipcDeser);
        System.out.printf("  convertArrowTableToCudf  (H2D):          %8.4fs  (%.1f GB/s)%n",
            h2d, h2dBW);

        System.out.println();
        System.out.println("-".repeat(70));
        System.out.println("SPARK SINGLE-THREAD MODEL  (additive, no pipelining)");
        System.out.println("-".repeat(70));
        System.out.printf("  T_p  (D2H + H2D):                        %8.4fs%n", d2h + h2d);
        System.out.printf("  T_s  (IPC ser + deser):                   %8.4fs%n", ipcSer + ipcDeser);
        System.out.printf("  Write total (D2H + IPC ser):              %8.4fs%n", writeTotal);
        System.out.printf("  Read total  (IPC deser + H2D):            %8.4fs%n", readTotal);
        System.out.println("  ──────────────────────────────────────────────");
        System.out.printf("  Full round-trip (write + read):           %8.4fs%n",
            writeTotal + readTotal);
        System.out.printf("  IPC stream size:                          %8.1f MB (%.1f%% of GPU data)%n",
            ipcMB, (dataMB > 0) ? ipcMB / dataMB * 100 : 0);
        System.out.printf("  Rows: %,d  |  Batches: %d  |  GPU data: %.1f MB%n",
            totalRows, nBatches, dataMB);
        System.out.println();
    }

    // ─── Entry point ────────────────────────────────────────────────────────

    public static void main(String[] args) {
        Map<String, String> argMap = parseArgs(args);

        String dataPath = argMap.get("data-path");
        if (dataPath == null) {
            throw new IllegalArgumentException(
                "Usage: --data-path PATH [--rows N] [--batch-size N] " +
                "[--warmup N] [--measured N] [--pool-fraction F]");
        }

        int maxRows    = Integer.parseInt(argMap.getOrDefault("rows", "-1"));
        int batchSize  = Integer.parseInt(argMap.getOrDefault("batch-size", "-1"));
        int warmup     = Integer.parseInt(argMap.getOrDefault("warmup",
                              String.valueOf(DEFAULT_WARMUP)));
        int measured   = Integer.parseInt(argMap.getOrDefault("measured",
                              String.valueOf(DEFAULT_MEASURED)));
        float rmmFrac  = Float.parseFloat(argMap.getOrDefault("pool-fraction",
                              String.valueOf(DEFAULT_RMM_FRACTION)));

        if (!Rmm.isInitialized()) {
            CudaMemInfo memInfo = Cuda.memGetInfo();
            long poolSize = (long) (memInfo.free * rmmFrac) & ~255L;
            Rmm.initialize(RmmAllocationMode.POOL, null, poolSize);
        }

        System.out.printf("Loading data from: %s%n", dataPath);
        Table fullTable = readParquetData(dataPath, maxRows);
        Table[] batches = splitIntoBatches(fullTable, batchSize);
        boolean ownsBatches = (batches.length > 1);

        int totalRows = 0;
        for (Table b : batches) totalRows += (int) b.getRowCount();
        int numCols = batches[0].getNumberOfColumns();
        double dataMB = getDataSizeMB(batches);
        int effBatchSize = batchSize <= 0 ? totalRows : batchSize;

        System.out.printf("  %,d rows x %d columns, %.1f MB, %d batches of %,d%n",
            totalRows, numCols, dataMB, batches.length, effBatchSize);
        System.out.printf("  warmup=%d, measured=%d%n%n", warmup, measured);

        // ── Run benchmarks ──────────────────────────────────────────────────
        System.out.println("=".repeat(70));
        System.out.println("TRANSFER OVERHEAD DECOMPOSITION  (JVM-side)");
        System.out.println("=".repeat(70));

        // 1. Arrow IPC write (convertCudfToArrowTable + writeArrowIPCArrowChunk)
        System.out.println("  [1/2] Arrow IPC write ...");
        SplitResult writeResult = benchArrowWrite(batches, numCols, warmup, measured);
        printTimes("  convertCudfToArrowTable  (D2H)", writeResult.phase1Ns, batches.length);
        printTimes("  writeArrowIPCArrowChunk  (IPC ser)", writeResult.phase2Ns, batches.length);
        printTimes("  write total", writeResult.totalNs, batches.length);
        syncGpu();

        // Pre-serialize IPC bytes for read benchmark
        byte[] ipcBytes = writeToIpcBytes(batches);
        double ipcMB = ipcBytes.length / (1024.0 * 1024.0);
        System.out.printf("  (IPC stream size: %.1f MB)%n%n", ipcMB);

        // 2. Arrow IPC read (readArrowIPCChunkToArrowTable + convertArrowTableToCudf)
        System.out.println("  [2/2] Arrow IPC read ...");
        SplitResult readResult = benchArrowRead(ipcBytes, effBatchSize, warmup, measured);
        printTimes("  readArrowIPCChunkToArrow (IPC deser)", readResult.phase1Ns, batches.length);
        printTimes("  convertArrowTableToCudf  (H2D)", readResult.phase2Ns, batches.length);
        printTimes("  read total", readResult.totalNs, batches.length);
        syncGpu();

        // ── Summary ─────────────────────────────────────────────────────────
        printSummary(writeResult, readResult, batches.length, totalRows, dataMB, ipcMB);

        // ── Cleanup ─────────────────────────────────────────────────────────
        if (ownsBatches) {
            for (Table b : batches) b.close();
        }
        fullTable.close();

        System.exit(0);
    }

    private static Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        int i = 0;
        while (i < args.length) {
            if (args[i].startsWith("--") && i + 1 < args.length && !args[i + 1].startsWith("--")) {
                map.put(args[i].substring(2), args[i + 1]);
                i += 2;
            } else {
                i++;
            }
        }
        return map;
    }
}
