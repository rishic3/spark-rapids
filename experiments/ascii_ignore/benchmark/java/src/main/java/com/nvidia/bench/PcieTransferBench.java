/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.bench;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.CudaMemInfo;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.Rmm;
import ai.rapids.cudf.RmmAllocationMode;
import ai.rapids.cudf.Table;

/**
 * JVM-side microbenchmark measuring pure PCIe transfer cost:
 *   D2H: ColumnVector.copyToHost()   — GPU device memory → host memory
 *   H2D: HostColumnVector.copyToDevice() — host memory → GPU device memory
 *
 * No Arrow IPC serialization is involved. This gives the raw PCIe bandwidth
 * baseline for comparison with the full Arrow IPC round-trip measured by
 * TransferOverheadBench.
 *
 * Usage:
 *   mvn compile exec:java -Dexec.mainClass=com.nvidia.bench.PcieTransferBench \
 *     -Dexec.args="--data-path /path/to/parquet --rows 5000000 --batch-size 100000"
 */
public class PcieTransferBench {

    private static final int DEFAULT_WARMUP = 3;
    private static final int DEFAULT_MEASURED = 5;
    private static final float DEFAULT_RMM_FRACTION = 0.5f;

    // ─── Benchmark operations ───────────────────────────────────────────────

    /**
     * Benchmark copyToHost: GPU ColumnVector → HostColumnVector for every
     * column in every batch. Returns sorted per-iteration total times.
     */
    static long[] benchCopyToHost(Table[] batches, int warmup, int measured) {
        for (int w = 0; w < warmup; w++) {
            for (Table batch : batches) {
                for (int c = 0; c < batch.getNumberOfColumns(); c++) {
                    try (HostColumnVector hcv = batch.getColumn(c).copyToHost()) {
                        // discard
                    }
                }
            }
            syncGpu();
        }

        long[] timesNs = new long[measured];
        for (int m = 0; m < measured; m++) {
            syncGpu();
            long t0 = System.nanoTime();
            for (Table batch : batches) {
                for (int c = 0; c < batch.getNumberOfColumns(); c++) {
                    try (HostColumnVector hcv = batch.getColumn(c).copyToHost()) {
                        // discard
                    }
                }
            }
            syncGpu();
            timesNs[m] = System.nanoTime() - t0;
        }
        Arrays.sort(timesNs);
        return timesNs;
    }

    /**
     * Benchmark copyToDevice: HostColumnVector → GPU ColumnVector for every
     * column in every batch. Pre-copies data to host first, then times H2D only.
     */
    static long[] benchCopyToDevice(Table[] batches, int warmup, int measured) {
        int numCols = batches[0].getNumberOfColumns();
        HostColumnVector[][] hostData = new HostColumnVector[batches.length][numCols];
        try {
            for (int b = 0; b < batches.length; b++) {
                for (int c = 0; c < numCols; c++) {
                    hostData[b][c] = batches[b].getColumn(c).copyToHost();
                }
            }

            for (int w = 0; w < warmup; w++) {
                for (HostColumnVector[] cols : hostData) {
                    for (HostColumnVector hcv : cols) {
                        try (ColumnVector cv = hcv.copyToDevice()) {
                            // discard
                        }
                    }
                }
                syncGpu();
            }

            long[] timesNs = new long[measured];
            for (int m = 0; m < measured; m++) {
                syncGpu();
                long t0 = System.nanoTime();
                for (HostColumnVector[] cols : hostData) {
                    for (HostColumnVector hcv : cols) {
                        try (ColumnVector cv = hcv.copyToDevice()) {
                            // discard
                        }
                    }
                }
                syncGpu();
                timesNs[m] = System.nanoTime() - t0;
            }
            Arrays.sort(timesNs);
            return timesNs;
        } finally {
            for (HostColumnVector[] cols : hostData) {
                for (HostColumnVector hcv : cols) {
                    if (hcv != null) {
                        try { hcv.close(); } catch (Exception ignore) {}
                    }
                }
            }
        }
    }

    // ─── Helpers (shared with TransferOverheadBench) ────────────────────────

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
        System.out.printf("  %-36s  median %8.4fs  best %8.4fs  (%.6fs/batch)%n",
            label, medianS, bestS, perBatch);
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

        System.out.println("=".repeat(70));
        System.out.println("PURE PCIe TRANSFER  (JVM-side, no Arrow IPC)");
        System.out.println("=".repeat(70));

        // 1. D2H: copyToHost
        System.out.println("  [1/2] copyToHost (D2H) ...");
        long[] d2hTimes = benchCopyToHost(batches, warmup, measured);
        printTimes("copyToHost  (D2H)", d2hTimes, batches.length);
        syncGpu();

        // 2. H2D: copyToDevice
        System.out.println("  [2/2] copyToDevice (H2D) ...");
        long[] h2dTimes = benchCopyToDevice(batches, warmup, measured);
        printTimes("copyToDevice (H2D)", h2dTimes, batches.length);
        syncGpu();

        // ── Summary ─────────────────────────────────────────────────────────
        double d2hMedian = d2hTimes[d2hTimes.length / 2] / 1e9;
        double h2dMedian = h2dTimes[h2dTimes.length / 2] / 1e9;
        double d2hBW = (d2hMedian > 0) ? dataMB / 1024.0 / d2hMedian : 0;
        double h2dBW = (h2dMedian > 0) ? dataMB / 1024.0 / h2dMedian : 0;

        System.out.println();
        System.out.println("-".repeat(70));
        System.out.println("SUMMARY  (median times)");
        System.out.println("-".repeat(70));
        System.out.printf("  copyToHost   (D2H):                      %8.4fs  (%.1f GB/s)%n",
            d2hMedian, d2hBW);
        System.out.printf("  copyToDevice (H2D):                      %8.4fs  (%.1f GB/s)%n",
            h2dMedian, h2dBW);
        System.out.println("  ──────────────────────────────────────────────");
        System.out.printf("  Round-trip (D2H + H2D):                  %8.4fs%n",
            d2hMedian + h2dMedian);
        System.out.printf("  Rows: %,d  |  Batches: %d  |  GPU data: %.1f MB%n",
            totalRows, batches.length, dataMB);
        System.out.println();

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
