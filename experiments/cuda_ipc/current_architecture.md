# Pandas UDF Data Flow in Spark-RAPIDS

## Spark UI Plan

For a read parquet -> Pandas UDF -> write parquet job:
```
GpuScan → GpuArrowEvalPython → GpuCoalesceBatches → GpuProject → GpuWriteFiles
```

The entire device→host→Python→host→device round-trip is hidden inside `GpuArrowEvalPython`.

## Detailed Data Flow

### JVM → Python (Writer Thread)

```
GPU ColumnarBatch
  │
  │  GpuColumnVector.from(batch) → cuDF Table (still on GPU)
  │
  ▼
tableWriter.write(table)                  ← cuDF ArrowIPCTableWriter.write()
  │
  │  Step 1: convertCudfToArrowTable()    ← JNI into libcudf
  │    Calls to_arrow_host kernel: D2H copy of GPU column data
  │    Constructs and returns host-side Arrow buffers (Arrow table on host)
  │
  │  Step 2: writeArrowIPCArrowChunk()    ← JNI into C++ Arrow IPC writer
  │    Takes host Arrow table buffers, serializes to IPC format
  │    Calls back into Java with IPC bytes:
  │
  ▼
BufferToStreamWriter.handleBuffer()       ← receives host IPC bytes (128 KB chunks)
  │  copies from HostMemoryBuffer → byte[] → DataOutputStream
  │
  ▼
Socket → Python worker
```

### Python Worker

PySpark's standard `worker.py` reads the Arrow IPC stream, deserializes into
Pandas DataFrame/Series, executes the UDF, serializes the result back to Arrow
IPC, and writes it to the return socket.

### Python → JVM (Reader Thread)

```
Socket → Python worker response
  │
  ▼
tableReader.getNextIfAvailable()           ← cuDF ArrowIPCStreamedTableReader
  │
  │  Step 1: readArrowIPCChunkToArrowTable()  ← JNI into Arrow C++ stream reader
  │    StreamToBufferProvider.readInto(): reads socket bytes into host buffer (128 KB chunks)
  │    Arrow C++ IPC reader deserializes into host Arrow record batches
  │    Repeats until accumulated rows reach target or EOF
  │
  │  Callback acquires GPU semaphore before H2D transfer
  │
  │  Step 2: convertArrowTableToCudf()        ← JNI into cudf::from_arrow
  │    Constructs cuDF columns (H2D copy) from host Arrow buffers
  │
  ▼
toBatch(table) → GPU ColumnarBatch
```

### Post-Python

`CombiningIterator` column-concatenates the original input batch (cached by
`BatchProducer`) with the UDF result batch. `GpuCoalesceBatches` merges the
small output batches back into larger ones. `GpuProject` prunes to only the
columns needed downstream.

## Key Code Locations

| Component | File |
|---|---|
| Exec node (orchestrator) | `sql-plugin/.../python/GpuArrowEvalPythonExec.scala` |
| Arrow write (GPU→socket) | `sql-plugin/.../python/GpuArrowWriter.scala` |
| Arrow read (socket→GPU) | `sql-plugin/.../python/GpuArrowReader.scala` |
| Runner (spawns writer/reader threads) | `sql-plugin/.../python/shims/GpuArrowPythonRunner.scala` |
| Reader iterator (protocol handling) | `sql-plugin/.../python/shims/GpuArrowPythonOutput.scala` |
| Combining + batching utils | `sql-plugin/.../python/BatchGroupUtils.scala` |
| cuDF two-phase write | `cudf java Table.java` → `ArrowIPCTableWriter.write()` → `convertCudfToArrowTable()` + `writeArrowIPCArrowChunk()` |

## NVTX Ranges (for Nsight Systems profiling)

| Range | What it covers |
|---|---|
| `WRITE_PYTHON_BATCH` | `tableWriter.write(table)`: `convertCudfToArrowTable` (D2H) + `writeArrowIPCArrowChunk` (IPC serialize) + `handleBuffer` (socket write) |
| `READ_PYTHON_BATCH` | `tableReader.getNextIfAvailable()`: `readArrowIPCChunkToArrowTable` (socket read + IPC deserialize) + `convertArrowTableToCudf` (H2D) → GPU ColumnarBatch |
