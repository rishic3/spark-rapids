# Pandas UDF Data Flow in Spark-RAPIDS

## Spark UI Plan

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
tableWriter.write(table)             ← cuDF ArrowIPCTableWriter.write()
  │
  │  JNI → libcudf C++: D2H copy + Arrow conversion + IPC serialization (fused)
  │  Calls back into Java as IPC bytes are produced:
  │
  ▼
BufferToStreamWriter.handleBuffer()  ← receives host IPC bytes (128 KB chunks)
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
tableReader.getNextIfAvailable()      ← cuDF StreamedTableReader
  │
  │  StreamToBufferProvider.readInto(): reads socket bytes into host buffer (128 KB chunks)
  │  JNI → libcudf C++: parse IPC bytes + H2D copy (fused)
  │  Callback acquires GPU semaphore before H2D transfer
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
| cuDF two-phase write | `cudf java Table.java` → `ArrowIPCTableWriter.write()` |

## NVTX Ranges (for Nsight Systems profiling)

| Range | What it covers |
|---|---|
| `WRITE_PYTHON_BATCH` | `tableWriter.write(table)`: fused D2H + Arrow IPC serialize + `handleBuffer` socket write |
| `READ_PYTHON_BATCH` | socket read + Arrow IPC deserialize + H2D copy → GPU ColumnarBatch |

Enable with: `--conf spark.rapids.sql.nvtx.enabled=true` and profile with
`nsys profile -t cuda,nvtx`.

## Note on `spark.rapids.sql.python.gpu.enabled`

This config does **not** change the JVM-side Arrow serialization path. The
cuDF `writeArrowIPCChunked` / `readArrowIPCChunked` path is always used. The
config only controls:

- Whether the RAPIDS Python daemon/worker module is used (for GPU memory init)
- Whether RMM is initialized in the Python process (for cuDF usage inside UDFs)
- Whether the `PythonWorkerSemaphore` limits concurrent Python workers
