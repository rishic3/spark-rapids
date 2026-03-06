# Engineering Report: Efficient JVM to Python Transfer in Spark Using CUDA IPC

**Date:** March 2026
**Scope:** spark-rapids, cudf, spark-rapids-jni, Apache Arrow

---

## 1. Executive Summary

### The Problem

When Spark RAPIDS executes Python UDFs (Pandas UDFs, `mapInPandas`, etc.), data must travel between the JVM executor and a separate Python worker process. The current path is:

```
GPU (cuDF Table) → D2H copy → Arrow IPC serialize → Socket → Python deserialize → Pandas DataFrame
                                                                                     ↓ (UDF executes)
GPU (cuDF Table) ← H2D copy ← Arrow IPC deserialize ← Socket ← Python serialize ← Pandas DataFrame
```

This involves **two full GPU↔Host memory copies** and **two Arrow IPC serialization/deserialization passes** per batch, which is expensive — especially for lightweight UDFs where serialization dominates compute time.

### Two Proposed Approaches

| Approach | Mechanism | Key Benefit | Key Risk |
|----------|-----------|-------------|----------|
| **CUDA IPC** | Pass GPU memory handles to Python process; data stays on GPU | Eliminates all D2H/H2D copies and serialization | RMM allocator compatibility; memory lifetime management |
| **Embedded Python** | Embed CPython within JVM via pybind11+JNI; execute UDFs in-process | Eliminates IPC, sockets, and process boundary | GIL contention; Python crash takes down JVM; requires shared libpython |

### Recommendation

**CUDA IPC is the more viable and impactful approach**, particularly using the modern `cudf::pack`/`cudf::unpack` mechanism. The embedded Python approach, while architecturally elegant, introduces unacceptable stability risks for production Spark deployments and offers only modest performance improvements (3–34%) that are bounded by Arrow serialization overhead rather than the D2H/H2D copy cost that CUDA IPC eliminates.

---

## 2. Current Implementation & Architecture

### 2.1 Data Flow (Current — As of Main Branches, March 2026)

The Python UDF execution path in spark-rapids is orchestrated by `GpuArrowEvalPythonExec`:

```
GpuScan → GpuArrowEvalPython → GpuCoalesceBatches → GpuProject → GpuWriteFiles
```

**Key Code Locations (spark-rapids):**

| Component | File |
|---|---|
| Exec node (orchestrator) | `sql-plugin/src/main/scala/org/apache/spark/sql/rapids/execution/python/GpuArrowEvalPythonExec.scala` |
| Arrow write (GPU→socket) | `sql-plugin/src/main/scala/org/apache/spark/sql/rapids/execution/python/GpuArrowWriter.scala` |
| Arrow read (socket→GPU) | `sql-plugin/src/main/scala/org/apache/spark/sql/rapids/execution/python/GpuArrowReader.scala` |
| Runner (Spark 3.2.x shim) | `sql-plugin/src/main/spark321/scala/.../python/shims/GpuArrowPythonRunner.scala` |
| Runner (Spark 3.4.1db+ shim) | `sql-plugin/src/main/spark341db/scala/.../python/shims/GpuArrowPythonRunner.scala` |
| Batching utilities | `sql-plugin/src/main/scala/org/apache/spark/sql/rapids/execution/python/BatchGroupUtils.scala` |
| Map in Pandas | `sql-plugin/.../python/GpuMapInPandasExec.scala` |
| Flat Map Groups | `sql-plugin/.../python/GpuFlatMapGroupsInPandasExec.scala` |
| Aggregate in Pandas | `sql-plugin/.../python/GpuAggregateInPandasExec.scala` |
| Window in Pandas | `sql-plugin/.../python/GpuWindowInPandasExecBase.scala` |
| Python Worker Semaphore | `sql-plugin/.../python/PythonWorkerSemaphore.scala` |
| UDF Arguments (new 2025) | `sql-plugin/.../python/GpuPythonArguments.scala` |

**Note on Shim Architecture:** The runner has different implementations per Spark version:
- **Spark 3.2–3.4.0:** `WriterThread.writeIteratorToStream()` — loops all batches in one call
- **Spark 3.4.1db+:** `Writer.writeNextInputToStream()` — writes one batch at a time, returns boolean (more efficient streaming model)

**JVM → Python (Writer Thread):**
1. `GpuColumnVector.from(batch)` → cuDF Table (on GPU)
2. `tableWriter.write(table)` calls:
   - `convertCudfToArrowTable()` — JNI into libcudf; D2H copy; constructs host Arrow buffers
   - `writeArrowIPCArrowChunk()` — JNI into Arrow C++ IPC writer; serializes to IPC format
3. `BufferToStreamWriter.handleBuffer()` — copies host IPC bytes to socket via 128KB chunks

**Python → JVM (Reader Thread):**
1. `tableReader.getNextIfAvailable()` calls:
   - `readArrowIPCChunkToArrowTable()` — reads socket bytes into host buffer; Arrow IPC deserializes
   - `convertArrowTableToCudf()` — JNI into `cudf::from_arrow`; H2D copy
2. Result becomes a GPU `ColumnarBatch`

### 2.2 cuDF Java Arrow IPC Code (Still Present)

The Arrow IPC writer/reader code **still exists** in cuDF's Java bindings:
- `Table.java:1890` — `ArrowIPCTableWriter` class
- `Table.java:2010` — `ArrowIPCStreamedTableReader` class
- `TableJni.cpp:2806` — `convertCudfToArrowTable` JNI implementation
- `TableJni.cpp:2941` — `convertArrowTableToCudf` JNI implementation

These rely on Arrow C++ for IPC serialization/deserialization and are the **current hot path** for Python UDF data transfer.

**cuDF C++ Arrow Interop (what remains after CUDA IPC removal):**
- `to_arrow_device()` / `from_arrow_device()` — ArrowDeviceArray interchange (C Data Interface)
- `to_arrow_host()` / `from_arrow_host()` — Host-side Arrow conversion
- The Arrow format serialization/deserialization is maintained; only the CUDA-specific Arrow IPC transport was removed

### 2.3 RMM Configuration

The default RMM pool allocator in spark-rapids is now **`ASYNC`** (CUDA stream-ordered memory allocator), configured via:
```
spark.rapids.memory.gpu.pool = "ASYNC"  (default)
```
Other options: `DEFAULT` (RMM pool), `ARENA`, `NONE`.

**This is a critical consideration for CUDA IPC** — see Section 4.2.

### 2.4 Kudo Serializer (New in 2025)

spark-rapids-jni now includes a comprehensive custom serialization framework used for shuffle:

**Host Serialization** (`KudoSerializer.java`):
- Binary format: 4-byte header + validity buffers + offset buffers + data buffers
- 4-byte alignment, Arrow-compatible (64-byte padding)
- API: `writeToStreamWithMetrics()`, `mergeOnHost()`, `mergeToTable()`

**GPU Serialization** (`KudoGpuSerializer.java` — NEW in 2025):
- `splitAndSerializeToDevice(Table, splits)` — splits and serializes entirely on GPU
- `assembleFromDeviceRaw(Schema, partitions, offsets)` — reassembles from device buffers
- Returns contiguous device buffers with metadata offsets

**Host Table Transfers** (`HostTable.java`):
- `fromTableAsync()` — async device-to-host copy
- `toTableAsync()` — async host-to-device copy
- `fromViewWithContiguousAllocation()` — shared buffer construction

This demonstrates that the spark-rapids team has already invested in **custom binary serialization** that bypasses Arrow IPC — a pattern directly applicable to CUDA IPC transfer. The `KudoGpuSerializer.splitAndSerializeToDevice` method is particularly relevant as it packs table data into contiguous device buffers, exactly what CUDA IPC needs.

### 2.5 contiguousSplit Usage in the Python UDF Path

`contiguousSplit` is used in three critical locations in the Python UDF pipeline:

1. **`RebatchingRoundoffIterator`** (`GpuArrowEvalPythonExec.scala:185`): Normalizes batches to target size multiples; splits oversized combined batches to extract excess rows
2. **`BatchGroupedIterator`** (`BatchGroupUtils.scala:302`): Splits grouped batches into one group per batch via `table.groupBy(...).contiguousSplitGroups()`
3. **`CombiningIterator`** (`BatchGroupUtils.scala:465`): Splits the last batch to align with Python output row count

All intermediate batches use `SpillableColumnarBatch` for RMM integration and memory spillability.

### 2.6 Current CUDA IPC Status: Not Implemented

A thorough search of all three repositories confirms: **no CUDA IPC code exists on any main branch today**. The PoC branches (wbo4958 forks) were never merged. Searches for `cuda.ipc`, `ipc`, `zerocopy`, `zero.copy`, `cudaIpc`, `IpcMemHandle` found only unrelated references (Parquet, ORC, Shuffle, UCX).

---

## 3. Historical Context & Blockers

### 3.1 Original PoC (July 2022)

**Author:** Bobby Wang (wbo4958), NVIDIA

**Design:** Pass CUDA IPC metadata (JSON) as Arrow IPC payload instead of actual data:
- JVM side: Extract CUDA IPC handle from GPU buffer via `cudaIpcGetMemHandle`, encode as base64 JSON with column metadata (type, offset, null count, etc.)
- Python side: Deserialize JSON from Arrow IPC, call `cudaIpcOpenMemHandle` to get GPU pointer, reconstruct cuDF columns using offsets

**PoC Branches:**
- `wbo4958/spark-rapids-jni:cuda-ipc` — Added `CudaIPC.java` / `CudaIPCJni.cpp` with `getCudaIpcMemHandle()` JNI wrapper
- `wbo4958/spark-rapids:cuda-ipc` — Modified `GpuMapInPandasExec` to send IPC metadata instead of data

### 3.2 Original Blockers

| Blocker | Description | Status (2026) |
|---------|-------------|---------------|
| **RebatchingRoundOffIterator / GpuCoalesceBatches** | These operators call `contiguousSplit()` which repacks data into a single contiguous buffer. After repacking, column data pointers are offsets within a larger allocation, and `cudaIpcGetMemHandle` on an interior pointer returns the handle for the **entire** allocation. The PoC couldn't determine the base allocation address from a column_view's data pointer. | **Resolved** — The `contiguousSplit()` / `pack()` approach actually *helps* CUDA IPC by ensuring all data is in one contiguous buffer with known metadata. |
| **RMM Pool Allocator Compatibility** | CUDA IPC documentation stated: "IPC API is not supported for `cudaMallocManaged` allocations." The question was whether RMM's pool/arena allocators (which sub-allocate from larger `cudaMalloc` allocations) would produce valid IPC handles. | **Partially Resolved** — RMM pool and arena allocators sub-allocate from `cudaMalloc` blocks, so IPC handles work but point to the entire underlying allocation. The `ASYNC` allocator (now the default) uses `cudaMallocAsync` which has its own IPC mechanism (`cudaMemPoolExportToShareableHandle`) that is different from the standard `cudaIpcGetMemHandle`. |
| **Memory Lifetime** | GPU buffers must remain alive while the Python process holds the IPC handle. The original PoC noted that iterating all ColumnarBatches and holding them alive could "blow up GPU memory rapidly." | **Still a concern** — Requires careful lifetime management with reference counting or explicit synchronization. |
| **Non-working Cases** | Operations before `mapInPandas` (filter, sort, repartition) caused failures because `GpuCoalesceBatches` repacked data. | **Resolved** — Using `pack()` as the serialization step deliberately creates a contiguous buffer, making this a feature rather than a bug. |

### 3.3 cuDF PR #11564 (Closed May 2025)

Jiaming Yuan's PR attempted to implement IPC directly in cuDF C++/Python. Key outcomes:
- **42 commits** of work, supporting RMM pool allocator, primitive types, and list types
- **Closed as stale** with the recommendation: *"a fresh start with the pack/unpack approach would be preferable"*, referencing patterns from **multi-GPU Polars** work
- The `pack/unpack` approach was specifically called out as simpler and more maintainable

### 3.4 cuDF Issue #10994 (Arrow CUDA IPC Removal)

The Arrow CUDA IPC code was **removed from libcudf C++** in July 2022 (PR #10995). Reasons:
- Legacy code from "pygdf" days
- Created dependency on Arrow CUDA (`libcuda.so`)
- Modern alternatives (`__cuda_array_interface__`, DLPack, pack/unpack) exist

**Important:** The Arrow IPC code in the **cuDF Java bindings** (`Table.java`) was NOT removed — it's still used by spark-rapids for the current Python UDF path.

### 3.5 cuDF Issue #11514 (Interchange Protocol)

Feature request for `to_ipc()` / `from_ipc()` methods in cuDF. Discussion converged on:
- Using existing `pack/unpack` mechanism rather than creating a new IPC protocol
- Arrow IPC considered "sufficient for now" for the immediate term
- **Still open** — no dedicated IPC interchange API was added

---

## 4. Modern Viability & Nuances

### 4.1 The Pack/Unpack Paradigm (Recommended Approach)

cuDF now has a mature, well-tested `pack`/`unpack` API that is the **ideal foundation** for CUDA IPC:

**C++ API** (`cudf/contiguous_split.hpp`):
```cpp
// Pack: Table → contiguous metadata (host) + data (device)
packed_columns pack(table_view const& input, ...);

// Unpack: metadata + device pointer → table_view (zero-copy)
table_view unpack(uint8_t const* metadata, uint8_t const* gpu_data);

// Metadata inspection without device data
class packed_metadata_view { ... };  // Query schema, types, sizes, null counts
```

**`packed_columns` structure:**
```cpp
struct packed_columns {
    std::unique_ptr<std::vector<uint8_t>> metadata;  // Host: column schema, offsets, types
    std::unique_ptr<rmm::device_buffer> gpu_data;     // Device: all column data contiguously
};
```

**Java API** (`Table.java`, `ContiguousTable.java`, `ChunkedPack.java`):
```java
// Pack via contiguousSplit (zero splits = single pack)
ContiguousTable[] table.contiguousSplit();  // Returns ContiguousTable with metadata + buffer

// ContiguousTable provides:
ByteBuffer getMetadataDirectBuffer();  // Host metadata
DeviceMemoryBuffer getBuffer();        // Single contiguous GPU buffer

// Reconstruct from metadata + buffer
Table.fromPackedTable(ByteBuffer metadata, DeviceMemoryBuffer buffer);

// ChunkedPack for memory-constrained scenarios
ChunkedPack table.makeChunkedPack(long bounceBufferSize);
```

**Python API** (`pylibcudf.contiguous_split`):
```python
packed = pylibcudf.contiguous_split.pack(table)
metadata, gpu_data = packed.release()

# Reconstruct from metadata + GPU memory pointer
table = pylibcudf.contiguous_split.unpack_from_memoryviews(metadata, gpu_data)
```

**Why this is ideal for CUDA IPC:**
1. `pack()` produces exactly what CUDA IPC needs: a **single contiguous GPU buffer** + **small host metadata**
2. Only the metadata (small, ~bytes per column) needs to cross the socket
3. The GPU buffer can be shared via `cudaIpcGetMemHandle` on its base address
4. `unpack()` reconstructs the table from metadata + GPU pointer — zero-copy on the receiving end

### 4.2 RMM Allocator Compatibility (Critical Constraint)

| Allocator | `cudaIpcGetMemHandle` | Notes |
|-----------|----------------------|-------|
| `cudaMalloc` (NONE) | **Works** | Direct CUDA allocation; IPC handles are valid |
| RMM Pool (DEFAULT) | **Works with caveats** | Sub-allocates from `cudaMalloc` blocks; IPC handle covers the entire underlying allocation, not just the sub-allocation. Receiver must use the correct offset. |
| RMM Arena (ARENA) | **Works with caveats** | Same sub-allocation concern as Pool |
| `cudaMallocAsync` (ASYNC) | **Does NOT work with standard IPC API** | Requires `cudaMemPoolExportToShareableHandle` / `cudaMemPoolImportFromShareableHandle` instead. Different API surface. |
| `cudaMallocManaged` | **Not supported** | CUDA IPC explicitly does not support managed memory |

**The ASYNC allocator is now the default in spark-rapids.** This means a naive `cudaIpcGetMemHandle` approach will fail out of the box.

**Solutions:**
1. **Use `pack()` with a non-async allocation:** The `pack()` function accepts an `rmm::device_async_resource_ref mr` parameter. By passing a `cudaMalloc`-backed resource (not the default async pool), the packed buffer will be allocated with standard CUDA malloc and will be compatible with `cudaIpcGetMemHandle`.
2. **Use the CUDA async IPC API:** CUDA 11.2+ supports IPC for stream-ordered allocations via `cudaMemPoolExportToShareableHandle`, but this is more complex and requires configuring the memory pool for IPC export.
3. **Allocate a staging buffer with `cudaMalloc`:** Pack into a separately-allocated staging buffer using the standard CUDA allocator, export its IPC handle, and have the Python side open it.

**Recommendation:** Option 1 or 3 — allocate the packed output buffer using a plain `cudaMalloc` resource specifically for IPC-exportable buffers. This is simple, doesn't change the default allocator for the rest of the system, and is fully compatible.

### 4.3 cuDF's New CudfTable Binary Format

cuDF has introduced an experimental `cudftable` binary format (`cudf/io/experimental/cudftable.hpp`, 2025) that wraps `pack`/`unpack` with file I/O:

```cpp
void write_cudftable(cudftable_writer_options const& options, ...);
packed_table read_cudftable(cudftable_reader_options const& options, ...);
```

This format serializes packed metadata + data with a simple header. While designed for file I/O, it validates that `pack/unpack` is being promoted as the **standard serialization mechanism** for cuDF tables.

### 4.4 Multi-GPU Polars Pattern (rapidsmpf)

The `cudf-polars` project's `rapidsmpf` module demonstrates the modern pattern for cross-process/cross-GPU data transfer:

```python
# Producer side
packed_data = PackedData.from_cudf_packed_columns(
    pylibcudf.contiguous_split.pack(plc_table, stream), ...)

# Transfer via shuffler/allgather
allgather.insert(0, packed_data)

# Consumer side
plc_result = unpack_and_concat(results, stream, ctx.br())
```

This is the exact pattern recommended by the cuDF maintainers when they closed PR #11564. The pack/unpack + transport mechanism is **production-proven** for multi-GPU shuffling in Polars.

### 4.5 Are the Old PoCs Still Viable?

**No, the old PoC branches are not directly viable.** They target:
- spark-rapids 22.08 (now on 25.x+)
- cudf 22.06 (now on 26.x+)
- PySpark 3.1.1 (now on 3.5+)
- The raw `cudaIpcGetMemHandle` approach without `pack/unpack`
- JSON-encoded metadata format (fragile, not schema-aware)

However, the **core concept is more viable than ever** because:
- `pack/unpack` solves the contiguous buffer + metadata problem cleanly
- The Java API already supports `ContiguousTable` with `getMetadataDirectBuffer()` + `getBuffer()`
- Python's `unpack_from_memoryviews(metadata, gpu_data)` can reconstruct from arbitrary GPU pointers
- The `packed_metadata_view` class enables schema introspection without device data

---

## 5. Implementation Blueprint

### 5.1 Architecture Overview

```
JVM → Python (CUDA IPC):
┌─────────────────────┐              ┌─────────────────────┐
│     JVM Process     │              │   Python Process    │
│                     │              │                     │
│  GPU ColumnarBatch  │              │                     │
│        │            │              │                     │
│        ▼            │              │                     │
│  table.contiguousSplit()           │                     │
│  or pack() with     │              │                     │
│  cudaMalloc resource│              │                     │
│        │            │              │                     │
│        ├─ metadata ─┼──(socket)──▶│── metadata          │
│        │  (bytes)   │              │      │              │
│        │            │              │      ▼              │
│        ├─ IPC handle┼──(socket)──▶│── cudaIpcOpenMem()  │
│        │  (64 bytes)│              │      │              │
│        │            │              │      ▼              │
│        │            │              │  unpack(metadata,   │
│        │            │              │         gpu_ptr)    │
│        │            │              │      │              │
│        │            │              │      ▼              │
│        │            │              │  cuDF Table         │
│        │            │              │  → Pandas DataFrame │
│        │            │              │  → Execute UDF      │
│        │            │              │      │              │
│        │            │              │  (reverse path for  │
│        │            │              │   return data)      │
│        ▼            │              │                     │
│  Keep buffer alive  │              │  cudaIpcCloseMem()  │
│  until Python done  │              │                     │
└─────────────────────┘              └─────────────────────┘

Data transferred over socket: ~metadata bytes + 64-byte IPC handle
Data NOT transferred: entire GPU buffer (stays on GPU!)
```

### 5.2 Implementation Steps

#### Phase 1: JNI Layer (spark-rapids-jni)

**New file:** `src/main/java/com/nvidia/spark/rapids/jni/CudaIpc.java`

```java
public class CudaIpc {
    /**
     * Get a CUDA IPC handle for a device memory buffer.
     * The buffer MUST have been allocated with cudaMalloc (not cudaMallocAsync).
     * @param devicePtr base address of the device memory buffer
     * @return 64-byte CUDA IPC memory handle
     */
    public static native byte[] getIpcMemHandle(long devicePtr);

    /**
     * Open a CUDA IPC handle received from another process.
     * @param handle 64-byte CUDA IPC memory handle
     * @return device pointer to the shared memory
     */
    public static native long openIpcMemHandle(byte[] handle);

    /**
     * Close a previously opened CUDA IPC memory handle.
     * @param devicePtr pointer returned by openIpcMemHandle
     */
    public static native void closeIpcMemHandle(long devicePtr);
}
```

**New file:** `src/main/cpp/src/CudaIpcJni.cpp`

```cpp
extern "C" {
JNIEXPORT jbyteArray JNICALL Java_com_nvidia_spark_rapids_jni_CudaIpc_getIpcMemHandle(
    JNIEnv* env, jclass, jlong device_ptr) {
    cudaIpcMemHandle_t handle;
    CUDA_CHECK(cudaIpcGetMemHandle(&handle, reinterpret_cast<void*>(device_ptr)));
    jbyteArray result = env->NewByteArray(sizeof(handle));
    env->SetByteArrayRegion(result, 0, sizeof(handle), reinterpret_cast<jbyte*>(&handle));
    return result;
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_CudaIpc_openIpcMemHandle(
    JNIEnv* env, jclass, jbyteArray handle_bytes) {
    cudaIpcMemHandle_t handle;
    env->GetByteArrayRegion(handle_bytes, 0, sizeof(handle), reinterpret_cast<jbyte*>(&handle));
    void* ptr;
    CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
    return reinterpret_cast<jlong>(ptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_CudaIpc_closeIpcMemHandle(
    JNIEnv* env, jclass, jlong device_ptr) {
    CUDA_CHECK(cudaIpcCloseMemHandle(reinterpret_cast<void*>(device_ptr)));
}
}
```

#### Phase 2: Spark-RAPIDS Plugin (Write Path)

**Modified:** `GpuArrowWriter.scala` (or new `GpuIpcWriter.scala`)

Replace the Arrow IPC write path with:

1. **Pack the table:** Call `table.contiguousSplit()` (no splits) to get a `ContiguousTable` with a single contiguous GPU buffer. Use a `cudaMalloc`-backed RMM resource for the allocation.
2. **Get IPC handle:** Call `CudaIpc.getIpcMemHandle(contiguousTable.getBuffer().getAddress())`
3. **Send over socket:**
   - Send metadata length (4 bytes) + metadata bytes (from `getMetadataDirectBuffer()`)
   - Send buffer size (8 bytes)
   - Send IPC handle (64 bytes)
4. **Hold reference** to the `ContiguousTable` until Python signals completion

**Key change in `GpuArrowPythonRunner`:** The writer thread must not close the `ContiguousTable` until the Python process has finished reading from the shared GPU memory. This requires a synchronization mechanism (e.g., the Python process sends a "done" signal back over the socket before the JVM releases the buffer).

#### Phase 3: Python Worker Modification

**Modified:** PySpark's `worker.py` or a new RAPIDS-specific worker

On the Python side, when IPC mode is enabled:

```python
import cupy
import pylibcudf as plc

def read_ipc_batch(socket):
    # Read metadata
    meta_len = struct.unpack('!I', socket.read(4))[0]
    metadata = socket.read(meta_len)

    # Read buffer size and IPC handle
    buf_size = struct.unpack('!Q', socket.read(8))[0]
    ipc_handle = socket.read(64)

    # Open IPC memory handle
    gpu_ptr = cupy.cuda.runtime.ipcOpenMemHandle(ipc_handle)

    # Create a device memory view from the pointer
    gpu_data = cupy.cuda.MemoryPointer(
        cupy.cuda.UnownedMemory(gpu_ptr, buf_size, owner=None), 0)

    # Unpack using pylibcudf
    table = plc.contiguous_split.unpack_from_memoryviews(
        memoryview(metadata), gpu_data_span)

    # Convert to pandas
    df = cudf.DataFrame.from_pylibcudf(table)
    pandas_df = df.to_pandas()

    return pandas_df

def close_ipc_handle(gpu_ptr):
    cupy.cuda.runtime.ipcCloseMemHandle(gpu_ptr)
```

#### Phase 4: Configuration & Fallback

**New configuration options in `RapidsConf.scala`:**

```scala
val PYTHON_CUDA_IPC_ENABLED = conf("spark.rapids.sql.python.cudaIpc.enabled")
    .doc("Enable CUDA IPC for zero-copy GPU data transfer to Python workers. " +
         "Requires cuDF and cupy in the Python environment. " +
         "Falls back to Arrow IPC if unavailable.")
    .booleanConf
    .createWithDefault(false)
```

**Fallback:** If the Python worker doesn't have cuDF/cupy, or if CUDA IPC handle creation fails, fall back to the existing Arrow IPC path transparently.

### 5.3 Specific Code Areas Requiring Modification

| Repository | File/Area | Change |
|---|---|---|
| **spark-rapids-jni** | New `src/main/java/com/nvidia/spark/rapids/jni/CudaIpc.java` + `src/main/cpp/src/CudaIpcJni.cpp` | JNI wrappers for CUDA IPC API |
| **spark-rapids** | `sql-plugin/.../python/GpuArrowEvalPythonExec.scala` | Add IPC mode selection logic in `RebatchingRoundoffIterator` |
| **spark-rapids** | `sql-plugin/.../python/GpuArrowWriter.scala` | New IPC write path: `start()`, `write()`, `reset()` |
| **spark-rapids** | `sql-plugin/.../python/GpuArrowReader.scala` | New IPC read path: `start()`, `readNext()` |
| **spark-rapids** | `sql-plugin/.../python/shims/GpuArrowPythonRunner.scala` (both spark321 and spark341db shims) | Buffer lifetime management; synchronization |
| **spark-rapids** | `sql-plugin/.../RapidsConf.scala` | New configuration options |
| **spark-rapids** | `sql-plugin/.../python/BatchGroupUtils.scala` | `CombiningIterator` uses `contiguousSplit`; ensure compatibility |
| **spark-rapids** | `sql-plugin/.../python/GpuMapInPandasExec.scala` | Entry point for `mapInPandas` UDFs |
| **cudf (Java)** | `java/src/main/java/ai/rapids/cudf/ContiguousTable.java` | Has `getMetadataDirectBuffer()` + `getBuffer()` — may add IPC handle accessor |
| **cudf (Java)** | `java/src/main/java/ai/rapids/cudf/ChunkedPack.java` | Memory-constrained packing: `hasNext()`, `next()`, `buildMetadata()` |
| **cudf (Java)** | `java/src/main/native/src/PackedColumnMetadataJni.cpp` + `ChunkedPackJni.cpp` | Native JNI for packed metadata |
| **cudf (Python)** | `python/pylibcudf/pylibcudf/contiguous_split.pyx` | Already has `unpack_from_memoryviews()` — no changes needed |
| **PySpark** | `worker.py` (or RAPIDS wrapper) | IPC-aware data reader using `pylibcudf` |

### 5.4 Potential Pitfalls

1. **ASYNC Allocator (Default):** The packed buffer MUST be allocated with `cudaMalloc`, not `cudaMallocAsync`. Use `rmm::mr::cuda_memory_resource` explicitly when calling `pack()` or `contiguousSplit()`.

2. **Memory Lifetime:** The JVM must hold the `ContiguousTable` (and its GPU buffer) alive until Python has finished processing. If the JVM releases it while Python still references it via IPC handle, corruption occurs. Implement explicit "done" signaling.

3. **Multi-GPU:** CUDA IPC works between processes on the **same GPU**. In multi-GPU setups, ensure the Python worker is bound to the same GPU as the JVM executor. This is typically managed by Spark's GPU resource discovery.

4. **Nested Types:** The `pack`/`unpack` mechanism handles nested types (lists, structs) correctly via the metadata format. The old PoC's JSON format did not support these well.

5. **Null Masks:** `pack()` correctly handles null bitmasks by including them in the contiguous buffer with appropriate offsets in metadata. No special handling needed.

6. **Return Path (Python → JVM):** The reverse direction is equally important. The Python process must `pack()` its result, get an IPC handle, and send it back. The JVM must then `openIpcMemHandle`, `unpack`, and copy the data into its own RMM-managed buffer before closing the IPC handle.

7. **Process Crash Safety:** If the Python process crashes while holding an IPC handle, CUDA should clean up the mapping. However, the JVM-side buffer reference must also be cleaned up. Use try-finally blocks and timeout mechanisms.

8. **Batching Strategy:** The current path sends multiple small Arrow IPC batches. With CUDA IPC, it's more efficient to send fewer, larger batches since the IPC handle overhead is fixed (64 bytes). Consider coalescing batches before packing.

9. **Python Environment Dependency:** CUDA IPC mode requires `cudf` and `cupy` (or equivalent) in the Python environment. This is a stronger requirement than the current path which only needs `pyarrow` and `pandas`.

### 5.5 Performance Expectations

| Scenario | Current (Arrow IPC) | With CUDA IPC | Expected Speedup |
|----------|-------------------|---------------|-----------------|
| Small UDF, large data | D2H + serialize + socket + deserialize + H2D | 64-byte handle + metadata bytes over socket | **10-100x on transfer** |
| Large UDF, small data | Transfer is small portion of total | Marginal improvement | **1-5%** |
| Lightweight UDF (parse_json-like) | Transfer dominates | Near-zero transfer | **50-90% on transfer** |

The key insight is that CUDA IPC eliminates the transfer cost almost entirely, making it proportional to the **metadata size** rather than the **data size**. For a 100MB batch, this means sending ~100 bytes instead of ~100MB.

---

## 6. Embedded Python Analysis (Alternative Approach)

### 6.1 Overview

Embed CPython within the JVM process via pybind11 + JNI, eliminating the IPC boundary entirely.

### 6.2 Measured Performance

Testing with 100K rows, 10K batch size:
- **Lightweight UDFs:** 20-34% speedup (parse_json: 29%)
- **Medium UDFs:** 5-10% (encode_decode: 10%, url_parse: 8%)
- **Heavy UDFs:** 1-5% (data_mask: 2%)

### 6.3 Why CUDA IPC is Preferred

| Criterion | Embedded Python | CUDA IPC |
|-----------|----------------|----------|
| **Stability risk** | High — Python crash kills JVM | Low — separate process isolation |
| **GIL** | Limits concurrency within executor | N/A — separate process |
| **Dependency** | Requires shared libpython; multi-version .so shipping | Requires cuDF/cupy in Python env |
| **Deployment complexity** | High — conda/pyenv shared Python requirements | Low — standard Python works |
| **Performance gain** | 3-34% (eliminates serde only) | Near-total elimination of transfer cost |
| **Code invasiveness** | Moderate — new JNI/pybind11 bridge | Moderate — new IPC protocol |
| **Spark compatibility** | Bypasses PySpark worker protocol | Works within PySpark worker model |

The embedded Python approach only eliminates **Arrow serialization** overhead (the CPU-side conversion), while CUDA IPC eliminates **both serialization AND the GPU↔Host memory copies**, which are the dominant cost for large batches.

---

## 7. Conclusions

### 7.1 Key Findings

1. **The `pack`/`unpack` mechanism in cuDF is mature and ready** — it's production-proven in multi-GPU Polars and provides exactly the right abstraction for CUDA IPC.

2. **The old PoC (2022) was on the right track** but used a fragile custom JSON metadata format. Today's `packed_columns` metadata is a robust, schema-aware binary format that handles all cuDF types including nested structures.

3. **The ASYNC allocator default is the main new constraint** — must allocate the IPC-exportable buffer with `cudaMalloc` explicitly.

4. **The cuDF PR #11564 closure (May 2025) explicitly endorsed the pack/unpack approach** — this is not speculative; the cuDF maintainers have indicated this is the right path.

5. **No new fundamental blockers exist** — the original issues (contiguous buffer requirement, metadata format, nested type support) are all solved by the current pack/unpack API.

### 7.2 Recommended Next Steps

1. **Prototype in spark-rapids-jni:** Implement the `CudaIpc` JNI wrapper (< 100 lines of code)
2. **Prototype the write path:** Modify `GpuArrowWriter` to pack + IPC handle instead of Arrow IPC serialize
3. **Prototype the Python reader:** Write a RAPIDS-specific Python module that reads IPC handles and unpacks via pylibcudf
4. **Benchmark:** Compare end-to-end Pandas UDF execution times with Arrow IPC vs CUDA IPC
5. **Handle the return path:** Implement Python → JVM CUDA IPC for UDF results
6. **Production hardening:** Buffer lifetime management, error handling, fallback to Arrow IPC
