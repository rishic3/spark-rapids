# Python Transfer Profiles

## Theoretical Analysis

### Preliminaries

Let $T_c$ be UDF compute time.  

Let $T_s$ be round-trip serialization time.
- 1 round trip: `writeArrowIPCArrowChunk + readArrowIPCChunkToArrowTable`.

Let $T_p$ be round-trip PCIe transfer time. We decompose $T_p$ into two hops:
- $T_{p,\mathrm{jvm}}$: JVM-side — `convertCudfToArrowTable + convertArrowTableToCudf`
- $T_{p,\mathrm{py}}$: Python-side — `cudf.Series(_) + _.to_pandas()`

Let $T^{(n)}$ be wall-clock time of the entire UDF op (including data transfer) with $n$ threads.

We consider a machine with $n$ CPU cores, 1 PCIe bus, and 1 GPU.

### Single Thread

#### Formulas

Consider using only 1 thread/task on the machine, processing $N$ rows.

##### CPU

```math
\Large T_{\mathrm{CPU}}^{(1)} = T_c^{\mathrm{CPU}} + T_s + T_{p,\mathrm{jvm}}
```

##### GPU

```math
\Large T_{\mathrm{GPU}}^{(1)} = T_c^{\mathrm{GPU}} + T_s + T_{p,\mathrm{jvm}} + T_{p,\mathrm{py}}
```

### Multiple Threads

#### Scaling

Consider running $n$ concurrent instances of the single-thread model, where *each thread* processes $N$ rows.  
We assume that scaling to $n$ threads results in:
- $T_s$: **linear scaling**, since serialization parellelizes across cores.
- $T_c$:
    - **linear scaling** for $T_c^{\mathrm{CPU}}$, since CPU compute parallelizes across cores.
    - **sublinear scaling** for $T_c^{\mathrm{GPU}}$, since kernels are roughly sequentialized on the GPU.
- $T_{p,\mathrm{jvm}}$, $T_{p,\mathrm{py}}$: **sublinear scaling**, since PCIe transfers will be sequentialized once the bus is saturated.

For a simplified model, we assume the components are additive (disregarding pipelining across threads).

#### Formulas

Assume the PCIe bus scales with some $1 \le p(n) \le n$, where 1 is perfect scaling, and $n$ is fully serialized. 
Similarly, assume GPU compute scales with some $1 \le g(n) \le n$. 

##### CPU

```math
\Large T_{\mathrm{CPU}}^{(n)} = T_c^{\mathrm{CPU}} + T_s + p(n)\ T_{p,\mathrm{jvm}}
```

##### GPU

```math
\Large T_{\mathrm{GPU}}^{(n)} = g(n)\ T_c^{\mathrm{GPU}} + T_s + p(n)\ \left(T_{p,\mathrm{jvm}} + T_{p,\mathrm{py}}\right)
```

---

## Empirical Analysis - `ascii_ignore`

Here we consider the [ascii_ignore](experiments/ascii_ignore/src/ascii_ignore_impl.py) UDF and its [GPU implementation](experiments/ascii_ignore/src/ascii_ignore_gpu_impl.py).

### Measuring $T^{(1)}$, $T_c$, and $T_{p,\mathrm{py}}$

The results below measure running the UDF in a Spark job with **1 thread, 1 task**. 

| Experiment          | UDF Impl | Op Time (s) | Compute (s) | % Compute | Py H2D (s) | Py D2H (s) | Overhead (s) | Actual Speedup | Compute Speedup |
| ------------------- | -------- | ----------- | ----------- | --------- | ----------- | ----------- | ------------ | -------------- | --------------- |
| 1M rows, 10k batch  | GPU      | 3.50        | 1.020       | 29.15%    | 0.570       | 0.136       | 1.774        | 1.34           | 4.61            |
|                     | CPU      | 4.70        | 3.631       | 77.26%    |             |             |              |                |                 |
| 1M rows, 100k batch | GPU      | 2.80        | 0.306       | 10.93%    | 0.566       | 0.167       | 1.761        | 1.79           | 16.33           |
|                     | CPU      | 5.00        | 3.815       | 76.30%    |             |             |              |                |                 |
| 5M rows, 10k batch  | GPU      | 13.50       | 4.976       | 36.86%    | 2.521       | 0.616       | 5.387        | 1.67           | 4.54            |
|                     | CPU      | 22.60       | 18.155      | 80.33%    |             |             |              |                |                 |
| 5M rows, 100k batch | GPU      | 10.90       | 1.485       | 13.62%    | 2.670       | 0.775       | 5.970        | 2.44           | 17.92           |
|                     | CPU      | 26.60       | 21.251      | 79.89%    |             |             |              |                |                 |
| 100M rows, 100k batch | GPU    | 210.00      | 30.013      | 14.29%    | 56.504      | 15.844      | 107.639      | 2.23           | 12.44           |
|                       | CPU    | 468.00      | 373.505     | 79.81%    |             |             |              |                |                 |

Where:

- Op Time = $T^{(1)}$
- Compute = $T_c$
- Py H2D + Py D2H = $T_{p,\mathrm{py}}$

### Measuring $T_s$ and $T_{p,\mathrm{jvm}}$

The results below run a pure Java microbenchmark, exercising the JNI calls invoked by `GpuArrowWriter` / `GpuArrowReader`.  
All times are medians of 5 measured runs.

| Experiment  | D2H (s) | IPC ser (s) | IPC deser (s) | H2D (s) | $T_{p,\mathrm{jvm}}$ (s) | $T_s$ (s) | Round-trip (s) |
| ----------- | ------- | ----------- | ------------- | ------- | ------------------------ | ---------- | -------------- |
| 100M, 100k  | 2.00*   | 9.84*       | 2.94*         | 1.92*   | 3.92*                    | 12.78*     | 16.72*         |
| 5M, 100k    | 0.100   | 0.492       | 0.147         | 0.096   | 0.196                    | 0.639      | 0.836          |
| 5M, 10k     | 0.129   | 0.487       | 0.124         | 0.117   | 0.246      | 0.611      | 0.858          |
| 1M, 100k    | 0.020   | 0.113       | 0.027         | 0.019   | 0.039      | 0.141      | 0.179          |
| 1M, 10k     | 0.026   | 0.111       | 0.024         | 0.023   | 0.049      | 0.135      | 0.183          |


\* 100M row values are linearly extrapolated from 5M/100k (×20 batches).

Where:

- D2H = `convertCudfToArrowTable`
- IPC ser = `writeArrowIPCArrowChunk`
- IPC deser = `readArrowIPCChunkToArrowTable`
- H2D = `convertArrowTableToCudf`
- D2H + H2D = $T_{p,\mathrm{jvm}}$
- IPC ser + IPC deser = $T_s$

### Extrapolation to $n$ Threads

#### Measured Parameters (single-thread, per-worker)

| Parameter                          | 5M, 100k | 5M, 10k | 1M, 100k | 1M, 10k |
| ---------------------------------- | -------- | ------- | -------- | ------- |
| $T_c^{\mathrm{CPU}}$ (s)          | 21.251   | 18.155  | 3.815    | 3.631   |
| $T_c^{\mathrm{GPU}}$ (s)          | 1.485    | 4.976   | 0.306    | 1.020   |
| $T_s$ (s)                         | 0.639    | 0.611   | 0.141    | 0.135   |
| $T_{p,\mathrm{jvm}}$ (s)          | 0.196    | 0.246   | 0.039    | 0.049   |
| $T_{p,\mathrm{py}}$ (s)           | 3.445    | 3.137   | 0.733    | 0.706   |

Note: The model underestimates observed Spark op times at $n=1$ (possibly Spark's internal Arrow-Pandas conversion, socket I/O, cuInit, etc.). The projections below are therefore conservative regarding overhead.

#### Projected CUDA IPC Speedup — 5M rows, 100k batch

We consider two models for $g(n)$ (GPU compute scaling):

- **$g(n) = n$**: GPU compute is fully serialized — $n$ workers time-slice.
- **$g(n) = \max(1,\, n/4)$**: GPU scales linearly up to 4 concurrent kernels, then serializes.

...and two models for $p(n)$ (PCIe/transfer scaling):

- **$p(n) = 1$**: Transfers fully parallelize across CPU cores (bus not saturated).
- **$p(n) = n$**: Transfers fully serialized (bus saturated, all workers queue).

We assume for simplicity that CUDA IPC eliminates transfer and serialization, ($T_s, T_{p,\mathrm{jvm}}, T_{p,\mathrm{py}} \to 0$):

```math
\Large T_{\mathrm{GPU,IPC}}^{(n)} = g(n)\ T_c^{\mathrm{GPU}}
```

The IPC speedup is therefore:

```math
\Large \frac{T_{\mathrm{GPU}}^{(n)}}{T_{\mathrm{GPU,IPC}}^{(n)}} = 1 + \frac{T_s + p(n)\left(T_{p,\mathrm{jvm}} + T_{p,\mathrm{py}}\right)}{g(n)\ T_c^{\mathrm{GPU}}}
```

##### Scenario A: $g(n) = n$, $p(n) = 1$

| $n$ | $T_\mathrm{CPU}^{(n)}$ | $T_\mathrm{GPU}^{(n)}$ | $T_\mathrm{GPU,IPC}^{(n)}$ | IPC Speedup |
| --- | ----------------------- | ----------------------- | --------------------------- | ----------- |
| 1   | 22.1                    | 5.8                     | 1.5                         | 3.9×        |
| 4   | 22.1                    | 10.2                    | 5.9                         | 1.7×        |
| 8   | 22.1                    | 16.2                    | 11.9                        | 1.4×        |
| 16  | 22.1                    | 28.0                    | 23.8                        | 1.2×        |
| 32  | 22.1                    | 51.8                    | 47.5                        | 1.1×        |

##### Scenario B: $g(n) = \max(1, n/4)$, $p(n) = 1$

| $n$ | $T_\mathrm{CPU}^{(n)}$ | $T_\mathrm{GPU}^{(n)}$ | $T_\mathrm{GPU,IPC}^{(n)}$ | IPC Speedup |
| --- | ----------------------- | ----------------------- | --------------------------- | ----------- |
| 1   | 22.1                    | 5.8                     | 1.5                         | 3.9×        |
| 4   | 22.1                    | 5.8                     | 1.5                         | 3.9×        |
| 8   | 22.1                    | 7.2                     | 3.0                         | 2.4×        |
| 16  | 22.1                    | 10.2                    | 5.9                         | 1.7×        |
| 32  | 22.1                    | 16.2                    | 11.9                        | 1.4×        |

##### Scenario C: $g(n) = n$, $p(n) = n$

| $n$ | $T_\mathrm{CPU}^{(n)}$ | $T_\mathrm{GPU}^{(n)}$ | $T_\mathrm{GPU,IPC}^{(n)}$ | IPC Speedup |
| --- | ----------------------- | ----------------------- | --------------------------- | ----------- |
| 1   | 22.1                    | 5.8                     | 1.5                         | 3.9×        |
| 4   | 22.7                    | 21.1                    | 5.9                         | 3.6×        |
| 8   | 23.5                    | 41.6                    | 11.9                        | 3.5×        |
| 16  | 25.0                    | 82.7                    | 23.8                        | 3.5×        |
| 32  | 28.2                    | 164.7                   | 47.5                        | 3.5×        |

---

## Empirical Analysis - `metric_status`

Here we consider [metric_status](experiments/ascii_ignore/src/metric_status_impl.py) UDF and its [GPU implementation](experiments/ascii_ignore/src/metric_status_gpu_impl.py).

### Measuring $T^{(1)}$, $T_c$, and $T_{p,\mathrm{py}}$

| Experiment          | UDF Impl | Op Time (s) | Compute (s) | % Compute | Py H2D (s) | Py D2H (s) | Overhead (s) | Actual Speedup | Compute Speedup |
| ------------------- | -------- | ----------- | ----------- | --------- | ----------- | ----------- | ------------ | -------------- | --------------- |
| 5M rows, 100k batch | GPU      | 6.20        | 0.479       | 7.7%      | 1.838       | 0.061       | 5.721        | 0.98           | 5.95            |
|                     | CPU      | 6.10        | 2.851       | 46.7%     |             |             |              |                |                 |

### Measuring $T_s$ and $T_{p,\mathrm{jvm}}$

| Experiment  | D2H (s) | IPC ser (s) | IPC deser (s) | H2D (s) | $T_{p,\mathrm{jvm}}$ (s) | $T_s$ (s) | Round-trip (s) |
| ----------- | ------- | ----------- | ------------- | ------- | ------------------------ | ---------- | -------------- |
| 5M, 100k    | 0.183   | 0.993       | 0.420         | 0.160   | 0.343                    | 1.413      | 1.756          |

### Extrapolation to $n$ Threads

#### Measured Parameters (single-thread, per-worker)

| Parameter                          | 5M, 100k |
| ---------------------------------- | -------- |
| $T_c^{\mathrm{CPU}}$ (s)          | 2.851    |
| $T_c^{\mathrm{GPU}}$ (s)          | 0.479    |
| $T_s$ (s)                         | 1.413    |
| $T_{p,\mathrm{jvm}}$ (s)          | 0.343    |
| $T_{p,\mathrm{py}}$ (s)           | 1.899    |

#### Projected CUDA IPC Speedup — 5M rows, 100k batch

##### Scenario A: $g(n) = n$, $p(n) = n$

| $n$ | $T_\mathrm{CPU}^{(n)}$ | $T_\mathrm{GPU}^{(n)}$ | $T_\mathrm{GPU,IPC}^{(n)}$ | IPC Speedup |
| --- | ----------------------- | ----------------------- | --------------------------- | ----------- |
| 1   | 4.6                     | 4.1                     | 0.5                         | 8.6×        |
| 4   | 5.6                     | 12.3                    | 1.9                         | 6.4×        |
| 8   | 7.0                     | 23.2                    | 3.8                         | 6.0×        |
| 16  | 9.8                     | 44.9                    | 7.7                         | 5.9×        |
| 32  | 15.2                    | 88.5                    | 15.3                        | 5.8×        |

##### Scenario B: $g(n) = \max(1, n/4)$, $p(n) = 1$

| $n$ | $T_\mathrm{CPU}^{(n)}$ | $T_\mathrm{GPU}^{(n)}$ | $T_\mathrm{GPU,IPC}^{(n)}$ | IPC Speedup |
| --- | ----------------------- | ----------------------- | --------------------------- | ----------- |
| 1   | 4.6                     | 4.1                     | 0.5                         | 8.6×        |
| 4   | 4.6                     | 4.1                     | 0.5                         | 8.6×        |
| 8   | 4.6                     | 4.6                     | 1.0                         | 4.8×        |
| 16  | 4.6                     | 5.6                     | 1.9                         | 2.9×        |
| 32  | 4.6                     | 7.5                     | 3.8                         | 2.0×        |

##### Scenario C: $g(n) = n$, $p(n) = 1$

| $n$ | $T_\mathrm{CPU}^{(n)}$ | $T_\mathrm{GPU}^{(n)}$ | $T_\mathrm{GPU,IPC}^{(n)}$ | IPC Speedup |
| --- | ----------------------- | ----------------------- | --------------------------- | ----------- |
| 1   | 4.6                     | 4.1                     | 0.5                         | 8.6×        |
| 4   | 4.6                     | 5.6                     | 1.9                         | 2.9×        |
| 8   | 4.6                     | 7.5                     | 3.8                         | 2.0×        |
| 16  | 4.6                     | 11.3                    | 7.7                         | 1.5×        |
| 32  | 4.6                     | 19.0                    | 15.3                        | 1.2×        |

