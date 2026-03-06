# Efficient JVM to Python Transfer in Spark

## Current Architecture

See [current architecture doc](current_architecture.md).

See [Spark RAPIDS repository](github.com/nvidia/spark-rapids).

## CUDA IPC Approach

### Relevant Github Links

Note - many of these issues and links are a few years old at this point.  
Any reference code may be changed, and certain issues may no longer be blockers. There also may be new blockers.

#### Issues

[Spark-RAPIDS Issue: CUDA IPC feature request](https://github.com/NVIDIA/spark-rapids/issues/5561)

[cuDF Issue: Interchange protocol for dataframe 'descriptor' feature request](https://github.com/rapidsai/cudf/issues/11514)

[cuDF Issue: How to reconstruct a dataframe](https://github.com/rapidsai/cudf/issues/11462)

[cuDF Issue: Removing Arrow IPC related code](https://github.com/rapidsai/cudf/issues/10994)

#### PRs / Branches

[cuDF Draft PR: CUDA IPC implementation](https://github.com/rapidsai/cudf/pull/11564)

[spark-rapids-jni Branch: PoC for CUDA IPC](https://github.com/wbo4958/spark-rapids-jni/tree/cuda-ipc)

[spark-rapids: PoC for CUDA IPC](https://github.com/wbo4958/spark-rapids/tree/cuda-ipc)

### Relevant Design Docs

Old design doc:

[Design Doc (PDF)](cuda_ipc_doc.pdf)

[Design Doc (TXT)](cuda_ipc_doc.txt)

## Embedded Python Approach

[Design Doc (MD)](embedded_python_doc.md)
