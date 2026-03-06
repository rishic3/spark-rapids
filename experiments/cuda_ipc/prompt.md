**Role and Objective**
You are an expert systems engineer and AI researcher. Your goal is to conduct a highly in-depth, comprehensive investigation into a complex feature implementation: "Efficient JVM to Python Transfer in Spark using CUDA IPC". This feature spans multiple repositories (Spark RAPIDS, cuDF, spark-rapids-jni) and has a historical context of issues, design docs, and proof-of-concept branches.

**Context and Starting Point**
Please start by thoroughly reading the root document: `spark/cuda_ipc/cuda_ipc.md`. This document contains references to local architecture files, design documents (PDF, TXT, and MD formats), and several external GitHub issues, pull requests, and branches.

**Required Actions & Methodology**
To successfully complete this research, you must do the following:

1. **Document Analysis:** Read and synthesize all local documents referenced in the root markdown file, including `current_architecture.md`, `cuda_ipc_doc.pdf` / `cuda_ipc_doc.txt`, and `embedded_python_doc.md`.
2. **External Context Gathering:** Use your web/browser tools or curl/gh commands to read through all linked GitHub issues and PRs to understand the historical context, blockers, and feature requests.
3. **Deep Codebase Exploration:** 
   - Feel free to `git clone` or fetch the relevant repositories (e.g., `NVIDIA/spark-rapids`, `rapidsai/cudf`, `wbo4958/spark-rapids-jni`).
   - Checkout the specific PoC branches mentioned in the document (`cuda-ipc`).
   - Use your **Task / subagent / explore tools** to concurrently investigate these repositories. Delegate repository exploration to subagents to manage context effectively and analyze the codebase structure, current implementation details, and differences between the PoC branches and the main branches.
4. **Current State Assessment:** Determine the delta between the historical PoCs/issues (which are a few years old) and the current state of these repositories today. Identify which blockers still exist, which have been resolved, and if new architectural constraints have emerged.

**Deliverable**
Produce a highly detailed, structured engineering report that includes:
*   **Executive Summary:** A high-level overview of the feature's goal (Efficient JVM to Python Transfer) and the two primary approaches (CUDA IPC vs. Embedded Python).
*   **Current Implementation & Architecture:** A detailed breakdown of how the systems are currently structured, based on your codebase exploration.
*   **Historical Context & Blockers:** A summary of the original challenges outlined in the old issues and PRs.
*   **Modern Viability & Nuances:** An in-depth analysis of how this feature could be implemented *today*. Are the old PoCs still viable? What has changed in cuDF or Spark RAPIDS that affects this? 
*   **Implementation Blueprint:** A proposed, updated path forward for implementing the CUDA IPC approach, highlighting specific code areas that will need modification and potential pitfalls to watch out for.

Please be exhaustive in your research. Take your time, launch subagents as necessary to dig into the codebases, and compile all your findings into the final report.
