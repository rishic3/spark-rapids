"""Plot projected T^(n) for CPU, GPU (current), and GPU (IPC) across thread counts.

Two f(n) scenarios are shown per panel:
  f=1           — GPU compute fully serialized (memory-bandwidth-bound)
  f=1/min(n,K)  — GPU scales linearly up to K concurrent kernels

In each scenario, the same f(n) is applied to both Current GPU and IPC GPU
so the comparison is apples-to-apples.
"""

import numpy as np
import matplotlib.pyplot as plt

CONFIGS = {
    "5M, 100k": {
        "Tc_cpu": 21.251, "Tc_gpu": 1.485,
        "Ts": 0.639, "Tp_jvm": 0.196, "Tp_py": 3.445,
    },
    "1M, 100k": {
        "Tc_cpu": 3.815, "Tc_gpu": 0.306,
        "Ts": 0.141, "Tp_jvm": 0.039, "Tp_py": 0.733,
    },
}

K = 4


def t_cpu(n, p):
    return (p["Tc_cpu"] + p["Ts"]) / n + p["Tp_jvm"]


def t_gpu(n, p, f):
    return f * p["Tc_gpu"] + p["Ts"] / n + p["Tp_jvm"] + p["Tp_py"]


def t_ipc(n, p, f):
    return f * p["Tc_gpu"]


def f_const(_n):
    return 1.0


def f_scale(n):
    return 1.0 / min(n, K)


ns = np.arange(1, 33)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, (label, p) in zip(axes, CONFIGS.items()):
    cpu = [t_cpu(n, p) for n in ns]
    gpu_f1 = [t_gpu(n, p, f_const(n)) for n in ns]
    gpu_fk = [t_gpu(n, p, f_scale(n)) for n in ns]
    ipc_f1 = [t_ipc(n, p, f_const(n)) for n in ns]
    ipc_fk = [t_ipc(n, p, f_scale(n)) for n in ns]

    ax.plot(ns, cpu, "o-", color="C0",
            label=r"$T_{\mathrm{CPU}}^{(n)}$", markersize=3)
    ax.plot(ns, gpu_f1, "s-", color="C1",
            label=r"GPU current ($f\!=\!1$)", markersize=3)
    ax.plot(ns, ipc_f1, "s--", color="C1", alpha=0.5,
            label=r"GPU IPC ($f\!=\!1$)", markersize=3)
    ax.plot(ns, gpu_fk, "D-", color="C3",
            label=r"GPU current ($f\!=\!1/\min(n," + str(K) + r")$)",
            markersize=3)
    ax.plot(ns, ipc_fk, "D--", color="C3", alpha=0.5,
            label=r"GPU IPC ($f\!=\!1/\min(n," + str(K) + r")$)",
            markersize=3)

    ax.set_xlabel("Threads (n)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{label} batch — Projected $T^{{(n)}}$")
    ax.legend(fontsize=7)
    ax.set_xlim(1, 32)
    ax.grid(True, alpha=0.3)

fig.tight_layout()
out = "results/projected_tn.png"
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.close(fig)
