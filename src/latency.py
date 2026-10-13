"""CPU latency benchmark shared by paper_results.

Reports median and p95 milliseconds PER PROMPT at batch size 1 and 16, plus the
CPU model and thread count, so latency numbers are only compared on the same
machine.
"""
from __future__ import annotations

import os
import platform
import statistics
import subprocess
import time

import numpy as np

BATCH_SIZES = (1, 16)
WARMUP_CALLS = 3


def cpu_model() -> str:
    """Human-readable CPU name (best effort, per OS)."""
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if system == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def thread_count() -> int | None:
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:
        return None


def benchmark(predict, texts: list[str], batch_sizes=BATCH_SIZES) -> dict:
    """predict(list[str]) is called on consecutive batches of *texts*.

    Returns {"batch_<n>": {"median_ms_per_prompt", "p95_ms_per_prompt", "calls"}}.
    """
    out = {}
    for bs in batch_sizes:
        batches = [texts[i : i + bs] for i in range(0, len(texts) - bs + 1, bs)] or [texts[:bs]]
        for batch in batches[:WARMUP_CALLS]:
            predict(batch)
        per_prompt = []
        for batch in batches:
            t0 = time.perf_counter()
            predict(batch)
            per_prompt.append((time.perf_counter() - t0) * 1000 / len(batch))
        out[f"batch_{bs}"] = {
            "median_ms_per_prompt": float(statistics.median(per_prompt)),
            "p95_ms_per_prompt": float(np.percentile(per_prompt, 95)),
            "mean_ms_per_prompt": float(statistics.fmean(per_prompt)),
            "calls": len(per_prompt),
        }
    return out


def machine_info() -> dict:
    return {
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": thread_count(),
        "platform": platform.platform(),
    }
