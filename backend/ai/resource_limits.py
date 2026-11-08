"""Central resource limits for interactive ML operations."""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any

from config import (MAX_CONCURRENT_JOBS, MAX_DATASET_ROWS, MAX_DATASET_SIZE_MB,
                    MAX_MEMORY_MB, MAX_OPTIMIZATION_TRIALS)


class ResourceLimitError(ValueError):
    pass


_jobs = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)


def limits() -> dict[str, int]:
    return {"max_dataset_rows": MAX_DATASET_ROWS, "max_dataset_size_bytes": MAX_DATASET_SIZE_MB * 1024 * 1024,
            "max_memory_bytes": MAX_MEMORY_MB * 1024 * 1024, "max_optimization_trials": MAX_OPTIMIZATION_TRIALS,
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS}


def violations(estimate: dict[str, Any], tool_name: str, configuration: dict[str, Any]) -> list[str]:
    current = limits(); problems = []
    if estimate.get("rows", 0) > current["max_dataset_rows"]:
        problems.append(f"Dataset rows ({estimate['rows']}) exceed the configured limit ({current['max_dataset_rows']})")
    if estimate.get("dataset_bytes") is not None and estimate["dataset_bytes"] > current["max_dataset_size_bytes"]:
        problems.append("Dataset size exceeds the configured limit")
    if estimate.get("estimated_memory_bytes") is not None and estimate["estimated_memory_bytes"] > current["max_memory_bytes"]:
        problems.append("Estimated memory exceeds the configured limit")
    if tool_name == "optimize_workspace":
        trials = configuration.get("n_trials") or configuration.get("n_iter") or configuration.get("optimization", {}).get("n_trials") or configuration.get("optimization", {}).get("n_iter")
        if trials and int(trials) > current["max_optimization_trials"]:
            problems.append(f"Optimization trials ({trials}) exceed the configured limit ({current['max_optimization_trials']})")
    return problems


def enforce(estimate: dict[str, Any], tool_name: str, configuration: dict[str, Any]) -> None:
    problems = violations(estimate, tool_name, configuration)
    if problems:
        raise ResourceLimitError("; ".join(problems))


@contextmanager
def job_slot():
    if not _jobs.acquire(blocking=False):
        raise ResourceLimitError("Maximum concurrent ML jobs reached")
    try:
        yield
    finally:
        _jobs.release()
