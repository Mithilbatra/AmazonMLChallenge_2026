"""Logging, timing, seeding and serialisation helpers."""
from __future__ import annotations

import json
import logging
import os
import pickle
import random
import sys
import time
import zlib
from contextlib import contextmanager
from typing import Any

import numpy as np

LOGGER_NAME = "er"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = get_logger()
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def rss_mb() -> float:
    """Current resident memory of this process in MB (Linux /proc; else peak)."""
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, AttributeError):
        return peak_rss_mb()


def peak_rss_mb() -> float:
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024 if sys.platform != "darwin" else peak / 2**20
    except (ImportError, OSError):
        return float("nan")


def release_memory() -> None:
    """Garbage-collect and hand freed heap pages back to the OS (glibc
    malloc_trim). Without the trim a long-lived notebook kernel keeps the
    peak memory of earlier stages reserved."""
    import gc
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _proc_status_mb(field: str) -> float | None:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def mem_str() -> str:
    """Memory report for the logs. On Linux, real process memory (anonymous)
    is shown separately from pages of memory-mapped files (the disk-backed
    feature matrix): the latter is page cache the OS can drop at any time and
    is not what causes out-of-memory kills."""
    anon, mapped = _proc_status_mb("RssAnon"), _proc_status_mb("RssFile")
    if anon is None:
        return f"RSS {rss_mb():,.0f} MB (peak {peak_rss_mb():,.0f} MB)"
    return f"mem {anon:,.0f} MB (+{mapped:,.0f} MB mapped files; peak total {peak_rss_mb():,.0f} MB)"


@contextmanager
def timer(name: str, store: dict | None = None):
    log = get_logger()
    log.info(">> %s ...  [%s]", name, mem_str())
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        log.info("<< %s done in %.1fs  [%s]", name, dt, mem_str())
        if store is not None:
            store[name] = round(dt, 3)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_pickle(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def stable_hash(text: str) -> int:
    """Deterministic 32-bit hash (Python's hash() is salted per process)."""
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return int(n_jobs)


def describe_counts(values) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }
