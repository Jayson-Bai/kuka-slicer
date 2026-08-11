"""Keep local slicing work from monopolising the workstation CPU.

The native Prusa bridge and numerical export code can both create worker
threads.  Limiting Python threads alone therefore is not sufficient.  This
module applies a temporary *process* affinity cap while a slicer task runs.
Windows background priority is available as an explicit opt-in, rather than a
default source of extra latency.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
from ctypes import wintypes
import os
import sys
import threading
from typing import Iterator


MAX_CPU_CORES_ENV = "KUKA_SLICER_MAX_CPU_CORES"
LOW_PRIORITY_ENV = "KUKA_SLICER_LOW_PRIORITY"
DEFAULT_MAX_CPU_CORES = 12
_NUMERIC_THREAD_ENVS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_TASK_LIMIT_LOCK = threading.RLock()


@dataclass(frozen=True)
class CpuLimitInfo:
    """Effective CPU limit applied to one local slicer task."""

    available_cores: int
    max_cores: int
    affinity_applied: bool
    priority_lowered: bool

    def to_metadata(self) -> dict[str, int | bool]:
        return {
            "available_cores": self.available_cores,
            "max_cores": self.max_cores,
            "affinity_applied": self.affinity_applied,
            "priority_lowered": self.priority_lowered,
        }


def configured_max_cpu_cores(available_cores: int | None = None) -> int:
    """Return the requested core cap, defaulting to at most 12 cores."""

    available = max(1, int(available_cores if available_cores is not None else (os.cpu_count() or 1)))
    requested = os.environ.get(MAX_CPU_CORES_ENV, "").strip()
    if requested:
        try:
            value = int(requested)
        except ValueError:
            value = 0
        if value > 0:
            return min(value, available)
    return min(DEFAULT_MAX_CPU_CORES, available)


def low_priority_requested() -> bool:
    """Return whether a caller explicitly wants background scheduling."""

    return os.environ.get(LOW_PRIORITY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def configure_numeric_thread_environment() -> int:
    """Set conservative numerical-library thread defaults before they import.

    Explicit user environment settings always win.  Affinity in
    :func:`limit_slicer_task` remains the final, library-independent cap.
    """

    max_cores = configured_max_cpu_cores()
    for name in _NUMERIC_THREAD_ENVS:
        os.environ.setdefault(name, str(max_cores))
    return max_cores


def _get_cpu_affinity() -> tuple[int, ...] | None:
    if sys.platform == "win32":
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        kernel32 = _windows_kernel32()
        if not kernel32.GetProcessAffinityMask(
            kernel32.GetCurrentProcess(),
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        ):
            return None
        return tuple(index for index in range(process_mask.value.bit_length()) if process_mask.value & (1 << index))
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return None


def _set_cpu_affinity(cpus: tuple[int, ...]) -> bool:
    if not cpus:
        return False
    if sys.platform == "win32":
        mask = sum(1 << cpu for cpu in cpus)
        kernel32 = _windows_kernel32()
        return bool(kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.c_size_t(mask)))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))
        return True
    return False


def _get_windows_priority() -> int | None:
    if sys.platform != "win32":
        return None
    kernel32 = _windows_kernel32()
    value = int(kernel32.GetPriorityClass(kernel32.GetCurrentProcess()))
    return value or None


def _set_windows_priority(priority: int) -> bool:
    if sys.platform != "win32":
        return False
    kernel32 = _windows_kernel32()
    return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), priority))


def _windows_kernel32():
    """Return Kernel32 with pointer-width-safe process API signatures."""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetProcessAffinityMask.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel32.GetPriorityClass.restype = wintypes.DWORD
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    return kernel32


@contextmanager
def limit_slicer_task() -> Iterator[CpuLimitInfo]:
    """Run one slicer task with an affinity and priority cap.

    The lock serialises local slicing jobs.  Affinity is process-wide, so this
    prevents two browser requests from combining into a full-CPU workload and
    makes nested calls safe.
    """

    with _TASK_LIMIT_LOCK:
        original_affinity = _get_cpu_affinity()
        available = len(original_affinity) if original_affinity else max(1, os.cpu_count() or 1)
        max_cores = configured_max_cpu_cores(available)
        selected = original_affinity[:max_cores] if original_affinity else ()
        affinity_applied = False
        if selected and len(selected) < len(original_affinity):
            try:
                affinity_applied = _set_cpu_affinity(selected)
            except OSError:
                affinity_applied = False

        original_priority = _get_windows_priority() if low_priority_requested() else None
        # CPU affinity is the default protection.  Lower scheduling priority
        # remains an opt-in background-mode setting because it can make a
        # single-threaded geometry phase unnecessarily slow.
        priority_lowered = bool(
            original_priority is not None and _set_windows_priority(0x00004000)
        )
        info = CpuLimitInfo(
            available_cores=available,
            max_cores=max_cores,
            affinity_applied=affinity_applied,
            priority_lowered=priority_lowered,
        )
        try:
            yield info
        finally:
            if priority_lowered and original_priority is not None:
                _set_windows_priority(original_priority)
            if affinity_applied and original_affinity is not None:
                _set_cpu_affinity(original_affinity)
