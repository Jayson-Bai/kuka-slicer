"""Keep local slicing work from monopolising the workstation CPU.

The native Prusa bridge and numerical export code can both create worker
threads. Limiting Python threads alone therefore is not sufficient. This module
applies temporary process affinity and working-set caps while a slicer task
runs. Windows background priority is available as an explicit opt-in, rather
than a default source of extra latency.
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
MAX_MEMORY_PERCENT_ENV = "KUKA_SLICER_MAX_MEMORY_PERCENT"
DEFAULT_RESOURCE_PERCENT = 70
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
    total_memory_bytes: int | None = None
    max_memory_bytes: int | None = None
    memory_cap_applied: bool = False

    def to_metadata(self) -> dict[str, int | bool]:
        return {
            "available_cores": self.available_cores,
            "max_cores": self.max_cores,
            "affinity_applied": self.affinity_applied,
            "priority_lowered": self.priority_lowered,
            "total_memory_bytes": self.total_memory_bytes,
            "max_memory_bytes": self.max_memory_bytes,
            "memory_cap_applied": self.memory_cap_applied,
        }


def configured_max_cpu_cores(available_cores: int | None = None) -> int:
    """Return a core cap that never exceeds 70% of local logical CPUs."""

    available = max(1, int(available_cores if available_cores is not None else (os.cpu_count() or 1)))
    budget = max(1, available * DEFAULT_RESOURCE_PERCENT // 100)
    requested = os.environ.get(MAX_CPU_CORES_ENV, "").strip()
    if requested:
        try:
            value = int(requested)
        except ValueError:
            value = 0
        if value > 0:
            # An explicit setting is useful for quieter machines, but must
            # never defeat the workstation-protection budget.
            return min(value, budget)
    return budget


def configured_max_memory_percent() -> int:
    """Return a memory percent cap, allowing environment overrides only down."""

    requested = os.environ.get(MAX_MEMORY_PERCENT_ENV, "").strip()
    if requested:
        try:
            value = int(requested)
        except ValueError:
            value = 0
        if value > 0:
            return min(DEFAULT_RESOURCE_PERCENT, max(1, value))
    return DEFAULT_RESOURCE_PERCENT


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


def _windows_total_physical_memory_bytes() -> int | None:
    """Return installed physical RAM through the native Windows API."""

    if sys.platform != "win32":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    kernel32 = _windows_kernel32()
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys) if status.ullTotalPhys > 0 else None


def _get_windows_working_set_limits() -> tuple[int, int, int] | None:
    """Return the current process working-set (minimum, maximum, flags)."""

    if sys.platform != "win32":
        return None
    minimum = ctypes.c_size_t()
    maximum = ctypes.c_size_t()
    flags = wintypes.DWORD()
    kernel32 = _windows_kernel32()
    if not kernel32.GetProcessWorkingSetSizeEx(
        kernel32.GetCurrentProcess(),
        ctypes.byref(minimum),
        ctypes.byref(maximum),
        ctypes.byref(flags),
    ):
        return None
    return int(minimum.value), int(maximum.value), int(flags.value)


def _set_windows_working_set_limits(
    minimum: int,
    maximum: int,
    flags: int,
) -> bool:
    if sys.platform != "win32" or maximum <= 0 or minimum < 0 or minimum > maximum:
        return False
    kernel32 = _windows_kernel32()
    return bool(
        kernel32.SetProcessWorkingSetSizeEx(
            kernel32.GetCurrentProcess(),
            ctypes.c_size_t(minimum),
            ctypes.c_size_t(maximum),
            wintypes.DWORD(flags),
        )
    )


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
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel32.GetProcessWorkingSetSizeEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.GetProcessWorkingSetSizeEx.restype = wintypes.BOOL
    kernel32.SetProcessWorkingSetSizeEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_size_t,
        ctypes.c_size_t,
        wintypes.DWORD,
    ]
    kernel32.SetProcessWorkingSetSizeEx.restype = wintypes.BOOL
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
        total_memory_bytes = _windows_total_physical_memory_bytes()
        max_memory_bytes = (
            total_memory_bytes * configured_max_memory_percent() // 100
            if total_memory_bytes is not None
            else None
        )
        original_working_set = _get_windows_working_set_limits()
        memory_cap_applied = False
        if (
            original_working_set is not None
            and max_memory_bytes is not None
            and original_working_set[0] <= max_memory_bytes
        ):
            try:
                memory_cap_applied = _set_windows_working_set_limits(
                    original_working_set[0],
                    max_memory_bytes,
                    original_working_set[2],
                )
            except OSError:
                memory_cap_applied = False
        info = CpuLimitInfo(
            available_cores=available,
            max_cores=max_cores,
            affinity_applied=affinity_applied,
            priority_lowered=priority_lowered,
            total_memory_bytes=total_memory_bytes,
            max_memory_bytes=max_memory_bytes,
            memory_cap_applied=memory_cap_applied,
        )
        try:
            yield info
        finally:
            if memory_cap_applied and original_working_set is not None:
                _set_windows_working_set_limits(*original_working_set)
            if priority_lowered and original_priority is not None:
                _set_windows_priority(original_priority)
            if affinity_applied and original_affinity is not None:
                _set_cpu_affinity(original_affinity)
