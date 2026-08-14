from __future__ import annotations

from types import SimpleNamespace

from kuka_slicer import cpu_limiter


def test_default_cpu_cap_uses_at_most_seventy_percent_of_available_cores(monkeypatch) -> None:
    monkeypatch.delenv(cpu_limiter.MAX_CPU_CORES_ENV, raising=False)

    assert cpu_limiter.configured_max_cpu_cores(16) == 11
    assert cpu_limiter.configured_max_cpu_cores(3) == 2
    assert cpu_limiter.configured_max_cpu_cores(1) == 1


def test_cpu_cap_environment_override_is_bounded_by_available_cores(monkeypatch) -> None:
    monkeypatch.setenv(cpu_limiter.MAX_CPU_CORES_ENV, "6")
    assert cpu_limiter.configured_max_cpu_cores(16) == 6

    monkeypatch.setenv(cpu_limiter.MAX_CPU_CORES_ENV, "999")
    assert cpu_limiter.configured_max_cpu_cores(8) == 5

    monkeypatch.setenv(cpu_limiter.MAX_CPU_CORES_ENV, "not-a-number")
    assert cpu_limiter.configured_max_cpu_cores(8) == 5


def test_task_limiter_applies_and_restores_affinity(monkeypatch) -> None:
    calls: list[tuple[int, ...]] = []
    monkeypatch.delenv(cpu_limiter.MAX_CPU_CORES_ENV, raising=False)
    available_cpus = tuple(range(2, 34, 2))
    monkeypatch.setattr(cpu_limiter, "_get_cpu_affinity", lambda: available_cpus)
    monkeypatch.setattr(
        cpu_limiter,
        "_set_cpu_affinity",
        lambda cpus: calls.append(cpus) or True,
    )
    monkeypatch.setattr(cpu_limiter, "_get_windows_priority", lambda: None)
    monkeypatch.setattr(cpu_limiter, "_windows_total_physical_memory_bytes", lambda: None)
    monkeypatch.setattr(cpu_limiter, "_get_windows_working_set_limits", lambda: None)

    with cpu_limiter.limit_slicer_task() as info:
        assert info.available_cores == 16
        assert info.max_cores == 11
        assert info.affinity_applied is True
        assert info.priority_lowered is False

    assert calls == [available_cpus[:11], available_cpus]


def test_low_priority_is_opt_in(monkeypatch) -> None:
    monkeypatch.delenv(cpu_limiter.LOW_PRIORITY_ENV, raising=False)
    assert cpu_limiter.low_priority_requested() is False

    monkeypatch.setenv(cpu_limiter.LOW_PRIORITY_ENV, "true")
    assert cpu_limiter.low_priority_requested() is True


def test_memory_percent_defaults_to_seventy_and_can_only_be_lowered(monkeypatch) -> None:
    monkeypatch.delenv(cpu_limiter.MAX_MEMORY_PERCENT_ENV, raising=False)
    assert cpu_limiter.configured_max_memory_percent() == 70

    monkeypatch.setenv(cpu_limiter.MAX_MEMORY_PERCENT_ENV, "45")
    assert cpu_limiter.configured_max_memory_percent() == 45

    monkeypatch.setenv(cpu_limiter.MAX_MEMORY_PERCENT_ENV, "90")
    assert cpu_limiter.configured_max_memory_percent() == 70


def test_windows_api_declarations_are_pointer_width_safe(monkeypatch) -> None:
    fake = SimpleNamespace(
        GetCurrentProcess=lambda: 1,
        GetProcessAffinityMask=lambda *_args: 1,
        SetProcessAffinityMask=lambda *_args: 1,
        GetPriorityClass=lambda *_args: 1,
        SetPriorityClass=lambda *_args: 1,
        GlobalMemoryStatusEx=lambda *_args: 1,
        GetProcessWorkingSetSizeEx=lambda *_args: 1,
        SetProcessWorkingSetSizeEx=lambda *_args: 1,
    )
    monkeypatch.setattr(cpu_limiter.ctypes, "WinDLL", lambda *_args, **_kwargs: fake)

    assert cpu_limiter._windows_kernel32() is fake
    assert fake.GetProcessAffinityMask.argtypes[0] is cpu_limiter.wintypes.HANDLE
    assert fake.SetProcessAffinityMask.argtypes[1] is cpu_limiter.ctypes.c_size_t
    assert fake.GetProcessWorkingSetSizeEx.argtypes[1] == cpu_limiter.ctypes.POINTER(
        cpu_limiter.ctypes.c_size_t
    )
