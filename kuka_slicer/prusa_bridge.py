"""Optional bridge to the bundled PrusaSlicer geometry extension.

The Python package must remain usable when the native extension is not built,
for example in a source checkout or on a platform without a matching wheel.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_NATIVE_MODULE_NAME = "kuka_slicer._native.prusa_bridge"


class PrusaBridgeUnavailable(RuntimeError):
    """Raised when a caller requests the optional compiled bridge."""


def _load_native() -> tuple[ModuleType | None, str]:
    try:
        return import_module(_NATIVE_MODULE_NAME), ""
    except (ImportError, OSError) as error:
        return None, str(error)


def bridge_info() -> dict[str, bool | str | None]:
    """Return native bridge availability without making it a runtime requirement."""

    native, reason = _load_native()
    if native is None:
        return {
            "available": False,
            "reason": reason or "Prusa native extension is not installed",
            "native_version": None,
        }
    native_version: Any = getattr(native, "__version__", None)
    return {
        "available": True,
        "reason": "",
        "native_version": str(native_version) if native_version is not None else None,
    }


def require_native() -> ModuleType:
    """Return the compiled extension or raise an actionable availability error."""

    native, reason = _load_native()
    if native is not None:
        return native
    detail = reason or "Prusa native extension is not installed"
    raise PrusaBridgeUnavailable(f"Prusa native bridge is not available: {detail}")
