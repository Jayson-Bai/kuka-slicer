"""Pure mathematical primitives for the graded-curvature surface preview."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class DoubleSineSurface:
    """The initial ``graded_surface_v1`` double-sine height field in millimetres."""

    amplitude_mm: float = 0.8
    wavelength_x_mm: float = 40.0
    wavelength_y_mm: float = 50.0
    phase_x_rad: float = 0.0
    phase_y_rad: float = 0.0
    z_reference_mm: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "amplitude_mm", _finite(self.amplitude_mm, "amplitude_mm"))
        object.__setattr__(self, "phase_x_rad", _finite(self.phase_x_rad, "phase_x_rad"))
        object.__setattr__(self, "phase_y_rad", _finite(self.phase_y_rad, "phase_y_rad"))
        object.__setattr__(self, "z_reference_mm", _finite(self.z_reference_mm, "z_reference_mm"))
        for name in ("wavelength_x_mm", "wavelength_y_mm"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def height(self, x_mm, y_mm):
        """Return ``H(x, y)`` while preserving NumPy array shapes."""

        x = np.asarray(x_mm, dtype=float)
        y = np.asarray(y_mm, dtype=float)
        return self.z_reference_mm + self.amplitude_mm * np.sin(
            (2.0 * np.pi * x) / self.wavelength_x_mm + self.phase_x_rad
        ) * np.sin((2.0 * np.pi * y) / self.wavelength_y_mm + self.phase_y_rad)

    def gradient(self, x_mm, y_mm) -> tuple[np.ndarray, np.ndarray]:
        """Return the analytical ``(dH/dx, dH/dy)`` gradient."""

        x = np.asarray(x_mm, dtype=float)
        y = np.asarray(y_mm, dtype=float)
        x_phase = (2.0 * np.pi * x) / self.wavelength_x_mm + self.phase_x_rad
        y_phase = (2.0 * np.pi * y) / self.wavelength_y_mm + self.phase_y_rad
        dx = (
            self.amplitude_mm
            * (2.0 * np.pi / self.wavelength_x_mm)
            * np.cos(x_phase)
            * np.sin(y_phase)
        )
        dy = (
            self.amplitude_mm
            * (2.0 * np.pi / self.wavelength_y_mm)
            * np.sin(x_phase)
            * np.cos(y_phase)
        )
        return dx, dy

    def sample_grid(
        self,
        *,
        width_mm: float,
        height_mm: float,
        samples: int = 48,
        x_min_mm: float | None = None,
        y_min_mm: float | None = None,
    ) -> "SurfaceGrid":
        width_mm = _finite(width_mm, "width_mm")
        height_mm = _finite(height_mm, "height_mm")
        if width_mm <= 0.0 or height_mm <= 0.0:
            raise ValueError("preview dimensions must be positive")
        if samples < 2:
            raise ValueError("samples must be at least 2")

        if x_min_mm is None:
            x_min_mm = -width_mm / 2.0
        if y_min_mm is None:
            y_min_mm = -height_mm / 2.0
        x_min_mm = _finite(x_min_mm, "x_min_mm")
        y_min_mm = _finite(y_min_mm, "y_min_mm")
        x_axis = np.linspace(x_min_mm, x_min_mm + width_mm, samples)
        y_axis = np.linspace(y_min_mm, y_min_mm + height_mm, samples)
        x, y = np.meshgrid(x_axis, y_axis)
        z = self.height(x, y)
        return SurfaceGrid(x=x, y=y, z=z, surface=self)


@dataclass(frozen=True, slots=True)
class SurfaceGrid:
    """A sampled surface grid and its basic geometry diagnostics."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    surface: DoubleSineSurface

    @property
    def max_slope(self) -> float:
        dx, dy = self.surface.gradient(self.x, self.y)
        return float(np.max(np.hypot(dx, dy)))

    def summary(self) -> dict[str, float]:
        return {
            "z_min_mm": float(np.min(self.z)),
            "z_max_mm": float(np.max(self.z)),
            "z_range_mm": float(np.max(self.z) - np.min(self.z)),
            "max_slope": self.max_slope,
        }
