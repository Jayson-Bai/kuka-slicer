"""Provider-specific height fields behind the shared lattice pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .contracts import ConformalLatticeSpec
from ..surface_preview.model import DoubleSineSurface


class HeightField(Protocol):
    """Small geometry boundary used by path embedding."""

    z_reference_mm: float

    def height(self, x_mm, y_mm): ...

    def gradient(self, x_mm, y_mm) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True, slots=True)
class PlanarSurface:
    """The Z=0 print plane without synthetic sinusoidal parameters."""

    z_reference_mm: float = 0.0

    def height(self, x_mm, y_mm):
        x, y = np.broadcast_arrays(
            np.asarray(x_mm, dtype=np.float64),
            np.asarray(y_mm, dtype=np.float64),
        )
        return np.full(x.shape, self.z_reference_mm, dtype=np.float64)

    def gradient(self, x_mm, y_mm) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.broadcast_arrays(
            np.asarray(x_mm, dtype=np.float64),
            np.asarray(y_mm, dtype=np.float64),
        )
        return np.zeros(x.shape, dtype=np.float64), np.zeros(y.shape, dtype=np.float64)


def height_field_from_spec(spec: ConformalLatticeSpec) -> HeightField:
    """Resolve a validated generated-source provider to one height field."""

    if spec.source_provider == "planar":
        return PlanarSurface()
    if spec.source_provider == "double_sine":
        values = spec.source_surface.get("double_sine")
        if not isinstance(values, dict):
            raise ValueError("double-sine source metadata is malformed")
        return DoubleSineSurface(
            amplitude_mm=float(values["amplitude_mm"]),
            wavelength_x_mm=float(values["wavelength_x_mm"]),
            wavelength_y_mm=float(values["wavelength_y_mm"]),
            phase_x_rad=float(values["phase_x_rad"]),
            phase_y_rad=float(values["phase_y_rad"]),
            z_reference_mm=float(values["z_reference_mm"]),
        )
    raise ValueError(f"source provider {spec.source_provider!r} is not a generated height field")
