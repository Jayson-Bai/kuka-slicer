"""Portable input/output contracts for the independent surface mapper."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
import re
from typing import Any, Mapping

import numpy as np

from ..external_npz import LOGICAL_LAYER_SEMANTICS_V1, SOURCE_NPZ_CONTRACT_ID
from ..surface_preview.model import DoubleSineSurface


_PATH_KEY = re.compile(r"^layer_(\d{4,})_([RFT])$")
_E_KEY = re.compile(r"^layer_(\d{4,})_([RF])_E$")


@dataclass(frozen=True, slots=True)
class SurfaceTarget:
    """The geometric target exported by the surface-preview package."""

    surface: DoubleSineSurface
    width_mm: float
    height_mm: float
    source_file_name: str
    source_sha256: str
    raw_config: dict[str, object]


@dataclass(slots=True)
class SourceNPZ:
    """Validated external source arrays, kept in their original padded layout."""

    arrays: dict[str, np.ndarray]
    meta: dict[str, object]
    source_name: str = "flat.npz"

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted({int(match.group(1)) for key in self.arrays if (match := _PATH_KEY.match(key))}))

    @property
    def path_keys(self) -> tuple[str, ...]:
        return tuple(sorted(key for key in self.arrays if _PATH_KEY.match(key)))

    @property
    def xy_bounds_mm(self) -> tuple[float, float, float, float]:
        points = [array[..., :2][np.isfinite(array[..., 0])] for key, array in self.arrays.items() if _PATH_KEY.match(key)]
        joined = np.vstack(points) if points else np.empty((0, 2), dtype=np.float64)
        if not joined.size:
            raise ValueError("source NPZ does not contain any path points")
        return (
            float(np.min(joined[:, 0])), float(np.min(joined[:, 1])),
            float(np.max(joined[:, 0])), float(np.max(joined[:, 1])),
        )

    @property
    def z_bounds_mm(self) -> tuple[float, float]:
        values = [array[..., 2][np.isfinite(array[..., 2])] for key, array in self.arrays.items() if _PATH_KEY.match(key)]
        joined = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
        if not joined.size:
            raise ValueError("source NPZ does not contain any path points")
        return float(np.min(joined)), float(np.max(joined))

    def to_bytes(self) -> bytes:
        """Serialise source arrays without changing padding, E values, or ordering."""

        output = BytesIO()
        arrays = dict(self.arrays)
        arrays["meta"] = np.array(json.dumps(self.meta, ensure_ascii=False))
        np.savez(output, **arrays)
        return output.getvalue()


def load_surface_target(data: bytes | str | Mapping[str, object]) -> SurfaceTarget:
    """Validate a geometry-only ``graded_surface_v1`` export."""

    if isinstance(data, bytes):
        raw: Any = json.loads(data.decode("utf-8"))
    elif isinstance(data, str):
        raw = json.loads(data)
    else:
        raw = dict(data)
    if not isinstance(raw, dict) or raw.get("format") != "graded_surface_v1":
        raise ValueError("surface config format must be graded_surface_v1")
    surface_data = _mapping(raw.get("surface"), "surface")
    if surface_data.get("type") != "double_sine_product":
        raise ValueError("only double_sine_product surface configs are currently supported")
    coordinate_system = _mapping(raw.get("coordinate_system"), "coordinate_system")
    if coordinate_system.get("plane") != "XY" or coordinate_system.get("origin") != "stl_xy_min":
        raise ValueError("surface config must use the stl_xy_min XY coordinate system")
    domain = _mapping(raw.get("domain"), "domain")
    source = _mapping(domain.get("source"), "domain.source")
    bounds = source.get("xy_bounds_mm")
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise ValueError("domain.source.xy_bounds_mm must contain four values")
    x_min, y_min, x_max, y_max = (_finite(value, "domain.source.xy_bounds_mm") for value in bounds)
    width_mm, height_mm = x_max - x_min, y_max - y_min
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("surface domain dimensions must be positive")
    return SurfaceTarget(
        surface=DoubleSineSurface(
            amplitude_mm=_finite(surface_data.get("amplitude_mm"), "surface.amplitude_mm"),
            wavelength_x_mm=_finite(surface_data.get("wavelength_x_mm"), "surface.wavelength_x_mm"),
            wavelength_y_mm=_finite(surface_data.get("wavelength_y_mm"), "surface.wavelength_y_mm"),
            phase_x_rad=_finite(surface_data.get("phase_x_rad"), "surface.phase_x_rad"),
            phase_y_rad=_finite(surface_data.get("phase_y_rad"), "surface.phase_y_rad"),
            z_reference_mm=_finite(surface_data.get("z_reference_mm"), "surface.z_reference_mm"),
        ),
        width_mm=width_mm,
        height_mm=height_mm,
        source_file_name=str(source.get("file_name", "model.stl")),
        source_sha256=str(source.get("sha256", "")),
        raw_config=raw,
    )


def read_source_npz(data: bytes | str, *, source_name: str = "flat.npz") -> SourceNPZ:
    """Read and validate the existing Core-preprocessor source contract."""

    binary = data.encode() if isinstance(data, str) else data
    try:
        with np.load(BytesIO(binary), allow_pickle=False) as archive:
            if "meta" not in archive.files:
                raise ValueError("source NPZ is missing meta")
            meta = json.loads(str(archive["meta"]))
            arrays = {key: np.array(archive[key], copy=True) for key in archive.files if key != "meta"}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read source NPZ: {exc}") from exc
    if not isinstance(meta, dict) or meta.get("format") != SOURCE_NPZ_CONTRACT_ID:
        raise ValueError(f"source NPZ format must be {SOURCE_NPZ_CONTRACT_ID}")
    semantics = meta.get("layer_semantics")
    if semantics is not None and semantics != LOGICAL_LAYER_SEMANTICS_V1:
        raise ValueError("source NPZ layer_semantics does not preserve logical layer ownership")
    source = SourceNPZ(arrays=arrays, meta=meta, source_name=source_name)
    if not source.path_keys:
        raise ValueError("source NPZ has no layer_XXXX_R/F/T path arrays")
    for key, array in arrays.items():
        if _PATH_KEY.match(key):
            _validate_path_array(key, array)
        elif _E_KEY.match(key):
            _validate_extrusion_array(key, array)
        else:
            raise ValueError(f"source NPZ contains unsupported array {key}")
    _validate_extrusion_ownership(arrays)
    return source


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _validate_path_array(key: str, array: np.ndarray) -> None:
    if array.ndim != 3 or array.shape[2] not in (3, 6):
        raise ValueError(f"{key} must have shape (path, point, 3|6)")
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{key} must use a numeric dtype")
    for path in np.asarray(array, dtype=np.float64):
        rows_are_empty = np.isnan(path).all(axis=1)
        rows_are_finite = np.isfinite(path).all(axis=1)
        if not np.all(rows_are_empty | rows_are_finite):
            raise ValueError(f"{key} may only use complete NaN padding rows")
        if np.any(rows_are_finite) and np.any(rows_are_empty[: np.flatnonzero(rows_are_finite)[-1] + 1]):
            raise ValueError(f"{key} may not have NaN padding inside a path")


def _validate_extrusion_array(key: str, array: np.ndarray) -> None:
    if array.ndim != 2 or array.dtype.kind not in "fiu":
        raise ValueError(f"{key} must have shape (path, point) and numeric dtype")
    if not np.all(np.isnan(array) | np.isfinite(array)):
        raise ValueError(f"{key} contains invalid extrusion values")


def _validate_extrusion_ownership(arrays: Mapping[str, np.ndarray]) -> None:
    """Ensure every optional E grid exactly follows its material-path grid."""

    for key, extrusion in arrays.items():
        match = _E_KEY.match(key)
        if not match:
            continue
        path_key = f"layer_{match.group(1)}_{match.group(2)}"
        path = arrays.get(path_key)
        if path is None:
            raise ValueError(f"{key} requires matching path array {path_key}")
        if extrusion.shape != path.shape[:2]:
            raise ValueError(f"{key} shape must match {path_key} path and point dimensions")
        if not np.array_equal(np.isfinite(extrusion), np.isfinite(path[..., 0])):
            raise ValueError(f"{key} finite values must match {path_key} NaN padding")
