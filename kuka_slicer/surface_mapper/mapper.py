"""Pure Z-only mapping from a planar source NPZ to a graded surface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Literal

import numpy as np

from .contracts import SourceNPZ, SurfaceTarget, _PATH_KEY
from .progression import LayerProgression, ProgressionCurve


OffsetMode = Literal["auto", "manual"]


@dataclass(frozen=True, slots=True)
class SurfaceMappingPlan:
    """Mapper-owned process choices, separate from the target geometry JSON."""

    progression: LayerProgression
    offset_mode: OffsetMode = "auto"
    z_offset_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.offset_mode not in ("auto", "manual"):
            raise ValueError("offset_mode must be auto or manual")
        if not np.isfinite(self.z_offset_mm):
            raise ValueError("z_offset_mm must be finite")

    @classmethod
    def default_for(cls, source: SourceNPZ, *, curve: ProgressionCurve = "smoothstep") -> "SurfaceMappingPlan":
        layers = source.layer_indices
        return cls(LayerProgression(layers[0], layers[-1], curve=curve))


@dataclass(frozen=True, slots=True)
class MappingResult:
    source: SourceNPZ
    applied_z_offset_mm: float
    alpha_by_layer: dict[int, float]
    source_z_bounds_mm: tuple[float, float]
    mapped_z_bounds_mm: tuple[float, float]
    xy_bounds_mm: tuple[float, float, float, float]


def map_source_job(source: SourceNPZ, target: SurfaceTarget, plan: SurfaceMappingPlan) -> MappingResult:
    """Map all R/F/T path points in Z while preserving their logical key and order."""

    _validate_domain(source, target)
    alpha_by_layer = {layer: plan.progression.alpha(layer) for layer in source.layer_indices}
    source_z_bounds = source.z_bounds_mm
    offset = _resolve_offset(source, target, alpha_by_layer, plan)
    arrays = {key: value.copy() for key, value in source.arrays.items()}
    for key, array in arrays.items():
        match = _PATH_KEY.match(key)
        if not match:
            continue
        alpha = alpha_by_layer[int(match.group(1))]
        valid = np.isfinite(array[..., 0])
        if not np.any(valid):
            continue
        mapped_height = target.surface.height(array[..., 0][valid], array[..., 1][valid])
        array[..., 2][valid] += alpha * mapped_height + offset
    meta = dict(source.meta)
    meta["surface_mapping"] = {
        "format": "surface_mapping_v1",
        "formula": "z_mapped = z_flat + alpha(logical_layer) * H(x,y) + z_offset_mm",
        "target_surface_format": "graded_surface_v1",
        "target_surface_sha256": _target_digest(target),
        "target_source_file_name": target.source_file_name,
        "target_source_sha256": target.source_sha256,
        "progression": {
            "basis": "logical_layer_index",
            "curve": plan.progression.curve,
            "start_logical_layer": plan.progression.start_logical_layer,
            "end_logical_layer": plan.progression.end_logical_layer,
            "alpha_by_layer": {str(key): value for key, value in alpha_by_layer.items()},
        },
        "z_offset_mm": offset,
        "xy": "preserved",
        "extrusion": "preserved_unrecalculated",
        "orientation": "preserved_unrecalculated",
    }
    mapped = SourceNPZ(arrays=arrays, meta=meta, source_name=source.source_name)
    return MappingResult(
        source=mapped,
        applied_z_offset_mm=offset,
        alpha_by_layer=alpha_by_layer,
        source_z_bounds_mm=source_z_bounds,
        mapped_z_bounds_mm=mapped.z_bounds_mm,
        xy_bounds_mm=mapped.xy_bounds_mm,
    )


def _resolve_offset(
    source: SourceNPZ,
    target: SurfaceTarget,
    alpha_by_layer: dict[int, float],
    plan: SurfaceMappingPlan,
) -> float:
    if plan.offset_mode == "manual":
        return float(plan.z_offset_mm)
    candidate_min = float("inf")
    for key, array in source.arrays.items():
        match = _PATH_KEY.match(key)
        if not match:
            continue
        valid = np.isfinite(array[..., 0])
        if not np.any(valid):
            continue
        alpha = alpha_by_layer[int(match.group(1))]
        values = array[..., 2][valid] + alpha * target.surface.height(array[..., 0][valid], array[..., 1][valid])
        candidate_min = min(candidate_min, float(np.min(values)))
    return max(0.0, source.z_bounds_mm[0] - candidate_min)


def _validate_domain(source: SourceNPZ, target: SurfaceTarget) -> None:
    x_min, y_min, x_max, y_max = source.xy_bounds_mm
    tolerance = 1e-6
    if x_min < -tolerance or y_min < -tolerance or x_max > target.width_mm + tolerance or y_max > target.height_mm + tolerance:
        raise ValueError(
            "source NPZ XY bounds do not fit inside the target surface STL projection "
            f"(source={source.xy_bounds_mm}, target=(0, 0, {target.width_mm}, {target.height_mm}))"
        )
    source_model = source.meta.get("source_model")
    if isinstance(source_model, dict) and source_model.get("sha256") and target.source_sha256:
        if source_model["sha256"] != target.source_sha256:
            raise ValueError("source NPZ was sliced from a different STL than the target surface config")


def _target_digest(target: SurfaceTarget) -> str:
    payload = json.dumps(target.raw_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
