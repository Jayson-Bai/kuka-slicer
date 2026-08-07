"""Pure Z-only mapping from a planar source NPZ to a graded surface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .contracts import SourceNPZ, SurfaceTarget, _PATH_KEY
from .progression import LayerProgression, ProgressionCurve


@dataclass(frozen=True, slots=True)
class SurfaceMappingPlan:
    """Mapper-owned process choices, separate from the target geometry JSON."""

    progression: LayerProgression

    @classmethod
    def default_for(cls, source: SourceNPZ, *, curve: ProgressionCurve = "smoothstep") -> "SurfaceMappingPlan":
        layers = source.layer_indices
        return cls(LayerProgression(layers[0], layers[-1], curve=curve))


@dataclass(frozen=True, slots=True)
class MappingResult:
    source: SourceNPZ
    alpha_by_layer: dict[int, float]
    source_z_bounds_mm: tuple[float, float]
    mapped_z_bounds_mm: tuple[float, float]
    xy_bounds_mm: tuple[float, float, float, float]


def map_source_job(source: SourceNPZ, target: SurfaceTarget, plan: SurfaceMappingPlan) -> MappingResult:
    """Map all R/F/T path points in Z while preserving their logical key and order."""

    _validate_domain(source, target)
    alpha_by_layer = {layer: plan.progression.alpha(layer) for layer in source.layer_indices}
    source_z_bounds = source.z_bounds_mm
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
        array[..., 2][valid] += alpha * mapped_height
    mapped = SourceNPZ(arrays=arrays, meta=dict(source.meta), source_name=source.source_name)
    mapped_z_bounds = mapped.z_bounds_mm
    if mapped_z_bounds[0] < 0.0:
        raise ValueError(
            "surface mapping would produce negative Z "
            f"(minimum {mapped_z_bounds[0]:.3f} mm); adjust the curve or its start/completion layers"
        )
    meta = dict(source.meta)
    meta["surface_mapping"] = {
        "format": "surface_mapping_v1",
        "formula": "z_mapped = z_flat + alpha(logical_layer) * H(x,y)",
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
        "z_validation": "all mapped points must be greater than or equal to 0 mm",
        "xy": "preserved",
        "extrusion": "preserved_unrecalculated",
        "orientation": "preserved_unrecalculated",
    }
    mapped = SourceNPZ(arrays=arrays, meta=meta, source_name=source.source_name)
    return MappingResult(
        source=mapped,
        alpha_by_layer=alpha_by_layer,
        source_z_bounds_mm=source_z_bounds,
        mapped_z_bounds_mm=mapped_z_bounds,
        xy_bounds_mm=mapped.xy_bounds_mm,
    )


def _validate_domain(source: SourceNPZ, target: SurfaceTarget) -> None:
    x_min, y_min, x_max, y_max = source.xy_bounds_mm
    tolerance = 1e-6
    source_model = source.meta.get("source_model")
    source_hash = source_model.get("sha256") if isinstance(source_model, dict) else None
    target_hash = target.source_sha256 or None
    if source_hash and target_hash and source_hash != target_hash:
        raise ValueError("source NPZ was sliced from a different STL than the target surface config")
    # Native Prusa output can intentionally include a skirt, brim, or startup
    # path outside the part projection.  A matching source-model hash is a
    # stronger identity check than those process-path bounds, so keep those
    # paths valid and map them with the same analytical H(x, y) field.
    if source_hash and target_hash and source_hash == target_hash:
        return
    if x_min < -tolerance or y_min < -tolerance or x_max > target.width_mm + tolerance or y_max > target.height_mm + tolerance:
        raise ValueError(
            "source NPZ XY bounds do not fit inside the target surface STL projection "
            f"(source={source.xy_bounds_mm}, target=(0, 0, {target.width_mm}, {target.height_mm}))"
        )


def _target_digest(target: SurfaceTarget) -> str:
    payload = json.dumps(target.raw_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
