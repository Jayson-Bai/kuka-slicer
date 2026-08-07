"""Map planar source paths onto a graded surface with KUKA tool orientation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

from .contracts import SourceNPZ, SurfaceTarget, _PATH_KEY
from .orientation import kuka_abc_for_surface
from .progression import LayerProgression
from .sampling import (
    SurfaceSamplingConfig,
    SurfaceSamplingResult,
    resample_material_paths,
)


@dataclass(frozen=True, slots=True)
class SurfaceMappingPlan:
    """Mapper-owned process choices, separate from the target geometry JSON."""

    progression: LayerProgression
    sampling: SurfaceSamplingConfig = field(default_factory=SurfaceSamplingConfig)

    @classmethod
    def default_for(cls, source: SourceNPZ) -> "SurfaceMappingPlan":
        layers = source.layer_indices
        return cls(LayerProgression(layers[0], layers[-1]))


@dataclass(frozen=True, slots=True)
class MappingResult:
    source: SourceNPZ
    flat_source: SourceNPZ
    alpha_by_layer: dict[int, float]
    source_z_bounds_mm: tuple[float, float]
    mapped_z_bounds_mm: tuple[float, float]
    xy_bounds_mm: tuple[float, float, float, float]
    sampling: SurfaceSamplingResult


def map_source_job(source: SourceNPZ, target: SurfaceTarget, plan: SurfaceMappingPlan) -> MappingResult:
    """Resample deposited paths, then map all R/F/T paths to XYZABC."""

    _validate_domain(source, target)
    sampling = resample_material_paths(source, plan.sampling)
    flat_source = sampling.source
    alpha_by_layer = {layer: plan.progression.alpha(layer) for layer in source.layer_indices}
    source_z_bounds = source.z_bounds_mm
    arrays = {key: value.copy() for key, value in flat_source.arrays.items()}
    for key, original in tuple(arrays.items()):
        match = _PATH_KEY.match(key)
        if not match:
            continue
        array = _with_kuka_orientation_columns(original)
        arrays[key] = array
        alpha = alpha_by_layer[int(match.group(1))]
        valid = np.isfinite(array[..., 0])
        if not np.any(valid):
            continue
        x = array[..., 0][valid]
        y = array[..., 1][valid]
        mapped_height = target.surface.height(x, y)
        array[..., 2][valid] += alpha * mapped_height
        dz_dx, dz_dy = _surface_gradient(target, x, y, alpha)
        a, b, c = kuka_abc_for_surface(dz_dx, dz_dy)
        array[..., 3][valid] = a
        array[..., 4][valid] = b
        array[..., 5][valid] = c
    mapped = SourceNPZ(arrays=arrays, meta=dict(source.meta), source_name=source.source_name)
    mapped_z_bounds = mapped.z_bounds_mm
    if mapped_z_bounds[0] < 0.0:
        raise ValueError(
            "surface mapping would produce negative Z "
            f"(minimum {mapped_z_bounds[0]:.3f} mm); adjust the curve or its surface start layer"
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
            "mode": "symmetric_flat_curved_flat",
            "surface_start_layer": plan.progression.surface_start_layer,
            "surface_return_layer": plan.progression.surface_return_layer,
            "peak_layers": list(plan.progression.peak_layers),
            "alpha_by_layer": {str(key): value for key, value in alpha_by_layer.items()},
        },
        "z_validation": "all mapped points must be greater than or equal to 0 mm",
        "xy": "R/F paths resampled in planar XY before analytical surface mapping; T paths preserved",
        "sampling": {
            "format": "surface_material_resampling_v1",
            "scope": "R/F only; T paths are preserved",
            "max_segment_length_mm": plan.sampling.max_segment_length_mm,
            "material_point_count_before": sampling.material_point_count_before,
            "material_point_count_after": sampling.material_point_count_after,
            "resampled_path_count": sampling.resampled_path_count,
            "new_points": "XYZ is sampled on the planar source segment; Z and ABC are recomputed from the analytical surface",
        },
        "extrusion": "preserved_unrecalculated",
        "orientation": {
            "mode": "surface_normal_kuka_zyx",
            "kuka_abc_order": "A=Z,B=Y,C=X",
            "frame": "relative_to_calibrated_flat_printing_pose",
            "tool_work_axis": "+X_TOOL",
            "tool_work_axis_direction": "opposite_upward_surface_normal",
            "roll_reference": "projected_workpiece_+Y_minimum_twist",
            "flat_reference": "+X_TOOL=-Z_BASE,+Y_TOOL=+Y_BASE,+Z_TOOL=+X_BASE",
        },
    }
    mapped = SourceNPZ(arrays=arrays, meta=meta, source_name=source.source_name)
    return MappingResult(
        source=mapped,
        flat_source=flat_source,
        alpha_by_layer=alpha_by_layer,
        source_z_bounds_mm=source_z_bounds,
        mapped_z_bounds_mm=mapped_z_bounds,
        xy_bounds_mm=mapped.xy_bounds_mm,
        sampling=sampling,
    )


def _with_kuka_orientation_columns(array: np.ndarray) -> np.ndarray:
    """Preserve padding and XYZ while normalising mapped path arrays to XYZABC."""

    if array.shape[-1] == 6:
        return np.asarray(array, dtype=np.float64).copy()
    expanded = np.full((*array.shape[:2], 6), np.nan, dtype=np.float64)
    expanded[..., :3] = array
    return expanded


def _surface_gradient(
    target: SurfaceTarget, x: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    surface = target.surface
    x_phase = (2.0 * np.pi / surface.wavelength_x_mm) * x + surface.phase_x_rad
    y_phase = (2.0 * np.pi / surface.wavelength_y_mm) * y + surface.phase_y_rad
    dz_dx = alpha * surface.amplitude_mm * (2.0 * np.pi / surface.wavelength_x_mm) * np.cos(x_phase) * np.sin(y_phase)
    dz_dy = alpha * surface.amplitude_mm * (2.0 * np.pi / surface.wavelength_y_mm) * np.sin(x_phase) * np.cos(y_phase)
    return dz_dx, dz_dy


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
