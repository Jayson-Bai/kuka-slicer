"""Material accounting from analytic path lengths (fast UI path).

The continuous-course workflow replaces the honeycomb wall graph entirely with
analytic X courses; the retained resin material is the rectangular perimeter,
optional grip partitions, and those courses.  All of these are functions of
the design JSON and the height field, so the length-based estimate never needs
the expensive LSCM parameterization / overlap scan / rosy-6 orientation solve.

Resin volume = path length * fixed 2 mm bead width * layer height * extrusion
scale.  Fiber feed = path length * fiber extrusion scale.  These are pure
"path theory" values; the older exact estimate (which also resolves course
visit order and Core fitting) remains available as estimate_material_exact.
"""
from __future__ import annotations

import hashlib
import json
import math
import copy
import threading

import numpy as np

RESIN_FIXED_BEAD_WIDTH_MM = 2.0
RESIN_FILAMENT_DIAMETER_MM = 1.75
RESIN_FILAMENT_AREA_MM2 = math.pi * (RESIN_FILAMENT_DIAMETER_MM / 2.0) ** 2
RESIN_FILAMENT_LENGTH_PER_MM3 = 1.0 / RESIN_FILAMENT_AREA_MM2

_estimate_cache: dict[tuple, dict] = {}
_estimate_cache_lock = threading.Lock()
_ESTIMATE_CACHE_LIMIT = 128


def _polyline_length_mm(points_xyz) -> float:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts[:, :3], axis=0), axis=1).sum())


def _layer_schedule(spec, core):
    """Return (alpha_by_layer, base_z_by_layer, layer_count) without embedding."""
    from .height_planning import nearest_resin_layer_count

    layer_height = float(core.resin.layer_height_mm)
    count = nearest_resin_layer_count(float(spec.part["final_height_mm"]), layer_height)
    base_z = (np.arange(count, dtype=np.float64) + 0.5) * layer_height
    if spec.source_provider == "planar":
        return np.zeros(count, dtype=np.float64), base_z, count
    from .layer_embedding import _symmetric_alphas_at_physical_z

    embedding = spec.layer_embedding
    transition_steps = (
        int(embedding["transition_step_count"])
        if embedding.get("transition_step_policy", "auto_to_midplane") == "manual"
        else None
    )
    alpha = _symmetric_alphas_at_physical_z(
        count,
        surface_start_layer=int(embedding["surface_start_layer"]),
        transition_step_count=transition_steps,
        positions_mm=base_z,
    )
    return alpha, base_z, count


def _conformal_perimeter_length(surface, alpha, base_z, bounds, samples) -> float:
    """3-D arc length of the rectangular boundary sampled on the grid."""
    x_min, y_min, x_max, y_max = bounds
    samples_x, samples_y = samples
    total = 0.0
    for xs, ys in (
        (np.linspace(x_min, x_max, samples_x), np.full(samples_x, y_min)),
        (np.full(samples_y, x_max), np.linspace(y_min, y_max, samples_y)),
        (np.linspace(x_max, x_min, samples_x), np.full(samples_x, y_max)),
        (np.full(samples_y, x_min), np.linspace(y_max, y_min, samples_y)),
    ):
        h = np.asarray(surface.height(xs, ys), dtype=np.float64)
        z = base_z + alpha * (h - surface.z_reference_mm)
        pts = np.column_stack((xs, ys, z))
        total += _polyline_length_mm(pts)
    return total


def _planar_perimeter_length(spec) -> float:
    return 2.0 * (float(spec.part["length_mm"]) + float(spec.part["width_mm"]))


def _course_length(path_xy, surface, alpha, base_z) -> float:
    xy = np.asarray(path_xy, dtype=np.float64)
    h = np.asarray(surface.height(xy[:, 0], xy[:, 1]), dtype=np.float64)
    z = base_z + alpha * (h - surface.z_reference_mm)
    return _polyline_length_mm(np.column_stack((xy, z)))


def _fiber_interface_indices(spec, alpha) -> tuple[int, ...]:
    from ..fiber_interlayers import plan_flat_resin_interlayers
    from .fiber_reinforcement import symmetric_curvature_fiber_schedule

    if spec.source_provider == "planar":
        schedule = plan_flat_resin_interlayers(range(len(alpha)))
        return schedule.after_resin_layer_indices
    _first, _last, selected = symmetric_curvature_fiber_schedule(alpha)
    return selected


def _compute_fast_estimate(config, *, fiber_enabled, core, core_explicit):
    from ..ui_server import _core_print_params_path
    from .contracts import load_conformal_lattice_spec
    from .continuous_course import build_continuous_course_plan
    from .surface_field import height_field_from_spec

    spec = load_conformal_lattice_spec(json.dumps(config).encode("utf-8"))
    is_planar = spec.source_provider == "planar"
    surface = height_field_from_spec(spec)
    alpha, base_z, layer_count = _layer_schedule(spec, core)
    plan = build_continuous_course_plan(spec)

    fiber_indices = _fiber_interface_indices(spec, alpha)
    conformal_bounds = None
    conformal_samples = None
    if not is_planar:
        conformal_bounds = tuple(float(v) for v in spec.source_surface["double_sine"]["xy_bounds_mm"])
        conformal_samples = tuple(int(v) for v in spec.source_surface["double_sine"]["samples"])

    # Perimeter length is layer-invariant for a given alpha schedule; conformal
    # alpha varies per layer so recompute per layer below.
    course_lengths: list[float] = []
    perimeters: list[float] = []
    for index in range(layer_count):
        a = float(alpha[index])
        bz = float(base_z[index])
        course_lengths.append(sum(_course_length(xy, surface, a, bz) for xy in plan.paths_xy))
        if is_planar:
            perimeters.append(_planar_perimeter_length(spec))
        else:
            perimeters.append(
                _conformal_perimeter_length(
                    surface, a, bz,
                    conformal_bounds,
                    conformal_samples,
                )
            )

    # Grip partitions only exist for tensile specimens (bending has no grip).
    auxiliaries = [0.0] * layer_count
    from .pipeline import _layered_partition_paths, _symmetric_grip_end_length
    if _symmetric_grip_end_length(spec) > 0.0:
        shim = type(
            "LayerEmbeddingShim", (), {"report": {
                "alpha_by_layer": alpha.tolist(),
                "base_z_by_layer_mm": base_z.tolist(),
            }}
        )()
        layered = _layered_partition_paths(spec, shim)
        auxiliaries = [
            sum(_polyline_length_mm(points) for _role, points, _normals in layer)
            for layer in layered
        ]

    resin_length_mm = float(sum(perimeters) + sum(auxiliaries) + sum(course_lengths))

    valid_fiber_layers = tuple(i for i in fiber_indices if 0 <= i < layer_count)
    fiber_length_mm = 0.0
    if fiber_enabled and valid_fiber_layers:
        fiber_length_mm = float(sum(course_lengths[i] for i in valid_fiber_layers))
        if spec.lattice.get("print_outer_contour_fiber", False):
            fiber_length_mm += float(sum(perimeters[i] for i in valid_fiber_layers))

    resin = core.resin
    fiber = core.fiber
    resin_volume_mm3 = (
        resin_length_mm * RESIN_FIXED_BEAD_WIDTH_MM * float(resin.layer_height_mm) * float(resin.extrusion_scale)
    )
    resin_feed_mm = resin_volume_mm3 * RESIN_FILAMENT_LENGTH_PER_MM3
    fiber_feed_mm = fiber_length_mm * float(fiber.e_per_mm())

    return {
        "R": {
            "path_length_mm": resin_length_mm,
            "volume_mm3": resin_volume_mm3,
            "feed_mm": resin_feed_mm,
            "layer_count": layer_count,
            "path_count": 0,
        },
        "F": {
            "path_length_mm": fiber_length_mm,
            "feed_mm": fiber_feed_mm,
            "layer_count": len(valid_fiber_layers) if fiber_enabled else 0,
            "path_count": 0,
        },
        "basis": "path_theory_length_only",
        "resin_e_per_mm": float(resin.e_per_mm()),
        "fiber_e_per_mm": float(fiber.e_per_mm()),
        "resin_layer_height_mm": float(resin.layer_height_mm),
        "fiber_layer_height_mm": float(fiber.layer_height_mm),
        "resin_extrusion_scale": float(resin.extrusion_scale),
        "bead_width_mm": RESIN_FIXED_BEAD_WIDTH_MM,
        "filament_diameter_mm": RESIN_FILAMENT_DIAMETER_MM,
        "design_kind": "planar" if is_planar else "conformal",
        "design_label": "平面件" if is_planar else "曲面件",
        "core_params_source": "explicit" if core_explicit else str(_core_print_params_path()),
        "core_params_sha256": hashlib.sha256(repr(core).encode()).hexdigest(),
        "fiber_layer_indices": list(valid_fiber_layers),
        "exclusions": ["brim", "primeline", "stationary priming", "retraction", "cut waste", "course-visit-order", "core-fitting"],
    }


def estimate_material(config, *, fiber_enabled=True, core_params=None, brim_params=None, include_auxiliary=False, exact=False):
    from ..ui_server import _load_core_print_params
    core_explicit = core_params is not None
    core = core_params if core_explicit else _load_core_print_params()
    core_sha = hashlib.sha256(repr(core).encode()).hexdigest()
    cache_key = (
        json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        bool(fiber_enabled),
        core_sha,
        bool(exact),
        bool(include_auxiliary),
    )
    with _estimate_cache_lock:
        cached = _estimate_cache.get(cache_key)
    if cached is not None:
        return copy.deepcopy(cached)

    if exact:
        result = _compute_exact_estimate(
            config, fiber_enabled=fiber_enabled, core=core, core_explicit=core_explicit,
            brim_params=brim_params, include_auxiliary=include_auxiliary,
        )
    else:
        result = _compute_fast_estimate(
            config, fiber_enabled=fiber_enabled, core=core, core_explicit=core_explicit,
        )
    with _estimate_cache_lock:
        if len(_estimate_cache) >= _ESTIMATE_CACHE_LIMIT:
            _estimate_cache.clear()
        _estimate_cache[cache_key] = copy.deepcopy(result)
    return result


def _compute_exact_estimate(config, *, fiber_enabled, core, core_explicit, brim_params, include_auxiliary):
    """Authoritative-source estimate with course ordering and Core fitting.

    Kept for calibration/comparison only. The UI fast path does not call this.
    """
    from .contracts import load_conformal_lattice_spec
    from .material_source import prepare_material_source
    from ..ui_server import _core_print_params_path, _load_prusa_params
    from .brim import ConformalBrimSettings, apply_conformal_brim

    spec = load_conformal_lattice_spec(json.dumps(config).encode("utf-8"))
    is_planar = spec.source_provider == "planar"
    job, run, _, _ = prepare_material_source(
        spec, core, {"conformal_fiber_enabled": [str(fiber_enabled).lower()]},
    )
    coast_distance_mm = float(getattr(core.resin, "coast_distance_mm", 0.0))
    result = _measure_source(job, coast_distance_mm)
    if include_auxiliary:
        saved = _load_prusa_params() if brim_params is None else brim_params
        brim = apply_conformal_brim(job, ConformalBrimSettings(
            enabled=saved.get("prusa_brim_enabled", False) is True,
            width_mm=float(saved.get("prusa_brim_width", 5.0)),
            brim_type=str(saved.get("prusa_brim_type", "outer_only")),
            separation_mm=float(saved.get("prusa_brim_separation", 0.0)),
            one_stroke=saved.get("prusa_brim_one_stroke", False) is True,
            line_width_mm=2.0,
        ))
        with_brim = _measure_source(job, coast_distance_mm)
        brim_volume = with_brim["R"]["volume_mm3"] - result["R"]["volume_mm3"]
        primeline_volume = (core.primeline_length_mm * core.resin.e_per_mm() * RESIN_FILAMENT_AREA_MM2
                           if core.primeline_enabled else 0.0)
        result["auxiliary"] = dict(brim_volume_mm3=brim_volume,
                                   primeline_volume_mm3=primeline_volume, brim=brim)
        result["total_deposition_resin_volume_mm3"] = with_brim["R"]["volume_mm3"] + primeline_volume
    result.update(
        basis="authoritative_source_paths_before_core_fitting",
        core_params_source="explicit" if core_explicit else str(_core_print_params_path()),
        core_params_sha256=hashlib.sha256(repr(core).encode()).hexdigest(),
        design_kind="planar" if is_planar else "conformal",
        design_label="平面件" if is_planar else "曲面件",
        resin_e_per_mm=core.resin.e_per_mm(),
        fiber_e_per_mm=core.fiber.e_per_mm(),
        resin_coast_distance_mm=coast_distance_mm,
        filament_diameter_mm=1.75,
        physical_stack_plan=run.layer_embedding.report.get("physical_stack_plan", {}),
        exclusions=["brim", "primeline", "stationary priming", "retraction", "cut waste"],
    )
    return result


def _measure_source(job, coast_distance_mm=0.0):
    coast = float(coast_distance_mm)
    if not math.isfinite(coast) or coast < 0:
        raise ValueError("coast distance must be finite and nonnegative")
    totals = {}
    for material in ("R", "F"):
        length = feed = removed = 0.0
        count = 0
        layers = set()
        for group in job.material_paths:
            if group.material != material:
                continue
            if group.extrusion is None or len(group.extrusion) != len(group.paths):
                raise ValueError("material accounting requires authoritative E profiles")
            for path, profile in zip(group.paths, group.extrusion):
                xyz = np.asarray(path, dtype=float)[:, :3]
                e = np.asarray(profile, dtype=float)
                if len(xyz) != len(e) or not np.isfinite(xyz).all() or not np.isfinite(e).all():
                    raise ValueError("invalid source path/E profile")
                if len(e) < 2:
                    continue
                if np.any(np.diff(e) < -1e-7):
                    raise ValueError("source deposition E must be monotonic")
                distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
                path_length = float(distance[-1])
                length += path_length
                feed += float(e[-1] - e[0])
                if material == "R" and coast > 0 and path_length > 2 * coast:
                    removed += float(e[-1] - np.interp(path_length - coast, distance, e))
                count += 1
                layers.add(group.layer_index)
        totals[material] = dict(path_length_mm=length, feed_mm=feed,
                                source_coasting_feed_mm=feed - removed,
                                coasting_removed_feed_mm=removed,
                                path_count=count, layer_count=len(layers))
    totals["R"]["volume_mm3"] = totals["R"]["feed_mm"] * RESIN_FILAMENT_AREA_MM2
    totals["R"]["source_coasting_volume_mm3"] = totals["R"]["source_coasting_feed_mm"] * RESIN_FILAMENT_AREA_MM2
    return totals
