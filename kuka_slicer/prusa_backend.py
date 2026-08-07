"""Full PrusaSlicer FFF path backend.

The native extension owns mesh slicing, perimeter offsets, infill, path order
and avoid-crossing-perimeters travel.  This module only adapts that motion to
the project's stable ExternalSourceJob representation.
"""

from __future__ import annotations

import numpy as np

from .external_npz import ExternalSourceJob, MaterialPaths, TravelPaths
from .prusa_bridge import require_native
from .stl_io import Mesh
from .slicer import (
    DEFAULT_MATERIAL_PROCESS,
    SliceConfig,
    _connect_brim_paths_one_stroke,
    orient_mesh_for_build_axis,
)


_PRUSA_PATTERN_NAMES = {
    "none": "rectilinear",
    "line": "line",
    "rectilinear": "rectilinear",
    "aligned_rectilinear": "alignedrectilinear",
    "grid": "grid",
    "triangles": "triangles",
    "gyroid": "gyroid",
    "concentric": "concentric",
    # The remaining choices are project names for angle schedules. Prusa owns
    # the detailed alternating-angle policy for its rectilinear fill.
    "zigzag": "rectilinear",
    "isotropic": "rectilinear",
    "zigzag_horizontal": "rectilinear",
    "zigzag_vertical": "rectilinear",
    "zigzag_plus45": "rectilinear",
    "zigzag_minus45": "rectilinear",
}

# These are actual deposited directions, not Prusa's base angles. The native
# bridge compensates for Prusa's own 90-degree odd-layer alternation.
_PRUSA_FILL_ANGLE_SCHEDULES = {
    "zigzag_horizontal": [0.0],
    "zigzag_vertical": [90.0],
    "zigzag_plus45": [45.0],
    "zigzag_minus45": [-45.0],
    "isotropic": [45.0, 0.0, -45.0, 90.0],
}


def slice_mesh_to_job_with_prusa(mesh: Mesh, config: SliceConfig) -> ExternalSourceJob:
    """Return complete Prusa material and travel paths for a resin part.

    G-code extrusion is represented as a cumulative E value per deposited XYZ
    point; empty moves are kept separately so old XYZ-only consumers remain
    compatible.
    """

    if config.material != "R":
        raise ValueError("Prusa slicing kernel currently supports resin material R only")
    if config.curve_mode != "flat":
        raise ValueError("Prusa slicing kernel currently supports flat Z layers only")
    pattern = _PRUSA_PATTERN_NAMES.get(config.infill_pattern)
    if pattern is None:
        raise ValueError(f"unsupported Prusa infill pattern: {config.infill_pattern!r}")
    fill_angle_schedule = _PRUSA_FILL_ANGLE_SCHEDULES.get(
        config.infill_pattern,
        [],
    )

    oriented_mesh = orient_mesh_for_build_axis(mesh, config.build_axis)
    triangles = np.asarray(oriented_mesh.triangles, dtype=np.float32)
    if triangles.size == 0:
        return ExternalSourceJob(material_paths=[], meta=_prusa_metadata(config, []))

    minimum = np.min(triangles.reshape(-1, 3), axis=0)
    # A temporary placement on the default 200 mm Prusa bed makes the FFF
    # planner robust to user meshes centred around a negative XY origin. The
    # translation is reversed in the returned paths before normal export.
    placement = np.array([10.0 - minimum[0], 10.0 - minimum[1], -minimum[2]], dtype=np.float64)
    placed = triangles + placement
    vertices = placed.reshape(-1, 3)
    faces = np.arange(vertices.shape[0], dtype=np.int32).reshape(-1, 3)

    native = require_native()
    result = native.slice_print_paths(
        vertices,
        faces,
        layer_height=float(config.layer_height),
        first_layer_height=float(config.first_layer_height),
        line_width=float(config.line_width),
        perimeter_count=int(config.perimeter_count if config.print_perimeters else 0),
        infill_density=float(config.infill_density if config.infill_pattern != "none" else 0.0),
        infill_pattern=pattern,
        fill_angle_schedule=fill_angle_schedule,
        perimeter_infill_overlap=float(config.contour_infill_overlap),
        raft_layers=int(config.prusa_raft.layer_count),
        raft_expansion=float(config.prusa_raft.expansion),
        raft_first_layer_density=float(config.prusa_raft.first_layer_density),
        raft_first_layer_expansion=float(config.prusa_raft.first_layer_expansion),
        raft_contact_distance=float(config.prusa_raft.contact_distance),
        raft_contact_layer_height=(
            0.0 if config.prusa_raft.contact_auto else float(config.prusa_raft.contact_layer_height)
        ),
        raft_contact_density=(
            0.0 if config.prusa_raft.contact_auto else float(config.prusa_raft.contact_density)
        ),
        raft_contact_extrusion_width=(
            0.0 if config.prusa_raft.contact_auto else float(config.prusa_raft.contact_extrusion_width)
        ),
        brim_enabled=bool(config.brim_enabled),
        brim_width=float(config.brim_width_mm),
        brim_type=config.brim_type,
        brim_separation=float(config.brim_separation_mm),
        perimeter_generator=config.prusa_geometry.perimeter_generator,
        gap_fill_enabled=config.prusa_geometry.gap_fill_enabled,
        infill_anchor=config.prusa_geometry.infill_anchor,
        infill_anchor_max=config.prusa_geometry.infill_anchor_max,
        external_perimeter_width=config.prusa_geometry.external_perimeter_width,
        perimeter_width=config.prusa_geometry.perimeter_width,
        infill_width=config.prusa_geometry.infill_width,
        xy_size_compensation=float(config.prusa_geometry.xy_size_compensation),
        elephant_foot_compensation=float(config.prusa_geometry.elephant_foot_compensation),
        avoid_crossing_max_detour=float(config.prusa_geometry.avoid_crossing_max_detour),
        seam_position=config.prusa_geometry.seam_position,
    )

    material_paths: list[MaterialPaths] = []
    travel_paths: list[TravelPaths] = []
    path_roles: dict[str, dict[str, list[str]]] = {"R": {}}
    motion_order: dict[str, list[dict[str, object]]] = {}
    source_layers = result.get("layers", [])
    for source_layer in source_layers:
        z = float(source_layer["z"]) - float(placement[2])
        if config.z_min is not None and z < config.z_min - config.tolerance:
            continue
        if config.z_max is not None and z > config.z_max + config.tolerance:
            continue

        layer_index = len(material_paths)
        paths = _restore_paths(source_layer.get("paths", []), placement)
        extrusion = [np.asarray(values, dtype=np.float64) for values in source_layer.get("extrusion", [])]
        roles = [str(role) for role in source_layer.get("roles", [])]
        travel = _restore_paths(source_layer.get("travel", []), placement)
        native_motions = source_layer.get("motions", [])
        ordered_motions: list[dict[str, object]] = []
        if isinstance(native_motions, list):
            for motion in native_motions:
                if not isinstance(motion, dict):
                    continue
                kind = motion.get("kind")
                index = motion.get("index")
                if (
                    kind == "deposit"
                    and isinstance(index, int)
                    and 0 <= index < len(paths)
                ) or (
                    kind == "travel"
                    and isinstance(index, int)
                    and 0 <= index < len(travel)
                ):
                    ordered_motions.append({"kind": kind, "index": index})
        if not ordered_motions:
            ordered_motions.extend(
                {"kind": "deposit", "index": index}
                for index in range(len(paths))
            )
            ordered_motions.extend(
                {"kind": "travel", "index": index}
                for index in range(len(travel))
            )

        if paths:
            if len(paths) != len(extrusion) or len(paths) != len(roles):
                raise RuntimeError("Prusa native bridge returned inconsistent deposited path data")
            if config.brim_one_stroke:
                paths, extrusion, roles, travel, ordered_motions = _apply_brim_one_stroke(
                    paths,
                    extrusion,
                    roles,
                    travel,
                    ordered_motions,
                    line_width=float(config.line_width),
                    tolerance=float(config.tolerance),
                )
            material_paths.append(MaterialPaths(layer_index, "R", paths, extrusion))
            path_roles["R"][str(layer_index)] = roles

        if travel:
            travel_paths.append(TravelPaths(layer_index, travel))
        if ordered_motions:
            motion_order[str(layer_index)] = ordered_motions

    return ExternalSourceJob(
        material_paths=material_paths,
        travel_paths=travel_paths,
        meta=_prusa_metadata(
            config,
            source_layers,
            path_roles,
            fill_angle_schedule,
            motion_order,
        ),
        native_gcode=(
            result["gcode"] if isinstance(result.get("gcode"), str) else None
        ),
        native_gcode_translation_mm=tuple(float(value) for value in -placement),
    )


def _restore_paths(raw_paths: object, placement: np.ndarray) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    for raw_path in raw_paths if isinstance(raw_paths, list) else []:
        path = np.asarray(raw_path, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
            continue
        path = path.copy()
        path -= placement
        paths.append(path)
    return paths


def _apply_brim_one_stroke(
    paths: list[np.ndarray],
    extrusion: list[np.ndarray],
    roles: list[str],
    travel: list[np.ndarray],
    motions: list[dict[str, object]],
    *,
    line_width: float,
    tolerance: float,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[str],
    list[np.ndarray],
    list[dict[str, object]],
]:
    """Apply the existing safe boundary connector to one Brim layer.

    The Prusa paths and their cumulative E values remain the source data.  A
    connector is retained only when it is a real deposited segment inside the
    Brim corridor.  Each geometrically connected Brim component becomes one
    deposited path; genuinely separate islands retain their own source travel.
    """

    brim_indices = [index for index, role in enumerate(roles) if role == "brim"]
    if len(brim_indices) < 2:
        return paths, extrusion, roles, travel, motions

    connected = _connect_brim_paths_one_stroke(
        [paths[index] for index in brim_indices],
        line_width,
        max(tolerance, 1e-6),
    )
    if not connected:
        return paths, extrusion, roles, travel, motions

    # The connector returns one chain per geometrically connected Brim
    # component.  Match each returned chain back to its source paths by the
    # exact source endpoints so independent islands remain independent while
    # the main component is still collapsed into one deposited path.
    endpoint_tolerance = max(tolerance * 50.0, 1e-4)
    remaining = set(brim_indices)
    components: list[tuple[list[int], np.ndarray]] = []
    for merged_xy in connected:
        merged_xy = np.asarray(merged_xy, dtype=np.float64)
        if merged_xy.ndim != 2 or merged_xy.shape[1] < 2:
            return paths, extrusion, roles, travel, motions
        merged_xy = merged_xy[:, :2]
        matched: list[int] = []
        for index in brim_indices:
            if index not in remaining:
                continue
            source = np.asarray(paths[index][:, :2], dtype=np.float64)
            start_distance = float(
                np.linalg.norm(merged_xy - source[0], axis=1).min()
            )
            end_distance = float(
                np.linalg.norm(merged_xy - source[-1], axis=1).min()
            )
            if min(start_distance, end_distance) <= endpoint_tolerance:
                matched.append(index)
        if matched:
            matched.sort()
            remaining.difference_update(matched)
            components.append((matched, merged_xy))
    if remaining:
        # A provenance mismatch is safer than silently dropping a source
        # stroke or its E profile.
        return paths, extrusion, roles, travel, motions

    merged_paths: dict[int, np.ndarray] = {}
    merged_extrusion: dict[int, np.ndarray] = {}
    component_by_path: dict[int, int] = {}
    for source_indices, merged_xy in components:
        representative = source_indices[0]
        component_by_path.update({index: representative for index in source_indices})
        if len(source_indices) == 1:
            merged_paths[representative] = paths[representative]
            merged_extrusion[representative] = extrusion[representative]
            continue

        rates: list[float] = []
        for index in source_indices:
            path = np.asarray(paths[index], dtype=np.float64)
            values = np.asarray(extrusion[index], dtype=np.float64)
            if values.shape[0] != path.shape[0] or values.shape[0] < 2:
                return paths, extrusion, roles, travel, motions
            length = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
            delta_e = float(values[-1] - values[0])
            if length > max(tolerance, 1e-9) and delta_e >= 0.0:
                rates.append(delta_e / length)
        if not rates:
            return paths, extrusion, roles, travel, motions

        # All Brim paths use the same native flow.  The median is robust to a
        # short clipped path at a corner and preserves the native path density.
        rate = float(np.median(np.asarray(rates, dtype=np.float64)))
        first_path = paths[representative]
        z_value = float(first_path[0, 2])
        merged = np.column_stack(
            (merged_xy[:, 0], merged_xy[:, 1], np.full(merged_xy.shape[0], z_value))
        )
        start_e = float(extrusion[representative][0])
        distances = np.linalg.norm(np.diff(merged[:, :2], axis=0), axis=1)
        merged_e = start_e + np.concatenate(([0.0], np.cumsum(distances * rate)))
        merged_paths[representative] = merged
        merged_extrusion[representative] = merged_e

    brim_set = set(brim_indices)

    kept_paths: list[np.ndarray] = []
    kept_extrusion: list[np.ndarray] = []
    kept_roles: list[str] = []
    old_to_new: dict[int, int] = {}
    for old_index, (path, values, role) in enumerate(zip(paths, extrusion, roles)):
        if old_index in brim_set:
            representative = component_by_path[old_index]
            if old_index != representative:
                continue
            path = merged_paths[representative]
            values = merged_extrusion[representative]
            role = "brim"
        old_to_new[old_index] = len(kept_paths)
        kept_paths.append(path)
        kept_extrusion.append(values)
        kept_roles.append(role)

    # Remove only the non-depositing moves between Brim strokes.  The single
    # merged deposited path replaces them with actual material motion.
    removed_travel: set[int] = set()
    last_deposit: int | None = None
    pending_travel: list[int] = []
    for motion in motions:
        kind = motion.get("kind")
        index = motion.get("index")
        if not isinstance(index, int):
            continue
        if kind == "travel":
            pending_travel.append(index)
        elif kind == "deposit":
            if (
                last_deposit in brim_set
                and index in brim_set
                and component_by_path.get(last_deposit) == component_by_path.get(index)
            ):
                removed_travel.update(pending_travel)
            pending_travel = []
            last_deposit = index

    kept_travel: list[np.ndarray] = []
    travel_to_new: dict[int, int] = {}
    for old_index, path in enumerate(travel):
        if old_index in removed_travel:
            continue
        travel_to_new[old_index] = len(kept_travel)
        kept_travel.append(path)

    kept_motions: list[dict[str, object]] = []
    emitted_components: set[int] = set()
    for motion in motions:
        kind = motion.get("kind")
        index = motion.get("index")
        if not isinstance(index, int):
            continue
        if kind == "deposit":
            if index in brim_set:
                representative = component_by_path[index]
                if representative in emitted_components:
                    continue
                emitted_components.add(representative)
                index = representative
            if index not in old_to_new:
                continue
            kept_motions.append({"kind": "deposit", "index": old_to_new[index]})
        elif kind == "travel":
            if index in removed_travel or index not in travel_to_new:
                continue
            kept_motions.append({"kind": "travel", "index": travel_to_new[index]})

    return kept_paths, kept_extrusion, kept_roles, kept_travel, kept_motions


def _prusa_metadata(
    config: SliceConfig,
    source_layers: list[object],
    path_roles: dict[str, dict[str, list[str]]] | None = None,
    fill_angle_schedule: list[float] | None = None,
    motion_order: dict[str, list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    return {
        "source": "kuka_slicer",
        "slicing": {
            "layer_height": config.layer_height,
            "first_layer_height": config.first_layer_height,
            "line_width": config.line_width,
            "planning_line_width": config.planning_line_width,
            "planning_line_width_applied": False,
            "infill_pattern": config.infill_pattern,
            "infill_density": config.infill_density,
            "infill_overlap": config.infill_overlap,
            "contour_infill_overlap": config.contour_infill_overlap,
            "prusa_perimeter_infill_overlap": config.contour_infill_overlap,
            "prusa_fill_angle_schedule": fill_angle_schedule or None,
            "build_axis": config.build_axis,
            "z_min": config.z_min,
            "z_max": config.z_max,
            "start_x_mm": config.start_x_mm,
            "start_y_mm": config.start_y_mm,
            "slicing_kernel": "prusa",
            "slicing_kernel_status": "native_full_path",
            "perimeter_count": config.perimeter_count,
            "print_perimeters": config.print_perimeters,
            "contour_source": "prusa",
            "path_planner": "prusa_fff",
            "travel_source": "prusa_avoid_crossing_perimeters",
            "source_layer_count": len(source_layers),
            "prusa_raft": config.prusa_raft.to_metadata(),
            "prusa_brim": {
                "enabled": config.brim_enabled,
                "width": config.brim_width_mm,
                "type": config.brim_type,
                "separation": config.brim_separation_mm,
                "one_stroke": config.brim_one_stroke,
            },
            "prusa_geometry": config.prusa_geometry.to_metadata(),
        },
        "path_roles": path_roles if path_roles is not None else {"R": {}},
        "motion_order": motion_order if motion_order is not None else {},
        "process_defaults": {
            "resin": DEFAULT_MATERIAL_PROCESS["R"],
            "fiber": DEFAULT_MATERIAL_PROCESS["F"],
        },
    }
