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
from .slicer import DEFAULT_MATERIAL_PROCESS, SliceConfig, orient_mesh_for_build_axis


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
    placement = np.array([10.0 - minimum[0], 10.0 - minimum[1], -minimum[2]], dtype=np.float32)
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
        if paths:
            if len(paths) != len(extrusion) or len(paths) != len(roles):
                raise RuntimeError("Prusa native bridge returned inconsistent deposited path data")
            material_paths.append(MaterialPaths(layer_index, "R", paths, extrusion))
            path_roles["R"][str(layer_index)] = roles

        travel = _restore_paths(source_layer.get("travel", []), placement)
        if travel:
            travel_paths.append(TravelPaths(layer_index, travel))
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
    )


def _restore_paths(raw_paths: object, placement: np.ndarray) -> list[np.ndarray]:
    paths: list[np.ndarray] = []
    for raw_path in raw_paths if isinstance(raw_paths, list) else []:
        path = np.asarray(raw_path, dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
            continue
        path = path.copy()
        path -= placement
        paths.append(path)
    return paths


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
            "slicing_kernel": "prusa",
            "slicing_kernel_status": "native_full_path",
            "perimeter_count": config.perimeter_count,
            "print_perimeters": config.print_perimeters,
            "contour_source": "prusa",
            "path_planner": "prusa_fff",
            "travel_source": "prusa_avoid_crossing_perimeters",
            "source_layer_count": len(source_layers),
            "prusa_raft": config.prusa_raft.to_metadata(),
            "prusa_geometry": config.prusa_geometry.to_metadata(),
        },
        "path_roles": path_roles if path_roles is not None else {"R": {}},
        "motion_order": motion_order if motion_order is not None else {},
        "process_defaults": {
            "resin": DEFAULT_MATERIAL_PROCESS["R"],
            "fiber": DEFAULT_MATERIAL_PROCESS["F"],
        },
    }
