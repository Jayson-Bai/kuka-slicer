"""Reserved continuous-fiber layer interfaces for conformal lattices.

The previous honeycomb-chain planner intentionally no longer lives here.  This
module keeps a small, stable boundary for a future fiber-path strategy without
creating fiber geometry or changing any resin trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .contracts import ConformalLatticeSpec
from .layer_embedding import LayerEmbedding
from .path_bridge import ConformalLatticePathGraph, _with_kuka_surface_orientation
from ..external_npz import ExternalSourceJob, MaterialPaths, TravelPaths
from ..honeycomb_pathing.planner import _Edge, _minimum_trail_cover


@dataclass(frozen=True, slots=True)
class FiberReinforcementResult:
    """Auditable reservation of future fiber interfaces, with no paths."""

    enabled: bool
    reserved: bool
    resin_layer_indices: tuple[int, ...]
    paths_per_layer: int
    fiber_layer_height_mm: float
    nominal_final_height_mm: float
    total_path_count: int
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class MixedWallFiberSettings:
    """Deprecated mixed-wall settings retained for compatibility tests."""

    enabled: bool
    first_after_resin_layer_physical: int
    last_after_resin_layer_physical: int
    double_track_offset_mm: float = 1.0


@dataclass(frozen=True, slots=True)
class ContinuousCourseFiberSettings:
    """Runtime-only material settings for the designer's red course plan."""

    enabled: bool
    first_after_resin_layer_physical: int
    last_after_resin_layer_physical: int


def derive_symmetric_curvature_fiber_interfaces(
    layer_embedding: LayerEmbedding,
) -> tuple[int, int]:
    """Derive resin/fiber interfaces from the design-owned curvature schedule.

    ``surface_start_layer`` is authored in the conformal designer. The main UI
    must not offer a second, manually editable layer window. A fiber is placed
    after the flat resin layer immediately before the first non-zero curvature
    layer, then mirrored about the stack: after the resin layer immediately
    following the last non-zero curvature layer.

    Returned values use the one-based physical resin-layer convention used by
    ``ContinuousCourseFiberSettings``.
    """

    alpha = np.asarray(layer_embedding.report.get("alpha_by_layer"), dtype=np.float64)
    if alpha.ndim != 1 or len(alpha) != len(layer_embedding.node_positions_xyz):
        raise ValueError("conformal layer embedding is missing a valid alpha_by_layer schedule")
    active = np.flatnonzero(np.abs(alpha) > 1e-12)
    if active.size == 0:
        raise ValueError("continuous fiber requires a design JSON with at least one non-zero-curvature layer")

    # Layer index i represents physical resin layer i + 1. The first fiber
    # follows the preceding layer i; after a curvature region ending at j, its
    # matching interface follows physical layer j + 2.
    first_after_resin_layer_physical = int(active[0])
    last_after_resin_layer_physical = int(active[-1]) + 2
    if first_after_resin_layer_physical < 1:
        raise ValueError(
            "the first non-zero-curvature layer has no preceding resin layer for supported continuous-fiber placement"
        )
    if last_after_resin_layer_physical > len(alpha):
        raise ValueError("the symmetric curvature schedule leaves no top resin layer after the final fiber interface")
    return first_after_resin_layer_physical, last_after_resin_layer_physical


def reserve_fiber_layer_interfaces(
    *,
    spec: ConformalLatticeSpec,
    layer_embedding: LayerEmbedding,
) -> FiberReinforcementResult:
    """Validate and retain the selected resin-layer interfaces only.

    No ``F`` material paths are emitted.  In particular this function never
    changes resin Z coordinates, material ordering, Core inputs, or height.
    A subsequent fiber planner can consume ``resin_layer_indices`` through the
    same result contract.
    """

    settings = spec.raw_config.get("fiber_reinforcement")
    if not isinstance(settings, Mapping):
        return _empty_result(spec, "not_configured")

    first_value = settings.get("first_after_resin_layer_physical")
    last_value = settings.get("last_after_resin_layer_physical")
    if first_value is None or last_value is None:
        return _empty_result(spec, "not_configured")
    first_physical = int(first_value)
    last_physical = int(last_value)
    layer_count = len(layer_embedding.node_positions_xyz)
    if first_physical < 1 or last_physical < first_physical:
        raise ValueError("fiber layer interfaces must be ordered positive physical resin layers")
    # This is a non-executing reservation.  A short development coupon must
    # still be sliceable even if its layer count cannot reach the saved
    # research window (for example, the default 2–19 interfaces).  The next
    # real planner owns the top-cap and material-specific feasibility check.
    selected_layers = tuple(
        layer_index
        for layer_index in range(first_physical - 1, last_physical)
        if 0 <= layer_index < max(0, layer_count - 1)
    )
    report: dict[str, object] = {
        "enabled": False,
        "reserved": True,
        "mode": "reserved_future_path_v1",
        "path_generation": "disabled_pending_replacement",
        "resin_layer_index_basis": "zero_based; physical layer = index + 1",
        "after_resin_physical_layers": [first_physical, last_physical],
        "resin_layer_indices": list(selected_layers),
        "active_window_clipped_to_available_resin_layers": last_physical >= layer_count,
        "paths_per_fiber_layer": 0,
        "total_fiber_path_count": 0,
        "automatic_resin_z_raise": False,
        "core_material_paths": "none",
        "nominal_resin_stack_height_mm": float(spec.part.get("final_height_mm", 0.0)),
        "nominal_final_height_mm": float(spec.part.get("final_height_mm", 0.0)),
    }
    return FiberReinforcementResult(
        enabled=False,
        reserved=True,
        resin_layer_indices=selected_layers,
        paths_per_layer=0,
        fiber_layer_height_mm=0.0,
        nominal_final_height_mm=float(spec.part.get("final_height_mm", 0.0)),
        total_path_count=0,
        report=report,
    )


@dataclass(frozen=True, slots=True)
class _MixedWallChainPlan:
    """Layer-invariant honeycomb courses expressed on the planar skeleton.

    A course owns complete honeycomb walls (and their mapping transition
    points), never individual triangle-clipped wall segments.  ``lane_offset``
    is applied only after the complete planar course has been embedded on a
    physical layer.
    """

    node_paths: tuple[tuple[int, ...], ...]
    lane_offsets_mm: tuple[float, ...]
    lane_roles: tuple[str, ...]
    primary_parent_edge_count: int
    secondary_parent_edge_count: int
    primary_wall_selection: str


def apply_mixed_wall_fiber_strategy(
    *,
    source_job: ExternalSourceJob,
    graph: ConformalLatticePathGraph,
    settings: MixedWallFiberSettings,
    fiber_layer_height_mm: float,
    fiber_e_per_mm: float,
) -> FiberReinforcementResult:
    """Use one complete-wall-chain plan for conformal resin and fiber paths.

    The designer owns only the honeycomb skeleton.  At the main-UI process
    boundary this function expands it into the selected mixed-wall layout:
    primary walls get two tangent-plane lanes and the remaining walls get one
    centre lane.  The resin honeycomb and the optional fiber share precisely
    the same course topology.  Fiber is merely emitted on the selected layer
    interfaces and raised by its material thickness.

    The former segment-per-path implementation is intentionally retained below
    as an unused compatibility reference.  It must not be called for a
    conformal job because its triangle-clipped segments cause pathological Core
    start/stop overhead.
    """

    if not np.isfinite(settings.double_track_offset_mm) or settings.double_track_offset_mm <= 0.0:
        raise ValueError("mixed-wall fiber double_track_offset_mm must be positive and finite")
    if any(group.material == "F" for group in source_job.material_paths):
        raise ValueError("conformal source job already contains F material paths")

    chain_plan = _plan_mixed_wall_chains(graph, settings)
    resin_paths_per_layer = _replace_honeycomb_resin_with_mixed_wall_chains(
        source_job,
        graph=graph,
        plan=chain_plan,
    )

    if not settings.enabled:
        height = _source_job_max_z(source_job)
        return FiberReinforcementResult(
            enabled=False,
            reserved=False,
            resin_layer_indices=(),
            paths_per_layer=0,
            fiber_layer_height_mm=0.0,
            nominal_final_height_mm=height,
            total_path_count=0,
            report={
                "enabled": False,
                "reserved": False,
                "mode": "uniform_mixed_wall_chain_v2",
                "configured_in": "main_ui_runtime",
                "reason": "disabled_in_main_ui",
                "primary_wall_selection": chain_plan.primary_wall_selection,
                "primary_axis_source": "automatic_longest_planar_span",
                "resin_path_strategy": "complete_planar_honeycomb_wall_chains_then_conformal_embedding",
                "resin_honeycomb_paths_per_layer": resin_paths_per_layer,
                "paths_per_fiber_layer": 0,
                "total_fiber_path_count": 0,
                "automatic_resin_z_raise": False,
                "core_material_paths": "none",
                "nominal_final_height_mm": height,
            },
        )
    if not np.isfinite(fiber_layer_height_mm) or fiber_layer_height_mm <= 0.0:
        raise ValueError("fiber_layer_height_mm must be positive and finite")
    if not np.isfinite(fiber_e_per_mm) or fiber_e_per_mm <= 0.0:
        raise ValueError("fiber_e_per_mm must be positive and finite")
    if settings.first_after_resin_layer_physical < 1:
        raise ValueError("first fiber interface must follow physical resin layer >= 1")
    if settings.last_after_resin_layer_physical < settings.first_after_resin_layer_physical:
        raise ValueError("last fiber interface must not precede the first")

    resin_layer_indices = sorted({int(group.layer_index) for group in source_job.material_paths if group.material == "R"})
    if not resin_layer_indices:
        raise ValueError("mixed-wall fiber strategy requires resin source paths")
    final_resin_layer = resin_layer_indices[-1]
    selected_layers = tuple(
        physical_layer - 1
        for physical_layer in range(
            settings.first_after_resin_layer_physical,
            settings.last_after_resin_layer_physical + 1,
        )
        if physical_layer - 1 in resin_layer_indices and physical_layer - 1 < final_resin_layer
    )
    if not selected_layers:
        return _empty_result_from_height(
            source_job,
            reason="requested_interfaces_not_available_before_top_resin_cap",
            settings=settings,
        )
    if max(selected_layers) >= len(graph.layer_node_positions_xyz):
        raise ValueError("fiber interface is outside the conformal layer embedding")

    fiber_groups: list[MaterialPaths] = []
    total_length_mm = 0.0
    for layer_index in selected_layers:
        prior_fiber_count = sum(previous < layer_index for previous in selected_layers)
        paths, profiles, length_mm = _render_mixed_wall_chain_paths(
            graph,
            chain_plan,
            layer_index=layer_index,
            e_per_mm=float(fiber_e_per_mm),
            z_offset_mm=prior_fiber_count * float(fiber_layer_height_mm) + float(fiber_layer_height_mm),
        )
        fiber_groups.append(MaterialPaths(layer_index, "F", paths, profiles))
        total_length_mm += length_mm

    _raise_resin_and_travel_z_after_fiber_interfaces(
        source_job,
        selected_layers=selected_layers,
        fiber_layer_height_mm=float(fiber_layer_height_mm),
    )
    source_job.material_paths.extend(fiber_groups)
    source_job.material_paths.sort(key=lambda group: (int(group.layer_index), 0 if group.material == "R" else 1))
    source_job.meta["fiber_reinforcement"] = {
        "enabled": True,
        "mode": "uniform_mixed_wall_chain_v2",
        "configured_in": "main_ui_runtime",
        "primary_axis_source": "automatic_longest_planar_span",
        "double_track_offset_mm": float(settings.double_track_offset_mm),
        "double_track_center_spacing_mm": float(settings.double_track_offset_mm) * 2.0,
        "first_after_resin_layer_physical": settings.first_after_resin_layer_physical,
        "last_after_resin_layer_physical": settings.last_after_resin_layer_physical,
        "layer_interface_source": "design_json_symmetric_nonzero_curvature",
        "active_resin_layer_indices": list(selected_layers),
        "primary_wall_selection": chain_plan.primary_wall_selection,
        "course_semantics": "complete supported honeycomb-wall chains; primary walls double, remaining walls single; no cross-void connector",
        "planar_skeleton_then_surface_embedding": True,
    }
    nominal_resin_height = _source_job_max_z(source_job) - len(selected_layers) * float(fiber_layer_height_mm)
    report: dict[str, object] = {
        "enabled": True,
        "reserved": False,
        "mode": "uniform_mixed_wall_chain_v2",
        "configured_in": "main_ui_runtime",
        "primary_axis_source": "automatic_longest_planar_span",
        "double_track_offset_mm": float(settings.double_track_offset_mm),
        "double_track_center_spacing_mm": float(settings.double_track_offset_mm) * 2.0,
        "after_resin_physical_layers": [settings.first_after_resin_layer_physical, settings.last_after_resin_layer_physical],
        "layer_interface_source": "design_json_symmetric_nonzero_curvature",
        "resin_layer_indices": list(selected_layers),
        "paths_per_fiber_layer": len(chain_plan.node_paths),
        "total_fiber_path_count": len(chain_plan.node_paths) * len(selected_layers),
        "resin_honeycomb_paths_per_layer": resin_paths_per_layer,
        "total_resin_honeycomb_path_count": resin_paths_per_layer * len(resin_layer_indices),
        "primary_parent_wall_count": chain_plan.primary_parent_edge_count,
        "primary_wall_selection": chain_plan.primary_wall_selection,
        "single_secondary_parent_wall_count": chain_plan.secondary_parent_edge_count,
        "total_fiber_length_mm": total_length_mm,
        "fiber_layer_height_mm": float(fiber_layer_height_mm),
        "automatic_resin_z_raise": True,
        "nominal_resin_stack_height_mm": nominal_resin_height,
        "nominal_final_height_mm": _source_job_max_z(source_job),
        "core_material_paths": "F",
        "course_semantics": "complete supported honeycomb-wall chains; no unsupported cross-void connection",
        "resin_path_strategy": "complete_planar_honeycomb_wall_chains_then_conformal_embedding",
        "planar_skeleton_then_surface_embedding": True,
        "junction_template_status": "chain endpoints preserve existing honeycomb junctions; dedicated widened-node template remains pending",
    }
    return FiberReinforcementResult(
        enabled=True,
        reserved=False,
        resin_layer_indices=selected_layers,
        paths_per_layer=len(chain_plan.node_paths),
        fiber_layer_height_mm=float(fiber_layer_height_mm),
        nominal_final_height_mm=float(report["nominal_final_height_mm"]),
        total_path_count=int(report["total_fiber_path_count"]),
        report=report,
    )


def apply_continuous_course_fiber_strategy(
    *,
    source_job: ExternalSourceJob,
    graph: ConformalLatticePathGraph,
    course_paths_by_layer: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...],
    settings: ContinuousCourseFiberSettings,
    fiber_layer_height_mm: float,
    fiber_e_per_mm: float,
) -> FiberReinforcementResult:
    """Replace legacy honeycomb walls with the preview's continuous courses.

    Only the central ``conformal_honeycomb_macro_partition`` material is
    replaced.  The pre-existing rectangular contour, separator walls and grip
    infill remain in their original roles and order.
    """

    if any(group.material == "F" for group in source_job.material_paths):
        raise ValueError("conformal source job already contains F material paths")
    resin_paths_per_layer = _replace_honeycomb_resin_with_continuous_courses(
        source_job,
        graph=graph,
        course_paths_by_layer=course_paths_by_layer,
    )
    if not settings.enabled:
        height = _source_job_max_z(source_job)
        return FiberReinforcementResult(
            enabled=False,
            reserved=False,
            resin_layer_indices=(),
            paths_per_layer=0,
            fiber_layer_height_mm=0.0,
            nominal_final_height_mm=height,
            total_path_count=0,
            report={
                "enabled": False,
                "reserved": False,
                "mode": "continuous_course_network_v1",
                "configured_in": "main_ui_runtime",
                "reason": "disabled_in_main_ui",
                "resin_path_strategy": "preview_continuous_courses_then_conformal_embedding",
                "resin_honeycomb_paths_per_layer": resin_paths_per_layer,
                "paths_per_fiber_layer": 0,
                "total_fiber_path_count": 0,
                "automatic_resin_z_raise": False,
                "core_material_paths": "none",
                "nominal_final_height_mm": height,
            },
        )
    if not np.isfinite(fiber_layer_height_mm) or fiber_layer_height_mm <= 0.0:
        raise ValueError("fiber_layer_height_mm must be positive and finite")
    if not np.isfinite(fiber_e_per_mm) or fiber_e_per_mm <= 0.0:
        raise ValueError("fiber_e_per_mm must be positive and finite")
    if settings.first_after_resin_layer_physical < 1:
        raise ValueError("first fiber interface must follow physical resin layer >= 1")
    if settings.last_after_resin_layer_physical < settings.first_after_resin_layer_physical:
        raise ValueError("last fiber interface must not precede the first")

    resin_layer_indices = sorted({int(group.layer_index) for group in source_job.material_paths if group.material == "R"})
    if not resin_layer_indices:
        raise ValueError("continuous course fiber strategy requires resin source paths")
    final_resin_layer = resin_layer_indices[-1]
    selected_layers = tuple(
        physical_layer - 1
        for physical_layer in range(settings.first_after_resin_layer_physical, settings.last_after_resin_layer_physical + 1)
        if physical_layer - 1 in resin_layer_indices and physical_layer - 1 < final_resin_layer
    )
    if not selected_layers:
        return _empty_continuous_course_result(source_job, "requested_interfaces_not_available_before_top_resin_cap", settings)
    fiber_groups: list[MaterialPaths] = []
    total_length_mm = 0.0
    work_bounds = _continuous_work_bounds(graph)
    fiber_route_specs: list[tuple[int, list[np.ndarray]]] = []
    fiber_travel_indexes: dict[str, list[int]] = {}
    for layer_index in selected_layers:
        layer_courses = _courses_for_layer(course_paths_by_layer, layer_index)
        prior_fibers = sum(previous < layer_index for previous in selected_layers)
        z_offset = (prior_fibers + 1) * float(fiber_layer_height_mm)
        paths, profiles, length_mm = _render_continuous_courses(layer_courses, float(fiber_e_per_mm), z_offset)
        course_records = [
            (path, profile, "conformal_continuous_course_fragment")
            for path, profile in zip(paths, profiles)
        ]
        lower, middle, upper = _continuous_course_sequence(
            course_records,
            current_xy=np.asarray([work_bounds[0], work_bounds[1]], dtype=np.float64),
        )
        ordered_records = [*lower, *middle, *upper]
        paths = [path for path, _profile, _role in ordered_records]
        profiles = [profile for _path, profile, _role in ordered_records]
        fiber_route_specs.append((layer_index, paths))
        fiber_groups.append(MaterialPaths(layer_index, "F", paths, profiles))
        total_length_mm += length_mm

    _raise_resin_and_travel_z_after_fiber_interfaces(
        source_job,
        selected_layers=selected_layers,
        fiber_layer_height_mm=float(fiber_layer_height_mm),
    )
    resin_roles_root = source_job.meta.get("path_roles", {}).get("R", {})
    for layer_index, paths in fiber_route_specs:
        resin_group = next(
            (group for group in source_job.material_paths if group.material == "R" and int(group.layer_index) == layer_index),
            None,
        )
        if resin_group is None or not isinstance(resin_roles_root, dict):
            raise ValueError("continuous fiber travel planner is missing its resin surface layer")
        resin_roles = resin_roles_root.get(str(layer_index), [])
        boundary_sources = [
            np.asarray(path, dtype=np.float64)
            for path, role in zip(resin_group.paths, resin_roles)
            if role in {"conformal_outer_boundary", "conformal_partition_wall"}
        ]
        if not boundary_sources:
            raise ValueError("continuous fiber travel planner is missing embedded perimeter/partition paths")
        connectors = [
            _boundary_aware_travel(
                paths[index - 1][-1],
                paths[index][0],
                bounds=work_bounds,
                boundary_sources=boundary_sources,
                # Resin has already been lifted for earlier fiber interfaces;
                # the fiber itself is exactly one fiber-layer above it.
                z_offset_mm=float(fiber_layer_height_mm),
            )
            for index in range(1, len(paths))
        ]
        travel_group = next((group for group in source_job.travel_paths if int(group.layer_index) == layer_index), None)
        if travel_group is None:
            travel_group = TravelPaths(layer_index, [])
            source_job.travel_paths.append(travel_group)
        first_connector_index = len(travel_group.paths)
        travel_group.paths.extend(connectors)
        fiber_travel_indexes[str(layer_index)] = list(range(first_connector_index, first_connector_index + len(connectors)))
    source_job.material_paths.extend(fiber_groups)
    source_job.material_paths.sort(key=lambda group: (int(group.layer_index), 0 if group.material == "R" else 1))
    source_job.travel_paths.sort(key=lambda group: int(group.layer_index))
    source_job.meta["fiber_travel_path_indexes"] = fiber_travel_indexes
    motion_order = source_job.meta.get("motion_order")
    if isinstance(motion_order, dict):
        for layer_index in selected_layers:
            records = motion_order.setdefault(str(layer_index), [])
            if not isinstance(records, list):
                raise ValueError("continuous fiber travel planner requires list motion-order records")
            for fiber_index in range(resin_paths_per_layer):
                if fiber_index:
                    records.append({"kind": "fiber_travel", "index": fiber_travel_indexes[str(layer_index)][fiber_index - 1]})
                records.append({"kind": "fiber_deposit", "index": fiber_index})
    source_job.meta["fiber_reinforcement"] = {
        "enabled": True,
        "mode": "continuous_course_network_v1",
        "configured_in": "main_ui_runtime",
        "active_resin_layer_indices": list(selected_layers),
        "course_semantics": "every rectangle-clipped continuous-course fragment is an independent resin/F path with normal Core travel/cut boundaries",
        "planar_skeleton_then_surface_embedding": True,
    }
    nominal_resin_height = _source_job_max_z(source_job) - len(selected_layers) * float(fiber_layer_height_mm)
    report = {
        "enabled": True,
        "reserved": False,
        "mode": "continuous_course_network_v1",
        "configured_in": "main_ui_runtime",
        "after_resin_physical_layers": [settings.first_after_resin_layer_physical, settings.last_after_resin_layer_physical],
        "layer_interface_source": "design_json_symmetric_nonzero_curvature",
        "resin_layer_indices": list(selected_layers),
        "paths_per_fiber_layer": resin_paths_per_layer,
        "total_fiber_path_count": resin_paths_per_layer * len(selected_layers),
        "resin_honeycomb_paths_per_layer": resin_paths_per_layer,
        "total_resin_honeycomb_path_count": resin_paths_per_layer * len(resin_layer_indices),
        "total_fiber_length_mm": total_length_mm,
        "fiber_layer_height_mm": float(fiber_layer_height_mm),
        "automatic_resin_z_raise": True,
        "nominal_resin_stack_height_mm": nominal_resin_height,
        "nominal_final_height_mm": _source_job_max_z(source_job),
        "core_material_paths": "F",
        "course_semantics": "every rectangle-clipped continuous-course fragment is an independent resin/F path with normal Core travel/cut boundaries",
        "resin_path_strategy": "preview_continuous_courses_then_conformal_embedding",
        "planar_skeleton_then_surface_embedding": True,
    }
    return FiberReinforcementResult(
        enabled=True,
        reserved=False,
        resin_layer_indices=selected_layers,
        paths_per_layer=resin_paths_per_layer,
        fiber_layer_height_mm=float(fiber_layer_height_mm),
        nominal_final_height_mm=float(report["nominal_final_height_mm"]),
        total_path_count=int(report["total_fiber_path_count"]),
        report=report,
    )


def _apply_mixed_wall_segment_strategy_legacy(
    *,
    source_job: ExternalSourceJob,
    graph: ConformalLatticePathGraph,
    settings: MixedWallFiberSettings,
    fiber_layer_height_mm: float,
    fiber_e_per_mm: float,
) -> FiberReinforcementResult:
    """Append supported mixed-wall F paths without changing lattice geometry.

    The design graph remains the sole source of resin walls.  This process-side
    adapter classifies straight honeycomb walls in the UI-selected global axis:
    those walls receive two tangent-plane lanes at ``±offset``; all oblique
    walls receive one centre lane.  Each course follows an existing resin edge.
    It intentionally never creates a fictitious connector across a cell void.
    """

    if not settings.enabled:
        return _empty_result_from_height(
            source_job,
            reason="disabled_in_main_ui",
            settings=settings,
        )
    if not np.isfinite(fiber_layer_height_mm) or fiber_layer_height_mm <= 0.0:
        raise ValueError("fiber_layer_height_mm must be positive and finite")
    if not np.isfinite(fiber_e_per_mm) or fiber_e_per_mm <= 0.0:
        raise ValueError("fiber_e_per_mm must be positive and finite")
    if settings.first_after_resin_layer_physical < 1:
        raise ValueError("first fiber interface must follow physical resin layer >= 1")
    if settings.last_after_resin_layer_physical < settings.first_after_resin_layer_physical:
        raise ValueError("last fiber interface must not precede the first")
    if not np.isfinite(settings.double_track_offset_mm) or settings.double_track_offset_mm <= 0.0:
        raise ValueError("mixed-wall fiber double_track_offset_mm must be positive and finite")
    if any(group.material == "F" for group in source_job.material_paths):
        raise ValueError("conformal source job already contains F material paths")

    resin_layer_indices = sorted({int(group.layer_index) for group in source_job.material_paths if group.material == "R"})
    if not resin_layer_indices:
        raise ValueError("mixed-wall fiber strategy requires resin source paths")
    final_resin_layer = resin_layer_indices[-1]
    selected_layers = tuple(
        physical_layer - 1
        for physical_layer in range(
            settings.first_after_resin_layer_physical,
            settings.last_after_resin_layer_physical + 1,
        )
        if physical_layer - 1 in resin_layer_indices and physical_layer - 1 < final_resin_layer
    )
    if not selected_layers:
        return _empty_result_from_height(
            source_job,
            reason="requested_interfaces_not_available_before_top_resin_cap",
            settings=settings,
        )
    if max(selected_layers) >= len(graph.layer_node_positions_xyz):
        raise ValueError("fiber interface is outside the conformal layer embedding")

    reference_positions = np.asarray(graph.layer_node_positions_xyz[0], dtype=np.float64)
    axis_index = _automatic_primary_axis(reference_positions)
    edge_nodes = np.asarray(graph.edge_node_ids, dtype=np.int64)
    primary_mask, primary_selection = _primary_wall_mask(reference_positions, edge_nodes, axis_index)
    if not bool(np.any(primary_mask)):
        raise ValueError("no honeycomb walls were found along the automatically selected primary span")

    fiber_groups: list[MaterialPaths] = []
    paths_per_layer = 0
    primary_edge_count = int(np.count_nonzero(primary_mask))
    oblique_edge_count = int(len(edge_nodes) - primary_edge_count)
    total_length_mm = 0.0
    for layer_index in selected_layers:
        prior_fiber_count = sum(previous < layer_index for previous in selected_layers)
        xyz = np.asarray(graph.layer_node_positions_xyz[layer_index], dtype=np.float64)
        normals = np.asarray(graph.layer_tool_normals_xyz[layer_index], dtype=np.float64)
        layer_paths: list[np.ndarray] = []
        layer_extrusion: list[np.ndarray] = []
        for edge_index, (first, second) in enumerate(edge_nodes):
            points = np.vstack((xyz[first], xyz[second]))
            point_normals = np.vstack((normals[first], normals[second]))
            if primary_mask[edge_index]:
                for side in (-1.0, 1.0):
                    offset_points = _offset_edge_in_surface_tangent(
                        points,
                        point_normals,
                        side * float(settings.double_track_offset_mm),
                    )
                    offset_points[:, 2] += prior_fiber_count * fiber_layer_height_mm + fiber_layer_height_mm
                    length_mm = float(np.linalg.norm(offset_points[1] - offset_points[0]))
                    layer_paths.append(_with_kuka_surface_orientation(offset_points, point_normals))
                    layer_extrusion.append(np.asarray((0.0, length_mm * fiber_e_per_mm), dtype=np.float64))
                    total_length_mm += length_mm
            else:
                centre_points = np.array(points, copy=True)
                centre_points[:, 2] += prior_fiber_count * fiber_layer_height_mm + fiber_layer_height_mm
                length_mm = float(np.linalg.norm(centre_points[1] - centre_points[0]))
                layer_paths.append(_with_kuka_surface_orientation(centre_points, point_normals))
                layer_extrusion.append(np.asarray((0.0, length_mm * fiber_e_per_mm), dtype=np.float64))
                total_length_mm += length_mm
        fiber_groups.append(MaterialPaths(layer_index, "F", layer_paths, layer_extrusion))
        paths_per_layer = len(layer_paths)

    _raise_resin_and_travel_z_after_fiber_interfaces(
        source_job,
        selected_layers=selected_layers,
        fiber_layer_height_mm=float(fiber_layer_height_mm),
    )
    source_job.material_paths.extend(fiber_groups)
    source_job.material_paths.sort(key=lambda group: (int(group.layer_index), 0 if group.material == "R" else 1))
    source_job.meta["fiber_reinforcement"] = {
        "enabled": True,
        "mode": "uniform_mixed_wall_v1",
        "configured_in": "main_ui_runtime",
        "primary_axis_source": "automatic_longest_planar_span",
        "double_track_offset_mm": float(settings.double_track_offset_mm),
        "double_track_center_spacing_mm": float(settings.double_track_offset_mm) * 2.0,
        "first_after_resin_layer_physical": settings.first_after_resin_layer_physical,
        "last_after_resin_layer_physical": settings.last_after_resin_layer_physical,
        "layer_interface_source": "design_json_symmetric_nonzero_curvature",
        "active_resin_layer_indices": list(selected_layers),
        "primary_wall_selection": primary_selection,
        "course_semantics": "supported edge courses; primary walls double, oblique walls single; no cross-void connector",
    }
    nominal_resin_height = _source_job_max_z(source_job) - len(selected_layers) * float(fiber_layer_height_mm)
    report: dict[str, object] = {
        "enabled": True,
        "reserved": False,
        "mode": "uniform_mixed_wall_v1",
        "configured_in": "main_ui_runtime",
        "primary_axis_source": "automatic_longest_planar_span",
        "double_track_offset_mm": float(settings.double_track_offset_mm),
        "double_track_center_spacing_mm": float(settings.double_track_offset_mm) * 2.0,
        "after_resin_physical_layers": [settings.first_after_resin_layer_physical, settings.last_after_resin_layer_physical],
        "layer_interface_source": "design_json_symmetric_nonzero_curvature",
        "resin_layer_indices": list(selected_layers),
        "paths_per_fiber_layer": paths_per_layer,
        "total_fiber_path_count": paths_per_layer * len(selected_layers),
        "primary_wall_edge_count": primary_edge_count,
        "primary_wall_selection": primary_selection,
        "single_oblique_wall_edge_count": oblique_edge_count,
        "total_fiber_length_mm": total_length_mm,
        "fiber_layer_height_mm": float(fiber_layer_height_mm),
        "automatic_resin_z_raise": True,
        "nominal_resin_stack_height_mm": nominal_resin_height,
        "nominal_final_height_mm": _source_job_max_z(source_job),
        "core_material_paths": "F",
        "course_semantics": "individual supported honeycomb edge courses; no unsupported cross-void connection",
        "junction_template_status": "pending_dedicated_node-template planner",
    }
    return FiberReinforcementResult(
        enabled=True,
        reserved=False,
        resin_layer_indices=selected_layers,
        paths_per_layer=paths_per_layer,
        fiber_layer_height_mm=float(fiber_layer_height_mm),
        nominal_final_height_mm=float(report["nominal_final_height_mm"]),
        total_path_count=int(report["total_fiber_path_count"]),
        report=report,
    )


def _plan_mixed_wall_chains(
    graph: ConformalLatticePathGraph,
    settings: MixedWallFiberSettings,
) -> _MixedWallChainPlan:
    """Restore complete planar honeycomb walls and cover them with courses."""

    parent_paths = _complete_parent_wall_node_paths(graph)
    reference_positions = np.asarray(graph.layer_node_positions_xyz[0], dtype=np.float64)
    parent_ids = tuple(sorted(parent_paths))
    endpoint_pairs = np.asarray(
        [[parent_paths[parent][0], parent_paths[parent][-1]] for parent in parent_ids],
        dtype=np.int64,
    )
    axis_index = _automatic_primary_axis(reference_positions)
    primary_mask, selection = _primary_wall_mask(reference_positions, endpoint_pairs, axis_index)
    if not bool(np.any(primary_mask)):
        raise ValueError("no honeycomb walls were found along the automatically selected primary span")

    primary_ids = tuple(parent for parent, is_primary in zip(parent_ids, primary_mask) if is_primary)
    secondary_ids = tuple(parent for parent, is_primary in zip(parent_ids, primary_mask) if not is_primary)
    primary_chains = _cover_parent_walls_with_complete_chains(parent_paths, primary_ids)
    secondary_chains = _cover_parent_walls_with_complete_chains(parent_paths, secondary_ids)
    node_paths: list[tuple[int, ...]] = []
    lane_offsets: list[float] = []
    lane_roles: list[str] = []
    # Do not add a centre track to a primary wall: its two ±1 mm courses are
    # the two tracks of the mixed wall.  Secondary walls stay as one centre
    # course.  This matches the reference uniform-mixed-wall topology.
    for chain in primary_chains:
        for offset in (-float(settings.double_track_offset_mm), float(settings.double_track_offset_mm)):
            node_paths.append(chain)
            lane_offsets.append(offset)
            lane_roles.append("double_primary_honeycomb_wall_chain")
    for chain in secondary_chains:
        node_paths.append(chain)
        lane_offsets.append(0.0)
        lane_roles.append("single_secondary_honeycomb_wall_chain")
    if not node_paths:
        raise ValueError("mixed-wall chain planner found no complete honeycomb courses")
    return _MixedWallChainPlan(
        node_paths=tuple(node_paths),
        lane_offsets_mm=tuple(lane_offsets),
        lane_roles=tuple(lane_roles),
        primary_parent_edge_count=len(primary_ids),
        secondary_parent_edge_count=len(secondary_ids),
        primary_wall_selection=selection,
    )


def _complete_parent_wall_node_paths(graph: ConformalLatticePathGraph) -> dict[int, tuple[int, ...]]:
    """Join triangle-clipped segments back into their original planar wall.

    ``edge_parent_id`` is assigned before an abstract honeycomb wall crosses
    surface triangles.  Rejoining by this identifier retains every mapping
    transition node in order, yet prevents those transitions from becoming
    artificial fiber/path starts.
    """

    edge_nodes = np.asarray(graph.edge_node_ids, dtype=np.int64)
    parent_ids = np.asarray(graph.edge_parent_id, dtype=np.int64)
    grouped: dict[int, list[tuple[int, int]]] = {}
    for (first, second), parent in zip(edge_nodes, parent_ids):
        grouped.setdefault(int(parent), []).append((int(first), int(second)))
    result: dict[int, tuple[int, ...]] = {}
    for parent, segments in grouped.items():
        adjacency: dict[int, list[tuple[int, int]]] = {}
        for segment_index, (first, second) in enumerate(segments):
            adjacency.setdefault(first, []).append((second, segment_index))
            adjacency.setdefault(second, []).append((first, segment_index))
        endpoints = sorted(node for node, entries in adjacency.items() if len(entries) == 1)
        if len(endpoints) != 2 or any(len(entries) > 2 for entries in adjacency.values()):
            raise ValueError("a conformal honeycomb parent wall must be a single non-branching mapped chain")
        nodes = [endpoints[0]]
        used: set[int] = set()
        previous: int | None = None
        current = endpoints[0]
        while True:
            choices = [(other, segment) for other, segment in adjacency[current] if segment not in used]
            if not choices:
                break
            other, segment = min(choices, key=lambda item: (item[0], item[1]))
            used.add(segment)
            nodes.append(other)
            previous, current = current, other
            if previous == current:
                raise ValueError("conformal honeycomb parent wall contains a zero-length segment")
        if len(used) != len(segments) or nodes[-1] != endpoints[1]:
            raise ValueError("failed to reconstruct a complete conformal honeycomb parent wall")
        result[parent] = tuple(nodes)
    return result


def _cover_parent_walls_with_complete_chains(
    parent_paths: Mapping[int, tuple[int, ...]],
    parent_ids: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Create an exact, no-cross-void trail cover from full honeycomb walls."""

    if not parent_ids:
        return ()
    parent_by_endpoint: dict[frozenset[int], int] = {}
    graph_edges: list[_Edge] = []
    for parent in parent_ids:
        nodes = parent_paths[parent]
        first, second = nodes[0], nodes[-1]
        key = frozenset((first, second))
        if len(key) != 2 or key in parent_by_endpoint:
            raise ValueError("complete honeycomb parent walls must have unique non-zero endpoint pairs")
        parent_by_endpoint[key] = parent
        graph_edges.append(_Edge((float(first), 0.0), (float(second), 0.0), 1.0))

    result: list[tuple[int, ...]] = []
    for raw_trail in _minimum_trail_cover(graph_edges):
        endpoints = [int(round(point[0])) for point in raw_trail]
        if len(endpoints) < 2:
            continue
        nodes: list[int] = []
        for first, second in zip(endpoints, endpoints[1:]):
            parent = parent_by_endpoint.get(frozenset((first, second)))
            if parent is None:
                raise ValueError("parent-wall trail cover introduced a non-honeycomb connector")
            wall = list(parent_paths[parent])
            if wall[0] != first:
                wall.reverse()
            if wall[0] != first or wall[-1] != second:
                raise ValueError("parent-wall trail orientation does not match its complete wall endpoints")
            nodes.extend(wall if not nodes else wall[1:])
        if len(nodes) >= 2:
            result.append(tuple(nodes))
    if not result:
        raise ValueError("complete honeycomb parent-wall cover produced no paths")
    return tuple(result)


def _render_mixed_wall_chain_paths(
    graph: ConformalLatticePathGraph,
    plan: _MixedWallChainPlan,
    *,
    layer_index: int,
    e_per_mm: float,
    z_offset_mm: float,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Embed a layer-invariant wall skeleton and create physical E profiles."""

    positions = np.asarray(graph.layer_node_positions_xyz[layer_index], dtype=np.float64)
    tool_normals = np.asarray(graph.layer_tool_normals_xyz[layer_index], dtype=np.float64)
    paths: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    total_length_mm = 0.0
    for nodes, lateral_offset in zip(plan.node_paths, plan.lane_offsets_mm):
        node_ids = np.asarray(nodes, dtype=np.int64)
        points = np.array(positions[node_ids], copy=True)
        normals = np.asarray(tool_normals[node_ids], dtype=np.float64)
        if abs(lateral_offset) > 1e-12:
            points = _offset_chain_in_surface_tangent(points, normals, lateral_offset)
        points[:, 2] += z_offset_mm
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths <= 1e-9):
            raise ValueError("complete honeycomb wall chain contains a zero-length mapped segment")
        profiles.append(np.concatenate(([0.0], np.cumsum(lengths * float(e_per_mm)))))
        paths.append(_with_kuka_surface_orientation(points, normals))
        total_length_mm += float(np.sum(lengths))
    return paths, profiles, total_length_mm


def _offset_chain_in_surface_tangent(points: np.ndarray, normals: np.ndarray, offset_mm: float) -> np.ndarray:
    """Offset a complete chain smoothly without splitting it at mesh borders."""

    result = np.empty_like(points)
    for index, point in enumerate(points):
        before = points[max(0, index - 1)]
        after = points[min(len(points) - 1, index + 1)]
        tangent = after - before
        tangent_length = float(np.linalg.norm(tangent))
        lateral = np.cross(normals[index], tangent / tangent_length) if tangent_length > 1e-12 else np.zeros(3)
        lateral_length = float(np.linalg.norm(lateral))
        result[index] = point if lateral_length <= 1e-12 else point + offset_mm * lateral / lateral_length
    return result


def _courses_for_layer(
    course_paths_by_layer: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...],
    layer_index: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if layer_index < 0 or layer_index >= len(course_paths_by_layer):
        raise ValueError("continuous course paths are missing a physical resin layer")
    courses = course_paths_by_layer[layer_index]
    if not courses:
        raise ValueError("continuous course plan has no printable paths for a physical resin layer")
    return courses


def _render_continuous_courses(
    courses: tuple[tuple[np.ndarray, np.ndarray], ...],
    e_per_mm: float,
    z_offset_mm: float,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    paths: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    total_length = 0.0
    for points_xyz, normals_xyz in courses:
        points = np.asarray(points_xyz, dtype=np.float64).copy()
        normals = np.asarray(normals_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or normals.shape != points.shape or len(points) < 2:
            raise ValueError("continuous course embedding must provide matching Nx3 points/normals")
        points[:, 2] += z_offset_mm
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        paths.append(_with_kuka_surface_orientation(points, normals))
        profiles.append(np.concatenate(([0.0], np.cumsum(lengths * e_per_mm))))
        total_length += float(lengths.sum())
    return paths, profiles, total_length


def _reverse_deposition_path(path: np.ndarray, profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reverse a path while retaining its zero-based cumulative-E meaning."""
    reversed_path = np.asarray(path, dtype=np.float64)[::-1].copy()
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or len(values) != len(reversed_path):
        raise ValueError("continuous-course path/profile lengths must match")
    return reversed_path, values[-1] - values[::-1]


def _orient_record_from_current(
    record: tuple[np.ndarray, np.ndarray, str],
    current_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    path, profile, role = record
    if len(path) < 2:
        raise ValueError("continuous-course travel planner received a one-point deposition path")
    if np.linalg.norm(path[-1, :2] - current_xy) < np.linalg.norm(path[0, :2] - current_xy):
        reversed_path, reversed_profile = _reverse_deposition_path(path, profile)
        return reversed_path, reversed_profile, role
    return path, profile, role


def _split_course_boundary_groups(
    records: list[tuple[np.ndarray, np.ndarray, str]],
) -> tuple[list[tuple[np.ndarray, np.ndarray, str]], list[tuple[np.ndarray, np.ndarray, str]], list[tuple[np.ndarray, np.ndarray, str]]]:
    """Separate lower/upper clipped fragments from through-course fragments."""
    endpoint_y = np.asarray([point[1] for path, _profile, _role in records for point in (path[0], path[-1])])
    if endpoint_y.size == 0:
        return [], [], []
    lower_y = float(endpoint_y.min())
    upper_y = float(endpoint_y.max())
    tolerance = 1e-6
    lower = [record for record in records if np.all(np.abs(record[0][[0, -1], 1] - lower_y) <= tolerance)]
    upper = [record for record in records if np.all(np.abs(record[0][[0, -1], 1] - upper_y) <= tolerance)]
    boundary_ids = {id(record) for record in [*lower, *upper]}
    middle = [record for record in records if id(record) not in boundary_ids]
    return lower, middle, upper


def _append_oriented_records(
    target: list[tuple[np.ndarray, np.ndarray, str]],
    records: list[tuple[np.ndarray, np.ndarray, str]],
    current_xy: np.ndarray,
) -> np.ndarray:
    for record in records:
        oriented = _orient_record_from_current(record, current_xy)
        target.append(oriented)
        current_xy = oriented[0][-1, :2]
    return current_xy


def _continuous_course_sequence(
    records: list[tuple[np.ndarray, np.ndarray, str]],
    *,
    current_xy: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray, str]], list[tuple[np.ndarray, np.ndarray, str]], list[tuple[np.ndarray, np.ndarray, str]]]:
    """Return lower, central and upper course visits in safe serpentine order."""
    lower, middle, upper = _split_course_boundary_groups(records)
    lower.sort(key=lambda record: float(np.mean(record[0][:, 0])))
    middle.sort(key=lambda record: float(np.mean(record[0][:, 1])))
    lower_ordered: list[tuple[np.ndarray, np.ndarray, str]] = []
    current_xy = _append_oriented_records(lower_ordered, lower, current_xy)
    middle_ordered: list[tuple[np.ndarray, np.ndarray, str]] = []
    current_xy = _append_oriented_records(middle_ordered, middle, current_xy)
    # After central serpentine courses, begin at the nearer upper fragment and
    # continue outward.  This keeps every connector on a rectangle boundary.
    upper.sort(
        key=lambda record: float(np.mean(record[0][:, 0])),
        reverse=bool(upper and current_xy[0] >= np.mean([np.mean(record[0][:, 0]) for record in upper])),
    )
    upper_ordered: list[tuple[np.ndarray, np.ndarray, str]] = []
    _append_oriented_records(upper_ordered, upper, current_xy)
    return lower_ordered, middle_ordered, upper_ordered


def _continuous_work_bounds(graph: ConformalLatticePathGraph) -> tuple[float, float, float, float]:
    config = graph.metadata.get("config", {})
    additional = config.get("additional", {}) if isinstance(config, Mapping) else {}
    part = additional.get("part", {}) if isinstance(additional, Mapping) else {}
    try:
        length = float(part["length_mm"])
        width = float(part["width_mm"])
        # Older/minimal rectangular design JSONs omit the optional grip field;
        # they are valid zero-grip designs and their outer perimeter alone is
        # sufficient for the safe-boundary Travel router.
        grip = float(part.get("symmetric_grip_end_length_mm", 0.0))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("continuous-course travel planner is missing rectangular part/grip metadata") from error
    if not (length > 2.0 * grip and width > 0.0):
        raise ValueError("continuous-course travel planner received invalid work bounds")
    return grip, 0.0, length - grip, width


def _order_resin_records_for_continuous_courses(
    retained: list[tuple[np.ndarray, np.ndarray, str]],
    courses: list[tuple[np.ndarray, np.ndarray, str]],
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Preserve walls/grips while visiting course fragments without long diagonals."""
    outer = [record for record in retained if record[2] == "conformal_outer_boundary"]
    partitions = [record for record in retained if record[2] == "conformal_partition_wall"]
    grips = [record for record in retained if record[2] == "conformal_grip_zigzag_x_one_stroke"]
    known = {id(record) for record in [*outer, *partitions, *grips]}
    other = [record for record in retained if id(record) not in known]
    partitions.sort(key=lambda record: float(np.mean(record[0][:, 0])))
    grips.sort(key=lambda record: float(np.mean(record[0][:, 0])))

    result: list[tuple[np.ndarray, np.ndarray, str]] = []
    current_xy = np.asarray([0.0, 0.0], dtype=np.float64)
    current_xy = _append_oriented_records(result, outer, current_xy)
    current_xy = _append_oriented_records(result, other, current_xy)
    if partitions:
        current_xy = _append_oriented_records(result, [partitions[0]], current_xy)
    if grips:
        current_xy = _append_oriented_records(result, [grips[0]], current_xy)

    lower, middle, upper = _continuous_course_sequence(courses, current_xy=current_xy)
    current_xy = _append_oriented_records(result, lower, current_xy)
    current_xy = _append_oriented_records(result, middle, current_xy)
    if len(partitions) > 1:
        current_xy = _append_oriented_records(result, [partitions[-1]], current_xy)
    if len(grips) > 1:
        current_xy = _append_oriented_records(result, [grips[-1]], current_xy)
    _append_oriented_records(result, upper, current_xy)
    return result


def _boundary_candidates(point_xy: np.ndarray, bounds: tuple[float, float, float, float]) -> list[tuple[float, np.ndarray]]:
    left, lower, right, upper = bounds
    width = right - left
    height = upper - lower
    bottom_x = min(right, max(left, float(point_xy[0])))
    right_y = min(upper, max(lower, float(point_xy[1])))
    top_x = bottom_x
    left_y = right_y
    return [
        (bottom_x - left, np.asarray([bottom_x, lower], dtype=np.float64)),
        (width + right_y - lower, np.asarray([right, right_y], dtype=np.float64)),
        (width + height + right - top_x, np.asarray([top_x, upper], dtype=np.float64)),
        (2.0 * width + height + upper - left_y, np.asarray([left, left_y], dtype=np.float64)),
    ]


def _perimeter_point(parameter: float, bounds: tuple[float, float, float, float]) -> np.ndarray:
    left, lower, right, upper = bounds
    width = right - left
    height = upper - lower
    perimeter = 2.0 * (width + height)
    value = parameter % perimeter
    if value <= width:
        return np.asarray([left + value, lower], dtype=np.float64)
    if value <= width + height:
        return np.asarray([right, lower + value - width], dtype=np.float64)
    if value <= 2.0 * width + height:
        return np.asarray([right - (value - width - height), upper], dtype=np.float64)
    return np.asarray([left, upper - (value - 2.0 * width - height)], dtype=np.float64)


def _perimeter_route(start_parameter: float, end_parameter: float, bounds: tuple[float, float, float, float], forward: bool) -> list[np.ndarray]:
    left, lower, right, upper = bounds
    del left, lower
    width = right - bounds[0]
    height = upper - bounds[1]
    perimeter = 2.0 * (width + height)
    if forward:
        distance = (end_parameter - start_parameter) % perimeter
        target = start_parameter + distance
        corner_parameters = [width, width + height, 2.0 * width + height, perimeter]
        points = [_perimeter_point(start_parameter, bounds)]
        for corner in corner_parameters:
            while corner <= start_parameter:
                corner += perimeter
            if corner < target - 1e-9:
                points.append(_perimeter_point(corner, bounds))
        points.append(_perimeter_point(end_parameter, bounds))
        return points
    reverse = _perimeter_route(end_parameter, start_parameter, bounds, True)
    return list(reversed(reverse))


def _boundary_pose_at_xy(
    xy: np.ndarray,
    boundary_sources: list[np.ndarray],
    *,
    z_offset_mm: float,
) -> np.ndarray:
    best: tuple[float, np.ndarray] | None = None
    for path in boundary_sources:
        for start, end in zip(path, path[1:]):
            delta = end[:2] - start[:2]
            length_sq = float(np.dot(delta, delta))
            fraction = 0.0 if length_sq <= 1e-16 else min(1.0, max(0.0, float(np.dot(xy - start[:2], delta) / length_sq)))
            candidate = start + fraction * (end - start)
            distance = float(np.linalg.norm(candidate[:2] - xy))
            if best is None or distance < best[0]:
                best = distance, candidate
    if best is None or best[0] > 1e-4:
        raise ValueError("continuous-course travel waypoint is not covered by an embedded boundary path")
    pose = best[1].copy()
    pose[2] += z_offset_mm
    return pose


def _boundary_aware_travel(
    start: np.ndarray,
    end: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    boundary_sources: list[np.ndarray],
    z_offset_mm: float = 0.0,
) -> np.ndarray:
    """Return the shortest rectangle-boundary route; never a through-pore chord."""
    if start.shape[0] != 6 or end.shape[0] != 6:
        raise ValueError("continuous-course travel requires XYZABC deposition endpoints")
    choices: list[tuple[float, bool, float, float, np.ndarray, np.ndarray]] = []
    left, lower, right, upper = bounds
    perimeter = 2.0 * ((right - left) + (upper - lower))
    for start_parameter, start_xy in _boundary_candidates(start[:2], bounds):
        for end_parameter, end_xy in _boundary_candidates(end[:2], bounds):
            forward_distance = (end_parameter - start_parameter) % perimeter
            reverse_distance = (start_parameter - end_parameter) % perimeter
            approach = float(np.linalg.norm(start[:2] - start_xy) + np.linalg.norm(end[:2] - end_xy))
            choices.append((approach + forward_distance, True, start_parameter, end_parameter, start_xy, end_xy))
            choices.append((approach + reverse_distance, False, start_parameter, end_parameter, start_xy, end_xy))
    _cost, forward, start_parameter, end_parameter, _start_xy, _end_xy = min(choices, key=lambda choice: choice[0])
    route_xy = [start[:2], *_perimeter_route(start_parameter, end_parameter, bounds, forward), end[:2]]
    points: list[np.ndarray] = []
    for index, xy in enumerate(route_xy):
        pose = start.copy() if index == 0 else end.copy() if index == len(route_xy) - 1 else _boundary_pose_at_xy(xy, boundary_sources, z_offset_mm=z_offset_mm)
        if not points or float(np.linalg.norm(pose[:3] - points[-1][:3])) > 1e-8:
            points.append(pose)
    return np.asarray(points, dtype=np.float64)


def _replace_honeycomb_resin_with_continuous_courses(
    source_job: ExternalSourceJob,
    *,
    graph: ConformalLatticePathGraph,
    course_paths_by_layer: tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...],
) -> int:
    """Replace only central legacy walls; preserve contour and grip materials."""

    extrusion_config = graph.metadata.get("config", {}).get("extrusion")
    if not isinstance(extrusion_config, Mapping):
        raise ValueError("conformal path graph is missing extrusion conversion metadata")
    bead_area = float(extrusion_config["bead_cross_section_area_mm2"])
    e_volume = float(extrusion_config["e_volume_per_unit_mm3"])
    if not np.isfinite(bead_area) or not np.isfinite(e_volume) or bead_area <= 0.0 or e_volume <= 0.0:
        raise ValueError("conformal path graph has invalid resin extrusion conversion metadata")
    resin_roles = source_job.meta.setdefault("path_roles", {}).get("R")
    if not isinstance(resin_roles, dict):
        raise ValueError("conformal source job is missing resin path role metadata")
    travel_groups: list[TravelPaths] = []
    motion_order = source_job.meta.setdefault("motion_order", {})
    if not isinstance(motion_order, dict):
        raise ValueError("conformal source job motion_order metadata must be an object")
    expected_count: int | None = None
    for group in source_job.material_paths:
        if group.material != "R":
            continue
        roles = resin_roles.get(str(group.layer_index))
        if not isinstance(roles, list) or len(roles) != len(group.paths):
            raise ValueError("conformal resin path roles do not match its source paths")
        if group.extrusion is None or len(group.extrusion) != len(group.paths):
            raise ValueError("conformal resin paths require matching physical E profiles")
        retained = [
            (path, extrusion, role)
            for path, extrusion, role in zip(group.paths, group.extrusion, roles)
            if role != "conformal_honeycomb_macro_partition"
        ]
        paths, profiles, _ = _render_continuous_courses(
            _courses_for_layer(course_paths_by_layer, int(group.layer_index)),
            bead_area / e_volume,
            0.0,
        )
        course_records = [
            (path, profile, "conformal_continuous_course_fragment")
            for path, profile in zip(paths, profiles)
        ]
        ordered = _order_resin_records_for_continuous_courses(retained, course_records)
        boundary_sources = [
            np.asarray(path, dtype=np.float64)
            for path, _profile, role in retained
            if role in {"conformal_outer_boundary", "conformal_partition_wall"}
        ]
        if not boundary_sources:
            raise ValueError("continuous-course travel planner is missing embedded perimeter/partition paths")
        work_bounds = _continuous_work_bounds(graph)
        travels = [
            _boundary_aware_travel(
                ordered[index - 1][0][-1],
                ordered[index][0][0],
                bounds=work_bounds,
                boundary_sources=boundary_sources,
            )
            for index in range(1, len(ordered))
        ]
        expected_count = len(paths) if expected_count is None else expected_count
        if len(paths) != expected_count:
            raise ValueError("continuous course count must be invariant across physical layers")
        group.paths = [path for path, _profile, _role in ordered]
        group.extrusion = [profile for _path, profile, _role in ordered]
        resin_roles[str(group.layer_index)] = [role for _path, _profile, role in ordered]
        travel_groups.append(TravelPaths(int(group.layer_index), travels))
        motion_order[str(group.layer_index)] = [
            record
            for index in range(len(ordered))
            for record in (
                ([] if index == 0 else [{"kind": "travel", "index": index - 1}])
                + [{"kind": "deposit", "index": index}]
            )
        ]
    source_job.travel_paths = travel_groups
    bridge = source_job.meta.get("conformal_lattice_path_bridge")
    if isinstance(bridge, dict):
        bridge["path_order"] = "preview continuous courses embedded per physical layer"
        bridge["trail_partition_status"] = "superseded_by_continuous_course_network_v1"
    return 0 if expected_count is None else expected_count


def _empty_continuous_course_result(
    source_job: ExternalSourceJob,
    reason: str,
    settings: ContinuousCourseFiberSettings,
) -> FiberReinforcementResult:
    height = _source_job_max_z(source_job)
    return FiberReinforcementResult(
        enabled=False,
        reserved=False,
        resin_layer_indices=(),
        paths_per_layer=0,
        fiber_layer_height_mm=0.0,
        nominal_final_height_mm=height,
        total_path_count=0,
        report={
            "enabled": False,
            "reserved": False,
            "mode": "continuous_course_network_v1",
            "reason": reason,
            "after_resin_physical_layers": [settings.first_after_resin_layer_physical, settings.last_after_resin_layer_physical],
            "paths_per_fiber_layer": 0,
            "total_fiber_path_count": 0,
            "automatic_resin_z_raise": False,
            "core_material_paths": "none",
            "nominal_final_height_mm": height,
        },
    )


def _replace_honeycomb_resin_with_mixed_wall_chains(
    source_job: ExternalSourceJob,
    *,
    graph: ConformalLatticePathGraph,
    plan: _MixedWallChainPlan,
) -> int:
    """Replace only honeycomb resin; retain perimeter and solid-grip paths."""

    extrusion_config = graph.metadata.get("config", {}).get("extrusion")
    if not isinstance(extrusion_config, Mapping):
        raise ValueError("conformal path graph is missing extrusion conversion metadata")
    bead_area = float(extrusion_config["bead_cross_section_area_mm2"])
    e_volume = float(extrusion_config["e_volume_per_unit_mm3"])
    if not np.isfinite(bead_area) or not np.isfinite(e_volume) or bead_area <= 0.0 or e_volume <= 0.0:
        raise ValueError("conformal path graph has invalid resin extrusion conversion metadata")
    resin_e_per_mm = bead_area / e_volume
    path_roles = source_job.meta.setdefault("path_roles", {})
    if not isinstance(path_roles, dict):
        raise ValueError("conformal source job path_roles metadata must be an object")
    resin_roles = path_roles.get("R")
    if not isinstance(resin_roles, dict):
        raise ValueError("conformal source job is missing resin path role metadata")
    for group in source_job.material_paths:
        if group.material != "R":
            continue
        roles = resin_roles.get(str(group.layer_index))
        if not isinstance(roles, list) or len(roles) != len(group.paths):
            raise ValueError("conformal resin path roles do not match its source paths")
        if group.extrusion is None or len(group.extrusion) != len(group.paths):
            raise ValueError("conformal resin paths require matching physical E profiles")
        retained = [
            (path, extrusion, role)
            for path, extrusion, role in zip(group.paths, group.extrusion, roles)
            if role != "conformal_honeycomb_macro_partition"
        ]
        paths, profiles, _length = _render_mixed_wall_chain_paths(
            graph,
            plan,
            layer_index=int(group.layer_index),
            e_per_mm=resin_e_per_mm,
            z_offset_mm=0.0,
        )
        group.paths = [path for path, _extrusion, _role in retained] + paths
        group.extrusion = [extrusion for _path, extrusion, _role in retained] + profiles
        resin_roles[str(group.layer_index)] = [role for _path, _extrusion, role in retained] + list(plan.lane_roles)

    # The old macro partition emits explicit zero-E travel links for its own
    # trail cover.  Those links do not belong to the new complete-wall courses;
    # Core now receives just the shared resin/fiber wall paths and handles the
    # inter-course non-depositing motion in its normal source-job stage.
    source_job.travel_paths = []
    bridge = source_job.meta.get("conformal_lattice_path_bridge")
    if isinstance(bridge, dict):
        bridge["path_order"] = "complete planar honeycomb-wall chains embedded per physical layer"
        bridge["trail_partition_status"] = "superseded_by_main_ui_uniform_mixed_wall_chain_v2"
    return len(plan.node_paths)


def _automatic_primary_axis(reference_positions: np.ndarray) -> int:
    """Select the longer planar part span for the deprecated fallback."""

    positions = np.asarray(reference_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] < 2 or len(positions) == 0:
        raise ValueError("reference_positions must contain planar XY coordinates")
    return int(np.argmax(np.ptp(positions[:, :2], axis=0)))


def _primary_wall_mask(
    positions: np.ndarray,
    edge_nodes: np.ndarray,
    axis_index: int,
) -> tuple[np.ndarray, str]:
    """Select the double-wall family without redefining the resin topology.

    A regular honeycomb can contain a wall family directly parallel to the
    requested global axis.  In the other valid orientation it instead carries
    load through a pair of inclined edges that form an axis-progressing zigzag
    chain.  The latter is the intended X-direction chain in the designer's
    default, vertical-wall honeycomb layout.  Treating only perfectly parallel
    edges as valid caused a false failure for that normal configuration.
    """

    deltas = positions[edge_nodes[:, 1], :2] - positions[edge_nodes[:, 0], :2]
    lengths = np.linalg.norm(deltas, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError("fiber planner received a zero-length honeycomb edge")
    axis_projection = np.abs(deltas[:, axis_index]) / lengths
    # A 15° window isolates a wall family directly parallel to the requested
    # axis whenever the selected honeycomb orientation contains one.
    direct = axis_projection >= float(np.cos(np.deg2rad(15.0)))
    if bool(np.any(direct)):
        return direct, "direct_parallel_wall_family"

    # Otherwise select both inclined families whose alternating segments form
    # a continuous zigzag progression along the requested axis. For a regular
    # hexagon their projection is sqrt(3)/2; 0.5 leaves a wide separation from
    # the perpendicular wall family (projection 0) without depending on an
    # exact tessellation angle.
    zigzag = axis_projection >= 0.5
    if bool(np.any(zigzag)):
        return zigzag, "axis_progressing_zigzag_wall_chain"
    return direct, "no_compatible_wall_family"


def _offset_edge_in_surface_tangent(points: np.ndarray, normals: np.ndarray, offset_mm: float) -> np.ndarray:
    tangent = points[1] - points[0]
    tangent_length = float(np.linalg.norm(tangent))
    if tangent_length <= 1e-9:
        raise ValueError("cannot offset a zero-length honeycomb fiber edge")
    tangent /= tangent_length
    lateral = np.cross(normals, tangent[None, :])
    lateral_lengths = np.linalg.norm(lateral, axis=1)
    if np.any(lateral_lengths <= 1e-9):
        raise ValueError("cannot derive a surface-tangent double-wall offset")
    return np.asarray(points, dtype=np.float64) + lateral / lateral_lengths[:, None] * offset_mm


def _raise_resin_and_travel_z_after_fiber_interfaces(
    source_job: ExternalSourceJob,
    *,
    selected_layers: tuple[int, ...],
    fiber_layer_height_mm: float,
) -> None:
    for group in [*source_job.material_paths, *source_job.travel_paths]:
        offset = sum(interface < int(group.layer_index) for interface in selected_layers) * fiber_layer_height_mm
        if offset == 0.0:
            continue
        group.paths = [np.asarray(path, dtype=np.float64).copy() for path in group.paths]
        for path in group.paths:
            path[:, 2] += offset


def _source_job_max_z(source_job: ExternalSourceJob) -> float:
    maxima = [float(np.max(np.asarray(path, dtype=np.float64)[:, 2])) for group in source_job.material_paths for path in group.paths if len(path)]
    if not maxima:
        raise ValueError("fiber planner cannot determine source-job Z extent")
    return max(maxima)


def _empty_result_from_height(
    source_job: ExternalSourceJob,
    *,
    reason: str,
    settings: MixedWallFiberSettings,
) -> FiberReinforcementResult:
    height = _source_job_max_z(source_job)
    return FiberReinforcementResult(
        enabled=False,
        reserved=False,
        resin_layer_indices=(),
        paths_per_layer=0,
        fiber_layer_height_mm=0.0,
        nominal_final_height_mm=height,
        total_path_count=0,
        report={
            "enabled": False,
            "reserved": False,
            "mode": "uniform_mixed_wall_v1",
            "configured_in": "main_ui_runtime",
            "reason": reason,
            "primary_axis_source": "automatic_longest_planar_span",
            "after_resin_physical_layers": [settings.first_after_resin_layer_physical, settings.last_after_resin_layer_physical],
            "paths_per_fiber_layer": 0,
            "total_fiber_path_count": 0,
            "automatic_resin_z_raise": False,
            "core_material_paths": "none",
            "nominal_final_height_mm": height,
        },
    )


def _empty_result(spec: ConformalLatticeSpec, reason: str) -> FiberReinforcementResult:
    height = float(spec.part.get("final_height_mm", 0.0))
    return FiberReinforcementResult(
        enabled=False,
        reserved=False,
        resin_layer_indices=(),
        paths_per_layer=0,
        fiber_layer_height_mm=0.0,
        nominal_final_height_mm=height,
        total_path_count=0,
        report={
            "enabled": False,
            "reserved": False,
            "mode": "reserved_future_path_v1",
            "path_generation": "disabled_pending_replacement",
            "reason": reason,
            "paths_per_fiber_layer": 0,
            "total_fiber_path_count": 0,
            "automatic_resin_z_raise": False,
            "core_material_paths": "none",
            "nominal_final_height_mm": height,
        },
    )
