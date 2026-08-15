"""Apply the project-owned Legacy resin planning policy to a G-code SourceJob.

The Prusa bridge remains responsible for slicing and emitting G-code.  This
module deliberately runs *after* that G-code has been parsed, so Core receives
one source representation while the project retains its established safe
infill-continuity policy.
"""

from __future__ import annotations

import importlib
import math

import numpy as np

from .honeycomb_pathing import HoleSafeTravelRouter, solid_geometry_at_z
from .slicer import (
    SliceConfig,
    _connect_brim_paths_one_stroke,
    _connect_zigzag_infill_paths,
    _legacy_path_merge_tolerance,
    merge_adjacent_connected_paths,
    optimize_triangle_infill_travel,
)
from .stl_io import Mesh


_ZIGZAG_PATTERNS = {
    "rectilinear", "zigzag", "isotropic", "zigzag_horizontal",
    "zigzag_vertical", "zigzag_plus45", "zigzag_minus45",
}
_TERMINAL_LOOP_MAX_RADIUS_MM = 0.5
_TERMINAL_LOOP_MAX_SUFFIX_POINTS = 12


def apply_legacy_resin_optimization(job, planning_mesh: Mesh, config: SliceConfig):
    """Return a SourceJob with Legacy infill optimization applied.

    ``planning_mesh`` must use the same oriented, placement-restored frame as
    ``job``. Native ``infill`` blocks are rewritten by the Legacy continuity policy.
    When the UI enables Brim one-stroke, native ``brim`` blocks use the same
    safe boundary connector in this G-code representation. Perimeters and all
    unrelated Prusa travel records are kept. New infill inter-trail travel is
    produced by ``HoleSafeTravelRouter`` rather than by drawing an unchecked
    chord through a void.
    """

    # Keep the SourceJob contract boundary runtime-only.  The slicer package
    # owns this G-code-local transformation; Core/export packages are not an
    # import-time dependency of it.
    source_npz = importlib.import_module("external_npz_preprocessor.source_npz")
    LayerPaths = source_npz.LayerPaths
    MaterialPath = source_npz.MaterialPath
    SourceJob = source_npz.SourceJob
    TravelPath = source_npz.TravelPath

    roles_by_layer = job.meta.get("path_roles", {}).get("R", {})
    original_order = job.meta.get("motion_order", {})
    if not isinstance(roles_by_layer, dict) or not isinstance(original_order, dict):
        return job

    updated_layers = []
    updated_order = dict(original_order)
    changed_layers: list[int] = []
    brim_changed_layers: list[int] = []
    terminal_loop_trim_layers: list[int] = []
    for layer in job.layers:
        roles = roles_by_layer.get(str(layer.index), [])
        motions = original_order.get(str(layer.index), [])
        if not isinstance(roles, list) or not isinstance(motions, list):
            updated_layers.append(layer)
            continue
        candidate_layer = layer
        candidate_roles = roles
        candidate_motions = motions
        changed = False
        if config.brim_one_stroke:
            brim_result = _optimize_brim_layer(
                candidate_layer,
                candidate_roles,
                candidate_motions,
                config,
                LayerPaths=LayerPaths,
                MaterialPath=MaterialPath,
            )
            if brim_result is not None:
                candidate_layer, candidate_roles, candidate_motions = brim_result
                changed = True
                brim_changed_layers.append(layer.index)
        trim_result = _trim_terminal_infill_loops(
            candidate_layer,
            candidate_roles,
            candidate_motions,
            planning_mesh,
            config,
            LayerPaths=LayerPaths,
            MaterialPath=MaterialPath,
            TravelPath=TravelPath,
        )
        if trim_result is not None:
            candidate_layer, candidate_motions = trim_result
            changed = True
            terminal_loop_trim_layers.append(layer.index)
        result = _optimize_layer(
            candidate_layer,
            candidate_roles,
            candidate_motions,
            planning_mesh,
            config,
            LayerPaths=LayerPaths,
            MaterialPath=MaterialPath,
            TravelPath=TravelPath,
        )
        if result is not None:
            candidate_layer, candidate_roles, candidate_motions = result
            changed = True
        if not changed:
            updated_layers.append(layer)
            continue
        updated_layers.append(candidate_layer)
        roles_by_layer = dict(roles_by_layer)
        roles_by_layer[str(layer.index)] = candidate_roles
        updated_order[str(layer.index)] = candidate_motions
        changed_layers.append(layer.index)

    if not changed_layers:
        return job
    meta = dict(job.meta)
    meta["path_roles"] = {**dict(job.meta.get("path_roles", {})), "R": roles_by_layer}
    meta["motion_order"] = updated_order
    meta["resin_source"] = "prusa_gcode_legacy_postprocess"
    meta["legacy_resin_optimized_layers"] = changed_layers
    meta["legacy_brim_optimized_layers"] = brim_changed_layers
    meta["prusa_terminal_infill_loop_trim_layers"] = terminal_loop_trim_layers
    return SourceJob(meta=meta, layers=updated_layers)


def _trim_terminal_infill_loops(
    layer,
    roles,
    motions,
    mesh,
    config,
    *,
    LayerPaths,
    MaterialPath,
    TravelPath,
):
    """Delete Prusa's tiny terminal infill loops and rebuild only their travel.

    The source G-code parser stores a positive-extrusion chain and the
    following empty move separately.  Trimming a deposited suffix therefore
    requires replacing that particular empty move so its start is the new
    material endpoint.  No perimeter, Brim, Fiber, or Core-owned data is
    touched.  A trim is skipped when a hole-safe replacement travel cannot be
    found, rather than leaving a disconnected motion sequence.
    """

    infill_indexes = [index for index, role in enumerate(roles) if role == "infill"]
    if not infill_indexes or not isinstance(motions, list):
        return None
    if any(index >= len(layer.resin_paths) for index in infill_indexes):
        return None
    try:
        z = float(layer.resin_paths[infill_indexes[0]].points[0, 2])
        solid = solid_geometry_at_z(mesh, z, float(config.tolerance))
    except (IndexError, ValueError):
        return None
    if solid.is_empty:
        return None

    suffix_counts = {
        index: _terminal_small_loop_point_count(layer.resin_paths[index].points)
        for index in infill_indexes
    }
    suffix_counts = {index: count for index, count in suffix_counts.items() if count is not None}
    if not suffix_counts:
        return None

    deposit_positions = {
        int(motion["index"]): position
        for position, motion in enumerate(motions)
        if motion.get("kind") == "deposit" and isinstance(motion.get("index"), int)
    }
    available = {index for index in suffix_counts if index in deposit_positions}
    if not available:
        return None
    source_infill = [
        np.asarray(layer.resin_paths[index].points, dtype=np.float64)
        for index in infill_indexes
    ]
    spacing = _infill_spacing(source_infill, config)
    router = HoleSafeTravelRouter(
        solid,
        {},
        spacing_mm=max(spacing * 0.5, float(config.tolerance) * 10.0),
    )

    trimmed_points = {}
    routes_after_deposit: dict[int, list[tuple[float, float]]] = {}
    skipped = set()
    for index in sorted(available):
        path = np.asarray(layer.resin_paths[index].points, dtype=np.float64)
        count = suffix_counts[index]
        candidate = path[: -count + 1].copy()
        if candidate.shape[0] < 2:
            skipped.add(index)
            continue
        position = deposit_positions[index]
        next_deposit_position = next(
            (
                following
                for following in range(position + 1, len(motions))
                if motions[following].get("kind") == "deposit"
                and isinstance(motions[following].get("index"), int)
            ),
            None,
        )
        if next_deposit_position is not None:
            next_index = int(motions[next_deposit_position]["index"])
            if not 0 <= next_index < len(layer.resin_paths):
                skipped.add(index)
                continue
            destination = np.asarray(layer.resin_paths[next_index].points, dtype=np.float64)[0, :2]
            route = router.route(tuple(candidate[-1, :2]), tuple(destination))
            if route is None or len(route) < 2:
                skipped.add(index)
                continue
            routes_after_deposit[position] = route
        trimmed_points[index] = candidate
    for index in skipped:
        trimmed_points.pop(index, None)
        routes_after_deposit.pop(deposit_positions[index], None)
    if not trimmed_points:
        return None

    resin_paths = []
    for index, path in enumerate(layer.resin_paths):
        points = trimmed_points.get(index)
        if points is None:
            resin_paths.append(path)
            continue
        # E is stored per source material chain.  Keeping the exact prefix of
        # Prusa's cumulative E samples removes precisely the extrusion that
        # belonged to the deleted arc without modifying subsequent chains.
        extrusion = None if path.extrusion is None else np.asarray(path.extrusion, dtype=np.float64)[: len(points)].copy()
        resin_paths.append(MaterialPath(path.material, len(resin_paths), points, extrusion))

    skipped_travel_positions = set()
    for position in routes_after_deposit:
        following = next(
            (
                index
                for index in range(position + 1, len(motions))
                if motions[index].get("kind") == "deposit"
            ),
            len(motions),
        )
        skipped_travel_positions.update(
            index
            for index in range(position + 1, following)
            if motions[index].get("kind") == "travel"
        )

    travel_paths = []
    travel_index_map = {}
    rewritten_motions = []
    for position, motion in enumerate(motions):
        kind, index = motion.get("kind"), motion.get("index")
        if kind == "deposit" and isinstance(index, int):
            rewritten_motions.append({"kind": "deposit", "index": index})
            route = routes_after_deposit.get(position)
            if route is not None:
                reference = resin_paths[index].points
                travel_paths.append(
                    TravelPath(len(travel_paths), _route_points(route, z, reference))
                )
                rewritten_motions.append({"kind": "travel", "index": len(travel_paths) - 1})
        elif kind == "travel" and isinstance(index, int):
            if position in skipped_travel_positions or not 0 <= index < len(layer.travel_paths):
                continue
            mapped = travel_index_map.get(index)
            if mapped is None:
                mapped = len(travel_paths)
                travel_index_map[index] = mapped
                original = layer.travel_paths[index]
                travel_paths.append(TravelPath(mapped, np.asarray(original.points, dtype=np.float64).copy()))
            rewritten_motions.append({"kind": "travel", "index": mapped})

    return (
        LayerPaths(layer.index, resin_paths, list(layer.fiber_paths), travel_paths),
        rewritten_motions,
    )


def _terminal_small_loop_point_count(points: np.ndarray) -> int | None:
    """Return the terminal Prusa loop's point count, or ``None`` when absent."""

    xy = np.asarray(points[:, :2], dtype=np.float64)
    for count in range(min(len(xy), _TERMINAL_LOOP_MAX_SUFFIX_POINTS), 3, -1):
        suffix = xy[-count:]
        vectors = np.diff(suffix, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths <= 1e-5):
            continue
        turns = np.degrees(np.arctan2(
            vectors[:-1, 0] * vectors[1:, 1] - vectors[:-1, 1] * vectors[1:, 0],
            np.sum(vectors[:-1] * vectors[1:], axis=1),
        ))
        turns = turns[np.abs(turns) >= 2.0]
        if len(turns) < 2 or np.any(np.sign(turns) != np.sign(turns[0])):
            continue
        radius, rms = _fit_circle(suffix)
        if (
            0.05 <= radius <= _TERMINAL_LOOP_MAX_RADIUS_MM
            and rms <= max(0.03, radius * 0.04)
        ):
            return count
    return None


def _fit_circle(points: np.ndarray) -> tuple[float, float]:
    x, y = points[:, 0], points[:, 1]
    center_x, center_y, constant = np.linalg.lstsq(
        np.column_stack((2.0 * x, 2.0 * y, np.ones(len(points)))),
        x * x + y * y,
        rcond=None,
    )[0]
    radius_squared = constant + center_x * center_x + center_y * center_y
    if radius_squared <= 0.0:
        return math.inf, math.inf
    radius = math.sqrt(radius_squared)
    radial = np.linalg.norm(points - np.asarray([center_x, center_y]), axis=1)
    return radius, float(np.sqrt(np.mean((radial - radius) ** 2)))


def _optimize_brim_layer(layer, roles, motions, config, *, LayerPaths, MaterialPath):
    """Apply the accepted Brim connector without leaving the G-code boundary.

    A returned component is accepted only after each source Brim stroke can be
    matched back to an endpoint on that component. This preserves distinct
    islands and their original travel instead of guessing an unsafe bridge.
    """

    brim_indexes = [index for index, role in enumerate(roles) if role == "brim"]
    if len(brim_indexes) < 2:
        return None
    source_paths = [layer.resin_paths[index] for index in brim_indexes]
    connected = _connect_brim_paths_one_stroke(
        [np.asarray(path.points, dtype=np.float64) for path in source_paths],
        float(config.line_width),
        max(float(config.tolerance), 1e-6),
    )
    if not connected:
        return None

    endpoint_tolerance = max(float(config.tolerance) * 50.0, 1e-4)
    remaining = set(brim_indexes)
    components: list[tuple[list[int], np.ndarray]] = []
    for merged_xy in connected:
        merged_xy = np.asarray(merged_xy, dtype=np.float64)
        if merged_xy.ndim != 2 or merged_xy.shape[0] < 2 or merged_xy.shape[1] < 2:
            return None
        matched = []
        for index in brim_indexes:
            if index not in remaining:
                continue
            points = np.asarray(layer.resin_paths[index].points, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 2:
                return None
            distances = np.linalg.norm(merged_xy[:, :2] - points[0, :2], axis=1)
            end_distances = np.linalg.norm(merged_xy[:, :2] - points[-1, :2], axis=1)
            if min(float(distances.min()), float(end_distances.min())) <= endpoint_tolerance:
                matched.append(index)
        if matched:
            matched.sort()
            remaining.difference_update(matched)
            components.append((matched, merged_xy[:, :2]))
    if remaining:
        return None

    merged_by_representative = {}
    component_by_index = {}
    for source_indexes, merged_xy in components:
        representative = source_indexes[0]
        component_by_index.update({index: representative for index in source_indexes})
        if len(source_indexes) == 1:
            merged_by_representative[representative] = layer.resin_paths[representative]
            continue
        originals = [layer.resin_paths[index] for index in source_indexes]
        rate = _median_extrusion_rate(originals)
        if rate is None:
            return None
        reference = np.asarray(originals[0].points, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[0] < 2:
            return None
        merged_points = np.column_stack((
            merged_xy,
            np.full(len(merged_xy), reference[0, 2]),
            np.tile(reference[0, 3:6], (len(merged_xy), 1)),
        ))
        merged_by_representative[representative] = _material_path(
            MaterialPath, merged_points, rate, float(reference[0, 2]), representative
        )

    brim_set = set(brim_indexes)
    new_resin = []
    new_roles = []
    old_to_new = {}
    for old_index, (path, role) in enumerate(zip(layer.resin_paths, roles)):
        if old_index in brim_set:
            representative = component_by_index[old_index]
            if old_index != representative:
                continue
            path = merged_by_representative[representative]
        old_to_new[old_index] = len(new_resin)
        new_resin.append(MaterialPath("R", len(new_resin), path.points, path.extrusion))
        new_roles.append(role)

    removed_travel = set()
    previous_deposit = None
    pending_travel = []
    for motion in motions:
        kind, index = motion.get("kind"), motion.get("index")
        if kind == "travel" and isinstance(index, int):
            pending_travel.append(index)
        elif kind == "deposit" and isinstance(index, int):
            if (
                previous_deposit in brim_set
                and index in brim_set
                and component_by_index.get(previous_deposit) == component_by_index.get(index)
            ):
                removed_travel.update(pending_travel)
            pending_travel = []
            previous_deposit = index
    new_travel = []
    travel_to_new = {}
    for old_index, path in enumerate(layer.travel_paths):
        if old_index in removed_travel:
            continue
        travel_to_new[old_index] = len(new_travel)
        new_travel.append(path)

    new_motions = []
    emitted_components = set()
    for motion in motions:
        kind, index = motion.get("kind"), motion.get("index")
        if not isinstance(index, int):
            continue
        if kind == "deposit":
            if index in brim_set:
                representative = component_by_index[index]
                if representative in emitted_components:
                    continue
                emitted_components.add(representative)
                index = representative
            if index in old_to_new:
                new_motions.append({"kind": "deposit", "index": old_to_new[index]})
        elif kind == "travel" and index in travel_to_new:
            new_motions.append({"kind": "travel", "index": travel_to_new[index]})
    return LayerPaths(layer.index, new_resin, list(layer.fiber_paths), new_travel), new_roles, new_motions


def _optimize_layer(layer, roles, motions, mesh, config, *, LayerPaths, MaterialPath, TravelPath):
    infill_indexes = [index for index, role in enumerate(roles) if role == "infill"]
    if len(infill_indexes) < 2:
        return None
    infill_set = set(infill_indexes)
    positions = [
        position for position, motion in enumerate(motions)
        if motion.get("kind") == "deposit" and motion.get("index") in infill_set
    ]
    if len(positions) != len(infill_indexes):
        return None
    first, last = min(positions), max(positions)
    if any(
        motion.get("kind") == "deposit" and motion.get("index") not in infill_set
        for motion in motions[first : last + 1]
    ):
        return None
    try:
        z = float(layer.resin_paths[infill_indexes[0]].points[0, 2])
        solid = solid_geometry_at_z(mesh, z, float(config.tolerance))
    except (IndexError, ValueError):
        return None
    if solid.is_empty:
        return None

    source_paths = [np.asarray(layer.resin_paths[index].points, dtype=np.float64) for index in infill_indexes]
    spacing = _infill_spacing(source_paths, config)
    if spacing <= config.tolerance:
        return None
    if config.infill_pattern in _ZIGZAG_PATTERNS and config.zigzag_path_optimization:
        optimized = _connect_zigzag_infill_paths(
            source_paths, solid, spacing, spacing, float(config.tolerance),
            solid_bead_width=float(config.line_width),
            follow_boundaries=bool(config.print_perimeters),
        )
        optimized = merge_adjacent_connected_paths(
            optimized, _legacy_path_merge_tolerance(config.line_width, config.tolerance)
        )
    elif config.infill_pattern == "triangles" and config.triangle_path_optimization:
        optimized = optimize_triangle_infill_travel(source_paths, float(config.tolerance))
        optimized = merge_adjacent_connected_paths(
            optimized, _legacy_path_merge_tolerance(config.line_width, config.tolerance)
        )
    else:
        return None
    if not optimized:
        return None

    ordered, connector_routes = _order_with_hole_safe_travel(optimized, solid, spacing, config.tolerance)
    if not ordered:
        return None
    rate = _median_extrusion_rate([layer.resin_paths[index] for index in infill_indexes])
    if rate is None:
        return None
    rebuilt = [_material_path(MaterialPath, path, rate, z, index) for index, path in enumerate(ordered)]

    first_infill = min(infill_indexes)
    path_index_map: dict[int, int] = {}
    new_resin = []
    new_roles = []
    for old_index, (path, role) in enumerate(zip(layer.resin_paths, roles)):
        if old_index in infill_set:
            if old_index != first_infill:
                continue
            for rebuilt_path in rebuilt:
                path_index_map_marker = len(new_resin)
                new_resin.append(MaterialPath("R", path_index_map_marker, rebuilt_path.points, rebuilt_path.extrusion))
                new_roles.append("infill")
            continue
        path_index_map[old_index] = len(new_resin)
        new_resin.append(MaterialPath(path.material, len(new_resin), path.points, path.extrusion))
        new_roles.append(role)

    new_travel = list(layer.travel_paths)
    optimizer_travel_indexes = []
    for route in connector_routes:
        if len(route) < 2:
            continue
        points = _route_points(route, z, ordered[0])
        optimizer_travel_indexes.append(len(new_travel))
        new_travel.append(TravelPath(len(new_travel), points))

    rebuilt_indexes = list(range(path_index_map.get(first_infill, 0), path_index_map.get(first_infill, 0) + len(rebuilt)))
    if not rebuilt_indexes:
        rebuilt_indexes = list(range(len(new_resin) - len(rebuilt), len(new_resin)))
    rewritten_motions = []
    inserted = False
    for position, motion in enumerate(motions):
        kind, index = motion.get("kind"), motion.get("index")
        if first <= position <= last and kind in {"deposit", "travel"}:
            if not inserted:
                for entry, rebuilt_index in enumerate(rebuilt_indexes):
                    if entry:
                        rewritten_motions.append({"kind": "travel", "index": optimizer_travel_indexes[entry - 1]})
                    rewritten_motions.append({"kind": "deposit", "index": rebuilt_index})
                inserted = True
            continue
        if kind == "deposit" and isinstance(index, int) and index in path_index_map:
            rewritten_motions.append({"kind": "deposit", "index": path_index_map[index]})
        elif kind == "travel" and isinstance(index, int) and 0 <= index < len(new_travel):
            rewritten_motions.append({"kind": "travel", "index": index})
    return LayerPaths(layer.index, new_resin, list(layer.fiber_paths), new_travel), new_roles, rewritten_motions


def _infill_spacing(paths, config: SliceConfig) -> float:
    fallback = float(config.line_width) / max(float(config.infill_density) / 100.0, 1e-9)
    directions, centers = [], []
    for path in paths:
        delta = np.diff(path[:, :2], axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        if not np.any(lengths > config.tolerance):
            continue
        direction = delta[int(np.argmax(lengths))]
        directions.append(direction / np.linalg.norm(direction))
        centers.append(np.mean(path[:, :2], axis=0))
    if len(directions) < 2:
        return fallback
    reference = directions[0]
    direction = np.mean([item if np.dot(item, reference) >= 0 else -item for item in directions], axis=0)
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= config.tolerance:
        return fallback
    levels = np.sort(np.asarray(centers) @ np.asarray([-direction[1], direction[0]]) / magnitude)
    deltas = np.diff(levels)
    useful = deltas[deltas >= max(config.tolerance * 20.0, fallback * 0.1)]
    return float(np.median(useful)) if useful.size else fallback


def _order_with_hole_safe_travel(paths, solid, spacing, tolerance):
    remaining = [np.asarray(path, dtype=np.float64) for path in paths]
    if not remaining:
        return [], []
    router = HoleSafeTravelRouter(solid, {}, spacing_mm=max(spacing * 0.5, tolerance * 10.0))
    ordered, routes = [remaining.pop(0)], []
    while remaining:
        start = ordered[-1][-1, :2]
        candidates = []
        for index, path in enumerate(remaining):
            for reverse in (False, True):
                oriented = path[::-1].copy() if reverse else path
                route = router.route(tuple(start), tuple(oriented[0, :2]))
                if route is not None:
                    candidates.append((sum(math.dist(a, b) for a, b in zip(route, route[1:])), index, oriented, route))
        if not candidates:
            return [], []
        _, index, path, route = min(candidates, key=lambda item: item[0])
        remaining.pop(index)
        ordered.append(path)
        routes.append(route)
    return ordered, routes


def _median_extrusion_rate(paths):
    rates = []
    for path in paths:
        if path.extrusion is None or len(path.points) < 2:
            continue
        length = float(np.linalg.norm(np.diff(path.points[:, :3], axis=0), axis=1).sum())
        delta = float(path.extrusion[-1] - path.extrusion[0])
        if length > 1e-9 and delta >= 0:
            rates.append(delta / length)
    return float(np.median(rates)) if rates else None


def _material_path(MaterialPath, xy_path, rate, z, order):
    values = np.asarray(xy_path, dtype=np.float64)
    if values.shape[1] >= 6:
        points = values.copy()
    else:
        points = np.column_stack((values[:, :2], np.full(len(values), z), np.zeros((len(values), 3))))
    lengths = np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1)
    extrusion = np.concatenate(([0.0], np.cumsum(lengths * rate)))
    return MaterialPath("R", order, points, extrusion)


def _route_points(route, z, reference):
    abc = np.asarray(reference[0, 3:6], dtype=np.float64) if reference.shape[1] >= 6 else np.zeros(3)
    return np.asarray([[x, y, z, *abc] for x, y in route], dtype=np.float64)
