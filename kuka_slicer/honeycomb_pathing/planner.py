"""STL-preserving honeycomb path planning.

This is an *adapter* around a completed Prusa slice, not a replacement for
PrusaSlicer. Deposited honeycomb walls are retained verbatim. Every move
between two non-incident wall regions remains an explicit, hole-safe travel
path instead of a zero-E gap hidden inside a material path.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np
from shapely.geometry import LineString, MultiPoint, Polygon
from shapely.ops import unary_union, voronoi_diagram

from ..external_npz import ExternalSourceJob, MaterialPaths, TravelPaths
from ..stl_io import Mesh
from .config import HoneycombTopology
from .closed_hexagon import build_closed_hexagon_double_wall_honeycomb
from .travel_router import HoleSafeTravelRouter, route_length


_NODE_DECIMALS = 5
_MIN_EDGE_MM = 1e-3


@dataclass(frozen=True, slots=True)
class _Edge:
    a: tuple[float, float]
    b: tuple[float, float]
    length: float


@dataclass(frozen=True, slots=True)
class _EulerRegion:
    """One closed, wall-only traversal of a native STL graph component."""

    traversals: tuple[tuple[tuple[float, float], tuple[float, float], int], ...]
    edge_visits: dict[int, int]
    repeated_length_mm: float


def apply_honeycomb_centerline_pathing(
    job: ExternalSourceJob,
    mesh: Mesh,
    *,
    line_width_mm: float,
    tolerance_mm: float,
    topology: HoneycombTopology = "native_single_motion",
) -> ExternalSourceJob:
    """Preserve native honeycomb walls and emit a partitioned path plan.

    The outer frame is regenerated from the STL section, while every internal
    deposited path remains the native Prusa geometry. True shared endpoints
    receive a smooth three-way flow taper. Non-incident regions are never
    joined by a material or zero-E pseudo-material segment: their shortest
    hole-safe connector is emitted as ``T`` travel.
    """

    if line_width_mm <= 0 or tolerance_mm <= 0:
        raise ValueError("honeycomb path planning requires positive width and tolerance")
    resin_groups = sorted(
        (group for group in job.material_paths if group.material == "R" and group.paths),
        key=lambda group: group.layer_index,
    )
    if not resin_groups:
        raise ValueError("honeycomb path planning requires native resin paths")

    reference_z = _layer_z(resin_groups[0])
    solid = _solid_geometry_at_z(mesh, reference_z, tolerance_mm)
    # Native Prusa labels every disconnected honeycomb-wall ring as an outer
    # contour.  Derive the frame from the STL section instead of selecting the
    # largest native ring, which can otherwise be only one honeycomb cell.
    frame = _outer_frame_from_solid(solid, reference_z)
    native_travel_groups = {group.layer_index: group for group in job.travel_paths}
    if topology == "closed_hexagon_double_wall":
        pattern = build_closed_hexagon_double_wall_honeycomb(
            tuple(float(value) for value in solid.bounds),
            cell_side_mm=_estimate_honeycomb_cell_side(solid, line_width_mm),
            line_width_mm=line_width_mm,
        )
        # Each generated loop is the exact boundary of one closed hexagonal
        # void.  The router uses this *replacement* void geometry, rather than
        # the source STL holes, so every non-deposition move is also checked
        # against the new topology.
        generated_solid = _solid_from_closed_hexagons(frame, pattern.loops)
        routing = _closed_loop_routing_graph(frame, pattern.loops)
        travel_router = HoleSafeTravelRouter(
            generated_solid,
            routing,
            spacing_mm=min(1.0, line_width_mm * 0.5),
        )
        ordered_template, travel_template = _order_closed_loops(
            frame,
            pattern.loops,
            travel_router,
        )
        topology_meta = {
            "topology": topology,
            "source": "post_prusa_outer_envelope",
            "topology_change": "closed honeycomb regenerated as strict regular hexagons",
            "original_holes": "replaced by regular closed hexagonal voids",
            "wall_edge_count": pattern.cell_count * 6,
            "minimum_trail_count": pattern.cell_count,
            "layer_path_count": pattern.cell_count + 1,
            "closed_hexagon_cell_count": pattern.cell_count,
            "closed_hexagon_row_count": pattern.row_count,
            "closed_hexagon_nominal_cell_side_mm": pattern.nominal_cell_side_mm,
            "closed_hexagon_centreline_side_mm": pattern.centreline_hex_side_mm,
            "wall_tracks_per_internal_rib": 2,
            "wall_track_spacing_mm": round(pattern.wall_track_spacing_mm, 6),
            "wall_track_controlled_overlap_mm": round(pattern.wall_track_overlap_mm, 6),
            "travel_count": len(travel_template),
            "travel_length_mm_per_layer": round(
                sum(route_length(route) for route in travel_template), 6
            ),
        }
    elif topology == "closed_minimum_trails":
        # The wall graph comes directly from the source STL section: the
        # actual honeycomb-hole centroids define its Voronoi centreline and
        # the graph is clipped to the STL solid.  Prusa is used only for the
        # nominal extrusion rate, never for wall XY geometry or connectivity.
        wall_edges = _stl_honeycomb_wall_edges(solid, tolerance_mm)
        # The rectangle is a standalone first deposition path.  It must not
        # be noded into, repeated by, or reoriented with the honeycomb graph.
        edges = wall_edges
        euler_regions, euler_meta = _eulerize_native_wall_components(edges)
        # Routing edges are deliberately separate from material edges.  Each
        # clipped boundary wall component is attached to the outer frame with
        # a verified-solid connector, so travel can use the perimeter instead
        # of falling back to a dense grid or cutting across a honeycomb void.
        routing, routing_meta = _stl_wall_routing_graph(edges, frame, solid)
        travel_router = HoleSafeTravelRouter(
            solid,
            routing,
            spacing_mm=min(1.0, line_width_mm * 0.5),
        )
        ordered_euler_regions, euler_travel_template = _order_euler_regions(
            frame,
            euler_regions,
            travel_router,
        )
        junctions = _junction_nodes(edges)
        topology_meta = {
            "topology": "stl_honeycomb_eulerized_regions",
            "source": "source_stl_hole_topology",
            "topology_change": "none; wall graph is derived from and clipped to the source STL section",
            "wall_edge_count": len(wall_edges),
            "outer_frame_policy": "one closed standalone first deposition path",
            "native_deposition_stroke_count": 0,
            **euler_meta,
            "junction_count": len(junctions),
            "junction_flow_scale": round(1.0 / 3.0, 6),
            "connector_extrusion": "none; inter-region moves are explicit T travel",
            **routing_meta,
        }
    else:
        # Do not reconstruct the honeycomb from centroids or Voronoi cells.
        # The native Prusa job is the authority for both the source topology
        # and its pre-optimised sequence of individual wall strokes.
        reference_frame = frame.copy()
        reference_frame[:, 2] = reference_z
        reference_native_paths = _native_paths_without_outer_frame(
            resin_groups[0],
            reference_frame,
            tolerance_mm=tolerance_mm,
        )
        travel_router = HoleSafeTravelRouter(
            solid,
            _native_routing_graph(
                reference_frame,
                reference_native_paths,
                native_travel_groups.get(resin_groups[0].layer_index),
                solid,
                tolerance_mm=tolerance_mm,
            ),
            spacing_mm=min(1.0, line_width_mm * 0.5),
        )
        ordered_template = None
        travel_template = None
        topology_meta = {
            "topology": "native_partitioned_trails",
            "source": "completed_prusa_honeycomb_paths",
            "topology_change": "none; native Prusa honeycomb paths retained verbatim",
            "connector_extrusion": "none; inter-region moves are explicit T travel",
        }
    material_paths: list[MaterialPaths] = []
    travel_paths: list[TravelPaths] = []
    roles: dict[str, list[str]] = {}
    motions: dict[str, list[dict[str, object]]] = {}
    endpoint_compensation: dict[str, object] | None = None
    for group in resin_groups:
        z = _layer_z(group)
        frame_at_z = frame.copy()
        frame_at_z[:, 2] = z
        if topology == "closed_hexagon_double_wall":
            ordered = ordered_template
            connecting_travel = travel_template
            paths = [frame_at_z, *[_trail_to_path(trail, z) for trail in ordered]]
        elif topology == "closed_minimum_trails":
            native_junctions = junctions
            # The vertical STL extrusion has the same XY section at every
            # resin layer, so reuse the already-validated direct-STL Euler
            # graph rather than recomputing its shortest wall retraces 20
            # times.
            layer_regions, layer_meta = ordered_euler_regions, euler_meta
            ordered_regions, connecting_travel = layer_regions, euler_travel_template
            paths, extrusion = _eulerized_region_profiles(
                frame_at_z,
                ordered_regions,
                z=float(frame_at_z[0, 2]),
                rate=_native_e_per_mm(group, tolerance_mm),
                junctions=native_junctions,
                tolerance_mm=tolerance_mm,
                taper_length_mm=line_width_mm,
            )
            ordered = [path[:, :2].tolist() for path in paths[1:]]
        else:
            native_paths = _native_paths_without_outer_frame(
                group,
                frame_at_z,
                tolerance_mm=tolerance_mm,
            )
            native_paths, connecting_travel = _order_native_paths_for_continuous_motion(
                frame_at_z,
                native_paths,
                router=travel_router,
            )
            paths = [frame_at_z, *[path for _index, path in native_paths]]
            ordered = [path[:, :2].tolist() for _index, path in native_paths]
        if topology == "closed_minimum_trails":
            taper_meta = _junction_taper_meta(native_junctions, line_width_mm)
        else:
            taper_plan, taper_meta = _endpoint_taper_plan(
                paths,
                tolerance_mm=tolerance_mm,
                line_width_mm=line_width_mm,
            )
            paths = _insert_endpoint_taper_points(paths, taper_plan)
            extrusion = _extrusion_profiles(
                paths,
                _native_e_per_mm(group, tolerance_mm),
                taper_plan=taper_plan,
            )
        if endpoint_compensation is None:
            endpoint_compensation = taper_meta
        if topology == "closed_hexagon_double_wall":
            material_paths.append(MaterialPaths(group.layer_index, "R", paths, extrusion))
            travel_paths.append(
                TravelPaths(group.layer_index, [_travel_to_path(path, z) for path in connecting_travel])
            )
            roles[str(group.layer_index)] = ["outer_contour", *("honeycomb_wall" for _ in ordered)]
            records: list[dict[str, object]] = [{"kind": "deposit", "index": 0}]
            for index, _trail in enumerate(ordered, start=1):
                records.append({"kind": "travel", "index": index - 1})
                records.append({"kind": "deposit", "index": index})
            motions[str(group.layer_index)] = records
            continue

        if topology == "closed_minimum_trails":
            # A physical honeycomb edge must never be reclassified as a
            # zero-extrusion connector merely to reduce the NPZ path count.
            # Keep every Euler trail as an explicit deposited path and every
            # transition as a separately visible, hole-safe travel.  This
            # preserves the STL wall topology in both the preview and the
            # manufactured bead; callers may batch these paths for Core
            # export, but must not merge them into pseudo-one-stroke motion.
            material_paths.append(MaterialPaths(group.layer_index, "R", paths, extrusion))
            if connecting_travel:
                travel_paths.append(
                    TravelPaths(group.layer_index, [_travel_to_path(path, z) for path in connecting_travel])
                )
            roles[str(group.layer_index)] = ["outer_contour", *("honeycomb_wall" for _ in ordered)]
            records = [{"kind": "deposit", "index": 0}]
            for index in range(1, len(paths)):
                records.append({"kind": "travel", "index": index - 1})
                records.append({"kind": "deposit", "index": index})
            motions[str(group.layer_index)] = records
            if group is resin_groups[0]:
                topology_meta.update(
                    {
                        "layer_path_count": len(paths),
                        "motion_path_count": len(paths),
                        "travel_count": len(connecting_travel),
                        "travel_length_mm_per_layer": round(
                            sum(route_length(path) for path in connecting_travel), 6
                        ),
                        **layer_meta,
                        "intra_region_zero_e_connector_count": 0,
                        "intra_region_zero_e_connector_length_mm_per_layer": 0.0,
                    }
                )
            continue

        material_paths.append(MaterialPaths(group.layer_index, "R", paths, extrusion))
        travel_paths.append(
            TravelPaths(group.layer_index, [_travel_to_path(path, z) for path in connecting_travel])
        )
        roles[str(group.layer_index)] = ["outer_contour", *("honeycomb_wall" for _ in ordered)]
        records = [{"kind": "deposit", "index": 0}]
        for index, _trail in enumerate(ordered, start=1):
            records.append({"kind": "travel", "index": index - 1})
            records.append({"kind": "deposit", "index": index})
        motions[str(group.layer_index)] = records
        if group is resin_groups[0] and topology == "native_single_motion":
            topology_meta.update(
                {
                    "native_deposition_stroke_count": len(native_paths),
                    "layer_path_count": len(paths),
                    "motion_path_count": len(paths),
                    "travel_count": len(connecting_travel),
                    "connector_length_mm_per_layer": round(
                        sum(route_length(route) for route in connecting_travel), 6
                    ),
                }
            )

    job.material_paths = material_paths
    job.travel_paths = travel_paths
    job.native_gcode = None
    job.native_gcode_translation_mm = None
    job.meta["path_roles"] = {"R": roles}
    job.meta["motion_order"] = motions
    job.meta["honeycomb_centerline_pathing"] = {
        "format": "honeycomb_partitioned_trails_v1",
        "outer_frame": "stl_cross_section_outer_boundary_copied_first_to_every_layer",
        "wall_representation": (
            "strict closed regular hexagon loops; adjacent loops form each two-track wall"
            if topology == "closed_hexagon_double_wall"
            else "STL-section honeycomb-wall centreline reconstructed from the source void topology; Prusa supplies only the nominal extrusion rate"
            if topology == "closed_minimum_trails"
            else "native deposited paths from the completed Prusa slice; no wall geometry is reconstructed"
        ),
        "edge_policy": (
            "each original STL wall receives its nominal total material; an Euler retrace divides that wall budget across its visits"
            if topology == "closed_minimum_trails"
            else "each generated wall segment deposited exactly once"
        ),
        "trail_policy": (
            "one non-retracing closed loop per strict hexagonal void"
            if topology == "closed_hexagon_double_wall"
            else "Eulerized STL wall regions; only original STL-derived wall edges may be repeated"
            if topology == "closed_minimum_trails"
            else "native wall regions remain partitioned; only genuine shared endpoints may be continuous"
        ),
        "travel_policy": (
            "shortest direct no-hole connector; solid-grid fallback only when blocked; emitted as T travel"
            if topology != "closed_hexagon_double_wall"
            else "shortest hole-safe connector: direct solid, wall graph, then solid grid fallback"
        ),
        "travel_optimizer": (
            "actual-route nearest neighbour with the best loop entry side"
            if topology == "closed_hexagon_double_wall"
            else "STL wall regions are greedily ordered by the shortest hole-safe travel route"
            if topology == "closed_minimum_trails"
            else "native wall regions are greedily reoriented by shortest no-hole travel"
        ),
        "junction_flow_compensation": endpoint_compensation or {},
        **topology_meta,
    }
    slicing = job.meta.get("slicing")
    if isinstance(slicing, dict):
        slicing["path_planner"] = "honeycomb_centerline_post_prusa"
        slicing["native_gcode_reusable"] = False
    return job


def _native_paths_without_outer_frame(
    group: MaterialPaths,
    frame: np.ndarray,
    *,
    tolerance_mm: float,
) -> list[tuple[int, np.ndarray]]:
    """Return the native deposited strokes, omitting an already-present frame.

    Some Prusa profiles emit the rectangular exterior as one path while
    others split the honeycomb boundary into short open strokes.  Only a path
    geometrically equal to the regenerated STL frame is removed; all other
    native strokes, including boundary-adjacent honeycomb walls, remain.
    """

    result: list[tuple[int, np.ndarray]] = []
    for index, source_path in enumerate(group.paths):
        path = np.asarray(source_path, dtype=np.float64).copy()
        if path.shape[0] < 2:
            continue
        path[:, 2] = frame[0, 2]
        path = _dedupe_points(path)
        if _is_same_outer_frame(path, frame, tolerance_mm=tolerance_mm):
            continue
        result.append((index, path))
    return result


def _native_wall_edges(paths: Iterable[np.ndarray]) -> list[_Edge]:
    """Node the already-sliced wall strokes without changing their geometry.

    This exists solely to identify true three-way material junctions for the
    flow taper.  The returned graph is never used to regenerate or replace a
    deposition path, so clipped honeycomb cells remain exactly as in the STL
    slice all the way to the external frame.
    """

    lines = [
        LineString(np.asarray(path, dtype=np.float64)[:, :2])
        for path in paths
        if np.asarray(path).shape[0] >= 2
    ]
    if not lines:
        raise ValueError("native STL honeycomb slice has no wall strokes")
    noded = unary_union(lines)
    edges: dict[tuple[tuple[float, float], tuple[float, float]], _Edge] = {}
    for line in _lines(noded):
        coordinates = list(line.coords)
        for first, second in zip(coordinates, coordinates[1:]):
            a, b = _node(first), _node(second)
            length = math.dist(a, b)
            if length <= _MIN_EDGE_MM:
                continue
            key = (a, b) if a <= b else (b, a)
            edges[key] = _Edge(a, b, length)
    if not edges:
        raise ValueError("native STL honeycomb slice has no usable wall edges")
    return list(edges.values())


def _stl_honeycomb_wall_edges(solid, tolerance: float) -> list[_Edge]:
    """Build the honeycomb centreline only from the STL section geometry.

    The sites are the actual voids in the section, not Prusa perimeter or
    infill samples.  Intersecting the resulting centreline with ``solid``
    keeps every emitted edge inside printable material and outside the holes.
    """

    holes = [ring for polygon in _polygons(solid) for ring in polygon.interiors]
    if len(holes) < 3:
        raise ValueError("STL honeycomb graph requires at least three voids")
    # Edge cells in the STL are clipped by the rectangle.  Their centroids are
    # not honeycomb lattice centres and caused the former centreline to break
    # at the left/bottom/right edges.  Infer the lattice only from full holes,
    # then mirror that lattice outside the frame before clipping it back to
    # the STL solid.  This retains one wall centreline, not two hole outlines.
    hole_polygons = [Polygon(ring) for ring in holes]
    nominal_area = float(np.median([polygon.area for polygon in hole_polygons]))
    full_sites = [
        polygon.centroid
        for polygon in hole_polygons
        if polygon.area >= nominal_area * 0.98
    ]
    if len(full_sites) < 3:
        full_sites = [polygon.centroid for polygon in hole_polygons]
    min_x, min_y, max_x, max_y = solid.bounds
    rows: dict[float, list[float]] = defaultdict(list)
    for point in full_sites:
        rows[round(point.y, 6)].append(point.x)
    row_y = sorted(rows)
    y_step = float(np.median(np.diff(row_y)))
    x_step = float(np.median([
        np.median(np.diff(sorted(values)))
        for values in rows.values() if len(values) > 1
    ]))
    base_y = row_y[0]
    base_x = min(rows[base_y])
    first_row = math.floor((min_y - 2 * y_step - base_y) / y_step)
    last_row = math.ceil((max_y + 2 * y_step - base_y) / y_step)
    site_coordinates = set()
    for row in range(first_row, last_row + 1):
        y = base_y + row * y_step
        x0 = base_x + (row % 2) * (x_step * 0.5)
        first_column = math.floor((min_x - 2 * x_step - x0) / x_step)
        last_column = math.ceil((max_x + 2 * x_step - x0) / x_step)
        for column in range(first_column, last_column + 1):
            site_coordinates.add((round(x0 + column * x_step, 8), round(y, 8)))
    sites = MultiPoint(list(site_coordinates))
    try:
        cells = voronoi_diagram(sites, envelope=solid.envelope, tolerance=tolerance, edges=False)
    except ValueError:
        cells = voronoi_diagram(sites, envelope=solid.envelope, tolerance=0.0, edges=False)
    noded = unary_union(
        unary_union([polygon.boundary for polygon in _polygons(cells)]).intersection(solid)
    )
    edges: dict[tuple[tuple[float, float], tuple[float, float]], _Edge] = {}
    for line in _lines(noded):
        coordinates = list(line.coords)
        for first, second in zip(coordinates, coordinates[1:]):
            a, b = _node(first), _node(second)
            length = math.dist(a, b)
            if length <= _MIN_EDGE_MM:
                continue
            key = (a, b) if a <= b else (b, a)
            edges[key] = _Edge(a, b, length)
    if not edges:
        raise ValueError("STL honeycomb graph has no printable wall edges")
    return list(edges.values())


def _stl_outer_frame_edges(
    frame: np.ndarray,
    wall_edges: list[_Edge],
    tolerance: float,
) -> list[_Edge]:
    """Node the source outer contour at honeycomb endpoints lying on it."""

    boundary_endpoints = [
        node
        for edge in wall_edges
        for node in (edge.a, edge.b)
    ]
    result: list[_Edge] = []
    frame_xy = [(float(point[0]), float(point[1])) for point in frame]
    for first, second in zip(frame_xy, frame_xy[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length_sq = dx * dx + dy * dy
        if length_sq <= _MIN_EDGE_MM * _MIN_EDGE_MM:
            continue
        nodes = [first, second]
        for candidate in boundary_endpoints:
            parameter = ((candidate[0] - first[0]) * dx + (candidate[1] - first[1]) * dy) / length_sq
            if -1e-8 <= parameter <= 1.0 + 1e-8:
                projected = (first[0] + parameter * dx, first[1] + parameter * dy)
                if math.dist(candidate, projected) <= tolerance:
                    nodes.append(candidate)
        nodes = sorted({_node(node) for node in nodes}, key=lambda node: math.dist(first, node))
        for start, end in zip(nodes, nodes[1:]):
            edge_length = math.dist(start, end)
            if edge_length > _MIN_EDGE_MM:
                result.append(_Edge(start, end, edge_length))
    return result


def _eulerize_native_wall_components(
    edges: list[_Edge],
) -> tuple[list[_EulerRegion], dict[str, object]]:
    """Cover every native wall edge continuously, retracing only wall edges.

    Each connected component is made Eulerian by pairing odd-degree nodes with
    shortest paths *on the existing wall graph*.  The added instances are
    therefore repeated deposition on an original STL wall, never a bridge
    through a hole or a newly invented boundary connection.
    """

    adjacency: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        adjacency[edge.a].append(index)
        adjacency[edge.b].append(index)
    regions: list[_EulerRegion] = []
    repeated_edges = 0
    repeated_length = 0.0
    for nodes, edge_ids in _components(adjacency, edges):
        odd = {node for node in nodes if len(adjacency[node]) % 2}
        repeated_ids: list[int] = []
        while odd:
            source = min(odd)
            target, route = _nearest_odd_wall_route(
                source,
                odd - {source},
                adjacency,
                edges,
            )
            odd.remove(source)
            odd.remove(target)
            repeated_ids.extend(route)
        expanded_ids = [*edge_ids, *repeated_ids]
        expanded = [edges[index] for index in expanded_ids]
        circuit = _euler_edges(expanded, min(nodes))
        if len(circuit) != len(expanded):
            raise ValueError("native STL wall Eulerization did not cover every wall edge")
        traversals = tuple(
            (start, end, expanded_ids[edge_index])
            for start, end, edge_index in circuit
        )
        visits: dict[int, int] = defaultdict(int)
        for _start, _end, edge_id in traversals:
            visits[edge_id] += 1
        if any(visits[edge_id] < 1 for edge_id in edge_ids):
            raise ValueError("native STL wall Eulerization dropped a wall edge")
        regions.append(
            _EulerRegion(
                traversals=traversals,
                edge_visits=dict(visits),
                repeated_length_mm=sum(edges[index].length for index in repeated_ids),
            )
        )
        repeated_edges += len(repeated_ids)
        repeated_length += sum(edges[index].length for index in repeated_ids)
    regions.sort(key=lambda region: (-len(region.traversals), region.traversals[0][0]))
    return regions, {
        "eulerized_region_count": len(regions),
        "repeated_wall_edge_count": repeated_edges,
        "repeated_wall_length_mm_per_layer": round(repeated_length, 6),
        "repeat_material_policy": "each original wall edge keeps its nominal total extrusion; repeated traversals divide that edge budget equally",
    }


def _nearest_odd_wall_route(
    source: tuple[float, float],
    targets: set[tuple[float, float]],
    adjacency,
    edges: list[_Edge],
) -> tuple[tuple[float, float], list[int]]:
    """Return the shortest existing-wall route from one odd node to another."""

    queue: list[tuple[float, tuple[float, float]]] = [(0.0, source)]
    distance = {source: 0.0}
    previous: dict[tuple[float, float], tuple[tuple[float, float], int]] = {}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distance[node] + 1e-12:
            continue
        if node in targets:
            route: list[int] = []
            current = node
            while current != source:
                parent, edge_id = previous[current]
                route.append(edge_id)
                current = parent
            return node, list(reversed(route))
        for edge_id in adjacency[node]:
            edge = edges[edge_id]
            other = edge.b if edge.a == node else edge.a
            proposal = cost + edge.length
            if proposal + 1e-12 >= distance.get(other, math.inf):
                continue
            distance[other] = proposal
            previous[other] = (node, edge_id)
            heapq.heappush(queue, (proposal, other))
    raise ValueError("odd native STL wall node has no same-wall pairing route")


def _order_euler_regions(
    frame: np.ndarray,
    regions: list[_EulerRegion],
    router: HoleSafeTravelRouter,
) -> tuple[list[_EulerRegion], list[list[tuple[float, float]]]]:
    """Greedily order closed wall regions using only hole-safe travel."""

    remaining = list(regions)
    ordered: list[_EulerRegion] = []
    travels: list[list[tuple[float, float]]] = []
    current = (float(frame[-1, 0]), float(frame[-1, 1]))
    while remaining:
        candidates = sorted(
            (
                (math.dist(current, region.traversals[0][0]), index, region)
                for index, region in enumerate(remaining)
            ),
            key=lambda item: item[:2],
        )
        proposals = []
        # Most neighbouring wall regions are joined by a direct safe move.
        # Avoid running the expensive grid router against every tiny clipped
        # boundary component before considering the local candidates.
        for _distance, index, region in candidates[: min(24, len(candidates))]:
            # A closed Euler circuit can start at any existing node.  Picking
            # the nearest visible entry avoids needless grid routing between
            # otherwise adjacent closed STL hole walls.
            for entry, traversal in enumerate(region.traversals):
                start = traversal[0]
                direct = [current, start]
                if router.allows(LineString(direct)):
                    rotated = _rotate_euler_region(region, entry)
                    proposals.append((route_length(direct), index, direct, rotated))
        if not proposals:
            for _distance, index, region in candidates[: min(12, len(candidates))]:
                route = router.route(current, region.traversals[0][0])
                if route is not None:
                    proposals.append((route_length(route), index, route, region))
        if not proposals:
            for _distance, index, region in candidates[12:]:
                route = router.route(current, region.traversals[0][0])
                if route is not None:
                    proposals.append((route_length(route), index, route, region))
                    break
        if not proposals:
            raise ValueError("cannot connect Eulerized native wall regions without crossing a hole")
        _length, index, route, region = min(proposals, key=lambda item: item[:2])
        remaining.pop(index)
        ordered.append(region)
        travels.append(route)
        current = region.traversals[-1][1]
    return ordered, travels


def _rotate_euler_region(region: _EulerRegion, entry: int) -> _EulerRegion:
    traversals = region.traversals[entry:] + region.traversals[:entry]
    return _EulerRegion(traversals, region.edge_visits, region.repeated_length_mm)


def _edge_routing_graph(edges: Iterable[_Edge]):
    """Expose only direct STL wall edges to the hole-safe travel router."""

    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]] = defaultdict(list)
    for edge in edges:
        _add_route_edge(graph, edge.a, edge.b)
    return graph


def _stl_wall_routing_graph(
    edges: list[_Edge],
    frame: np.ndarray,
    solid,
) -> tuple[dict[tuple[float, float], list[tuple[tuple[float, float], float]]], dict[str, object]]:
    """Build a travel-only network between STL wall components and the rim.

    Voronoi edges clipped to an STL section can leave short, disconnected wall
    fragments at the outside boundary.  Those fragments must not be joined by
    newly deposited geometry.  For travel only, attach each boundary-reachable
    component to a sampled outer frame through the shortest straight segment
    wholly covered by the STL solid.  A component with no such direct route is
    deliberately left separate; it will use the solid-only fallback once,
    rather than receiving an invalid bridge through a void.
    """

    graph = _edge_routing_graph(edges)
    frame_xy = [(float(point[0]), float(point[1])) for point in frame]
    frame_nodes: list[tuple[float, float]] = []
    for first, second in zip(frame_xy, frame_xy[1:]):
        length = math.dist(first, second)
        steps = max(1, int(math.ceil(length / 1.0)))
        segment = [
            _node((
                first[0] + (second[0] - first[0]) * step / steps,
                first[1] + (second[1] - first[1]) * step / steps,
            ))
            for step in range(steps + 1)
        ]
        for start, end in zip(segment, segment[1:]):
            _add_route_edge(graph, start, end)
        frame_nodes.extend(segment)
    frame_nodes = list(dict.fromkeys(frame_nodes))

    adjacency: dict[tuple[float, float], list[int]] = defaultdict(list)
    for edge_id, edge in enumerate(edges):
        adjacency[edge.a].append(edge_id)
        adjacency[edge.b].append(edge_id)
    attached = 0
    unattached = 0
    connector_length = 0.0
    for nodes, _edge_ids in _components(adjacency, edges):
        best: tuple[float, tuple[float, float], tuple[float, float]] | None = None
        for node in nodes:
            nearby_frame_nodes = heapq.nsmallest(
                24,
                ((math.dist(node, candidate), candidate) for candidate in frame_nodes),
                key=lambda item: item[0],
            )
            for distance, candidate in nearby_frame_nodes:
                if best is not None and distance >= best[0]:
                    break
                if solid.covers(LineString((node, candidate))):
                    best = (distance, node, candidate)
                    break
        if best is None:
            # The large interior honeycomb component is intentionally not
            # forced through a new link: a straight segment to the rim would
            # cross one or more voids.  It remains a separate component and
            # is reached once, when necessary, by the router's solid-grid
            # fallback.  Boundary fragments still use the inexpensive rim
            # network above.
            unattached += 1
            continue
        distance, node, candidate = best
        if distance > _MIN_EDGE_MM:
            _add_route_edge(graph, node, candidate)
            attached += 1
            connector_length += distance
    return graph, {
        "travel_routing_connector_count": attached,
        "travel_routing_connector_length_mm": round(connector_length, 6),
        "travel_routing_connector_policy": "travel-only direct connectors fully covered by the source STL solid",
        "travel_routing_unattached_wall_component_count": unattached,
    }


def _eulerized_region_profiles(
    frame: np.ndarray | None,
    regions: list[_EulerRegion],
    *,
    z: float,
    rate: float,
    junctions: set[tuple[float, float]],
    tolerance_mm: float,
    taper_length_mm: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sample Euler regions and distribute wall material across retraces."""

    paths: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    current_e = 0.0
    if frame is not None:
        paths.append(np.asarray(frame, dtype=np.float64).copy())
        frame_length = np.linalg.norm(np.diff(paths[0][:, :3], axis=0), axis=1)
        frame_profile = current_e + np.concatenate(([0.0], np.cumsum(frame_length * rate)))
        profiles.append(frame_profile)
        current_e = float(frame_profile[-1])
    for region in regions:
        path, edge_ids = _sample_euler_region(region, z, taper_length_mm)
        distances = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(distances)))
        multipliers = _euler_junction_multipliers(
            path,
            cumulative,
            junctions,
            tolerance_mm=tolerance_mm,
            taper_length_mm=taper_length_mm,
        )
        edge_scales = np.asarray(
            [1.0 / region.edge_visits[edge_id] for edge_id in edge_ids],
            dtype=np.float64,
        )
        increments = distances * rate * edge_scales * (multipliers[:-1] + multipliers[1:]) * 0.5
        profile = current_e + np.concatenate(([0.0], np.cumsum(increments)))
        paths.append(path)
        profiles.append(profile)
        current_e = float(profile[-1])
    return paths, profiles


def _sample_euler_region(
    region: _EulerRegion,
    z: float,
    taper_length_mm: float,
) -> tuple[np.ndarray, list[int]]:
    """Add only collinear points so flow ramps remain smooth on wall edges."""

    if not region.traversals:
        raise ValueError("Eulerized native wall region is empty")
    points = [np.asarray((*region.traversals[0][0], z), dtype=np.float64)]
    edge_ids: list[int] = []
    spacing = max(float(taper_length_mm) * 0.25, 0.25)
    current = region.traversals[0][0]
    for start, end, edge_id in region.traversals:
        if math.dist(current, start) > 1e-6:
            raise ValueError("Eulerized native wall traversal is discontinuous")
        length = math.dist(start, end)
        steps = max(1, int(math.ceil(length / spacing)))
        first = np.asarray((*start, z), dtype=np.float64)
        second = np.asarray((*end, z), dtype=np.float64)
        for step in range(1, steps + 1):
            points.append(first + (step / steps) * (second - first))
            edge_ids.append(edge_id)
        current = end
    return np.asarray(points, dtype=np.float64), edge_ids


def _euler_junction_multipliers(
    path: np.ndarray,
    cumulative: np.ndarray,
    junctions: set[tuple[float, float]],
    *,
    tolerance_mm: float,
    taper_length_mm: float,
) -> np.ndarray:
    if not junctions:
        return np.ones(path.shape[0], dtype=np.float64)
    keys = {_point_key(node, tolerance_mm) for node in junctions}
    positions = np.asarray(
        [
            cumulative[index]
            for index, point in enumerate(path)
            if _point_key(point[:2], tolerance_mm) in keys
        ],
        dtype=np.float64,
    )
    if positions.size == 0:
        return np.ones(path.shape[0], dtype=np.float64)
    positions.sort()
    total = float(cumulative[-1])
    insertion = np.searchsorted(positions, cumulative, side="left")
    right = positions[np.minimum(insertion, positions.size - 1)]
    left = positions[np.maximum(insertion - 1, 0)]
    nearest = np.minimum(np.abs(cumulative - left), np.abs(right - cumulative))
    if total > 0.0:
        nearest = np.minimum(nearest, np.minimum(cumulative + positions[0], total - cumulative + positions[-1]))
    return _endpoint_flow_multiplier(nearest, 1.0 / 3.0, taper_length_mm)


def _is_same_outer_frame(path: np.ndarray, frame: np.ndarray, *, tolerance_mm: float) -> bool:
    if path.shape[0] < 4 or np.linalg.norm(path[0, :2] - path[-1, :2]) > tolerance_mm:
        return False
    candidate = LineString(path[:, :2])
    reference = LineString(frame[:, :2])
    return candidate.hausdorff_distance(reference) <= tolerance_mm


def _native_routing_graph(
    frame: np.ndarray,
    native_paths: list[tuple[int, np.ndarray]],
    native_travel: TravelPaths | None,
    solid,
    *,
    tolerance_mm: float,
):
    """Build a safe route graph from existing Prusa path geometry.

    It is deliberately a routing aid only: all internal deposited paths still
    come directly from Prusa.  Valid native travel segments add useful bridges
    to that graph, avoiding a dense grid search for every new connection.
    """

    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]] = defaultdict(list)
    allowed = solid.buffer(max(float(tolerance_mm), 1e-6))
    candidates = [frame, *(path for _index, path in native_paths)]
    if native_travel is not None:
        candidates.extend(np.asarray(path, dtype=np.float64) for path in native_travel.paths)
    for path in candidates:
        points = np.asarray(path, dtype=np.float64)
        if points.shape[0] < 2:
            continue
        for first, second in zip(points[:-1], points[1:]):
            line = LineString((first[:2], second[:2]))
            if not allowed.covers(line):
                continue
            start = (float(first[0]), float(first[1]))
            end = (float(second[0]), float(second[1]))
            if math.dist(start, end) > _MIN_EDGE_MM:
                _add_route_edge(graph, start, end)
    return graph


def _order_native_paths_for_continuous_motion(
    frame: np.ndarray,
    native_paths: list[tuple[int, np.ndarray]],
    *,
    router: HoleSafeTravelRouter,
) -> tuple[list[tuple[int, np.ndarray]], list[list[tuple[float, float]]]]:
    """Reorder native wall strokes by their shortest safe entry connection.

    Prusa owns the wall geometry, but its original ordering can include long
    jumps between distant cells.  The first direct connector that is safe is
    also the shortest possible connector for the current endpoint, so it is
    accepted immediately.  Only when every direct candidate crosses a hole do
    we invoke the slower exact grid fallback.  Paths are only reversed or
    reordered; their points and wall topology are never regenerated.
    """

    if not native_paths:
        return [], []
    remaining = [(index, np.asarray(path, dtype=np.float64)) for index, path in native_paths]
    ordered: list[tuple[int, np.ndarray]] = []
    routes: list[list[tuple[float, float]]] = []
    previous_path = frame
    while remaining:
        current = (float(previous_path[-1, 0]), float(previous_path[-1, 1]))
        candidates: list[tuple[float, int, int, np.ndarray]] = []
        for position, (source_index, path) in enumerate(remaining):
            variants = (path, path[::-1].copy())
            for direction, candidate in enumerate(variants):
                candidates.append(
                    (
                        math.dist(current, (float(candidate[0, 0]), float(candidate[0, 1]))),
                        position,
                        direction,
                        candidate,
                    )
                )
        candidates.sort(key=lambda item: item[:3])
        chosen: tuple[int, np.ndarray, list[tuple[float, float]]] | None = None
        # The nearest few endpoints contain the useful local choices in a
        # regular honeycomb.  Testing every remaining segment against every
        # hole creates a quadratic geometry workload without changing the
        # process strategy, so keep the direct-route scan deliberately local.
        local_candidates = candidates[: min(12, len(candidates))]
        for _lower_bound, position, _direction, candidate in local_candidates:
            direct = [
                current,
                (float(candidate[0, 0]), float(candidate[0, 1])),
            ]
            if router.allows(LineString(direct)):
                chosen = (position, candidate, direct)
                break
        if chosen is None:
            # Direct links are blocked by holes.  Only a small, distance-sorted
            # shortlist needs the full grid route in normal honeycomb layouts;
            # retain the complete fallback for unusual disconnected sections.
            routed: list[tuple[float, int, np.ndarray, list[tuple[float, float]]]] = []
            for _lower_bound, position, _direction, candidate in local_candidates:
                route = router.route(
                    current,
                    (float(candidate[0, 0]), float(candidate[0, 1])),
                )
                if route is not None:
                    routed.append((route_length(route), position, candidate, route))
            if not routed:
                for _lower_bound, position, _direction, candidate in candidates[len(local_candidates):]:
                    route = router.route(
                        current,
                        (float(candidate[0, 0]), float(candidate[0, 1])),
                    )
                    if route is not None:
                        routed.append((route_length(route), position, candidate, route))
                        break
            if not routed:
                raise ValueError("cannot connect native honeycomb paths without crossing a hole")
            _length, position, candidate, route = min(routed, key=lambda item: item[:2])
            chosen = (position, candidate, route)
        position, path, route = chosen
        source_index, _source_path = remaining.pop(position)
        ordered.append((source_index, path))
        routes.append(route)
        previous_path = path
    return ordered, routes


def _combine_with_zero_extrusion_connectors(
    paths: list[np.ndarray],
    extrusion: list[np.ndarray],
    connectors: list[list[tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Join deposited paths with zero-E connector segments into one path.

    Each original deposited segment retains its E increment exactly.  Connector
    points repeat the previous cumulative E value, a format accepted by the
    downstream source contract and therefore a genuine single motion path
    rather than a visual-only concatenation.
    """

    if not paths or len(paths) != len(extrusion):
        raise ValueError("continuous honeycomb path requires matching paths and extrusion")
    if len(connectors) != len(paths) - 1:
        raise ValueError("continuous honeycomb connectors must join every neighbouring path")

    points: list[np.ndarray] = [np.asarray(paths[0][0], dtype=np.float64)]
    cumulative_e: list[float] = [0.0]

    def append_deposition(path: np.ndarray, profile: np.ndarray) -> None:
        if path.shape[0] != profile.shape[0]:
            raise ValueError("extrusion profile must align with its deposition path")
        increments = np.diff(np.asarray(profile, dtype=np.float64))
        if np.any(increments < -1e-9):
            raise ValueError("continuous honeycomb path cannot contain negative extrusion")
        for point, increment in zip(path[1:], increments):
            points.append(np.asarray(point, dtype=np.float64))
            cumulative_e.append(cumulative_e[-1] + max(float(increment), 0.0))

    append_deposition(paths[0], extrusion[0])
    for connector, path, profile in zip(connectors, paths[1:], extrusion[1:]):
        connector_path = _travel_to_path(connector, float(path[0, 2]))
        if np.linalg.norm(connector_path[0] - points[-1]) > 1e-5:
            raise ValueError("continuous honeycomb connector does not start at the preceding path")
        if np.linalg.norm(connector_path[-1] - path[0]) > 1e-5:
            raise ValueError("continuous honeycomb connector does not end at the following path")
        for point in connector_path[1:]:
            points.append(np.asarray(point, dtype=np.float64))
            cumulative_e.append(cumulative_e[-1])
        append_deposition(path, profile)
    return np.vstack(points), np.asarray(cumulative_e, dtype=np.float64)


def _stl_region_connector_limit(solid, line_width_mm: float) -> float:
    """Keep local, same-region travel compact while splitting long jumps."""

    cell_side = _estimate_honeycomb_cell_side(solid, line_width_mm)
    # A region may cover several adjacent cells, but must not silently become
    # the former whole-layer pseudo-one-stroke. Six cell sides retains local
    # routing while keeping the downstream path count practical.
    return max(float(line_width_mm) * 6.0, cell_side * 6.0)


def _partition_stl_trails(
    paths: list[np.ndarray],
    extrusion: list[np.ndarray],
    connectors: list[list[tuple[float, float]]],
    *,
    max_intra_region_travel_mm: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[tuple[float, float]]], dict[str, object]]:
    """Pack nearby edge-disjoint trails into compact regions.

    The graph trail cover remains the topology authority. A short hole-safe
    transition may stay inside one process region as a zero-E move, which
    avoids thousands of tiny NPZ paths and their downstream overhead. A longer
    jump always starts a new region and is emitted as an explicit ``T`` path.
    """

    if not paths or len(paths) != len(extrusion):
        raise ValueError("STL trail partition requires aligned deposition paths and E profiles")
    if len(connectors) != len(paths) - 1:
        raise ValueError("STL trail partition requires one connector per neighbouring trail")
    region_paths: list[np.ndarray] = []
    region_extrusion: list[np.ndarray] = []
    inter_region: list[list[tuple[float, float]]] = []
    current_paths = [paths[0]]
    current_extrusion = [extrusion[0]]
    current_connectors: list[list[tuple[float, float]]] = []
    local_connector_count = 0
    local_connector_length = 0.0
    for connector, path, profile in zip(connectors, paths[1:], extrusion[1:]):
        length = route_length(connector)
        if length <= max_intra_region_travel_mm + 1e-9:
            current_connectors.append(connector)
            current_paths.append(path)
            current_extrusion.append(profile)
            local_connector_count += 1
            local_connector_length += length
            continue
        combined_path, combined_e = _combine_with_zero_extrusion_connectors(
            current_paths,
            current_extrusion,
            current_connectors,
        )
        region_paths.append(combined_path)
        region_extrusion.append(combined_e)
        inter_region.append(connector)
        current_paths = [path]
        current_extrusion = [profile]
        current_connectors = []
    combined_path, combined_e = _combine_with_zero_extrusion_connectors(
        current_paths,
        current_extrusion,
        current_connectors,
    )
    region_paths.append(combined_path)
    region_extrusion.append(combined_e)
    return region_paths, region_extrusion, inter_region, {
        "layer_path_count": len(region_paths),
        "motion_path_count": len(region_paths),
        "travel_count": len(inter_region),
        "inter_region_travel_length_mm_per_layer": round(
            sum(route_length(path) for path in inter_region), 6
        ),
        "intra_region_zero_e_connector_count": local_connector_count,
        "intra_region_zero_e_connector_length_mm_per_layer": round(local_connector_length, 6),
        "max_intra_region_travel_mm": round(float(max_intra_region_travel_mm), 6),
    }


def _first_layer_outer_frame(job: ExternalSourceJob, group: MaterialPaths) -> np.ndarray:
    roles = job.meta.get("path_roles", {})
    layer_roles = roles.get("R", {}).get(str(group.layer_index), []) if isinstance(roles, dict) else []
    candidates: list[tuple[float, np.ndarray]] = []
    for index, path in enumerate(group.paths):
        role = layer_roles[index] if isinstance(layer_roles, list) and index < len(layer_roles) else ""
        if role != "outer_contour" or path.shape[0] < 4:
            continue
        span = np.ptp(path[:, :2], axis=0)
        candidates.append((float(span[0] * span[1]), path))
    if not candidates:
        raise ValueError("native Prusa result has no first-layer outer contour to use as the frame")
    _, frame = max(candidates, key=lambda item: item[0])
    if not np.allclose(frame[0, :2], frame[-1, :2], atol=1e-4, rtol=0.0):
        frame = np.vstack((frame, frame[0]))
    return np.asarray(frame, dtype=np.float64)


def _outer_frame_from_solid(solid, z: float) -> np.ndarray:
    """Return the largest STL-section exterior as one closed XYZ frame."""

    polygons = list(_polygons(solid))
    if not polygons:
        raise ValueError("STL section has no exterior boundary for the honeycomb frame")
    outer = max(polygons, key=lambda polygon: float(polygon.area))
    xy = np.asarray(outer.exterior.coords, dtype=np.float64)
    if xy.shape[0] < 4:
        raise ValueError("STL section outer boundary must contain at least three edges")
    frame = np.column_stack((xy, np.full(xy.shape[0], float(z), dtype=np.float64)))
    if not np.allclose(frame[0, :2], frame[-1, :2], atol=1e-8, rtol=0.0):
        frame = np.vstack((frame, frame[0]))
    return frame


def _solid_geometry_at_z(mesh: Mesh, z: float, tolerance: float):
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for triangle in np.asarray(mesh.triangles, dtype=np.float64):
        points: list[np.ndarray] = []
        for start, end in ((0, 1), (1, 2), (2, 0)):
            p0, p1 = triangle[start], triangle[end]
            if abs(float(p1[2] - p0[2])) <= tolerance:
                continue
            if z < min(p0[2], p1[2]) - tolerance or z > max(p0[2], p1[2]) + tolerance:
                continue
            point = p0 + ((z - p0[2]) / (p1[2] - p0[2])) * (p1 - p0)
            if not any(np.linalg.norm(point[:2] - existing[:2]) <= tolerance for existing in points):
                points.append(point)
        if len(points) == 2 and np.linalg.norm(points[0][:2] - points[1][:2]) > tolerance:
            segments.append((points[0][:2], points[1][:2]))
    contours = _stitch_segments(segments, tolerance)
    rings = [np.asarray(contour, dtype=np.float64) for contour in contours if len(contour) >= 3]
    polygons = [Polygon(_open_ring(ring)) for ring in rings]
    polygons = [polygon.buffer(0) if not polygon.is_valid else polygon for polygon in polygons]
    shells: list[Polygon] = []
    holes: list[Polygon] = []
    for polygon in polygons:
        if any(other.contains(polygon.representative_point()) and other.area > polygon.area for other in polygons):
            holes.append(polygon)
        else:
            shells.append(polygon)
    result = []
    for shell in shells:
        shell_holes = [list(hole.exterior.coords) for hole in holes if shell.contains(hole.representative_point())]
        result.append(Polygon(shell.exterior.coords, shell_holes))
    solid = unary_union(result).buffer(0) if result else Polygon()
    if solid.is_empty:
        raise ValueError("STL layer has no solid printable honeycomb geometry")
    return solid


def _stitch_segments(segments: list[tuple[np.ndarray, np.ndarray]], tolerance: float) -> list[list[np.ndarray]]:
    indexed = [(np.asarray(a), np.asarray(b)) for a, b in segments]
    by_end: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (a, b) in enumerate(indexed):
        by_end[_point_key(a, tolerance)].append(index)
        by_end[_point_key(b, tolerance)].append(index)
    unused = set(range(len(indexed)))
    paths: list[list[np.ndarray]] = []
    while unused:
        index = unused.pop()
        a, b = indexed[index]
        path = [a, b]
        for at_end in (True, False):
            while True:
                endpoint = path[-1] if at_end else path[0]
                next_index = next((candidate for candidate in by_end[_point_key(endpoint, tolerance)] if candidate in unused), None)
                if next_index is None:
                    break
                unused.remove(next_index)
                first, second = indexed[next_index]
                next_point = second if np.linalg.norm(endpoint - first) <= tolerance else first
                if at_end:
                    path.append(next_point)
                else:
                    path.insert(0, next_point)
        paths.append(path)
    return paths


def _estimate_honeycomb_cell_side(solid, line_width_mm: float) -> float:
    """Infer the source honeycomb's nominal pointy-hexagon side length."""

    centers_y = sorted(
        float(Polygon(ring).centroid.y)
        for polygon in _polygons(solid)
        for ring in polygon.interiors
    )
    unique_y: list[float] = []
    for value in centers_y:
        if not unique_y or abs(value - unique_y[-1]) > 1e-3:
            unique_y.append(value)
    pitches = [
        right - left
        for left, right in zip(unique_y, unique_y[1:])
        if right - left > max(float(line_width_mm), 1e-3)
    ]
    if pitches:
        return float(np.median(pitches) / 1.5)
    # A compact fallback still leaves a meaningful regular-cell pattern on a
    # small or irregular sample with too few holes to infer the source pitch.
    return max(float(line_width_mm) * 2.5, 1.0)


def _minimum_trail_cover(edges: list[_Edge]) -> list[list[tuple[float, float]]]:
    adjacency: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        adjacency[edge.a].append(index)
        adjacency[edge.b].append(index)
    trails: list[list[tuple[float, float]]] = []
    for component_nodes, component_edges in _components(adjacency, edges):
        odd = sorted(node for node in component_nodes if len(adjacency[node]) % 2)
        all_edges = [edges[index] for index in component_edges]
        # Virtual pairs turn the graph Eulerian.  Removing them afterwards is
        # the exact minimum non-repeating trail cover.
        for left, right in _pair_odds(odd):
            all_edges.append(_Edge(left, right, 0.0))
        virtual_start = len(component_edges)
        circuit = _euler_edges(all_edges, component_nodes[0])
        current: list[tuple[float, float]] = []
        for start, end, edge_index in circuit:
            if edge_index >= virtual_start:
                if len(current) >= 2:
                    trails.append(current)
                current = []
                continue
            if not current:
                current = [start]
            current.append(end)
        if len(current) >= 2:
            trails.append(current)
    return trails


def _junction_nodes(edges: list[_Edge]) -> set[tuple[float, float]]:
    """Return only genuine three-way vertices of the STL wall graph."""

    degree: dict[tuple[float, float], int] = defaultdict(int)
    for edge in edges:
        degree[edge.a] += 1
        degree[edge.b] += 1
    return {node for node, count in degree.items() if count == 3}


def _insert_junction_taper_points(
    paths: list[np.ndarray],
    junctions: set[tuple[float, float]],
    *,
    tolerance_mm: float,
    taper_length_mm: float,
) -> list[np.ndarray]:
    """Add controller points around true STL three-way wall junctions.

    No XY coordinate is moved or invented here: every added point is an
    interpolation along an already accepted STL-derived wall edge.
    """

    if not junctions or taper_length_mm <= tolerance_mm:
        return [np.asarray(path, dtype=np.float64).copy() for path in paths]
    result: list[np.ndarray] = []
    for path in paths:
        points = np.asarray(path, dtype=np.float64)
        if points.shape[0] < 2:
            result.append(points.copy())
            continue
        expanded = [points[0].copy()]
        for first, second in zip(points[:-1], points[1:]):
            length = float(np.linalg.norm(second[:3] - first[:3]))
            fractions = {0.0, 1.0}
            if length > taper_length_mm + tolerance_mm:
                if _is_junction(first, junctions, tolerance_mm):
                    fractions.update((0.25 * taper_length_mm / length, 0.5 * taper_length_mm / length, 0.75 * taper_length_mm / length))
                if _is_junction(second, junctions, tolerance_mm):
                    fractions.update((1.0 - 0.75 * taper_length_mm / length, 1.0 - 0.5 * taper_length_mm / length, 1.0 - 0.25 * taper_length_mm / length))
            for fraction in sorted(value for value in fractions if value > 1e-12):
                expanded.append(first + float(fraction) * (second - first))
        result.append(_dedupe_points(np.asarray(expanded, dtype=np.float64)))
    return result


def _junction_extrusion_profiles(
    paths: list[np.ndarray],
    rate: float,
    junctions: set[tuple[float, float]],
    *,
    tolerance_mm: float,
    taper_length_mm: float,
) -> list[np.ndarray]:
    """Build cumulative E with a smooth one-third flow at each true junction."""

    current = 0.0
    profiles: list[np.ndarray] = []
    for path in paths:
        distances = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(distances)))
        junction_indices = [
            index for index, point in enumerate(path)
            if _is_junction(point, junctions, tolerance_mm)
        ]
        if junction_indices:
            nearest = np.min(
                np.abs(cumulative[:, None] - cumulative[np.asarray(junction_indices)][None, :]),
                axis=1,
            )
            multipliers = _endpoint_flow_multiplier(nearest, 1.0 / 3.0, taper_length_mm)
        else:
            multipliers = np.ones(path.shape[0], dtype=np.float64)
        increments = distances * rate * (multipliers[:-1] + multipliers[1:]) * 0.5
        profile = current + np.concatenate(([0.0], np.cumsum(increments)))
        profiles.append(profile)
        current = float(profile[-1]) if profile.size else current
    return profiles


def _junction_taper_meta(junctions: set[tuple[float, float]], line_width_mm: float) -> dict[str, object]:
    return {
        "format": "stl_three_way_junction_taper_v1",
        "junction_count": len(junctions),
        "junction_flow_scale": round(1.0 / 3.0, 6),
        "taper_length_mm": float(line_width_mm),
        "profile": "quintic smoothstep from one-third at a three-way node to nominal over one line width",
        "geometry": "only existing STL-derived wall edges are sampled; no material bridge is added",
    }


def _is_junction(point, junctions: set[tuple[float, float]], tolerance_mm: float) -> bool:
    key = _point_key(point[:2], tolerance_mm)
    return any(_point_key(node, tolerance_mm) == key for node in junctions)


def _components(adjacency, edges):
    pending = set(adjacency)
    while pending:
        start = next(iter(pending))
        stack = [start]
        nodes: list[tuple[float, float]] = []
        edge_ids: set[int] = set()
        pending.remove(start)
        while stack:
            node = stack.pop()
            nodes.append(node)
            for edge_id in adjacency[node]:
                edge_ids.add(edge_id)
                edge = edges[edge_id]
                other = edge.b if edge.a == node else edge.a
                if other in pending:
                    pending.remove(other)
                    stack.append(other)
        yield nodes, sorted(edge_ids)


def _pair_odds(nodes: list[tuple[float, float]]):
    remaining = list(nodes)
    while remaining:
        start = remaining.pop(0)
        end = min(remaining, key=lambda candidate: math.dist(start, candidate))
        remaining.remove(end)
        yield start, end


def _euler_edges(edges: list[_Edge], start: tuple[float, float]):
    adjacency: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, edge in enumerate(edges):
        adjacency[edge.a].append(index)
        adjacency[edge.b].append(index)
    used: set[int] = set()
    stack: list[tuple[tuple[float, float], int | None, tuple[float, float] | None]] = [(start, None, None)]
    out: list[tuple[tuple[float, float], tuple[float, float], int]] = []
    while stack:
        node, _incoming, _previous = stack[-1]
        edge_id = next((candidate for candidate in adjacency[node] if candidate not in used), None)
        if edge_id is not None:
            used.add(edge_id)
            edge = edges[edge_id]
            other = edge.b if edge.a == node else edge.a
            stack.append((other, edge_id, node))
            continue
        node, incoming, previous = stack.pop()
        if incoming is not None and previous is not None:
            out.append((previous, node, incoming))
    return list(reversed(out))


def _routing_graph(edges: list[_Edge], solid):
    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]] = defaultdict(list)
    for edge in edges:
        _add_route_edge(graph, edge.a, edge.b)
    components = list(_routing_components(graph))
    if len(components) > 1:
        connected = set(max(components, key=len))
        for component in sorted((item for item in components if set(item) != connected), key=len, reverse=True):
            connector = _shortest_visible_connector(component, connected, solid)
            if connector is None:
                continue
            left, right = connector
            _add_route_edge(graph, left, right)
            connected.update(component)
    return graph


def _routing_components(graph):
    pending = set(graph)
    while pending:
        start = pending.pop()
        component = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for other, _cost in graph[node]:
                if other in pending:
                    pending.remove(other)
                    component.add(other)
                    stack.append(other)
        yield component


def _shortest_visible_connector(source, target, solid):
    best = None
    for left in source:
        for right in target:
            distance = math.dist(left, right)
            if best is not None and distance >= best[0]:
                continue
            if solid.covers(LineString((left, right))):
                best = (distance, left, right)
    return None if best is None else (best[1], best[2])


def _order_trails(frame: np.ndarray, trails, travel_router: HoleSafeTravelRouter):
    remaining = [list(trail) for trail in trails]
    ordered: list[list[tuple[float, float]]] = []
    travels: list[list[tuple[float, float]]] = []
    current = (float(frame[-1, 0]), float(frame[-1, 1]))
    while remaining:
        # The old planner accepted the first endpoint that happened to be
        # closest in straight-line distance.  A straight line can be a poor
        # proxy near honeycomb holes, so evaluate real constrained lengths.
        # Euclidean distance is nevertheless a strict lower bound on any
        # connector, allowing us to stop as soon as later candidates cannot
        # beat the best safe route already found.
        best = None
        candidates = []
        for index, trail in enumerate(remaining):
            for reverse in (False, True):
                candidate = list(reversed(trail)) if reverse else trail
                candidates.append((math.dist(current, candidate[0]), index, reverse, candidate))
        for lower_bound, index, reverse, candidate in sorted(candidates, key=lambda item: item[:3]):
            if best is not None and lower_bound >= best[0] - 1e-8:
                break
            route = travel_router.route(current, candidate[0])
            if route is None:
                continue
            proposal = (route_length(route), index, reverse, candidate, route)
            if best is None or proposal[:3] < best[:3]:
                best = proposal
        if best is None:
            raise ValueError("cannot route honeycomb travel without crossing a hole")
        _cost, index, _reverse, trail, route = best
        ordered.append(trail)
        travels.append(route)
        current = trail[-1]
        remaining.pop(index)
    _two_opt_improve(frame, ordered, travels, travel_router)
    return ordered, travels


def _solid_from_closed_hexagons(frame: np.ndarray, loops: Iterable[np.ndarray]):
    """Build the replacement solid whose voids are the generated hexagons."""

    outer = Polygon([(float(x), float(y)) for x, y in frame[:, :2]])
    if outer.is_empty or not outer.is_valid:
        outer = outer.buffer(0)
    holes = []
    for loop in loops:
        hexagon = Polygon([(float(x), float(y)) for x, y in loop[:, :2]])
        if hexagon.is_empty or not hexagon.is_valid or not outer.contains(hexagon):
            raise ValueError("generated closed hexagon lies outside the outer frame")
        holes.append(list(hexagon.exterior.coords))
    result = Polygon(outer.exterior.coords, holes)
    if result.is_empty or not result.is_valid:
        raise ValueError("generated closed-hexagon honeycomb is invalid")
    return result


def _closed_loop_routing_graph(frame: np.ndarray, loops: Iterable[np.ndarray]):
    """Use every printed boundary as an exact, no-hole travel graph."""

    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]] = defaultdict(list)
    for path in (frame[:, :2], *loops):
        for first, second in zip(path, path[1:]):
            _add_route_edge(
                graph,
                (float(first[0]), float(first[1])),
                (float(second[0]), float(second[1])),
            )
    return graph


def _order_closed_loops(
    frame: np.ndarray,
    loops: Iterable[np.ndarray],
    travel_router: HoleSafeTravelRouter,
):
    """Greedily order closed hexagons using their nearest valid entry side.

    A closed loop can start from any of its six side midpoints without changing
    the printed hexagon.  Choosing that point per move removes the avoidable
    lead-in distance that would result from fixing all loops to one vertex.
    This remains linear in the number of emitted paths after candidate
    evaluation; unlike generic 2-opt it is intentionally bounded for large
    honeycomb arrays.
    """

    remaining = [np.asarray(loop, dtype=np.float64) for loop in loops]
    ordered: list[list[tuple[float, float]]] = []
    travels: list[list[tuple[float, float]]] = []
    current = (float(frame[-1, 0]), float(frame[-1, 1]))
    while remaining:
        candidates = []
        for index, loop in enumerate(remaining):
            for variant_index, candidate in enumerate(_closed_loop_entry_variants(loop)):
                candidates.append(
                    (math.dist(current, candidate[0]), index, variant_index, candidate)
                )
        best = None
        for lower_bound, index, variant_index, candidate in sorted(
            candidates,
            key=lambda item: item[:3],
        ):
            if best is not None and lower_bound >= best[0] - 1e-8:
                break
            route = travel_router.route(current, candidate[0])
            if route is None:
                continue
            proposal = (route_length(route), index, variant_index, candidate, route)
            if best is None or proposal[:3] < best[:3]:
                best = proposal
        if best is None:
            raise ValueError("cannot route closed-hexagon travel without crossing a hole")
        _cost, index, _variant_index, loop, route = best
        ordered.append(loop)
        travels.append(route)
        current = loop[-1]
        remaining.pop(index)
    return ordered, travels


def _closed_loop_entry_variants(loop: np.ndarray) -> list[list[tuple[float, float]]]:
    """Return the twelve geometrically identical start/direction variants."""

    vertices = [tuple(map(float, point)) for point in loop[:-1, :2]]
    if len(vertices) != 6:
        raise ValueError("closed-hexagon loop must have exactly six vertices")
    variants: list[list[tuple[float, float]]] = []
    for index in range(6):
        first = vertices[index]
        second = vertices[(index + 1) % 6]
        start = ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)
        forward = [start, *[vertices[(index + 1 + step) % 6] for step in range(6)], start]
        backward = [start, *[vertices[(index - step) % 6] for step in range(6)], start]
        variants.extend((forward, backward))
    return variants


def _two_opt_improve(frame, ordered, travels, travel_router: HoleSafeTravelRouter) -> None:
    """Improve a route order without changing a wall edge or using a void.

    Reversing a contiguous block and reversing every trail in that block keeps
    all its internal safe links valid.  The two boundary links are rerouted and
    accepted only when the total travel length strictly falls.  Two passes
    give a useful improvement on large honeycomb graphs without an expensive
    global travelling-salesperson solve.
    """

    tolerance = 1e-8
    for _pass in range(2):
        improved = False
        for first_index in range(len(ordered) - 1):
            before = (
                (float(frame[-1, 0]), float(frame[-1, 1]))
                if first_index == 0
                else ordered[first_index - 1][-1]
            )
            for last_index in range(first_index + 1, len(ordered)):
                reversed_block = [
                    list(reversed(trail))
                    for trail in reversed(ordered[first_index:last_index + 1])
                ]
                after_index = last_index + 1
                old_length = route_length(travels[first_index])
                lower_bound = math.dist(before, reversed_block[0][0])
                if after_index < len(ordered):
                    old_length += route_length(travels[after_index])
                    lower_bound += math.dist(reversed_block[-1][-1], ordered[after_index][0])
                # The interior links simply reverse, so only the two boundary
                # links can change.  Their Euclidean lengths are lower bounds
                # for their new safe routes; skip the expensive GEOS/Dijkstra
                # checks when they cannot improve the existing boundaries.
                if lower_bound >= old_length - tolerance:
                    continue
                new_before = travel_router.route(before, reversed_block[0][0])
                if new_before is None:
                    continue
                new_after = None
                if after_index < len(ordered):
                    new_after = travel_router.route(
                        reversed_block[-1][-1], ordered[after_index][0]
                    )
                    if new_after is None:
                        continue
                new_length = route_length(new_before)
                if new_after is not None:
                    new_length += route_length(new_after)
                if new_length + tolerance >= old_length:
                    continue
                ordered[first_index:last_index + 1] = reversed_block
                travels[first_index] = new_before
                travels[first_index + 1:last_index + 1] = [
                    list(reversed(route))
                    for route in reversed(travels[first_index + 1:last_index + 1])
                ]
                if new_after is not None:
                    travels[after_index] = new_after
                improved = True
                break
            if improved:
                break
        if not improved:
            return


def _add_route_edge(graph, first, second):
    distance = math.dist(first, second)
    graph[first].append((second, distance))
    graph[second].append((first, distance))


def _trail_to_path(trail, z):
    points = np.asarray([[x, y, z] for x, y in trail], dtype=np.float64)
    return _dedupe_points(points)


def _travel_to_path(route, z):
    points = np.asarray([[x, y, z] for x, y in route], dtype=np.float64)
    return _dedupe_points(points)


def _dedupe_points(points):
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) > 1e-8:
            kept.append(point)
    if len(kept) < 2:
        kept.append(kept[0].copy())
    return np.asarray(kept, dtype=np.float64)


def _native_e_per_mm(group: MaterialPaths, tolerance: float) -> float:
    if group.extrusion is None:
        return 1.0
    rates = []
    for path, values in zip(group.paths, group.extrusion):
        distance = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1).sum()
        delta = float(values[-1] - values[0])
        if distance > tolerance and delta > 0:
            rates.append(delta / distance)
    return float(np.median(rates)) if rates else 1.0


def _endpoint_taper_plan(
    paths: list[np.ndarray],
    *,
    tolerance_mm: float,
    line_width_mm: float,
) -> tuple[list[tuple[float, float, float]], dict[str, object]]:
    """Plan short flow ramps for repeated open-path endpoints.

    An E value cannot be lowered at a mathematical point: material is emitted
    over a segment.  The plan therefore reduces the first/last bead length at
    a repeated endpoint and smoothly returns to nominal flow one line width
    away. At a genuine three-way honeycomb node, each incident wall starts or
    ends at one third of nominal flow so the three local contributions sum to
    one nominal bead volume.
    """

    endpoint_counts: dict[tuple[int, int], int] = defaultdict(int)
    path_keys: list[tuple[tuple[int, int] | None, tuple[int, int] | None]] = []
    for path in paths:
        if path.shape[0] < 2 or np.linalg.norm(path[0, :2] - path[-1, :2]) <= tolerance_mm:
            path_keys.append((None, None))
            continue
        start = _point_key(path[0, :2], tolerance_mm)
        end = _point_key(path[-1, :2], tolerance_mm)
        endpoint_counts[start] += 1
        endpoint_counts[end] += 1
        path_keys.append((start, end))

    maximum_visits = max(endpoint_counts.values(), default=0)
    if maximum_visits > 3:
        raise ValueError(
            "honeycomb endpoint recurrence exceeds the permitted three visits; "
            "split the wall graph before exporting"
        )
    taper_length = max(float(line_width_mm), float(tolerance_mm) * 10.0)
    plan: list[tuple[float, float, float]] = []
    for start_key, end_key in path_keys:
        start_count = 0 if start_key is None else endpoint_counts[start_key]
        end_count = 0 if end_key is None else endpoint_counts[end_key]
        start_scale = 1.0 if start_count <= 1 else 1.0 / start_count
        end_scale = 1.0 if end_count <= 1 else 1.0 / end_count
        plan.append((start_scale, end_scale, taper_length))

    shared = [count for count in endpoint_counts.values() if count > 1]
    return plan, {
        "format": "shared_endpoint_flow_taper_v1",
        "endpoint_repeat_limit": 3,
        "shared_endpoint_count": len(shared),
        "maximum_endpoint_visits": maximum_visits,
        "taper_length_mm": taper_length,
        "flow_scale": {
            "two_visits": 0.5,
            "three_visits": round(1.0 / 3.0, 6),
        },
        "profile": "quintic smoothstep from reduced endpoint flow to nominal over one line width",
        "edge_policy": "each wall edge is still deposited exactly once",
    }


def _extrusion_profiles(
    paths,
    rate,
    *,
    taper_plan: list[tuple[float, float, float]] | None = None,
):
    if taper_plan is not None and len(taper_plan) != len(paths):
        raise ValueError("endpoint taper plan must align with material paths")
    current = 0.0
    profiles = []
    for index, path in enumerate(paths):
        distances = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
        if taper_plan is None:
            increments = distances * rate
        else:
            start_scale, end_scale, taper_length = taper_plan[index]
            cumulative = np.concatenate(([0.0], np.cumsum(distances)))
            total_length = float(cumulative[-1])
            multipliers = np.minimum(
                _endpoint_flow_multiplier(cumulative, start_scale, taper_length),
                _endpoint_flow_multiplier(total_length - cumulative, end_scale, taper_length),
            )
            increments = distances * rate * (multipliers[:-1] + multipliers[1:]) * 0.5
        profile = current + np.concatenate(([0.0], np.cumsum(increments)))
        profiles.append(profile)
        current = float(profile[-1])
    return profiles


def _insert_endpoint_taper_points(
    paths: list[np.ndarray],
    taper_plan: list[tuple[float, float, float]],
) -> list[np.ndarray]:
    """Split only at flow-ramp transitions, preserving path geometry exactly.

    A controller can vary E only between trajectory points.  These inserted
    points keep a one-line-width taper local to a shared end instead of
    lowering the E value over a complete long honeycomb edge.
    """

    if len(paths) != len(taper_plan):
        raise ValueError("endpoint taper plan must align with material paths")
    return [
        _insert_taper_points_one(path, start_scale, end_scale, taper_length)
        for path, (start_scale, end_scale, taper_length) in zip(paths, taper_plan)
    ]


def _insert_taper_points_one(
    path: np.ndarray,
    start_scale: float,
    end_scale: float,
    taper_length: float,
) -> np.ndarray:
    if path.shape[0] < 2 or (start_scale >= 1.0 and end_scale >= 1.0):
        return np.asarray(path, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total = float(cumulative[-1])
    if total <= 1e-12 or taper_length <= 0.0:
        return np.asarray(path, dtype=np.float64)

    breaks = {float(value) for value in cumulative}
    if start_scale < 1.0 and taper_length < total:
        breaks.add(float(taper_length))
    if end_scale < 1.0 and taper_length < total:
        breaks.add(float(total - taper_length))

    # A quintic ramp has zero slope at both ends. Split it into four equal
    # controller segments so exported cumulative E follows that smooth
    # profile instead of creating a sudden flow jump at the node.
    for fraction in (0.25, 0.5, 0.75):
        if start_scale < 1.0 and fraction * taper_length < total:
            breaks.add(float(fraction * taper_length))
        if end_scale < 1.0 and fraction * taper_length < total:
            breaks.add(float(total - fraction * taper_length))

    targets = sorted(breaks)
    points = []
    for target in targets:
        segment_index = min(
            int(np.searchsorted(cumulative, target, side="right") - 1),
            len(segment_lengths) - 1,
        )
        segment_index = max(0, segment_index)
        length = float(segment_lengths[segment_index])
        ratio = 0.0 if length <= 1e-12 else (target - cumulative[segment_index]) / length
        ratio = float(np.clip(ratio, 0.0, 1.0))
        points.append(path[segment_index] + ratio * (path[segment_index + 1] - path[segment_index]))
    return np.asarray(points, dtype=np.float64)


def _endpoint_flow_multiplier(distance_from_endpoint, endpoint_scale: float, taper_length: float):
    if endpoint_scale >= 1.0 or taper_length <= 0.0:
        return np.ones_like(distance_from_endpoint, dtype=np.float64)
    ramp = np.clip(np.asarray(distance_from_endpoint, dtype=np.float64) / taper_length, 0.0, 1.0)
    # Quintic smoothstep has zero slope at 0 and 1, avoiding an abrupt change
    # in commanded E/mm at the start or end of a shared honeycomb node.
    smooth = ramp * ramp * ramp * (ramp * (ramp * 6.0 - 15.0) + 10.0)
    return endpoint_scale + (1.0 - endpoint_scale) * smooth


def _start_flow_value(distance: float, endpoint_scale: float, taper_length: float) -> float:
    return float(_endpoint_flow_multiplier(np.asarray([distance]), endpoint_scale, taper_length)[0])


def _end_flow_value(distance: float, total: float, endpoint_scale: float, taper_length: float) -> float:
    return float(_endpoint_flow_multiplier(np.asarray([total - distance]), endpoint_scale, taper_length)[0])




def _layer_z(group: MaterialPaths) -> float:
    values = [float(path[0, 2]) for path in group.paths if path.shape[0]]
    if not values:
        raise ValueError("resin layer has no Z coordinate")
    return float(np.median(values))


def _polygons(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            yield from _polygons(item)


def _lines(geometry):
    if geometry.geom_type == "LineString":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            yield from _lines(item)


def _open_ring(points: Iterable[np.ndarray]):
    values = [(float(point[0]), float(point[1])) for point in points]
    return values[:-1] if len(values) > 1 and values[0] == values[-1] else values


def _node(point) -> tuple[float, float]:
    return (round(float(point[0]), _NODE_DECIMALS), round(float(point[1]), _NODE_DECIMALS))


def _point_key(point, tolerance: float):
    scale = 1.0 / max(tolerance, 1e-9)
    return (int(round(float(point[0]) * scale)), int(round(float(point[1]) * scale)))


def _route_length(route):
    return sum(math.dist(first, second) for first, second in zip(route, route[1:]))
