"""Stable conformal-edge bridge and macro-partition planner for External NPZ.

The geometry graph remains the authority: every structural edge receives one
deposition traversal per physical 2 mm bead lane. A minimum non-repeating
trail cover and zero-E graph connectors compact that graph before it is handed
to the existing Core XYZABC pipeline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import heapq
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from ..external_npz import ExternalSourceJob, MaterialPaths, TravelPaths, write_external_source_npz
from .contracts import canonical_json_sha256
from .layer_embedding import LayerEmbedding
from .lattice_generator import ConformalLatticeGeometry
from ..honeycomb_pathing.planner import _Edge, _minimum_trail_cover


_EDGE_ID_BITS = 32
_EDGE_ID_COMPONENT_MAX = (1 << _EDGE_ID_BITS) - 1


@dataclass(frozen=True, slots=True)
class ExtrusionVolumeModel:
    """Explicit conversion from 3-D deposited length to cumulative E.

    ``bead_cross_section_area_mm2`` defines the intended deposited volume per
    millimetre of centreline.  ``e_volume_per_unit_mm3`` defines the machine's
    E unit; it is deliberately required rather than inferred from a global
    printer profile or a flat XY reference path.
    """

    bead_cross_section_area_mm2: float
    e_volume_per_unit_mm3: float
    preview_line_width_mm: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("bead_cross_section_area_mm2", self.bead_cross_section_area_mm2),
            ("e_volume_per_unit_mm3", self.e_volume_per_unit_mm3),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.preview_line_width_mm is not None and (
            not np.isfinite(self.preview_line_width_mm) or self.preview_line_width_mm <= 0.0
        ):
            raise ValueError("preview_line_width_mm must be positive and finite when supplied")

    def metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "format": "conformal_edge_volume_e_v1",
            "bead_cross_section_area_mm2": float(self.bead_cross_section_area_mm2),
            "e_volume_per_unit_mm3": float(self.e_volume_per_unit_mm3),
            "formula": "delta_E = 3d_edge_length_mm * bead_cross_section_area_mm2 / e_volume_per_unit_mm3",
            "uses_actual_3d_length": True,
            "requires_xy_preservation": False,
        }
        if self.preview_line_width_mm is not None:
            metadata["preview_line_width_mm"] = float(self.preview_line_width_mm)
        return metadata


@dataclass(frozen=True, slots=True)
class ConformalLatticePathGraph:
    """Stable IDs and per-layer geometry before trail/partition planning."""

    node_ids: np.ndarray
    edge_ids: np.ndarray
    edge_node_ids: np.ndarray
    edge_parent_id: np.ndarray
    edge_segment_id: np.ndarray
    edge_source_triangle_id: np.ndarray
    layer_node_positions_xyz: np.ndarray
    edge_length_mm: np.ndarray
    deposited_volume_mm3: np.ndarray
    cumulative_extrusion_e: np.ndarray
    node_normals_xyz: np.ndarray
    wall_bead_count: int
    nominal_bead_width_mm: float
    report: dict[str, object]
    metadata: dict[str, object]

    def to_external_source_job(self, *, material: Literal["R", "F"] = "R") -> ExternalSourceJob:
        """Plan graph edges into non-repeating macro partitions for Core."""

        if material not in ("R", "F"):
            raise ValueError("material must be R or F")
        material_paths: list[MaterialPaths] = []
        travel_paths: list[TravelPaths] = []
        edge_ids_by_layer: dict[str, list[int]] = {}
        planning_reports: list[dict[str, object]] = []
        extrusion_config = self.metadata["config"]["extrusion"]
        for layer_index, positions in enumerate(self.layer_node_positions_xyz):
            paths, extrusion, travels, planning_report = _plan_layer_macro_partitions(
                self.edge_node_ids,
                positions,
                self.node_normals_xyz,
                bead_count=self.wall_bead_count,
                bead_width_mm=self.nominal_bead_width_mm,
                bead_area_mm2=float(extrusion_config["bead_cross_section_area_mm2"]),
                e_volume_per_unit_mm3=float(extrusion_config["e_volume_per_unit_mm3"]),
            )
            material_paths.append(MaterialPaths(layer_index, material, paths, extrusion))
            travel_paths.append(TravelPaths(layer_index, travels))
            edge_ids_by_layer[str(layer_index)] = [int(value) for value in self.edge_ids]
            planning_reports.append(planning_report)

        bridge_meta = {
            **self.metadata,
            "material": material,
            "path_order": "deterministic non-repeating trail cover with shortest existing-edge zero-E connectors",
            "edge_ids_by_layer": edge_ids_by_layer,
            "trail_partition_status": "planned_from_conformal_structural_graph",
            "trail_partition": planning_reports[0] if planning_reports else {},
            "wall_bead_lanes": self.wall_bead_count,
            "nominal_bead_width_mm": self.nominal_bead_width_mm,
            "core_handoff": "external_layer_paths_v1 XYZ only; downstream Core remains responsible for final XYZABC",
        }
        job_metadata: dict[str, object] = {
            "conformal_lattice_path_bridge": bridge_meta,
            "path_roles": {
                material: {
                    str(layer): ["conformal_honeycomb_macro_partition"] * len(material_paths[layer].paths)
                    for layer in range(len(self.layer_node_positions_xyz))
                }
            },
            "extrusion_compensation": {
                "format": "conformal_edge_volume_e_v1",
                "scope": "per-edge cumulative E from actual 3-D arc length and explicit bead volume",
                "requires_xy_preservation": False,
                "replaces_legacy_arc_length_ratio": True,
            },
        }
        extrusion = self.metadata.get("config", {}).get("extrusion") if isinstance(self.metadata.get("config"), Mapping) else None
        preview_line_width = extrusion.get("preview_line_width_mm") if isinstance(extrusion, Mapping) else None
        if isinstance(preview_line_width, (int, float)) and not isinstance(preview_line_width, bool):
            job_metadata["slicing"] = {"resolved_config": {"line_width": float(preview_line_width)}}
        return ExternalSourceJob(
            material_paths=material_paths,
            travel_paths=travel_paths,
            meta=job_metadata,
        )


def build_conformal_lattice_path_graph(
    geometry: ConformalLatticeGeometry,
    extrusion: ExtrusionVolumeModel,
    *,
    layer_embedding: LayerEmbedding | None = None,
    node_normals_xyz: np.ndarray | None = None,
    wall_bead_count: int = 1,
    nominal_bead_width_mm: float = 2.0,
    config_metadata: Mapping[str, object] | None = None,
) -> ConformalLatticePathGraph:
    """Turn verified structural edges into a deterministic, un-routed graph.

    The geometry's parent/segment IDs are encoded into 64-bit edge IDs.  This
    makes an edge's identity independent of the in-memory row ordering while
    keeping the source edge provenance recoverable without a side lookup.
    """

    _validate_geometry(geometry)
    source_surface_sha256, solver_seed = _required_geometry_provenance(geometry)
    positions, embedding_mode = _layer_positions(geometry, layer_embedding)
    normals = _node_normals_for_paths(geometry, node_normals_xyz)
    if not isinstance(wall_bead_count, int) or isinstance(wall_bead_count, bool) or wall_bead_count < 1:
        raise ValueError("wall_bead_count must be a positive integer")
    if not np.isfinite(nominal_bead_width_mm) or nominal_bead_width_mm <= 0.0:
        raise ValueError("nominal_bead_width_mm must be positive and finite")
    edge_ids, edge_nodes, parent_ids, segment_ids, source_faces = _stable_edges(geometry)
    starts = positions[:, edge_nodes[:, 0], :]
    ends = positions[:, edge_nodes[:, 1], :]
    lengths = np.linalg.norm(ends - starts, axis=2)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 1e-12):
        raise ValueError("validated structural edges must have finite non-zero 3-D length in every output layer")
    volume = lengths * float(extrusion.bead_cross_section_area_mm2)
    cumulative_e = np.stack(
        (np.zeros_like(volume), volume / float(extrusion.e_volume_per_unit_mm3)),
        axis=2,
    )
    config = {
        "layer_embedding_mode": embedding_mode,
        "extrusion": extrusion.metadata(),
        "additional": dict(config_metadata or {}),
    }
    metadata = {
        "format": "conformal_lattice_path_graph_v1",
        "source_surface_sha256": source_surface_sha256,
        "geometry_format": geometry.metadata.get("format"),
        "geometry_config_sha256": geometry.metadata.get("config_sha256"),
        "random_seed": solver_seed,
        "solver_seed": solver_seed,
        "config": config,
        "config_sha256": canonical_json_sha256(config),
        "quality": {
            "zero_length_edge_count": 0,
            "minimum_edge_length_mm": float(np.min(lengths)),
            "maximum_edge_length_mm": float(np.max(lengths)),
            "all_structural_edges_are_two_points": True,
            "edge_identity_unique": True,
        },
    }
    report = {
        "layer_count": int(len(positions)),
        "node_count": int(len(geometry.lattice_nodes_xyz)),
        "edge_count_per_layer": int(len(edge_ids)),
        "total_deposited_volume_mm3": float(np.sum(volume)),
        "trail_partition_status": "planned_when_adapted_to_external_source_job",
        "legacy_surface_mapper_used": False,
        "legacy_honeycomb_pathing_used": False,
    }
    return ConformalLatticePathGraph(
        node_ids=_readonly(np.arange(len(geometry.lattice_nodes_xyz), dtype=np.int64)),
        edge_ids=_readonly(edge_ids),
        edge_node_ids=_readonly(edge_nodes),
        edge_parent_id=_readonly(parent_ids),
        edge_segment_id=_readonly(segment_ids),
        edge_source_triangle_id=_readonly(source_faces),
        layer_node_positions_xyz=_readonly(positions),
        edge_length_mm=_readonly(lengths),
        deposited_volume_mm3=_readonly(volume),
        cumulative_extrusion_e=_readonly(cumulative_e),
        node_normals_xyz=_readonly(normals),
        wall_bead_count=wall_bead_count,
        nominal_bead_width_mm=float(nominal_bead_width_mm),
        report=report,
        metadata=metadata,
    )


def write_conformal_lattice_external_npz(
    graph: ConformalLatticePathGraph,
    output_path: str | Path,
    *,
    material: Literal["R", "F"] = "R",
) -> Path:
    """Write an ``external_layer_paths_v1`` source NPZ from a path graph."""

    destination = Path(output_path)
    job = graph.to_external_source_job(material=material)
    write_external_source_npz(job, destination)
    return destination


def _validate_geometry(geometry: ConformalLatticeGeometry) -> None:
    edge_count = len(geometry.lattice_edges)
    if edge_count == 0:
        raise ValueError("path bridge requires at least one validated structural edge")
    if geometry.lattice_edges.shape != (edge_count, 2):
        raise ValueError("lattice_edges must have shape (edge_count, 2)")
    for name, values in (
        ("lattice_edge_parent_id", geometry.lattice_edge_parent_id),
        ("lattice_edge_segment_id", geometry.lattice_edge_segment_id),
        ("lattice_edge_source_triangle_id", geometry.lattice_edge_source_triangle_id),
    ):
        if np.asarray(values).shape != (edge_count,):
            raise ValueError(f"{name} must contain one value per lattice edge")
    if not np.all((geometry.lattice_edges >= 0) & (geometry.lattice_edges < len(geometry.lattice_nodes_xyz))):
        raise ValueError("lattice edge nodes are outside the lattice node table")


def _required_geometry_provenance(geometry: ConformalLatticeGeometry) -> tuple[str, int]:
    source_surface_sha256 = geometry.metadata.get("source_surface_sha256")
    solver_seed = geometry.metadata.get("random_seed")
    if not isinstance(source_surface_sha256, str) or len(source_surface_sha256) != 64:
        raise ValueError("path bridge requires the geometry source_surface_sha256 provenance")
    if not isinstance(solver_seed, int) or isinstance(solver_seed, bool) or solver_seed < 0:
        raise ValueError("path bridge requires the geometry non-negative random_seed provenance")
    return source_surface_sha256, solver_seed


def _layer_positions(
    geometry: ConformalLatticeGeometry,
    layer_embedding: LayerEmbedding | None,
) -> tuple[np.ndarray, str]:
    if layer_embedding is None:
        return np.asarray(geometry.lattice_nodes_xyz, dtype=np.float64)[None, :, :], "surface_geometry_only"
    positions = np.asarray(layer_embedding.node_positions_xyz, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != geometry.lattice_nodes_xyz.shape or not np.all(np.isfinite(positions)):
        raise ValueError("layer embedding positions must be finite and match geometry lattice nodes")
    if not np.array_equal(layer_embedding.lattice_edges, geometry.lattice_edges):
        raise ValueError("layer embedding must preserve the exact validated lattice edge topology")
    return positions, layer_embedding.mode


def _node_normals_for_paths(
    geometry: ConformalLatticeGeometry,
    supplied: np.ndarray | None,
) -> np.ndarray:
    """Use explicit surface normals for lateral multi-bead lane offsets."""

    if supplied is None:
        # Geometry-only callers retain a deterministic, horizontal fallback.
        normals = np.zeros_like(geometry.lattice_nodes_xyz, dtype=np.float64)
        normals[:, 2] = 1.0
    else:
        normals = np.asarray(supplied, dtype=np.float64)
    if normals.shape != geometry.lattice_nodes_xyz.shape or not np.all(np.isfinite(normals)):
        raise ValueError("node_normals_xyz must be finite and match lattice node positions")
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("node_normals_xyz contains a zero-length normal")
    return normals / lengths[:, None]


def _stable_edges(
    geometry: ConformalLatticeGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent = np.asarray(geometry.lattice_edge_parent_id, dtype=np.int64)
    segment = np.asarray(geometry.lattice_edge_segment_id, dtype=np.int64)
    if np.any(parent < 0) or np.any(parent > _EDGE_ID_COMPONENT_MAX) or np.any(segment < 0) or np.any(segment > _EDGE_ID_COMPONENT_MAX):
        raise ValueError("edge parent and segment IDs must fit the stable 32-bit ID components")
    edge_ids = (parent << _EDGE_ID_BITS) | segment
    if len(np.unique(edge_ids)) != len(edge_ids):
        raise ValueError("geometry contains duplicate parent/segment stable edge IDs")
    raw_nodes = np.asarray(geometry.lattice_edges, dtype=np.int64)
    nodes = np.sort(raw_nodes, axis=1)
    faces = np.asarray(geometry.lattice_edge_source_triangle_id, dtype=np.int64)
    order = np.argsort(edge_ids, kind="stable")
    return edge_ids[order], nodes[order], parent[order], segment[order], faces[order]


def _plan_layer_macro_partitions(
    edge_nodes: np.ndarray,
    positions: np.ndarray,
    normals: np.ndarray,
    *,
    bead_count: int,
    bead_width_mm: float,
    bead_area_mm2: float,
    e_volume_per_unit_mm3: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, object]]:
    """Reuse the legacy exact trail cover on stable node IDs, not STL holes."""

    graph_edges = [
        _Edge((float(first), 0.0), (float(second), 0.0), float(np.linalg.norm(positions[first] - positions[second])))
        for first, second in np.asarray(edge_nodes, dtype=np.int64)
    ]
    trails = [
        [int(round(node[0])) for node in trail]
        for trail in _minimum_trail_cover(graph_edges)
    ]
    adjacency = _node_adjacency(edge_nodes, positions)
    components = _ordered_trail_components(trails, adjacency)
    macro_node_paths: list[tuple[list[int], list[bool]]] = []
    travel_node_paths: list[list[int]] = []
    previous_end: int | None = None
    for component in components:
        node_path, deposited = _join_component_trails(component, adjacency)
        if previous_end is not None:
            # Separate components have no structural-edge route.  Preserve the
            # old macro-partition meaning with an explicit zero-E T motion.
            travel_node_paths.append([previous_end, node_path[0]])
        macro_node_paths.append((node_path, deposited))
        previous_end = node_path[-1]

    paths: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    lane_offsets = [
        (lane - (bead_count - 1) / 2.0) * bead_width_mm
        for lane in range(bead_count)
    ]
    for nodes, deposited in macro_node_paths:
        for offset in lane_offsets:
            lane_points = _offset_lane_points(nodes, positions, normals, offset)
            paths.append(lane_points)
            profiles.append(
                _profile_for_deposition_segments(
                    lane_points,
                    deposited,
                    bead_area_mm2=bead_area_mm2,
                    e_volume_per_unit_mm3=e_volume_per_unit_mm3,
                )
            )
    travels = [positions[np.asarray(nodes, dtype=np.int64)].copy() for nodes in travel_node_paths]
    return paths, profiles, travels, {
        "strategy": "minimum_non_repeating_trail_cover_with_existing_edge_zero_e_connectors",
        "structural_edge_count": int(len(edge_nodes)),
        "minimum_non_repeating_trail_count": int(len(trails)),
        "macro_partition_count": int(len(macro_node_paths)),
        "intra_partition_zero_e_connector_count": int(max(0, len(trails) - len(macro_node_paths))),
        "inter_partition_travel_count": int(len(travel_node_paths)),
        "wall_bead_count": int(bead_count),
        "bead_lane_offsets_mm": lane_offsets,
        "structural_edge_deposition": "each edge is deposited once per 2 mm bead lane; graph-route connector segments keep E unchanged",
    }


def _node_adjacency(edge_nodes: np.ndarray, positions: np.ndarray) -> dict[int, list[tuple[int, float]]]:
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for first, second in np.asarray(edge_nodes, dtype=np.int64):
        length = float(np.linalg.norm(positions[first] - positions[second]))
        adjacency[int(first)].append((int(second), length))
        adjacency[int(second)].append((int(first), length))
    for entries in adjacency.values():
        entries.sort(key=lambda item: item[0])
    return adjacency


def _ordered_trail_components(
    trails: list[list[int]], adjacency: dict[int, list[tuple[int, float]]]
) -> list[list[list[int]]]:
    remaining = list(enumerate(trails))
    components: list[list[list[int]]] = []
    while remaining:
        _index, first = remaining.pop(0)
        nodes = set(first)
        group = [first]
        changed = True
        while changed:
            changed = False
            kept: list[tuple[int, list[int]]] = []
            for index, trail in remaining:
                if any(_node_reachable(node, nodes, adjacency) for node in (trail[0], trail[-1])):
                    group.append(trail)
                    nodes.update(trail)
                    changed = True
                else:
                    kept.append((index, trail))
            remaining = kept
        components.append(group)
    return components


def _node_reachable(source: int, targets: set[int], adjacency: dict[int, list[tuple[int, float]]]) -> bool:
    frontier = [source]
    seen = {source}
    while frontier:
        node = frontier.pop()
        if node in targets:
            return True
        for other, _length in adjacency.get(node, []):
            if other not in seen:
                seen.add(other)
                frontier.append(other)
    return False


def _join_component_trails(
    trails: list[list[int]], adjacency: dict[int, list[tuple[int, float]]]
) -> tuple[list[int], list[bool]]:
    """Deterministically pick shortest graph connectors without redeposition."""

    remaining = [list(trail) for trail in trails]
    current = min(remaining, key=lambda trail: (min(trail[0], trail[-1]), tuple(trail)))
    if current[-1] < current[0]:
        current.reverse()
    remaining.remove(current)
    nodes = list(current)
    deposited = [True] * (len(current) - 1)
    while remaining:
        choices: list[tuple[float, tuple[int, ...], int, list[int], list[int]]] = []
        for index, trail in enumerate(remaining):
            for oriented in (trail, list(reversed(trail))):
                connector = _shortest_node_route(nodes[-1], oriented[0], adjacency)
                if connector is not None:
                    choices.append((
                        _node_route_length(connector, adjacency), tuple(oriented), index, oriented, connector
                    ))
        if not choices:
            raise ValueError("connected conformal lattice trails have no existing-edge connector")
        _length, _tie, index, trail, connector = min(choices, key=lambda item: item[:2])
        nodes.extend(connector[1:])
        deposited.extend([False] * (len(connector) - 1))
        nodes.extend(trail[1:])
        deposited.extend([True] * (len(trail) - 1))
        remaining.pop(index)
    return nodes, deposited


def _shortest_node_route(
    source: int, target: int, adjacency: dict[int, list[tuple[int, float]]]
) -> list[int] | None:
    queue: list[tuple[float, int]] = [(0.0, source)]
    distances = {source: 0.0}
    previous: dict[int, int] = {}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distances[node] + 1e-12:
            continue
        if node == target:
            route = [node]
            while route[-1] != source:
                route.append(previous[route[-1]])
            return list(reversed(route))
        for other, length in adjacency.get(node, []):
            proposal = cost + length
            if proposal + 1e-12 < distances.get(other, float("inf")):
                distances[other] = proposal
                previous[other] = node
                heapq.heappush(queue, (proposal, other))
    return None


def _node_route_length(route: list[int], adjacency: dict[int, list[tuple[int, float]]]) -> float:
    lengths = {(node, other): length for node, entries in adjacency.items() for other, length in entries}
    return sum(lengths[(first, second)] for first, second in zip(route, route[1:]))


def _offset_lane_points(nodes: list[int], positions: np.ndarray, normals: np.ndarray, offset_mm: float) -> np.ndarray:
    raw = positions[np.asarray(nodes, dtype=np.int64)]
    if abs(offset_mm) <= 1e-12:
        return raw.copy()
    result = np.empty_like(raw)
    for index, (node, point) in enumerate(zip(nodes, raw)):
        before = raw[max(0, index - 1)]
        after = raw[min(len(raw) - 1, index + 1)]
        tangent = after - before
        tangent_length = float(np.linalg.norm(tangent))
        lateral = np.cross(normals[node], tangent / tangent_length) if tangent_length > 1e-12 else np.zeros(3)
        lateral_length = float(np.linalg.norm(lateral))
        result[index] = point if lateral_length <= 1e-12 else point + offset_mm * lateral / lateral_length
    return result


def _profile_for_deposition_segments(
    points: np.ndarray,
    deposited: list[bool],
    *,
    bead_area_mm2: float,
    e_volume_per_unit_mm3: float,
) -> np.ndarray:
    if len(deposited) != len(points) - 1:
        raise ValueError("deposition flags must align with planned macro path segments")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    increments = np.where(np.asarray(deposited, dtype=bool), lengths * bead_area_mm2 / e_volume_per_unit_mm3, 0.0)
    return np.concatenate(([0.0], np.cumsum(increments)))


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result
