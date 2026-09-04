"""Phase-domain triangular lattice, hexagonal dual, clipping, and surface map.

This module deliberately creates geometry only.  It neither creates print paths
nor calls the legacy STL-hole honeycomb reconstruction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Mapping

import numpy as np

from .mesh_domain import SurfaceMeshDomain
from .inverse_mapping import PhaseSurfaceInverseMapper, locate_phase_point, phase_point_key, phase_segment_intervals
from .orientation_field import OrientationField
from .parameterization import LSCMParameterization
from .phase_coordinates import PhaseCoordinates
from .scalar_fields import DesignFieldResult


BoundaryMode = Literal["clip", "inset"]
_EPSILON = 1e-10
_LATTICE_BASIS = np.asarray([[1.0, 0.5], [0.0, math.sqrt(3.0) / 2.0]])
_HEXAGON_OFFSETS = np.asarray(
    [[math.cos(math.pi / 6.0 + index * math.pi / 3.0), math.sin(math.pi / 6.0 + index * math.pi / 3.0)] for index in range(6)],
    dtype=np.float64,
) / math.sqrt(3.0)


@dataclass(frozen=True, slots=True)
class ConformalLatticeGeometry:
    """Geometry-sidecar payload for ``conformal_lattice_geometry_v1.npz``."""

    lattice_nodes_phase: np.ndarray
    lattice_nodes_uv: np.ndarray
    lattice_nodes_xyz: np.ndarray
    lattice_edges: np.ndarray
    lattice_edge_parent_id: np.ndarray
    lattice_edge_segment_id: np.ndarray
    lattice_edge_source_triangle_id: np.ndarray
    triangular_lattice_points_phase: np.ndarray
    cell_offsets: np.ndarray
    cell_node_indices: np.ndarray
    cell_parent_id: np.ndarray
    cell_valence: np.ndarray
    cell_defect_code: np.ndarray
    cell_is_boundary: np.ndarray
    source_triangle_id_per_node: np.ndarray
    barycentric_weights_per_node: np.ndarray
    mapping_residual_per_node: np.ndarray
    report: dict[str, object]
    metadata: dict[str, object]

    def save_npz(
        self,
        path: str | Path,
        *,
        domain: SurfaceMeshDomain,
        parameterization: LSCMParameterization,
        design_fields: DesignFieldResult,
        orientation: OrientationField,
        phase: PhaseCoordinates,
        fill_validation: "FillRatioValidation | None" = None,
    ) -> Path:
        """Persist the Gate 5 sidecar without reusing path-layer NPZ contracts."""

        destination = Path(path)
        if destination.name != "conformal_lattice_geometry_v1.npz":
            raise ValueError("Gate 5 output filename must be conformal_lattice_geometry_v1.npz")
        if fill_validation is not None and len(fill_validation.realized_fill_ratio_per_cell) != len(self.cell_valence):
            raise ValueError("fill validation must contain one realized fill ratio per cell")
        realized_fill = (
            np.full(len(self.cell_valence), np.nan, dtype=np.float64)
            if fill_validation is None
            else fill_validation.realized_fill_ratio_per_cell
        )
        meta = {
            **self.metadata,
            "report": self.report,
            "fill_ratio_validation": None if fill_validation is None else fill_validation.report,
        }
        np.savez_compressed(
            destination,
            surface_vertices_xyz=domain.vertices,
            surface_faces=domain.faces,
            surface_uv=parameterization.uv,
            surface_vertex_normals=orientation.vertex_normals_xyz,
            conformal_ratio_per_face=parameterization.quality.conformal_ratio_per_face,
            angle_error_deg_per_face=parameterization.quality.angle_error_deg_per_face,
            area_scale_per_face=parameterization.quality.area_scale_per_face,
            target_fill_ratio_per_vertex=design_fields.target_fill_ratio,
            realized_fill_ratio_per_cell=realized_fill,
            target_cell_size_mm_per_vertex=design_fields.target_cell_size_mm,
            orientation_rosy6_real=orientation.rosy6_real,
            orientation_rosy6_imag=orientation.rosy6_imag,
            phi_p_per_vertex=phase.phi_p,
            phi_q_per_vertex=phase.phi_q,
            phase_integrability_residual_per_face=phase.quality.residual_per_face,
            lattice_nodes_phase=self.lattice_nodes_phase,
            lattice_nodes_uv=self.lattice_nodes_uv,
            lattice_nodes_xyz=self.lattice_nodes_xyz,
            lattice_edges=self.lattice_edges,
            lattice_edge_parent_id=self.lattice_edge_parent_id,
            lattice_edge_segment_id=self.lattice_edge_segment_id,
            lattice_edge_source_triangle_id=self.lattice_edge_source_triangle_id,
            triangular_lattice_points_phase=self.triangular_lattice_points_phase,
            cell_offsets=self.cell_offsets,
            cell_node_indices=self.cell_node_indices,
            cell_parent_id=self.cell_parent_id,
            cell_valence=self.cell_valence,
            cell_defect_code=self.cell_defect_code,
            cell_is_boundary=self.cell_is_boundary,
            source_triangle_id_per_node=self.source_triangle_id_per_node,
            barycentric_weights_per_node=self.barycentric_weights_per_node,
            mapping_residual_per_node=self.mapping_residual_per_node,
            meta=np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
        )
        return destination


def generate_conformal_lattice_geometry(
    domain: SurfaceMeshDomain,
    parameterization: LSCMParameterization,
    design_fields: DesignFieldResult,
    orientation: OrientationField,
    phase: PhaseCoordinates,
    *,
    boundary_mode: BoundaryMode = "clip",
    phase_origin: np.ndarray | tuple[float, float] = (0.0, 0.0),
    random_seed: int = 0,
    config_metadata: Mapping[str, object] | None = None,
) -> ConformalLatticeGeometry:
    """Generate the triangular-lattice Voronoi dual and map it to the surface.

    ``clip`` preserves every boundary-intersecting lattice edge and stores the
    resulting cell fragments; ``inset`` retains only whole hexagons.  The
    source-triangle and barycentric payload makes the UV-to-surface mapping
    auditable and avoids a direct 3D chord shortcut.
    """

    _validate_inputs(domain, parameterization, design_fields, orientation, phase, boundary_mode, phase_origin, random_seed)
    phase_vertices = np.column_stack((phase.phi_p, phase.phi_q))
    origin = np.asarray(phase_origin, dtype=np.float64)
    centers = _triangular_lattice_centers(phase_vertices, origin)
    polygons = centers[:, None, :] + _HEXAGON_OFFSETS[None, :, :]
    abstract_vertices, abstract_edges, cell_abstract_nodes = _dual_hex_topology(polygons)
    edge_intervals = [phase_segment_intervals(abstract_vertices[left], abstract_vertices[right], phase_vertices, domain.faces) for left, right in abstract_edges]
    nodes = PhaseSurfaceInverseMapper(domain, parameterization.uv, phase_vertices)
    abstract_edge_index = {tuple(edge): index for index, edge in enumerate(abstract_edges)}
    edges, edge_parent, edge_segment, edge_face = _map_clipped_edges(abstract_vertices, abstract_edges, edge_intervals, nodes)
    cell_rows: list[list[int]] = []
    cell_parent: list[int] = []
    cell_boundary: list[bool] = []
    full = np.asarray([_cell_is_fully_inside(cell, abstract_edge_index, edge_intervals) for cell in cell_abstract_nodes], dtype=bool)
    for parent_id, abstract_node_ids in enumerate(cell_abstract_nodes):
        if full[parent_id]:
            cell_rows.append([nodes.add(abstract_vertices[node_id]) for node_id in abstract_node_ids])
            cell_parent.append(parent_id)
            cell_boundary.append(False)
        elif boundary_mode == "clip":
            for fragment in _clipped_cell_fragments(polygons[parent_id], phase_vertices, domain.faces):
                cell_rows.append([nodes.add(point) for point in fragment])
                cell_parent.append(parent_id)
                cell_boundary.append(True)
    offsets, indices = _ragged_indices(cell_rows)
    valence = np.asarray([len(row) for row in cell_rows], dtype=np.int64)
    boundary = np.asarray(cell_boundary, dtype=bool)
    defect = np.where(boundary, 1, np.where(valence == 6, 0, 2)).astype(np.int64)
    report = _topology_report(
        domain,
        design_fields,
        orientation,
        phase_vertices,
        centers,
        cell_rows,
        cell_parent,
        boundary,
        defect,
    )
    metadata = _metadata(domain, parameterization, phase, boundary_mode, origin, random_seed, config_metadata)
    return ConformalLatticeGeometry(
        lattice_nodes_phase=_readonly(np.asarray(nodes.phase, dtype=np.float64).reshape((-1, 2))),
        lattice_nodes_uv=_readonly(np.asarray(nodes.uv, dtype=np.float64).reshape((-1, 2))),
        lattice_nodes_xyz=_readonly(np.asarray(nodes.xyz, dtype=np.float64).reshape((-1, 3))),
        lattice_edges=_readonly(np.asarray(edges, dtype=np.int64).reshape((-1, 2))),
        lattice_edge_parent_id=_readonly(np.asarray(edge_parent, dtype=np.int64)),
        lattice_edge_segment_id=_readonly(np.asarray(edge_segment, dtype=np.int64)),
        lattice_edge_source_triangle_id=_readonly(np.asarray(edge_face, dtype=np.int64)),
        triangular_lattice_points_phase=_readonly(centers),
        cell_offsets=_readonly(offsets),
        cell_node_indices=_readonly(indices),
        cell_parent_id=_readonly(np.asarray(cell_parent, dtype=np.int64)),
        cell_valence=_readonly(valence),
        cell_defect_code=_readonly(defect),
        cell_is_boundary=_readonly(boundary),
        source_triangle_id_per_node=_readonly(np.asarray(nodes.face_ids, dtype=np.int64)),
        barycentric_weights_per_node=_readonly(np.asarray(nodes.barycentric, dtype=np.float64).reshape((-1, 3))),
        mapping_residual_per_node=_readonly(np.asarray(nodes.mapping_residual, dtype=np.float64)),
        report=report,
        metadata=metadata,
    )


def _validate_inputs(
    domain: SurfaceMeshDomain,
    parameterization: LSCMParameterization,
    design_fields: DesignFieldResult,
    orientation: OrientationField,
    phase: PhaseCoordinates,
    boundary_mode: str,
    phase_origin: np.ndarray | tuple[float, float],
    random_seed: int,
) -> None:
    count = len(domain.vertices)
    if boundary_mode not in ("clip", "inset"):
        raise ValueError("Gate 5 supports boundary_mode=clip or inset")
    origin = np.asarray(phase_origin, dtype=np.float64)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("phase_origin must contain two finite values")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    if parameterization.uv.shape != (count, 2) or phase.phi_p.shape != (count,) or phase.phi_q.shape != (count,):
        raise ValueError("domain, parameterization, and phase coordinates must share vertices")
    if design_fields.target_fill_ratio.shape != (count,) or orientation.vertex_normals_xyz.shape != (count, 3):
        raise ValueError("design and orientation fields must share the domain vertices")
    if phase.quality.flipped_phase_triangle_count or phase.quality.degenerate_phase_triangle_count or phase.quality.overlapping_phase_face_pairs:
        raise ValueError("Gate 5 cannot continue from an invalid phase map")


def _triangular_lattice_centers(phase_vertices: np.ndarray, origin: np.ndarray) -> np.ndarray:
    inverse_basis = np.linalg.inv(_LATTICE_BASIS)
    bbox_corners = np.asarray(
        [[x, y] for x in (np.min(phase_vertices[:, 0]), np.max(phase_vertices[:, 0])) for y in (np.min(phase_vertices[:, 1]), np.max(phase_vertices[:, 1]))]
    )
    lattice_coordinates = (inverse_basis @ (bbox_corners - origin).T).T
    lower = np.floor(np.min(lattice_coordinates, axis=0)).astype(int) - 2
    upper = np.ceil(np.max(lattice_coordinates, axis=0)).astype(int) + 2
    centers = [origin + _LATTICE_BASIS @ np.asarray([i, j], dtype=np.float64) for i in range(lower[0], upper[0] + 1) for j in range(lower[1], upper[1] + 1)]
    return np.asarray(centers, dtype=np.float64)


def _dual_hex_topology(polygons: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    points: list[np.ndarray] = []
    index: dict[tuple[int, int], int] = {}
    cells: list[list[int]] = []
    edge_set: set[tuple[int, int]] = set()
    for polygon in polygons:
        cell: list[int] = []
        for point in polygon:
            key = phase_point_key(point)
            node_id = index.get(key)
            if node_id is None:
                node_id = len(points)
                index[key] = node_id
                points.append(point)
            cell.append(node_id)
        cells.append(cell)
        for start, end in zip(cell, cell[1:] + cell[:1]):
            edge_set.add((min(start, end), max(start, end)))
    return np.asarray(points, dtype=np.float64), np.asarray(sorted(edge_set), dtype=np.int64), cells


def _map_clipped_edges(
    vertices: np.ndarray,
    abstract_edges: np.ndarray,
    edge_intervals: list[list[tuple[float, float, int]]],
    nodes: PhaseSurfaceInverseMapper,
) -> tuple[list[list[int]], list[int], list[int], list[int]]:
    segments: list[list[int]] = []
    parent_ids: list[int] = []
    segment_ids: list[int] = []
    faces: list[int] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for edge_id, ((left, right), intervals) in enumerate(zip(abstract_edges, edge_intervals)):
        start, direction = vertices[left], vertices[right] - vertices[left]
        segment_number = 0
        for lower, upper, face_id in intervals:
            first, second = start + lower * direction, start + upper * direction
            key = tuple(sorted((phase_point_key(first), phase_point_key(second))))
            if key in seen:
                continue
            seen.add(key)
            segments.append([nodes.add(first), nodes.add(second)])
            parent_ids.append(edge_id)
            segment_ids.append(segment_number)
            faces.append(face_id)
            segment_number += 1
    return segments, parent_ids, segment_ids, faces


def _cell_is_fully_inside(
    cell: list[int],
    abstract_edge_index: Mapping[tuple[int, int], int],
    edge_intervals: list[list[tuple[float, float, int]]],
) -> bool:
    for left, right in zip(cell, cell[1:] + cell[:1]):
        edge_id = abstract_edge_index[(min(left, right), max(left, right))]
        coverage = _merged_coverage(edge_intervals[edge_id])
        if len(coverage) != 1 or coverage[0][0] > _EPSILON or coverage[0][1] < 1.0 - _EPSILON:
            return False
    return True


def _merged_coverage(intervals: list[tuple[float, float, int]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for lower, upper, _ in sorted(intervals):
        if not result or lower > result[-1][1] + _EPSILON:
            result.append((lower, upper))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], upper))
    return result


def _clipped_cell_fragments(polygon: np.ndarray, phase_vertices: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    fragments: list[np.ndarray] = []
    for face in faces:
        fragment = _clip_polygon_to_ccw_triangle(polygon, phase_vertices[face])
        if len(fragment) >= 3 and abs(_polygon_area(fragment)) > _EPSILON:
            fragments.append(fragment)
    return fragments


def _clip_polygon_to_ccw_triangle(polygon: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    output = [point.copy() for point in polygon]
    for edge_start, edge_end in zip(triangle, np.vstack((triangle[1:], triangle[:1]))):
        if not output:
            break
        input_points, output = output, []
        previous = input_points[-1]
        previous_inside = _cross2d(edge_end - edge_start, previous - edge_start) >= -_EPSILON
        for current in input_points:
            current_inside = _cross2d(edge_end - edge_start, current - edge_start) >= -_EPSILON
            if current_inside != previous_inside:
                segment = current - previous
                denominator = _cross2d(edge_end - edge_start, segment)
                if abs(denominator) > _EPSILON:
                    t = -_cross2d(edge_end - edge_start, previous - edge_start) / denominator
                    output.append(previous + t * segment)
            if current_inside:
                output.append(current)
            previous, previous_inside = current, current_inside
    return np.asarray(output, dtype=np.float64)


def _ragged_indices(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for index, row in enumerate(rows):
        offsets[index + 1] = offsets[index] + len(row)
    return offsets, np.asarray([item for row in rows for item in row], dtype=np.int64)


def _topology_report(
    domain: SurfaceMeshDomain,
    design_fields: DesignFieldResult,
    orientation: OrientationField,
    phase_vertices: np.ndarray,
    centers: np.ndarray,
    cell_rows: list[list[int]],
    cell_parent: list[int],
    boundary: np.ndarray,
    defect: np.ndarray,
) -> dict[str, object]:
    interior = ~boundary
    valence = np.asarray([len(row) for row in cell_rows], dtype=np.int64)
    singularity_points = [phase_vertices[item.vertex] for item in orientation.singularities if 0 <= item.vertex < len(phase_vertices)]
    defects = [index for index, code in enumerate(defect) if code != 0]
    near_singularities = 0
    for index in defects:
        parent = cell_parent[index]
        if singularity_points and min(float(np.linalg.norm(centers[parent] - point)) for point in singularity_points) <= 1.0:
            near_singularities += 1
    log_size = np.log(design_fields.target_cell_size_mm)
    density_gradient = float(np.max(log_size) - np.min(log_size))
    face_gradient = np.asarray([np.ptp(log_size[face]) for face in domain.faces], dtype=np.float64)
    threshold = float(np.quantile(face_gradient, 0.75)) if len(face_gradient) else math.inf
    density_gradient_defects = 0
    if threshold > _EPSILON:
        for index in defects:
            face_id, _ = locate_phase_point(centers[cell_parent[index]], phase_vertices, domain.faces)
            if face_id is not None and face_gradient[face_id] >= threshold:
                density_gradient_defects += 1
    return {
        "triangular_lattice_point_count": int(len(centers)),
        "cell_fragment_count": int(len(cell_rows)),
        "interior_cell_count": int(np.count_nonzero(interior)),
        "boundary_clipped_cell_count": int(np.count_nonzero(boundary)),
        "interior_six_valence_ratio": float(np.mean(valence[interior] == 6)) if np.any(interior) else math.nan,
        "five_valence_cell_count": int(np.count_nonzero(valence[interior] == 5)),
        "seven_valence_cell_count": int(np.count_nonzero(valence[interior] == 7)),
        "other_interior_valence_cell_count": int(np.count_nonzero(interior & ~np.isin(valence, [5, 6, 7]))),
        "defect_count_near_orientation_singularity": near_singularities,
        "orientation_singularity_count": len(singularity_points),
        "density_log_cell_size_range": density_gradient,
        "density_gradient_defect_count": density_gradient_defects,
        "seam_defect_count": 0,
        "seam_defect_note": "explicit seam-pair metadata is not available on this Gate 1 domain",
        "source_mesh_seam_edge_count": int(domain.report.get("seam_edge_count", 0)),
        "defect_cell_indices": defects,
        "defect_parent_phase": [centers[cell_parent[index]].tolist() for index in defects],
        "realized_fill_ratio_status": "not_evaluated_until_gate_6",
    }


def _metadata(
    domain: SurfaceMeshDomain,
    parameterization: LSCMParameterization,
    phase: PhaseCoordinates,
    boundary_mode: BoundaryMode,
    phase_origin: np.ndarray,
    random_seed: int,
    config_metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    config = dict(config_metadata or {})
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "format": "conformal_lattice_geometry_v1",
        "source_surface_sha256": domain.input_sha256,
        "parameterization_solver": parameterization.solver,
        "phase_solver": phase.solver,
        "boundary_mode": boundary_mode,
        "phase_origin": phase_origin.tolist(),
        "random_seed": random_seed,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "config": config,
    }


def _polygon_area(points: np.ndarray) -> float:
    return 0.5 * sum(_cross2d(points[index], points[(index + 1) % len(points)]) for index in range(len(points)))


def _cross2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result
