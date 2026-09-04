"""Measured wall-coverage fill ratios on the Gate 5 surface lattice geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .lattice_generator import ConformalLatticeGeometry
from .mesh_domain import SurfaceMeshDomain
from .parameterization import LSCMParameterization
from .phase_coordinates import PhaseCoordinates
from .scalar_fields import DesignFieldResult


_EPSILON = 1e-10


@dataclass(frozen=True, slots=True)
class FillRatioValidation:
    """Local measured coverage and a non-mutating next-iteration suggestion."""

    target_fill_ratio_per_cell: np.ndarray
    realized_fill_ratio_per_cell: np.ndarray
    error_per_cell: np.ndarray
    cell_surface_area_mm2: np.ndarray
    wall_covered_area_mm2: np.ndarray
    evaluation_mask: np.ndarray
    suggested_cell_scale_factor_per_cell: np.ndarray
    report: dict[str, object]

    def heatmap_payload(self) -> dict[str, object]:
        return {
            "target_fill_ratio_per_cell": self.target_fill_ratio_per_cell.tolist(),
            "realized_fill_ratio_per_cell": self.realized_fill_ratio_per_cell.tolist(),
            "fill_ratio_error_per_cell": self.error_per_cell.tolist(),
            "suggested_cell_scale_factor_per_cell": self.suggested_cell_scale_factor_per_cell.tolist(),
            "report": self.report,
        }


def validate_realized_fill_ratio(
    domain: SurfaceMeshDomain,
    parameterization: LSCMParameterization,
    design_fields: DesignFieldResult,
    phase: PhaseCoordinates,
    geometry: ConformalLatticeGeometry,
    *,
    wall_width_mm: float | None = None,
    samples_per_triangle_side: int = 10,
) -> FillRatioValidation:
    """Measure wall coverage inside each generated cell's surface window.

    The wall is the union of radius ``wall_width_mm / 2`` capsules around the
    already mapped Gate 5 edge segments.  Coverage is sampled on each source
    surface triangle, rather than inferred from a nominal hexagon formula.
    The suggested correction is informational; callers must rerun Gates 4--5
    explicitly if they choose to apply it.
    """

    width = _wall_width(design_fields, wall_width_mm)
    if not isinstance(samples_per_triangle_side, int) or samples_per_triangle_side < 2:
        raise ValueError("samples_per_triangle_side must be an integer >= 2")
    _validate_inputs(domain, parameterization, design_fields, phase, geometry)
    phase_vertices = np.column_stack((phase.phi_p, phase.phi_q))
    candidate_edges = _candidate_edges_by_face(domain, geometry)
    cell_count = len(geometry.cell_valence)
    target = np.full(cell_count, np.nan, dtype=np.float64)
    realized = np.full(cell_count, np.nan, dtype=np.float64)
    area = np.zeros(cell_count, dtype=np.float64)
    covered = np.zeros(cell_count, dtype=np.float64)
    for cell_id, node_indices in enumerate(_cell_nodes(geometry)):
        polygon = geometry.lattice_nodes_phase[node_indices]
        for face_id, fragment in _phase_polygon_fragments(polygon, phase_vertices, domain.faces):
            fragment_area, fragment_covered, target_integral = _measure_fragment(
                fragment,
                face_id,
                phase_vertices[domain.faces[face_id]],
                domain,
                design_fields.target_fill_ratio,
                geometry,
                candidate_edges[face_id],
                width,
                samples_per_triangle_side,
            )
            area[cell_id] += fragment_area
            covered[cell_id] += fragment_covered
            target[cell_id] = 0.0 if math.isnan(target[cell_id]) else target[cell_id]
            target[cell_id] += target_integral
        if area[cell_id] > _EPSILON:
            target[cell_id] /= area[cell_id]
            realized[cell_id] = covered[cell_id] / area[cell_id]
    error = realized - target
    valid = np.isfinite(target) & np.isfinite(realized) & (area > _EPSILON)
    correction = np.full(cell_count, np.nan, dtype=np.float64)
    correction[valid] = np.clip(realized[valid] / target[valid], 0.5, 2.0)
    report = _report(geometry, target, realized, error, valid, width, samples_per_triangle_side)
    return FillRatioValidation(
        target_fill_ratio_per_cell=_readonly(target),
        realized_fill_ratio_per_cell=_readonly(realized),
        error_per_cell=_readonly(error),
        cell_surface_area_mm2=_readonly(area),
        wall_covered_area_mm2=_readonly(covered),
        evaluation_mask=_readonly(valid),
        suggested_cell_scale_factor_per_cell=_readonly(correction),
        report=report,
    )


def _wall_width(design_fields: DesignFieldResult, requested: float | None) -> float:
    value = design_fields.report.get("wall_width_mm") if requested is None else requested
    try:
        width = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("wall_width_mm must be declared by the design field or caller") from exc
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("wall_width_mm must be finite and positive")
    return width


def _validate_inputs(
    domain: SurfaceMeshDomain,
    parameterization: LSCMParameterization,
    design_fields: DesignFieldResult,
    phase: PhaseCoordinates,
    geometry: ConformalLatticeGeometry,
) -> None:
    vertex_count = len(domain.vertices)
    if parameterization.uv.shape != (vertex_count, 2) or phase.phi_p.shape != (vertex_count,) or phase.phi_q.shape != (vertex_count,):
        raise ValueError("domain, parameterization, and phase data must share vertices")
    if design_fields.target_fill_ratio.shape != (vertex_count,):
        raise ValueError("design fields must provide target fill ratio per domain vertex")
    if geometry.lattice_nodes_phase.shape != geometry.lattice_nodes_uv.shape or len(geometry.lattice_nodes_xyz) != len(geometry.lattice_nodes_phase):
        raise ValueError("geometry node phase, UV, and XYZ arrays must align")
    if len(geometry.cell_offsets) != len(geometry.cell_valence) + 1:
        raise ValueError("geometry cell offsets are malformed")


def _candidate_edges_by_face(domain: SurfaceMeshDomain, geometry: ConformalLatticeGeometry) -> list[np.ndarray]:
    neighbours = [set([index]) for index in range(len(domain.faces))]
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_id, face in enumerate(domain.faces):
        for left, right in zip(face, np.roll(face, -1)):
            edge_faces.setdefault((min(int(left), int(right)), max(int(left), int(right))), []).append(face_id)
    for adjacent in edge_faces.values():
        for face_id in adjacent:
            neighbours[face_id].update(adjacent)
    source_face = geometry.lattice_edge_source_triangle_id
    return [np.flatnonzero(np.isin(source_face, sorted(adjacent))) for adjacent in neighbours]


def _cell_nodes(geometry: ConformalLatticeGeometry) -> list[np.ndarray]:
    return [geometry.cell_node_indices[geometry.cell_offsets[index] : geometry.cell_offsets[index + 1]] for index in range(len(geometry.cell_valence))]


def _phase_polygon_fragments(polygon: np.ndarray, phase_vertices: np.ndarray, faces: np.ndarray) -> list[tuple[int, np.ndarray]]:
    fragments: list[tuple[int, np.ndarray]] = []
    for face_id, face in enumerate(faces):
        fragment = _clip_polygon_to_triangle(polygon, phase_vertices[face])
        if len(fragment) >= 3 and abs(_polygon_area(fragment)) > _EPSILON:
            fragments.append((face_id, fragment))
    return fragments


def _measure_fragment(
    fragment: np.ndarray,
    face_id: int,
    phase_face: np.ndarray,
    domain: SurfaceMeshDomain,
    target_vertex_values: np.ndarray,
    geometry: ConformalLatticeGeometry,
    candidate_edge_indices: np.ndarray,
    wall_width_mm: float,
    samples_per_triangle_side: int,
) -> tuple[float, float, float]:
    face = domain.faces[face_id]
    fragment_area = 0.0
    fragment_covered = 0.0
    target_integral = 0.0
    for index in range(1, len(fragment) - 1):
        phase_triangle = np.asarray([fragment[0], fragment[index], fragment[index + 1]], dtype=np.float64)
        barycentric_values = [_barycentric(point, phase_face) for point in phase_triangle]
        if any(value is None for value in barycentric_values):
            raise ValueError("cell fragment cannot be mapped to its phase triangle")
        barycentric = np.asarray(barycentric_values, dtype=np.float64)
        xyz_triangle = barycentric @ domain.vertices[face]
        surface_area = 0.5 * float(np.linalg.norm(np.cross(xyz_triangle[1] - xyz_triangle[0], xyz_triangle[2] - xyz_triangle[0])))
        if surface_area <= _EPSILON:
            continue
        target_triangle = barycentric @ target_vertex_values[face]
        sample_count = samples_per_triangle_side * samples_per_triangle_side
        covered_fraction = 0.0
        if len(candidate_edge_indices):
            segment_nodes = geometry.lattice_edges[candidate_edge_indices]
            segments = geometry.lattice_nodes_xyz[segment_nodes]
            covered_fraction = float(
                np.mean([_inside_wall(bary @ xyz_triangle, segments, wall_width_mm / 2.0) for bary in _uniform_triangle_samples(sample_count)])
            )
        fragment_area += surface_area
        fragment_covered += surface_area * covered_fraction
        target_integral += surface_area * float(np.mean(target_triangle))
    return fragment_area, fragment_covered, target_integral


def _inside_wall(point: np.ndarray, segments: np.ndarray, radius: float) -> bool:
    starts, ends = segments[:, 0], segments[:, 1]
    direction = ends - starts
    denominator = np.sum(direction * direction, axis=1)
    projection = np.divide(
        np.sum((point - starts) * direction, axis=1),
        denominator,
        out=np.zeros(len(segments), dtype=np.float64),
        where=denominator > _EPSILON,
    )
    closest = starts + np.clip(projection, 0.0, 1.0)[:, None] * direction
    return bool(np.any(np.sum(np.square(point - closest), axis=1) <= radius * radius))


def _uniform_triangle_samples(count: int) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for index in range(1, count + 1):
        radius = math.sqrt(_halton(index, 2))
        angle = _halton(index, 3)
        samples.append(np.asarray([1.0 - radius, radius * (1.0 - angle), radius * angle], dtype=np.float64))
    return samples


def _halton(index: int, base: int) -> float:
    fraction, value = 1.0, 0.0
    current = index
    while current:
        fraction /= base
        value += fraction * (current % base)
        current //= base
    return value


def _clip_polygon_to_triangle(polygon: np.ndarray, triangle: np.ndarray) -> np.ndarray:
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


def _report(
    geometry: ConformalLatticeGeometry,
    target: np.ndarray,
    realized: np.ndarray,
    error: np.ndarray,
    valid: np.ndarray,
    wall_width_mm: float,
    samples_per_triangle_side: int,
) -> dict[str, object]:
    interior = valid & ~geometry.cell_is_boundary
    return {
        "measurement_model": "union_of_3d_wall_segment_capsules_sampled_on_surface_triangles",
        "wall_width_mm": wall_width_mm,
        "samples_per_triangle_side": samples_per_triangle_side,
        "evaluated_cell_count": int(np.count_nonzero(valid)),
        "interior_evaluated_cell_count": int(np.count_nonzero(interior)),
        "mae": _mean_abs(error[valid]),
        "p95_absolute_error": _p95_abs(error[valid]),
        "max_absolute_error": _max_abs(error[valid]),
        "interior_mae": _mean_abs(error[interior]),
        "interior_p95_absolute_error": _p95_abs(error[interior]),
        "correction_policy": "suggested factors are clipped to [0.5, 2.0] and require an explicit Gate 4--5 rerun",
        "target_not_reused_as_realized": True,
    }


def _mean_abs(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values))) if len(values) else math.nan


def _p95_abs(values: np.ndarray) -> float:
    return float(np.quantile(np.abs(values), 0.95)) if len(values) else math.nan


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if len(values) else math.nan


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    matrix = np.column_stack((triangle[0] - triangle[2], triangle[1] - triangle[2]))
    if abs(float(np.linalg.det(matrix))) <= _EPSILON:
        return None
    first_two = np.linalg.solve(matrix, point - triangle[2])
    return np.asarray([first_two[0], first_two[1], 1.0 - first_two[0] - first_two[1]], dtype=np.float64)


def _polygon_area(points: np.ndarray) -> float:
    return 0.5 * sum(_cross2d(points[index], points[(index + 1) % len(points)]) for index in range(len(points)))


def _cross2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result
