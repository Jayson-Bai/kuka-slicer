"""Least-squares UV phase coordinates for the conformal lattice pipeline.

The two solved scalar fields are *not* paths.  Their gradients provide a
surface-aware coordinate system; Gate 5 will turn its level sets into hexagonal
topology only after this module has accepted the phase-map quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import __version__ as scipy_version
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr

from .distortion import nonadjacent_triangle_overlap_pairs
from .mesh_domain import SurfaceMeshDomain
from .orientation_field import OrientationField
from .parameterization import LSCMParameterization
from .scalar_fields import DesignFieldResult


_PHASE_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class PhaseAnchor:
    """A user-selected phase origin, expressed at one existing mesh vertex."""

    vertex: int
    phi_p: float = 0.0
    phi_q: float = 0.0


@dataclass(frozen=True, slots=True)
class PhaseQuality:
    residual_per_face: np.ndarray
    jacobian_det_per_face: np.ndarray
    flipped_phase_triangle_count: int
    degenerate_phase_triangle_count: int
    overlapping_phase_face_pairs: tuple[tuple[int, int], ...]
    summary: dict[str, float | int]


@dataclass(frozen=True, slots=True)
class PhaseCoordinates:
    """Gate 4 output, prior to any curve tracing or inverse surface map."""

    phi_p: np.ndarray
    phi_q: np.ndarray
    p_per_face: np.ndarray
    q_per_face: np.ndarray
    grad_phi_p_per_face: np.ndarray
    grad_phi_q_per_face: np.ndarray
    physical_cell_size_mm_per_vertex: np.ndarray
    conformal_length_scale_per_vertex: np.ndarray
    cell_size_uv_per_vertex: np.ndarray
    phase_anchor: PhaseAnchor
    solver: dict[str, float | int | str]
    quality: PhaseQuality

    def heatmap_payload(self) -> dict[str, object]:
        return {
            "phi_p": self.phi_p.tolist(),
            "phi_q": self.phi_q.tolist(),
            "phase_residual": self.quality.residual_per_face.tolist(),
            "phase_jacobian_det": self.quality.jacobian_det_per_face.tolist(),
            "cell_size_uv": self.cell_size_uv_per_vertex.tolist(),
            "report": self.quality.summary,
        }


def solve_phase_coordinates(
    domain: SurfaceMeshDomain,
    parameterization: LSCMParameterization,
    design_fields: DesignFieldResult,
    orientation: OrientationField,
    *,
    phase_anchor: PhaseAnchor | None = None,
    cell_size_mm_override: np.ndarray | None = None,
    solver_tolerance: float = 1e-10,
) -> PhaseCoordinates:
    """Fit two scalar phase fields to the requested UV target gradients.

    The objective is ``sum_t A_t (|grad(phi_p)-p|^2 +
    |grad(phi_q)-q|^2)``.  It remains meaningful for a non-integrable target;
    the reported residual must then be considered before continuing to Gate 5.
    Flips, degenerate phase triangles, and non-adjacent phase overlaps are hard
    errors because no later reverse map may be attempted from such a result.
    """

    _validate_inputs(domain, parameterization, design_fields, orientation, cell_size_mm_override, solver_tolerance)
    anchor = _phase_anchor(phase_anchor, len(domain.vertices))
    gradients, uv_areas = _triangle_gradient_operator(parameterization.uv, domain.faces)
    scale = _conformal_length_scale_at_vertices(domain, parameterization)
    physical_cell_size = (
        np.asarray(design_fields.target_cell_size_mm, dtype=np.float64)
        if cell_size_mm_override is None
        else np.asarray(cell_size_mm_override, dtype=np.float64)
    )
    cell_size_uv = physical_cell_size / scale
    p, q = _target_phase_gradients(domain, parameterization.uv, orientation, cell_size_uv)
    phi_p, p_diagnostics = _solve_weighted_phase(gradients, domain.faces, len(domain.vertices), uv_areas, p, anchor.vertex, anchor.phi_p, solver_tolerance)
    phi_q, q_diagnostics = _solve_weighted_phase(gradients, domain.faces, len(domain.vertices), uv_areas, q, anchor.vertex, anchor.phi_q, solver_tolerance)
    grad_phi_p = np.einsum("fvi,fv->fi", gradients, phi_p[domain.faces])
    grad_phi_q = np.einsum("fvi,fv->fi", gradients, phi_q[domain.faces])
    quality = evaluate_phase_quality(domain, phi_p, phi_q, p, q, grad_phi_p, grad_phi_q, uv_areas)
    require_valid_phase(quality)
    return PhaseCoordinates(
        phi_p=_readonly(phi_p),
        phi_q=_readonly(phi_q),
        p_per_face=_readonly(p),
        q_per_face=_readonly(q),
        grad_phi_p_per_face=_readonly(grad_phi_p),
        grad_phi_q_per_face=_readonly(grad_phi_q),
        physical_cell_size_mm_per_vertex=_readonly(physical_cell_size),
        conformal_length_scale_per_vertex=_readonly(scale),
        cell_size_uv_per_vertex=_readonly(cell_size_uv),
        phase_anchor=anchor,
        solver={
            "backend": "scipy.sparse.linalg.lsmr",
            "scipy_version": scipy_version,
            "tolerance": float(solver_tolerance),
            "physical_cell_size_source": "design_field" if cell_size_mm_override is None else "gate_6_override",
            "p_stop_code": p_diagnostics[0],
            "p_iterations": p_diagnostics[1],
            "p_residual_norm": p_diagnostics[2],
            "q_stop_code": q_diagnostics[0],
            "q_iterations": q_diagnostics[1],
            "q_residual_norm": q_diagnostics[2],
        },
        quality=quality,
    )


def evaluate_phase_quality(
    domain: SurfaceMeshDomain,
    phi_p: np.ndarray,
    phi_q: np.ndarray,
    p_per_face: np.ndarray,
    q_per_face: np.ndarray,
    grad_phi_p_per_face: np.ndarray | None = None,
    grad_phi_q_per_face: np.ndarray | None = None,
    uv_areas: np.ndarray | None = None,
) -> PhaseQuality:
    """Measure residual, orientation and global injectivity of a phase map."""

    count = len(domain.vertices)
    p_values = np.asarray(phi_p, dtype=np.float64)
    q_values = np.asarray(phi_q, dtype=np.float64)
    if p_values.shape != (count,) or q_values.shape != (count,) or not np.all(np.isfinite(p_values)) or not np.all(np.isfinite(q_values)):
        raise ValueError("phase coordinates must be finite per-vertex arrays")
    expected = (len(domain.faces), 2)
    targets_p, targets_q = np.asarray(p_per_face, dtype=np.float64), np.asarray(q_per_face, dtype=np.float64)
    if targets_p.shape != expected or targets_q.shape != expected:
        raise ValueError("phase targets must be per-face two-dimensional arrays")
    if grad_phi_p_per_face is None or grad_phi_q_per_face is None or uv_areas is None:
        # This public helper is also used when diagnosing a candidate map.
        raise ValueError("phase quality requires per-face gradients and UV areas from the parameterization")
    gradient_p = np.asarray(grad_phi_p_per_face, dtype=np.float64)
    gradient_q = np.asarray(grad_phi_q_per_face, dtype=np.float64)
    weights = np.asarray(uv_areas, dtype=np.float64)
    if gradient_p.shape != expected or gradient_q.shape != expected or weights.shape != (len(domain.faces),):
        raise ValueError("phase gradients and UV areas must match the mesh faces")
    residual = np.sqrt(np.sum(np.square(gradient_p - targets_p) + np.square(gradient_q - targets_q), axis=1))
    jacobian = gradient_p[:, 0] * gradient_q[:, 1] - gradient_p[:, 1] * gradient_q[:, 0]
    flipped = int(np.count_nonzero(jacobian < -_PHASE_EPSILON))
    degenerate = int(np.count_nonzero(np.abs(jacobian) <= _PHASE_EPSILON))
    overlaps = nonadjacent_triangle_overlap_pairs(np.column_stack((p_values, q_values)), domain.faces)
    weighted_square = float(np.sum(weights * np.square(residual)))
    total_area = float(np.sum(weights))
    summary: dict[str, float | int] = {
        "face_count": int(len(domain.faces)),
        "integral_residual": weighted_square,
        "integral_residual_per_uv_area": weighted_square / total_area if total_area > 0.0 else math.nan,
        "rms_residual_per_uv_area": math.sqrt(weighted_square / total_area) if total_area > 0.0 else math.nan,
        "max_residual": float(np.max(residual)) if len(residual) else math.nan,
        "phase_jacobian_det_min": float(np.min(jacobian)) if len(jacobian) else math.nan,
        "phase_jacobian_det_max": float(np.max(jacobian)) if len(jacobian) else math.nan,
        "flipped_phase_triangle_count": flipped,
        "degenerate_phase_triangle_count": degenerate,
        "overlapping_phase_face_pair_count": int(len(overlaps)),
    }
    return PhaseQuality(
        residual_per_face=_readonly(residual),
        jacobian_det_per_face=_readonly(jacobian),
        flipped_phase_triangle_count=flipped,
        degenerate_phase_triangle_count=degenerate,
        overlapping_phase_face_pairs=overlaps,
        summary=summary,
    )


def require_valid_phase(quality: PhaseQuality) -> None:
    """Reject phase maps that cannot safely proceed to inverse mapping."""

    remediation = "reduce density gradient, increase field smoothing, or modify phase/orientation constraints"
    if quality.degenerate_phase_triangle_count:
        raise ValueError(f"phase map has {quality.degenerate_phase_triangle_count} degenerate triangles; {remediation}")
    if quality.flipped_phase_triangle_count:
        raise ValueError(f"phase map has {quality.flipped_phase_triangle_count} flipped triangles; {remediation}")
    if quality.overlapping_phase_face_pairs:
        raise ValueError(f"phase map has non-adjacent triangle overlaps; {remediation}")


def _validate_inputs(
    domain: SurfaceMeshDomain,
    parameterization: LSCMParameterization,
    design_fields: DesignFieldResult,
    orientation: OrientationField,
    cell_size_mm_override: np.ndarray | None,
    solver_tolerance: float,
) -> None:
    count = len(domain.vertices)
    if parameterization.uv.shape != (count, 2):
        raise ValueError("parameterization must belong to the supplied domain")
    if parameterization.quality.flipped_face_count or parameterization.quality.degenerate_uv_face_count or parameterization.quality.overlapping_uv_face_pairs:
        raise ValueError("phase coordinates require a valid Gate 2 UV parameterization")
    if design_fields.target_cell_size_mm.shape != (count,) or np.any(design_fields.target_cell_size_mm <= 0.0):
        raise ValueError("design field must provide one positive physical cell size per vertex")
    if cell_size_mm_override is not None:
        override = np.asarray(cell_size_mm_override, dtype=np.float64)
        if override.shape != (count,) or not np.all(np.isfinite(override)) or np.any(override <= 0.0):
            raise ValueError("cell_size_mm_override must be a finite positive per-vertex array")
    if orientation.tangent_vectors_xyz.shape != (count, 3):
        raise ValueError("orientation field must belong to the supplied domain")
    if not math.isfinite(solver_tolerance) or not 0.0 < solver_tolerance < 1.0:
        raise ValueError("solver_tolerance must be finite and in (0, 1)")


def _phase_anchor(anchor: PhaseAnchor | None, vertex_count: int) -> PhaseAnchor:
    selected = PhaseAnchor(0) if anchor is None else anchor
    if not 0 <= selected.vertex < vertex_count:
        raise ValueError("phase anchor vertex is out of range")
    if not math.isfinite(selected.phi_p) or not math.isfinite(selected.phi_q):
        raise ValueError("phase anchor values must be finite")
    return selected


def _triangle_gradient_operator(uv: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    operator = np.empty((len(faces), 3, 2), dtype=np.float64)
    areas = np.empty(len(faces), dtype=np.float64)
    for face_index, face in enumerate(faces):
        triangle = uv[face]
        double_area = _cross2d(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if double_area <= _PHASE_EPSILON:
            raise ValueError("phase coordinates require consistently oriented non-degenerate UV triangles")
        operator[face_index, 0] = np.asarray([triangle[1, 1] - triangle[2, 1], triangle[2, 0] - triangle[1, 0]]) / double_area
        operator[face_index, 1] = np.asarray([triangle[2, 1] - triangle[0, 1], triangle[0, 0] - triangle[2, 0]]) / double_area
        operator[face_index, 2] = np.asarray([triangle[0, 1] - triangle[1, 1], triangle[1, 0] - triangle[0, 0]]) / double_area
        areas[face_index] = 0.5 * double_area
    return operator, areas


def _conformal_length_scale_at_vertices(domain: SurfaceMeshDomain, parameterization: LSCMParameterization) -> np.ndarray:
    face_scale = np.sqrt(np.asarray(parameterization.quality.area_scale_per_face, dtype=np.float64))
    if face_scale.shape != (len(domain.faces),) or np.any(~np.isfinite(face_scale)) or np.any(face_scale <= 0.0):
        raise ValueError("Gate 2 conformal area scale must be finite and positive")
    weights = np.zeros(len(domain.vertices), dtype=np.float64)
    totals = np.zeros(len(domain.vertices), dtype=np.float64)
    for face_index, face in enumerate(domain.faces):
        triangle = parameterization.uv[face]
        area = 0.5 * abs(_cross2d(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        totals[face] += area * face_scale[face_index]
        weights[face] += area
    scale = totals / weights
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("could not derive a positive conformal length scale at every vertex")
    return scale


def _target_phase_gradients(
    domain: SurfaceMeshDomain,
    uv: np.ndarray,
    orientation: OrientationField,
    cell_size_uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.empty((len(domain.faces), 2), dtype=np.float64)
    q = np.empty_like(p)
    for face_index, face in enumerate(domain.faces):
        direction_xyz = np.sum(orientation.tangent_vectors_xyz[face], axis=0)
        normal = np.cross(domain.vertices[face[1]] - domain.vertices[face[0]], domain.vertices[face[2]] - domain.vertices[face[0]])
        normal /= np.linalg.norm(normal)
        direction_xyz -= normal * float(np.dot(direction_xyz, normal))
        if np.linalg.norm(direction_xyz) <= _PHASE_EPSILON:
            direction_xyz = domain.vertices[face[1]] - domain.vertices[face[0]]
        direction_xyz /= np.linalg.norm(direction_xyz)
        source_edges = np.column_stack((domain.vertices[face[1]] - domain.vertices[face[0]], domain.vertices[face[2]] - domain.vertices[face[0]]))
        uv_edges = np.column_stack((uv[face[1]] - uv[face[0]], uv[face[2]] - uv[face[0]]))
        uv_direction = np.linalg.lstsq(source_edges @ np.linalg.inv(uv_edges), direction_xyz, rcond=None)[0]
        uv_direction /= np.linalg.norm(uv_direction)
        face_cell_size = float(np.mean(cell_size_uv[face]))
        p[face_index] = uv_direction / face_cell_size
        q[face_index] = np.asarray([-uv_direction[1], uv_direction[0]]) / face_cell_size
    return p, q


def _solve_weighted_phase(
    gradients: np.ndarray,
    faces: np.ndarray,
    vertex_count: int,
    areas: np.ndarray,
    target: np.ndarray,
    anchor_vertex: int,
    anchor_value: float,
    tolerance: float,
) -> tuple[np.ndarray, tuple[int, int, float]]:
    face_count = len(faces)
    rows = np.repeat(np.arange(2 * face_count, dtype=np.int64), 3)
    columns = np.empty(6 * face_count, dtype=np.int64)
    values = np.empty(6 * face_count, dtype=np.float64)
    for face_index, face in enumerate(faces):
        columns[6 * face_index : 6 * face_index + 3] = face
        columns[6 * face_index + 3 : 6 * face_index + 6] = face
        values[6 * face_index : 6 * face_index + 3] = gradients[face_index, :, 0]
        values[6 * face_index + 3 : 6 * face_index + 6] = gradients[face_index, :, 1]
    matrix = csr_matrix((values, (rows, columns)), shape=(2 * face_count, vertex_count))
    row_weights = np.repeat(np.sqrt(areas), 2)
    weighted_matrix = matrix.multiply(row_weights[:, None]).tocsr()
    weighted_target = target.reshape(-1) * row_weights
    free = np.delete(np.arange(vertex_count, dtype=np.int64), anchor_vertex)
    right_hand_side = weighted_target - weighted_matrix[:, anchor_vertex].toarray().ravel() * anchor_value
    # A non-integrable target is expected to retain a residual.  Give LSMR
    # enough iterations to reach its least-squares stopping criterion instead
    # of mistaking the default ``min(m, n)`` cap for a numerical failure.
    solved = lsmr(
        weighted_matrix[:, free],
        right_hand_side,
        atol=tolerance,
        btol=tolerance,
        maxiter=max(100, 10 * vertex_count),
    )
    solution, stop_code, iterations, residual_norm = solved[:4]
    if int(stop_code) not in (1, 2):
        raise ValueError(f"phase sparse solver did not converge (stop code {stop_code})")
    result = np.empty(vertex_count, dtype=np.float64)
    result[free] = solution
    result[anchor_vertex] = anchor_value
    return result, (int(stop_code), int(iterations), float(residual_norm))


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


def _cross2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])
