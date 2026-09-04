"""Auditable phase-to-UV-to-surface barycentric inverse mapping."""

from __future__ import annotations

import math

import numpy as np

from .mesh_domain import SurfaceMeshDomain


_EPSILON = 1e-10


class PhaseSurfaceInverseMapper:
    """Deduplicate phase nodes while retaining their explicit face provenance."""

    def __init__(self, domain: SurfaceMeshDomain, uv: np.ndarray, phase_vertices: np.ndarray) -> None:
        self._domain, self._uv, self._phase = domain, uv, phase_vertices
        self._indices: dict[tuple[int, int], int] = {}
        self.phase: list[np.ndarray] = []
        self.uv: list[np.ndarray] = []
        self.xyz: list[np.ndarray] = []
        self.face_ids: list[int] = []
        self.barycentric: list[np.ndarray] = []
        self.mapping_residual: list[float] = []

    def add(self, point: np.ndarray) -> int:
        key = phase_point_key(point)
        current = self._indices.get(key)
        if current is not None:
            return current
        face_id, barycentric = locate_phase_point(point, self._phase, self._domain.faces)
        if face_id is None or barycentric is None:
            raise ValueError("lattice node lies outside the valid phase domain")
        face = self._domain.faces[face_id]
        current = len(self.phase)
        self._indices[key] = current
        self.phase.append(np.asarray(point, dtype=np.float64))
        self.uv.append(barycentric @ self._uv[face])
        self.xyz.append(barycentric @ self._domain.vertices[face])
        self.face_ids.append(face_id)
        self.barycentric.append(barycentric)
        self.mapping_residual.append(float(np.linalg.norm(point - barycentric @ self._phase[face])))
        return current


def phase_segment_intervals(start: np.ndarray, end: np.ndarray, phase_vertices: np.ndarray, faces: np.ndarray) -> list[tuple[float, float, int]]:
    """Split a phase-space segment into its intersections with source triangles."""

    direction = end - start
    intervals: list[tuple[float, float, int]] = []
    for face_index, face in enumerate(faces):
        triangle = phase_vertices[face]
        lower, upper = 0.0, 1.0
        for edge_start, edge_end in zip(triangle, np.vstack((triangle[1:], triangle[:1]))):
            numerator = _cross2d(edge_end - edge_start, start - edge_start)
            slope = _cross2d(edge_end - edge_start, direction)
            if abs(slope) <= _EPSILON:
                if numerator < -_EPSILON:
                    lower, upper = 1.0, 0.0
                    break
            elif slope > 0.0:
                lower = max(lower, -numerator / slope)
            else:
                upper = min(upper, -numerator / slope)
            if lower > upper + _EPSILON:
                break
        if upper - lower > _EPSILON:
            intervals.append((max(0.0, lower), min(1.0, upper), face_index))
    return intervals


def locate_phase_point(point: np.ndarray, phase_vertices: np.ndarray, faces: np.ndarray) -> tuple[int | None, np.ndarray | None]:
    """Find a phase triangle deterministically and return its barycentric map."""

    for face_id, face in enumerate(faces):
        barycentric = _barycentric(point, phase_vertices[face])
        if barycentric is not None and np.all(barycentric >= -_EPSILON) and np.all(barycentric <= 1.0 + _EPSILON):
            clamped = np.clip(barycentric, 0.0, 1.0)
            return face_id, clamped / np.sum(clamped)
    return None, None


def phase_point_key(point: np.ndarray) -> tuple[int, int]:
    return tuple(np.rint(np.asarray(point, dtype=np.float64) * 1e10).astype(np.int64))  # type: ignore[return-value]


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    matrix = np.column_stack((triangle[0] - triangle[2], triangle[1] - triangle[2]))
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) <= _EPSILON:
        return None
    first_two = np.linalg.solve(matrix, point - triangle[2])
    return np.asarray([first_two[0], first_two[1], 1.0 - first_two[0] - first_two[1]], dtype=np.float64)


def _cross2d(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])
