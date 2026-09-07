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
        self._phase_triangles = phase_vertices[domain.faces]
        self._phase_lookup = _PhaseTriangleLookup(self._phase_triangles)
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
        face_id, barycentric = self._phase_lookup.locate(point)
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

    def segment_intervals(self, start: np.ndarray, end: np.ndarray) -> list[tuple[float, float, int]]:
        """Clip one lattice edge using this mapper's cached phase index."""

        return self._phase_lookup.segment_intervals(start, end, self._phase, self._domain.faces)

    def candidate_faces_for_bounds(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Return a conservative ordered face set for a phase-space AABB."""

        return self._phase_lookup.candidate_faces_for_bounds(lower, upper)


def phase_segment_intervals(
    start: np.ndarray,
    end: np.ndarray,
    phase_vertices: np.ndarray,
    faces: np.ndarray,
    *,
    candidate_face_ids: np.ndarray | None = None,
) -> list[tuple[float, float, int]]:
    """Split a phase-space segment into its intersections with source triangles."""

    # The previous per-face Python loop made a normal 150 x 100 mm UI design
    # scale with ``abstract edge count * source triangle count``.  The same
    # half-plane interval clipping is evaluated in one NumPy batch instead.
    direction = end - start
    face_ids = np.arange(len(faces), dtype=np.int64) if candidate_face_ids is None else np.asarray(candidate_face_ids, dtype=np.int64)
    if not len(face_ids):
        return []
    triangles = phase_vertices[faces[face_ids]]
    edge_start = triangles
    edge_end = np.roll(triangles, -1, axis=1)
    edge = edge_end - edge_start
    numerator = _cross2d_batch(edge, start - edge_start)
    slope = _cross2d_batch(edge, direction)
    parallel = np.abs(slope) <= _EPSILON
    rejected = np.any(parallel & (numerator < -_EPSILON), axis=1)
    roots = np.divide(-numerator, slope, out=np.zeros_like(numerator), where=~parallel)
    lower = np.maximum(0.0, np.max(np.where(slope > _EPSILON, roots, -np.inf), axis=1))
    upper = np.minimum(1.0, np.min(np.where(slope < -_EPSILON, roots, np.inf), axis=1))
    valid = ~rejected & (upper - lower > _EPSILON)
    return [(float(lower[index]), float(upper[index]), int(face_ids[index])) for index in np.flatnonzero(valid)]


def locate_phase_point(point: np.ndarray, phase_vertices: np.ndarray, faces: np.ndarray) -> tuple[int | None, np.ndarray | None]:
    """Find a phase triangle deterministically and return its barycentric map."""

    return _locate_phase_point_in_triangles(point, phase_vertices[faces])


def _locate_phase_point_in_triangles(point: np.ndarray, triangles: np.ndarray) -> tuple[int | None, np.ndarray | None]:
    """Vectorised equivalent of checking every phase triangle in face order."""

    first = triangles[:, 0] - triangles[:, 2]
    second = triangles[:, 1] - triangles[:, 2]
    delta = np.asarray(point, dtype=np.float64) - triangles[:, 2]
    determinant = _cross2d_batch(first, second)
    nondegenerate = np.abs(determinant) > _EPSILON
    first_weight = np.divide(
        _cross2d_batch(delta, second), determinant,
        out=np.zeros_like(determinant), where=nondegenerate,
    )
    second_weight = np.divide(
        _cross2d_batch(first, delta), determinant,
        out=np.zeros_like(determinant), where=nondegenerate,
    )
    barycentric = np.column_stack((first_weight, second_weight, 1.0 - first_weight - second_weight))
    inside = nondegenerate & np.all(barycentric >= -_EPSILON, axis=1) & np.all(barycentric <= 1.0 + _EPSILON, axis=1)
    matches = np.flatnonzero(inside)
    if not len(matches):
        return None, None
    face_id = int(matches[0])
    clamped = np.clip(barycentric[face_id], 0.0, 1.0)
    return face_id, clamped / np.sum(clamped)


class _PhaseTriangleLookup:
    """Read-only broad phase for repeated point-to-phase-triangle lookups."""

    def __init__(self, triangles: np.ndarray) -> None:
        self._triangles = triangles
        lower = np.min(triangles, axis=1)
        upper = np.max(triangles, axis=1)
        self._origin = np.min(lower, axis=0)
        extent = np.max(upper, axis=0) - self._origin
        self._resolution = max(1, int(math.ceil(math.sqrt(len(triangles)))))
        self._extent = np.where(extent > _EPSILON, extent, 1.0)
        first_cell = np.floor((lower - self._origin) * self._resolution / self._extent).astype(np.int64)
        last_cell = np.floor((upper - self._origin) * self._resolution / self._extent).astype(np.int64)
        first_cell = np.clip(first_cell, 0, self._resolution - 1)
        last_cell = np.clip(last_cell, 0, self._resolution - 1)
        self._buckets: dict[tuple[int, int], list[int]] = {}
        for face_id, (start, end) in enumerate(zip(first_cell, last_cell)):
            for x in range(int(start[0]), int(end[0]) + 1):
                for y in range(int(start[1]), int(end[1]) + 1):
                    self._buckets.setdefault((x, y), []).append(face_id)

    def locate(self, point: np.ndarray) -> tuple[int | None, np.ndarray | None]:
        cell = np.floor((np.asarray(point, dtype=np.float64) - self._origin) * self._resolution / self._extent).astype(np.int64)
        cell = np.clip(cell, 0, self._resolution - 1)
        for face_id in self._buckets.get((int(cell[0]), int(cell[1])), []):
            barycentric = _barycentric(point, self._triangles[face_id])
            if barycentric is not None and np.all(barycentric >= -_EPSILON) and np.all(barycentric <= 1.0 + _EPSILON):
                clamped = np.clip(barycentric, 0.0, 1.0)
                return face_id, clamped / np.sum(clamped)
        # The fallback keeps the public numerical contract intact if a point
        # lies on an extreme floating-point bucket boundary.
        return _locate_phase_point_in_triangles(point, self._triangles)

    def candidate_faces_for_bounds(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        lower = np.minimum(start, end)
        upper = np.maximum(start, end)
        first = np.floor((lower - self._origin) * self._resolution / self._extent).astype(np.int64)
        last = np.floor((upper - self._origin) * self._resolution / self._extent).astype(np.int64)
        first = np.clip(first, 0, self._resolution - 1)
        last = np.clip(last, 0, self._resolution - 1)
        candidates: set[int] = set()
        for x in range(int(first[0]), int(last[0]) + 1):
            for y in range(int(first[1]), int(last[1]) + 1):
                candidates.update(self._buckets.get((x, y), ()))
        # An empty bucket can only occur outside the known phase extent.  The
        # caller still gets the exact full-domain rejection in that case.
        return np.asarray(sorted(candidates), dtype=np.int64)

    def segment_intervals(self, start: np.ndarray, end: np.ndarray, phase_vertices: np.ndarray, faces: np.ndarray) -> list[tuple[float, float, int]]:
        candidates = self.candidate_faces_for_bounds(start, end)
        return phase_segment_intervals(
            start,
            end,
            phase_vertices,
            faces,
            candidate_face_ids=candidates,
        )


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


def _cross2d_batch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Broadcasting 2D cross product for a leading face/edge dimension."""

    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
