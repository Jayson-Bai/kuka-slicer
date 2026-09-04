"""Per-triangle distortion metrics for a UV parameterization."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .mesh_domain import SurfaceMeshDomain


@dataclass(frozen=True, slots=True)
class ConformalQuality:
    """Numerical evidence for one surface-to-UV parameterization."""

    conformal_ratio_per_face: np.ndarray
    angle_error_deg_per_face: np.ndarray
    area_scale_per_face: np.ndarray
    uv_signed_area_per_face: np.ndarray
    flipped_face_count: int
    degenerate_uv_face_count: int
    overlapping_uv_face_pairs: tuple[tuple[int, int], ...]
    summary: dict[str, float | int]


def evaluate_conformal_quality(domain: SurfaceMeshDomain, uv: np.ndarray) -> ConformalQuality:
    """Measure UV-to-surface Jacobians and reject malformed input arrays.

    The ratio is ``sigma_1 / sigma_2`` of each face Jacobian from UV to the
    3D triangle's local tangent coordinates.  It is therefore 1 for a locally
    isometric map, while still allowing a uniform local scale change.
    """

    coordinates = np.asarray(uv, dtype=np.float64)
    if coordinates.shape != (len(domain.vertices), 2) or not np.all(np.isfinite(coordinates)):
        raise ValueError("UV coordinates must be a finite (vertex_count, 2) array")
    face_count = len(domain.faces)
    ratio = np.empty(face_count, dtype=np.float64)
    angle_error = np.empty(face_count, dtype=np.float64)
    area_scale = np.empty(face_count, dtype=np.float64)
    signed_area = np.empty(face_count, dtype=np.float64)
    source_areas = np.empty(face_count, dtype=np.float64)
    degenerate = 0

    for face_index, face in enumerate(domain.faces):
        triangle = domain.vertices[face]
        mapped = coordinates[face]
        local = _triangle_local_coordinates(triangle)
        uv_edges = np.column_stack((mapped[1] - mapped[0], mapped[2] - mapped[0]))
        determinant = float(np.linalg.det(uv_edges))
        signed_area[face_index] = 0.5 * determinant
        # ``local`` stores triangle points as columns; its last two columns are
        # the local edge matrix.
        source_areas[face_index] = 0.5 * abs(np.linalg.det(local[:, 1:]))
        if abs(determinant) <= 1e-14:
            ratio[face_index] = math.inf
            angle_error[face_index] = math.inf
            area_scale[face_index] = 0.0
            degenerate += 1
            continue
        source_edges = local[:, 1:]
        jacobian = source_edges @ np.linalg.inv(uv_edges)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        sigma_1, sigma_2 = (float(value) for value in singular_values)
        ratio[face_index] = sigma_1 / sigma_2 if sigma_2 > 0.0 else math.inf
        area_scale[face_index] = sigma_1 * sigma_2
        angle_error[face_index] = _max_angle_error_deg(local.T, mapped)

    flips = int(np.count_nonzero(signed_area < -1e-14))
    overlaps = nonadjacent_triangle_overlap_pairs(coordinates, domain.faces)
    finite_ratio = ratio[np.isfinite(ratio)]
    weights = source_areas[np.isfinite(ratio)]
    summary: dict[str, float | int] = {
        "face_count": int(face_count),
        "flipped_face_count": flips,
        "degenerate_uv_face_count": int(degenerate),
        "overlapping_uv_face_pair_count": int(len(overlaps)),
        "conformal_ratio_area_weighted_mean": _weighted_mean(finite_ratio, weights),
        "conformal_ratio_p95": _quantile(finite_ratio, 0.95),
        "conformal_ratio_p99": _quantile(finite_ratio, 0.99),
        "conformal_ratio_max": float(np.max(ratio)) if len(ratio) else math.nan,
        "angle_error_deg_max": float(np.max(angle_error)) if len(angle_error) else math.nan,
    }
    return ConformalQuality(
        conformal_ratio_per_face=_readonly(ratio),
        angle_error_deg_per_face=_readonly(angle_error),
        area_scale_per_face=_readonly(area_scale),
        uv_signed_area_per_face=_readonly(signed_area),
        flipped_face_count=flips,
        degenerate_uv_face_count=degenerate,
        overlapping_uv_face_pairs=overlaps,
        summary=summary,
    )


def require_valid_uv(quality: ConformalQuality, *, max_conformal_ratio: float | None = None) -> None:
    """Apply the non-negotiable Gate 2 quality checks and optional ratio limit."""

    if quality.degenerate_uv_face_count:
        raise ValueError(f"UV parameterization has {quality.degenerate_uv_face_count} degenerate triangles")
    if quality.flipped_face_count:
        raise ValueError(f"UV parameterization has {quality.flipped_face_count} flipped triangles")
    if quality.overlapping_uv_face_pairs:
        raise ValueError("UV parameterization has non-adjacent triangle overlaps")
    if max_conformal_ratio is not None:
        if not math.isfinite(max_conformal_ratio) or max_conformal_ratio < 1.0:
            raise ValueError("max_conformal_ratio must be finite and >= 1")
        maximum = float(quality.summary["conformal_ratio_max"])
        if maximum > max_conformal_ratio:
            raise ValueError(f"conformal ratio {maximum:.6g} exceeds limit {max_conformal_ratio:.6g}")


def _triangle_local_coordinates(triangle: np.ndarray) -> np.ndarray:
    first_edge = triangle[1] - triangle[0]
    second_edge = triangle[2] - triangle[0]
    first_length = float(np.linalg.norm(first_edge))
    if first_length <= 0.0:
        raise ValueError("surface mesh contains a degenerate triangle")
    x = float(np.dot(second_edge, first_edge) / first_length)
    y_squared = float(np.dot(second_edge, second_edge) - x * x)
    y = math.sqrt(max(0.0, y_squared))
    if y <= 1e-14:
        raise ValueError("surface mesh contains a degenerate triangle")
    return np.asarray([[0.0, first_length, x], [0.0, 0.0, y]], dtype=np.float64)


def _max_angle_error_deg(source: np.ndarray, uv: np.ndarray) -> float:
    source_angles = _triangle_angles_deg(source)
    uv_angles = _triangle_angles_deg(uv)
    return float(np.max(np.abs(source_angles - uv_angles)))


def _triangle_angles_deg(points: np.ndarray) -> np.ndarray:
    lengths = np.asarray(
        [np.linalg.norm(points[1] - points[2]), np.linalg.norm(points[0] - points[2]), np.linalg.norm(points[0] - points[1])],
        dtype=np.float64,
    )
    angles = np.empty(3, dtype=np.float64)
    for index in range(3):
        adjacent = [item for item in range(3) if item != index]
        cosine = (lengths[adjacent[0]] ** 2 + lengths[adjacent[1]] ** 2 - lengths[index] ** 2) / (
            2.0 * lengths[adjacent[0]] * lengths[adjacent[1]]
        )
        angles[index] = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
    return angles


def nonadjacent_triangle_overlap_pairs(coordinates: np.ndarray, faces: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return overlapping pairs after excluding triangles that share a vertex.

    This geometric predicate is intentionally independent of UV semantics so
    later gates can apply the same non-adjacent-overlap rule to phase space.
    """

    points = np.asarray(coordinates, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("coordinates must be a finite (vertex_count, 2) array")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must be a (face_count, 3) array")
    if len(triangles) and (np.any(triangles < 0) or np.any(triangles >= len(points))):
        raise ValueError("faces reference a coordinate outside the array")
    return tuple(_overlap_pairs_2d(points, triangles))


def _overlap_pairs_2d(uv: np.ndarray, faces: np.ndarray) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left, face in enumerate(faces):
        first = uv[face]
        first_min, first_max = np.min(first, axis=0), np.max(first, axis=0)
        for right in range(left + 1, len(faces)):
            if set(face).intersection(faces[right]):
                continue
            second = uv[faces[right]]
            if np.any(first_max < np.min(second, axis=0)) or np.any(np.max(second, axis=0) < first_min):
                continue
            if _triangles_overlap_2d(first, second):
                pairs.append((left, right))
    return pairs


def _triangles_overlap_2d(first: np.ndarray, second: np.ndarray, tolerance: float = 1e-12) -> bool:
    for triangle, other in ((first, second), (second, first)):
        for point in triangle:
            if _point_in_triangle(point, other, tolerance):
                return True
    for triangle, other in ((first, second), (second, first)):
        for index in range(3):
            if _segments_intersect(triangle[index], triangle[(index + 1) % 3], other[index], other[(index + 1) % 3], tolerance):
                return True
    return False


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray, tolerance: float) -> bool:
    signs = []
    for index in range(3):
        start, end = triangle[index], triangle[(index + 1) % 3]
        signs.append((end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0]))
    return min(signs) >= -tolerance or max(signs) <= tolerance


def _segments_intersect(first_start, first_end, second_start, second_end, tolerance: float) -> bool:
    def cross(origin, left, right) -> float:
        return float((left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (right[0] - origin[0]))

    values = (
        cross(first_start, first_end, second_start),
        cross(first_start, first_end, second_end),
        cross(second_start, second_end, first_start),
        cross(second_start, second_end, first_end),
    )
    return min(values[0], values[1]) <= tolerance <= max(values[0], values[1]) and min(values[2], values[3]) <= tolerance <= max(values[2], values[3])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights)) if len(values) else math.nan


def _quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile)) if len(values) else math.nan


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result
