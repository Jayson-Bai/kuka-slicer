"""Sparse least-squares conformal parameterization (LSCM)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy import __version__ as scipy_version
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsmr

from .distortion import ConformalQuality, evaluate_conformal_quality, require_valid_uv
from .contracts import ConformalLatticeSpec
from .mesh_domain import SurfaceMeshDomain


AnchorStrategy = Literal["farthest_boundary_pair", "user"]


@dataclass(frozen=True, slots=True)
class LSCMParameterization:
    """UV coordinates, anchors, solver diagnostics, and quality evidence."""

    uv: np.ndarray
    anchor_vertices: tuple[int, int]
    anchor_xyz: np.ndarray
    anchor_strategy: AnchorStrategy
    solver: dict[str, float | int | str]
    quality: ConformalQuality

    def metadata(self) -> dict[str, object]:
        return {
            "method": "lscm",
            "anchor_strategy": self.anchor_strategy,
            "anchors": [
                {
                    "vertex": int(vertex),
                    "xyz": self.anchor_xyz[index].tolist(),
                    "uv": self.uv[vertex].tolist(),
                }
                for index, vertex in enumerate(self.anchor_vertices)
            ],
            "solver": self.solver,
            "quality": self.quality.summary,
        }

def parameterize_lscm(
    domain: SurfaceMeshDomain,
    *,
    anchors: Sequence[int] | None = None,
    solver_tolerance: float = 1e-10,
    max_conformal_ratio: float | None = None,
) -> LSCMParameterization:
    """Solve an LSCM UV map with two fixed boundary anchors.

    Gate 2 accepts exactly one connected, orientable surface patch with one
    boundary loop.  Closed or multi-boundary domains must first receive an
    explicit seam in Gate 1; no automatic topology fallback occurs here.
    """

    if domain.report.get("connected_component_count") != 1:
        raise ValueError("LSCM requires one connected surface patch")
    if len(domain.boundary_loops) != 1:
        raise ValueError("LSCM requires exactly one boundary loop; provide an explicit seam first")
    if not math.isfinite(solver_tolerance) or not 0.0 < solver_tolerance < 1.0:
        raise ValueError("solver_tolerance must be finite and in (0, 1)")
    if anchors is None:
        selected = farthest_boundary_anchors(domain)
        strategy: AnchorStrategy = "farthest_boundary_pair"
    else:
        selected = _user_anchors(domain, anchors)
        strategy = "user"

    matrix = _lscm_matrix(domain)
    vertex_count = len(domain.vertices)
    fixed_columns = np.asarray([selected[0], selected[1], vertex_count + selected[0], vertex_count + selected[1]], dtype=np.int64)
    fixed_values = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    all_columns = np.arange(2 * vertex_count, dtype=np.int64)
    free = np.setdiff1d(all_columns, fixed_columns, assume_unique=False)
    right_hand_side = -np.asarray(matrix[:, fixed_columns] @ fixed_values).ravel()
    solved = lsmr(matrix[:, free], right_hand_side, atol=solver_tolerance, btol=solver_tolerance)
    solution, stop_code, iterations, residual_norm, normal_residual_norm = solved[:5]
    if int(stop_code) not in (1, 2):
        raise ValueError(f"LSCM sparse solver did not converge (stop code {stop_code})")
    coefficients = np.empty(2 * vertex_count, dtype=np.float64)
    coefficients[free] = solution
    coefficients[fixed_columns] = fixed_values
    uv = np.column_stack((coefficients[:vertex_count], coefficients[vertex_count:]))
    uv = _normalize_orientation(domain, uv)
    quality = evaluate_conformal_quality(domain, uv)
    require_valid_uv(quality, max_conformal_ratio=max_conformal_ratio)
    return LSCMParameterization(
        uv=_readonly(uv),
        anchor_vertices=selected,
        anchor_xyz=_readonly(np.asarray(domain.vertices[list(selected)], dtype=np.float64)),
        anchor_strategy=strategy,
        solver={
            "backend": "scipy.sparse.linalg.lsmr",
            "scipy_version": scipy_version,
            "tolerance": float(solver_tolerance),
            "stop_code": int(stop_code),
            "iterations": int(iterations),
            "residual_norm": float(residual_norm),
            "normal_residual_norm": float(normal_residual_norm),
        },
        quality=quality,
    )


def parameterize_spec_lscm(
    spec: ConformalLatticeSpec,
    domain: SurfaceMeshDomain,
    *,
    solver_tolerance: float = 1e-10,
) -> LSCMParameterization:
    """Parameterize a prepared domain using its declared contract settings."""

    parameterization = spec.parameterization
    if parameterization["method"] != "lscm":
        raise ValueError("Gate 2 only supports parameterization.method=lscm")
    strategy = parameterization["anchor_strategy"]
    anchors = parameterization.get("anchors") if strategy == "user" else None
    limit = spec.quality_limits.get("max_conformal_ratio")
    return parameterize_lscm(
        domain,
        anchors=anchors if isinstance(anchors, list) else None,
        solver_tolerance=solver_tolerance,
        max_conformal_ratio=float(limit) if limit is not None else None,
    )


def farthest_boundary_anchors(domain: SurfaceMeshDomain) -> tuple[int, int]:
    """Select a deterministic geodesically farthest pair on the longest loop."""

    if not domain.boundary_loops:
        raise ValueError("cannot select LSCM anchors without a boundary loop")
    loop = max(domain.boundary_loops, key=lambda candidate: _loop_length(domain.vertices, candidate))
    if len(loop) < 2:
        raise ValueError("boundary loop needs at least two vertices")
    graph = _edge_length_graph(domain)
    distances = dijkstra(graph, directed=False, indices=loop)
    best_distance = -math.inf
    best_pair: tuple[int, int] | None = None
    for left_index, left in enumerate(loop):
        for right in loop[left_index + 1 :]:
            distance = float(distances[left_index, int(right)])
            pair = (int(left), int(right)) if int(left) < int(right) else (int(right), int(left))
            if math.isfinite(distance) and (distance > best_distance + 1e-12 or (abs(distance - best_distance) <= 1e-12 and (best_pair is None or pair < best_pair))):
                best_distance, best_pair = distance, pair
    if best_pair is None:
        raise ValueError("could not find a connected pair of boundary anchors")
    return best_pair


def _user_anchors(domain: SurfaceMeshDomain, anchors: Sequence[int]) -> tuple[int, int]:
    if len(anchors) != 2:
        raise ValueError("LSCM requires exactly two user anchor vertices")
    first, second = (int(value) for value in anchors)
    if first == second or min(first, second) < 0 or max(first, second) >= len(domain.vertices):
        raise ValueError("LSCM user anchors must be two distinct in-range vertices")
    boundary = set(domain.boundary_loops[0].tolist())
    if first not in boundary or second not in boundary:
        raise ValueError("LSCM user anchors must lie on the boundary loop")
    return first, second


def _lscm_matrix(domain: SurfaceMeshDomain):
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    vertex_count = len(domain.vertices)
    for face_index, face in enumerate(domain.faces):
        local = _local_triangle(domain.vertices[face])
        double_area = abs(float(np.linalg.det(local[:, 1:])))
        gradients = np.asarray(
            [
                [local[1, 1] - local[1, 2], local[0, 2] - local[0, 1]],
                [local[1, 2] - local[1, 0], local[0, 0] - local[0, 2]],
                [local[1, 0] - local[1, 1], local[0, 1] - local[0, 0]],
            ],
            dtype=np.float64,
        ) / double_area
        weight = math.sqrt(0.5 * double_area)
        first_row, second_row = 2 * face_index, 2 * face_index + 1
        for local_index, vertex in enumerate(face):
            gradient_x, gradient_y = gradients[local_index]
            rows.extend((first_row, first_row, second_row, second_row))
            columns.extend((int(vertex), vertex_count + int(vertex), int(vertex), vertex_count + int(vertex)))
            values.extend((weight * gradient_x, -weight * gradient_y, weight * gradient_y, weight * gradient_x))
    return coo_matrix((values, (rows, columns)), shape=(2 * len(domain.faces), 2 * vertex_count)).tocsr()


def _local_triangle(triangle: np.ndarray) -> np.ndarray:
    first_edge = triangle[1] - triangle[0]
    second_edge = triangle[2] - triangle[0]
    first_length = float(np.linalg.norm(first_edge))
    if first_length <= 1e-14:
        raise ValueError("cannot parameterize a degenerate surface triangle")
    x = float(np.dot(second_edge, first_edge) / first_length)
    y_squared = float(np.dot(second_edge, second_edge) - x * x)
    y = math.sqrt(max(0.0, y_squared))
    if y <= 1e-14:
        raise ValueError("cannot parameterize a degenerate surface triangle")
    return np.asarray([[0.0, first_length, x], [0.0, 0.0, y]], dtype=np.float64)


def _edge_length_graph(domain: SurfaceMeshDomain):
    lengths: dict[tuple[int, int], float] = {}
    for face in domain.faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = (int(first), int(second)) if first < second else (int(second), int(first))
            lengths[key] = float(np.linalg.norm(domain.vertices[key[1]] - domain.vertices[key[0]]))
    rows, columns, values = [], [], []
    for (first, second), length in lengths.items():
        rows.extend((first, second))
        columns.extend((second, first))
        values.extend((length, length))
    return coo_matrix((values, (rows, columns)), shape=(len(domain.vertices), len(domain.vertices))).tocsr()


def _loop_length(vertices: np.ndarray, loop: np.ndarray) -> float:
    ring = np.append(loop, loop[0])
    return float(np.sum(np.linalg.norm(vertices[ring[1:]] - vertices[ring[:-1]], axis=1)))


def _normalize_orientation(domain: SurfaceMeshDomain, uv: np.ndarray) -> np.ndarray:
    signed_area = []
    for face in domain.faces:
        edge = np.column_stack((uv[face[1]] - uv[face[0]], uv[face[2]] - uv[face[0]]))
        signed_area.append(float(np.linalg.det(edge)))
    if float(np.sum(signed_area)) < 0.0:
        result = np.array(uv, copy=True)
        result[:, 1] *= -1.0
        return result
    return uv


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result
