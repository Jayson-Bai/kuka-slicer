"""Six-fold rotationally symmetric (6-RoSy) surface direction fields."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .mesh_domain import SurfaceMeshDomain


OrientationMode = Literal["global_axis", "principal_curvature"]


@dataclass(frozen=True, slots=True)
class DirectionSingularity:
    vertex: int
    rosy6_winding: int
    index: float
    reason: str


@dataclass(frozen=True, slots=True)
class OrientationField:
    """A direction field represented in local frames by ``exp(i * 6 * theta)``."""

    tangent_vectors_xyz: np.ndarray
    vertex_normals_xyz: np.ndarray
    rosy6: np.ndarray
    principal_curvature_values: np.ndarray | None
    principal_direction_defined: np.ndarray | None
    singularities: tuple[DirectionSingularity, ...]
    report: dict[str, object]

    @property
    def rosy6_real(self) -> np.ndarray:
        return self.rosy6.real

    @property
    def rosy6_imag(self) -> np.ndarray:
        return self.rosy6.imag

    def heatmap_payload(self) -> dict[str, object]:
        return {
            "orientation_rosy6_real": self.rosy6_real.tolist(),
            "orientation_rosy6_imag": self.rosy6_imag.tolist(),
            "tangent_vectors_xyz": self.tangent_vectors_xyz.tolist(),
            "singularities": [
                {
                    "vertex": singularity.vertex,
                    "rosy6_winding": singularity.rosy6_winding,
                    "index": singularity.index,
                    "reason": singularity.reason,
                }
                for singularity in self.singularities
            ],
            "report": self.report,
        }


def build_orientation_field(
    domain: SurfaceMeshDomain,
    *,
    mode: OrientationMode = "global_axis",
    global_axis_xyz: np.ndarray | None = None,
    smoothing_iterations: int = 0,
    constraint_weight: float = 1.0,
) -> OrientationField:
    """Build and optionally smooth a 6-RoSy field in transported tangent frames."""

    if smoothing_iterations < 0:
        raise ValueError("smoothing_iterations must be non-negative")
    if not math.isfinite(constraint_weight) or constraint_weight < 0.0:
        raise ValueError("constraint_weight must be finite and non-negative")
    normals = vertex_normals(domain)
    first_basis, second_basis = tangent_frames(normals)
    if mode == "global_axis":
        axis = np.asarray(global_axis_xyz if global_axis_xyz is not None else [1.0, 0.0, 0.0], dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-14:
            raise ValueError("global_axis_xyz must be a finite non-zero 3D vector")
        tangent = np.asarray([_project_global_axis(axis, normal, first_basis[index]) for index, normal in enumerate(normals)])
        principal_values = None
        defined = None
        source_metadata = {"mode": "global_axis", "global_axis_xyz": axis.tolist()}
    elif mode == "principal_curvature":
        tangent, principal_values, defined = principal_curvature_directions(domain, normals, first_basis, second_basis)
        source_metadata = {
            "mode": "principal_curvature",
            "undefined_principal_direction_vertex_count": int(np.count_nonzero(~defined)),
        }
    else:
        raise ValueError("Gate 3 supports global_axis and principal_curvature orientation modes")
    initial = _vectors_to_rosy6(tangent, first_basis, second_basis)
    rosy6 = smooth_rosy6(
        domain,
        initial,
        first_basis,
        second_basis,
        iterations=smoothing_iterations,
        constraint_weight=constraint_weight,
    )
    tangent = _rosy6_to_vectors(rosy6, first_basis, second_basis)
    singularities = find_rosy6_singularities(domain, rosy6, first_basis, second_basis)
    if defined is not None:
        singularities = tuple(
            sorted(
                (*singularities, *(DirectionSingularity(int(index), 0, 0.0, "principal_direction_undefined") for index in np.flatnonzero(~defined))),
                key=lambda item: (item.vertex, item.reason),
            )
        )
    return OrientationField(
        tangent_vectors_xyz=_readonly(tangent),
        vertex_normals_xyz=_readonly(normals),
        rosy6=_readonly(rosy6),
        principal_curvature_values=_readonly(principal_values) if principal_values is not None else None,
        principal_direction_defined=_readonly(defined) if defined is not None else None,
        singularities=singularities,
        report={
            **source_metadata,
            "smoothing_iterations": smoothing_iterations,
            "constraint_weight": constraint_weight,
            "smoothness_energy": rosy6_smoothness_energy(domain, rosy6, first_basis, second_basis),
            "singularity_count": len(singularities),
        },
    )


def vertex_normals(domain: SurfaceMeshDomain) -> np.ndarray:
    """Compute area-weighted vertex normals from the already oriented faces."""

    normals = np.zeros_like(domain.vertices, dtype=np.float64)
    for face in domain.faces:
        triangle = domain.vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normals[face] += normal
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-14):
        raise ValueError("surface mesh has a vertex with undefined normal")
    return normals / lengths[:, None]


def tangent_frames(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct deterministic right-handed tangent frames at all vertices."""

    first = np.empty_like(normals, dtype=np.float64)
    second = np.empty_like(normals, dtype=np.float64)
    for index, normal in enumerate(normals):
        reference = np.asarray([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.asarray([1.0, 0.0, 0.0])
        basis = reference - np.dot(reference, normal) * normal
        basis /= np.linalg.norm(basis)
        first[index] = basis
        second[index] = np.cross(normal, basis)
    return first, second


def principal_curvature_directions(
    domain: SurfaceMeshDomain,
    normals: np.ndarray,
    first_basis: np.ndarray,
    second_basis: np.ndarray,
    *,
    anisotropy_tolerance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a symmetric normal-variation tensor in each local tangent frame."""

    neighbors = _vertex_neighbors(domain)
    directions = np.empty_like(normals, dtype=np.float64)
    curvatures = np.empty((len(normals), 2), dtype=np.float64)
    defined = np.ones(len(normals), dtype=bool)
    fallback_axis = np.asarray([1.0, 0.0, 0.0])
    for vertex, adjacent in enumerate(neighbors):
        if len(adjacent) < 2:
            directions[vertex] = _project_global_axis(fallback_axis, normals[vertex], first_basis[vertex])
            curvatures[vertex] = 0.0
            defined[vertex] = False
            continue
        position_deltas = domain.vertices[adjacent] - domain.vertices[vertex]
        normal_deltas = normals[adjacent] - normals[vertex]
        position_2d = np.vstack((position_deltas @ first_basis[vertex], position_deltas @ second_basis[vertex]))
        normal_2d = np.vstack((normal_deltas @ first_basis[vertex], normal_deltas @ second_basis[vertex]))
        gram = position_2d @ position_2d.T
        if np.linalg.cond(gram) > 1e10:
            directions[vertex] = _project_global_axis(fallback_axis, normals[vertex], first_basis[vertex])
            curvatures[vertex] = 0.0
            defined[vertex] = False
            continue
        shape_operator = -(normal_2d @ position_2d.T) @ np.linalg.inv(gram)
        shape_operator = 0.5 * (shape_operator + shape_operator.T)
        values, vectors = np.linalg.eigh(shape_operator)
        order = np.argsort(np.abs(values))[::-1]
        values, vectors = values[order], vectors[:, order]
        curvatures[vertex] = values
        if abs(values[0] - values[1]) <= anisotropy_tolerance * max(1.0, abs(values[0]), abs(values[1])):
            directions[vertex] = _project_global_axis(fallback_axis, normals[vertex], first_basis[vertex])
            defined[vertex] = False
        else:
            directions[vertex] = vectors[0, 0] * first_basis[vertex] + vectors[1, 0] * second_basis[vertex]
    return directions, curvatures, defined


def smooth_rosy6(
    domain: SurfaceMeshDomain,
    initial: np.ndarray,
    first_basis: np.ndarray,
    second_basis: np.ndarray,
    *,
    iterations: int,
    constraint_weight: float,
) -> np.ndarray:
    """Smooth complex RoSy values after tangent-plane transport, never raw angles."""

    field = _unit_rosy6(initial)
    if iterations == 0:
        return field
    neighbors = _vertex_neighbors(domain)
    for _ in range(iterations):
        updated = np.empty_like(field)
        for vertex, adjacent in enumerate(neighbors):
            accumulator = constraint_weight * initial[vertex]
            for neighbor in adjacent:
                accumulator += _transport_rosy6(field[neighbor], neighbor, vertex, first_basis, second_basis)
            updated[vertex] = initial[vertex] if abs(accumulator) <= 1e-14 else accumulator / abs(accumulator)
        field = updated
    return field


def rosy6_smoothness_energy(domain: SurfaceMeshDomain, field: np.ndarray, first_basis: np.ndarray, second_basis: np.ndarray) -> float:
    energy = 0.0
    for left, right in _mesh_edges(domain):
        difference = field[left] - _transport_rosy6(field[right], right, left, first_basis, second_basis)
        energy += float(abs(difference) ** 2)
    return energy


def find_rosy6_singularities(
    domain: SurfaceMeshDomain,
    field: np.ndarray,
    first_basis: np.ndarray,
    second_basis: np.ndarray,
) -> tuple[DirectionSingularity, ...]:
    """Estimate discrete RoSy winding around each interior one-ring."""

    neighbors = _vertex_neighbors(domain)
    boundary = {int(vertex) for loop in domain.boundary_loops for vertex in loop}
    singularities = []
    for vertex, adjacent in enumerate(neighbors):
        if vertex in boundary or len(adjacent) < 3:
            continue
        ordered = sorted(
            adjacent,
            key=lambda neighbor: math.atan2(
                np.dot(domain.vertices[neighbor] - domain.vertices[vertex], second_basis[vertex]),
                np.dot(domain.vertices[neighbor] - domain.vertices[vertex], first_basis[vertex]),
            ),
        )
        phases = [math.atan2(_transport_rosy6(field[neighbor], neighbor, vertex, first_basis, second_basis).imag, _transport_rosy6(field[neighbor], neighbor, vertex, first_basis, second_basis).real) for neighbor in ordered]
        winding = round(sum(_wrapped_angle(phases[(index + 1) % len(phases)] - phases[index]) for index in range(len(phases))) / (2.0 * math.pi))
        if winding:
            singularities.append(DirectionSingularity(vertex, int(winding), float(winding) / 6.0, "discrete_rosy6_winding"))
    return tuple(singularities)


def _project_global_axis(axis: np.ndarray, normal: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    tangent = axis - np.dot(axis, normal) * normal
    if np.linalg.norm(tangent) <= 1e-14:
        tangent = fallback
    return tangent / np.linalg.norm(tangent)


def _vectors_to_rosy6(vectors: np.ndarray, first_basis: np.ndarray, second_basis: np.ndarray) -> np.ndarray:
    angle = np.arctan2(np.sum(vectors * second_basis, axis=1), np.sum(vectors * first_basis, axis=1))
    return np.exp(1j * 6.0 * angle)


def _rosy6_to_vectors(field: np.ndarray, first_basis: np.ndarray, second_basis: np.ndarray) -> np.ndarray:
    angle = np.angle(field) / 6.0
    return np.cos(angle)[:, None] * first_basis + np.sin(angle)[:, None] * second_basis


def _transport_rosy6(value: complex, source: int, target: int, first_basis: np.ndarray, second_basis: np.ndarray) -> complex:
    angle = math.atan2(value.imag, value.real) / 6.0
    direction = math.cos(angle) * first_basis[source] + math.sin(angle) * second_basis[source]
    target_direction = direction - np.dot(direction, np.cross(first_basis[target], second_basis[target])) * np.cross(first_basis[target], second_basis[target])
    length = np.linalg.norm(target_direction)
    if length <= 1e-14:
        target_direction = first_basis[target]
    else:
        target_direction /= length
    target_angle = math.atan2(np.dot(target_direction, second_basis[target]), np.dot(target_direction, first_basis[target]))
    return complex(np.exp(1j * 6.0 * target_angle))


def _vertex_neighbors(domain: SurfaceMeshDomain) -> list[np.ndarray]:
    groups: list[set[int]] = [set() for _ in domain.vertices]
    for face in domain.faces:
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            groups[int(first)].add(int(second))
            groups[int(second)].add(int(first))
    return [np.asarray(sorted(group), dtype=np.int64) for group in groups]


def _mesh_edges(domain: SurfaceMeshDomain) -> list[tuple[int, int]]:
    edges = set()
    for face in domain.faces:
        edges.update((tuple(sorted((int(face[0]), int(face[1])))), tuple(sorted((int(face[1]), int(face[2])))), tuple(sorted((int(face[2]), int(face[0]))))))
    return sorted(edges)


def _unit_rosy6(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128)
    if result.ndim != 1 or np.any(np.abs(result) <= 1e-14):
        raise ValueError("rosy6 field must be non-zero at every vertex")
    return result / np.abs(result)


def _wrapped_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values)
    result.setflags(write=False)
    return result
