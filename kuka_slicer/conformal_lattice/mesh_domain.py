"""Triangular target-surface preparation for the conformal-lattice pipeline.

The code in this module only creates and validates an indexed surface domain.
It intentionally contains no UV solve, lattice construction, path ordering, or
printer-specific output.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..stl_io import load_stl_bytes
from ..surface_preview.model import DoubleSineSurface
from .contracts import ConformalLatticeSpec, double_sine_source_sha256


@dataclass(frozen=True, slots=True)
class SurfaceMeshDomain:
    """A clean, consistently oriented triangular surface and its provenance."""

    vertices: np.ndarray
    faces: np.ndarray
    boundary_loops: tuple[np.ndarray, ...]
    source_vertex_index: np.ndarray
    source_face_index: np.ndarray
    report: dict[str, object]

    @property
    def input_sha256(self) -> str:
        return str(self.report["input_sha256"])


def prepare_surface_mesh_domain(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    input_sha256: str | None = None,
    merge_tolerance_mm: float = 1e-9,
    area_tolerance_mm2: float = 1e-12,
) -> SurfaceMeshDomain:
    """Clean, orient, and validate an external indexed triangle mesh.

    Duplicate vertices within ``merge_tolerance_mm`` are merged, zero-area
    faces are removed, and every later topology decision is recorded in the
    returned report.  Non-manifold vertices/edges, duplicate faces and
    self-intersections are hard failures rather than silent repairs.
    """

    source_vertices = _vertices(vertices)
    source_faces = _faces(faces, len(source_vertices))
    if merge_tolerance_mm <= 0.0:
        raise ValueError("merge_tolerance_mm must be positive")
    if area_tolerance_mm2 < 0.0:
        raise ValueError("area_tolerance_mm2 must be non-negative")

    cleaned_vertices, vertex_map = _merge_vertices(source_vertices, merge_tolerance_mm)
    cleaned_faces = vertex_map[source_faces]
    retained = _nondegenerate_faces(cleaned_vertices, cleaned_faces, area_tolerance_mm2)
    cleaned_faces = cleaned_faces[retained]
    source_face_index = np.flatnonzero(retained).astype(np.int64)
    if not len(cleaned_faces):
        raise ValueError("mesh has no non-degenerate triangles after cleanup")
    _reject_duplicate_faces(cleaned_faces)
    oriented_faces, orientation_flips = _orient_faces(cleaned_faces)
    topology = _topology(oriented_faces, len(cleaned_vertices))
    _reject_nonmanifold(topology)
    intersections = _self_intersection_pairs(cleaned_vertices, oriented_faces)
    if intersections:
        preview = ", ".join(f"({left},{right})" for left, right in intersections[:5])
        raise ValueError(f"mesh has self-intersecting non-adjacent triangles: {preview}")

    return SurfaceMeshDomain(
        vertices=_readonly(cleaned_vertices),
        faces=_readonly(oriented_faces),
        boundary_loops=tuple(_readonly(loop) for loop in topology["boundary_loops"]),
        source_vertex_index=_readonly(_first_source_index(vertex_map, len(cleaned_vertices))),
        source_face_index=_readonly(source_face_index),
        report={
            "input_sha256": input_sha256 or indexed_mesh_sha256(source_vertices, source_faces),
            "input_vertex_count": int(len(source_vertices)),
            "input_face_count": int(len(source_faces)),
            "vertex_count": int(len(cleaned_vertices)),
            "face_count": int(len(oriented_faces)),
            "merged_vertex_count": int(len(source_vertices) - len(cleaned_vertices)),
            "removed_degenerate_face_count": int(len(source_faces) - len(oriented_faces)),
            "orientation_flipped_face_count": int(orientation_flips),
            "connected_component_count": int(topology["component_count"]),
            "boundary_loop_count": int(len(topology["boundary_loops"])),
            "boundary_edge_count": int(topology["boundary_edge_count"]),
            "nonmanifold_edge_count": 0,
            "nonmanifold_vertex_count": 0,
            "duplicate_face_count": 0,
            "self_intersection_count": 0,
            "seam_edge_count": 0,
        },
    )


def cut_mesh_along_edges(
    domain: SurfaceMeshDomain,
    seam_edges: Sequence[Sequence[int]],
) -> SurfaceMeshDomain:
    """Create explicit seams by duplicating vertices on selected interior edges.

    Edge indices refer to the cleaned ``domain.vertices`` array.  The returned
    domain may have several charts; a later parameterization gate must reject
    unsupported topology rather than silently select one chart.
    """

    seam_keys = {_edge_key(edge) for edge in seam_edges}
    if not seam_keys:
        return domain
    edge_faces = _edge_faces(domain.faces)
    unknown = sorted(key for key in seam_keys if key not in edge_faces)
    if unknown:
        raise ValueError(f"seam edge is not present in the mesh: {unknown[0]}")
    boundary = sorted(key for key in seam_keys if len(edge_faces[key]) != 2)
    if boundary:
        raise ValueError(f"seam edge must be an interior manifold edge: {boundary[0]}")

    parent = list(range(len(domain.faces)))
    for key, adjacent in edge_faces.items():
        if len(adjacent) == 2 and key not in seam_keys:
            _union(parent, adjacent[0], adjacent[1])
    chart_for_face = np.asarray([_find(parent, face) for face in range(len(domain.faces))], dtype=np.int64)
    chart_ids = {value: index for index, value in enumerate(sorted(set(chart_for_face.tolist())))}
    chart_for_face = np.asarray([chart_ids[value] for value in chart_for_face], dtype=np.int64)

    vertex_lookup: dict[tuple[int, int], int] = {}
    vertices: list[np.ndarray] = []
    source_vertices: list[int] = []
    cut_faces = np.empty_like(domain.faces)
    for face_index, face in enumerate(domain.faces):
        chart = int(chart_for_face[face_index])
        for corner, old_vertex in enumerate(face):
            key = (chart, int(old_vertex))
            new_vertex = vertex_lookup.get(key)
            if new_vertex is None:
                new_vertex = len(vertices)
                vertex_lookup[key] = new_vertex
                vertices.append(domain.vertices[old_vertex])
                source_vertices.append(int(domain.source_vertex_index[old_vertex]))
            cut_faces[face_index, corner] = new_vertex

    cut_vertices = np.asarray(vertices, dtype=np.float64)
    topology = _topology(cut_faces, len(cut_vertices))
    _reject_nonmanifold(topology)
    return SurfaceMeshDomain(
        vertices=_readonly(cut_vertices),
        faces=_readonly(cut_faces),
        boundary_loops=tuple(_readonly(loop) for loop in topology["boundary_loops"]),
        source_vertex_index=_readonly(np.asarray(source_vertices, dtype=np.int64)),
        source_face_index=domain.source_face_index,
        report={
            **domain.report,
            "vertex_count": int(len(cut_vertices)),
            "face_count": int(len(cut_faces)),
            "connected_component_count": int(topology["component_count"]),
            "boundary_loop_count": int(len(topology["boundary_loops"])),
            "boundary_edge_count": int(topology["boundary_edge_count"]),
            "seam_edge_count": int(len(seam_keys)),
            "cut_vertex_count": int(len(cut_vertices) - len(domain.vertices)),
        },
    )


def build_double_sine_surface_domain(spec: ConformalLatticeSpec) -> SurfaceMeshDomain:
    """Triangulate the spec's analytical height field with deterministic indexing."""

    if spec.source_provider != "double_sine":
        raise ValueError("double-sine domain construction requires a double_sine source")
    source = spec.source_surface
    expected_hash = double_sine_source_sha256(source)
    if spec.source_sha256 != expected_hash:
        raise ValueError("double_sine source_surface.sha256 does not match its canonical configuration")
    values = source["double_sine"]
    if not isinstance(values, dict):  # guarded by the contract loader
        raise ValueError("source_surface.double_sine must be an object")
    surface = DoubleSineSurface(
        amplitude_mm=float(values["amplitude_mm"]),
        wavelength_x_mm=float(values["wavelength_x_mm"]),
        wavelength_y_mm=float(values["wavelength_y_mm"]),
        phase_x_rad=float(values["phase_x_rad"]),
        phase_y_rad=float(values["phase_y_rad"]),
        z_reference_mm=float(values["z_reference_mm"]),
    )
    x_min, y_min, x_max, y_max = (float(value) for value in values["xy_bounds_mm"])
    samples_x, samples_y = (int(value) for value in values["samples"])
    x = np.linspace(x_min, x_max, samples_x)
    y = np.linspace(y_min, y_max, samples_y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    vertices = np.column_stack((xx.ravel(), yy.ravel(), surface.height(xx, yy).ravel()))
    faces: list[list[int]] = []
    for row in range(samples_y - 1):
        for column in range(samples_x - 1):
            bottom_left = row * samples_x + column
            bottom_right = bottom_left + 1
            top_left = bottom_left + samples_x
            top_right = top_left + 1
            faces.extend(([bottom_left, bottom_right, top_right], [bottom_left, top_right, top_left]))
    domain = prepare_surface_mesh_domain(vertices, np.asarray(faces, dtype=np.int64), input_sha256=spec.source_sha256)
    return _with_report(domain, {"provider": "double_sine", "source_file": spec.source_file, "generated": True})


def load_triangle_mesh_domain(spec: ConformalLatticeSpec, source_bytes: bytes) -> SurfaceMeshDomain:
    """Load an external STL or indexed NPZ mesh and verify its declared hash."""

    if spec.source_provider != "triangle_mesh":
        raise ValueError("external mesh loading requires a triangle_mesh source")
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != spec.source_sha256:
        raise ValueError("external triangle-mesh SHA-256 does not match source_surface.sha256")
    suffix = Path(spec.source_file).suffix.lower()
    if suffix == ".npz":
        try:
            with np.load(BytesIO(source_bytes), allow_pickle=False) as archive:
                if set(archive.files) != {"vertices", "faces"}:
                    raise ValueError("indexed triangle NPZ must contain only vertices and faces")
                vertices = np.asarray(archive["vertices"])
                faces = np.asarray(archive["faces"])
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not read indexed triangle NPZ: {exc}") from exc
    else:
        mesh = load_stl_bytes(source_bytes)
        vertices = mesh.triangles.reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    domain = prepare_surface_mesh_domain(vertices, faces, input_sha256=actual_hash)
    return _with_report(domain, {"provider": "triangle_mesh", "source_file": spec.source_file, "generated": False})


def indexed_mesh_sha256(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Hash an indexed mesh with a stable, shape-aware binary encoding."""

    digest = hashlib.sha256()
    for array in (np.asarray(vertices, dtype="<f8"), np.asarray(faces, dtype="<i8")):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _with_report(domain: SurfaceMeshDomain, additions: dict[str, object]) -> SurfaceMeshDomain:
    return SurfaceMeshDomain(
        vertices=domain.vertices,
        faces=domain.faces,
        boundary_loops=domain.boundary_loops,
        source_vertex_index=domain.source_vertex_index,
        source_face_index=domain.source_face_index,
        report={**domain.report, **additions},
    )


def _vertices(value: np.ndarray) -> np.ndarray:
    vertices = np.asarray(value, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("mesh vertices must have shape (n, 3) with n >= 3")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("mesh vertices must be finite")
    return vertices


def _faces(value: np.ndarray, vertex_count: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != 3 or not len(raw):
        raise ValueError("mesh faces must have shape (m, 3) with m >= 1")
    if raw.dtype.kind not in "iu" and not np.all(np.equal(raw, np.floor(raw))):
        raise ValueError("mesh faces must contain integer vertex indices")
    faces = np.asarray(raw, dtype=np.int64)
    if np.any(faces < 0) or np.any(faces >= vertex_count):
        raise ValueError("mesh faces contain an out-of-range vertex index")
    return faces


def _merge_vertices(vertices: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    lookup: dict[tuple[int, int, int], int] = {}
    merged: list[np.ndarray] = []
    mapping = np.empty(len(vertices), dtype=np.int64)
    for index, vertex in enumerate(vertices):
        key = tuple(np.rint(vertex / tolerance).astype(np.int64).tolist())
        target = lookup.get(key)
        if target is None:
            target = len(merged)
            lookup[key] = target
            merged.append(vertex)
        mapping[index] = target
    return np.asarray(merged, dtype=np.float64), mapping


def _nondegenerate_faces(vertices: np.ndarray, faces: np.ndarray, tolerance: float) -> np.ndarray:
    distinct = np.logical_and.reduce((faces[:, 0] != faces[:, 1], faces[:, 1] != faces[:, 2], faces[:, 2] != faces[:, 0]))
    triangles = vertices[faces]
    twice_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    return distinct & (0.5 * twice_area > tolerance)


def _reject_duplicate_faces(faces: np.ndarray) -> None:
    canonical = np.sort(faces, axis=1)
    _, counts = np.unique(canonical, axis=0, return_counts=True)
    if np.any(counts > 1):
        raise ValueError("mesh contains duplicate triangles")


def _edge_key(edge: Sequence[int]) -> tuple[int, int]:
    if len(edge) != 2:
        raise ValueError("an edge must contain exactly two vertex indices")
    left, right = (int(value) for value in edge)
    if left == right:
        raise ValueError("an edge must contain two distinct vertex indices")
    return (left, right) if left < right else (right, left)


def _edge_faces(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    result: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            result.setdefault(_edge_key(edge), []).append(face_index)
    return result


def _orient_faces(faces: np.ndarray) -> tuple[np.ndarray, int]:
    result = np.array(faces, copy=True)
    edge_entries: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for face_index, face in enumerate(result):
        for edge in ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0]))):
            edge_entries.setdefault(_edge_key(edge), []).append((face_index, edge))
    adjacency: list[list[tuple[int, bool]]] = [[] for _ in result]
    for entries in edge_entries.values():
        if len(entries) == 2:
            (first, first_direction), (second, second_direction) = entries
            adjacency[first].append((second, first_direction == second_direction))
            adjacency[second].append((first, first_direction == second_direction))
    flip = np.zeros(len(result), dtype=bool)
    visited = np.zeros(len(result), dtype=bool)
    for start in range(len(result)):
        if visited[start]:
            continue
        visited[start] = True
        queue = [start]
        while queue:
            current = queue.pop()
            for neighbor, same_direction in adjacency[current]:
                expected = bool(flip[current]) ^ same_direction
                if visited[neighbor]:
                    if bool(flip[neighbor]) != expected:
                        raise ValueError("mesh cannot be consistently oriented")
                    continue
                flip[neighbor] = expected
                visited[neighbor] = True
                queue.append(neighbor)
    result[flip] = result[flip][:, [0, 2, 1]]
    return result, int(np.count_nonzero(flip))


def _topology(faces: np.ndarray, vertex_count: int) -> dict[str, object]:
    edge_faces = _edge_faces(faces)
    boundary_directed: list[tuple[int, int]] = []
    adjacency: list[list[int]] = [[] for _ in faces]
    for key, entries in edge_faces.items():
        if len(entries) == 1:
            face = faces[entries[0]]
            for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                if _edge_key((left, right)) == key:
                    boundary_directed.append((int(left), int(right)))
                    break
        elif len(entries) == 2:
            left, right = entries
            adjacency[left].append(right)
            adjacency[right].append(left)
    nonmanifold_edges = [key for key, entries in edge_faces.items() if len(entries) > 2]
    nonmanifold_vertices = _nonmanifold_vertices(faces, edge_faces, vertex_count)
    component_count = _component_count(adjacency)
    # Do not let a malformed boundary mask the more specific topology failure.
    loops = [] if nonmanifold_edges or nonmanifold_vertices else _boundary_loops(boundary_directed)
    return {
        "edge_faces": edge_faces,
        "component_count": component_count,
        "boundary_loops": loops,
        "boundary_edge_count": len(boundary_directed),
        "nonmanifold_edges": nonmanifold_edges,
        "nonmanifold_vertices": nonmanifold_vertices,
    }


def _component_count(adjacency: list[list[int]]) -> int:
    visited = [False] * len(adjacency)
    count = 0
    for start in range(len(adjacency)):
        if visited[start]:
            continue
        count += 1
        visited[start] = True
        pending = [start]
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    pending.append(neighbor)
    return count


def _boundary_loops(edges: Iterable[tuple[int, int]]) -> list[np.ndarray]:
    outgoing: dict[int, int] = {}
    incoming: set[int] = set()
    for left, right in edges:
        if left in outgoing or right in incoming:
            raise ValueError("mesh boundary is branched and therefore non-manifold")
        outgoing[left] = right
        incoming.add(right)
    loops: list[np.ndarray] = []
    used: set[int] = set()
    for start in sorted(outgoing):
        if start in used:
            continue
        loop = [start]
        current = start
        while True:
            used.add(current)
            current = outgoing.get(current, -1)
            if current < 0:
                raise ValueError("mesh boundary does not form a closed loop")
            if current == start:
                break
            if current in used:
                raise ValueError("mesh boundary loop intersects itself")
            loop.append(current)
        minimum = min(range(len(loop)), key=loop.__getitem__)
        loops.append(np.asarray(loop[minimum:] + loop[:minimum], dtype=np.int64))
    return loops


def _nonmanifold_vertices(
    faces: np.ndarray, edge_faces: dict[tuple[int, int], list[int]], vertex_count: int
) -> list[int]:
    incident: list[list[int]] = [[] for _ in range(vertex_count)]
    for face_index, face in enumerate(faces):
        for vertex in face:
            incident[int(vertex)].append(face_index)
    bad: list[int] = []
    for vertex, face_indices in enumerate(incident):
        if len(face_indices) <= 1:
            continue
        neighboring: dict[int, list[int]] = {face: [] for face in face_indices}
        for key, shared_faces in edge_faces.items():
            if vertex in key and len(shared_faces) == 2:
                left, right = shared_faces
                neighboring[left].append(right)
                neighboring[right].append(left)
        seen = {face_indices[0]}
        pending = [face_indices[0]]
        while pending:
            current = pending.pop()
            for neighbor in neighboring[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        if len(seen) != len(face_indices):
            bad.append(vertex)
    return bad


def _reject_nonmanifold(topology: dict[str, object]) -> None:
    edges = topology["nonmanifold_edges"]
    vertices = topology["nonmanifold_vertices"]
    if edges:
        raise ValueError(f"mesh has non-manifold edge {edges[0]}")
    if vertices:
        raise ValueError(f"mesh has non-manifold vertex {vertices[0]}")


def _self_intersection_pairs(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[int, int]]:
    triangles = vertices[faces]
    pairs: list[tuple[int, int]] = []
    for left in range(len(triangles)):
        lower, upper = np.min(triangles[left], axis=0), np.max(triangles[left], axis=0)
        for right in range(left + 1, len(triangles)):
            if set(faces[left]).intersection(faces[right]):
                continue
            other_lower, other_upper = np.min(triangles[right], axis=0), np.max(triangles[right], axis=0)
            if np.any(upper < other_lower) or np.any(other_upper < lower):
                continue
            if _triangles_intersect(triangles[left], triangles[right]):
                pairs.append((left, right))
    return pairs


def _triangles_intersect(first: np.ndarray, second: np.ndarray, tolerance: float = 1e-10) -> bool:
    first_normal = np.cross(first[1] - first[0], first[2] - first[0])
    second_normal = np.cross(second[1] - second[0], second[2] - second[0])
    first_distance = (second - first[0]) @ first_normal
    second_distance = (first - second[0]) @ second_normal
    if np.all(first_distance > tolerance) or np.all(first_distance < -tolerance):
        return False
    if np.all(second_distance > tolerance) or np.all(second_distance < -tolerance):
        return False
    if np.linalg.norm(np.cross(first_normal, second_normal)) <= tolerance and np.max(np.abs(first_distance)) <= tolerance:
        return _coplanar_triangles_intersect(first, second, first_normal, tolerance)
    first_edges = ((first[0], first[1]), (first[1], first[2]), (first[2], first[0]))
    second_edges = ((second[0], second[1]), (second[1], second[2]), (second[2], second[0]))
    return any(_segment_intersects_triangle(start, end, second, tolerance) for start, end in first_edges) or any(
        _segment_intersects_triangle(start, end, first, tolerance) for start, end in second_edges
    )


def _segment_intersects_triangle(start: np.ndarray, end: np.ndarray, triangle: np.ndarray, tolerance: float) -> bool:
    direction = end - start
    first_edge, second_edge = triangle[1] - triangle[0], triangle[2] - triangle[0]
    cross = np.cross(direction, second_edge)
    determinant = float(np.dot(first_edge, cross))
    if abs(determinant) <= tolerance:
        return False
    inverse = 1.0 / determinant
    offset = start - triangle[0]
    u = inverse * float(np.dot(offset, cross))
    if u < -tolerance or u > 1.0 + tolerance:
        return False
    q = np.cross(offset, first_edge)
    v = inverse * float(np.dot(direction, q))
    if v < -tolerance or u + v > 1.0 + tolerance:
        return False
    t = inverse * float(np.dot(second_edge, q))
    return -tolerance <= t <= 1.0 + tolerance


def _coplanar_triangles_intersect(first: np.ndarray, second: np.ndarray, normal: np.ndarray, tolerance: float) -> bool:
    axis = int(np.argmax(np.abs(normal)))
    keep = [index for index in range(3) if index != axis]
    left, right = first[:, keep], second[:, keep]
    for polygon, other in ((left, right), (right, left)):
        for index in range(3):
            if _point_in_triangle_2d(polygon[index], other, tolerance):
                return True
    for polygon, other in ((left, right), (right, left)):
        for index in range(3):
            if _segments_intersect_2d(polygon[index], polygon[(index + 1) % 3], other[index], other[(index + 1) % 3], tolerance):
                return True
    return False


def _point_in_triangle_2d(point: np.ndarray, triangle: np.ndarray, tolerance: float) -> bool:
    signs = []
    for index in range(3):
        left, right = triangle[index], triangle[(index + 1) % 3]
        signs.append((right[0] - left[0]) * (point[1] - left[1]) - (right[1] - left[1]) * (point[0] - left[0]))
    return min(signs) >= -tolerance or max(signs) <= tolerance


def _segments_intersect_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, tolerance: float) -> bool:
    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        return float((first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0]))

    values = (cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b))
    return min(values[0], values[1]) <= tolerance <= max(values[0], values[1]) and min(values[2], values[3]) <= tolerance <= max(values[2], values[3])


def _first_source_index(mapping: np.ndarray, target_count: int) -> np.ndarray:
    result = np.full(target_count, -1, dtype=np.int64)
    for source, target in enumerate(mapping):
        if result[target] < 0:
            result[target] = source
    return result


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value)
    result.setflags(write=False)
    return result


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: list[int], left: int, right: int) -> None:
    left_root, right_root = _find(parent, left), _find(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root
