from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
import math
from typing import Callable, Literal

import numpy as np
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import linemerge, nearest_points, unary_union

from .external_npz import ExternalSourceJob, Material, MaterialPaths
from .stl_io import Mesh

CurveMode = Literal["flat", "sinusoidal"]
InfillPattern = Literal[
    "contour",
    "contour_offset",
    "lines_x",
    "lines_y",
    "grid",
    "triangles",
    "gyroid",
    "diagonal",
    "alternating_diagonal",
]
BuildAxis = Literal["x", "y", "z"]

DEFAULT_RESIN_LAYER_HEIGHT_MM = 0.5
DEFAULT_RESIN_LINE_WIDTH_MM = 2.0
DEFAULT_RESIN_INFILL_DENSITY_PERCENT = 100.0
DEFAULT_RESIN_INFILL_OVERLAP_PERCENT = 10.0
DEFAULT_FIBER_LAYER_HEIGHT_MM = 0.1
DEFAULT_FIBER_LINE_WIDTH_MM = 1.0
DEFAULT_RESIN_PERIMETER_COUNT = 2
DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES = 150.0
DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR = 0.35

DEFAULT_MATERIAL_PROCESS = {
    "R": {
        "layer_height_mm": DEFAULT_RESIN_LAYER_HEIGHT_MM,
        "line_width_mm": DEFAULT_RESIN_LINE_WIDTH_MM,
    },
    "F": {
        "layer_height_mm": DEFAULT_FIBER_LAYER_HEIGHT_MM,
        "line_width_mm": DEFAULT_FIBER_LINE_WIDTH_MM,
    },
}


@dataclass(frozen=True)
class SliceConfig:
    layer_height: float | None = None
    line_width: float | None = None
    material: Material = "R"
    z_min: float | None = None
    z_max: float | None = None
    tolerance: float = 1e-5
    curve_mode: CurveMode = "flat"
    curve_amplitude: float = 0.0
    curve_period: float = 50.0
    infill_pattern: InfillPattern = "lines_x"
    infill_density: float = DEFAULT_RESIN_INFILL_DENSITY_PERCENT
    infill_overlap: float = DEFAULT_RESIN_INFILL_OVERLAP_PERCENT
    build_axis: BuildAxis = "z"
    force_cap_angle: float | None = None

    def __post_init__(self) -> None:
        if self.material not in ("R", "F"):
            raise ValueError("material must be R or F")
        if self.layer_height is None:
            object.__setattr__(
                self,
                "layer_height",
                DEFAULT_MATERIAL_PROCESS[self.material]["layer_height_mm"],
            )
        if self.line_width is None:
            object.__setattr__(
                self,
                "line_width",
                DEFAULT_MATERIAL_PROCESS[self.material]["line_width_mm"],
            )
        if self.layer_height <= 0:
            raise ValueError("layer_height must be positive")
        if self.line_width <= 0:
            raise ValueError("line_width must be positive")
        if self.curve_period <= 0:
            raise ValueError("curve_period must be positive")
        if self.infill_pattern not in (
            "contour",
            "contour_offset",
            "lines_x",
            "lines_y",
            "grid",
            "triangles",
            "gyroid",
            "diagonal",
            "alternating_diagonal",
        ):
            raise ValueError("unsupported infill_pattern")
        if self.infill_density < 0 or self.infill_density > 100:
            raise ValueError("infill_density must be in the range [0, 100]")
        if self.infill_overlap < 0 or self.infill_overlap >= 100:
            raise ValueError("infill_overlap must be in the range [0, 100)")
        if self.build_axis not in ("x", "y", "z"):
            raise ValueError("build_axis must be x, y, or z")


@dataclass(frozen=True)
class RaftLayerConfig:
    outward_offset: float = 5.0
    layer_height: float = DEFAULT_RESIN_LAYER_HEIGHT_MM
    infill_density: float = DEFAULT_RESIN_INFILL_DENSITY_PERCENT

    def __post_init__(self) -> None:
        if self.outward_offset < 0:
            raise ValueError("raft outward offset must be non-negative")
        if self.layer_height <= 0:
            raise ValueError("raft layer height must be positive")
        if self.infill_density <= 0 or self.infill_density > 100:
            raise ValueError("raft infill density must be in the range (0, 100]")


def slice_mesh_to_job(mesh: Mesh, config: SliceConfig) -> ExternalSourceJob:
    mesh = orient_mesh_for_build_axis(mesh, config.build_axis)
    z_values = _layer_z_values(mesh, config)
    material_paths: list[MaterialPaths] = []
    path_roles: dict[str, dict[str, list[str]]] = {"R": {}}
    z_projector = _build_z_projector(config)
    constant_section_paths = _constant_section_paths_for_two_plane_extrusion(
        mesh,
        z_values,
        config.tolerance,
    )
    cached_constant_resin_paths_2d: list[np.ndarray] | None = None
    cached_constant_roles: list[str] | None = None

    for layer_index, base_z in enumerate(z_values):
        if constant_section_paths is None:
            segments = _intersect_mesh_at_z(mesh.triangles, float(base_z), config.tolerance)
            paths_2d = _stitch_segments(segments, config.tolerance)
        else:
            paths_2d = [path.copy() for path in constant_section_paths]
        if config.material == "R":
            layer_config = _part_layer_resin_config(config, layer_index, len(z_values))
            if (
                constant_section_paths is not None
                and layer_config.infill_pattern == "triangles"
                and cached_constant_resin_paths_2d is not None
                and cached_constant_roles is not None
            ):
                paths_2d = [path.copy() for path in cached_constant_resin_paths_2d]
                roles = list(cached_constant_roles)
            else:
                paths_2d, roles = _build_resin_paths(paths_2d, layer_config, layer_index)
                if constant_section_paths is not None and layer_config.infill_pattern == "triangles":
                    cached_constant_resin_paths_2d = [path.copy() for path in paths_2d]
                    cached_constant_roles = list(roles)
            path_roles["R"][str(layer_index)] = roles
        paths_3d = [_path_2d_to_3d(path, float(base_z), z_projector) for path in paths_2d]
        if paths_3d:
            material_paths.append(MaterialPaths(layer_index, config.material, paths_3d))

    meta = {
        "source": "kuka_slicer",
        "slicing": {
            "layer_height": config.layer_height,
            "line_width": config.line_width,
            "z_min": float(z_values[0]) if len(z_values) else None,
            "z_max": float(z_values[-1]) if len(z_values) else None,
            "curve_mode": config.curve_mode,
            "curve_amplitude": config.curve_amplitude,
            "curve_period": config.curve_period,
            "infill_pattern": config.infill_pattern,
            "infill_density": config.infill_density,
            "infill_overlap": config.infill_overlap,
            "build_axis": config.build_axis,
            "perimeter_count": DEFAULT_RESIN_PERIMETER_COUNT,
            "forced_part_cap_layers": {
                "infill_pattern": "zigzag_diagonal",
                "infill_density": 100.0,
                "bottom_angle_degrees": 45.0,
                "top_angle_degrees": -45.0,
            },
        },
        "path_roles": path_roles,
        "process_defaults": {
            "resin": {
                "layer_height_mm": DEFAULT_RESIN_LAYER_HEIGHT_MM,
                "line_width_mm": DEFAULT_RESIN_LINE_WIDTH_MM,
            },
            "fiber": {
                "layer_height_mm": DEFAULT_FIBER_LAYER_HEIGHT_MM,
                "line_width_mm": DEFAULT_FIBER_LINE_WIDTH_MM,
            },
        },
    }
    return ExternalSourceJob(material_paths=material_paths, meta=meta)


def _part_layer_resin_config(config: SliceConfig, layer_index: int, layer_count: int) -> SliceConfig:
    if layer_count <= 0:
        return config
    if layer_index == 0:
        return replace(config, infill_pattern="diagonal", infill_density=100.0, force_cap_angle=45.0)
    if layer_index == layer_count - 1:
        return replace(config, infill_pattern="diagonal", infill_density=100.0, force_cap_angle=-45.0)
    return config


def _constant_section_paths_for_two_plane_extrusion(
    mesh: Mesh,
    z_values: np.ndarray,
    tolerance: float,
) -> list[np.ndarray] | None:
    if len(z_values) == 0:
        return None
    z_levels = np.unique(np.round(mesh.triangles[:, :, 2].reshape(-1), 6))
    if len(z_levels) != 2:
        return None
    thickness = float(z_levels[-1] - z_levels[0])
    if thickness <= tolerance:
        return None
    segments = _intersect_mesh_at_z(mesh.triangles, float(z_values[0]), tolerance)
    paths = _stitch_segments(segments, tolerance)
    return paths or None


def add_raft_to_job(
    job: ExternalSourceJob,
    mesh: Mesh,
    config: SliceConfig,
    raft_layers: list[RaftLayerConfig],
    top_gap: float,
) -> float:
    """Insert resin raft layers before the part and shift existing paths upward."""

    if not raft_layers:
        return 0.0
    if top_gap < 0:
        raise ValueError("raft top gap must be non-negative")

    oriented_mesh = orient_mesh_for_build_axis(mesh, config.build_axis)
    footprint = _raft_footprint_geometry(oriented_mesh, config)
    if footprint.is_empty:
        return 0.0

    raft_height = sum(layer.layer_height for layer in raft_layers)
    z_shift = raft_height + top_gap
    raft_count = len(raft_layers)

    shifted_groups = [
        MaterialPaths(
            group.layer_index + raft_count,
            group.material,
            [_shift_path_z(path, z_shift) for path in group.paths],
        )
        for group in job.material_paths
    ]

    path_roles = job.meta.setdefault("path_roles", {})
    if not isinstance(path_roles, dict):
        path_roles = {}
        job.meta["path_roles"] = path_roles
    resin_roles = path_roles.setdefault("R", {})
    if not isinstance(resin_roles, dict):
        resin_roles = {}
        path_roles["R"] = resin_roles
    shifted_roles = {
        str(int(layer_index) + raft_count): roles
        for layer_index, roles in resin_roles.items()
    }
    resin_roles.clear()
    resin_roles.update(shifted_roles)

    raft_groups: list[MaterialPaths] = []
    current_z = 0.0
    for layer_index, raft_layer in enumerate(raft_layers):
        current_z += raft_layer.layer_height
        paths_2d, roles = _raft_paths_for_layer(
            footprint,
            config,
            raft_layer,
            layer_index,
            contact_layer=layer_index == raft_count - 1,
        )
        paths_3d = [_path_2d_to_constant_z(path, current_z) for path in paths_2d]
        if paths_3d:
            raft_groups.append(MaterialPaths(layer_index, "R", paths_3d))
            resin_roles[str(layer_index)] = roles

    job.material_paths = raft_groups + shifted_groups
    job.material_paths.sort(key=lambda group: (group.layer_index, 0 if group.material == "R" else 1))
    job.meta["raft"] = {
        "layer_count": raft_count,
        "top_gap": top_gap,
        "layers": [
            {
                "outward_offset": layer.outward_offset,
                "layer_height": layer.layer_height,
                "infill_density": layer.infill_density,
            }
            for layer in raft_layers
        ],
    }
    return z_shift


def normalize_job_xy_origin(job: ExternalSourceJob) -> tuple[float, float]:
    """Translate all exported paths so the lower-left XY bound is at (0, 0)."""

    bounds = _job_xy_bounds(job)
    if bounds is None:
        return (0.0, 0.0)
    min_x, min_y, _, _ = bounds
    if abs(min_x) <= 1e-7 and abs(min_y) <= 1e-7:
        return (0.0, 0.0)

    for group in job.material_paths:
        for path in group.paths:
            path[:, 0] -= np.float32(min_x)
            path[:, 1] -= np.float32(min_y)

    job.meta["xy_origin_normalization"] = {
        "applied": True,
        "source_min_x": float(min_x),
        "source_min_y": float(min_y),
        "translation_x": float(-min_x),
        "translation_y": float(-min_y),
    }
    return (float(-min_x), float(-min_y))


def _job_xy_bounds(job: ExternalSourceJob) -> tuple[float, float, float, float] | None:
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf
    found = False
    for group in job.material_paths:
        for path in group.paths:
            if path.size == 0:
                continue
            found = True
            min_x = min(min_x, float(np.min(path[:, 0])))
            min_y = min(min_y, float(np.min(path[:, 1])))
            max_x = max(max_x, float(np.max(path[:, 0])))
            max_y = max(max_y, float(np.max(path[:, 1])))
    if not found:
        return None
    return (min_x, min_y, max_x, max_y)


def orient_mesh_for_build_axis(mesh: Mesh, build_axis: BuildAxis) -> Mesh:
    """Map the selected source build axis onto output Z before slicing/export."""

    if build_axis == "z":
        return mesh

    triangles = mesh.triangles.copy()
    if build_axis == "y":
        triangles = triangles[:, :, [0, 2, 1]]
    elif build_axis == "x":
        triangles = triangles[:, :, [1, 2, 0]]
    else:
        raise ValueError("build_axis must be x, y, or z")
    return Mesh(triangles)


def _raft_footprint_geometry(mesh: Mesh, config: SliceConfig):
    z_values = _layer_z_values(mesh, config)
    if len(z_values) == 0:
        return Polygon()
    segments = _intersect_mesh_at_z(mesh.triangles, float(z_values[0]), config.tolerance)
    contours = [path for path in _stitch_segments(segments, config.tolerance) if path.shape[0] >= 3]
    return _solid_geometry_from_contours(contours)


def _raft_paths_for_layer(
    footprint,
    config: SliceConfig,
    raft_layer: RaftLayerConfig,
    layer_index: int,
    contact_layer: bool = False,
) -> tuple[list[np.ndarray], list[str]]:
    geometry = footprint.buffer(raft_layer.outward_offset, join_style="round")
    if geometry.is_empty:
        return [], []

    perimeters, roles = _perimeter_paths_from_geometry(
        geometry,
        config.line_width,
        _resin_path_spacing(config.line_width, config.infill_overlap),
        DEFAULT_RESIN_PERIMETER_COUNT,
        config.tolerance,
    )
    infill_geometry = geometry.buffer(
        -_infill_geometry_inset(config),
        join_style="round",
    )
    if contact_layer:
        filled = _raft_lattice_infill_paths(infill_geometry, config, raft_layer.infill_density)
    else:
        filled = _raft_zigzag_infill_paths(infill_geometry, config, raft_layer.infill_density)
    return perimeters + filled, roles + ["infill"] * len(filled)


def _raft_lattice_infill_paths(
    geometry,
    config: SliceConfig,
    infill_density: float,
) -> list[np.ndarray]:
    if geometry.is_empty:
        return []

    spacing = _line_infill_spacing(
        config.line_width,
        infill_density,
        config.infill_overlap,
    )
    filled = _gyroid_infill_geometry(geometry, spacing, config.tolerance)
    return _connect_infill_paths_within_geometry(filled, geometry, config.tolerance)


def _connect_infill_paths_within_geometry(
    paths: list[np.ndarray],
    geometry,
    tolerance: float,
) -> list[np.ndarray]:
    if len(paths) <= 1:
        return paths

    remaining = [np.asarray(path, dtype=np.float32) for path in paths if path.shape[0] >= 2]
    if len(remaining) <= 1:
        return remaining

    connected: list[np.ndarray] = []
    current = remaining.pop(0)
    while remaining:
        best_index = -1
        best_path: np.ndarray | None = None
        best_distance = math.inf
        for index, candidate in enumerate(remaining):
            for oriented in _path_connection_orientations(candidate, current[-1, :2], tolerance):
                start = oriented[0, :2]
                connector = LineString(
                    [
                        tuple(float(value) for value in current[-1, :2]),
                        tuple(float(value) for value in start),
                    ]
                )
                if not geometry.buffer(max(tolerance * 10.0, 1e-6)).covers(connector):
                    continue
                if not _connector_clear_of_paths(
                    connector,
                    [current, *remaining],
                    tolerance,
                ):
                    continue
                distance = float(connector.length)
                if distance < best_distance:
                    best_index = index
                    best_path = oriented
                    best_distance = distance

        if best_index < 0:
            connected.append(current)
            current = remaining.pop(0)
            continue

        remaining.pop(best_index)
        if best_path is None:
            continue
        next_path = best_path
        if np.linalg.norm(current[-1, :2] - next_path[0, :2]) <= tolerance:
            current = np.vstack([current, next_path[1:]])
        else:
            current = np.vstack([current, next_path])

    connected.append(current)
    return connected


def _connector_clear_of_paths(
    connector: LineString,
    paths: list[np.ndarray],
    tolerance: float,
) -> bool:
    path_lines = [
        LineString([(float(point[0]), float(point[1])) for point in path])
        for path in paths
        if path.shape[0] >= 2
    ]
    path_lines = [line for line in path_lines if not line.is_empty]
    if not path_lines:
        return True

    intersection = connector.intersection(unary_union(path_lines))
    if intersection.is_empty:
        return True
    if float(getattr(intersection, "length", 0.0)) > tolerance:
        return False

    endpoint_tolerance = max(tolerance * 20.0, 1e-5)
    allowed = Point(connector.coords[0]).buffer(endpoint_tolerance).union(
        Point(connector.coords[-1]).buffer(endpoint_tolerance)
    )
    return allowed.covers(intersection)


def _path_connection_orientations(
    path: np.ndarray,
    target_start: np.ndarray,
    tolerance: float,
) -> list[np.ndarray]:
    path = np.asarray(path, dtype=np.float32)
    if not _is_closed_path(path, tolerance):
        return [path, path[::-1]]

    ring = _dedupe_consecutive(path[:-1], tolerance)
    if ring.shape[0] < 2:
        return [path]
    distances = np.linalg.norm(ring[:, :2] - target_start[:2], axis=1)
    start_index = int(np.argmin(distances))
    rotated = np.vstack([ring[start_index:], ring[:start_index], ring[start_index]])
    reversed_ring = ring[::-1]
    reverse_distances = np.linalg.norm(reversed_ring[:, :2] - target_start[:2], axis=1)
    reverse_start_index = int(np.argmin(reverse_distances))
    reversed_rotated = np.vstack([
        reversed_ring[reverse_start_index:],
        reversed_ring[:reverse_start_index],
        reversed_ring[reverse_start_index],
    ])
    return [
        np.asarray(rotated, dtype=np.float32),
        np.asarray(reversed_rotated, dtype=np.float32),
    ]


def _raft_zigzag_infill_paths(
    geometry,
    config: SliceConfig,
    infill_density: float,
) -> list[np.ndarray]:
    if geometry.is_empty:
        return []

    spacing = _line_infill_spacing(
        config.line_width,
        infill_density,
        config.infill_overlap,
    )
    filled = _zigzag_infill_geometry(
        geometry,
        spacing,
        0.0,
        config.tolerance,
        max_connector_distance=spacing * 1.6,
    )
    filled = _smooth_resin_infill_paths(
        filled,
        geometry,
        config.line_width * DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR,
        DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES,
        config.tolerance,
    )
    return optimize_open_path_travel(filled, config.tolerance)


def _shift_path_z(path: np.ndarray, z_shift: float) -> np.ndarray:
    shifted = np.asarray(path, dtype=np.float32).copy()
    shifted[:, 2] += np.float32(z_shift)
    return shifted


def _path_2d_to_constant_z(path: np.ndarray, z: float) -> np.ndarray:
    return np.asarray(
        [[float(point[0]), float(point[1]), float(z)] for point in path],
        dtype=np.float32,
    )


def _layer_z_values(mesh: Mesh, config: SliceConfig) -> np.ndarray:
    z_min = mesh.z_min if config.z_min is None else config.z_min
    z_max = mesh.z_max if config.z_max is None else config.z_max
    if z_min > z_max:
        raise ValueError("z_min must be <= z_max")

    # Start one layer above the bottom and include the top cap when it falls on
    # the layer grid. Horizontal faces are ignored by the intersection code, but
    # vertical side faces still provide the top boundary.
    start = z_min if config.z_min is not None else z_min + config.layer_height
    end = z_max
    if start > end:
        start = (z_min + z_max) / 2.0
        end = start
    count = int(math.floor((end - start) / config.layer_height)) + 1
    z_values = start + np.arange(max(count, 0), dtype=np.float32) * config.layer_height
    return z_values[z_values <= z_max + config.tolerance]


def _intersect_mesh_at_z(triangles: np.ndarray, z: float, tolerance: float) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for triangle in triangles:
        points = _intersect_triangle_at_z(triangle, z, tolerance)
        if len(points) == 2 and np.linalg.norm(points[0] - points[1]) > tolerance:
            segments.append(np.asarray(points, dtype=np.float32))
    return segments


def _intersect_triangle_at_z(
    triangle: np.ndarray, z: float, tolerance: float
) -> list[np.ndarray]:
    intersections: list[np.ndarray] = []
    for edge_start, edge_end in ((0, 1), (1, 2), (2, 0)):
        p0 = triangle[edge_start]
        p1 = triangle[edge_end]
        z0 = float(p0[2])
        z1 = float(p1[2])

        if abs(z0 - z1) <= tolerance:
            continue
        if (z < min(z0, z1) - tolerance) or (z > max(z0, z1) + tolerance):
            continue

        t = (z - z0) / (z1 - z0)
        if t < -tolerance or t > 1.0 + tolerance:
            continue
        point = p0 + np.float32(t) * (p1 - p0)
        _append_unique_point(intersections, point[:2], tolerance)
    return intersections


def _append_unique_point(points: list[np.ndarray], point: np.ndarray, tolerance: float) -> None:
    for existing in points:
        if np.linalg.norm(existing - point) <= tolerance:
            return
    points.append(np.asarray(point, dtype=np.float32))


def _stitch_segments(segments: list[np.ndarray], tolerance: float) -> list[np.ndarray]:
    if not segments:
        return []

    indexed_segments = [
        (
            np.asarray(segment[0], dtype=np.float32),
            np.asarray(segment[1], dtype=np.float32),
        )
        for segment in segments
    ]
    endpoint_map: dict[tuple[int, int], set[int]] = defaultdict(set)
    for index, (a, b) in enumerate(indexed_segments):
        endpoint_map[_point_key(a, tolerance)].add(index)
        endpoint_map[_point_key(b, tolerance)].add(index)

    unused = set(range(len(indexed_segments)))
    paths: list[np.ndarray] = []

    while unused:
        current_index = unused.pop()
        a, b = indexed_segments[current_index]
        path = deque([a, b])

        _extend_stitched_path(path, unused, indexed_segments, endpoint_map, tolerance, append_right=True)
        _extend_stitched_path(path, unused, indexed_segments, endpoint_map, tolerance, append_right=False)

        cleaned = _dedupe_consecutive(np.asarray(list(path), dtype=np.float32), tolerance)
        if cleaned.shape[0] >= 2:
            paths.append(cleaned)
    return paths


def _extend_stitched_path(
    path: deque[np.ndarray],
    unused: set[int],
    segments: list[tuple[np.ndarray, np.ndarray]],
    endpoint_map: dict[tuple[int, int], set[int]],
    tolerance: float,
    append_right: bool,
) -> None:
    while True:
        endpoint = path[-1] if append_right else path[0]
        connected_indices = endpoint_map.get(_point_key(endpoint, tolerance), set())
        next_index = next((index for index in connected_indices if index in unused), None)
        if next_index is None:
            return

        unused.remove(next_index)
        a, b = segments[next_index]
        if _close(endpoint, a, tolerance):
            next_point = b
        elif _close(endpoint, b, tolerance):
            next_point = a
        else:
            continue

        if append_right:
            path.append(next_point)
        else:
            path.appendleft(next_point)


def _point_key(point: np.ndarray, tolerance: float) -> tuple[int, int]:
    scale = 1.0 / max(tolerance, 1e-9)
    return (int(round(float(point[0]) * scale)), int(round(float(point[1]) * scale)))


def _close(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    delta = a - b
    return bool(float(delta[0] * delta[0] + delta[1] * delta[1]) <= tolerance * tolerance)


def _dedupe_consecutive(path: np.ndarray, tolerance: float) -> np.ndarray:
    points = [path[0]]
    for point in path[1:]:
        if not _close(points[-1], point, tolerance):
            points.append(point)
    return np.asarray(points, dtype=np.float32)


def _apply_resin_infill(
    paths: list[np.ndarray], config: SliceConfig, layer_index: int = 0
) -> list[np.ndarray]:
    return _build_resin_paths(paths, config, layer_index)[0]


def _build_resin_paths(
    paths: list[np.ndarray], config: SliceConfig, layer_index: int = 0
) -> tuple[list[np.ndarray], list[str]]:
    contours = [path for path in paths if path.shape[0] >= 3]
    solid_geometry = _solid_geometry_from_contours(contours)
    if solid_geometry.is_empty:
        return paths, ["outer_contour" if path.shape[0] > 2 else "infill" for path in paths]

    perimeters, roles = _perimeter_paths_from_geometry(
        solid_geometry,
        config.line_width,
        _resin_path_spacing(config.line_width, config.infill_overlap),
        DEFAULT_RESIN_PERIMETER_COUNT,
        config.tolerance,
    )
    if config.infill_pattern == "contour":
        return perimeters, roles

    infill_geometry = solid_geometry.buffer(
        -_infill_geometry_inset(config),
        join_style="round",
    )
    filled = _infill_paths_for_geometry(
        infill_geometry,
        config,
        layer_index,
        config.infill_density,
    )
    return perimeters + filled, roles + ["infill"] * len(filled)


def _infill_paths_for_geometry(
    geometry,
    config: SliceConfig,
    layer_index: int,
    infill_density: float,
) -> list[np.ndarray]:
    if geometry.is_empty:
        return []
    if infill_density <= 0:
        return []

    filled: list[np.ndarray] = []
    line_spacing = _line_infill_spacing(
        config.line_width,
        infill_density,
        config.infill_overlap,
    )
    grid_spacing = _grid_infill_spacing(
        config.line_width,
        infill_density,
        config.infill_overlap,
    )
    triangle_spacing = _multi_axis_infill_spacing(
        config.line_width,
        infill_density,
        3,
        config.infill_overlap,
    )

    if config.infill_pattern == "lines_x":
        filled.extend(_zigzag_infill_geometry(geometry, line_spacing, 0.0, config.tolerance))
    elif config.infill_pattern == "contour_offset":
        filled.extend(
            _concentric_infill_geometry(
                geometry,
                config.line_width,
                _resin_path_spacing(config.line_width, config.infill_overlap),
                config.tolerance,
            )
        )
    elif config.infill_pattern == "lines_y":
        filled.extend(_zigzag_infill_geometry(geometry, line_spacing, 90.0, config.tolerance))
    elif config.infill_pattern == "diagonal":
        angle = 45.0 if config.force_cap_angle is None else config.force_cap_angle
        filled.extend(
            _zigzag_infill_geometry(
                geometry,
                line_spacing,
                angle,
                config.tolerance,
                allow_boundary_route=True,
            )
        )
    elif config.infill_pattern == "alternating_diagonal":
        angle = 45.0 if layer_index % 2 == 0 else -45.0
        filled.extend(_zigzag_infill_geometry(geometry, line_spacing, angle, config.tolerance))
    elif config.infill_pattern == "grid":
        filled.extend(_zigzag_infill_geometry(geometry, grid_spacing, 0.0, config.tolerance))
        filled.extend(_zigzag_infill_geometry(geometry, grid_spacing, 90.0, config.tolerance))
    elif config.infill_pattern == "triangles":
        filled.extend(
            _triangular_lattice_infill_geometry(
                geometry,
                triangle_spacing,
                max(
                    config.line_width,
                    min(config.line_width * 1.6, triangle_spacing * 0.38),
                ),
                _triangular_connector_length_factor(infill_density),
                _triangular_connector_requires_lattice_direction(infill_density),
                config.tolerance,
            )
        )
    elif config.infill_pattern == "gyroid":
        filled.extend(_gyroid_infill_geometry(geometry, line_spacing, config.tolerance))

    if config.infill_pattern not in ("triangles", "gyroid"):
        smoothing_radius = config.line_width * DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR
        smoothing_cut_fraction = 0.35
        if config.infill_pattern in ("diagonal", "alternating_diagonal"):
            # Keep the bend as a small line-width-sized fillet. Larger radii
            # turn boundary turns into semicircles and force excessive splits.
            smoothing_radius = config.line_width * 0.2
            smoothing_cut_fraction = 0.3
        filled = _smooth_resin_infill_paths(
            filled,
            geometry,
            smoothing_radius,
            DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES,
            config.tolerance,
            cut_fraction=smoothing_cut_fraction,
        )
    return optimize_open_path_travel(filled, config.tolerance)


def _gyroid_infill_geometry(geometry, spacing: float, tolerance: float) -> list[np.ndarray]:
    if spacing <= 0:
        raise ValueError("infill spacing must be positive")
    if geometry.is_empty:
        return []

    min_x, min_y, max_x, max_y = geometry.bounds
    wavelength = max(spacing * 4.0, spacing + tolerance)
    wave_number = 2.0 * math.pi / wavelength
    sample_step = max(min(spacing * 0.25, wavelength / 18.0), tolerance * 20.0)
    padding = wavelength
    z_phase = math.pi * 0.37

    xs = np.arange(min_x - padding, max_x + padding + sample_step, sample_step)
    ys = np.arange(min_y - padding, max_y + padding + sample_step, sample_step)
    if xs.size < 2 or ys.size < 2:
        return []

    x_grid, y_grid = np.meshgrid(xs, ys, indexing="xy")
    values = (
        np.sin(wave_number * x_grid) * np.cos(wave_number * y_grid)
        + np.sin(wave_number * y_grid) * math.cos(z_phase)
        + math.sin(z_phase) * np.cos(wave_number * x_grid)
    )

    segments: list[LineString] = []
    for row in range(ys.size - 1):
        y0 = float(ys[row])
        y1 = float(ys[row + 1])
        for col in range(xs.size - 1):
            x0 = float(xs[col])
            x1 = float(xs[col + 1])
            corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            corner_values = (
                float(values[row, col]),
                float(values[row, col + 1]),
                float(values[row + 1, col + 1]),
                float(values[row + 1, col]),
            )
            intersections = _marching_square_zero_crossings(corners, corner_values, tolerance)
            if len(intersections) == 2:
                segments.append(LineString(intersections))
            elif len(intersections) == 4:
                center_value = float(
                    math.sin(wave_number * ((x0 + x1) * 0.5))
                    * math.cos(wave_number * ((y0 + y1) * 0.5))
                    + math.sin(wave_number * ((y0 + y1) * 0.5)) * math.cos(z_phase)
                    + math.sin(z_phase) * math.cos(wave_number * ((x0 + x1) * 0.5))
                )
                for start, end in _split_saddle_zero_crossings(
                    intersections,
                    corner_values,
                    center_value,
                ):
                    segments.append(LineString((start, end)))

    if not segments:
        return []

    merged = linemerge(unary_union(segments))
    clipped = geometry.intersection(merged)
    min_length = max(spacing * 0.45, tolerance * 10.0)
    paths: list[np.ndarray] = []
    for segment in _extract_line_segments(clipped, tolerance):
        if segment.length < min_length:
            continue
        coords = list(segment.coords)
        if len(coords) < 2:
            continue
        path = np.asarray([[float(x), float(y)] for x, y in coords], dtype=np.float32)
        path = _dedupe_consecutive(path, tolerance)
        if path.shape[0] >= 2:
            paths.append(path)
    return optimize_open_path_travel(paths, tolerance)


def _marching_square_zero_crossings(
    corners: tuple[tuple[float, float], ...],
    values: tuple[float, float, float, float],
    tolerance: float,
) -> list[tuple[float, float]]:
    edge_indices = ((0, 1), (1, 2), (2, 3), (3, 0))
    points: list[tuple[float, float]] = []
    for start_index, end_index in edge_indices:
        start_value = values[start_index]
        end_value = values[end_index]
        if start_value > 0 and end_value > 0:
            continue
        if start_value < 0 and end_value < 0:
            continue

        start = corners[start_index]
        end = corners[end_index]
        denominator = start_value - end_value
        if abs(denominator) <= tolerance:
            ratio = 0.5
        else:
            ratio = start_value / denominator
        ratio = min(1.0, max(0.0, ratio))
        point = (
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        )
        if not any(math.hypot(point[0] - prior[0], point[1] - prior[1]) <= tolerance for prior in points):
            points.append(point)
    return points


def _split_saddle_zero_crossings(
    intersections: list[tuple[float, float]],
    corner_values: tuple[float, float, float, float],
    center_value: float,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    positive_diagonal_02 = corner_values[0] >= 0 and corner_values[2] >= 0
    if positive_diagonal_02 == (center_value >= 0):
        return [(intersections[3], intersections[0]), (intersections[1], intersections[2])]
    return [(intersections[0], intersections[1]), (intersections[2], intersections[3])]


def optimize_open_path_travel(paths: list[np.ndarray], tolerance: float = 1e-5) -> list[np.ndarray]:
    """Order and reverse open paths to reduce travel between consecutive paths."""

    remaining = [np.asarray(path, dtype=np.float32) for path in paths]
    if len(remaining) <= 1:
        return remaining

    ordered: list[np.ndarray] = [remaining.pop(0)]
    while remaining:
        current_end = ordered[-1][-1, :2]
        best_index = 0
        best_reverse = False
        best_distance = math.inf

        for index, path in enumerate(remaining):
            if _is_closed_path(path, tolerance):
                distance = float(np.linalg.norm(path[0, :2] - current_end))
                reverse = False
            else:
                start_distance = float(np.linalg.norm(path[0, :2] - current_end))
                end_distance = float(np.linalg.norm(path[-1, :2] - current_end))
                if end_distance < start_distance:
                    distance = end_distance
                    reverse = True
                else:
                    distance = start_distance
                    reverse = False

            if distance < best_distance:
                best_index = index
                best_reverse = reverse
                best_distance = distance

        selected = remaining.pop(best_index)
        if best_reverse:
            selected = selected[::-1].copy()
        ordered.append(selected)

    return ordered


def _smooth_resin_infill_paths(
    paths: list[np.ndarray],
    geometry,
    max_radius: float,
    angle_threshold_degrees: float,
    tolerance: float,
    cut_fraction: float = 0.35,
) -> list[np.ndarray]:
    if max_radius <= tolerance or not paths:
        return paths

    # Allow the tool centerline to use the material carried by the line width
    # near a boundary, while still keeping the transition out of holes.
    safe_geometry = geometry.buffer(max(tolerance * 10.0, min(max_radius * 0.25, max_radius)), join_style="round")
    smoothed: list[np.ndarray] = []
    for path in paths:
        smoothed.extend(
            _smooth_path_corners_into_paths(
                path,
                max_radius,
                angle_threshold_degrees,
                tolerance,
                safe_geometry,
                cut_fraction,
            )
        )
    return smoothed


def _smooth_path_corners_into_paths(
    path: np.ndarray,
    max_radius: float,
    angle_threshold_degrees: float,
    tolerance: float,
    safe_geometry,
    cut_fraction: float,
) -> list[np.ndarray]:
    points = _dedupe_consecutive(np.asarray(path[:, :2], dtype=np.float32), tolerance)
    if points.shape[0] < 3 or _is_closed_path(points, tolerance):
        return [_smooth_path_corners(
            points,
            max_radius,
            angle_threshold_degrees,
            tolerance,
            safe_geometry=safe_geometry,
            cut_fraction=cut_fraction,
        )]

    result: list[np.ndarray] = [points[0]]
    split_paths: list[np.ndarray] = []
    for index in range(1, points.shape[0] - 1):
        previous_point = points[index - 1]
        current_point = points[index]
        next_point = points[index + 1]
        rounded = _rounded_corner_points(
            previous_point,
            current_point,
            next_point,
            max_radius,
            angle_threshold_degrees,
            tolerance,
            cut_fraction,
        )
        rounded_is_safe = rounded is not None and safe_geometry.covers(
            LineString([(float(point[0]), float(point[1])) for point in rounded])
        )
        if rounded_is_safe:
            result.extend(rounded)
            continue

        if rounded is not None:
            result.append(current_point)
            if len(result) >= 2:
                split_paths.append(_dedupe_consecutive(np.asarray(result, dtype=np.float32), tolerance))
            result = [current_point, next_point]
        else:
            result.append(current_point)

    result.append(points[-1])
    if len(result) >= 2:
        split_paths.append(_dedupe_consecutive(np.asarray(result, dtype=np.float32), tolerance))
    return [item for item in split_paths if item.shape[0] >= 2]


def _smooth_path_corners(
    path: np.ndarray,
    max_radius: float,
    angle_threshold_degrees: float,
    tolerance: float,
    safe_geometry=None,
    cut_fraction: float = 0.35,
) -> np.ndarray:
    points = _dedupe_consecutive(np.asarray(path[:, :2], dtype=np.float32), tolerance)
    if points.shape[0] < 3:
        return path

    closed = _is_closed_path(points, tolerance)
    if closed:
        points = points[:-1]
    if points.shape[0] < 3:
        return path

    result: list[np.ndarray] = []
    count = points.shape[0]
    start_index = 0 if closed else 1
    end_index = count if closed else count - 1

    if not closed:
        result.append(points[0])

    for index in range(start_index, end_index):
        previous_point = points[(index - 1) % count]
        current_point = points[index]
        next_point = points[(index + 1) % count]
        rounded = _rounded_corner_points(
            previous_point,
            current_point,
            next_point,
            max_radius,
            angle_threshold_degrees,
            tolerance,
            cut_fraction,
        )
        if rounded is None or (
            safe_geometry is not None
            and not safe_geometry.covers(
                LineString([(float(point[0]), float(point[1])) for point in rounded])
            )
        ):
            result.append(current_point)
        else:
            result.extend(rounded)

    if not closed:
        result.append(points[-1])
    elif result:
        result.append(result[0])

    return _dedupe_consecutive(np.asarray(result, dtype=np.float32), tolerance)


def _rounded_corner_points(
    previous_point: np.ndarray,
    current_point: np.ndarray,
    next_point: np.ndarray,
    max_radius: float,
    angle_threshold_degrees: float,
    tolerance: float,
    cut_fraction: float = 0.35,
) -> list[np.ndarray] | None:
    incoming = previous_point - current_point
    outgoing = next_point - current_point
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    if incoming_length <= tolerance or outgoing_length <= tolerance:
        return None

    incoming_unit = incoming / incoming_length
    outgoing_unit = outgoing / outgoing_length
    cosine = float(np.clip(np.dot(incoming_unit, outgoing_unit), -1.0, 1.0))
    angle_degrees = math.degrees(math.acos(cosine))
    if angle_degrees >= angle_threshold_degrees:
        return None

    cut_fraction = min(max(float(cut_fraction), 0.05), 0.8)
    cut_distance = min(max_radius, incoming_length * cut_fraction, outgoing_length * cut_fraction)
    if cut_distance <= tolerance:
        return None

    start = current_point + incoming_unit * cut_distance
    end = current_point + outgoing_unit * cut_distance
    bisector = incoming_unit + outgoing_unit
    bisector_length = float(np.linalg.norm(bisector))
    if bisector_length <= tolerance:
        return None

    half_angle = math.radians(angle_degrees) * 0.5
    cosine_half = math.cos(half_angle)
    if abs(cosine_half) <= tolerance:
        return None

    center = current_point + (bisector / bisector_length) * (cut_distance / cosine_half)
    start_angle = math.atan2(float(start[1] - center[1]), float(start[0] - center[0]))
    end_angle = math.atan2(float(end[1] - center[1]), float(end[0] - center[0]))
    delta = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    steps = max(5, int(abs(delta) / (math.pi / 12.0)) + 1)

    rounded: list[np.ndarray] = []
    radius = float(np.linalg.norm(start - center))
    for step in range(steps):
        t = step / (steps - 1)
        angle = start_angle + delta * t
        point = np.asarray(
            [center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius],
            dtype=np.float32,
        )
        rounded.append(point)
    return rounded


def _is_closed_path(path: np.ndarray, tolerance: float) -> bool:
    return path.shape[0] > 2 and _close(path[0, :2], path[-1, :2], tolerance)


def _solid_geometry_from_contours(contours: list[np.ndarray]):
    contour_roles = _contours_with_boundary_roles(contours)
    shells = [(contour, Polygon(_open_ring(contour))) for contour, role in contour_roles if role == "outer_boundary"]
    holes = [(contour, Polygon(_open_ring(contour))) for contour, role in contour_roles if role == "inner_boundary"]
    polygons: list[Polygon] = []

    for shell_contour, shell_polygon in shells:
        if not shell_polygon.is_valid:
            shell_polygon = shell_polygon.buffer(0)
        shell_holes: list[list[tuple[float, float]]] = []
        for hole_contour, hole_polygon in holes:
            point = hole_polygon.representative_point()
            if shell_polygon.contains(point):
                shell_holes.append(_open_ring(hole_contour))
        polygon = Polygon(_open_ring(shell_contour), holes=shell_holes)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            polygons.append(polygon)

    if not polygons:
        return Polygon()
    return unary_union(polygons)


def _open_ring(contour: np.ndarray) -> list[tuple[float, float]]:
    points = np.asarray(contour[:, :2], dtype=np.float32)
    if points.shape[0] > 1 and _close(points[0], points[-1], 1e-6):
        points = points[:-1]
    return [(float(point[0]), float(point[1])) for point in points]


def _perimeter_paths_from_geometry(
    geometry,
    line_width: float,
    path_spacing: float,
    perimeter_count: int,
    tolerance: float,
) -> tuple[list[np.ndarray], list[str]]:
    paths: list[np.ndarray] = []
    roles: list[str] = []
    for perimeter_index in range(perimeter_count):
        offset_distance = line_width * 0.5 + perimeter_index * path_spacing
        offset_geometry = geometry.buffer(-offset_distance, join_style="round")
        if offset_geometry.is_empty:
            continue
        for polygon in _iter_polygons(offset_geometry):
            exterior = _coords_to_path(polygon.exterior.coords, tolerance)
            if exterior.shape[0] >= 3:
                paths.append(exterior)
                roles.append("outer_contour" if perimeter_index == 0 else "inner_contour")
            for interior in polygon.interiors:
                inner = _coords_to_path(interior.coords, tolerance)
                if inner.shape[0] >= 3:
                    paths.append(inner)
                    roles.append("outer_contour" if perimeter_index == 0 else "inner_contour")
    return paths, roles


def _iter_polygons(geometry):
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            if isinstance(item, Polygon):
                yield item
            elif isinstance(item, MultiPolygon):
                yield from item.geoms


def _coords_to_path(coords, tolerance: float) -> np.ndarray:
    points = np.asarray([[float(x), float(y)] for x, y in coords], dtype=np.float32)
    if points.shape[0] > 1 and not _close(points[0], points[-1], tolerance):
        points = np.vstack([points, points[0]])
    return _dedupe_consecutive(points, tolerance)


def _parallel_infill_geometry(
    geometry,
    spacing: float,
    angle_degrees: float,
    tolerance: float,
    scan_origin: float,
) -> list[np.ndarray]:
    if spacing <= 0:
        raise ValueError("infill spacing must be positive")
    if geometry.is_empty:
        return []

    rotated = affinity.rotate(geometry, -angle_degrees, origin=(0, 0), use_radians=False)
    min_x, min_y, max_x, max_y = rotated.bounds
    padding = spacing * 2.0
    first_index = math.ceil((min_y + tolerance - scan_origin) / spacing)
    last_index = math.floor((max_y - tolerance - scan_origin) / spacing)
    paths: list[np.ndarray] = []

    for scan_index in range(first_index, last_index + 1):
        scan_y = scan_origin + scan_index * spacing
        line = LineString([(min_x - padding, scan_y), (max_x + padding, scan_y)])
        intersection = rotated.intersection(line)
        for segment in _extract_line_segments(intersection, tolerance):
            restored = affinity.rotate(segment, angle_degrees, origin=(0, 0), use_radians=False)
            clipped = geometry.intersection(restored)
            for clipped_segment in _extract_line_segments(clipped, tolerance):
                coords = list(clipped_segment.coords)
                if len(coords) >= 2:
                    path = np.asarray([[float(x), float(y)] for x, y in coords], dtype=np.float32)
                    if np.linalg.norm(path[-1] - path[0]) > tolerance:
                        paths.append(path)
    return paths


def _concentric_infill_geometry(
    geometry,
    line_width: float,
    path_spacing: float,
    tolerance: float,
) -> list[np.ndarray]:
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if path_spacing <= 0:
        raise ValueError("path_spacing must be positive")
    if geometry.is_empty:
        return []

    max_offset = _max_geometry_offset(geometry, tolerance)
    if max_offset * 2.0 < line_width - tolerance:
        return []

    offsets = _uniform_concentric_offsets(max_offset, line_width, path_spacing)
    paths: list[np.ndarray] = []
    for offset in offsets:
        paths.extend(_concentric_paths_at_offset(geometry, offset, tolerance))
    paths = _filter_concentric_paths_by_spacing(paths, path_spacing, tolerance)
    return optimize_open_path_travel(paths, tolerance)


def _uncovered_region_fallback_paths(
    geometry,
    paths: list[np.ndarray],
    line_width: float,
    path_spacing: float,
    tolerance: float,
) -> list[np.ndarray]:
    if geometry.is_empty or not paths:
        return []

    lines = [
        LineString([(float(point[0]), float(point[1])) for point in path])
        for path in paths
        if path.shape[0] >= 2
    ]
    if not lines:
        return _linear_fallback_infill_geometry(geometry, path_spacing, 0.0, tolerance)

    coverage_radius = min(line_width * 0.5, path_spacing * 0.45)
    covered = unary_union(
        [
            line.buffer(coverage_radius, cap_style="round", join_style="round")
            for line in lines
            if not line.is_empty
        ]
    )
    residual = geometry.difference(covered)
    if residual.is_empty:
        return []

    fallback: list[np.ndarray] = []
    minimum_area = max((line_width * line_width) * 0.2, tolerance * 100.0)
    residual_polygons = [
        polygon for polygon in _iter_polygons(residual)
        if polygon.area >= minimum_area
    ]
    residual_polygons.sort(key=lambda polygon: polygon.area, reverse=True)
    for polygon in residual_polygons[:6]:
        fallback.extend(_centerline_fallback_infill_geometry(polygon, 0.0, tolerance))
    return fallback


def _linear_fallback_infill_geometry(
    geometry,
    spacing: float,
    angle_degrees: float,
    tolerance: float,
) -> list[np.ndarray]:
    if geometry.is_empty:
        return []

    min_x, min_y, max_x, max_y = geometry.bounds
    if min(max_x - min_x, max_y - min_y) < spacing - tolerance:
        return _centerline_fallback_infill_geometry(geometry, angle_degrees, tolerance)

    candidates = [
        _zigzag_infill_geometry(geometry, spacing, angle_degrees, tolerance),
        _zigzag_infill_geometry(geometry, spacing, angle_degrees + 90.0, tolerance),
    ]
    best = max(candidates, key=_paths_total_length, default=[])
    centerline = _centerline_fallback_infill_geometry(geometry, angle_degrees, tolerance)
    if best and _paths_total_length(best) > _paths_total_length(centerline) * 1.2:
        return best
    return centerline


def _paths_total_length(paths: list[np.ndarray]) -> float:
    return sum(
        float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
        for path in paths
        if path.shape[0] >= 2
    )


def _centerline_fallback_infill_geometry(
    geometry,
    angle_degrees: float,
    tolerance: float,
) -> list[np.ndarray]:
    rotated = affinity.rotate(geometry, -angle_degrees, origin=(0, 0), use_radians=False)
    min_x, min_y, max_x, max_y = rotated.bounds
    point = rotated.representative_point()
    scan_y = float(point.y)
    horizontal = LineString([(min_x - tolerance, scan_y), (max_x + tolerance, scan_y)])
    vertical_x = float(point.x)
    vertical = LineString([(vertical_x, min_y - tolerance), (vertical_x, max_y + tolerance)])
    segments = list(_extract_line_segments(rotated.intersection(horizontal), tolerance))
    segments.extend(_extract_line_segments(rotated.intersection(vertical), tolerance))
    if not segments:
        return []

    longest = max(segments, key=lambda segment: segment.length)
    restored = affinity.rotate(longest, angle_degrees, origin=(0, 0), use_radians=False)
    path = np.asarray([[float(x), float(y)] for x, y in restored.coords], dtype=np.float32)
    path = _dedupe_consecutive(path, tolerance)
    return [path] if path.shape[0] >= 2 else []


def _uniform_concentric_offsets(
    max_offset: float,
    line_width: float,
    path_spacing: float,
) -> list[float]:
    first_offset = line_width * 0.5
    if max_offset <= first_offset:
        return [max_offset]

    offsets: list[float] = []
    offset = first_offset
    while offset <= max_offset + 1e-9:
        offsets.append(offset)
        offset += path_spacing

    residual = max_offset - offsets[-1]
    if residual > line_width * 0.2:
        offsets.append(max_offset)
    return offsets


def _filter_concentric_paths_by_spacing(
    paths: list[np.ndarray], path_spacing: float, tolerance: float
) -> list[np.ndarray]:
    accepted: list[np.ndarray] = []
    accepted_lines: list[LineString] = []
    minimum_spacing = path_spacing * 0.95

    for path in paths:
        if path.shape[0] < 2:
            continue
        line = LineString([(float(point[0]), float(point[1])) for point in path])
        if line.is_empty:
            continue
        if any(line.distance(existing) < minimum_spacing - tolerance for existing in accepted_lines):
            continue
        accepted.append(path)
        accepted_lines.append(line)
    return accepted


def _concentric_paths_at_offset(
    geometry,
    offset_distance: float,
    tolerance: float,
) -> list[np.ndarray]:
    offset_geometry = geometry.buffer(-offset_distance, join_style="round")
    if offset_geometry.is_empty:
        return []

    paths: list[np.ndarray] = []
    for polygon in _iter_polygons(offset_geometry):
        exterior = _coords_to_path(polygon.exterior.coords, tolerance)
        exterior = _normalize_degenerate_ring(exterior, tolerance)
        if exterior.shape[0] >= 3:
            paths.append(exterior)
        elif exterior.shape[0] == 2:
            paths.append(exterior)
        for interior in polygon.interiors:
            inner = _coords_to_path(interior.coords, tolerance)
            inner = _normalize_degenerate_ring(inner, tolerance)
            if inner.shape[0] >= 3:
                paths.append(inner)
            elif inner.shape[0] == 2:
                paths.append(inner)
    return paths


def _normalize_degenerate_ring(path: np.ndarray, tolerance: float) -> np.ndarray:
    if path.shape[0] < 3 or not _is_closed_path(path, tolerance):
        return path

    unique = _dedupe_consecutive(path[:-1], tolerance)
    if unique.shape[0] < 3:
        return unique

    width = float(np.max(unique[:, 0]) - np.min(unique[:, 0]))
    height = float(np.max(unique[:, 1]) - np.min(unique[:, 1]))
    if min(width, height) > tolerance:
        return path

    best_pair = (0, 1)
    best_distance = -1.0
    for first_index in range(unique.shape[0]):
        for second_index in range(first_index + 1, unique.shape[0]):
            distance = float(np.linalg.norm(unique[first_index] - unique[second_index]))
            if distance > best_distance:
                best_pair = (first_index, second_index)
                best_distance = distance
    if best_distance <= tolerance:
        return unique[:1]
    return np.asarray([unique[best_pair[0]], unique[best_pair[1]]], dtype=np.float32)


def _max_geometry_offset(geometry, tolerance: float) -> float:
    min_x, min_y, max_x, max_y = geometry.bounds
    empty_offset = max(max_x - min_x, max_y - min_y, tolerance)
    while not geometry.buffer(-empty_offset, join_style="round").is_empty:
        empty_offset *= 2.0
    return _max_concentric_offset(geometry, 0.0, empty_offset, tolerance)


def _max_concentric_offset(
    geometry,
    valid_offset: float,
    empty_offset: float,
    tolerance: float,
) -> float:
    low = valid_offset
    high = empty_offset
    for _ in range(24):
        mid = (low + high) * 0.5
        if geometry.buffer(-mid, join_style="round").is_empty:
            high = mid
        else:
            low = mid
        if high - low <= max(tolerance, 1e-6):
            break
    return low


def _zigzag_infill_geometry(
    geometry,
    spacing: float,
    angle_degrees: float,
    tolerance: float,
    max_connector_distance: float | None = None,
    allow_boundary_route: bool = False,
) -> list[np.ndarray]:
    if spacing <= 0:
        raise ValueError("infill spacing must be positive")
    if geometry.is_empty:
        return []

    rotated = affinity.rotate(geometry, -angle_degrees, origin=(0, 0), use_radians=False)
    min_x, min_y, max_x, max_y = rotated.bounds
    padding = spacing * 2.0
    scan_y = math.ceil((min_y + tolerance) / spacing) * spacing
    row_index = 0
    chains: list[list[tuple[float, float]]] = []
    active_chain_indices: list[int] = []

    while scan_y < max_y - tolerance:
        line = LineString([(min_x - padding, scan_y), (max_x + padding, scan_y)])
        row_segments = []
        for segment in _extract_line_segments(rotated.intersection(line), tolerance):
            coords = sorted(list(segment.coords), key=lambda point: point[0])
            if len(coords) < 2:
                continue
            start = (float(coords[0][0]), float(coords[0][1]))
            end = (float(coords[-1][0]), float(coords[-1][1]))
            if math.dist(start, end) > tolerance:
                row_segments.append((start, end))

        row_segments.sort(key=lambda item: item[0][0])
        candidate_segments = list(reversed(row_segments)) if row_index % 2 == 1 else row_segments

        next_active_chain_indices: list[int] = []
        used_active_indices: set[int] = set()
        for segment_start, segment_end in candidate_segments:
            preferred = (
                (segment_end, segment_start)
                if row_index % 2 == 1
                else (segment_start, segment_end)
            )
            alternate = (preferred[1], preferred[0])
            orientations = [preferred, alternate]
            best_active_index: int | None = None
            best_orientation = preferred
            best_distance = math.inf
            for active_index, chain_index in enumerate(active_chain_indices):
                if active_index in used_active_indices:
                    continue
                chain = chains[chain_index]
                for orientation_index, (start, end) in enumerate(orientations):
                    distance = math.dist(chain[-1], start)
                    if orientation_index > 0:
                        distance += spacing * 0.25
                    if distance >= best_distance:
                        continue
                    connector_path = _connector_path_within_geometry(
                        rotated,
                        chain[-1],
                        start,
                        tolerance,
                        chains,
                        max_connector_distance,
                        allow_boundary_route,
                    )
                    if connector_path is not None:
                        best_active_index = active_index
                        best_orientation = (start, end)
                        best_distance = distance

            start, end = best_orientation

            if best_active_index is None:
                chains.append([start, end])
                next_active_chain_indices.append(len(chains) - 1)
                continue

            used_active_indices.add(best_active_index)
            best_chain_index = active_chain_indices[best_active_index]
            chain = chains[best_chain_index]
            connector_path = _connector_path_within_geometry(
                rotated,
                chain[-1],
                start,
                tolerance,
                chains,
                max_connector_distance,
                allow_boundary_route,
            )
            if connector_path is not None:
                chain.extend(connector_path[1:])
            chain.append(end)
            next_active_chain_indices.append(best_chain_index)

        scan_y += spacing
        row_index += 1
        active_chain_indices = next_active_chain_indices

    paths: list[np.ndarray] = []
    for chain in chains:
        if len(chain) < 2:
            continue
        restored = affinity.rotate(LineString(chain), angle_degrees, origin=(0, 0), use_radians=False)
        path = np.asarray([[float(x), float(y)] for x, y in restored.coords], dtype=np.float32)
        path = _dedupe_consecutive(path, tolerance)
        if path.shape[0] >= 2 and np.linalg.norm(path[-1] - path[0]) > tolerance:
            paths.append(path)
    return paths


def _triangular_lattice_infill_geometry(
    geometry,
    spacing: float,
    minimum_feature_length: float,
    connector_length_factor: float,
    connector_requires_lattice_direction: bool,
    tolerance: float,
) -> list[np.ndarray]:
    if spacing <= 0:
        raise ValueError("infill spacing must be positive")
    if geometry.is_empty:
        return []

    edges, coordinates = _optimized_triangular_lattice_edges(
        geometry,
        spacing,
        minimum_feature_length,
        connector_length_factor,
        connector_requires_lattice_direction,
        tolerance,
    )
    if not edges:
        return []
    return _trace_unique_graph_edges(
        edges,
        coordinates,
        tolerance,
        geometry,
        spacing * connector_length_factor,
        minimum_feature_length,
        connector_requires_lattice_direction,
    )


def _triangular_connector_length_factor(density_percent: float) -> float:
    if density_percent >= 70.0:
        return 1.5
    if density_percent >= 60.0:
        return 1.5
    return 1.25


def _triangular_connector_requires_lattice_direction(density_percent: float) -> bool:
    return True


def _optimized_triangular_lattice_edges(
    geometry,
    spacing: float,
    minimum_feature_length: float,
    connector_length_factor: float,
    connector_requires_lattice_direction: bool,
    tolerance: float,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[int, int], tuple[float, float]]]:
    best_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    best_coordinates: dict[tuple[int, int], tuple[float, float]] = {}
    best_score: tuple[int, int, int, int] | None = None

    for phase_fraction in (0.0, 0.2, 0.4, 0.6, 0.8):
        edges, coordinates = _triangular_lattice_edges_at_phase(
            geometry,
            spacing,
            spacing * phase_fraction,
            minimum_feature_length,
            tolerance,
        )
        if not edges:
            continue
        traced_paths = _trace_unique_graph_edges(
            edges,
            coordinates,
            tolerance,
            geometry,
            spacing * connector_length_factor,
            minimum_feature_length,
            connector_requires_lattice_direction,
        )
        score = _triangular_lattice_phase_score(
            edges,
            coordinates,
            minimum_feature_length,
            tolerance,
            len(traced_paths),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_edges = edges
            best_coordinates = coordinates

    return best_edges, best_coordinates


def _triangular_lattice_edges_at_phase(
    geometry,
    spacing: float,
    scan_origin: float,
    minimum_feature_length: float,
    tolerance: float,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[int, int], tuple[float, float]]]:
    candidate_paths: list[np.ndarray] = []
    for angle in (0.0, 60.0, -60.0):
        angle_paths = _parallel_infill_geometry(
            geometry,
            spacing,
            angle,
            tolerance,
            scan_origin,
        )
        if not angle_paths:
            rotated = affinity.rotate(geometry, -angle, origin=(0, 0), use_radians=False)
            _, min_y, _, max_y = rotated.bounds
            angle_paths = _parallel_infill_geometry(
                geometry,
                spacing,
                angle,
                tolerance,
                (min_y + max_y) * 0.5,
            )
        candidate_paths.extend(angle_paths)
    if not candidate_paths:
        return [], {}

    noded = unary_union(
        [
            LineString([(float(point[0]), float(point[1])) for point in path])
            for path in candidate_paths
            if path.shape[0] >= 2
        ]
    )
    edges, coordinates = _unique_line_graph_edges(noded, tolerance)
    edges = _filter_short_lattice_edges(edges, coordinates, minimum_feature_length, tolerance)
    return edges, coordinates


def _triangular_lattice_phase_score(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    minimum_feature_length: float,
    tolerance: float,
    path_count: int,
) -> tuple[int, int, int, int, int]:
    directions = _triangular_lattice_directions(edges, coordinates, tolerance)
    direction_penalty = 0 if directions == {0, 60, 120} else 1
    short_edges = sum(
        1
        for start, end in edges
        if math.dist(coordinates[start], coordinates[end]) < minimum_feature_length * 1.35 - tolerance
    )
    odd_vertices = _graph_odd_vertex_count(edges)
    return (direction_penalty, short_edges + path_count, path_count, odd_vertices, -len(edges))


def _graph_odd_vertex_count(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> int:
    degree: dict[tuple[int, int], int] = defaultdict(int)
    for start, end in edges:
        degree[start] += 1
        degree[end] += 1
    return sum(1 for value in degree.values() if value % 2 == 1)


def _unique_line_graph_edges(
    geometry,
    tolerance: float,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[int, int], tuple[float, float]]]:
    key_step = max(tolerance * 100.0, 1e-4)
    coordinates: dict[tuple[int, int], tuple[float, float]] = {}
    edges_by_key: dict[
        tuple[tuple[int, int], tuple[int, int]],
        tuple[tuple[int, int], tuple[int, int]],
    ] = {}

    def point_key(point: tuple[float, float]) -> tuple[int, int]:
        key = (round(float(point[0]) / key_step), round(float(point[1]) / key_step))
        coordinates.setdefault(key, (float(point[0]), float(point[1])))
        return key

    for line in _extract_line_segments(geometry, tolerance):
        coords = [(float(x), float(y)) for x, y in line.coords]
        for start, end in zip(coords, coords[1:]):
            if math.dist(start, end) <= tolerance:
                continue
            start_key = point_key(start)
            end_key = point_key(end)
            if start_key == end_key:
                continue
            edge_key = tuple(sorted((start_key, end_key)))
            edges_by_key[edge_key] = (start_key, end_key)

    return list(edges_by_key.values()), coordinates


def _filter_short_lattice_edges(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    minimum_length: float,
    tolerance: float,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if minimum_length <= tolerance:
        return edges
    filtered = [
        (start, end)
        for start, end in edges
        if math.dist(coordinates[start], coordinates[end]) >= minimum_length - tolerance
    ]
    if not filtered:
        return edges
    if _triangular_lattice_directions(filtered, coordinates, tolerance) != {0, 60, 120}:
        return edges
    return filtered


def _prune_triangular_lattice_dangling_edges(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    tolerance: float,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(edges) < 12:
        return edges

    pruned = list(edges)
    while True:
        degree: dict[tuple[int, int], int] = defaultdict(int)
        for start, end in pruned:
            degree[start] += 1
            degree[end] += 1
        next_edges = [
            (start, end)
            for start, end in pruned
            if degree[start] > 1 and degree[end] > 1
        ]
        if len(next_edges) == len(pruned):
            break
        pruned = next_edges
        if not pruned:
            return edges

    if len(pruned) < len(edges) * 0.25:
        return edges
    if _triangular_lattice_directions(pruned, coordinates, tolerance) != {0, 60, 120}:
        return edges
    return pruned


def _triangular_lattice_directions(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    tolerance: float,
) -> set[int]:
    directions: set[int] = set()
    for start, end in edges:
        dx = coordinates[end][0] - coordinates[start][0]
        dy = coordinates[end][1] - coordinates[start][1]
        if math.hypot(dx, dy) <= max(1.0, tolerance):
            continue
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        for expected in (0.0, 60.0, 120.0):
            if abs(angle - expected) < 2.0:
                directions.add(int(expected))
    return directions


def _trace_unique_graph_edges(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    tolerance: float,
    geometry=None,
    max_connector_length: float | None = None,
    minimum_connector_length: float = 0.0,
    connector_requires_lattice_direction: bool = True,
) -> list[np.ndarray]:
    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for edge_index, (start, end) in enumerate(edges):
        adjacency[start].append((edge_index, end))
        adjacency[end].append((edge_index, start))

    paths: list[np.ndarray] = []
    remaining_vertices = set(adjacency)
    while remaining_vertices:
        seed = remaining_vertices.pop()
        component_vertices = _graph_component_vertices(seed, adjacency)
        remaining_vertices.difference_update(component_vertices)
        component_edges = {
            edge_index
            for vertex in component_vertices
            for edge_index, _ in adjacency[vertex]
        }
        paths.extend(
            _euler_trails_for_component(
                component_vertices,
                component_edges,
                edges,
                coordinates,
                tolerance,
                geometry,
                max_connector_length,
                minimum_connector_length,
                connector_requires_lattice_direction,
            )
        )

    return optimize_open_path_travel(paths, tolerance)


def _graph_component_vertices(
    seed: tuple[int, int],
    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]],
) -> set[tuple[int, int]]:
    component = {seed}
    queue: deque[tuple[int, int]] = deque([seed])
    while queue:
        current = queue.popleft()
        for _, next_vertex in adjacency[current]:
            if next_vertex in component:
                continue
            component.add(next_vertex)
            queue.append(next_vertex)
    return component


def _euler_trails_for_component(
    component_vertices: set[tuple[int, int]],
    component_edges: set[int],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    tolerance: float,
    geometry=None,
    max_connector_length: float | None = None,
    minimum_connector_length: float = 0.0,
    connector_requires_lattice_direction: bool = True,
) -> list[np.ndarray]:
    euler_edges: dict[int, tuple[tuple[int, int], tuple[int, int], bool]] = {
        edge_index: (start, end, False)
        for edge_index, (start, end) in enumerate(edges)
        if edge_index in component_edges
    }
    existing_graph = unary_union(
        [
            LineString([coordinates[start], coordinates[end]])
            for start, end, _ in euler_edges.values()
        ]
    )
    odd_vertices = [
        vertex
        for vertex in component_vertices
        if sum(1 for edge_index in component_edges if vertex in edges[edge_index]) % 2 == 1
    ]
    paired_odds = _pair_odd_vertices_for_connectors(
        odd_vertices,
        coordinates,
        geometry,
        existing_graph,
        max_connector_length,
        minimum_connector_length,
        connector_requires_lattice_direction,
        tolerance,
    )
    next_dummy_id = len(edges)
    for start, end, is_dummy in paired_odds:
        euler_edges[next_dummy_id] = (start, end, is_dummy)
        next_dummy_id += 1

    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for edge_index, (start, end, _) in euler_edges.items():
        adjacency[start].append((edge_index, end))
        adjacency[end].append((edge_index, start))

    start_vertex = min(component_vertices, key=lambda key: (coordinates[key][1], coordinates[key][0]))
    circuit_vertices, circuit_edges = _hierholzer_circuit(start_vertex, adjacency)
    trail_vertices = _split_circuit_at_dummy_edges(circuit_vertices, circuit_edges, euler_edges)

    paths: list[np.ndarray] = []
    for path_vertices in trail_vertices:
        if len(path_vertices) < 2:
            continue
        path = np.asarray([coordinates[key] for key in path_vertices], dtype=np.float32)
        path = _dedupe_consecutive(path, tolerance)
        if path.shape[0] >= 2:
            paths.append(path)
    return paths


def _pair_odd_vertices_for_connectors(
    odd_vertices: list[tuple[int, int]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    geometry,
    existing_graph,
    max_connector_length: float | None,
    minimum_connector_length: float,
    connector_requires_lattice_direction: bool,
    tolerance: float,
) -> list[tuple[tuple[int, int], tuple[int, int], bool]]:
    remaining = set(odd_vertices)
    pairs: list[tuple[tuple[int, int], tuple[int, int], bool]] = []
    printed_connectors: list[LineString] = []

    candidates: list[tuple[float, tuple[int, int], tuple[int, int], LineString]] = []
    for index, start in enumerate(odd_vertices):
        for end in odd_vertices[index + 1 :]:
            connector = LineString([coordinates[start], coordinates[end]])
            if _printable_odd_connector(
                connector,
                geometry,
                existing_graph,
                [],
                max_connector_length,
                minimum_connector_length,
                connector_requires_lattice_direction,
                tolerance,
            ):
                candidates.append((connector.length, start, end, connector))

    for _, start, end, connector in sorted(candidates, key=lambda item: item[0]):
        if start not in remaining or end not in remaining:
            continue
        if not all(_connector_has_no_line_overlap(connector, existing, tolerance) for existing in printed_connectors):
            continue
        remaining.remove(start)
        remaining.remove(end)
        printed_connectors.append(connector)
        pairs.append((start, end, False))

    dummy_pairs = _pair_odd_vertices(list(remaining), coordinates)
    pairs.extend((start, end, True) for start, end in dummy_pairs)
    return pairs


def _printable_odd_connector(
    connector: LineString,
    geometry,
    existing_graph,
    printed_connectors: list[LineString],
    max_connector_length: float | None,
    minimum_connector_length: float,
    connector_requires_lattice_direction: bool,
    tolerance: float,
) -> bool:
    if connector.length <= tolerance:
        return False
    if connector.length < minimum_connector_length - tolerance:
        return False
    if connector_requires_lattice_direction and not _line_matches_triangular_lattice_direction(connector, tolerance):
        return False
    if max_connector_length is not None and connector.length > max_connector_length + tolerance:
        return False
    if geometry is not None:
        if not geometry.buffer(max(tolerance * 10.0, 1e-6)).covers(connector):
            return False
        if _connector_runs_too_close_to_boundary(
            connector,
            geometry,
            minimum_connector_length * 0.6,
            tolerance,
        ):
            return False

    if not _connector_has_no_line_overlap(connector, existing_graph, tolerance):
        return False
    for existing in printed_connectors:
        if not _connector_has_no_line_overlap(connector, existing, tolerance):
            return False
    return True


def _line_matches_triangular_lattice_direction(line: LineString, tolerance: float) -> bool:
    coords = list(line.coords)
    if len(coords) < 2:
        return False
    dx = float(coords[-1][0] - coords[0][0])
    dy = float(coords[-1][1] - coords[0][1])
    if math.hypot(dx, dy) <= tolerance:
        return False
    angle = math.degrees(math.atan2(dy, dx)) % 180.0
    return any(abs(angle - expected) < 2.0 for expected in (0.0, 60.0, 120.0))


def _connector_runs_too_close_to_boundary(
    connector: LineString,
    geometry,
    clearance: float,
    tolerance: float,
) -> bool:
    if clearance <= tolerance:
        return False
    boundary = geometry.boundary
    for fraction in (0.25, 0.5, 0.75):
        sample = connector.interpolate(fraction, normalized=True)
        if sample.distance(boundary) < clearance - tolerance:
            return True
    return False


def _line_angle_degrees(line: LineString) -> float | None:
    coords = list(line.coords)
    if len(coords) < 2:
        return None
    dx = float(coords[-1][0] - coords[0][0])
    dy = float(coords[-1][1] - coords[0][1])
    if math.hypot(dx, dy) <= 1e-12:
        return None
    return math.degrees(math.atan2(dy, dx)) % 180.0


def _has_parallel_boundary_segment_near(
    point,
    boundary,
    angle_degrees: float,
    clearance: float,
    tolerance: float,
) -> bool:
    for segment in _boundary_segments(boundary, tolerance):
        if point.distance(segment) > clearance + tolerance:
            continue
        boundary_angle = _line_angle_degrees(segment)
        if boundary_angle is None:
            continue
        delta = abs((angle_degrees - boundary_angle + 90.0) % 180.0 - 90.0)
        if delta <= 10.0:
            return True
    return False


def _boundary_segments(boundary, tolerance: float):
    if isinstance(boundary, LineString):
        coords = list(boundary.coords)
        for start, end in zip(coords, coords[1:]):
            segment = LineString([start, end])
            if segment.length > tolerance:
                yield segment
    elif isinstance(boundary, MultiLineString):
        for line in boundary.geoms:
            yield from _boundary_segments(line, tolerance)
    elif isinstance(boundary, GeometryCollection):
        for item in boundary.geoms:
            yield from _boundary_segments(item, tolerance)


def _connector_has_no_line_overlap(
    connector: LineString,
    other,
    tolerance: float,
) -> bool:
    intersection = connector.intersection(other)
    if intersection.is_empty:
        return True
    return float(getattr(intersection, "length", 0.0)) <= tolerance


def _pair_odd_vertices(
    odd_vertices: list[tuple[int, int]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    remaining = set(odd_vertices)
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while remaining:
        start = min(remaining, key=lambda key: (coordinates[key][1], coordinates[key][0]))
        remaining.remove(start)
        end = min(remaining, key=lambda key: math.dist(coordinates[start], coordinates[key]))
        remaining.remove(end)
        pairs.append((start, end))
    return pairs


def _hierholzer_circuit(
    start: tuple[int, int],
    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]],
) -> tuple[list[tuple[int, int]], list[int]]:
    unused = {edge_index for entries in adjacency.values() for edge_index, _ in entries}
    stack: list[tuple[tuple[int, int], int | None]] = [(start, None)]
    circuit_vertices: list[tuple[int, int]] = []
    circuit_edges: list[int] = []

    while stack:
        current, incoming_edge = stack[-1]
        while adjacency[current] and adjacency[current][-1][0] not in unused:
            adjacency[current].pop()
        if not adjacency[current]:
            circuit_vertices.append(current)
            if incoming_edge is not None:
                circuit_edges.append(incoming_edge)
            stack.pop()
            continue
        edge_index, next_vertex = adjacency[current].pop()
        if edge_index not in unused:
            continue
        unused.remove(edge_index)
        stack.append((next_vertex, edge_index))

    circuit_vertices.reverse()
    circuit_edges.reverse()
    return circuit_vertices, circuit_edges


def _split_circuit_at_dummy_edges(
    circuit_vertices: list[tuple[int, int]],
    circuit_edges: list[int],
    euler_edges: dict[int, tuple[tuple[int, int], tuple[int, int], bool]],
) -> list[list[tuple[int, int]]]:
    if not circuit_edges:
        return []
    dummy_positions = [
        index for index, edge_index in enumerate(circuit_edges) if euler_edges[edge_index][2]
    ]
    if not dummy_positions:
        return [circuit_vertices]

    split_paths: list[list[tuple[int, int]]] = []
    start_index = (dummy_positions[0] + 1) % len(circuit_edges)
    ordered_edge_indices = [
        (start_index + offset) % len(circuit_edges) for offset in range(len(circuit_edges))
    ]
    current_path = [circuit_vertices[start_index]]
    for edge_position in ordered_edge_indices:
        edge_index = circuit_edges[edge_position]
        next_vertex = circuit_vertices[(edge_position + 1) % len(circuit_vertices)]
        if euler_edges[edge_index][2]:
            if len(current_path) >= 2:
                split_paths.append(current_path)
            current_path = [next_vertex]
            continue
        current_path.append(next_vertex)
    if len(current_path) >= 2:
        split_paths.append(current_path)
    return split_paths


def _graph_trace_start_vertex(
    unused_edges: set[int],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
) -> tuple[int, int]:
    vertices = {vertex for edge_index in unused_edges for vertex in edges[edge_index]}
    odd_vertices = [
        vertex
        for vertex in vertices
        if sum(1 for edge_index, _ in adjacency[vertex] if edge_index in unused_edges) % 2 == 1
    ]
    candidates = odd_vertices or list(vertices)
    return min(candidates, key=lambda key: (coordinates[key][1], coordinates[key][0]))


def _next_graph_edge(
    current: tuple[int, int],
    previous_vector: tuple[float, float] | None,
    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]],
    unused_edges: set[int],
    coordinates: dict[tuple[int, int], tuple[float, float]],
) -> tuple[int, tuple[int, int]] | None:
    candidates = [
        (edge_index, next_vertex)
        for edge_index, next_vertex in adjacency[current]
        if edge_index in unused_edges
    ]
    if not candidates:
        return None
    if previous_vector is None:
        return min(candidates, key=lambda item: (coordinates[item[1]][1], coordinates[item[1]][0]))

    previous_angle = math.atan2(previous_vector[1], previous_vector[0])

    def turn_cost(candidate: tuple[int, tuple[int, int]]) -> tuple[float, float]:
        _, next_vertex = candidate
        dx = coordinates[next_vertex][0] - coordinates[current][0]
        dy = coordinates[next_vertex][1] - coordinates[current][1]
        angle = math.atan2(dy, dx)
        turn = abs((angle - previous_angle + math.pi) % (2.0 * math.pi) - math.pi)
        distance = math.hypot(dx, dy)
        return (turn, -distance)

    return min(candidates, key=turn_cost)


def _triangular_wave_infill_geometry(
    geometry,
    spacing: float,
    tolerance: float,
) -> list[np.ndarray]:
    if spacing <= 0:
        raise ValueError("infill spacing must be positive")
    if geometry.is_empty:
        return []

    min_x, min_y, max_x, max_y = geometry.bounds
    height = spacing
    half_period = height / math.sqrt(3.0)
    padding = max(spacing * 2.0, half_period * 4.0)
    row_y = math.floor((min_y - padding) / height) * height
    paths: list[np.ndarray] = []
    row_index = 0

    while row_y <= max_y + padding:
        x = min_x - padding
        high_y = row_y + height
        points: list[tuple[float, float]] = []
        point_index = 0
        while x <= max_x + padding:
            if row_index % 2 == 0:
                y = row_y if point_index % 2 == 0 else high_y
            else:
                y = high_y if point_index % 2 == 0 else row_y
            points.append((float(x), float(y)))
            x += half_period
            point_index += 1

        clipped = geometry.intersection(LineString(points))
        row_segments = sorted(
            _extract_line_segments(clipped, tolerance),
            key=lambda segment: min(point[0] for point in segment.coords),
            reverse=bool(row_index % 2),
        )
        for segment in row_segments:
            coords = list(segment.coords)
            if row_index % 2:
                coords = list(reversed(coords))
            path = np.asarray([[float(x), float(y)] for x, y in coords], dtype=np.float32)
            path = _dedupe_consecutive(path, tolerance)
            if path.shape[0] >= 2 and np.linalg.norm(path[-1] - path[0]) > tolerance:
                paths.append(path)

        row_y += height
        row_index += 1

    return paths


def _connector_within_geometry(
    geometry,
    start: tuple[float, float],
    end: tuple[float, float],
    tolerance: float,
    existing_chains: list[list[tuple[float, float]]] | None = None,
    max_distance: float | None = None,
    allow_boundary_route: bool = False,
) -> bool:
    return _connector_path_within_geometry(
        geometry,
        start,
        end,
        tolerance,
        existing_chains,
        max_distance,
        allow_boundary_route,
    ) is not None


def _connector_path_within_geometry(
    geometry,
    start: tuple[float, float],
    end: tuple[float, float],
    tolerance: float,
    existing_chains: list[list[tuple[float, float]]] | None = None,
    max_distance: float | None = None,
    allow_boundary_route: bool = False,
) -> list[tuple[float, float]] | None:
    if _points_close_2d(start, end, tolerance):
        return [start, end]
    if max_distance is not None and math.dist(start, end) > max_distance + tolerance:
        return None

    safe_geometry = geometry.buffer(max(tolerance * 10.0, 1e-6), join_style="round")
    connector = LineString([start, end])
    if safe_geometry.covers(connector) and _connector_clear_of_chains(
        connector, start, end, existing_chains, tolerance
    ):
        return [start, end]

    if not allow_boundary_route:
        return None

    boundary = geometry.boundary
    boundary_safe_geometry = geometry.buffer(0.15, join_style="round")
    candidates: list[tuple[float, list[tuple[float, float]]]] = []
    for ring in _extract_line_segments(boundary, tolerance):
        if not _is_closed_path(np.asarray(ring.coords, dtype=np.float32), tolerance):
            continue
        ring_length = float(ring.length)
        if ring_length <= tolerance:
            continue
        start_distance = float(ring.project(Point(start)))
        end_distance = float(ring.project(Point(end)))
        for direction in (1.0, -1.0):
            if direction > 0:
                delta = (end_distance - start_distance) % ring_length
            else:
                delta = (start_distance - end_distance) % ring_length
            samples = max(3, int(delta / max(tolerance * 100.0, 0.35)) + 1)
            boundary_points: list[tuple[float, float]] = []
            for index in range(samples + 1):
                distance = start_distance + direction * delta * index / samples
                distance %= ring_length
                point = ring.interpolate(distance)
                boundary_points.append((float(point.x), float(point.y)))
            route = [start, boundary_points[0], *boundary_points[1:-1], boundary_points[-1], end]
            route_line = LineString(route)
            if not route_line.is_simple:
                continue
            if not boundary_safe_geometry.covers(route_line):
                continue
            if not _connector_clear_of_chains(
                route_line,
                start,
                end,
                existing_chains,
                tolerance,
                boundary,
            ):
                continue
            route_length = float(route_line.length)
            candidates.append((route_length, route))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _connector_clear_of_chains(
    connector: LineString,
    start: tuple[float, float],
    end: tuple[float, float],
    existing_chains: list[list[tuple[float, float]]] | None,
    tolerance: float,
    allowed_boundary=None,
) -> bool:
    if existing_chains is None:
        return True
    for chain in existing_chains:
        if len(chain) < 2:
            continue
        intersection = connector.intersection(LineString(chain))
        if intersection.is_empty:
            continue
        if allowed_boundary is not None and allowed_boundary.buffer(max(tolerance * 10.0, 1e-6)).covers(intersection):
            continue
        endpoint_tolerance = max(tolerance * 10.0, 1e-6)
        allowed = Point(start).buffer(endpoint_tolerance).union(Point(end).buffer(endpoint_tolerance))
        if not allowed.covers(intersection):
            return False
    return True


def _points_close_2d(
    a: tuple[float, float],
    b: tuple[float, float],
    tolerance: float,
) -> bool:
    return math.dist(a, b) <= tolerance


def _extract_line_segments(geometry, tolerance: float):
    if geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        if geometry.length > tolerance:
            yield geometry
    elif isinstance(geometry, MultiLineString):
        for line in geometry.geoms:
            if line.length > tolerance:
                yield line
    elif isinstance(geometry, GeometryCollection):
        for item in geometry.geoms:
            yield from _extract_line_segments(item, tolerance)


def _contours_with_boundary_roles(contours: list[np.ndarray]) -> list[tuple[np.ndarray, str]]:
    contour_polygons: list[tuple[int, Polygon]] = []
    for index, contour in enumerate(contours):
        if contour.shape[0] < 3:
            continue
        polygon = Polygon(_open_ring(contour))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty:
            contour_polygons.append((index, polygon))

    result: list[tuple[np.ndarray, str]] = []
    for index, polygon in contour_polygons:
        containing_count = 0
        for other_index, other_polygon in contour_polygons:
            if other_index == index or other_polygon.area <= polygon.area:
                continue
            if other_polygon.contains(polygon) or other_polygon.covers(polygon):
                containing_count += 1
        role = "inner_boundary" if containing_count % 2 == 1 else "outer_boundary"
        result.append((contours[index], role))
    result.sort(key=lambda item: 0 if item[1] == "outer_boundary" else 1)
    return result


def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    inside = False
    points = polygon[:, :2]
    point_count = points.shape[0]
    for index in range(point_count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % point_count]
        if (float(y0) > y) == (float(y1) > y):
            continue
        crossing_x = float(x0) + (float(x1) - float(x0)) * (y - float(y0)) / (float(y1) - float(y0))
        if x < crossing_x:
            inside = not inside
    return inside


def _offset_contour_into_material(
    contour: np.ndarray, distance: float, boundary_role: str, tolerance: float
) -> np.ndarray:
    candidates = [
        _offset_contour(contour, distance, tolerance),
        _offset_contour(contour, -distance, tolerance),
    ]
    candidates = [candidate for candidate in candidates if candidate.shape[0] >= 3]
    if not candidates:
        return contour

    original_area = abs(_signed_area(contour))
    if boundary_role == "outer_boundary":
        smaller = [candidate for candidate in candidates if abs(_signed_area(candidate)) < original_area]
        return min(smaller or candidates, key=lambda candidate: abs(_signed_area(candidate)))
    larger = [candidate for candidate in candidates if abs(_signed_area(candidate)) > original_area]
    return max(larger or candidates, key=lambda candidate: abs(_signed_area(candidate)))


def _offset_contour(contour: np.ndarray, signed_distance: float, tolerance: float) -> np.ndarray:
    polygon = np.asarray(contour[:, :2], dtype=np.float32)
    if _close(polygon[0], polygon[-1], tolerance):
        polygon = polygon[:-1]
    if polygon.shape[0] < 3:
        return polygon

    offset_lines: list[tuple[np.ndarray, np.ndarray]] = []
    point_count = polygon.shape[0]
    for index in range(point_count):
        p0 = polygon[index]
        p1 = polygon[(index + 1) % point_count]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length <= tolerance:
            continue
        normal = np.asarray([-edge[1] / length, edge[0] / length], dtype=np.float32)
        offset_lines.append((p0 + normal * signed_distance, p1 + normal * signed_distance))

    if len(offset_lines) < 3:
        return polygon

    result: list[np.ndarray] = []
    for index in range(len(offset_lines)):
        prev_line = offset_lines[index - 1]
        current_line = offset_lines[index]
        intersection = _line_intersection(prev_line[0], prev_line[1], current_line[0], current_line[1], tolerance)
        if intersection is None:
            intersection = current_line[0]
        if not result or not _close(result[-1], intersection, tolerance):
            result.append(intersection)

    if result and not _close(result[0], result[-1], tolerance):
        result.append(result[0])
    return np.asarray(result, dtype=np.float32)


def _line_intersection(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, tolerance: float
) -> np.ndarray | None:
    da = a1 - a0
    db = b1 - b0
    cross = float(da[0] * db[1] - da[1] * db[0])
    if abs(cross) <= tolerance:
        return None
    delta = b0 - a0
    t = float(delta[0] * db[1] - delta[1] * db[0]) / cross
    return np.asarray(a0 + da * t, dtype=np.float32)


def _signed_area(contour: np.ndarray) -> float:
    polygon = np.asarray(contour[:, :2], dtype=np.float32)
    if polygon.shape[0] < 3:
        return 0.0
    if _close(polygon[0], polygon[-1], 1e-6):
        polygon = polygon[:-1]
    shifted = np.roll(polygon, -1, axis=0)
    return float(0.5 * np.sum(polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1]))


def _line_infill_spacing(
    line_width: float,
    density_percent: float,
    overlap_percent: float = 0.0,
) -> float:
    if density_percent <= 0:
        raise ValueError("density_percent must be positive for non-empty infill")
    return _resin_path_spacing(line_width, overlap_percent) / (density_percent / 100.0)


def _grid_infill_spacing(
    line_width: float,
    density_percent: float,
    overlap_percent: float = 0.0,
) -> float:
    return _multi_axis_infill_spacing(line_width, density_percent, 2, overlap_percent)


def _multi_axis_infill_spacing(
    line_width: float,
    density_percent: float,
    axis_count: int,
    overlap_percent: float = 0.0,
) -> float:
    if axis_count <= 0:
        raise ValueError("axis_count must be positive")
    if density_percent <= 0:
        raise ValueError("density_percent must be positive for non-empty infill")
    density = density_percent / 100.0
    path_spacing = _resin_path_spacing(line_width, overlap_percent)
    if density >= 0.999:
        return path_spacing
    per_axis_coverage = 1.0 - math.pow(1.0 - density, 1.0 / axis_count)
    return path_spacing / per_axis_coverage


def _infill_geometry_inset(config: SliceConfig) -> float:
    return max(
        config.line_width * 0.5,
        DEFAULT_RESIN_PERIMETER_COUNT
        * _resin_path_spacing(config.line_width, config.infill_overlap),
    )


def _resin_path_spacing(line_width: float, overlap_percent: float) -> float:
    return line_width - _resin_overlap_width(line_width, overlap_percent)


def _resin_overlap_width(line_width: float, overlap_percent: float) -> float:
    return line_width * overlap_percent / 100.0


def _parallel_infill(
    polygons: list[np.ndarray], spacing: float, angle_degrees: float, tolerance: float
) -> list[np.ndarray]:
    if spacing <= 0:
        raise ValueError("infill spacing must be positive")

    transformed_polygons: list[np.ndarray] = []
    for polygon in polygons:
        if polygon.shape[0] < 3:
            continue
        transformed = _rotate_points(polygon, -angle_degrees)
        if _close(transformed[0], transformed[-1], tolerance):
            transformed = transformed[:-1]
        if transformed.shape[0] >= 3:
            transformed_polygons.append(transformed)

    if not transformed_polygons:
        return []

    min_v = min(float(np.min(polygon[:, 1])) for polygon in transformed_polygons)
    max_v = max(float(np.max(polygon[:, 1])) for polygon in transformed_polygons)
    scan_v = math.ceil((min_v + tolerance) / spacing) * spacing
    paths: list[np.ndarray] = []

    while scan_v < max_v - tolerance:
        intersections: list[float] = []
        for polygon in transformed_polygons:
            intersections.extend(_scanline_intersections(polygon, scan_v, tolerance))
        intersections.sort()
        for start_u, end_u in _paired_intersections(intersections, tolerance):
            if end_u - start_u <= tolerance:
                continue
            segment = np.asarray([[start_u, scan_v], [end_u, scan_v]], dtype=np.float32)
            paths.append(_rotate_points(segment, angle_degrees))
        scan_v += spacing

    return paths


def _scanline_intersections(
    polygon: np.ndarray, scan_v: float, tolerance: float
) -> list[float]:
    intersections: list[float] = []
    point_count = polygon.shape[0]
    for index in range(point_count):
        p0 = polygon[index]
        p1 = polygon[(index + 1) % point_count]
        v0 = float(p0[1])
        v1 = float(p1[1])

        if abs(v0 - v1) <= tolerance:
            continue
        lower = min(v0, v1)
        upper = max(v0, v1)
        if scan_v < lower or scan_v >= upper:
            continue

        t = (scan_v - v0) / (v1 - v0)
        u = float(p0[0] + np.float32(t) * (p1[0] - p0[0]))
        intersections.append(u)

    intersections.sort()
    return intersections


def _paired_intersections(intersections: list[float], tolerance: float) -> list[tuple[float, float]]:
    deduped: list[float] = []
    for value in intersections:
        if not deduped or abs(value - deduped[-1]) > tolerance:
            deduped.append(value)

    pairs: list[tuple[float, float]] = []
    for index in range(0, len(deduped) - 1, 2):
        pairs.append((deduped[index], deduped[index + 1]))
    return pairs


def _rotate_points(points: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rotation = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return np.asarray(points, dtype=np.float32) @ rotation.T


def _build_z_projector(config: SliceConfig) -> Callable[[float, float, float], float]:
    if config.curve_mode == "flat":
        return lambda x, y, base_z: base_z
    if config.curve_mode == "sinusoidal":
        return lambda x, y, base_z: base_z + config.curve_amplitude * math.sin(
            (2.0 * math.pi * x) / config.curve_period
        )
    raise ValueError(f"unsupported curve mode: {config.curve_mode}")


def _path_2d_to_3d(
    path: np.ndarray, base_z: float, z_projector: Callable[[float, float, float], float]
) -> np.ndarray:
    result = np.empty((path.shape[0], 3), dtype=np.float32)
    result[:, :2] = path
    for index, (x, y) in enumerate(path):
        result[index, 2] = z_projector(float(x), float(y), base_z)
    return result
