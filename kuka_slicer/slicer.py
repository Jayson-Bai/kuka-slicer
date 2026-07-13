from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Callable, Literal

import numpy as np
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
from shapely.ops import linemerge, unary_union

from .external_npz import ExternalSourceJob, Material, MaterialPaths
from .stl_io import Mesh

CurveMode = Literal["flat", "sinusoidal"]
InfillPattern = Literal[
    "none",
    "rectilinear",
    "aligned_rectilinear",
    "line",
    "grid",
    "triangles",
    "gyroid",
    "concentric",
    "zigzag",
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
MIN_GEOMETRY_TOLERANCE_MM = 1e-5
MAX_GEOMETRY_TOLERANCE_MM = 1e-2

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


def recommended_geometry_tolerance(layer_height: float, line_width: float) -> float:
    print_scale = min(layer_height, line_width)
    tolerance = print_scale * 0.001
    return min(max(tolerance, MIN_GEOMETRY_TOLERANCE_MM), MAX_GEOMETRY_TOLERANCE_MM)


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
    infill_pattern: InfillPattern = "rectilinear"
    infill_density: float = DEFAULT_RESIN_INFILL_DENSITY_PERCENT
    infill_overlap: float = DEFAULT_RESIN_INFILL_OVERLAP_PERCENT
    build_axis: BuildAxis = "z"
    perimeter_count: int = DEFAULT_RESIN_PERIMETER_COUNT
    smoothing_angle: float = DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES
    smoothing_radius_factor: float = DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR

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
            "none",
            "rectilinear",
            "aligned_rectilinear",
            "line",
            "grid",
            "triangles",
            "gyroid",
            "concentric",
            "zigzag",
        ):
            raise ValueError("unsupported infill_pattern")
        if self.infill_density < 0 or self.infill_density > 100:
            raise ValueError("infill_density must be in the range [0, 100]")
        if self.infill_overlap < 0 or self.infill_overlap >= 100:
            raise ValueError("infill_overlap must be in the range [0, 100)")
        if self.build_axis not in ("x", "y", "z"):
            raise ValueError("build_axis must be x, y, or z")
        if self.perimeter_count < 1:
            raise ValueError("perimeter_count must be at least 1")
        if self.smoothing_angle <= 0 or self.smoothing_angle >= 180:
            raise ValueError("smoothing_angle must be in the range (0, 180)")
        if self.smoothing_radius_factor < 0:
            raise ValueError("smoothing_radius_factor must be non-negative")


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
            if (
                constant_section_paths is not None
                and config.infill_pattern == "triangles"
                and cached_constant_resin_paths_2d is not None
                and cached_constant_roles is not None
            ):
                paths_2d = [path.copy() for path in cached_constant_resin_paths_2d]
                roles = list(cached_constant_roles)
            else:
                paths_2d, roles = _build_resin_paths(paths_2d, config, layer_index)
                if constant_section_paths is not None and config.infill_pattern == "triangles":
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
            "perimeter_count": config.perimeter_count,
            "smoothing_angle": config.smoothing_angle,
            "smoothing_radius_factor": config.smoothing_radius_factor,
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
        config.perimeter_count,
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
    return filled
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
    )
    filled = _smooth_resin_infill_paths(
        filled,
        geometry,
        config.line_width * DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR,
        DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES,
        config.tolerance,
    )
    return filled


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
        config.perimeter_count,
        config.tolerance,
    )
    if config.infill_pattern == "none":
        return perimeters, roles

    infill_geometry = _resin_infill_surface_geometry(solid_geometry, config)
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

    if config.infill_pattern == "rectilinear":
        angle = 45.0 if layer_index % 2 == 0 else -45.0
        filled.extend(_zigzag_infill_geometry(geometry, line_spacing, angle, config.tolerance))
    elif config.infill_pattern == "aligned_rectilinear":
        filled.extend(_zigzag_infill_geometry(geometry, line_spacing, 0.0, config.tolerance))
    elif config.infill_pattern == "line":
        filled.extend(_zigzag_infill_geometry(geometry, line_spacing, 90.0, config.tolerance))
    elif config.infill_pattern == "concentric":
        filled.extend(
            _concentric_infill_geometry(
                geometry,
                config.line_width,
                _resin_path_spacing(config.line_width, config.infill_overlap),
                config.tolerance,
            )
        )
    elif config.infill_pattern == "zigzag":
        angle = 45.0 if layer_index % 2 == 0 else -45.0
        filled.extend(
            _zigzag_infill_geometry(
                geometry,
                line_spacing,
                angle,
                config.tolerance,
            )
        )
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
                config.tolerance,
            )
        )
    elif config.infill_pattern == "gyroid":
        filled.extend(_gyroid_infill_geometry(geometry, line_spacing, config.tolerance))

    if config.infill_pattern not in ("triangles", "gyroid"):
        smoothing_radius = config.line_width * config.smoothing_radius_factor
        smoothing_cut_fraction = 0.35
        if config.infill_pattern in ("rectilinear", "zigzag"):
            # Keep the bend as a small line-width-sized fillet. Larger radii
            # turn boundary turns into semicircles and force excessive splits.
            smoothing_radius = config.line_width * 0.2
            smoothing_cut_fraction = 0.3
        filled = _smooth_resin_infill_paths(
            filled,
            geometry,
            smoothing_radius,
            config.smoothing_angle,
            config.tolerance,
            cut_fraction=smoothing_cut_fraction,
        )
    return filled


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
    return paths


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


def _libslic3r_offset_geometry(geometry, distance: float, tolerance: float):
    """Offset geometry using libslic3r-like miter joins and cleanup semantics."""

    if geometry.is_empty or abs(distance) <= tolerance:
        return geometry
    offset = geometry.buffer(distance, join_style="mitre", mitre_limit=3.0)
    if offset.is_empty:
        return offset
    if not offset.is_valid:
        offset = offset.buffer(0)
    return offset


def _resin_infill_surface_geometry(geometry, config: SliceConfig):
    if geometry.is_empty:
        return geometry

    path_spacing = _resin_path_spacing(config.line_width, config.infill_overlap)
    last_perimeter_centerline = config.line_width * 0.5 + (
        config.perimeter_count - 1
    ) * path_spacing
    infill_surface = _libslic3r_offset_geometry(
        geometry,
        -last_perimeter_centerline,
        config.tolerance,
    )
    if infill_surface.is_empty:
        return infill_surface

    overlap_offset = _libslic3r_fill_surface_overlap_offset(
        path_spacing,
        config.line_width,
        config.infill_overlap,
    )
    return _libslic3r_offset_geometry(infill_surface, overlap_offset, config.tolerance)


def _libslic3r_fill_surface_overlap_offset(
    line_spacing: float,
    line_width: float,
    overlap_percent: float,
) -> float:
    return _resin_overlap_width(line_width, overlap_percent) - 0.5 * line_spacing


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
        offset_geometry = _libslic3r_offset_geometry(geometry, -offset_distance, tolerance)
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
    return paths


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
) -> list[np.ndarray]:
    if spacing <= 0:
        raise ValueError("infill spacing must be positive")
    if geometry.is_empty:
        return []

    rotated = affinity.rotate(geometry, -angle_degrees, origin=(0, 0), use_radians=False)
    min_x, min_y, max_x, max_y = rotated.bounds
    padding = spacing * 2.0
    scan_y = math.ceil((min_y + tolerance) / spacing) * spacing
    paths: list[np.ndarray] = []

    while scan_y < max_y - tolerance:
        line = LineString([(min_x - padding, scan_y), (max_x + padding, scan_y)])
        for segment in _extract_line_segments(rotated.intersection(line), tolerance):
            coords = list(segment.coords)
            if len(coords) < 2:
                continue
            restored = affinity.rotate(
                LineString(coords), angle_degrees, origin=(0, 0), use_radians=False
            )
            path = np.asarray([[float(x), float(y)] for x, y in restored.coords], dtype=np.float32)
            path = _dedupe_consecutive(path, tolerance)
            if path.shape[0] >= 2 and np.linalg.norm(path[-1] - path[0]) > tolerance:
                paths.append(path)

        scan_y += spacing
    return paths


def _triangular_lattice_infill_geometry(
    geometry,
    spacing: float,
    minimum_feature_length: float,
    tolerance: float,
) -> list[np.ndarray]:
    if spacing <= 0:
        raise ValueError("infill spacing must be positive")
    if geometry.is_empty:
        return []

    edges, coordinates = _triangular_lattice_edges_at_phase(
        geometry,
        spacing,
        0.0,
        minimum_feature_length,
        tolerance,
    )
    if not edges:
        return []
    return [
        np.asarray([coordinates[start], coordinates[end]], dtype=np.float32)
        for start, end in edges
    ]


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
        config.perimeter_count * _resin_path_spacing(config.line_width, config.infill_overlap),
    )


def _resin_path_spacing(line_width: float, overlap_percent: float) -> float:
    return line_width - _resin_overlap_width(line_width, overlap_percent)


def _resin_overlap_width(line_width: float, overlap_percent: float) -> float:
    return line_width * overlap_percent / 100.0


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
