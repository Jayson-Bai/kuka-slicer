from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .external_npz import ExternalSourceJob, MaterialPaths
from .stl_io import Mesh
from .slicer import (
    DEFAULT_MATERIAL_PROCESS,
    PART_BOTTOM_ZIGZAG_ANGLE_DEGREES,
    PART_TOP_ZIGZAG_ANGLE_DEGREES,
    SliceConfig,
    _build_z_projector,
    _connect_zigzag_infill_paths,
    _isotropic_part_layer_angle,
    _isotropic_schedule_metadata,
    _layer_z_values,
    _libslic3r_fill_surface_overlap_offset,
    _path_2d_to_3d,
    _resin_infill_surface_geometry,
    _resin_path_spacing,
    _solid_geometry_from_contours,
    _zigzag_infill_geometry,
    orient_mesh_for_build_axis,
)

PYSLM_NATIVE_PATTERNS = {
    "none",
    "line",
    "aligned_rectilinear",
    "rectilinear",
    "zigzag",
}
PROJECT_ZIGZAG_PATTERNS = {"zigzag", "isotropic"}
SUPPORTED_PYSLM_PATTERNS = PYSLM_NATIVE_PATTERNS | {"isotropic"}


def slice_mesh_to_job_with_pyslm(mesh: Mesh, config: SliceConfig) -> ExternalSourceJob:
    """Slice with PySLM while preserving the project's ExternalSourceJob contract."""

    _validate_pyslm_config(config)
    pyslm_core, pyslm_hatching, pyslm_geometry, trimesh = _load_pyslm_modules()

    oriented_mesh = orient_mesh_for_build_axis(mesh, config.build_axis)
    part = _part_from_mesh(pyslm_core, trimesh, oriented_mesh)
    z_values = _layer_z_values(oriented_mesh, config)
    z_projector = _build_z_projector(config)

    material_paths: list[MaterialPaths] = []
    path_roles: dict[str, dict[str, list[str]]] = {"R": {}}

    for layer_index, base_z in enumerate(z_values):
        layer_config = _effective_layer_config(config, layer_index, len(z_values))
        slice_z = float(base_z)
        if abs(slice_z - oriented_mesh.z_max) <= max(config.tolerance * 10.0, 1e-7):
            top_sample_offset = max(
                config.tolerance * 2.0,
                config.layer_height * 1e-4,
            )
            slice_z = max(oriented_mesh.z_min, slice_z - top_sample_offset)
        boundary_paths = part.getVectorSlice(
            slice_z,
            returnCoordPaths=True,
            fixPolygons=config.pyslm.fix_polygons,
            simplificationFactor=config.pyslm.simplification_factor,
            simplificationPreserveTopology=config.pyslm.simplification_preserve_topology,
            simplificationFactorMode=config.pyslm.simplification_mode,
        )
        boundary_paths = [_closed_xy_path(path, layer_config.tolerance) for path in boundary_paths]
        boundary_paths = [path for path in boundary_paths if path.shape[0] >= 3]
        if not boundary_paths:
            continue

        layer_paths_2d, roles = _pyslm_layer_paths(
            boundary_paths,
            layer_config,
            layer_index,
            len(z_values),
            pyslm_hatching,
            pyslm_geometry,
        )
        if config.material == "R":
            path_roles["R"][str(layer_index)] = roles
        layer_paths_3d = [
            _path_2d_to_3d(path, float(base_z), z_projector)
            for path in layer_paths_2d
            if path.shape[0] >= 2
        ]
        if layer_paths_3d:
            material_paths.append(MaterialPaths(layer_index, config.material, layer_paths_3d))

    return ExternalSourceJob(
        material_paths=material_paths,
        meta={
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
                "slicing_kernel": config.slicing_kernel,
                "slicing_kernel_status": "experimental",
                "perimeter_count": config.perimeter_count,
                "smoothing_angle": config.smoothing_angle,
                "smoothing_radius_factor": config.smoothing_radius_factor,
                "part_cap_layers": (
                    {
                        "bottom": 0 if len(z_values) else None,
                        "top": len(z_values) - 1 if len(z_values) else None,
                        "infill_pattern": "zigzag",
                        "infill_density": 100.0,
                        "bottom_angle_degrees": PART_BOTTOM_ZIGZAG_ANGLE_DEGREES,
                        "top_angle_degrees": PART_TOP_ZIGZAG_ANGLE_DEGREES,
                    }
                    if config.material == "R"
                    else None
                ),
                "isotropic_schedule": (
                    _isotropic_schedule_metadata(len(z_values), config.layer_height)
                    if config.material == "R" and config.infill_pattern == "isotropic"
                    else None
                ),
                "pyslm": {
                    "hatcher": config.pyslm.hatcher,
                    "native_patterns": sorted(PYSLM_NATIVE_PATTERNS),
                    "hatch_sort": config.pyslm.hatch_sort,
                    "hatch_angle": config.pyslm.hatch_angle,
                    "layer_angle_increment": config.pyslm.layer_angle_increment,
                    "hatch_distance": config.pyslm.hatch_distance,
                    "contour_offset": config.pyslm.contour_offset,
                    "spot_compensation": config.pyslm.spot_compensation,
                    "volume_offset_hatch": config.pyslm.volume_offset_hatch,
                    "num_outer_contours": config.pyslm.num_outer_contours,
                    "num_inner_contours": config.pyslm.num_inner_contours,
                    "scan_contour_first": config.pyslm.scan_contour_first,
                    "stripe_width": config.pyslm.stripe_width,
                    "stripe_overlap": config.pyslm.stripe_overlap,
                    "stripe_offset": config.pyslm.stripe_offset,
                    "island_width": config.pyslm.island_width,
                    "island_overlap": config.pyslm.island_overlap,
                    "island_offset": config.pyslm.island_offset,
                    "fix_polygons": config.pyslm.fix_polygons,
                    "simplification_factor": config.pyslm.simplification_factor,
                    "simplification_preserve_topology": config.pyslm.simplification_preserve_topology,
                    "simplification_mode": config.pyslm.simplification_mode,
                },
            },
            "path_roles": path_roles,
            "process_defaults": {
                "resin": {
                    "layer_height_mm": DEFAULT_MATERIAL_PROCESS["R"]["layer_height_mm"],
                    "line_width_mm": DEFAULT_MATERIAL_PROCESS["R"]["line_width_mm"],
                },
                "fiber": {
                    "layer_height_mm": DEFAULT_MATERIAL_PROCESS["F"]["layer_height_mm"],
                    "line_width_mm": DEFAULT_MATERIAL_PROCESS["F"]["line_width_mm"],
                },
            },
        },
    )


def _validate_pyslm_config(config: SliceConfig) -> None:
    if config.material != "R":
        raise ValueError("PySLM slicing kernel currently supports resin material R only")
    if config.infill_pattern not in SUPPORTED_PYSLM_PATTERNS:
        raise ValueError(
            "PySLM kernel currently supports "
            f"{', '.join(sorted(SUPPORTED_PYSLM_PATTERNS))}; "
            f"got {config.infill_pattern!r}"
        )


def _load_pyslm_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import trimesh
        from pyslm import core as pyslm_core
        from pyslm import geometry as pyslm_geometry
        from pyslm import hatching as pyslm_hatching
    except ImportError as exc:
        raise ImportError(
            "PySLM slicing kernel requires the optional 'pyslm' dependency set. "
            "Install it with: python -m pip install 'kuka-slicer[pyslm]'"
        ) from exc
    return pyslm_core, pyslm_hatching, pyslm_geometry, trimesh


def _part_from_mesh(pyslm_core: Any, trimesh_module: Any, mesh: Mesh) -> Any:
    vertices = mesh.triangles.reshape(-1, 3).astype(np.float64)
    faces = np.arange(vertices.shape[0], dtype=np.int64).reshape(-1, 3)
    trimesh_mesh = trimesh_module.Trimesh(vertices=vertices, faces=faces, process=True)
    part = pyslm_core.Part("kuka_slicer_part")
    part.setGeometryByMesh(trimesh_mesh)
    return part


def _effective_layer_config(
    config: SliceConfig,
    layer_index: int,
    layer_count: int,
) -> SliceConfig:
    if config.material == "R" and config.infill_pattern == "isotropic":
        return replace(
            config,
            infill_density=(
                100.0
                if layer_index in {0, layer_count - 1}
                else config.infill_density
            ),
            pyslm=replace(
                config.pyslm,
                hatch_angle=_isotropic_part_layer_angle(layer_index, layer_count),
            ),
        )
    if config.material == "R" and layer_index == 0:
        return replace(
            config,
            infill_pattern="zigzag",
            infill_density=100.0,
            pyslm=replace(config.pyslm, hatch_angle=PART_BOTTOM_ZIGZAG_ANGLE_DEGREES),
        )
    if config.material == "R" and layer_index == layer_count - 1:
        return replace(
            config,
            infill_pattern="zigzag",
            infill_density=100.0,
            pyslm=replace(config.pyslm, hatch_angle=PART_TOP_ZIGZAG_ANGLE_DEGREES),
        )
    return config


def _pyslm_layer_paths(
    boundary_paths: list[np.ndarray],
    config: SliceConfig,
    layer_index: int,
    layer_count: int,
    pyslm_hatching: Any,
    pyslm_geometry: Any,
) -> tuple[list[np.ndarray], list[str]]:
    hatcher = _make_hatcher(pyslm_hatching, config, layer_index, layer_count)
    hatcher.hatchingEnabled = config.infill_pattern != "none" and config.infill_density > 0

    layer = hatcher.hatch([_open_path_for_pyslm(path) for path in boundary_paths])
    if layer is None:
        return [], []

    paths: list[np.ndarray] = []
    roles: list[str] = []
    for geometry in layer.geometry:
        if isinstance(geometry, pyslm_geometry.ContourGeometry):
            path = _closed_xy_path(geometry.coords, config.tolerance)
            if path.shape[0] >= 3:
                paths.append(path)
                roles.append("outer_contour" if geometry.subType == "outer" else "inner_contour")
        elif (
            isinstance(geometry, pyslm_geometry.HatchGeometry)
            and config.infill_pattern not in PROJECT_ZIGZAG_PATTERNS
        ):
            hatch_coords = np.asarray(geometry.coords, dtype=np.float32).reshape(-1, 2, 2)
            for hatch in hatch_coords:
                path = np.asarray(hatch, dtype=np.float32)
                if np.linalg.norm(path[1] - path[0]) > config.tolerance:
                    paths.append(path)
                    roles.append("infill")

    if config.infill_pattern in PROJECT_ZIGZAG_PATTERNS and config.infill_density > 0:
        solid_geometry = _solid_geometry_from_contours(boundary_paths)
        infill_geometry = _resin_infill_surface_geometry(solid_geometry, config)
        spacing = _pyslm_hatch_spacing(config)
        infill_paths = _zigzag_infill_geometry(
            infill_geometry,
            spacing,
            _pyslm_hatch_angle(config, layer_index, layer_count),
            config.tolerance,
        )
        infill_paths = _connect_zigzag_infill_paths(
            infill_paths,
            infill_geometry,
            spacing,
            config.tolerance,
        )
        paths.extend(infill_paths)
        roles.extend(["infill"] * len(infill_paths))
    return paths, roles


def _make_hatcher(pyslm_hatching: Any, config: SliceConfig, layer_index: int, layer_count: int) -> Any:
    settings = config.pyslm
    hatcher_class = {
        "basic": pyslm_hatching.Hatcher,
        "stripe": pyslm_hatching.StripeHatcher,
        "island": pyslm_hatching.IslandHatcher,
        "basic_island": pyslm_hatching.BasicIslandHatcher,
    }[settings.hatcher]
    hatcher = hatcher_class()
    hatcher.scanContourFirst = settings.scan_contour_first
    if settings.num_outer_contours is None:
        hatcher.numOuterContours = 1 if config.perimeter_count >= 1 else 0
    else:
        hatcher.numOuterContours = settings.num_outer_contours
    if settings.num_inner_contours is None:
        hatcher.numInnerContours = max(0, config.perimeter_count - hatcher.numOuterContours)
    else:
        hatcher.numInnerContours = settings.num_inner_contours
    hatcher.spotCompensation = (
        config.line_width * 0.5
        if settings.spot_compensation is None
        else settings.spot_compensation
    )
    hatcher.contourOffset = (
        _resin_path_spacing(config.line_width, config.infill_overlap)
        if settings.contour_offset is None
        else settings.contour_offset
    )
    hatcher.volumeOffsetHatch = (
        _volume_offset_between_contour_and_hatch(config)
        if settings.volume_offset_hatch is None
        else settings.volume_offset_hatch
    )
    hatcher.hatchDistance = _pyslm_hatch_spacing(config)
    hatcher.hatchAngle = _pyslm_hatch_angle(config, layer_index, layer_count)
    hatcher.layerAngleIncrement = layer_index * settings.layer_angle_increment

    if settings.hatcher == "stripe":
        hatcher.stripeWidth = settings.stripe_width
        hatcher.stripeOverlap = settings.stripe_overlap
        hatcher.stripeOffset = settings.stripe_offset
    elif settings.hatcher in ("island", "basic_island"):
        hatcher.islandWidth = settings.island_width
        hatcher.islandOverlap = settings.island_overlap
        hatcher.islandOffset = settings.island_offset

    sort_method = _pyslm_sort_method(pyslm_hatching, settings.hatch_sort, hatcher.hatchAngle)
    if sort_method is not None:
        hatcher.hatchSortMethod = sort_method
    return hatcher


def _pyslm_sort_method(pyslm_hatching: Any, sort_name: str, hatch_angle: float) -> Any:
    if sort_name == "none":
        return None
    if sort_name == "alternate":
        return pyslm_hatching.AlternateSort()
    if sort_name == "unidirectional":
        return pyslm_hatching.UnidirectionalSort()
    if sort_name == "linear":
        return pyslm_hatching.LinearSort(hatch_angle)
    if sort_name == "directional":
        return pyslm_hatching.HatchDirectionalSort()
    raise ValueError(f"unsupported PySLM hatch sort: {sort_name}")


def _volume_offset_between_contour_and_hatch(config: SliceConfig) -> float:
    path_spacing = _resin_path_spacing(config.line_width, config.infill_overlap)
    overlap_offset = _libslic3r_fill_surface_overlap_offset(
        path_spacing,
        config.line_width,
        config.infill_overlap,
    )
    return -overlap_offset


def _pyslm_hatch_spacing(config: SliceConfig) -> float:
    if config.pyslm.hatch_distance is not None:
        return config.pyslm.hatch_distance
    if config.infill_density <= 0:
        return config.line_width
    path_spacing = _resin_path_spacing(config.line_width, config.infill_overlap)
    return path_spacing / (config.infill_density / 100.0)


def _pyslm_hatch_angle(config: SliceConfig, layer_index: int, layer_count: int) -> float:
    if config.material == "R" and config.infill_pattern == "isotropic":
        return _isotropic_part_layer_angle(layer_index, layer_count)
    if config.material == "R" and config.infill_pattern == "zigzag":
        if layer_index == 0:
            return PART_BOTTOM_ZIGZAG_ANGLE_DEGREES
        if layer_index == layer_count - 1:
            return PART_TOP_ZIGZAG_ANGLE_DEGREES
    if config.pyslm.hatch_angle is None:
        angle = 45.0 if config.infill_pattern in {"rectilinear", "zigzag"} else 0.0
    else:
        angle = float(config.pyslm.hatch_angle)
    if config.infill_pattern in {"rectilinear", "zigzag"} and layer_index % 2:
        angle = -angle
    return angle


def _closed_xy_path(path: Any, tolerance: float) -> np.ndarray:
    points = np.asarray(path, dtype=np.float32)[:, :2]
    if points.shape[0] > 1 and np.linalg.norm(points[0] - points[-1]) <= tolerance:
        points = points[:-1]
    if points.shape[0] > 1:
        points = np.vstack([points, points[0]])
    return points


def _open_path_for_pyslm(path: np.ndarray) -> np.ndarray:
    points = np.asarray(path, dtype=np.float32)[:, :2]
    if points.shape[0] > 1 and np.linalg.norm(points[0] - points[-1]) <= 1e-6:
        return points[:-1]
    return points
