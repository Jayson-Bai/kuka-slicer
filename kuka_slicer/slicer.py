from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
import math
from typing import Callable, Literal

import numpy as np
from shapely import affinity, maximum_inscribed_circle, prepare
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import linemerge, nearest_points, unary_union
from shapely.strtree import STRtree

from .external_npz import ExternalSourceJob, Material, MaterialPaths
from .stl_io import Mesh

CurveMode = Literal["flat", "sinusoidal"]
InfillPattern = Literal[
    "none",
    "zigzag_horizontal",
    "zigzag_vertical",
    "zigzag_plus45",
    "zigzag_minus45",
    "rectilinear",
    "aligned_rectilinear",
    "line",
    "grid",
    "triangles",
    "gyroid",
    "concentric",
    "zigzag",
    "isotropic",
]
BuildAxis = Literal["x", "y", "z"]
SlicingKernel = Literal["legacy", "pyslm"]
PySLMHatcher = Literal["basic", "stripe", "island", "basic_island"]
PySLMHatchSort = Literal["none", "alternate", "unidirectional", "linear", "directional"]
PySLMSimplificationMode = Literal["absolute", "bound"]
RaftInfillPattern = Literal["concentric", "zigzag"]

FIXED_ZIGZAG_ANGLES = {
    "zigzag_horizontal": 0.0,
    "zigzag_vertical": 90.0,
    "zigzag_plus45": 45.0,
    "zigzag_minus45": -45.0,
}
FIXED_ZIGZAG_PATTERNS = tuple(FIXED_ZIGZAG_ANGLES)


def _fixed_zigzag_angle(pattern: str) -> float | None:
    return FIXED_ZIGZAG_ANGLES.get(pattern)

DEFAULT_RESIN_LAYER_HEIGHT_MM = 0.5
DEFAULT_RESIN_LINE_WIDTH_MM = 2.0
DEFAULT_RESIN_PLANNING_LINE_WIDTH_MM = 2.3
DEFAULT_RESIN_INFILL_DENSITY_PERCENT = 100.0
DEFAULT_RESIN_INFILL_OVERLAP_PERCENT = 10.0
DEFAULT_RESIN_CONTOUR_INFILL_OVERLAP_PERCENT = 2.0
DEFAULT_FIBER_LAYER_HEIGHT_MM = 0.1
DEFAULT_FIBER_LINE_WIDTH_MM = 1.0
DEFAULT_RESIN_PERIMETER_COUNT = 2
DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES = 150.0
DEFAULT_PRUSA_CONTINUITY_SMOOTHING_ANGLE_DEGREES = 120.0
DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR = 0.35
MAX_SOLID_FILL_RECONNECT_PASSES = 5
GYROID_WAVELENGTH_FACTOR = 2.35
DEFAULT_RAFT_LAYER_COUNT = 2
DEFAULT_RAFT_OUTWARD_OFFSETS_MM = (15.0, 10.0)
DEFAULT_RAFT_TOP_GAP_MM = 0.0
CONCENTRIC_RESIDUAL_GAP_TOLERANCE_MM = 0.5
CONCENTRIC_MINIMUM_PATH_LENGTH_MM = 0.5
RAFT_BOTTOM_ZIGZAG_ANGLE_DEGREES = 90.0
RAFT_TOP_ZIGZAG_ANGLE_DEGREES = -45.0
PART_BOTTOM_ZIGZAG_ANGLE_DEGREES = 0.0
PART_TOP_ZIGZAG_ANGLE_DEGREES = 45.0
ISOTROPIC_LAYER_HEIGHT_MM = 0.5
ISOTROPIC_REPEAT_HEIGHT_MM = 2.0
ISOTROPIC_CAP_HEIGHT_MM = 1.0
ISOTROPIC_REPEAT_ANGLES_DEGREES = (45.0, 0.0, -45.0, 90.0)
STRICT_MEASURED_PATTERN_ANGLE_SCHEDULES = {
    "grid": (0.0, 90.0),
    "triangles": (0.0, 60.0, 120.0),
    "gyroid": (45.0, -45.0),
}
MINIMUM_RESIDUAL_CORRECTION_NOVEL_AREA_FRACTION = 0.4
FULL_DENSITY_TARGET_UNCOVERED_DIAMETER_FACTOR = 0.65
MAX_FULL_DENSITY_COVERAGE_REPAIR_PASSES = 8
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


@dataclass(frozen=True)
class PySLMStrategyDefaults:
    """Recommended stripe/island dimensions for the current resin scale."""

    width: float
    overlap: float
    offset: float = 0.5


def recommended_pyslm_strategy_defaults(
    layer_height: float,
    line_width: float,
) -> PySLMStrategyDefaults:
    """Return scale-aware defaults for PySLM stripe and island strategies.

    Stripe/island dimensions are XY scan settings, so line width is the primary
    scale. Layer height provides a lower bound for very small line widths. The
    offset is PySLM's dimensionless half-hatch-spacing default, not a distance.
    """

    if (
        not math.isfinite(layer_height)
        or not math.isfinite(line_width)
        or layer_height <= 0
        or line_width <= 0
    ):
        raise ValueError("layer_height and line_width must be positive")
    return PySLMStrategyDefaults(
        width=max(line_width * 5.0, layer_height * 10.0),
        overlap=min(0.1, line_width * 0.05, layer_height * 0.2),
    )


@dataclass(frozen=True)
class PySLMConfig:
    """Native PySLM controls kept separate from the legacy slicer options."""

    hatcher: PySLMHatcher = "basic"
    hatch_angle: float | None = None
    layer_angle_increment: float = 0.0
    hatch_distance: float | None = None
    contour_offset: float | None = None
    spot_compensation: float | None = None
    volume_offset_hatch: float | None = None
    num_outer_contours: int | None = None
    num_inner_contours: int | None = None
    scan_contour_first: bool = True
    hatch_sort: PySLMHatchSort = "none"
    stripe_width: float = 5.0
    stripe_overlap: float = 0.1
    stripe_offset: float = 0.5
    island_width: float = 5.0
    island_overlap: float = 0.1
    island_offset: float = 0.5
    fix_polygons: bool = True
    simplification_factor: float | None = None
    simplification_preserve_topology: bool = True
    simplification_mode: PySLMSimplificationMode = "absolute"

    def __post_init__(self) -> None:
        if self.hatcher not in ("basic", "stripe", "island", "basic_island"):
            raise ValueError("pyslm hatcher must be basic, stripe, island, or basic_island")
        if self.hatch_sort not in (
            "none",
            "alternate",
            "unidirectional",
            "linear",
            "directional",
        ):
            raise ValueError("unsupported pyslm hatch_sort")
        if self.simplification_mode not in ("absolute", "bound"):
            raise ValueError("pyslm simplification_mode must be absolute or bound")
        if self.hatch_angle is not None and (
            not math.isfinite(self.hatch_angle) or self.hatch_angle < -180 or self.hatch_angle > 180
        ):
            raise ValueError("pyslm hatch_angle must be in the range [-180, 180]")
        if not math.isfinite(self.layer_angle_increment):
            raise ValueError("pyslm layer_angle_increment must be finite")
        if self.hatch_distance is not None and (
            not math.isfinite(self.hatch_distance) or self.hatch_distance <= 0
        ):
            raise ValueError("pyslm hatch_distance must be positive")
        for name, value in (
            ("contour_offset", self.contour_offset),
            ("spot_compensation", self.spot_compensation),
            ("simplification_factor", self.simplification_factor),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"pyslm {name} must be non-negative")
        if self.volume_offset_hatch is not None and not math.isfinite(self.volume_offset_hatch):
            raise ValueError("pyslm volume_offset_hatch must be finite")
        for name, value in (
            ("num_outer_contours", self.num_outer_contours),
            ("num_inner_contours", self.num_inner_contours),
        ):
            if value is not None and value < 0:
                raise ValueError(f"pyslm {name} must be non-negative")
        for name, value in (
            ("stripe_width", self.stripe_width),
            ("island_width", self.island_width),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"pyslm {name} must be positive")
        for name, value in (
            ("stripe_overlap", self.stripe_overlap),
            ("stripe_offset", self.stripe_offset),
            ("island_overlap", self.island_overlap),
            ("island_offset", self.island_offset),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"pyslm {name} must be non-negative")


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
    slicing_kernel: SlicingKernel = "legacy"
    pyslm: PySLMConfig = field(default_factory=PySLMConfig)
    perimeter_count: int = DEFAULT_RESIN_PERIMETER_COUNT
    print_perimeters: bool = True
    triangle_path_optimization: bool = True
    zigzag_path_optimization: bool = True
    planning_line_width: float | None = None
    contour_infill_overlap: float = DEFAULT_RESIN_CONTOUR_INFILL_OVERLAP_PERCENT
    first_layer_height: float | None = None
    # Deprecated compatibility inputs. They are intentionally ignored; final
    # toolpaths are no longer rounded or split by a corner-angle constraint.
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
        if self.first_layer_height is None:
            object.__setattr__(self, "first_layer_height", self.layer_height)
        if not math.isfinite(self.first_layer_height) or self.first_layer_height <= 0:
            raise ValueError("first_layer_height must be positive")
        if not math.isfinite(self.line_width) or self.line_width <= 0:
            raise ValueError("line_width must be positive")
        if self.planning_line_width is not None and (
            not math.isfinite(self.planning_line_width)
            or self.planning_line_width <= 0
        ):
            raise ValueError("planning_line_width must be positive")
        if self.curve_period <= 0:
            raise ValueError("curve_period must be positive")
        if self.infill_pattern not in (
            "none",
            "zigzag_horizontal",
            "zigzag_vertical",
            "zigzag_plus45",
            "zigzag_minus45",
            "rectilinear",
            "aligned_rectilinear",
            "line",
            "grid",
            "triangles",
            "gyroid",
            "concentric",
            "zigzag",
            "isotropic",
        ):
            raise ValueError("unsupported infill_pattern")
        if self.infill_density < 0 or self.infill_density > 100:
            raise ValueError("infill_density must be in the range [0, 100]")
        if self.infill_overlap < 0 or self.infill_overlap >= 100:
            raise ValueError("infill_overlap must be in the range [0, 100)")
        if (
            not math.isfinite(self.contour_infill_overlap)
            or self.contour_infill_overlap < 0
            or self.contour_infill_overlap >= 100
        ):
            raise ValueError(
                "contour_infill_overlap must be in the range [0, 100)"
            )
        if self.build_axis not in ("x", "y", "z"):
            raise ValueError("build_axis must be x, y, or z")
        if self.slicing_kernel not in ("legacy", "pyslm"):
            raise ValueError("slicing_kernel must be legacy or pyslm")
        if self.perimeter_count < 1:
            raise ValueError("perimeter_count must be at least 1")


def _strict_measured_pattern_angle(
    config: SliceConfig,
    layer_index: int,
) -> float | None:
    """Return the safe single-axis angle for a crossing pattern in strict mode.

    Grid, triangle and gyroid centerlines can cross or approach each other below
    the requested pitch within one layer.  When a measured planning footprint is
    active, retain the requested layer-to-layer directional intent but execute a
    single zigzag direction per layer so the existing spacing contract remains
    physically meaningful.
    """

    schedule = STRICT_MEASURED_PATTERN_ANGLE_SCHEDULES.get(config.infill_pattern)
    if (
        schedule is None
        or config.material != "R"
        or config.slicing_kernel != "legacy"
        or config.planning_line_width is None
        or config.infill_density <= 0
    ):
        return None
    return float(schedule[layer_index % len(schedule)])


def _strict_measured_pattern_execution(config: SliceConfig) -> dict[str, object] | None:
    """Describe the explicit safety fallback stored in exported metadata."""

    schedule = STRICT_MEASURED_PATTERN_ANGLE_SCHEDULES.get(config.infill_pattern)
    if _strict_measured_pattern_angle(config, 0) is None or schedule is None:
        return None
    return {
        "applied": True,
        "requested_pattern": config.infill_pattern,
        "effective_pattern": "zigzag",
        "strategy": "single_axis_per_layer",
        "angle_schedule_degrees": list(schedule),
        "same_layer_crossings_disabled": True,
        "reason": "measured_width_maximum_overlap_contract",
    }


@dataclass(frozen=True)
class RaftLayerConfig:
    outward_offset: float = 5.0
    layer_height: float = DEFAULT_RESIN_LAYER_HEIGHT_MM
    infill_density: float = DEFAULT_RESIN_INFILL_DENSITY_PERCENT
    infill_pattern: RaftInfillPattern | None = None

    def __post_init__(self) -> None:
        if self.outward_offset < 0:
            raise ValueError("raft outward offset must be non-negative")
        if self.layer_height <= 0:
            raise ValueError("raft layer height must be positive")
        if self.infill_density <= 0 or self.infill_density > 100:
            raise ValueError("raft infill density must be in the range (0, 100]")
        if self.infill_pattern not in (None, "concentric", "zigzag"):
            raise ValueError("raft infill pattern must be concentric or zigzag")


def slice_mesh_to_job(mesh: Mesh, config: SliceConfig) -> ExternalSourceJob:
    if config.infill_pattern == "isotropic":
        _validate_isotropic_infill_schedule(mesh, config)
    if config.slicing_kernel == "pyslm":
        from .pyslm_backend import slice_mesh_to_job_with_pyslm

        # PySLM owns its hatch-distance semantics.  Never let the Prusa-only
        # measured footprint partially alter PySLM's infill inset while its
        # hatch spacing still uses the nominal width.
        backend_config = replace(config, planning_line_width=None)
        job = slice_mesh_to_job_with_pyslm(mesh, backend_config)
        _record_line_width_contract(job, backend_config, planning_applied=False)
        return job
    job = _slice_mesh_to_job_legacy(mesh, config)
    _record_line_width_contract(job, config, planning_applied=True)
    return job


def _resin_planning_line_width(config: SliceConfig) -> float:
    """Return the measured bead footprint used only for resin XY planning."""

    if config.material != "R" or config.planning_line_width is None:
        return float(config.line_width)
    return float(config.planning_line_width)


def _resin_maximum_overlap_spacing(config: SliceConfig) -> float:
    """Return the centreline pitch that exactly matches the configured overlap."""

    return _resin_path_spacing(
        _resin_planning_line_width(config),
        config.infill_overlap,
    )


def _resin_contour_infill_maximum_overlap_spacing(config: SliceConfig) -> float:
    """Return the configured contour-to-infill seam spacing."""

    return _resin_path_spacing(
        _resin_planning_line_width(config),
        config.contour_infill_overlap,
    )


def _resin_contour_infill_spacing(config: SliceConfig) -> float:
    """Return the conservative planning spacing for the contour seam.

    This independently configured seam overlap does not read ``infill_overlap``;
    that value continues to control only spacing between infill runs.
    """

    semantic_spacing = _resin_contour_infill_maximum_overlap_spacing(config)
    # Offset clipping followed by float32 path serialization can consume more
    # clearance than a scan-to-scan comparison.  Reserve two standard strict
    # planning margins so the emitted seam never exceeds its configured cap.
    numerical_margin = max(
        config.tolerance * 32.0,
        _resin_planning_line_width(config) * 2e-5,
        2e-6,
    )
    return semantic_spacing + numerical_margin


def _resin_planning_spacing_safety_margin(config: SliceConfig) -> float:
    """Keep numerical geometry operations on the conservative side of the pitch.

    Several GEOS containment checks intentionally expand a safe corridor by up to
    ten geometry tolerances.  A strict measured-width plan starts slightly farther
    away so those numerical allowances can never turn into extra physical overlap.
    The margin affects XY centreline placement only; it is not an extrusion value.
    """

    if config.material != "R" or config.planning_line_width is None:
        return 0.0
    planning_width = _resin_planning_line_width(config)
    return max(config.tolerance * 16.0, planning_width * 1e-5, 1e-7)


def _resin_planning_path_spacing(config: SliceConfig) -> float:
    """Return the conservative pitch used to generate measured-width infill."""

    return _resin_maximum_overlap_spacing(
        config
    ) + _resin_planning_spacing_safety_margin(config)


def _record_line_width_contract(
    job: ExternalSourceJob,
    config: SliceConfig,
    *,
    planning_applied: bool,
) -> None:
    """Keep nominal process width separate from the path-planning footprint."""

    slicing = job.meta.get("slicing")
    if not isinstance(slicing, dict):
        return
    slicing["line_width"] = float(config.line_width)
    slicing["planning_line_width"] = _resin_planning_line_width(config)
    slicing["planning_line_width_applied"] = bool(
        planning_applied and config.material == "R"
    )
    slicing["planning_line_width_changes_extrusion"] = False
    slicing["maximum_overlap_spacing"] = _resin_maximum_overlap_spacing(config)
    slicing["planning_spacing_safety_margin"] = (
        _resin_planning_spacing_safety_margin(config)
    )
    slicing["planning_path_spacing"] = _resin_planning_path_spacing(config)
    slicing["contour_infill_overlap_percent"] = (
        config.contour_infill_overlap
    )
    slicing["contour_infill_spacing"] = _resin_contour_infill_spacing(config)
    slicing["contour_infill_maximum_overlap_spacing"] = (
        _resin_contour_infill_maximum_overlap_spacing(config)
    )
    slicing["contour_infill_overlap_uses_ui"] = True
    slicing["maximum_overlap_enforced"] = bool(
        planning_applied
        and config.material == "R"
        and config.planning_line_width is not None
        and config.infill_pattern != "concentric"
    )
    slicing["maximum_overlap_scope"] = (
        "infill_nonlocal_runs_endcaps_and_closed_ring_footprints_with_independent_contour_infill_seam"
        if slicing["maximum_overlap_enforced"]
        else None
    )
    slicing["perimeter_spacing_uses_nominal_line_width"] = bool(
        config.slicing_kernel == "legacy"
    )
    slicing["concentric_fast_fill"] = bool(config.infill_pattern == "concentric")
    slicing["concentric_residual_gap_tolerance"] = (
        CONCENTRIC_RESIDUAL_GAP_TOLERANCE_MM
        if config.infill_pattern == "concentric"
        else None
    )
    slicing["concentric_minimum_path_length"] = (
        CONCENTRIC_MINIMUM_PATH_LENGTH_MM
        if config.infill_pattern == "concentric"
        else None
    )


def _slice_mesh_to_job_legacy(mesh: Mesh, config: SliceConfig) -> ExternalSourceJob:
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
    cached_constant_resin_layers: dict[
        tuple[SliceConfig, tuple[str, float | int] | None],
        tuple[list[np.ndarray], list[str]],
    ] = {}

    for layer_index, base_z in enumerate(z_values):
        is_part_cap_layer = config.material == "R" and layer_index in {
            0,
            len(z_values) - 1,
        }
        forced_cap_angle = None
        fixed_zigzag_angle = _fixed_zigzag_angle(config.infill_pattern)
        if fixed_zigzag_angle is not None:
            forced_cap_angle = fixed_zigzag_angle
        elif config.material == "R" and layer_index == 0:
            forced_cap_angle = PART_BOTTOM_ZIGZAG_ANGLE_DEGREES
        elif config.material == "R" and layer_index == len(z_values) - 1:
            forced_cap_angle = PART_TOP_ZIGZAG_ANGLE_DEGREES
        if is_part_cap_layer:
            layer_config = (
                replace(config, infill_density=100.0)
                if fixed_zigzag_angle is not None
                else replace(config, infill_pattern="zigzag", infill_density=100.0)
            )
        elif config.infill_pattern == "isotropic":
            forced_cap_angle = _isotropic_part_layer_angle(layer_index, len(z_values))
            layer_config = replace(config, infill_pattern="zigzag")
        elif (
            strict_pattern_angle := _strict_measured_pattern_angle(config, layer_index)
        ) is not None:
            # Preserve the selected pattern's layer-to-layer directions while
            # removing unsafe same-layer crossings in measured-width mode.
            forced_cap_angle = strict_pattern_angle
            layer_config = replace(config, infill_pattern="zigzag")
        else:
            layer_config = config
        if constant_section_paths is None:
            segments = _intersect_mesh_at_z(mesh.triangles, float(base_z), config.tolerance)
            paths_2d = _stitch_segments(segments, config.tolerance)
        else:
            paths_2d = [path.copy() for path in constant_section_paths]
        if config.material == "R":
            cache_key = None
            if constant_section_paths is not None:
                if forced_cap_angle is not None:
                    phase_key: tuple[str, float | int] | None = (
                        "forced",
                        float(forced_cap_angle),
                    )
                elif _fixed_zigzag_angle(layer_config.infill_pattern) is not None:
                    phase_key = ("fixed", float(_fixed_zigzag_angle(layer_config.infill_pattern)))
                elif layer_config.infill_pattern in ("rectilinear", "zigzag"):
                    phase_key = ("parity", layer_index % 2)
                else:
                    phase_key = None
                cache_key = (layer_config, phase_key)

            cached_layer = (
                cached_constant_resin_layers.get(cache_key)
                if cache_key is not None
                else None
            )
            if cached_layer is not None:
                cached_paths, cached_roles = cached_layer
                paths_2d = [path.copy() for path in cached_paths]
                roles = list(cached_roles)
            else:
                paths_2d, roles = _build_resin_paths(
                    paths_2d,
                    layer_config,
                    layer_index,
                    forced_zigzag_angle=forced_cap_angle,
                )
                if cache_key is not None:
                    cached_constant_resin_layers[cache_key] = (
                        [path.copy() for path in paths_2d],
                        list(roles),
                    )
            path_roles["R"][str(layer_index)] = roles
        paths_3d = [_path_2d_to_3d(path, float(base_z), z_projector) for path in paths_2d]
        if paths_3d:
            material_paths.append(MaterialPaths(layer_index, config.material, paths_3d))

    meta = {
        "source": "kuka_slicer",
        "slicing": {
            "layer_height": config.layer_height,
            "first_layer_height": config.first_layer_height,
            "line_width": config.line_width,
            "z_min": float(z_values[0]) if len(z_values) else None,
            "z_max": float(z_values[-1]) if len(z_values) else None,
            "curve_mode": config.curve_mode,
            "curve_amplitude": config.curve_amplitude,
            "curve_period": config.curve_period,
            "infill_pattern": config.infill_pattern,
            "effective_infill_pattern": (
                "zigzag"
                if _strict_measured_pattern_execution(config) is not None
                else config.infill_pattern
            ),
            "infill_pattern_execution": _strict_measured_pattern_execution(config),
            "infill_density": config.infill_density,
            "infill_overlap": config.infill_overlap,
            "triangle_path_optimization": config.triangle_path_optimization,
            "triangle_path_merge_tolerance": (
                _legacy_path_merge_tolerance(config.line_width, config.tolerance)
                if config.triangle_path_optimization and config.infill_pattern == "triangles"
                else None
            ),
            "zigzag_path_optimization": config.zigzag_path_optimization,
            "zigzag_path_merge_tolerance": (
                _legacy_path_merge_tolerance(config.line_width, config.tolerance)
                if config.zigzag_path_optimization
                else None
            ),
            "build_axis": config.build_axis,
            "slicing_kernel": config.slicing_kernel,
            "perimeter_count": config.perimeter_count,
            "print_perimeters": config.print_perimeters,
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


def _validate_isotropic_infill_schedule(mesh: Mesh, config: SliceConfig) -> int:
    if config.material != "R":
        raise ValueError("各向同性填充仅支持树脂材料 R")
    tolerance = max(config.tolerance * 10.0, 1e-4)
    if not math.isclose(
        config.layer_height,
        ISOTROPIC_LAYER_HEIGHT_MM,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "各向同性填充要求层高固定为 0.5 mm；"
            f"当前层高为 {config.layer_height:g} mm，已拒绝切片"
        )

    oriented_mesh = orient_mesh_for_build_axis(mesh, config.build_axis)
    z_min = oriented_mesh.z_min if config.z_min is None else config.z_min
    z_max = oriented_mesh.z_max if config.z_max is None else config.z_max
    if z_min > z_max:
        raise ValueError("z_min must be <= z_max")
    part_height = float(z_max - z_min)
    repeat_value = (
        part_height - ISOTROPIC_CAP_HEIGHT_MM
    ) / ISOTROPIC_REPEAT_HEIGHT_MM
    repeat_count = int(round(repeat_value))
    expected_height = (
        repeat_count * ISOTROPIC_REPEAT_HEIGHT_MM + ISOTROPIC_CAP_HEIGHT_MM
    )
    if repeat_count < 1 or not math.isclose(
        part_height,
        expected_height,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "各向同性填充要求零件有效高度为 2N+1 mm（N>=1），"
            "用于底层 0.5 mm、N 组四方向 2 mm 和顶层 0.5 mm；"
            f"当前高度为 {part_height:g} mm，无法完整完成四方向循环，已拒绝切片"
        )

    layer_count = len(_layer_z_values(oriented_mesh, config))
    expected_layer_count = repeat_count * len(ISOTROPIC_REPEAT_ANGLES_DEGREES) + 2
    if layer_count != expected_layer_count:
        raise ValueError(
            "各向同性填充的有效切层数必须为 4N+2；"
            f"当前得到 {layer_count} 层，期望 {expected_layer_count} 层，已拒绝切片"
        )
    return repeat_count


def _isotropic_part_layer_angle(layer_index: int, layer_count: int) -> float:
    if layer_index == 0:
        return PART_BOTTOM_ZIGZAG_ANGLE_DEGREES
    if layer_index == layer_count - 1:
        return PART_TOP_ZIGZAG_ANGLE_DEGREES
    return ISOTROPIC_REPEAT_ANGLES_DEGREES[
        (layer_index - 1) % len(ISOTROPIC_REPEAT_ANGLES_DEGREES)
    ]


def _isotropic_schedule_metadata(
    layer_count: int,
    layer_height: float,
) -> dict[str, object]:
    return {
        "layer_height_mm": layer_height,
        "repeat_count": (layer_count - 2) // len(ISOTROPIC_REPEAT_ANGLES_DEGREES),
        "bottom_angle_degrees": PART_BOTTOM_ZIGZAG_ANGLE_DEGREES,
        "repeat_angles_degrees": list(ISOTROPIC_REPEAT_ANGLES_DEGREES),
        "top_angle_degrees": PART_TOP_ZIGZAG_ANGLE_DEGREES,
        "layer_angles_degrees": [
            _isotropic_part_layer_angle(index, layer_count)
            for index in range(layer_count)
        ],
    }


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

    def section_paths_and_geometry(z_value: float):
        segments = _intersect_mesh_at_z(mesh.triangles, z_value, tolerance)
        section_paths = _stitch_segments(segments, tolerance)
        section_geometry = _solid_geometry_from_contours(
            [path for path in section_paths if path.shape[0] >= 3]
        )
        return section_paths, section_geometry

    reference_paths, reference_geometry = section_paths_and_geometry(
        float(z_values[0])
    )
    if not reference_paths or reference_geometry.is_empty:
        return None

    # Two vertex planes do not by themselves prove a constant extrusion: a
    # frustum and an arbitrary two-profile loft have the same Z-level count.
    # Intersections are cheap compared with bead-aware path planning, so verify
    # every actual layer before enabling the direction cache.  This retains the
    # exact-model speed-up without ever copying a first-layer toolpath into a
    # geometrically different layer.
    reference_boundary = reference_geometry.boundary
    reference_signature = sorted(
        (len(polygon.interiors) for polygon in _iter_polygons(reference_geometry))
    )
    for raw_z_value in z_values[1:]:
        _, section_geometry = section_paths_and_geometry(float(raw_z_value))
        if section_geometry.is_empty:
            return None
        section_signature = sorted(
            (len(polygon.interiors) for polygon in _iter_polygons(section_geometry))
        )
        if section_signature != reference_signature:
            return None
        boundary_distance = reference_boundary.hausdorff_distance(
            section_geometry.boundary
        )
        if boundary_distance > tolerance:
            return None
        area_tolerance = max(
            tolerance * max(
                float(reference_boundary.length),
                float(section_geometry.boundary.length),
                1.0,
            ),
            tolerance * tolerance * 10.0,
        )
        if reference_geometry.symmetric_difference(section_geometry).area > area_tolerance:
            return None
    return reference_paths


def add_raft_to_job(
    job: ExternalSourceJob,
    mesh: Mesh,
    config: SliceConfig,
    raft_layers: list[RaftLayerConfig],
    top_gap: float = DEFAULT_RAFT_TOP_GAP_MM,
) -> float:
    """Insert resin raft layers before the part and shift existing paths upward."""

    if not raft_layers:
        return 0.0
    if len(raft_layers) != DEFAULT_RAFT_LAYER_COUNT:
        raise ValueError(f"raft layer count is fixed at {DEFAULT_RAFT_LAYER_COUNT}")
    top_gap = DEFAULT_RAFT_TOP_GAP_MM
    raft_layers = [
        RaftLayerConfig(
            outward_offset=layer.outward_offset,
            infill_density=config.infill_density if config.infill_density > 0 else DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
        )
        for layer in raft_layers
    ]

    oriented_mesh = orient_mesh_for_build_axis(mesh, config.build_axis)
    part_projection = _part_projection_geometry(oriented_mesh, config)
    if part_projection.is_empty:
        return 0.0
    reserved_voids = _raft_reserved_void_geometry(part_projection, config.tolerance)

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
            part_projection,
            reserved_voids,
            config,
            raft_layer,
            layer_index,
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
        "fixed_patterns": [
            {
                "layer_index": 0,
                "infill_pattern": "zigzag",
                "angle_degrees": RAFT_BOTTOM_ZIGZAG_ANGLE_DEGREES,
            },
            {
                "layer_index": 1,
                "infill_pattern": "zigzag",
                "angle_degrees": RAFT_TOP_ZIGZAG_ANGLE_DEGREES,
            },
        ],
        "layers": [
            {
                "outward_offset": layer.outward_offset,
                "layer_height": layer.layer_height,
                "infill_density": layer.infill_density,
                "infill_pattern": "zigzag",
                "angle_degrees": (
                    RAFT_BOTTOM_ZIGZAG_ANGLE_DEGREES
                    if index == 0
                    else RAFT_TOP_ZIGZAG_ANGLE_DEGREES
                ),
            }
            for index, layer in enumerate(raft_layers)
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
    return _part_projection_geometry(mesh, config)


def _part_projection_geometry(mesh: Mesh, config: SliceConfig):
    z_values = _layer_z_values(mesh, config)
    if len(z_values) == 0:
        return Polygon()
    sections = []
    for base_z in z_values:
        segments = _intersect_mesh_at_z(mesh.triangles, float(base_z), config.tolerance)
        contours = [
            path
            for path in _stitch_segments(segments, config.tolerance)
            if path.shape[0] >= 3
        ]
        section = _solid_geometry_from_contours(contours)
        if not section.is_empty:
            sections.append(section)
    if not sections:
        return Polygon()
    projection = unary_union(sections)
    if not projection.is_valid:
        projection = projection.buffer(0)
    return projection


def _raft_paths_for_layer(
    footprint,
    reserved_voids,
    config: SliceConfig,
    raft_layer: RaftLayerConfig,
    layer_index: int,
) -> tuple[list[np.ndarray], list[str]]:
    geometry = _raft_geometry_for_layer(
        footprint,
        reserved_voids,
        raft_layer.outward_offset,
        config.tolerance,
    )
    if geometry.is_empty:
        return [], []

    if not config.print_perimeters and config.planning_line_width is not None:
        # With no contour carrier, the nominal bead is also the boundary
        # carrier. The measured contour-to-infill contract would reject some
        # boundary hatches at narrow concave transitions, so use nominal
        # spacing consistently for this explicit contour-free mode.
        config = replace(config, planning_line_width=None)

    perimeter_path_spacing = _resin_path_spacing(
        config.line_width,
        config.infill_overlap,
    )
    planning_width = _resin_planning_line_width(config)
    infill_path_spacing = _resin_planning_path_spacing(config)
    contour_infill_spacing = _resin_contour_infill_spacing(config)
    perimeters, roles = _perimeter_paths_from_geometry(
        geometry,
        config.line_width,
        perimeter_path_spacing,
        config.perimeter_count,
        config.tolerance,
    )
    # When contours are disabled, the infill itself must reach the physical
    # boundary.  Do not keep the omitted perimeter rings as a planning wall:
    # that would reserve the very band the replacement infill is meant to fill.
    planning_perimeters = perimeters if config.print_perimeters else []
    infill_geometry = _resin_infill_surface_geometry(geometry, config)
    if infill_geometry.is_empty:
        return (perimeters, roles) if config.print_perimeters else ([], [])
    angle = (
        RAFT_BOTTOM_ZIGZAG_ANGLE_DEGREES
        if layer_index == 0
        else RAFT_TOP_ZIGZAG_ANGLE_DEGREES
    )
    filled = _raft_zigzag_infill_paths(
        infill_geometry,
        config,
        raft_layer.infill_density,
        angle_degrees=angle,
    )
    last_perimeter_linework = (
        None if config.print_perimeters else GeometryCollection()
    )
    measured_width_validated = False
    coverage_direct_allowed = None
    if raft_layer.infill_density >= 100.0 - 1e-9:
        used_strict_unconnected_fallback = False
        if config.print_perimeters:
            last_perimeter_linework = _last_perimeter_linework(
                geometry,
                config.line_width,
                perimeter_path_spacing,
                config.perimeter_count,
                config.tolerance,
            )
        centerline_regions = _solid_residual_centerline_regions(
            geometry,
            infill_geometry,
            planning_perimeters,
            last_perimeter_linework,
            planning_width,
            infill_path_spacing,
            config.tolerance,
            enforce_maximum_overlap=config.planning_line_width is not None,
            wall_clearance=contour_infill_spacing,
        )
        coverage_direct_allowed = (
            None if centerline_regions is None else centerline_regions[2]
        )
        if config.planning_line_width is None:
            filled = _reroute_residual_solid_bead_gaps(
                geometry,
                infill_geometry,
                planning_perimeters,
                filled,
                last_perimeter_linework,
                planning_width,
                infill_path_spacing,
                config.tolerance,
                centerline_regions=centerline_regions,
                minimum_wall_clearance=contour_infill_spacing,
            )
        try:
            filled = _finish_solid_fill_paths(
                geometry,
                infill_geometry,
                planning_perimeters,
                filled,
                last_perimeter_linework,
                config,
                centerline_regions=centerline_regions,
            )
        except ValueError as exc:
            if (
                config.planning_line_width is None
                or "cannot satisfy the configured maximum overlap" not in str(exc)
            ):
                raise
            # A strict zero-overlap return can be valid against the printable
            # boundary while approaching a non-local hatch by less than one
            # measured bead width. Keep the overlap contract and make the layer
            # printable by removing only those continuity returns, then rerun
            # the same measured-width finisher and postcondition.
            filled = _strict_unconnected_solid_zigzag_fallback(
                infill_geometry,
                config,
                angle,
            )
            used_strict_unconnected_fallback = True
            filled = _finish_solid_fill_paths(
                geometry,
                infill_geometry,
                planning_perimeters,
                filled,
                last_perimeter_linework,
                config,
                centerline_regions=centerline_regions,
            )
        if (
            config.planning_line_width is not None
            and not used_strict_unconnected_fallback
        ):
            reconnect_direct_allowed = (
                centerline_regions[2]
                if centerline_regions is not None
                else _solid_reconnect_centerline_region(
                    geometry,
                    planning_perimeters,
                    last_perimeter_linework,
                    planning_width,
                    contour_infill_spacing,
                    config.tolerance,
                )
            )
            for _ in range(MAX_SOLID_FILL_RECONNECT_PASSES):
                reconnected = _reconnect_finished_solid_fill_paths(
                    planning_perimeters,
                    filled,
                    last_perimeter_linework,
                    config,
                    direct_allowed=reconnect_direct_allowed,
                )
                if len(reconnected) == len(filled):
                    break
                filled = reconnected
        if config.planning_line_width is not None:
            pre_coverage_path_count = len(filled)
            filled = _optimize_full_density_coverage(
                geometry,
                infill_geometry,
                planning_perimeters,
                filled,
                planning_width,
                config.tolerance,
                direct_allowed=coverage_direct_allowed,
            )
            filled = _maximize_full_density_continuity(
                geometry,
                infill_geometry,
                planning_perimeters,
                filled,
                config,
                last_perimeter_linework=last_perimeter_linework,
                allow_overlap_relaxation=(
                    len(filled) > pre_coverage_path_count
                ),
                enable_detour_absorption=False,
            )
        measured_width_validated = config.planning_line_width is not None
    if (
        config.planning_line_width is not None
        and filled
        and not measured_width_validated
        and config.infill_pattern != "concentric"
    ):
        if last_perimeter_linework is None:
            last_perimeter_linework = _last_perimeter_linework(
                geometry,
                config.line_width,
                perimeter_path_spacing,
                config.perimeter_count,
                config.tolerance,
            )
        filled = _require_measured_width_infill(
            filled,
            last_perimeter_linework,
            config,
        )
    if config.print_perimeters:
        return perimeters + filled, roles + ["infill"] * len(filled)
    return filled, ["infill"] * len(filled)


def _boundary_paths_from_geometry(
    geometry,
    path_spacing: float,
    perimeter_count: int,
    tolerance: float,
) -> tuple[list[np.ndarray], list[str]]:
    paths: list[np.ndarray] = []
    roles: list[str] = []
    for perimeter_index in range(perimeter_count):
        offset_geometry = (
            geometry
            if perimeter_index == 0
            else _libslic3r_offset_geometry(
                geometry,
                -perimeter_index * path_spacing,
                tolerance,
            )
        )
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


def _raft_geometry_for_layer(
    footprint,
    reserved_voids,
    outward_offset: float,
    tolerance: float,
):
    geometry = footprint.buffer(outward_offset, join_style="round")
    if reserved_voids.is_empty:
        return geometry
    exterior_extensions = _raft_exterior_void_extensions(
        footprint,
        outward_offset,
        tolerance,
    )
    void_geometry = unary_union([reserved_voids, exterior_extensions])
    geometry = geometry.difference(void_geometry)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def _raft_exterior_void_extensions(
    footprint,
    outward_offset: float,
    tolerance: float,
):
    """Carry projection openings through the outward raft band without widening them."""

    if footprint.is_empty or outward_offset <= tolerance:
        return Polygon()

    outer_silhouette = footprint.convex_hull
    silhouette_center = np.asarray(
        [outer_silhouette.centroid.x, outer_silhouette.centroid.y],
        dtype=np.float64,
    )
    extension_distance = outward_offset + max(tolerance * 10.0, 1e-5)
    extension_pieces: list[Polygon] = []
    exterior_voids = _nondegenerate_polygons(
        outer_silhouette.difference(footprint),
        tolerance,
    )

    for exterior_void in exterior_voids:
        mouth_geometry = exterior_void.boundary.intersection(outer_silhouette.boundary)
        for mouth_line in _extract_line_segments(mouth_geometry, tolerance):
            coordinates = np.asarray(mouth_line.coords, dtype=np.float64)
            extended_segments: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for start, end in zip(coordinates[:-1], coordinates[1:]):
                direction = end - start
                length = float(np.linalg.norm(direction))
                if length <= tolerance:
                    continue

                normal = np.asarray([-direction[1], direction[0]], dtype=np.float64) / length
                midpoint = (start + end) * 0.5
                if float(np.dot(normal, midpoint - silhouette_center)) < 0.0:
                    normal = -normal
                extended_start = start + normal * extension_distance
                extended_end = end + normal * extension_distance
                extension_pieces.append(
                    Polygon([start, end, extended_end, extended_start])
                )
                extended_segments.append((start, extended_start, extended_end, end))

            for previous, current in zip(extended_segments[:-1], extended_segments[1:]):
                if np.linalg.norm(previous[2] - current[1]) <= tolerance:
                    continue
                extension_pieces.append(
                    Polygon([previous[3], previous[2], current[1]])
                )

    if not extension_pieces:
        return Polygon()
    extensions = unary_union(extension_pieces)
    if not extensions.is_valid:
        extensions = extensions.buffer(0)
    return extensions


def _raft_reserved_void_geometry(part_projection, tolerance: float):
    if part_projection.is_empty:
        return Polygon()

    outer_silhouette = part_projection.convex_hull
    voids = _nondegenerate_polygons(
        outer_silhouette.difference(part_projection),
        tolerance,
    )
    holes = _interior_holes_geometry(part_projection, tolerance)
    if not holes.is_empty:
        voids.extend(_nondegenerate_polygons(holes, tolerance))

    combined = unary_union([void for void in voids if not void.is_empty])
    if combined.is_empty:
        return Polygon()
    if not combined.is_valid:
        combined = combined.buffer(0)
    return combined


def _nondegenerate_polygons(geometry, tolerance: float) -> list[Polygon]:
    minimum_area = max(tolerance * tolerance, 0.01)
    return [
        polygon
        for polygon in _iter_polygons(geometry)
        if polygon.area > minimum_area
    ]


def _interior_holes_geometry(geometry, tolerance: float):
    holes: list[Polygon] = []
    for polygon in _iter_polygons(geometry):
        for interior in polygon.interiors:
            hole = Polygon(interior.coords)
            if hole.area > max(tolerance * tolerance, 0.01):
                holes.append(hole)
    if not holes:
        return Polygon()
    return unary_union(holes)


def _raft_lattice_infill_paths(
    geometry,
    config: SliceConfig,
    infill_density: float,
) -> list[np.ndarray]:
    if geometry.is_empty:
        return []

    planning_width = _resin_planning_line_width(config)
    path_spacing = _resin_planning_path_spacing(config)
    spacing = path_spacing / (infill_density / 100.0)
    filled = _gyroid_infill_geometry(geometry, spacing, config.tolerance)
    return _connect_boundary_infill_paths(
        filled,
        geometry,
        spacing,
        path_spacing,
        config.tolerance,
        adjacent_scanlines_only=False,
    )


def _raft_zigzag_infill_paths(
    geometry,
    config: SliceConfig,
    infill_density: float,
    angle_degrees: float = 0.0,
) -> list[np.ndarray]:
    if geometry.is_empty:
        return []

    planning_width = _resin_planning_line_width(config)
    path_spacing = _resin_planning_path_spacing(config)
    spacing = path_spacing / (infill_density / 100.0)
    if infill_density >= 100.0 - 1e-9:
        filled = _solid_zigzag_infill_paths(
            geometry,
            spacing,
            planning_width,
            angle_degrees,
            path_spacing,
            config.tolerance,
            smoothing_corner_cut=(
                config.line_width * 0.04
                if config.planning_line_width is not None
                else config.line_width * 0.15
            ),
            enforce_maximum_overlap=config.planning_line_width is not None,
            maximum_overlap_spacing=(
                _resin_maximum_overlap_spacing(config)
                if config.planning_line_width is not None
                else None
            ),
            connect_adjacent=True,
            follow_boundaries=config.print_perimeters,
        )
    else:
        filled = _zigzag_infill_geometry(
            geometry,
            spacing,
            angle_degrees,
            config.tolerance,
        )
        if config.planning_line_width is not None:
            # Clipping an oblique sparse hatch at a corner can leave a tiny
            # centreline stub.  Printing that sub-quarter-pitch fragment is a
            # resin blob, and connecting it creates a tight hairpin.  Omit it
            # before continuity planning; the missing footprint is smaller
            # than the configured bead-overlap resolution.
            minimum_fragment_length = path_spacing * 0.25
            filled = [
                path
                for path in filled
                if _open_path_length(path) >= minimum_fragment_length
            ]
        filled = _connect_zigzag_infill_paths(
            filled,
            geometry,
            spacing,
            path_spacing,
            config.tolerance,
            solid_bead_width=planning_width,
            solid_smoothing_corner_cut=0.0,
            maximum_connector_overlap_spacing=(
                _resin_maximum_overlap_spacing(config)
                if config.planning_line_width is not None
                else None
            ),
            follow_boundaries=config.print_perimeters,
        )
    return _filter_paths_covered_by_geometry(
        filled,
        geometry,
        config.tolerance,
        boundary_allowance=0.25 if not config.print_perimeters else 0.0,
    )


def _strict_unconnected_solid_zigzag_fallback(
    geometry,
    config: SliceConfig,
    angle_degrees: float,
) -> list[np.ndarray]:
    """Return strict independent hatches when material-bearing returns fail."""

    planning_width = _resin_planning_line_width(config)
    path_spacing = _resin_planning_path_spacing(config)
    paths = _solid_zigzag_infill_paths(
        geometry,
        path_spacing,
        planning_width,
        angle_degrees,
        path_spacing,
        config.tolerance,
        smoothing_corner_cut=0.0,
        enforce_maximum_overlap=True,
        maximum_overlap_spacing=_resin_maximum_overlap_spacing(config),
        connect_adjacent=False,
        follow_boundaries=config.print_perimeters,
    )
    minimum_fragment_length = path_spacing * 0.25
    paths = [
        path
        for path in paths
        if _open_path_length(path) >= minimum_fragment_length
    ]
    return _filter_paths_covered_by_geometry(
        paths,
        geometry,
        config.tolerance,
        boundary_allowance=0.25 if not config.print_perimeters else 0.0,
    )


def _filter_paths_covered_by_geometry(
    paths: list[np.ndarray],
    geometry,
    tolerance: float,
    *,
    boundary_allowance: float = 0.0,
) -> list[np.ndarray]:
    filtered: list[np.ndarray] = []
    safe_geometry = geometry.buffer(
        max(tolerance * 10.0, 1e-7) + boundary_allowance,
        join_style="round",
    )
    for path in paths:
        # Clipping/cleanup can leave a degenerate one-point stub.  GEOS
        # rejects that input when constructing a LineString; it cannot carry
        # any printable length, so discard it at the geometry boundary.
        if path.ndim != 2 or path.shape[0] < 2:
            continue
        line = LineString([(float(point[0]), float(point[1])) for point in path[:, :2]])
        if safe_geometry.covers(line):
            filtered.append(path)
    return filtered


def _open_path_length(path: np.ndarray) -> float:
    if path.shape[0] < 2:
        return 0.0
    differences = np.diff(np.asarray(path[:, :2], dtype=np.float64), axis=0)
    return float(np.linalg.norm(differences, axis=1).sum())


def _vectors_parallel(first: np.ndarray, second: np.ndarray, max_angle_degrees: float) -> bool:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0 or second_norm <= 0:
        return False
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(abs(cosine)))
    return angle <= max_angle_degrees


def _vector_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 0 or second_norm <= 0:
        return 0.0
    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(abs(cosine)))


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
    start = (
        z_min
        if config.z_min is not None and config.infill_pattern != "isotropic"
        else z_min + config.first_layer_height
    )
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
    paths: list[np.ndarray],
    config: SliceConfig,
    layer_index: int = 0,
    forced_zigzag_angle: float | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    strict_pattern_angle = _strict_measured_pattern_angle(config, layer_index)
    if strict_pattern_angle is not None:
        # Direct callers do not pass through the layer scheduler above.  Apply
        # the same explicit single-axis safety fallback here as a backstop.
        config = replace(config, infill_pattern="zigzag")
        if forced_zigzag_angle is None:
            forced_zigzag_angle = strict_pattern_angle
    if not config.print_perimeters and config.planning_line_width is not None:
        # Boundary-replacement infill has no printed contour to satisfy the
        # measured contour overlap contract against.
        config = replace(config, planning_line_width=None)
    contours = [path for path in paths if path.shape[0] >= 3]
    solid_geometry = _solid_geometry_from_contours(contours)
    if solid_geometry.is_empty:
        if not config.print_perimeters:
            return [], []
        return paths, ["outer_contour" if path.shape[0] > 2 else "infill" for path in paths]

    perimeter_path_spacing = _resin_path_spacing(
        config.line_width,
        config.infill_overlap,
    )
    planning_width = _resin_planning_line_width(config)
    infill_path_spacing = _resin_planning_path_spacing(config)
    contour_infill_spacing = _resin_contour_infill_spacing(config)
    perimeters, roles = _perimeter_paths_from_geometry(
        solid_geometry,
        config.line_width,
        perimeter_path_spacing,
        config.perimeter_count,
        config.tolerance,
    )
    # With no contour output, all boundary-clearance planning must be removed
    # together with the omitted contours; otherwise the fill stops several mm
    # short of every outer, concave, and hole boundary.
    planning_perimeters = perimeters if config.print_perimeters else []
    if config.infill_pattern == "none":
        return (perimeters, roles) if config.print_perimeters else ([], [])

    infill_geometry = _resin_infill_surface_geometry(solid_geometry, config)
    filled = _infill_paths_for_geometry(
        infill_geometry,
        config,
        layer_index,
        config.infill_density,
        forced_zigzag_angle=forced_zigzag_angle,
    )
    last_perimeter_linework = (
        None if config.print_perimeters else GeometryCollection()
    )
    measured_width_validated = False
    coverage_direct_allowed = None
    if (
        config.infill_density >= 100.0 - 1e-9
        and config.infill_pattern
        in (
            "rectilinear",
            "aligned_rectilinear",
            "line",
            "zigzag",
            *FIXED_ZIGZAG_PATTERNS,
        )
    ):
        if config.print_perimeters:
            last_perimeter_linework = _last_perimeter_linework(
                solid_geometry,
                config.line_width,
                perimeter_path_spacing,
                config.perimeter_count,
                config.tolerance,
            )
        centerline_regions = _solid_residual_centerline_regions(
            solid_geometry,
            infill_geometry,
            planning_perimeters,
            last_perimeter_linework,
            planning_width,
            infill_path_spacing,
            config.tolerance,
            enforce_maximum_overlap=config.planning_line_width is not None,
            wall_clearance=contour_infill_spacing,
        )
        coverage_direct_allowed = (
            None if centerline_regions is None else centerline_regions[2]
        )
        if config.planning_line_width is None:
            filled = _reroute_residual_solid_bead_gaps(
                solid_geometry,
                infill_geometry,
                planning_perimeters,
                filled,
                last_perimeter_linework,
                planning_width,
                infill_path_spacing,
                config.tolerance,
                centerline_regions=centerline_regions,
                minimum_wall_clearance=contour_infill_spacing,
            )
        filled = _finish_solid_fill_paths(
            solid_geometry,
            infill_geometry,
            planning_perimeters,
            filled,
            last_perimeter_linework,
            config,
            centerline_regions=centerline_regions,
        )
        if config.planning_line_width is not None:
            reconnect_direct_allowed = (
                centerline_regions[2]
                if centerline_regions is not None
                else _solid_reconnect_centerline_region(
                    solid_geometry,
                    planning_perimeters,
                    last_perimeter_linework,
                    planning_width,
                    contour_infill_spacing,
                    config.tolerance,
                )
            )
            for _ in range(MAX_SOLID_FILL_RECONNECT_PASSES):
                reconnected = _reconnect_finished_solid_fill_paths(
                    planning_perimeters,
                    filled,
                    last_perimeter_linework,
                    config,
                    direct_allowed=reconnect_direct_allowed,
                )
                if len(reconnected) == len(filled):
                    break
                filled = reconnected
        if config.planning_line_width is not None:
            pre_coverage_path_count = len(filled)
            filled = _optimize_full_density_coverage(
                solid_geometry,
                infill_geometry,
                planning_perimeters,
                filled,
                planning_width,
                config.tolerance,
                direct_allowed=coverage_direct_allowed,
            )
            filled = _maximize_full_density_continuity(
                solid_geometry,
                infill_geometry,
                planning_perimeters,
                filled,
                config,
                last_perimeter_linework=last_perimeter_linework,
                allow_overlap_relaxation=(
                    len(filled) > pre_coverage_path_count
                ),
            )
        measured_width_validated = config.planning_line_width is not None
    if config.infill_pattern == "triangles" and config.triangle_path_optimization:
        triangle_merge_tolerance = _legacy_path_merge_tolerance(
            config.line_width,
            config.tolerance,
        )
        filled = optimize_triangle_infill_travel(filled, config.tolerance)
        filled = merge_adjacent_connected_paths(filled, triangle_merge_tolerance)
    if (
        config.planning_line_width is not None
        and filled
        and not measured_width_validated
        and config.infill_pattern != "concentric"
    ):
        if last_perimeter_linework is None:
            last_perimeter_linework = _last_perimeter_linework(
                solid_geometry,
                config.line_width,
                perimeter_path_spacing,
                config.perimeter_count,
                config.tolerance,
            )
        filled = _require_measured_width_infill(
            filled,
            last_perimeter_linework,
            config,
        )
    if config.print_perimeters:
        return perimeters + filled, roles + ["infill"] * len(filled)
    return filled, ["infill"] * len(filled)


def _infill_paths_for_geometry(
    geometry,
    config: SliceConfig,
    layer_index: int,
    infill_density: float,
    forced_zigzag_angle: float | None = None,
) -> list[np.ndarray]:
    if geometry.is_empty:
        return []
    if infill_density <= 0:
        return []

    filled: list[np.ndarray] = []
    planning_width = _resin_planning_line_width(config)
    path_spacing = _resin_planning_path_spacing(config)
    density_fraction = infill_density / 100.0
    line_spacing = path_spacing / density_fraction
    grid_spacing = path_spacing * 2.0 / density_fraction
    triangle_spacing = path_spacing * 3.0 / density_fraction
    single_axis_patterns = (
        "rectilinear",
        "aligned_rectilinear",
        "line",
        "zigzag",
        *FIXED_ZIGZAG_PATTERNS,
    )
    solid_single_axis = (
        infill_density >= 100.0 - 1e-9 and config.infill_pattern in single_axis_patterns
    )

    def single_axis_paths(angle: float) -> list[np.ndarray]:
        if solid_single_axis:
            return _solid_zigzag_infill_paths(
                geometry,
                line_spacing,
                planning_width,
                angle,
                path_spacing,
                config.tolerance,
                smoothing_corner_cut=(
                    config.line_width * 0.04
                    if config.planning_line_width is not None
                    else config.line_width * 0.15
                ),
                enforce_maximum_overlap=config.planning_line_width is not None,
                maximum_overlap_spacing=(
                    _resin_maximum_overlap_spacing(config)
                    if config.planning_line_width is not None
                    else None
                ),
                connect_adjacent=True,
                follow_boundaries=config.print_perimeters,
            )
        single_axis = _zigzag_infill_geometry(
            geometry,
            line_spacing,
            angle,
            config.tolerance,
        )
        if config.planning_line_width is not None:
            minimum_fragment_length = path_spacing * 0.25
            single_axis = [
                path
                for path in single_axis
                if _open_path_length(path) >= minimum_fragment_length
            ]
        return single_axis

    if config.infill_pattern == "rectilinear":
        angle = 45.0 if layer_index % 2 == 0 else -45.0
        filled.extend(single_axis_paths(angle))
    elif config.infill_pattern == "aligned_rectilinear":
        filled.extend(single_axis_paths(0.0))
    elif config.infill_pattern == "line":
        filled.extend(single_axis_paths(90.0))
    elif config.infill_pattern == "concentric":
        # Concentric is a dedicated fast full-fill mode. Emit independent
        # direct offsets at one measured bead width, ignoring run-density and
        # run-overlap contracts. Only residual voids wider than 0.5 mm receive
        # a centered supplemental path; no ring-to-ring seam is introduced.
        filled.extend(
            _fast_concentric_infill_geometry(
                geometry,
                planning_width,
                config.tolerance,
            )
        )
    elif (fixed_zigzag_angle := _fixed_zigzag_angle(config.infill_pattern)) is not None:
        filled.extend(single_axis_paths(fixed_zigzag_angle))
    elif config.infill_pattern == "zigzag":
        angle = forced_zigzag_angle
        if angle is None:
            angle = 45.0 if layer_index % 2 == 0 else -45.0
        filled.extend(single_axis_paths(angle))
    elif config.infill_pattern == "grid":
        filled.extend(
            _multi_axis_lattice_infill_geometry(
                geometry,
                grid_spacing,
                (0.0, 90.0),
                config.tolerance,
            )
        )
    elif config.infill_pattern == "triangles":
        filled.extend(
            _triangular_lattice_infill_geometry(
                geometry,
                triangle_spacing,
                max(
                    planning_width,
                    min(planning_width * 1.6, triangle_spacing * 0.38),
                ),
                config.tolerance,
            )
        )
    elif config.infill_pattern == "gyroid":
        filled.extend(_gyroid_infill_geometry(geometry, line_spacing, config.tolerance))

    if (
        config.infill_pattern in single_axis_patterns
        and not solid_single_axis
    ):
        filled = _connect_zigzag_infill_paths(
            filled,
            geometry,
            line_spacing,
            path_spacing,
            config.tolerance,
            solid_bead_width=planning_width,
            solid_smoothing_corner_cut=0.0,
            maximum_connector_overlap_spacing=(
                _resin_maximum_overlap_spacing(config)
                if config.planning_line_width is not None
                else None
            ),
            follow_boundaries=config.print_perimeters,
        )
    elif config.infill_pattern in ("grid", "triangles", "gyroid"):
        pattern_spacing = {
            "grid": grid_spacing,
            "triangles": triangle_spacing,
            "gyroid": line_spacing,
        }[config.infill_pattern]
        filled = _connect_boundary_infill_paths(
            filled,
            geometry,
            pattern_spacing,
            path_spacing,
            config.tolerance,
            adjacent_scanlines_only=False,
        )
    zigzag_like_pattern = config.infill_pattern in ("zigzag", *FIXED_ZIGZAG_PATTERNS)
    zigzag_merge_tolerance = (
        _legacy_path_merge_tolerance(config.line_width, config.tolerance)
        if zigzag_like_pattern and config.zigzag_path_optimization
        else None
    )
    if zigzag_like_pattern and config.zigzag_path_optimization:
        filled = optimize_open_path_travel(filled, config.tolerance)
        filled = merge_adjacent_connected_paths(filled, zigzag_merge_tolerance)

    return filled

def _solid_residual_centerline_regions(
    solid_geometry,
    infill_geometry,
    perimeter_paths: list[np.ndarray],
    last_perimeter_linework,
    line_width: float,
    path_spacing: float,
    tolerance: float,
    *,
    enforce_maximum_overlap: bool = False,
    wall_clearance: float | None = None,
):
    """Build the shared physical centerline limits for gap repair and smoothing."""

    spacing_adjustment = _solid_spacing_adjustment_limit(path_spacing, line_width)
    if spacing_adjustment <= tolerance:
        return None

    bead_radius = line_width * 0.5
    physical_centerlines = solid_geometry.buffer(
        -(bead_radius - max(tolerance * 2.0, 1e-7)),
        join_style="round",
    )
    # Never buy residual coverage by moving an infill centreline closer to the
    # last perimeter than the independently configured contour-to-infill seam
    # pitch.  The infill-run overlap must not leak into this boundary contract
    # during residual repair or smoothing.
    minimum_wall_clearance = max(
        tolerance,
        (
            wall_clearance
            if wall_clearance is not None
            else (
                path_spacing
                if enforce_maximum_overlap
                else path_spacing - spacing_adjustment
            )
        ),
    )
    actual_perimeter_lines = [
        LineString(path[:, :2])
        for path in perimeter_paths
        if path.shape[0] >= 2
    ]
    wall_lines = list(actual_perimeter_lines)
    if not last_perimeter_linework.is_empty:
        wall_lines.append(last_perimeter_linework)
    wall_linework = unary_union(wall_lines) if wall_lines else GeometryCollection()
    if wall_linework.is_empty:
        direct_allowed = physical_centerlines
    else:
        wall_exclusion_distance = max(
            tolerance,
            minimum_wall_clearance - tolerance * 2.0,
        )
        # Buffering the already-unioned, heavily overlapping perimeter
        # linework makes GEOS node all of those overlaps before it can build
        # the offset.  Buffer each immutable wall first and union the areas
        # instead.  Buffer distributes over set union, so this is the same
        # exclusion geometry while avoiding several seconds of redundant
        # noding on large constant-section parts and rafts.
        wall_exclusion = unary_union(
            [
                wall_line.buffer(
                    wall_exclusion_distance,
                    join_style="round",
                )
                for wall_line in wall_lines
            ]
        )
        direct_allowed = physical_centerlines.difference(
            wall_exclusion
        )
    safe_surface_allowed = infill_geometry.buffer(
        spacing_adjustment,
        join_style="round",
    ).intersection(direct_allowed)
    return (
        spacing_adjustment,
        safe_surface_allowed,
        direct_allowed,
        minimum_wall_clearance,
        wall_linework,
    )


def _solid_reconnect_centerline_region(
    solid_geometry,
    perimeter_paths: list[np.ndarray],
    last_perimeter_linework,
    line_width: float,
    minimum_wall_clearance: float,
    tolerance: float,
):
    """Return the strict physical corridor used only for trail reconnection.

    At zero configured overlap the conservative planning pitch can be slightly
    wider than the measured bead, so residual-gap adjustment is correctly
    disabled. Reconnection still has a well-defined safe corridor, however:
    physical bead centres inside the part and outside the full wall-clearance
    exclusion. Building it separately lets the validated reconnector run
    without enabling any residual-fill or overlap relaxation.
    """

    bead_radius = line_width * 0.5
    physical_centerlines = solid_geometry.buffer(
        -(bead_radius - max(tolerance * 2.0, 1e-7)),
        join_style="round",
    )
    wall_lines = [
        LineString(path[:, :2])
        for path in perimeter_paths
        if path.shape[0] >= 2
    ]
    if not last_perimeter_linework.is_empty:
        wall_lines.append(last_perimeter_linework)
    if not wall_lines:
        return physical_centerlines
    exclusion_distance = max(
        tolerance,
        minimum_wall_clearance - tolerance * 2.0,
    )
    wall_exclusion = unary_union(
        [
            wall_line.buffer(exclusion_distance, join_style="round")
            for wall_line in wall_lines
        ]
    )
    return physical_centerlines.difference(wall_exclusion)


def _remove_smoothing_micro_segments(
    path: np.ndarray,
    safe_geometry,
    minimum_length: float,
    tolerance: float,
) -> np.ndarray:
    """Drop sub-visible boundary fragments before fitting a continuous fillet."""

    points = [
        np.asarray(point[:2], dtype=np.float32)
        for point in _dedupe_consecutive(path[:, :2], tolerance)
    ]
    changed = True
    while changed and len(points) > 2:
        changed = False
        simplified = [points[0]]
        for index in range(1, len(points) - 1):
            previous = simplified[-1]
            current = points[index]
            following = points[index + 1]
            if (
                min(
                    float(np.linalg.norm(current - previous)),
                    float(np.linalg.norm(following - current)),
                )
                < minimum_length
                and safe_geometry.covers(LineString([previous, following]))
            ):
                changed = True
                continue
            simplified.append(current)
        simplified.append(points[-1])
        points = simplified

    candidate = _dedupe_consecutive(
        np.asarray(points, dtype=np.float32),
        tolerance,
    )
    if candidate.shape[0] < 2:
        return np.asarray(path[:, :2], dtype=np.float32)
    original_line = LineString(path[:, :2])
    candidate_line = LineString(candidate)
    if (
        not safe_geometry.covers(candidate_line)
        or (original_line.is_simple and not candidate_line.is_simple)
    ):
        return np.asarray(path[:, :2], dtype=np.float32)
    return candidate


def _effective_path_corner_candidates(
    path: np.ndarray,
    minimum_span: float,
    angle_threshold_degrees: float,
    tolerance: float,
) -> list[tuple[float, int, int, int]]:
    """Find turns that remain sharp when numerical/arc micro-segments are grouped."""

    points = np.asarray(path[:, :2], dtype=np.float64)
    point_count = points.shape[0]
    if point_count < 3:
        return []

    indices = np.arange(1, point_count - 1, dtype=np.intp)
    previous_indices = indices - 1
    next_indices = indices + 1

    # Advance every still-too-short side together.  Batched norm evaluates
    # the same float64 distances as the former scalar loop, including the
    # strict ``< minimum_span`` boundary, without making three NumPy calls per
    # path vertex on every effective-corner pass.
    active_positions = np.arange(indices.size, dtype=np.intp)
    while active_positions.size:
        distances = np.linalg.norm(
            points[indices[active_positions]]
            - points[previous_indices[active_positions]],
            axis=1,
        )
        too_close = distances < minimum_span
        if not np.any(too_close):
            break
        active_positions = active_positions[too_close]
        previous_indices[active_positions] -= 1
        active_positions = active_positions[
            previous_indices[active_positions] >= 0
        ]

    active_positions = np.arange(indices.size, dtype=np.intp)
    while active_positions.size:
        distances = np.linalg.norm(
            points[next_indices[active_positions]]
            - points[indices[active_positions]],
            axis=1,
        )
        too_close = distances < minimum_span
        if not np.any(too_close):
            break
        active_positions = active_positions[too_close]
        next_indices[active_positions] += 1
        active_positions = active_positions[
            next_indices[active_positions] < point_count
        ]

    has_two_sides = (previous_indices >= 0) & (next_indices < point_count)
    indices = indices[has_two_sides]
    previous_indices = previous_indices[has_two_sides]
    next_indices = next_indices[has_two_sides]
    if not indices.size:
        return []

    incoming = points[previous_indices] - points[indices]
    outgoing = points[next_indices] - points[indices]
    incoming_lengths = np.linalg.norm(incoming, axis=1)
    outgoing_lengths = np.linalg.norm(outgoing, axis=1)
    nondegenerate = (
        np.minimum(incoming_lengths, outgoing_lengths) > tolerance
    )
    indices = indices[nondegenerate]
    previous_indices = previous_indices[nondegenerate]
    next_indices = next_indices[nondegenerate]
    incoming = incoming[nondegenerate]
    outgoing = outgoing[nondegenerate]
    incoming_lengths = incoming_lengths[nondegenerate]
    outgoing_lengths = outgoing_lengths[nondegenerate]
    if not indices.size:
        return []

    cosines = np.clip(
        np.sum(incoming * outgoing, axis=1)
        / (incoming_lengths * outgoing_lengths),
        -1.0,
        1.0,
    )
    angles = np.degrees(np.arccos(cosines))
    sharp = angles < angle_threshold_degrees - 1e-6
    candidates = [
        (float(angle), int(index), int(previous_index), int(next_index))
        for angle, index, previous_index, next_index in zip(
            angles[sharp],
            indices[sharp],
            previous_indices[sharp],
            next_indices[sharp],
        )
    ]
    candidates.sort(key=lambda item: item[0])
    return candidates


def _smooth_effective_path_corners(
    path: np.ndarray,
    safe_geometry,
    corner_cut: float,
    angle_threshold_degrees: float,
    minimum_span: float,
    tolerance: float,
) -> np.ndarray:
    """Refit tiny-radius fillets over a wider path window."""

    current = np.asarray(path[:, :2], dtype=np.float32)
    blocked_points: set[tuple[int, int]] = set()
    # There is deliberately no arbitrary iteration ceiling here.  A rejected
    # corner is permanently blocked at the effective-span resolution, while an
    # accepted replacement must strictly reduce the path-wide angle deficit
    # below.  The non-negative deficit therefore proves forward progress.
    while True:
        candidates = _effective_path_corner_candidates(
            current,
            minimum_span,
            angle_threshold_degrees,
            tolerance,
        )
        candidate = next(
            (
                item
                for item in candidates
                if (
                    round(float(current[item[1], 0]) / minimum_span),
                    round(float(current[item[1], 1]) / minimum_span),
                )
                not in blocked_points
            ),
            None,
        )
        if candidate is None:
            break
        _, index, _, _ = candidate
        point_key = (
            round(float(current[index, 0]) / minimum_span),
            round(float(current[index, 1]) / minimum_span),
        )
        accepted = False
        for window_span in (
            corner_cut,
            corner_cut * 0.75,
            corner_cut * 0.5,
            max(minimum_span * 2.0, corner_cut * 0.25),
            max(minimum_span * 3.0, corner_cut * 0.1),
            minimum_span * 2.0,
        ):
            previous_index = index - 1
            while (
                previous_index >= 0
                and float(
                    np.linalg.norm(current[index, :2] - current[previous_index, :2])
                )
                < window_span
            ):
                previous_index -= 1
            next_index = index + 1
            while (
                next_index < current.shape[0]
                and float(np.linalg.norm(current[next_index, :2] - current[index, :2]))
                < window_span
            ):
                next_index += 1
            if previous_index < 0 or next_index >= current.shape[0]:
                continue
            rounded = _rounded_corner_points(
                current[previous_index],
                current[index],
                current[next_index],
                corner_cut,
                angle_threshold_degrees,
                tolerance,
                cut_fraction=0.8,
            )
            if rounded is None:
                continue
            replacement = np.asarray(rounded, dtype=np.float32)
            maximum_transition_segment = max(
                corner_cut * 4.0,
                minimum_span * 10.0,
            )
            transition_start_count = max(
                1,
                int(
                    math.ceil(
                        float(
                            np.linalg.norm(
                                replacement[0] - current[previous_index]
                            )
                        )
                        / maximum_transition_segment
                    )
                ),
            )
            transition_end_count = max(
                1,
                int(
                    math.ceil(
                        float(np.linalg.norm(current[next_index] - replacement[-1]))
                        / maximum_transition_segment
                    )
                ),
            )
            transition_start = np.linspace(
                current[previous_index],
                replacement[0],
                transition_start_count + 1,
                dtype=np.float32,
            )
            transition_end = np.linspace(
                replacement[-1],
                current[next_index],
                transition_end_count + 1,
                dtype=np.float32,
            )
            local_replacement = _dedupe_consecutive(
                np.vstack(
                    (
                        transition_start,
                        replacement[1:],
                        transition_end[1:],
                    )
                ).astype(np.float32),
                tolerance,
            )
            local_line = LineString(local_replacement)
            if not safe_geometry.covers(local_line):
                continue
            proposal = _dedupe_consecutive(
                np.vstack(
                    (
                        current[:previous_index],
                        local_replacement,
                        current[next_index + 1 :],
                    )
                ).astype(np.float32),
                tolerance,
            )
            if LineString(current).is_simple and not LineString(proposal).is_simple:
                continue

            # ``local_replacement`` is tangent to the two rays that meet at
            # ``current[index]``, but those rays need not be tangent to the
            # path immediately outside the selected window.  In particular,
            # refitting a few samples of an existing tiny arc can leave a
            # short backwards hook at either splice.  Such a hook can remain
            # geometrically simple while turning a pair of mild 146-degree
            # bends into several 90-degree bends.  Compare the effective
            # (micro-segment-aware) corner deficit in a small splice context
            # and accept only a strict, non-regressing improvement.
            context_start = previous_index - 1
            while (
                context_start > 0
                and float(
                    np.linalg.norm(
                        current[previous_index, :2] - current[context_start, :2]
                    )
                )
                < minimum_span
            ):
                context_start -= 1
            context_end = next_index + 1
            while (
                context_end < current.shape[0] - 1
                and float(
                    np.linalg.norm(
                        current[context_end, :2] - current[next_index, :2]
                    )
                )
                < minimum_span
            ):
                context_end += 1
            before_context = current[context_start : context_end + 1]
            after_context = _dedupe_consecutive(
                np.vstack(
                    (
                        current[context_start:previous_index],
                        local_replacement,
                        current[next_index + 1 : context_end + 1],
                    )
                ).astype(np.float32),
                tolerance,
            )
            before_corners = _effective_path_corner_candidates(
                before_context,
                minimum_span,
                angle_threshold_degrees,
                tolerance,
            )
            after_corners = _effective_path_corner_candidates(
                after_context,
                minimum_span,
                angle_threshold_degrees,
                tolerance,
            )
            if not before_corners:
                continue
            before_minimum = min(item[0] for item in before_corners)
            after_minimum = min(
                (item[0] for item in after_corners),
                default=angle_threshold_degrees,
            )
            before_deficit = sum(
                angle_threshold_degrees - item[0] for item in before_corners
            )
            after_deficit = sum(
                angle_threshold_degrees - item[0] for item in after_corners
            )
            angle_epsilon = 1e-3
            if (
                after_minimum < before_minimum - angle_epsilon
                or after_deficit >= before_deficit - angle_epsilon
            ):
                continue
            proposal_corners = _effective_path_corner_candidates(
                proposal,
                minimum_span,
                angle_threshold_degrees,
                tolerance,
            )
            current_minimum = min(item[0] for item in candidates)
            proposal_minimum = min(
                (item[0] for item in proposal_corners),
                default=angle_threshold_degrees,
            )
            current_deficit = sum(
                angle_threshold_degrees - item[0] for item in candidates
            )
            proposal_deficit = sum(
                angle_threshold_degrees - item[0] for item in proposal_corners
            )
            if (
                proposal_minimum < current_minimum - angle_epsilon
                or proposal_deficit >= current_deficit - angle_epsilon
            ):
                continue
            current = proposal
            accepted = True
            break
        if not accepted:
            blocked_points.add(point_key)
    return current


def _split_path_at_unresolved_effective_corners(
    path: np.ndarray,
    minimum_span: float,
    angle_threshold_degrees: float,
    tolerance: float,
) -> list[np.ndarray]:
    """Keep a simple centerline and minimally cut unsmoothable hard turns."""

    points = _dedupe_consecutive(
        np.asarray(path[:, :2], dtype=np.float32),
        tolerance,
    )
    if points.shape[0] < 2:
        return []
    if not LineString(points).is_simple:
        raise ValueError("solid-fill corner splitting requires a simple baseline")

    pieces = [points]
    while True:
        changed = False
        next_pieces: list[np.ndarray] = []
        for piece in pieces:
            candidates = _effective_path_corner_candidates(
                piece,
                minimum_span,
                angle_threshold_degrees,
                tolerance,
            )
            if not candidates:
                next_pieces.append(piece)
                continue

            # Each effective corner is supported by the open vertex interval
            # (previous_index, next_index).  A cut anywhere in that interval
            # makes the turn a path endpoint.  Greedily stabbing intervals by
            # increasing right endpoint gives the minimum cut count for this
            # candidate set, while preserving original path order.
            cut_indices: list[int] = []
            for _, center_index, previous_index, next_index in sorted(
                candidates,
                key=lambda item: (item[3], item[2], item[1]),
            ):
                if any(
                    previous_index < cut_index < next_index
                    for cut_index in cut_indices
                ):
                    continue
                cut_index = next_index - 1
                if not (
                    0 < cut_index < piece.shape[0] - 1
                    and previous_index < cut_index < next_index
                ):
                    # Numerical edge cases still have a valid central vertex.
                    cut_index = center_index
                if 0 < cut_index < piece.shape[0] - 1:
                    cut_indices.append(cut_index)

            cut_indices = sorted(set(cut_indices))
            if not cut_indices:
                # An effective corner always has an interior centre; retaining
                # it would violate the configured threshold, so fail closed.
                raise ValueError("unable to split unresolved solid-fill corner")

            start_index = 0
            for cut_index in cut_indices:
                next_pieces.append(piece[start_index : cut_index + 1].copy())
                start_index = cut_index
            next_pieces.append(piece[start_index:].copy())
            changed = True

        pieces = next_pieces
        if not changed:
            return pieces


def _solid_fill_spacing_postcondition(
    paths: list[np.ndarray],
    wall_linework,
    minimum_spacing: float,
    tolerance: float,
    *,
    bead_width: float | None = None,
    minimum_wall_spacing: float | None = None,
    allow_boundary_bridges: bool = False,
) -> bool:
    """Verify the physical pitch after all continuity and smoothing passes.

    Consecutive polyline segments necessarily meet at a vertex, and a rounded
    U-turn necessarily remains close to its incident hatch.  Those are one
    continuous deposited bead, not two neighbouring passes.  The material-pile
    failure mode is instead a pair of non-local, material-length runs (including
    returns in the same one-stroke path).  Collinear micro-segments are first
    coalesced so numerical sampling cannot hide an under-spaced long hatch.
    """

    if minimum_spacing <= 0:
        raise ValueError("minimum_spacing must be positive")
    if minimum_wall_spacing is None:
        minimum_wall_spacing = minimum_spacing
    if minimum_wall_spacing <= 0:
        raise ValueError("minimum_wall_spacing must be positive")
    if bead_width is not None and bead_width <= 0:
        raise ValueError("bead_width must be positive")
    comparison_epsilon = max(1e-6, minimum_spacing * 1e-7)
    wall_comparison_epsilon = max(1e-6, minimum_wall_spacing * 1e-7)

    # The sharp-corner finisher is allowed to split one deposited trail into
    # several arrays.  Validate the physical centreline, not that incidental
    # array partition: otherwise a tight return can evade the same-path checks
    # simply by putting every sampled segment in a separate path.  Merge only
    # exact, non-branching endpoint chains.  Branched, overlapping, or
    # self-intersecting components remain separate and therefore receive no
    # continuity exemption below.
    path_arrays = [np.asarray(path) for path in paths]
    endpoint_merge_tolerance = max(tolerance * 2.0, comparison_epsilon)
    merge_parents = list(range(len(path_arrays)))

    def merge_root(index: int) -> int:
        while merge_parents[index] != index:
            merge_parents[index] = merge_parents[merge_parents[index]]
            index = merge_parents[index]
        return index

    def union_merge_components(first: int, second: int) -> None:
        first_root = merge_root(first)
        second_root = merge_root(second)
        if first_root != second_root:
            merge_parents[second_root] = first_root

    merge_endpoints = [
        (
            np.asarray(path[0, :2], dtype=np.float64),
            np.asarray(path[-1, :2], dtype=np.float64),
        )
        if path.ndim == 2 and path.shape[0] >= 2
        else None
        for path in path_arrays
    ]
    for first_index, first_endpoints in enumerate(merge_endpoints):
        if first_endpoints is None:
            continue
        for second_index in range(first_index + 1, len(merge_endpoints)):
            second_endpoints = merge_endpoints[second_index]
            if second_endpoints is None:
                continue
            if any(
                float(np.linalg.norm(first_endpoint - second_endpoint))
                <= endpoint_merge_tolerance
                for first_endpoint in first_endpoints
                for second_endpoint in second_endpoints
            ):
                union_merge_components(first_index, second_index)

    merge_members: dict[int, list[int]] = defaultdict(list)
    for path_index, endpoints in enumerate(merge_endpoints):
        if endpoints is not None:
            merge_members[merge_root(path_index)].append(path_index)
    merged_replacements: dict[int, np.ndarray] = {}
    merged_consumed: set[int] = set()
    for members in merge_members.values():
        if len(members) < 2:
            continue
        member_lines = [
            LineString(np.asarray(path_arrays[index][:, :2], dtype=np.float64))
            for index in members
        ]
        merged = linemerge(MultiLineString(member_lines))
        total_length = sum(float(line.length) for line in member_lines)
        if (
            isinstance(merged, LineString)
            and merged.is_simple
            and abs(float(merged.length) - total_length)
            <= endpoint_merge_tolerance * max(1, len(member_lines))
        ):
            representative = min(members)
            merged_replacements[representative] = np.asarray(
                merged.coords,
                dtype=np.float64,
            )
            merged_consumed.update(index for index in members if index != representative)
    paths = [
        merged_replacements.get(index, path)
        for index, path in enumerate(path_arrays)
        if index not in merged_consumed
    ]

    path_lines = [
        LineString(np.asarray(path[:, :2], dtype=np.float64))
        for path in paths
        if path.shape[0] >= 2
    ]
    if not path_lines:
        return True
    if any(not line.is_simple for line in path_lines):
        return False

    # A tiny closed ring can be made entirely of short arc samples and thus
    # contain no material-length directional run below.  Its opposite sides
    # still receive the same 2.2 mm bead and may completely stack.  Eroding the
    # enclosed region by half the required pitch is the closed-curve analogue
    # of the run-spacing check: an empty or split core proves that non-local
    # sides of the ring approach more closely than the allowed centre distance.
    closed_curve_radius = max(
        0.0,
        (minimum_spacing if bead_width is None else bead_width) * 0.5
        - comparison_epsilon,
    )
    if closed_curve_radius > 0:
        for path in paths:
            points = np.asarray(path[:, :2], dtype=np.float64)
            if points.shape[0] < 4 or not _close(
                points[0], points[-1], tolerance
            ):
                continue
            enclosed = Polygon(points)
            if enclosed.is_empty or not enclosed.is_valid:
                return False
            closed_core = enclosed.buffer(
                -closed_curve_radius,
                join_style="round",
            )
            if closed_core.is_empty or not isinstance(closed_core, Polygon):
                return False

    # Corner fallback may split one logical trail into several paths, including
    # a very short point-to-point bridge.  Reconstruct those exact endpoint
    # components using only geometry-scale tolerance; nearby but independent
    # paths must never gain the same exemption.
    component_parents = list(range(len(paths)))

    def component_root(index: int) -> int:
        while component_parents[index] != index:
            component_parents[index] = component_parents[component_parents[index]]
            index = component_parents[index]
        return index

    def union_components(first: int, second: int) -> None:
        first_root = component_root(first)
        second_root = component_root(second)
        if first_root != second_root:
            component_parents[second_root] = first_root

    endpoint_tolerance = max(tolerance * 2.0, comparison_epsilon)
    direct_endpoint_links: dict[
        tuple[int, int], list[tuple[int, int]]
    ] = defaultdict(list)
    path_endpoints = [
        (
            np.asarray(path[0, :2], dtype=np.float64),
            np.asarray(path[-1, :2], dtype=np.float64),
        )
        if path.shape[0] >= 2
        else None
        for path in paths
    ]
    for first_index, first_endpoints in enumerate(path_endpoints):
        if first_endpoints is None:
            continue
        for second_index in range(first_index + 1, len(path_endpoints)):
            second_endpoints = path_endpoints[second_index]
            if second_endpoints is None:
                continue
            matching_sides = [
                (first_side, second_side)
                for first_side, first_endpoint in enumerate(first_endpoints)
                for second_side, second_endpoint in enumerate(second_endpoints)
                if float(np.linalg.norm(first_endpoint - second_endpoint))
                <= endpoint_tolerance
            ]
            if matching_sides:
                union_components(first_index, second_index)
                direct_endpoint_links[(first_index, second_index)].extend(
                    matching_sides
                )
    if (
        wall_linework is not None
        and not wall_linework.is_empty
        and unary_union(path_lines).distance(wall_linework)
        < minimum_wall_spacing - wall_comparison_epsilon
    ):
        return False

    # A directional run represents a material pass, not an individual sample
    # chord from a fillet.  Keeping the threshold at a quarter pitch prevents
    # medium-density arc samples from being mistaken for parallel return arms;
    # the raw fallback below still audits curves made entirely of short chords.
    minimum_run_length = max(minimum_spacing * 0.25, tolerance * 20.0)
    collinear_cosine = math.cos(math.radians(1.0))
    run_lines: list[LineString] = []
    run_records: list[
        tuple[int, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]
    ] = []
    run_topology: list[tuple[float, float, float, bool]] = []
    path_topology_points: list[np.ndarray] = []
    path_topology_cumulative: list[np.ndarray] = []
    path_topology_turn_cumulative: list[np.ndarray] = []
    path_topology_lines: list[LineString] = []
    raw_segment_lines: list[LineString] = []
    raw_segment_records: list[tuple[int, int]] = []
    raw_segment_lengths: list[float] = []

    def append_run(
        path_index: int,
        run_points: np.ndarray,
        start_position: float,
        end_position: float,
        path_length: float,
        is_closed: bool,
    ) -> None:
        if run_points.shape[0] < 2:
            return
        run = LineString(run_points)
        if run.length < minimum_run_length:
            return
        start_point = np.asarray(run_points[0], dtype=np.float64)
        end_point = np.asarray(run_points[-1], dtype=np.float64)
        direction = end_point - start_point
        direction_length = float(np.linalg.norm(direction))
        if direction_length <= tolerance:
            return
        direction /= direction_length
        projection = sorted(
            (
                float(np.dot(start_point, direction)),
                float(np.dot(end_point, direction)),
            )
        )
        run_lines.append(run)
        run_records.append(
            (
                path_index,
                start_point,
                end_point,
                direction,
                (projection[0], projection[1]),
            )
        )
        run_topology.append(
            (start_position, end_position, path_length, is_closed)
        )

    for path_index, path in enumerate(paths):
        points = np.asarray(path[:, :2], dtype=np.float64)
        segment_vectors = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        cumulative_lengths = np.concatenate(
            (np.asarray([0.0]), np.cumsum(segment_lengths))
        )
        path_topology_points.append(points)
        path_topology_cumulative.append(cumulative_lengths)
        path_topology_lines.append(LineString(points))
        turn_cumulative = np.zeros(segment_lengths.shape[0], dtype=np.float64)
        prior_direction: np.ndarray | None = None
        total_turn = 0.0
        for segment_index, (vector, length) in enumerate(
            zip(segment_vectors, segment_lengths)
        ):
            if float(length) > tolerance:
                direction = vector / float(length)
                if prior_direction is not None:
                    total_turn += abs(
                        math.atan2(
                            float(
                                prior_direction[0] * direction[1]
                                - prior_direction[1] * direction[0]
                            ),
                            float(np.dot(prior_direction, direction)),
                        )
                    )
                prior_direction = direction
            turn_cumulative[segment_index] = total_turn
        path_topology_turn_cumulative.append(turn_cumulative)
        path_length = float(cumulative_lengths[-1])
        is_closed = _close(points[0], points[-1], tolerance)
        for segment_index, length in enumerate(segment_lengths):
            if float(length) <= tolerance:
                continue
            raw_segment_lines.append(
                LineString(points[segment_index : segment_index + 2])
            )
            raw_segment_records.append((path_index, segment_index))
            raw_segment_lengths.append(float(length))
        raw_runs: list[tuple[np.ndarray, float, float]] = []
        run_start: int | None = None
        run_end: int | None = None
        run_direction: np.ndarray | None = None
        for segment_index, (vector, length) in enumerate(
            zip(segment_vectors, segment_lengths)
        ):
            length = float(length)
            if length <= tolerance:
                continue
            direction = vector / length
            if run_start is None:
                run_start = segment_index
                run_end = segment_index + 1
                run_direction = direction
                continue
            if float(np.dot(run_direction, direction)) >= collinear_cosine:
                run_end = segment_index + 1
                chord = points[run_end] - points[run_start]
                chord_length = float(np.linalg.norm(chord))
                if chord_length > tolerance:
                    run_direction = chord / chord_length
                continue
            raw_runs.append(
                (
                    points[run_start : run_end + 1],
                    float(cumulative_lengths[run_start]),
                    float(cumulative_lengths[run_end]),
                )
            )
            run_start = segment_index
            run_end = segment_index + 1
            run_direction = direction
        if run_start is not None and run_end is not None:
            raw_runs.append(
                (
                    points[run_start : run_end + 1],
                    float(cumulative_lengths[run_start]),
                    float(cumulative_lengths[run_end]),
                )
            )

        # A closed ring may start halfway along a straight side.  Join its last
        # and first directional runs before the material-length filter so two
        # short halves cannot hide a long folded return.
        if is_closed and len(raw_runs) >= 2:
            first_vector = raw_runs[0][0][-1] - raw_runs[0][0][0]
            last_vector = raw_runs[-1][0][-1] - raw_runs[-1][0][0]
            first_length = float(np.linalg.norm(first_vector))
            last_length = float(np.linalg.norm(last_vector))
            if (
                first_length > tolerance
                and last_length > tolerance
                and float(
                    np.dot(
                        first_vector / first_length,
                        last_vector / last_length,
                    )
                )
                >= collinear_cosine
            ):
                raw_runs[0] = (
                    np.vstack((raw_runs[-1][0][:-1], raw_runs[0][0])),
                    raw_runs[-1][1],
                    raw_runs[0][2],
                )
                raw_runs.pop()
        for run_points, start_position, end_position in raw_runs:
            append_run(
                path_index,
                run_points,
                start_position,
                end_position,
                path_length,
                is_closed,
            )

    query_distance = max(0.0, minimum_spacing - comparison_epsilon)
    local_bead_width = minimum_spacing if bead_width is None else bead_width
    component_members: dict[int, list[int]] = defaultdict(list)
    for path_index, endpoints in enumerate(path_endpoints):
        if endpoints is not None:
            component_members[component_root(path_index)].append(path_index)
    merged_component_lines: dict[int, LineString] = {}
    for root, members in component_members.items():
        if len(members) < 2:
            continue
        merged = linemerge(
            MultiLineString(
                [
                    LineString(path_topology_points[path_index])
                    for path_index in members
                ]
            )
        )
        if isinstance(merged, LineString) and merged.is_simple and not merged.is_ring:
            merged_component_lines[root] = merged

    def path_interval_points(
        path_index: int,
        first_position: float,
        second_position: float,
        *,
        wraps: bool = False,
    ) -> np.ndarray:
        lower = min(first_position, second_position)
        upper = max(first_position, second_position)
        points = path_topology_points[path_index]
        cumulative = path_topology_cumulative[path_index]
        source_line = path_topology_lines[path_index]
        lower_point = np.asarray(
            source_line.interpolate(lower).coords[0], dtype=np.float64
        )
        upper_point = np.asarray(
            source_line.interpolate(upper).coords[0], dtype=np.float64
        )
        if wraps:
            after_upper = points[cumulative > upper + comparison_epsilon]
            before_lower = points[cumulative < lower - comparison_epsilon]
            return np.vstack(
                (
                    upper_point,
                    after_upper,
                    before_lower,
                    lower_point,
                )
            )
        interior = points[
            (cumulative > lower + comparison_epsilon)
            & (cumulative < upper - comparison_epsilon)
        ]
        return np.vstack((lower_point, interior, upper_point))

    def absolute_heading_change(local_points: np.ndarray) -> float:
        vectors = np.diff(local_points, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        vectors = vectors[lengths > tolerance]
        if vectors.shape[0] < 2:
            return 0.0
        headings = np.arctan2(vectors[:, 1], vectors[:, 0])
        heading_steps = np.arctan2(
            np.sin(np.diff(headings)),
            np.cos(np.diff(headings)),
        )
        return float(np.sum(np.abs(heading_steps)))

    def points_form_compact_local_turn(local_points: np.ndarray) -> bool:
        # A genuine continuous U-turn has both a short material route (checked
        # by the callers) and a compact spatial footprint.  The small 1.6-bead
        # diagonal allowance covers a discretised semicircular connector plus
        # its incident tangent samples; a later loop-back cannot qualify merely
        # because its two nearest points are spatially close.
        span = np.ptp(local_points, axis=0)
        if (
            float(np.linalg.norm(span))
            > local_bead_width * 1.6 + endpoint_tolerance
        ):
            return False
        # One generated hatch-end connector reverses direction once.  Permit a
        # small discretisation/smoothing allowance; a materially larger turn
        # is an almost-loop, not a local U-turn exemption.
        heading_change = absolute_heading_change(local_points)
        if heading_change > math.pi + math.radians(10.0):
            return False
        return True

    def same_path_interval_absolute_turn(
        path_index: int,
        first_position: float,
        second_position: float,
    ) -> float:
        cumulative = path_topology_cumulative[path_index]
        path_length = float(cumulative[-1])
        points = path_topology_points[path_index]
        direct_distance = abs(first_position - second_position)
        wraps = (
            _close(points[0], points[-1], tolerance)
            and path_length - direct_distance < direct_distance
        )
        return absolute_heading_change(
            path_interval_points(
                path_index,
                first_position,
                second_position,
                wraps=wraps,
            )
        )

    def same_path_positions_form_compact_local_turn(
        path_index: int,
        first_position: float,
        second_position: float,
    ) -> bool:
        """Distinguish a compact continuous bend from a later loop-back."""

        cumulative = path_topology_cumulative[path_index]
        path_length = float(cumulative[-1])
        points = path_topology_points[path_index]
        is_closed = _close(points[0], points[-1], tolerance)
        direct_distance = abs(first_position - second_position)
        wraps = is_closed and path_length - direct_distance < direct_distance
        material_distance = (
            path_length - direct_distance if wraps else direct_distance
        )
        if material_distance <= minimum_run_length + endpoint_tolerance:
            return True
        if material_distance > math.pi * local_bead_width + endpoint_tolerance:
            return False
        return points_form_compact_local_turn(
            path_interval_points(
                path_index,
                first_position,
                second_position,
                wraps=wraps,
            )
        )

    def different_path_positions_form_compact_local_turn(
        first_path: int,
        first_position: float,
        second_path: int,
        second_position: float,
    ) -> bool:
        links = direct_endpoint_links.get((first_path, second_path))
        reverse_links = False
        if links is None:
            links = direct_endpoint_links.get((second_path, first_path))
            reverse_links = links is not None
        if links:
            first_length = float(path_topology_cumulative[first_path][-1])
            second_length = float(path_topology_cumulative[second_path][-1])
            candidates: list[tuple[float, int, int]] = []
            for raw_first_side, raw_second_side in links:
                first_side, second_side = (
                    (raw_second_side, raw_first_side)
                    if reverse_links
                    else (raw_first_side, raw_second_side)
                )
                first_endpoint_position = (
                    0.0 if first_side == 0 else first_length
                )
                second_endpoint_position = (
                    0.0 if second_side == 0 else second_length
                )
                candidates.append(
                    (
                        abs(first_position - first_endpoint_position)
                        + abs(second_position - second_endpoint_position),
                        first_side,
                        second_side,
                    )
                )
            material_distance, first_side, second_side = min(
                candidates,
                key=lambda item: item[0],
            )
            if material_distance <= minimum_spacing * 0.3 + endpoint_tolerance:
                return True
            if (
                material_distance
                <= math.pi * local_bead_width + endpoint_tolerance
            ):
                first_endpoint_position = (
                    0.0 if first_side == 0 else first_length
                )
                second_endpoint_position = (
                    0.0 if second_side == 0 else second_length
                )
                first_interval = path_interval_points(
                    first_path,
                    first_position,
                    first_endpoint_position,
                )
                if first_position > first_endpoint_position:
                    first_interval = first_interval[::-1]
                second_interval = path_interval_points(
                    second_path,
                    second_endpoint_position,
                    second_position,
                )
                if second_endpoint_position > second_position:
                    second_interval = second_interval[::-1]
                if points_form_compact_local_turn(
                    np.vstack((first_interval, second_interval))
                ):
                    return True

        # The hard-corner splitter may insert one or more exact, tiny bridge
        # paths.  Rebuild only non-branching endpoint chains and apply the same
        # short-route/compact-turn proof to the actual merged material route.
        # A merely transitive component receives no blanket exemption: the
        # route must itself remain local, so a long A-C-B detour is rejected.
        root = component_root(first_path)
        if root != component_root(second_path):
            return False
        merged = merged_component_lines.get(root)
        if merged is None:
            return False
        first_point = path_topology_lines[first_path].interpolate(
            first_position
        )
        second_point = path_topology_lines[second_path].interpolate(
            second_position
        )
        merged_first = float(merged.project(first_point))
        merged_second = float(merged.project(second_point))
        material_distance = abs(merged_first - merged_second)
        if material_distance <= minimum_spacing * 0.3 + endpoint_tolerance:
            return True
        if material_distance > math.pi * local_bead_width + endpoint_tolerance:
            return False
        lower = min(merged_first, merged_second)
        upper = max(merged_first, merged_second)
        merged_points = np.asarray(merged.coords, dtype=np.float64)
        merged_lengths = np.linalg.norm(np.diff(merged_points, axis=0), axis=1)
        merged_cumulative = np.concatenate(
            (np.asarray([0.0]), np.cumsum(merged_lengths))
        )
        interval = np.vstack(
            (
                np.asarray(merged.interpolate(lower).coords[0]),
                merged_points[
                    (merged_cumulative > lower + comparison_epsilon)
                    & (merged_cumulative < upper - comparison_epsilon)
                ],
                np.asarray(merged.interpolate(upper).coords[0]),
            )
        )
        return points_form_compact_local_turn(interval)

    def run_path_position(
        run_index: int,
        run_line: LineString,
        point: Point,
    ) -> float:
        start_position, _, path_length, is_closed = run_topology[run_index]
        position = start_position + float(run_line.project(point))
        if is_closed and path_length > 0:
            position %= path_length
        return position

    def incident_runs_have_safe_turn_radius(
        first_index: int,
        second_index: int,
        run_distance: float,
    ) -> bool:
        """Apply a sampling-independent radius check to one local bend.

        Arc samples may be arbitrarily dense or sparse, but the incident
        material-length runs retain the physical turn angle and their minimum
        separation.  For tangent rays, ``distance / (2 sin(theta / 2))`` is the
        bend radius.  A short boundary-following bridge between two already
        safe, pitch-separated hatch arms is handled as an explicit three-run
        topology below; every true two-arm return is checked regardless of the
        arc sampling density.
        """

        first_path, _, _, first_direction, _ = run_records[first_index]
        second_path, _, _, second_direction, _ = run_records[second_index]
        if first_path != second_path:
            return False
        cosine = float(
            np.clip(np.dot(first_direction, second_direction), -1.0, 1.0)
        )
        turn_angle = math.acos(cosine)
        if turn_angle <= math.radians(90.0) + 1e-9:
            return True
        sine_half_turn = math.sin(turn_angle * 0.5)
        if sine_half_turn <= 1e-12:
            return True
        effective_radius = run_distance / (2.0 * sine_half_turn)
        minimum_radius = (
            minimum_spacing
            * 0.5
            * sine_half_turn**10
        )
        radius_tolerance = max(endpoint_tolerance, minimum_spacing * 1e-4)
        if effective_radius >= minimum_radius - radius_tolerance:
            return True

        # A generated boundary join can contain three material-length runs:
        # hatch arm -> short boundary bridge -> next hatch arm.  The bridge may
        # locally approach its incident arm, but it is not a second return pass
        # when the two outer hatch arms themselves are antiparallel and remain
        # at or beyond the configured pitch.  Make that exception structural,
        # rather than applying the former blanket <=150-degree exemption which
        # also admitted arbitrarily tight 135-149 degree hairpins.
        if turn_angle > math.radians(135.1):
            return False
        path_run_indices = sorted(
            (
                run_index
                for run_index, record in enumerate(run_records)
                if record[0] == first_path
            ),
            key=lambda run_index: run_topology[run_index][0],
        )
        if run_topology[first_index][3] or len(path_run_indices) < 3:
            return False
        positions = {
            run_index: position
            for position, run_index in enumerate(path_run_indices)
        }
        first_order = positions[first_index]
        second_order = positions[second_index]
        bridge_candidates: list[tuple[int, int, int]] = []
        if second_order == first_order + 1 and first_order > 0:
            bridge_candidates.append(
                (
                    first_index,
                    path_run_indices[first_order - 1],
                    second_index,
                )
            )
        if first_order == second_order + 1 and second_order > 0:
            bridge_candidates.append(
                (
                    second_index,
                    path_run_indices[second_order - 1],
                    first_index,
                )
            )
        if second_order == first_order + 1 and second_order + 1 < len(path_run_indices):
            bridge_candidates.append(
                (
                    second_index,
                    first_index,
                    path_run_indices[second_order + 1],
                )
            )
        if first_order == second_order + 1 and first_order + 1 < len(path_run_indices):
            bridge_candidates.append(
                (
                    first_index,
                    second_index,
                    path_run_indices[first_order + 1],
                )
        )

        for bridge_index, outer_index, other_index in bridge_candidates:
            # This exemption is only for a connector that actually follows the
            # bead-aware infill boundary beside the innermost perimeter.  That
            # geometric anchor distinguishes it from an arbitrary third segment
            # appended to a tight hairpin, and also permits low-density hatches
            # whose legitimate boundary bridge is longer than one bead width.
            if not allow_boundary_bridges:
                if wall_linework is None or wall_linework.is_empty:
                    continue
                bridge_wall_distance = run_lines[bridge_index].distance(
                    wall_linework
                )
                if bridge_wall_distance > (
                    minimum_spacing
                    + max(local_bead_width * 0.1, endpoint_tolerance)
                ):
                    continue
            bridge_start, bridge_end, _, _ = run_topology[bridge_index]
            outer_start, outer_end, _, _ = run_topology[outer_index]
            other_start, other_end, _, _ = run_topology[other_index]
            ordered = sorted(
                (
                    (outer_start, outer_end, outer_index),
                    (bridge_start, bridge_end, bridge_index),
                    (other_start, other_end, other_index),
                ),
                key=lambda item: item[0],
            )
            if ordered[1][2] != bridge_index:
                continue
            if (
                ordered[1][0] - ordered[0][1]
                > minimum_spacing + endpoint_tolerance
                or ordered[2][0] - ordered[1][1]
                > minimum_spacing + endpoint_tolerance
            ):
                continue
            outer_direction = run_records[outer_index][3]
            other_direction = run_records[other_index][3]
            if float(np.dot(outer_direction, other_direction)) > -math.cos(
                math.radians(10.0)
            ):
                continue
            if run_lines[outer_index].distance(run_lines[other_index]) < (
                minimum_spacing - comparison_epsilon
            ):
                continue
            return True
        return False

    # Directional-run aggregation is intentionally optimized for long hatch
    # ridges.  Curves sampled into many sub-run segments need an exact fallback
    # so a spiral or two independent arcs cannot evade the physical spacing
    # contract merely because every individual segment is short.
    if len(raw_segment_lines) >= 2:
        raw_tree = STRtree(raw_segment_lines)
        for first_index, first_line in enumerate(raw_segment_lines):
            first_path, first_segment = raw_segment_records[first_index]
            for raw_second_index in raw_tree.query(
                first_line,
                predicate="dwithin",
                distance=query_distance,
            ):
                second_index = int(raw_second_index)
                if second_index <= first_index:
                    continue
                second_line = raw_segment_lines[second_index]
                if (
                    first_line.distance(second_line)
                    >= minimum_spacing - comparison_epsilon
                ):
                    continue
                second_path, second_segment = raw_segment_records[second_index]
                # Material-length directional runs are checked again below
                # after collinear aggregation.  The raw fallback exists for
                # finely sampled curves, so do not duplicate the expensive
                # exact topology work when both segments already qualify.
                if (
                    raw_segment_lengths[first_index] >= minimum_run_length
                    and raw_segment_lengths[second_index] >= minimum_run_length
                ):
                    continue
                if first_path == second_path:
                    lower_segment = min(first_segment, second_segment)
                    upper_segment = max(first_segment, second_segment)
                    cumulative = path_topology_cumulative[first_path]
                    material_gap = max(
                        0.0,
                        float(
                            cumulative[upper_segment]
                            - cumulative[lower_segment + 1]
                        ),
                    )
                    points = path_topology_points[first_path]
                    is_closed = _close(points[0], points[-1], tolerance)
                    if is_closed:
                        path_length = float(cumulative[-1])
                        material_gap = min(
                            material_gap,
                            max(0.0, path_length - material_gap),
                        )
                    # Nearby samples along one deposited centreline are the
                    # unavoidable local material continuation.  Non-local
                    # returns, spirals, and almost-loops remain beyond one
                    # requested pitch and still take the exact path below.
                    turn_cumulative = path_topology_turn_cumulative[first_path]
                    material_turn = float(
                        turn_cumulative[upper_segment]
                        - turn_cumulative[lower_segment]
                    )
                    if (
                        not is_closed
                        and material_gap
                        <= minimum_run_length + endpoint_tolerance
                        and material_turn <= math.radians(135.0)
                    ):
                        continue
                    if (
                        not is_closed
                        and material_gap
                        <= minimum_spacing + endpoint_tolerance
                        and material_turn <= math.radians(90.0)
                    ):
                        continue
                first_nearest, second_nearest = nearest_points(
                    first_line,
                    second_line,
                )
                first_position = float(
                    path_topology_cumulative[first_path][first_segment]
                    + first_line.project(first_nearest)
                )
                second_position = float(
                    path_topology_cumulative[second_path][second_segment]
                    + second_line.project(second_nearest)
                )
                if first_path == second_path:
                    if same_path_positions_form_compact_local_turn(
                        first_path,
                        first_position,
                        second_position,
                    ):
                        continue
                    return False
                if different_path_positions_form_compact_local_turn(
                    first_path,
                    first_position,
                    second_path,
                    second_position,
                ):
                    continue
                return False

    if len(run_lines) < 2:
        return True

    tree = STRtree(run_lines)
    for first_index, first_line in enumerate(run_lines):
        first_path = run_records[first_index][0]
        for raw_second_index in tree.query(
            first_line,
            predicate="dwithin",
            distance=query_distance,
        ):
            second_index = int(raw_second_index)
            if second_index <= first_index:
                continue
            second_path = run_records[second_index][0]
            second_line = run_lines[second_index]
            same_component = component_root(first_path) == component_root(second_path)
            first_nearest, second_nearest = nearest_points(
                first_line,
                second_line,
            )
            first_position = run_path_position(
                first_index,
                first_line,
                first_nearest,
            )
            second_position = run_path_position(
                second_index,
                second_line,
                second_nearest,
            )
            locally_adjacent = (
                (
                    first_path == second_path
                    and same_path_positions_form_compact_local_turn(
                        first_path,
                        first_position,
                        second_position,
                    )
                )
                or (
                    first_path != second_path
                    and same_component
                    and different_path_positions_form_compact_local_turn(
                        first_path,
                        first_position,
                        second_path,
                        second_position,
                    )
                )
            )
            distance = first_line.distance(second_line)
            if distance >= minimum_spacing - comparison_epsilon:
                continue
            if same_component and locally_adjacent:
                # Exact non-branching endpoint chains were normalised above.
                # Any remaining cross-path component is branched/ambiguous and
                # must not gain a local-turn exemption.  For one logical path,
                # validate the incident runs before the short-route fast path;
                # this is what makes a tight U-turn independent of arc sampling.
                if first_path != second_path:
                    return False
                if not incident_runs_have_safe_turn_radius(
                    first_index,
                    second_index,
                    distance,
                ):
                    return False
                if first_path == second_path:
                    path_length = float(
                        path_topology_cumulative[first_path][-1]
                    )
                    local_material_distance = abs(
                        first_position - second_position
                    )
                    if _close(
                        path_topology_points[first_path][0],
                        path_topology_points[first_path][-1],
                        tolerance,
                    ):
                        local_material_distance = min(
                            local_material_distance,
                            path_length - local_material_distance,
                        )
                    # Consecutive directional pieces inside one sampled fillet
                    # are one local curve, not a second pass.  The raw-segment
                    # guard still checks the whole curve for later loop-backs.
                    if local_material_distance <= (
                        minimum_spacing + endpoint_tolerance
                    ):
                        continue
                    local_interval_turn = same_path_interval_absolute_turn(
                        first_path,
                        first_position,
                        second_position,
                    )
                    if local_interval_turn <= math.radians(95.0):
                        continue
                # A compact local bend may be split across exact path boundaries
                # when the sharp-corner finisher inserts a tiny bridge.  Apply the
                # same physical corridor test to that reconstructed component as
                # to an unsplit curve.  Only a fully close run lasting almost a
                # complete pitch is a second material pass; a shorter curved
                # sample is intrinsic to the one local turn.
                first_close_length = float(
                    first_line.intersection(
                        second_line.buffer(
                            query_distance,
                            cap_style="round",
                            join_style="round",
                        )
                    ).length
                )
                second_close_length = float(
                    second_line.intersection(
                        first_line.buffer(
                            query_distance,
                            cap_style="round",
                            join_style="round",
                        )
                    ).length
                )
                directions_are_parallel = abs(
                    float(
                        np.dot(
                            run_records[first_index][3],
                            run_records[second_index][3],
                        )
                    )
                ) >= math.cos(math.radians(25.0))
                sustained_run_length = minimum_spacing * 0.95
                if (
                    directions_are_parallel
                    and (
                        (
                            float(first_line.length)
                            >= sustained_run_length - endpoint_tolerance
                            and first_close_length
                            >= float(first_line.length) - endpoint_tolerance
                        )
                        or (
                            float(second_line.length)
                            >= sustained_run_length - endpoint_tolerance
                            and second_close_length
                            >= float(second_line.length) - endpoint_tolerance
                        )
                    )
                ):
                    return False
                if min(first_close_length, second_close_length) <= (
                    minimum_spacing + endpoint_tolerance
                ):
                    continue
            return False
    return True


def _trim_close_collinear_solid_fill_endcaps(
    paths: list[np.ndarray],
    target_spacing: float,
    tolerance: float,
) -> list[np.ndarray]:
    """Separate facing hatch endcaps without moving the neighbouring runs.

    A scanline can be split by the bead-aware wall exclusion into two
    collinear pieces.  When that excluded interval is shorter than the
    measured-width pitch, two independent starts/stops would still overlap at
    their round endcaps even though all parallel hatch levels are correctly
    spaced.  A direct bridge is not necessarily legal: on a curved wall it can
    cut through the wall exclusion.  Conservatively retract both facing ends
    along their existing straight terminal segments instead.  Exact endpoint
    components created by the hard-corner splitter are one logical trail and
    are deliberately left unchanged.
    """

    if target_spacing <= 0:
        raise ValueError("target_spacing must be positive")
    if len(paths) < 2:
        return paths

    repaired = [np.asarray(path, dtype=np.float32).copy() for path in paths]
    endpoint_tolerance = max(tolerance * 2.0, target_spacing * 1e-7, 1e-6)
    component_parents = list(range(len(repaired)))

    def component_root(index: int) -> int:
        while component_parents[index] != index:
            component_parents[index] = component_parents[
                component_parents[index]
            ]
            index = component_parents[index]
        return index

    def union_components(first: int, second: int) -> None:
        first_root = component_root(first)
        second_root = component_root(second)
        if first_root != second_root:
            component_parents[second_root] = first_root

    path_endpoints = [
        (
            np.asarray(path[0, :2], dtype=np.float64),
            np.asarray(path[-1, :2], dtype=np.float64),
        )
        if path.shape[0] >= 2 and not _is_closed_path(path, tolerance)
        else None
        for path in repaired
    ]
    for first_index, first_endpoints in enumerate(path_endpoints):
        if first_endpoints is None:
            continue
        for second_index in range(first_index + 1, len(path_endpoints)):
            second_endpoints = path_endpoints[second_index]
            if second_endpoints is None:
                continue
            if any(
                float(np.linalg.norm(first_endpoint - second_endpoint))
                <= endpoint_tolerance
                for first_endpoint in first_endpoints
                for second_endpoint in second_endpoints
            ):
                union_components(first_index, second_index)

    endpoints: list[tuple[int, int]] = [
        (path_index, side)
        for path_index, path in enumerate(repaired)
        if path.shape[0] >= 2 and not _is_closed_path(path, tolerance)
        for side in (0, -1)
    ]
    candidate_pairs: list[tuple[float, int, int]] = []
    collinear_cosine = math.cos(math.radians(1.0))
    for first_endpoint_index, (first_path, first_side) in enumerate(endpoints):
        first_point = np.asarray(
            repaired[first_path][first_side, :2], dtype=np.float64
        )
        first_neighbour = np.asarray(
            repaired[first_path][1 if first_side == 0 else -2, :2],
            dtype=np.float64,
        )
        first_inward = first_neighbour - first_point
        first_length = float(np.linalg.norm(first_inward))
        if first_length <= endpoint_tolerance:
            continue
        first_inward /= first_length
        for second_endpoint_index in range(
            first_endpoint_index + 1, len(endpoints)
        ):
            second_path, second_side = endpoints[second_endpoint_index]
            if component_root(first_path) == component_root(second_path):
                continue
            second_point = np.asarray(
                repaired[second_path][second_side, :2], dtype=np.float64
            )
            gap = second_point - first_point
            gap_length = float(np.linalg.norm(gap))
            if not (
                endpoint_tolerance < gap_length
                < target_spacing - endpoint_tolerance
            ):
                continue
            second_neighbour = np.asarray(
                repaired[second_path][1 if second_side == 0 else -2, :2],
                dtype=np.float64,
            )
            second_inward = second_neighbour - second_point
            second_length = float(np.linalg.norm(second_inward))
            if second_length <= endpoint_tolerance:
                continue
            second_inward /= second_length
            gap_direction = gap / gap_length
            if (
                abs(float(np.dot(first_inward, second_inward)))
                < collinear_cosine
                or float(np.dot(first_inward, gap_direction))
                > -collinear_cosine
                or float(np.dot(second_inward, -gap_direction))
                > -collinear_cosine
            ):
                continue
            candidate_pairs.append(
                (gap_length, first_endpoint_index, second_endpoint_index)
            )

    # Resolve the closest independent cap pair first.  An endpoint is moved at
    # most once; any ambiguous multi-neighbour case remains for the strict
    # postcondition to reject rather than being repaired speculatively.
    used_endpoints: set[int] = set()
    safety = max(tolerance * 0.02, target_spacing * 1e-7, 1e-7)
    minimum_remaining_segment = max(tolerance * 2.0, safety * 2.0)
    for gap_length, first_endpoint_index, second_endpoint_index in sorted(
        candidate_pairs
    ):
        if (
            first_endpoint_index in used_endpoints
            or second_endpoint_index in used_endpoints
        ):
            continue
        first_path, first_side = endpoints[first_endpoint_index]
        second_path, second_side = endpoints[second_endpoint_index]
        first_point = np.asarray(
            repaired[first_path][first_side, :2], dtype=np.float64
        )
        second_point = np.asarray(
            repaired[second_path][second_side, :2], dtype=np.float64
        )
        first_neighbour = np.asarray(
            repaired[first_path][1 if first_side == 0 else -2, :2],
            dtype=np.float64,
        )
        second_neighbour = np.asarray(
            repaired[second_path][1 if second_side == 0 else -2, :2],
            dtype=np.float64,
        )
        first_vector = first_neighbour - first_point
        second_vector = second_neighbour - second_point
        first_length = float(np.linalg.norm(first_vector))
        second_length = float(np.linalg.norm(second_vector))
        retraction = (target_spacing - gap_length) * 0.5 + safety
        if (
            retraction >= first_length - minimum_remaining_segment
            or retraction >= second_length - minimum_remaining_segment
        ):
            continue
        repaired[first_path][first_side, :2] = (
            first_point + first_vector / first_length * retraction
        ).astype(np.float32)
        repaired[second_path][second_side, :2] = (
            second_point + second_vector / second_length * retraction
        ).astype(np.float32)
        used_endpoints.update((first_endpoint_index, second_endpoint_index))
    return repaired


def _measured_width_infill_result(
    paths: list[np.ndarray],
    last_perimeter_linework,
    config: SliceConfig,
) -> tuple[list[np.ndarray], bool]:
    """Return conservatively repaired paths and their strict spacing result."""

    if config.planning_line_width is None:
        return paths, True
    repaired = _trim_close_collinear_solid_fill_endcaps(
        paths,
        _resin_planning_path_spacing(config),
        config.tolerance,
    )
    return repaired, _solid_fill_spacing_postcondition(
        repaired,
        last_perimeter_linework,
        _resin_maximum_overlap_spacing(config),
        config.tolerance,
        bead_width=_resin_planning_line_width(config),
        minimum_wall_spacing=_resin_contour_infill_maximum_overlap_spacing(config),
    )


def _require_measured_width_infill(
    paths: list[np.ndarray],
    last_perimeter_linework,
    config: SliceConfig,
) -> list[np.ndarray]:
    repaired, valid = _measured_width_infill_result(
        paths,
        last_perimeter_linework,
        config,
    )
    # Coverage and spacing are optimized in separate stages.  A boundary-gap
    # correction may intentionally consume a small amount of the overlap budget
    # to keep a nominally solid layer closed.  Do not turn that local conflict
    # into a failed export here; retain the conservative end-cap repair and let
    # the later layer-wide optimizer minimize the remaining overlap after it has
    # established the best attainable physical coverage.
    return repaired


def _finish_solid_fill_paths(
    solid_geometry,
    infill_geometry,
    perimeter_paths: list[np.ndarray],
    infill_paths: list[np.ndarray],
    last_perimeter_linework,
    config: SliceConfig,
    *,
    centerline_regions=None,
) -> list[np.ndarray]:
    """Validate final spacing without rounding or splitting continuous paths."""

    return _require_measured_width_infill(
        infill_paths,
        last_perimeter_linework,
        config,
    )


def _reconnect_finished_solid_fill_paths(
    perimeter_paths: list[np.ndarray],
    infill_paths: list[np.ndarray],
    last_perimeter_linework,
    config: SliceConfig,
    *,
    direct_allowed,
) -> list[np.ndarray]:
    """Join compatible finished trails by replacing their short end tails.

    A boundary arc added at the original endpoints can be centerline-clear yet
    still form a sharp, over-deposited hook beside a hole.  Instead, retract a
    fraction of one measured pitch from both incident hatches and replace those
    tails with a tangent cubic return.  Every proposal is checked against the
    complete layer for measured-width spacing, effective corner angle, simple
    topology, and loss of physical bead coverage before it may reduce a stop.
    """

    if (
        config.planning_line_width is None
        or len(infill_paths) < 2
        or direct_allowed is None
        or direct_allowed.is_empty
    ):
        return infill_paths

    planning_width = _resin_planning_line_width(config)
    path_spacing = _resin_planning_path_spacing(config)
    maximum_overlap_spacing = _resin_maximum_overlap_spacing(config)
    tolerance = config.tolerance
    safe_geometry = direct_allowed.buffer(
        max(tolerance * 10.0, 1e-7),
        join_style="round",
    )
    effective_span = max(tolerance * 20.0, config.line_width * 0.005)
    source_paths = [
        np.asarray(path[:, :2], dtype=np.float32).copy()
        for path in infill_paths
        if path.shape[0] >= 2
    ]
    if len(source_paths) < 2:
        return infill_paths

    bead_radius = planning_width * 0.5
    baseline_coverage = _round_bead_coverage(
        [*perimeter_paths, *source_paths],
        bead_radius,
    )
    maximum_lost_diameter = planning_width * 0.60
    # Prefer short asymmetric trims: one side makes room for curvature while
    # the other preserves coverage.  The relaxed coverage guard below permits
    # a small local underfill instead of rejecting every non-identical return.
    trim_factors = (
        (0.1, 0.1),
        (0.1, 0.2),
        (0.2, 0.1),
        (0.1, 0.3),
        (0.3, 0.1),
        (0.2, 0.2),
        (0.1, 0.5),
        (0.5, 0.1),
    )

    def trim_path_end(path: np.ndarray, distance: float) -> np.ndarray | None:
        reversed_trimmed = _trim_open_path_start(
            path[::-1].copy(),
            distance,
            tolerance,
        )
        return (
            None
            if reversed_trimmed is None
            else reversed_trimmed[::-1].copy()
        )

    def unit_direction(vector: np.ndarray) -> np.ndarray | None:
        length = float(np.linalg.norm(vector))
        if length <= tolerance:
            return None
        return np.asarray(vector, dtype=np.float64) / length

    def tangent_cubic(
        start: np.ndarray,
        end: np.ndarray,
        start_tangent: np.ndarray,
        end_tangent: np.ndarray,
    ) -> np.ndarray:
        chord_length = float(np.linalg.norm(end - start))
        handle_length = chord_length * 0.12
        parameters = np.linspace(0.0, 1.0, 33, dtype=np.float64)[:, None]
        one_minus = 1.0 - parameters
        first_control = start + start_tangent * handle_length
        second_control = end - end_tangent * handle_length
        curve = (
            one_minus**3 * start
            + 3.0 * one_minus**2 * parameters * first_control
            + 3.0 * one_minus * parameters**2 * second_control
            + parameters**3 * end
        ).astype(np.float32)
        curve[0] = np.asarray(start, dtype=np.float32)
        curve[-1] = np.asarray(end, dtype=np.float32)
        return curve

    def layer_is_valid(paths: list[np.ndarray]) -> bool:
        return _solid_fill_spacing_postcondition(
            paths,
            last_perimeter_linework,
            maximum_overlap_spacing,
            tolerance,
            bead_width=planning_width,
            minimum_wall_spacing=_resin_contour_infill_maximum_overlap_spacing(config),
        )

    def coverage_is_preserved(paths: list[np.ndarray]) -> bool:
        proposed_coverage = _round_bead_coverage(
            [*perimeter_paths, *paths],
            bead_radius,
        )
        lost_coverage = baseline_coverage.difference(proposed_coverage)
        return (
            lost_coverage.is_empty
            or _maximum_inscribed_diameter(lost_coverage, tolerance)
            <= maximum_lost_diameter + tolerance
        )

    candidates: list[
        tuple[int, int, float, float, np.ndarray]
    ] = []
    minimum_gap = path_spacing * 0.95
    # Complex hole-split Zigzag layers can leave safe component endpoints more
    # than four pitches apart after the local adjacent-scan planner has done
    # all it can. Search farther here; every longer proposal still has to pass
    # the complete safe-region, corner, spacing, and coverage postconditions.
    maximum_gap = path_spacing * 12.0
    for first_index, first_source in enumerate(source_paths):
        for second_index in range(first_index + 1, len(source_paths)):
            second_source = source_paths[second_index]
            for first_side in (0, 1):
                first_oriented = (
                    first_source if first_side == 1 else first_source[::-1].copy()
                )
                for second_side in (0, 1):
                    second_oriented = (
                        second_source
                        if second_side == 0
                        else second_source[::-1].copy()
                    )
                    endpoint_gap = float(
                        np.linalg.norm(
                            second_oriented[0, :2] - first_oriented[-1, :2]
                        )
                    )
                    if not (minimum_gap <= endpoint_gap <= maximum_gap):
                        continue
                    start_tangent = unit_direction(
                        first_oriented[-1, :2] - first_oriented[-2, :2]
                    )
                    end_tangent = unit_direction(
                        second_oriented[1, :2] - second_oriented[0, :2]
                    )
                    if (
                        start_tangent is None
                        or end_tangent is None
                    ):
                        continue

                    for first_factor, second_factor in trim_factors:
                        first_trimmed = trim_path_end(
                            first_oriented,
                            path_spacing * first_factor,
                        )
                        second_trimmed = _trim_open_path_start(
                            second_oriented,
                            path_spacing * second_factor,
                            tolerance,
                        )
                        if first_trimmed is None or second_trimmed is None:
                            continue
                        start_tangent = unit_direction(
                            first_trimmed[-1, :2] - first_trimmed[-2, :2]
                        )
                        end_tangent = unit_direction(
                            second_trimmed[1, :2] - second_trimmed[0, :2]
                        )
                        if start_tangent is None or end_tangent is None:
                            continue
                        connector = tangent_cubic(
                            np.asarray(first_trimmed[-1, :2], dtype=np.float64),
                            np.asarray(second_trimmed[0, :2], dtype=np.float64),
                            start_tangent,
                            end_tangent,
                        )
                        chain = _dedupe_consecutive(
                            np.vstack(
                                (
                                    first_trimmed,
                                    connector[1:],
                                    second_trimmed[1:],
                                )
                            ).astype(np.float32),
                            tolerance,
                        )
                        chain_line = LineString(chain[:, :2])
                        if (
                            chain.shape[0] < 2
                            or not chain_line.is_simple
                            or not safe_geometry.covers(chain_line)
                        ):
                            continue
                        trim_total = path_spacing * (
                            first_factor + second_factor
                        )
                        candidates.append(
                            (
                                first_index,
                                second_index,
                                trim_total,
                                endpoint_gap,
                                chain,
                            )
                        )

    if not candidates:
        return source_paths

    selected: dict[int, tuple[int, np.ndarray]] = {}
    consumed: set[int] = set()

    def assembled_paths(
        extra: tuple[int, int, np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        local_selected = dict(selected)
        local_consumed = set(consumed)
        if extra is not None:
            first_index, second_index, chain = extra
            local_selected[first_index] = (second_index, chain)
            local_consumed.update((first_index, second_index))
        assembled: list[np.ndarray] = []
        for path_index, path in enumerate(source_paths):
            selection = local_selected.get(path_index)
            if selection is not None:
                assembled.append(selection[1])
            elif path_index not in local_consumed:
                assembled.append(path)
        return assembled

    # Each accepted pair returns as a new trail on the next pass, so it may be
    # extended again until no validated connection remains.  Complete-layer
    # checks prevent individually safe returns from conflicting.
    maximum_layer_trials = max(8, len(source_paths) * 2)
    layer_trials = 0
    for first_index, second_index, trim_total, endpoint_gap, chain in sorted(
        candidates,
        key=lambda item: (item[2], item[3], item[0], item[1]),
    ):
        if first_index in consumed or second_index in consumed:
            continue
        if layer_trials >= maximum_layer_trials:
            break
        proposed = assembled_paths((first_index, second_index, chain))
        layer_trials += 1
        if not layer_is_valid(proposed) or not coverage_is_preserved(proposed):
            continue
        selected[first_index] = (second_index, chain)
        consumed.update((first_index, second_index))

    reconnected = assembled_paths()
    return reconnected if len(reconnected) < len(source_paths) else source_paths


def _reroute_residual_solid_bead_gaps(
    solid_geometry,
    infill_geometry,
    perimeter_paths: list[np.ndarray],
    infill_paths: list[np.ndarray],
    last_perimeter_linework,
    line_width: float,
    path_spacing: float,
    tolerance: float,
    *,
    centerline_regions=None,
    minimum_wall_clearance: float | None = None,
) -> list[np.ndarray]:
    """Fill visible fixed-width residuals by detouring existing solid paths.

    Narrow necks between a hole and a concave wall can disappear from the
    normally inset infill surface even though the two 2 mm perimeter beads do
    not quite meet.  Adding another standalone stroke would create a stop and a
    start blob.  Instead, this routine replaces at most a short interval of an
    existing infill trail with a triangular visit through the uncovered pocket.
    Path count and centerline continuity are preserved.
    """

    if (
        solid_geometry.is_empty
        or infill_geometry.is_empty
        or not infill_paths
        or line_width <= 0
        or path_spacing <= 0
    ):
        return infill_paths

    regions = centerline_regions
    if regions is None:
        regions = _solid_residual_centerline_regions(
            solid_geometry,
            infill_geometry,
            perimeter_paths,
            last_perimeter_linework,
            line_width,
            path_spacing,
            tolerance,
            wall_clearance=minimum_wall_clearance,
        )
    if regions is None:
        return infill_paths
    (
        spacing_adjustment,
        safe_surface_allowed,
        direct_allowed,
        minimum_wall_clearance,
        wall_linework,
    ) = regions
    bead_radius = line_width * 0.5
    evaluation_region = solid_geometry.buffer(
        -line_width * 0.15,
        join_style="mitre",
    )
    if direct_allowed.is_empty or evaluation_region.is_empty:
        return infill_paths

    # These centerline regions are immutable during the correction search.
    # Preparing them turns the thousands of repeated ``covers`` predicates
    # into indexed lookups without changing any geometric decision.
    prepare(direct_allowed)
    prepare(safe_surface_allowed)

    paths = [np.asarray(path[:, :2], dtype=np.float32).copy() for path in infill_paths]
    original_path_corridors = [
        LineString(path).buffer(
            max(tolerance * 2.0, 1e-7),
            cap_style="round",
            join_style="round",
        )
        for path in paths
    ]
    for corridor in original_path_corridors:
        prepare(corridor)
    original_path_endpoints = [
        (
            np.asarray(path[0, :2], dtype=np.float64),
            np.asarray(path[-1, :2], dtype=np.float64),
        )
        for path in paths
    ]
    fixed_paths = [
        np.asarray(path[:, :2], dtype=np.float32)
        for path in perimeter_paths
        if path.shape[0] >= 2
    ]

    def build_path_search_cache(path_index: int):
        path = paths[path_index]
        path_line = LineString(path)
        searchable_segments: list[tuple[LineString, float, float]] = []
        cumulative_start = 0.0
        for segment_start, segment_end in zip(path[:-1], path[1:]):
            segment = LineString([segment_start[:2], segment_end[:2]])
            segment_length = float(segment.length)
            segment_offset = cumulative_start
            cumulative_start += segment_length
            if (
                segment_length > tolerance
                and original_path_corridors[path_index].covers(segment)
            ):
                searchable_segments.append(
                    (segment, segment_offset, segment_length)
                )
        return path_line, float(path_line.length), searchable_segments

    path_search_cache = [build_path_search_cache(index) for index in range(len(paths))]
    fixed_path_beads = [_round_bead_for_path(path, bead_radius) for path in fixed_paths]
    path_beads = [_round_bead_for_path(path, bead_radius) for path in paths]
    original_total_length = sum(_open_path_length(path) for path in paths)
    added_length_budget = max(line_width * 2.0, original_total_length * 0.015)
    maximum_corrections = 20
    # A half-bead residual is still clearly visible in a nominally solid
    # layer.  Drive corrections down to 40% of the physical bead width while
    # retaining the dose/added-length guards below, so tighter coverage cannot
    # be bought by stacking a second 2 mm stroke onto an existing one.
    target_void_diameter = line_width * 0.4
    minimum_component_area = line_width * line_width * 0.0075
    mic_tolerance = max(tolerance * 20.0, line_width * 0.005)
    blocked_centers: set[tuple[int, int]] = set()
    added_length = 0.0

    for _ in range(maximum_corrections):
        coverage = unary_union(fixed_path_beads + path_beads)
        uncovered = evaluation_region.difference(coverage)
        candidates: list[tuple[float, object, np.ndarray, tuple[int, int]]] = []
        for component in _iter_polygons(uncovered):
            if component.area < minimum_component_area:
                continue
            circle = maximum_inscribed_circle(component, tolerance=mic_tolerance)
            if circle.is_empty:
                continue
            diameter = float(circle.length * 2.0)
            if diameter <= target_void_diameter:
                continue
            center = np.asarray(circle.coords[0], dtype=np.float64)
            center_key = (
                round(float(center[0]) / mic_tolerance),
                round(float(center[1]) / mic_tolerance),
            )
            if center_key not in blocked_centers:
                candidates.append((diameter, component, center, center_key))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0], reverse=True)

        changed = False
        for _, component, center, center_key in candidates:
            center_point = Point(float(center[0]), float(center[1]))
            apex_candidates: list[np.ndarray] = []
            for candidate_region in (safe_surface_allowed, direct_allowed):
                if candidate_region.is_empty:
                    continue
                candidate_point = (
                    center_point
                    if candidate_region.covers(center_point)
                    else nearest_points(center_point, candidate_region)[1]
                )
                candidate = np.asarray(candidate_point.coords[0], dtype=np.float64)
                if float(np.linalg.norm(candidate - center)) >= bead_radius - tolerance:
                    continue
                if not any(
                    float(np.linalg.norm(candidate - existing)) <= tolerance * 10.0
                    for existing in apex_candidates
                ):
                    apex_candidates.append(candidate)
            if not apex_candidates:
                blocked_centers.add(center_key)
                continue

            best_proposal: tuple[
                tuple[float, float, float, int, float],
                int,
                np.ndarray,
                float,
            ] | None = None
            for apex in apex_candidates:
                options: list[tuple[float, int, float, float, float]] = []
                apex_geometry = Point(float(apex[0]), float(apex[1]))
                for path_index, path in enumerate(paths):
                    path_line, path_length, searchable_segments = path_search_cache[
                        path_index
                    ]
                    for segment, segment_offset, segment_length in searchable_segments:
                        distance = float(segment.distance(apex_geometry))
                        local_projection = float(segment.project(apex_geometry))
                        left = min(bead_radius, local_projection)
                        right = min(bead_radius, segment_length - local_projection)
                        if (
                            tolerance * 10.0 < distance < line_width * 1.5
                            and left + right >= bead_radius * 0.55
                            and min(left, right) > tolerance * 10.0
                        ):
                            options.append(
                                (
                                    distance,
                                    path_index,
                                    segment_offset,
                                    local_projection,
                                    segment_length,
                                )
                            )
                    # A nearest point can be an interior smoothing vertex.  In
                    # that case each incident micro-segment has zero usable
                    # distance on one side even though the continuous path has
                    # ample room for a safe detour.  Add a path-distance option
                    # that may straddle original vertices; the immutable
                    # corridor and retained-bead checks below keep it from
                    # spanning a previously inserted correction.
                    path_projection = float(path_line.project(apex_geometry))
                    path_distance = float(path_line.distance(apex_geometry))
                    path_left = min(bead_radius, path_projection)
                    path_right = min(bead_radius, path_length - path_projection)
                    projection_point = path_line.interpolate(path_projection)
                    if (
                        tolerance * 10.0
                        < path_distance
                        < line_width * 1.5
                        and path_left + path_right >= bead_radius * 0.55
                        and min(path_left, path_right) > tolerance * 10.0
                        and original_path_corridors[path_index].covers(projection_point)
                    ):
                        options.append(
                            (
                                path_distance,
                                path_index,
                                0.0,
                                path_projection,
                                path_length,
                            )
                        )

                    # If the nearest printable point lies beyond a genuine
                    # free end, a two-sided triangular replacement is
                    # impossible.  Extend that same continuous trail once,
                    # provided the short tail neither doubles back nor touches
                    # another trail.  This covers narrow-neck pockets without
                    # introducing a separate start/stop or a retraced stroke.
                    for endpoint_side, endpoint_index in ((0, 0), (1, -1)):
                        endpoint_projection_distance = (
                            path_projection
                            if endpoint_side == 0
                            else path_length - path_projection
                        )
                        if endpoint_projection_distance > tolerance * 10.0:
                            continue
                        endpoint = np.asarray(path[endpoint_index, :2], dtype=np.float64)
                        if (
                            float(
                                np.linalg.norm(
                                    endpoint
                                    - original_path_endpoints[path_index][endpoint_side]
                                )
                            )
                            > tolerance * 10.0
                        ):
                            continue
                        tail_line = LineString([endpoint, apex])
                        tail_length = float(tail_line.length)
                        if not (tolerance * 10.0 < tail_length <= line_width):
                            continue
                        if not direct_allowed.covers(tail_line):
                            continue
                        if (
                            not wall_linework.is_empty
                            and tail_line.distance(wall_linework)
                            < minimum_wall_clearance - tolerance * 5.0
                        ):
                            continue
                        if center_point.distance(tail_line) >= bead_radius - tolerance:
                            continue
                        old_line = path_line
                        if _has_unexpected_linework_intersection(
                            tail_line,
                            old_line,
                            (endpoint,),
                            tolerance,
                        ):
                            continue
                        if any(
                            tail_line.distance(path_search_cache[other_index][0])
                            <= tolerance * 2.0
                            for other_index in range(len(paths))
                            if other_index != path_index
                        ):
                            continue
                        existing_ray = np.asarray(
                            (
                                path[1, :2] - path[0, :2]
                                if endpoint_side == 0
                                else path[-2, :2] - path[-1, :2]
                            ),
                            dtype=np.float64,
                        )
                        tail_ray = np.asarray(apex - endpoint, dtype=np.float64)
                        if (
                            _acute_angle_degrees(
                                existing_ray,
                                tail_ray,
                                tolerance,
                            )
                            < 15.0
                        ):
                            continue
                        new_path = _dedupe_consecutive(
                            (
                                np.vstack((apex, path))
                                if endpoint_side == 0
                                else np.vstack((path, apex))
                            ).astype(np.float32),
                            tolerance * 2.0,
                        )
                        new_line = LineString(new_path)
                        if not new_line.is_simple:
                            continue
                        tail_bead = tail_line.buffer(
                            bead_radius,
                            cap_style="round",
                            join_style="round",
                        )
                        component_gain = component.intersection(tail_bead).area
                        if component_gain < component.area * 0.25:
                            continue
                        if not _residual_correction_has_sufficient_novel_area(
                            component_gain,
                            tail_length,
                            line_width,
                        ):
                            continue
                        if added_length + tail_length > added_length_budget:
                            continue
                        remaining_diameter = _maximum_inscribed_diameter(
                            component.difference(tail_bead),
                            mic_tolerance,
                        )
                        score = (
                            remaining_diameter,
                            -float(component_gain),
                            tail_length,
                            path_index,
                            -1.0 if endpoint_side == 0 else path_length,
                        )
                        if best_proposal is None or score < best_proposal[0]:
                            best_proposal = (
                                score,
                                path_index,
                                new_path,
                                tail_length,
                            )
                options.sort(key=lambda item: item[0])

                for (
                    _,
                    path_index,
                    cumulative_start,
                    local_projection,
                    segment_length,
                ) in options:
                    old_path = paths[path_index]
                    old_line = path_search_cache[path_index][0]
                    for interval_factor in (1.0, 0.75, 0.5):
                        left = min(
                            bead_radius * interval_factor,
                            local_projection,
                        )
                        right = min(
                            bead_radius * interval_factor,
                            segment_length - local_projection,
                        )
                        if (
                            left + right < bead_radius * 0.55
                            or min(left, right) <= tolerance * 10.0
                        ):
                            continue
                        proposal = _replace_path_interval_with_detour(
                            old_path,
                            cumulative_start + local_projection - left,
                            cumulative_start + local_projection + right,
                            apex,
                            tolerance,
                        )
                        if proposal is None:
                            continue
                        new_path, original_interval, detour = proposal
                        if not original_path_corridors[path_index].covers(original_interval):
                            continue
                        detour_line = LineString(detour)
                        if not direct_allowed.covers(detour_line):
                            continue
                        if (
                            not wall_linework.is_empty
                            and detour_line.distance(wall_linework)
                            < minimum_wall_clearance - tolerance * 5.0
                        ):
                            continue
                        if center_point.distance(detour_line) >= bead_radius - tolerance:
                            continue
                        base_vector = np.asarray(
                            detour[-1] - detour[0], dtype=np.float64
                        )
                        first_leg = np.asarray(
                            detour[1] - detour[0], dtype=np.float64
                        )
                        second_leg = np.asarray(
                            detour[-1] - detour[1], dtype=np.float64
                        )
                        if (
                            _acute_angle_degrees(base_vector, first_leg, tolerance) < 15.0
                            or _acute_angle_degrees(base_vector, second_leg, tolerance) < 15.0
                            or _acute_angle_degrees(first_leg, second_leg, tolerance) < 15.0
                        ):
                            continue

                        new_line = LineString(new_path)
                        if old_line.is_simple and not new_line.is_simple:
                            continue
                        if any(
                            detour_line.distance(path_search_cache[other_index][0])
                            <= tolerance * 2.0
                            for other_index in range(len(paths))
                            if other_index != path_index
                        ):
                            continue
                        length_increase = float(new_line.length - old_line.length)
                        if (
                            length_increase <= tolerance
                            or length_increase > line_width * 2.0
                            or added_length + length_increase > added_length_budget
                        ):
                            continue

                        retained_bead = detour_line.buffer(
                            bead_radius * 0.95,
                            cap_style="round",
                            join_style="round",
                        )
                        if (
                            original_interval.difference(retained_bead).length
                            > tolerance * 10.0
                        ):
                            continue
                        detour_bead = detour_line.buffer(
                            bead_radius,
                            cap_style="round",
                            join_style="round",
                        )
                        component_gain = component.intersection(detour_bead).area
                        if component_gain < component.area * 0.25:
                            continue
                        if not _residual_correction_has_sufficient_novel_area(
                            component_gain,
                            length_increase,
                            line_width,
                        ):
                            continue

                        remaining_diameter = _maximum_inscribed_diameter(
                            component.difference(detour_bead),
                            mic_tolerance,
                        )
                        score = (
                            remaining_diameter,
                            -float(component_gain),
                            length_increase,
                            path_index,
                            cumulative_start + local_projection,
                        )
                        if best_proposal is None or score < best_proposal[0]:
                            best_proposal = (
                                score,
                                path_index,
                                new_path,
                                length_increase,
                            )
            if best_proposal is not None:
                _, path_index, new_path, length_increase = best_proposal
                paths[path_index] = new_path
                path_search_cache[path_index] = build_path_search_cache(path_index)
                path_beads[path_index] = _round_bead_for_path(new_path, bead_radius)
                added_length += length_increase
                changed = True
                break
            blocked_centers.add(center_key)
        if not changed:
            break
    return paths


def _round_bead_for_path(path: np.ndarray, bead_radius: float):
    return LineString(path[:, :2]).buffer(
        bead_radius,
        cap_style="round",
        join_style="round",
        quad_segs=8,
    )


def _round_bead_coverage(paths: list[np.ndarray], bead_radius: float):
    beads = [
        _round_bead_for_path(path, bead_radius)
        for path in paths
        if path.shape[0] >= 2
    ]
    return unary_union(beads) if beads else GeometryCollection()


def _residual_correction_has_sufficient_novel_area(
    novel_area: float,
    added_length: float,
    line_width: float,
) -> bool:
    """Reject a correction that would deposit mostly onto existing 2 mm beads."""

    if added_length <= 0 or line_width <= 0:
        return False
    minimum_novel_area = (
        added_length
        * line_width
        * MINIMUM_RESIDUAL_CORRECTION_NOVEL_AREA_FRACTION
    )
    return novel_area >= minimum_novel_area


def _maximum_inscribed_diameter(geometry, tolerance: float) -> float:
    maximum = 0.0
    for polygon in _iter_polygons(geometry):
        if polygon.area <= tolerance * tolerance:
            continue
        circle = maximum_inscribed_circle(polygon, tolerance=tolerance)
        if not circle.is_empty:
            maximum = max(maximum, float(circle.length * 2.0))
    return maximum


def _full_density_coverage_metrics(
    solid_geometry,
    perimeter_paths: list[np.ndarray],
    infill_paths: list[np.ndarray],
    line_width: float,
    tolerance: float,
):
    """Measure physical uncovered space independently from path-spacing rules."""

    evaluation_region = solid_geometry.buffer(
        -line_width * 0.15,
        join_style="mitre",
    )
    if evaluation_region.is_empty:
        return 0.0, 0.0, GeometryCollection(), GeometryCollection()
    coverage = _round_bead_coverage(
        [*perimeter_paths, *infill_paths],
        line_width * 0.5,
    )
    uncovered = evaluation_region.difference(coverage)
    return (
        _maximum_inscribed_diameter(
            uncovered,
            max(tolerance * 20.0, line_width * 0.005),
        ),
        float(uncovered.area),
        uncovered,
        coverage,
    )


def _coverage_repair_line_candidates(
    solid_geometry,
    infill_geometry,
    uncovered,
    direct_allowed,
    existing_linework,
    line_width: float,
    target_void_diameter: float,
    tolerance: float,
) -> list[np.ndarray]:
    """Build coverage candidates without coupling them to continuity decisions."""

    bead_radius = line_width * 0.5
    candidates: list[np.ndarray] = []
    seen: set[bytes] = set()

    def append_segment(segment: LineString) -> None:
        if segment.length <= max(tolerance * 20.0, line_width * 2.0):
            return
        if (
            not existing_linework.is_empty
            and segment.difference(
                existing_linework.buffer(max(tolerance * 2.0, 1e-7))
            ).length
            <= tolerance
        ):
            return
        path = _dedupe_consecutive(
            np.asarray(segment.coords, dtype=np.float32),
            tolerance,
        )
        if path.shape[0] < 2 or not LineString(path).is_simple:
            return
        key = np.round(
            path / max(tolerance * 20.0, line_width * 1e-4)
        ).astype(np.int64).tobytes()
        if key not in seen:
            seen.add(key)
            candidates.append(path)

    # If the strict seam corridor cannot reach a remaining pocket, coverage has
    # priority.  Search the complete physical centerline region locally and let
    # the caller's overlap score select the least depositing correction.
    physical_centerlines = solid_geometry.buffer(
        -(bead_radius - max(tolerance * 2.0, 1e-7)),
        join_style="round",
    )
    allowed_regions = [
        region
        for region in (direct_allowed, physical_centerlines)
        if region is not None and not region.is_empty
    ]
    widest_components: list[tuple[float, object, np.ndarray]] = []
    mic_tolerance = max(tolerance * 20.0, line_width * 0.005)
    for component in _iter_polygons(uncovered):
        circle = maximum_inscribed_circle(component, tolerance=mic_tolerance)
        if circle.is_empty or len(circle.coords) < 2:
            continue
        diameter = float(circle.length * 2.0)
        if diameter <= target_void_diameter + tolerance:
            continue
        widest_components.append(
            (
                diameter,
                component,
                np.asarray(circle.coords[0], dtype=np.float64),
            )
        )
    widest_components.sort(key=lambda item: item[0], reverse=True)

    # Use short seam arcs centered on the actual widest pockets.  Buffering all
    # uncovered slivers at once can join a chain of scallops into most of a
    # contour ring, which covers well but unnecessarily reprints every hatch
    # cap it passes.  Local windows keep the same coverage reach with far less
    # repeated dose and shorter independent strokes.
    boundary_reach = max(
        tolerance * 20.0,
        bead_radius - target_void_diameter * 0.5,
    )
    for _, component, center in widest_components[:4]:
        local_window = Point(float(center[0]), float(center[1])).buffer(
            line_width * 1.5,
            join_style="round",
        )
        boundary_linework = infill_geometry.boundary.intersection(
            component.buffer(boundary_reach, join_style="round").intersection(
                local_window
            )
        )
        for segment in _extract_line_segments(boundary_linework, tolerance):
            append_segment(segment)

    for diameter, component, center in widest_components[:4]:
        center_point = Point(float(center[0]), float(center[1]))
        for allowed in allowed_regions:
            anchor_geometry = (
                center_point
                if allowed.covers(center_point)
                else nearest_points(center_point, allowed)[1]
            )
            anchor = np.asarray(anchor_geometry.coords[0], dtype=np.float64)
            if float(np.linalg.norm(anchor - center)) >= bead_radius - tolerance:
                continue
            probe_half_length = min(
                line_width * 2.0,
                max(line_width * 0.35, diameter),
            )
            local_region = allowed.intersection(
                component.buffer(bead_radius * 0.95, join_style="round")
            )
            for angle_degrees in range(0, 180, 30):
                angle = math.radians(float(angle_degrees))
                direction = np.asarray(
                    [math.cos(angle), math.sin(angle)],
                    dtype=np.float64,
                )
                probe = LineString(
                    [
                        anchor - direction * probe_half_length,
                        anchor + direction * probe_half_length,
                    ]
                )
                segments = list(
                    _extract_line_segments(local_region.intersection(probe), tolerance)
                )
                if not segments:
                    continue
                segment = min(
                    segments,
                    key=lambda item: (
                        float(item.distance(anchor_geometry)),
                        -float(item.length),
                    ),
                )
                append_segment(segment)
            # Strict and physical regions share the same anchor in the common
            # case; avoid manufacturing duplicate direction candidates.
            if allowed.covers(center_point):
                break
    candidates.sort(
        key=lambda path: (
            -float(
                uncovered.intersection(
                    _round_bead_for_path(path, bead_radius)
                ).area
            ),
            float(LineString(path).length),
        )
    )
    return candidates[:24]


def _optimize_full_density_coverage(
    solid_geometry,
    infill_geometry,
    perimeter_paths: list[np.ndarray],
    infill_paths: list[np.ndarray],
    line_width: float,
    tolerance: float,
    *,
    direct_allowed=None,
) -> list[np.ndarray]:
    """Greedily reach a coverage-local optimum, then minimize added overlap.

    Coverage is the primary objective.  Candidate overlap and added path length
    are compared only after maximum void diameter and uncovered area, so an
    impossible local spacing conflict never turns into either a failed export
    or a deliberately retained hole.
    """

    if (
        solid_geometry.is_empty
        or infill_geometry.is_empty
        or line_width <= 0
        or not infill_paths
    ):
        return infill_paths

    target_void_diameter = max(
        tolerance * 40.0,
        line_width * FULL_DENSITY_TARGET_UNCOVERED_DIAMETER_FACTOR,
    )
    paths = [np.asarray(path[:, :2], dtype=np.float32).copy() for path in infill_paths]
    minimum_area_improvement = max(tolerance * tolerance * 10.0, line_width**2 * 1e-6)
    contour_lines = [
        LineString(path[:, :2])
        for path in perimeter_paths
        if path.shape[0] >= 2
    ]
    contour_exclusion = (
        unary_union(contour_lines).buffer(
            max(tolerance * 10.0, 1e-7),
            cap_style="round",
            join_style="round",
        )
        if contour_lines
        else GeometryCollection()
    )
    # Coverage repair must not undo the continuity stage by manufacturing a
    # forest of short standalone strokes.  At most one local correction per
    # already continuous trail is available; coverage chooses where those
    # scarce corrections go, and overlap/length break ties.
    maximum_correction_paths = min(3, max(2, len(infill_paths)))

    correction_paths: list[np.ndarray] = []
    for _ in range(MAX_FULL_DENSITY_COVERAGE_REPAIR_PASSES):
        if len(correction_paths) >= maximum_correction_paths:
            break
        current_diameter, current_area, uncovered, coverage = (
            _full_density_coverage_metrics(
                solid_geometry,
                perimeter_paths,
                paths,
                line_width,
                tolerance,
            )
        )
        if current_diameter <= target_void_diameter + tolerance:
            break
        existing_linework = unary_union(
            [LineString(path[:, :2]) for path in paths if path.shape[0] >= 2]
        )
        candidates = _coverage_repair_line_candidates(
            solid_geometry,
            infill_geometry,
            uncovered,
            direct_allowed,
            existing_linework,
            line_width,
            target_void_diameter,
            tolerance,
        )
        ranked_candidates: list[
            tuple[tuple[float, float, float, float], np.ndarray, object]
        ] = []
        for candidate in candidates:
            candidate_line = LineString(candidate[:, :2])
            if (
                not contour_exclusion.is_empty
                and candidate_line.intersects(contour_exclusion)
            ):
                continue
            candidate_bead = _round_bead_for_path(candidate, line_width * 0.5)
            novel_area = float(uncovered.intersection(candidate_bead).area)
            if novel_area <= minimum_area_improvement:
                continue
            remaining_uncovered = uncovered.difference(candidate_bead)
            remaining_diameter = _maximum_inscribed_diameter(
                remaining_uncovered,
                max(tolerance * 20.0, line_width * 0.005),
            )
            remaining_area = float(remaining_uncovered.area)
            overlap_area = float(candidate_bead.intersection(coverage).area)
            # Coverage is lexicographically primary. Quantizing changes below
            # visual resolution lets two practically equal repairs be decided
            # by deposited overlap and then by motion length.
            diameter_resolution = max(tolerance * 20.0, line_width * 0.02)
            area_resolution = max(
                tolerance * tolerance * 20.0,
                line_width * line_width * 0.02,
            )
            ranked_candidates.append(
                (
                    (
                        round(remaining_diameter / diameter_resolution),
                        round(remaining_area / area_resolution),
                        overlap_area,
                        float(candidate_line.length),
                    ),
                    candidate,
                    candidate_bead,
                )
            )
        accepted_batch = (
            [min(ranked_candidates, key=lambda item: item[0])[1]]
            if ranked_candidates
            else []
        )
        if not accepted_batch:
            break
        proposed_paths = [*paths, *accepted_batch]
        proposed_diameter, proposed_area, _, _ = _full_density_coverage_metrics(
            solid_geometry,
            perimeter_paths,
            proposed_paths,
            line_width,
            tolerance,
        )
        diameter_improved = proposed_diameter < current_diameter - tolerance * 2.0
        area_improved = proposed_area < current_area - minimum_area_improvement
        if not diameter_improved and not area_improved:
            break
        paths = proposed_paths
        correction_paths.extend(accepted_batch)

    if len(correction_paths) < 2:
        return paths
    base_paths = paths[: len(paths) - len(correction_paths)]
    correction_linework = unary_union(
        [LineString(path[:, :2]) for path in correction_paths]
    )
    merged_linework = (
        correction_linework
        if correction_linework.geom_type == "LineString"
        else linemerge(correction_linework)
    )
    merged_corrections = [
        _dedupe_consecutive(
            np.asarray(segment.coords, dtype=np.float32),
            tolerance,
        )
        for segment in _extract_line_segments(merged_linework, tolerance)
    ]
    return base_paths + [path for path in merged_corrections if path.shape[0] >= 2]


def _maximize_full_density_continuity(
    solid_geometry,
    infill_geometry,
    perimeter_paths: list[np.ndarray],
    infill_paths: list[np.ndarray],
    config: SliceConfig,
    *,
    last_perimeter_linework,
    allow_overlap_relaxation: bool,
    enable_detour_absorption: bool = True,
) -> list[np.ndarray]:
    """Join every safely connectable final trail without reopening coverage.

    Strict scan spacing has already been planned and coverage repair has already
    selected the minimum useful seam additions.  This final stage therefore
    treats path count as its primary objective.  A boundary connection is
    rejected only when it leaves the physical bead-center region, creates a
    self intersection or sharp effective corner, crosses a third trail, or
    measurably removes coverage.  It does not reapply the global maximum-
    overlap predicate that caused otherwise printable seam links to be dropped.
    """

    if (
        config.planning_line_width is None
        or len(infill_paths) < 2
        or solid_geometry.is_empty
        or infill_geometry.is_empty
    ):
        return infill_paths

    line_width = _resin_planning_line_width(config)
    tolerance = config.tolerance
    bead_radius = line_width * 0.5
    numerical_allowance = max(tolerance * 10.0, 1e-7)
    seam_geometry = infill_geometry.buffer(
        numerical_allowance,
        join_style="round",
    )
    physical_centerlines = solid_geometry.buffer(
        -(bead_radius - max(tolerance * 2.0, 1e-7)),
        join_style="round",
    ).buffer(numerical_allowance, join_style="round")
    if physical_centerlines.is_empty:
        return infill_paths
    boundary_rings = _infill_boundary_rings(infill_geometry, tolerance)
    if not boundary_rings:
        return infill_paths

    paths = [
        np.asarray(path[:, :2], dtype=np.float32).copy()
        for path in infill_paths
        if path.shape[0] >= 2
    ]
    contour_lines = [
        LineString(path[:, :2])
        for path in perimeter_paths
        if path.shape[0] >= 2
    ]
    if (
        last_perimeter_linework is not None
        and not last_perimeter_linework.is_empty
    ):
        contour_lines.append(last_perimeter_linework)
    contour_exclusion = (
        unary_union(contour_lines).buffer(
            numerical_allowance,
            cap_style="round",
            join_style="round",
        )
        if contour_lines
        else GeometryCollection()
    )
    maximum_connector_length = line_width * 8.0
    maximum_third_path_overlap = max(
        tolerance * tolerance * 20.0,
        line_width * line_width * 0.01,
    )
    target_void_diameter = max(
        tolerance * 40.0,
        line_width * FULL_DENSITY_TARGET_UNCOVERED_DIAMETER_FACTOR,
    )
    area_allowance = max(
        tolerance * tolerance * 20.0,
        line_width * line_width * 0.001,
    )
    cubic_parameters = np.linspace(
        0.0,
        1.0,
        33,
        dtype=np.float64,
    )[:, None]
    cubic_one_minus = 1.0 - cubic_parameters

    def tangent_cubic_connector(
        first_oriented: np.ndarray,
        second_oriented: np.ndarray,
    ) -> np.ndarray | None:
        start_tangent = np.asarray(
            first_oriented[-1, :2] - first_oriented[-2, :2],
            dtype=np.float64,
        )
        end_tangent = np.asarray(
            second_oriented[1, :2] - second_oriented[0, :2],
            dtype=np.float64,
        )
        start_tangent_length = float(np.linalg.norm(start_tangent))
        end_tangent_length = float(np.linalg.norm(end_tangent))
        start = np.asarray(first_oriented[-1, :2], dtype=np.float64)
        end = np.asarray(second_oriented[0, :2], dtype=np.float64)
        chord_length = float(np.linalg.norm(end - start))
        if min(
            start_tangent_length,
            end_tangent_length,
            chord_length,
        ) <= tolerance:
            return None
        start_tangent /= start_tangent_length
        end_tangent /= end_tangent_length
        chord_direction = (end - start) / chord_length
        chord_normal = np.asarray(
            [-chord_direction[1], chord_direction[0]],
            dtype=np.float64,
        )
        normal_bump = (
            16.0
            * cubic_parameters**2
            * cubic_one_minus**2
        )
        for handle_factor in (0.12, 0.2, 0.3, 0.4):
            handle_length = chord_length * handle_factor
            first_control = start + start_tangent * handle_length
            second_control = end - end_tangent * handle_length
            base_connector = (
                cubic_one_minus**3 * start
                + 3.0
                * cubic_one_minus**2
                * cubic_parameters
                * first_control
                + 3.0
                * cubic_one_minus
                * cubic_parameters**2
                * second_control
                + cubic_parameters**3 * end
            )
            for normal_factor in (0.0,):
                tangent_connector = (
                    base_connector
                    + normal_bump
                    * chord_normal
                    * line_width
                    * normal_factor
                ).astype(np.float32)
                tangent_connector[0] = start.astype(np.float32)
                tangent_connector[-1] = end.astype(np.float32)
                tangent_line = LineString(tangent_connector)
                tangent_chain = _dedupe_consecutive(
                    np.vstack(
                        (
                            first_oriented,
                            tangent_connector[1:],
                            second_oriented[1:],
                        )
                    ).astype(np.float32),
                    tolerance,
                )
                if (
                    LineString(tangent_chain).is_simple
                    and physical_centerlines.covers(tangent_line)
                    and (
                        contour_exclusion.is_empty
                        or not tangent_line.intersects(contour_exclusion)
                    )
                ):
                    return tangent_connector
        return None

    def projected_boundary_connector(
        first_point: np.ndarray,
        second_point: np.ndarray,
    ) -> np.ndarray | None:
        """Follow a nearby fill boundary when exact endpoint snapping fails."""

        first = Point(float(first_point[0]), float(first_point[1]))
        second = Point(float(second_point[0]), float(second_point[1]))
        maximum_snap_distance = line_width
        candidates: list[np.ndarray] = []
        for ring in boundary_rings:
            if (
                ring.line.distance(first) > maximum_snap_distance
                or ring.line.distance(second) > maximum_snap_distance
            ):
                continue
            candidate = _shortest_infill_ring_arc(
                ring,
                first_point,
                second_point,
                tolerance,
            )
            candidate_line = LineString(candidate[:, :2])
            if (
                candidate_line.length <= maximum_connector_length
                and physical_centerlines.covers(candidate_line)
                and (
                    contour_exclusion.is_empty
                    or not candidate_line.intersects(contour_exclusion)
                )
            ):
                candidates.append(candidate)
        return (
            min(candidates, key=_open_path_length)
            if candidates
            else None
        )

    while len(paths) >= 2:
        baseline_coverage = _round_bead_coverage(
            [*perimeter_paths, *paths],
            bead_radius,
        )
        baseline_diameter, baseline_uncovered_area, _, _ = (
            _full_density_coverage_metrics(
                solid_geometry,
                perimeter_paths,
                paths,
                line_width,
                tolerance,
            )
        )
        proposals: list[
            tuple[tuple[float, float, float, int, int], int, int, np.ndarray]
        ] = []
        detour_proposals: list[
            tuple[
                tuple[float, float, float, int, int],
                int,
                int,
                list[np.ndarray],
            ]
        ] = []
        for first_index, first_path in enumerate(paths):
            for second_index in range(first_index + 1, len(paths)):
                second_path = paths[second_index]
                third_path_coverage = _round_bead_coverage(
                    [
                        path
                        for path_index, path in enumerate(paths)
                        if path_index not in (first_index, second_index)
                    ],
                    bead_radius,
                )
                for first_side in (0, 1):
                    first_endpoint = first_path[0 if first_side == 0 else -1, :2]
                    first_oriented = (
                        first_path[::-1].copy()
                        if first_side == 0
                        else first_path
                    )
                    for second_side in (0, 1):
                        second_endpoint = second_path[
                            0 if second_side == 0 else -1,
                            :2,
                        ]
                        second_oriented = (
                            second_path
                            if second_side == 0
                            else second_path[::-1].copy()
                        )
                        connector = _infill_endpoint_connector(
                            first_endpoint,
                            second_endpoint,
                            boundary_rings,
                            seam_geometry,
                            line_width,
                            tolerance,
                        )
                        uses_projected_boundary = False
                        if connector is None:
                            connector = projected_boundary_connector(
                                first_endpoint,
                                second_endpoint,
                            )
                            uses_projected_boundary = connector is not None
                        if connector is None:
                            connector = np.asarray(
                                [first_endpoint, second_endpoint],
                                dtype=np.float32,
                            )
                            connector_region = physical_centerlines
                        else:
                            connector = np.asarray(
                                connector[:, :2],
                                dtype=np.float32,
                            )
                            if not _close(
                                connector[0, :2],
                                first_oriented[-1, :2],
                                tolerance,
                            ):
                                connector = connector[::-1].copy()
                            connector_region = (
                                physical_centerlines
                                if uses_projected_boundary
                                else seam_geometry
                            )
                        tangent_preserves_endcaps = False

                        # Prefer a cubic whose endpoint tangents continue both
                        # trails. Unlike a post-hoc fillet, this preserves the
                        # original end caps and cannot open a new coverage gap.
                        tangent_connector = (
                            None
                            if uses_projected_boundary
                            else tangent_cubic_connector(
                                first_oriented,
                                second_oriented,
                            )
                        )
                        if tangent_connector is not None:
                            connector = tangent_connector
                            connector_region = physical_centerlines
                            tangent_preserves_endcaps = True
                        else:
                            # Two visually adjacent endpoints may point in the
                            # same direction. Continuing their exact end rays
                            # makes the cubic graze one of its source trails.
                            # Retract only the local end caps and retry; the
                            # complete coverage check below decides whether the
                            # replacement remains acceptable.
                            trimmed_connection_found = False
                            for first_factor, second_factor in (
                                (0.05, 0.05),
                                (0.10, 0.10),
                                (0.10, 0.20),
                                (0.20, 0.10),
                                (0.20, 0.20),
                            ):
                                reversed_first = _trim_open_path_start(
                                    first_oriented[::-1].copy(),
                                    line_width * first_factor,
                                    tolerance,
                                )
                                trimmed_second = _trim_open_path_start(
                                    second_oriented,
                                    line_width * second_factor,
                                    tolerance,
                                )
                                if (
                                    reversed_first is None
                                    or trimmed_second is None
                                ):
                                    continue
                                trimmed_first = reversed_first[::-1].copy()
                                tangent_connector = tangent_cubic_connector(
                                    trimmed_first,
                                    trimmed_second,
                                )
                                if tangent_connector is None:
                                    continue
                                first_oriented = trimmed_first
                                second_oriented = trimmed_second
                                connector = tangent_connector
                                connector_region = physical_centerlines
                                trimmed_connection_found = True
                                break
                            if trimmed_connection_found:
                                tangent_preserves_endcaps = False
                        connector_length = _open_path_length(connector)
                        if connector_length > maximum_connector_length:
                            continue
                        connector_line = LineString(connector[:, :2])
                        if not connector_region.covers(connector_line):
                            continue
                        if (
                            not contour_exclusion.is_empty
                            and connector_line.intersects(contour_exclusion)
                        ):
                            continue
                        connector_bead = connector_line.buffer(
                            bead_radius,
                            cap_style="round",
                            join_style="round",
                            quad_segs=8,
                        )
                        third_path_overlap = (
                            0.0
                            if third_path_coverage.is_empty
                            else float(
                                connector_bead.intersection(
                                    third_path_coverage
                                ).area
                            )
                        )
                        if third_path_overlap > maximum_third_path_overlap:
                            continue

                        chain = _dedupe_consecutive(
                            np.vstack(
                                (
                                    first_oriented,
                                    connector[1:],
                                    second_oriented[1:],
                                )
                            ).astype(np.float32),
                            tolerance,
                        )
                        if chain.shape[0] < 2 or not LineString(chain).is_simple:
                            continue
                        chain_line = LineString(chain[:, :2])
                        if (
                            not chain_line.is_simple
                            or not physical_centerlines.covers(chain_line)
                            or (
                                not contour_exclusion.is_empty
                                and chain_line.intersects(contour_exclusion)
                            )
                        ):
                            continue

                        proposed_paths = [
                            path
                            for path_index, path in enumerate(paths)
                            if path_index not in (first_index, second_index)
                        ]
                        proposed_paths.append(chain)
                        if (
                            not allow_overlap_relaxation
                            and not _solid_fill_spacing_postcondition(
                                proposed_paths,
                                last_perimeter_linework,
                                _resin_maximum_overlap_spacing(config),
                                tolerance,
                                bead_width=line_width,
                                minimum_wall_spacing=(
                                    _resin_contour_infill_maximum_overlap_spacing(
                                        config
                                    )
                                ),
                            )
                        ):
                            continue

                        if tangent_preserves_endcaps:
                            # The original two trails are embedded unchanged in
                            # the merged chain, so its bead union is a strict
                            # superset. Avoid an expensive whole-layer MIC pass.
                            coverage_gain = float(
                                connector_bead.difference(
                                    baseline_coverage
                                ).area
                            )
                        else:
                            proposed_coverage = _round_bead_coverage(
                                [*perimeter_paths, *proposed_paths],
                                bead_radius,
                            )
                            (
                                proposed_diameter,
                                proposed_uncovered_area,
                                _,
                                _,
                            ) = _full_density_coverage_metrics(
                                solid_geometry,
                                perimeter_paths,
                                proposed_paths,
                                line_width,
                                tolerance,
                            )
                            if (
                                proposed_diameter
                                > max(
                                    baseline_diameter,
                                    target_void_diameter,
                                )
                                + tolerance * 2.0
                                or proposed_uncovered_area
                                > baseline_uncovered_area + area_allowance
                            ):
                                continue
                            coverage_gain = float(
                                proposed_coverage.difference(
                                    baseline_coverage
                                ).area
                            )
                        proposals.append(
                            (
                                (
                                    third_path_overlap,
                                    -coverage_gain,
                                    connector_length,
                                    first_index,
                                    second_index,
                                ),
                                first_index,
                                second_index,
                                chain,
                            )
                        )

        # A short isolated trail may sit between two portions of one longer
        # trail. Connecting either pair of endpoints directly then crosses the
        # host before reaching its nominal endpoint. Replace that host interval
        # with the isolated trail instead: this is the continuous, non-retraced
        # route a human naturally draws through both visible gaps.
        detour_guest_paths = paths if enable_detour_absorption else []
        for guest_index, guest_path in enumerate(detour_guest_paths):
            for host_index, host_path in enumerate(paths):
                if guest_index == host_index:
                    continue
                third_path_coverage = _round_bead_coverage(
                    [
                        path
                        for path_index, path in enumerate(paths)
                        if path_index not in (guest_index, host_index)
                    ],
                    bead_radius,
                )
                host_line = LineString(host_path[:, :2])
                host_length = float(host_line.length)
                for guest_oriented in (
                    guest_path,
                    guest_path[::-1].copy(),
                ):
                    guest_start = Point(
                        float(guest_oriented[0, 0]),
                        float(guest_oriented[0, 1]),
                    )
                    guest_end = Point(
                        float(guest_oriented[-1, 0]),
                        float(guest_oriented[-1, 1]),
                    )
                    host_start_point = nearest_points(
                        guest_start,
                        host_line,
                    )[1]
                    host_end_point = nearest_points(
                        guest_end,
                        host_line,
                    )[1]
                    host_start_distance = float(
                        host_line.project(host_start_point)
                    )
                    host_end_distance = float(
                        host_line.project(host_end_point)
                    )
                    if (
                        host_start_distance >= host_end_distance - tolerance
                        or host_end_distance - host_start_distance
                        <= line_width
                    ):
                        continue
                    first_connector = LineString(
                        [host_start_point.coords[0], guest_start.coords[0]]
                    )
                    second_connector = LineString(
                        [guest_end.coords[0], host_end_point.coords[0]]
                    )
                    connector_length = float(
                        first_connector.length + second_connector.length
                    )
                    if (
                        first_connector.length > line_width * 2.0
                        or second_connector.length > line_width * 2.0
                        or connector_length > maximum_connector_length
                    ):
                        continue
                    connector_linework = unary_union(
                        [first_connector, second_connector]
                    )
                    if (
                        not physical_centerlines.covers(connector_linework)
                        or (
                            not contour_exclusion.is_empty
                            and connector_linework.intersects(
                                contour_exclusion
                            )
                        )
                    ):
                        continue

                    if host_start_distance <= tolerance:
                        host_prefix = np.asarray(
                            [host_path[0, :2]],
                            dtype=np.float32,
                        )
                    else:
                        host_prefix = _open_path_prefix(
                            host_path,
                            host_start_distance,
                            tolerance,
                        )
                    if host_end_distance >= host_length - tolerance:
                        host_suffix = np.asarray(
                            [host_path[-1, :2]],
                            dtype=np.float32,
                        )
                    else:
                        host_suffix = _trim_open_path_start(
                            host_path,
                            host_end_distance,
                            tolerance,
                        )
                    if host_prefix is None or host_suffix is None:
                        continue
                    host_after_start = _trim_open_path_start(
                        host_path,
                        host_start_distance,
                        tolerance,
                    )
                    if host_after_start is None:
                        continue
                    host_residual = _open_path_prefix(
                        host_after_start,
                        host_end_distance - host_start_distance,
                        tolerance,
                    )
                    if (
                        host_residual is None
                        or host_residual.shape[0] < 2
                        or _open_path_length(host_residual)
                        < line_width * 2.0
                    ):
                        continue
                    chain = _dedupe_consecutive(
                        np.vstack(
                            (
                                host_prefix,
                                np.asarray(
                                    guest_oriented,
                                    dtype=np.float32,
                                ),
                                host_suffix,
                            )
                        ).astype(np.float32),
                        tolerance,
                    )
                    if chain.shape[0] < 2 or not LineString(chain).is_simple:
                        continue
                    chain_line = LineString(chain[:, :2])
                    if (
                        not chain_line.is_simple
                        or not physical_centerlines.covers(chain_line)
                        or (
                            not contour_exclusion.is_empty
                            and chain_line.intersects(contour_exclusion)
                        )
                    ):
                        continue
                    residual_start_projection = nearest_points(
                        Point(
                            float(host_residual[0, 0]),
                            float(host_residual[0, 1]),
                        ),
                        chain_line,
                    )[1]
                    residual_end_projection = nearest_points(
                        Point(
                            float(host_residual[-1, 0]),
                            float(host_residual[-1, 1]),
                        ),
                        chain_line,
                    )[1]
                    if (
                        residual_start_projection.distance(
                            Point(
                                float(host_residual[0, 0]),
                                float(host_residual[0, 1]),
                            )
                        )
                        > line_width * 0.25
                        or residual_end_projection.distance(
                            Point(
                                float(host_residual[-1, 0]),
                                float(host_residual[-1, 1]),
                            )
                        )
                        > line_width * 0.25
                    ):
                        continue
                    host_residual = _dedupe_consecutive(
                        np.vstack(
                            (
                                np.asarray(
                                    residual_start_projection.coords[0],
                                    dtype=np.float32,
                                ),
                                host_residual,
                                np.asarray(
                                    residual_end_projection.coords[0],
                                    dtype=np.float32,
                                ),
                            )
                        ).astype(np.float32),
                        tolerance,
                    )
                    residual_line = LineString(host_residual[:, :2])
                    if (
                        not residual_line.is_simple
                        or not physical_centerlines.covers(residual_line)
                        or (
                            not contour_exclusion.is_empty
                            and residual_line.intersects(contour_exclusion)
                        )
                    ):
                        continue
                    connector_bead = connector_linework.buffer(
                        bead_radius,
                        cap_style="round",
                        join_style="round",
                        quad_segs=8,
                    )
                    third_path_overlap = (
                        0.0
                        if third_path_coverage.is_empty
                        else float(
                            connector_bead.intersection(
                                third_path_coverage
                            ).area
                        )
                    )
                    if third_path_overlap > maximum_third_path_overlap:
                        continue
                    proposed_paths = [
                        path
                        for path_index, path in enumerate(paths)
                        if path_index not in (guest_index, host_index)
                    ]
                    proposed_paths.extend((chain, host_residual))
                    if (
                        not allow_overlap_relaxation
                        and not _solid_fill_spacing_postcondition(
                            proposed_paths,
                            last_perimeter_linework,
                            _resin_maximum_overlap_spacing(config),
                            tolerance,
                            bead_width=line_width,
                            minimum_wall_spacing=(
                                _resin_contour_infill_maximum_overlap_spacing(
                                    config
                                )
                            ),
                        )
                    ):
                        continue
                    proposed_coverage = _round_bead_coverage(
                        [*perimeter_paths, *proposed_paths],
                        bead_radius,
                    )
                    (
                        proposed_diameter,
                        proposed_uncovered_area,
                        _,
                        _,
                    ) = _full_density_coverage_metrics(
                        solid_geometry,
                        perimeter_paths,
                        proposed_paths,
                        line_width,
                        tolerance,
                    )
                    if (
                        proposed_diameter
                        > max(baseline_diameter, target_void_diameter)
                        + tolerance * 2.0
                        or proposed_uncovered_area
                        > baseline_uncovered_area + area_allowance
                    ):
                        continue
                    coverage_gain = float(
                        proposed_coverage.difference(
                            baseline_coverage
                        ).area
                    )
                    detour_proposals.append(
                        (
                            (
                                third_path_overlap,
                                -coverage_gain,
                                connector_length,
                                guest_index,
                                host_index,
                            ),
                            guest_index,
                            host_index,
                            [chain, host_residual],
                        )
                    )

        if not proposals:
            if detour_proposals:
                _, guest_index, host_index, replacements = min(
                    detour_proposals,
                    key=lambda item: item[0],
                )
                paths = [
                    path
                    for path_index, path in enumerate(paths)
                    if path_index not in (guest_index, host_index)
                ]
                paths.extend(replacements)
            break
        _, first_index, second_index, chain = min(
            proposals,
            key=lambda item: item[0],
        )
        paths = [
            path
            for path_index, path in enumerate(paths)
            if path_index not in (first_index, second_index)
        ]
        paths.append(chain)

    return optimize_open_path_travel(paths, tolerance)


def _replace_path_interval_with_detour(
    path: np.ndarray,
    start_distance: float,
    end_distance: float,
    apex: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, LineString, np.ndarray] | None:
    if path.shape[0] < 2 or not (tolerance < start_distance < end_distance):
        return None
    points = np.asarray(path[:, :2], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if end_distance >= total_length - tolerance:
        return None

    line = LineString(points)
    start = np.asarray(line.interpolate(start_distance).coords[0], dtype=np.float64)
    end = np.asarray(line.interpolate(end_distance).coords[0], dtype=np.float64)
    middle = points[
        (cumulative > start_distance + tolerance)
        & (cumulative < end_distance - tolerance)
    ]
    original_interval = LineString(np.vstack((start, middle, end)))
    detour = np.asarray([start, apex, end], dtype=np.float32)
    prefix = points[cumulative < start_distance - tolerance]
    suffix = points[cumulative > end_distance + tolerance]
    new_path = _dedupe_consecutive(
        np.vstack((prefix, detour, suffix)).astype(np.float32),
        tolerance,
    )
    if new_path.shape[0] < 2:
        return None
    return new_path, original_interval, detour


def _gyroid_infill_geometry(geometry, spacing: float, tolerance: float) -> list[np.ndarray]:
    if spacing <= 0:
        raise ValueError("infill spacing must be positive")
    if geometry.is_empty:
        return []

    min_x, min_y, max_x, max_y = geometry.bounds
    wavelength = max(spacing * GYROID_WAVELENGTH_FACTOR, spacing + tolerance)
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


def optimize_triangle_infill_travel(
    paths: list[np.ndarray],
    tolerance: float = 1e-5,
) -> list[np.ndarray]:
    """Choose the best greedy open-path order for legacy triangle infill.

    Triangle lattice paths are intentionally kept as separate open paths so
    different 0/60/120 degree families are never joined with an extrusion
    connector. Trying representative starting points improves endpoint
    ordering without changing path geometry or adding travel segments.
    """

    if len(paths) <= 2:
        return optimize_open_path_travel(paths, tolerance)

    step = max(1, len(paths) // 8)
    starts = set(range(0, len(paths), step))
    starts.update((0, len(paths) // 2, len(paths) - 1))
    candidates = [
        optimize_open_path_travel(paths[start:] + paths[:start], tolerance)
        for start in sorted(starts)
    ]
    return min(candidates, key=_open_path_travel_length)


def _open_path_travel_length(paths: list[np.ndarray]) -> float:
    return sum(
        float(np.linalg.norm(paths[index][0, :2] - paths[index - 1][-1, :2]))
        for index in range(1, len(paths))
    )


def _legacy_path_merge_tolerance(line_width: float, tolerance: float) -> float:
    """Limit post-planning joins to numerical endpoint coincidence.

    ``line_width`` remains in the signature for metadata/API compatibility,
    but a fraction of the bead width is not a safe merge tolerance: those
    links would bypass the bead-aware connector clearance checks.
    """

    del line_width
    return tolerance


def merge_adjacent_connected_paths(
    paths: list[np.ndarray],
    tolerance: float = 1e-5,
) -> list[np.ndarray]:
    """Merge consecutive open paths whose endpoints coincide numerically.

    This deliberately does not search for nearby paths. If a caller supplies
    a looser tolerance, both near-endpoints are retained so the original first
    segment is never silently replaced by an unvalidated diagonal.
    """

    merged: list[np.ndarray] = []
    for path in paths:
        current = np.asarray(path, dtype=np.float32)
        if current.shape[0] == 0:
            continue
        if (
            merged
            and current.shape[0] >= 2
            and merged[-1].shape[0] >= 2
            and not _is_closed_path(merged[-1], tolerance)
            and not _is_closed_path(current, tolerance)
            and _close(merged[-1][-1, :2], current[0, :2], tolerance)
        ):
            endpoint_distance = float(
                np.linalg.norm(merged[-1][-1, :2] - current[0, :2])
            )
            continuation = current[1:] if endpoint_distance <= 1e-12 else current
            merged[-1] = np.vstack((merged[-1], continuation))
        else:
            merged.append(current)
    return merged


def _smooth_resin_infill_paths(
    paths: list[np.ndarray],
    geometry,
    max_radius: float,
    angle_threshold_degrees: float,
    tolerance: float,
    cut_fraction: float = 0.35,
    merge_tolerance: float | None = None,
) -> list[np.ndarray]:
    if max_radius <= tolerance or not paths:
        return (
            merge_adjacent_connected_paths(paths, merge_tolerance)
            if merge_tolerance is not None
            else paths
        )

    # ``geometry`` is already the bead-aware centerline corridor.  Only add a
    # numerical epsilon here; expanding by the corner radius would move a
    # smoothed connector back into the wall-overlap zone.
    safe_geometry = geometry.buffer(max(tolerance * 10.0, 1e-7), join_style="round")
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
    if merge_tolerance is not None:
        return merge_adjacent_connected_paths(smoothed, merge_tolerance)
    return smoothed


def _connect_zigzag_infill_paths(
    paths: list[np.ndarray],
    geometry,
    spacing: float,
    minimum_clearance: float,
    tolerance: float,
    *,
    maximum_spacing: float | None = None,
    solid_bead_width: float | None = None,
    wall_seam_clearance: float | None = None,
    solid_smoothing_angle_degrees: float | None = None,
    solid_smoothing_corner_cut: float | None = None,
    maximum_connector_overlap_spacing: float | None = None,
    follow_boundaries: bool = True,
) -> list[np.ndarray]:
    """Join adjacent scanlines along the bead-aware centerline boundary."""

    return _connect_boundary_infill_paths(
        paths,
        geometry,
        spacing,
        minimum_clearance,
        tolerance,
        adjacent_scanlines_only=True,
        maximum_scan_spacing=maximum_spacing,
        solid_bead_width=solid_bead_width,
        wall_seam_clearance=wall_seam_clearance,
        solid_smoothing_angle_degrees=solid_smoothing_angle_degrees,
        solid_smoothing_corner_cut=solid_smoothing_corner_cut,
        maximum_connector_overlap_spacing=maximum_connector_overlap_spacing,
        follow_boundaries=follow_boundaries,
    )


def _tangent_u_turn_connector(
    paths: list[np.ndarray],
    first_endpoint: int,
    second_endpoint: int,
    boundary_connector: np.ndarray,
    safe_geometry,
    tolerance: float,
) -> np.ndarray | None:
    """Replace an aligned adjacent-scan connector with a tangent semicircle."""

    first_path = np.asarray(paths[first_endpoint // 2][:, :2], dtype=np.float64)
    second_path = np.asarray(paths[second_endpoint // 2][:, :2], dtype=np.float64)
    if first_path.shape[0] < 2 or second_path.shape[0] < 2:
        return None

    first_index = 0 if first_endpoint % 2 == 0 else -1
    first_neighbor_index = 1 if first_endpoint % 2 == 0 else -2
    second_index = 0 if second_endpoint % 2 == 0 else -1
    second_neighbor_index = 1 if second_endpoint % 2 == 0 else -2
    start = first_path[first_index]
    end = second_path[second_index]
    first_neighbor = first_path[first_neighbor_index]
    second_neighbor = second_path[second_neighbor_index]

    desired_start = start - first_neighbor
    desired_end = second_neighbor - end
    first_length = float(np.linalg.norm(desired_start))
    second_length = float(np.linalg.norm(desired_end))
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    if min(first_length, second_length, chord_length) <= tolerance:
        return None
    desired_start /= first_length
    desired_end /= second_length
    chord_unit = chord / chord_length
    if (
        abs(float(np.dot(desired_start, chord_unit))) > 0.05
        or abs(float(np.dot(desired_end, chord_unit))) > 0.05
        or float(np.dot(desired_start, desired_end)) > -0.995
    ):
        # Real STL boundaries can shift the two adjacent hatch endpoints a
        # little in X while the hatch directions remain a true U-turn.  The
        # circular connector above is then not tangent enough, but dropping
        # the link would destroy Zigzag continuity.  Use a short cubic that
        # preserves both endpoint tangents and stays inside the already
        # inset fill surface.
        if float(np.dot(desired_start, desired_end)) > -0.995:
            return None
        for handle_factor in (0.12, 0.2, 0.3, 0.4):
            handle_length = chord_length * handle_factor
            parameters = np.linspace(0.0, 1.0, 17, dtype=np.float64)[:, None]
            one_minus = 1.0 - parameters
            first_control = start + desired_start * handle_length
            second_control = end - desired_end * handle_length
            curve = (
                one_minus**3 * start
                + 3.0 * one_minus**2 * parameters * first_control
                + 3.0 * one_minus * parameters**2 * second_control
                + parameters**3 * end
            ).astype(np.float32)
            curve[0] = start.astype(np.float32)
            curve[-1] = end.astype(np.float32)
            curve_line = LineString(curve)
            if curve_line.is_simple and safe_geometry.covers(curve_line):
                return curve
        return None

    center = (start + end) * 0.5
    start_radius = start - center
    start_angle = math.atan2(float(start_radius[1]), float(start_radius[0]))
    sample_count = 19
    boundary_line = LineString(boundary_connector[:, :2])
    candidates: list[tuple[float, np.ndarray]] = []
    for direction in (1.0, -1.0):
        angles = start_angle + direction * np.linspace(0.0, math.pi, sample_count)
        arc = np.column_stack(
            (
                center[0] + np.cos(angles) * chord_length * 0.5,
                center[1] + np.sin(angles) * chord_length * 0.5,
            )
        ).astype(np.float32)
        arc[0] = start
        arc[-1] = end
        arc_start_direction = np.asarray(arc[1] - arc[0], dtype=np.float64)
        arc_end_direction = np.asarray(arc[-1] - arc[-2], dtype=np.float64)
        arc_start_direction /= float(np.linalg.norm(arc_start_direction))
        arc_end_direction /= float(np.linalg.norm(arc_end_direction))
        if (
            float(np.dot(arc_start_direction, desired_start)) < 0.99
            or float(np.dot(arc_end_direction, desired_end)) < 0.99
        ):
            continue
        arc_line = LineString(arc)
        if not arc_line.is_simple or not safe_geometry.covers(arc_line):
            continue
        candidates.append((arc_line.hausdorff_distance(boundary_line), arc))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _interior_tangent_connector(
    paths: list[np.ndarray],
    first_endpoint: int,
    second_endpoint: int,
    safe_geometry,
    spacing: float,
    tolerance: float,
) -> np.ndarray | None:
    """Make a short tangent connection without tracing the fill boundary."""

    first_path = np.asarray(paths[first_endpoint // 2][:, :2], dtype=np.float64)
    second_path = np.asarray(paths[second_endpoint // 2][:, :2], dtype=np.float64)
    if first_path.shape[0] < 2 or second_path.shape[0] < 2:
        return None

    first_index = 0 if first_endpoint % 2 == 0 else -1
    first_neighbor_index = 1 if first_endpoint % 2 == 0 else -2
    second_index = 0 if second_endpoint % 2 == 0 else -1
    second_neighbor_index = 1 if second_endpoint % 2 == 0 else -2
    start = first_path[first_index]
    end = second_path[second_index]
    start_tangent = start - first_path[first_neighbor_index]
    end_tangent = second_path[second_neighbor_index] - end
    start_length = float(np.linalg.norm(start_tangent))
    end_length = float(np.linalg.norm(end_tangent))
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    if min(start_length, end_length, chord_length) <= tolerance:
        return None
    start_tangent /= start_length
    end_tangent /= end_length
    if float(np.dot(start_tangent, end_tangent)) > -0.5:
        return None

    allowance = min(spacing * 0.15, 0.25)
    for handle_factor in (0.12, 0.2, 0.3, 0.4):
        handle_length = chord_length * handle_factor
        parameters = np.linspace(0.0, 1.0, 17, dtype=np.float64)[:, None]
        one_minus = 1.0 - parameters
        first_control = start + start_tangent * handle_length
        second_control = end - end_tangent * handle_length
        curve = (
            one_minus**3 * start
            + 3.0 * one_minus**2 * parameters * first_control
            + 3.0 * one_minus * parameters**2 * second_control
            + parameters**3 * end
        ).astype(np.float32)
        curve[0] = start.astype(np.float32)
        curve[-1] = end.astype(np.float32)
        curve_line = LineString(curve)
        if curve_line.is_simple and safe_geometry.buffer(
            allowance,
            join_style="round",
        ).covers(curve_line):
            return curve
    return None


def _connect_boundary_infill_paths(
    paths: list[np.ndarray],
    geometry,
    spacing: float,
    minimum_clearance: float,
    tolerance: float,
    *,
    adjacent_scanlines_only: bool,
    maximum_scan_spacing: float | None = None,
    solid_bead_width: float | None = None,
    wall_seam_clearance: float | None = None,
    solid_smoothing_angle_degrees: float | None = None,
    solid_smoothing_corner_cut: float | None = None,
    maximum_connector_overlap_spacing: float | None = None,
    follow_boundaries: bool = True,
) -> list[np.ndarray]:
    """Chain open infill trails with safe direct or boundary-following links."""

    if len(paths) < 2 or spacing <= 0 or geometry.is_empty:
        return paths

    open_indices = [
        index
        for index, path in enumerate(paths)
        if path.shape[0] >= 2 and not _is_closed_path(path, tolerance)
    ]
    if len(open_indices) < 2:
        return paths

    safe_geometry = geometry.buffer(max(tolerance * 10.0, 1e-7), join_style="round")
    boundary_rings = (
        _infill_boundary_rings(geometry, tolerance)
        if follow_boundaries
        else []
    )
    path_lines = {
        index: LineString(
            [(float(point[0]), float(point[1])) for point in path[:, :2]]
        )
        for index, path in enumerate(paths)
        if path.shape[0] >= 2
    }
    endpoint_points = {
        2 * index + side: np.asarray(
            paths[index][0 if side == 0 else -1, :2], dtype=np.float32
        )
        for index in open_indices
        for side in (0, 1)
    }
    scan_levels = (
        _infill_scan_levels(paths, open_indices, tolerance)
        if adjacent_scanlines_only
        else {}
    )
    connector_spacing = (
        max(spacing, maximum_scan_spacing)
        if maximum_scan_spacing is not None
        else spacing
    )
    adjacent_level_tolerance = max(spacing * 0.01, tolerance * 20.0)
    if adjacent_scanlines_only:
        ordered_indices = sorted(open_indices, key=lambda index: (scan_levels[index], index))
        minimum_level_delta = (
            spacing - adjacent_level_tolerance
            if maximum_scan_spacing is not None
            else spacing - max(spacing * 0.1, tolerance * 20.0)
        )
        maximum_level_delta = (
            maximum_scan_spacing + adjacent_level_tolerance
            if maximum_scan_spacing is not None
            else spacing + max(spacing * 0.1, tolerance * 20.0)
        )
        candidate_index_pairs: list[tuple[int, int]] = []
        for first_position, first_index in enumerate(ordered_indices):
            first_level = scan_levels[first_index]
            for second_index in ordered_indices[first_position + 1 :]:
                level_delta = scan_levels[second_index] - first_level
                if level_delta > maximum_level_delta:
                    break
                if level_delta >= minimum_level_delta:
                    candidate_index_pairs.append((first_index, second_index))
    else:
        candidate_index_pairs = [
            (first_index, second_index)
            for first_position, first_index in enumerate(open_indices)
            for second_index in open_indices[first_position + 1 :]
        ]

    candidates: list[tuple[float, int, int, np.ndarray, bool]] = []
    for first_index, second_index in candidate_index_pairs:
        for first_side in (0, 1):
            first_endpoint = 2 * first_index + first_side
            for second_side in (0, 1):
                second_endpoint = 2 * second_index + second_side
                direct_distance = float(
                    np.linalg.norm(
                        endpoint_points[first_endpoint]
                        - endpoint_points[second_endpoint]
                    )
                )
                # A continuity link is printed with material.  Routes whose
                # endpoints are more than eight pitches apart add too much
                # non-pattern material and are not useful candidates.  This
                # inexpensive test also avoids quadratic Shapely projection
                # work for dense gyroid layers.
                if direct_distance > connector_spacing * 8.0:
                    continue
                connector_points = None
                if not follow_boundaries and adjacent_scanlines_only:
                    connector_points = _interior_tangent_connector(
                        paths,
                        first_endpoint,
                        second_endpoint,
                        safe_geometry,
                        connector_spacing,
                        tolerance,
                    )
                if connector_points is None:
                    connector_points = _infill_endpoint_connector(
                        endpoint_points[first_endpoint],
                        endpoint_points[second_endpoint],
                        boundary_rings,
                        safe_geometry,
                        connector_spacing,
                        tolerance,
                    )
                if connector_points is None:
                    continue
                connector_score = _open_path_length(connector_points)
                tangent_u_turn = None
                if adjacent_scanlines_only and solid_bead_width is not None:
                    tangent_u_turn = _tangent_u_turn_connector(
                        paths,
                        first_endpoint,
                        second_endpoint,
                        connector_points,
                        safe_geometry,
                        tolerance,
                    )
                is_tangent_u_turn = tangent_u_turn is not None
                if tangent_u_turn is not None:
                    connector_points = tangent_u_turn
                candidates.append(
                    (
                        connector_score,
                        first_endpoint,
                        second_endpoint,
                        connector_points,
                        is_tangent_u_turn,
                    )
                )
    candidates_by_endpoint: dict[
        int,
        list[tuple[float, int, np.ndarray, bool]],
    ] = defaultdict(list)
    for (
        length,
        first_endpoint,
        second_endpoint,
        connector_points,
        is_tangent_u_turn,
    ) in candidates:
        candidates_by_endpoint[first_endpoint].append(
            (length, second_endpoint, connector_points, is_tangent_u_turn)
        )
        candidates_by_endpoint[second_endpoint].append(
            (length, first_endpoint, connector_points, is_tangent_u_turn)
        )
    for endpoint_candidates in candidates_by_endpoint.values():
        endpoint_candidates.sort(key=lambda item: (item[0], item[1]))

    unused = set(open_indices)
    accepted: list[tuple[int, int, np.ndarray]] = []
    connector_by_endpoint: dict[int, tuple[int, np.ndarray, bool]] = {}
    components: list[list[int]] = []
    reverse_solid_scan_order = (
        adjacent_scanlines_only
        and maximum_scan_spacing is not None
        and maximum_connector_overlap_spacing is not None
    )
    while unused:
        # Work back from the far scan side so hole-split branches keep their
        # adjacent continuation available instead of burying it mid-chain.
        start_index = (
            max(unused, key=lambda index: (scan_levels[index], index))
            if reverse_solid_scan_order
            else min(unused)
        )
        unused.remove(start_index)
        component = [start_index]
        left_endpoint = 2 * start_index
        right_endpoint = 2 * start_index + 1

        while True:
            extensions: list[tuple[float, str, int, int, np.ndarray, bool]] = []
            for side, current_endpoint in (
                ("left", left_endpoint),
                ("right", right_endpoint),
            ):
                for (
                    length,
                    next_endpoint,
                    connector_points,
                    is_tangent_u_turn,
                ) in candidates_by_endpoint.get(current_endpoint, []):
                    next_index = next_endpoint // 2
                    if next_index not in unused:
                        continue
                    connector = LineString(
                        [(float(point[0]), float(point[1])) for point in connector_points]
                    )
                    if not _resin_connector_is_clear(
                        connector,
                        paths,
                        path_lines,
                        current_endpoint,
                        next_endpoint,
                        accepted,
                        tolerance,
                        minimum_clearance=minimum_clearance,
                        maximum_overlap_spacing=(
                            maximum_connector_overlap_spacing
                        ),
                        bead_width=solid_bead_width,
                        safe_geometry=safe_geometry,
                        smoothing_angle_degrees=(
                            solid_smoothing_angle_degrees
                        ),
                        smoothing_corner_cut=solid_smoothing_corner_cut,
                    ):
                        continue
                    extensions.append(
                        (
                            length,
                            side,
                            current_endpoint,
                            next_endpoint,
                            connector_points,
                            is_tangent_u_turn,
                        )
                    )
                    break
            if not extensions:
                break

            (
                _,
                side,
                current_endpoint,
                next_endpoint,
                connector_points,
                is_tangent_u_turn,
            ) = min(
                extensions,
                key=lambda item: (item[0], item[3]),
            )
            next_index = next_endpoint // 2
            unused.remove(next_index)
            component.append(next_index)
            accepted.append((current_endpoint, next_endpoint, connector_points))
            connector_by_endpoint[current_endpoint] = (
                next_endpoint,
                connector_points,
                is_tangent_u_turn,
            )
            connector_by_endpoint[next_endpoint] = (
                current_endpoint,
                connector_points,
                is_tangent_u_turn,
            )
            opposite_endpoint = 2 * next_index + (1 - next_endpoint % 2)
            if side == "left":
                left_endpoint = opposite_endpoint
            else:
                right_endpoint = opposite_endpoint
        components.append(component)

    # The first pass grows one chain at a time and deliberately never revisits
    # a path once it belongs to a component.  On hole-split scanlines that can
    # leave two completed chains whose free endpoints still have a safe
    # boundary connector between them.  Merge those components globally before
    # materializing the polylines.  Endpoint degree remains at most two and a
    # union-find guard prevents closed cycles/retracing; every added connector
    # passes the same full clearance checks as a first-pass connector.
    component_parent = {index: index for index in open_indices}

    def component_root(index: int) -> int:
        while component_parent[index] != index:
            component_parent[index] = component_parent[component_parent[index]]
            index = component_parent[index]
        return index

    def union_components(first: int, second: int) -> None:
        first_root = component_root(first)
        second_root = component_root(second)
        if first_root != second_root:
            component_parent[second_root] = first_root

    for first_endpoint, second_endpoint, _ in accepted:
        union_components(first_endpoint // 2, second_endpoint // 2)

    for (
        _,
        first_endpoint,
        second_endpoint,
        connector_points,
        is_tangent_u_turn,
    ) in sorted(candidates, key=lambda item: (item[0], item[1], item[2])):
        if (
            first_endpoint in connector_by_endpoint
            or second_endpoint in connector_by_endpoint
        ):
            continue
        first_index = first_endpoint // 2
        second_index = second_endpoint // 2
        if component_root(first_index) == component_root(second_index):
            continue
        connector = LineString(
            [(float(point[0]), float(point[1])) for point in connector_points]
        )
        if not _resin_connector_is_clear(
            connector,
            paths,
            path_lines,
            first_endpoint,
            second_endpoint,
            accepted,
            tolerance,
            minimum_clearance=minimum_clearance,
            maximum_overlap_spacing=maximum_connector_overlap_spacing,
            bead_width=solid_bead_width,
            safe_geometry=safe_geometry,
            smoothing_angle_degrees=solid_smoothing_angle_degrees,
            smoothing_corner_cut=solid_smoothing_corner_cut,
        ):
            continue
        accepted.append((first_endpoint, second_endpoint, connector_points))
        connector_by_endpoint[first_endpoint] = (
            second_endpoint,
            connector_points,
            is_tangent_u_turn,
        )
        connector_by_endpoint[second_endpoint] = (
            first_endpoint,
            connector_points,
            is_tangent_u_turn,
        )
        union_components(first_index, second_index)

    merged_components: dict[int, list[int]] = defaultdict(list)
    for component in components:
        if component:
            merged_components[component_root(component[0])].extend(component)
    components = list(merged_components.values())

    connected_paths: list[np.ndarray] = []
    connected_indices: set[int] = set()
    compensate_wall_seams = (
        adjacent_scanlines_only
        and solid_bead_width is not None
        and wall_seam_clearance is not None
        and solid_bead_width > tolerance
        and wall_seam_clearance < solid_bead_width - tolerance
    )
    compensated_wall_gaps: set[tuple[int, int]] = set()
    compensated_wall_lines: list[LineString] = []
    accepted_connector_linework = (
        unary_union(
            [
                LineString(
                    [(float(point[0]), float(point[1])) for point in connector]
                )
                for _, _, connector in accepted
            ]
        )
        if accepted
        else GeometryCollection()
    )

    def validated_chain_chunks(
        chain: np.ndarray,
        connector_ranges: list[tuple[int, int]],
        component_indices: list[int],
    ) -> list[np.ndarray]:
        """Drop only connectors that remain unsafe after the final fillet."""

        should_validate = (
            maximum_connector_overlap_spacing is not None
            and solid_bead_width is not None
        )
        if not should_validate or not connector_ranges:
            return [_dedupe_consecutive(chain, tolerance)]

        def chain_is_valid(candidate: np.ndarray) -> bool:
            candidate_paths = [candidate]
            if (
                solid_smoothing_corner_cut is not None
                and solid_smoothing_corner_cut > tolerance
            ):
                candidate_paths = _smooth_resin_infill_paths(
                    candidate_paths,
                    safe_geometry,
                    solid_smoothing_corner_cut,
                    (
                        DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES
                        if solid_smoothing_angle_degrees is None
                        else solid_smoothing_angle_degrees
                    ),
                    tolerance,
                    cut_fraction=0.3,
                )
            return _solid_fill_spacing_postcondition(
                candidate_paths,
                Polygon(),
                maximum_connector_overlap_spacing,
                tolerance,
                bead_width=solid_bead_width,
                allow_boundary_bridges=True,
            )

        normalized_chain = _dedupe_consecutive(chain, tolerance)
        if chain_is_valid(normalized_chain):
            return [normalized_chain]

        base_segments: list[np.ndarray] = []
        connectors: list[np.ndarray] = []
        base_start = 0
        for entry_index, exit_index in connector_ranges:
            base_segments.append(chain[base_start : entry_index + 1])
            connectors.append(chain[entry_index : exit_index + 1])
            base_start = exit_index
        base_segments.append(chain[base_start:])

        chunks: list[np.ndarray] = []
        current = base_segments[0]
        for connector, next_base in zip(connectors, base_segments[1:]):
            candidate = _dedupe_consecutive(
                np.vstack((current, connector[1:], next_base[1:])).astype(
                    np.float32
                ),
                tolerance,
            )
            if candidate.shape[0] >= 2 and chain_is_valid(candidate):
                current = candidate
                continue
            if current.shape[0] >= 2:
                chunks.append(_dedupe_consecutive(current, tolerance))
            current = next_base
        if current.shape[0] >= 2:
            chunks.append(_dedupe_consecutive(current, tolerance))
        if chunks and all(chain_is_valid(chunk) for chunk in chunks):
            return chunks

        # Seam compensation can make a retained base chunk depend on the
        # connector that was just removed.  If that happens, fall all the way
        # back to the original independent hatches for this component.  Those
        # are the proven bead-aware baseline and losing continuity is safer
        # than exporting a tight hook or failing the whole slice.
        return [
            np.asarray(paths[index][:, :2], dtype=np.float32).copy()
            for index in component_indices
            if paths[index].shape[0] >= 2
        ]

    for component in components:
        start_endpoint = next(
            endpoint
            for index in component
            for endpoint in (2 * index, 2 * index + 1)
            if endpoint not in connector_by_endpoint
        )
        chain_points: list[np.ndarray] = []
        chain_connector_ranges: list[tuple[int, int]] = []
        current_endpoint: int | None = start_endpoint
        incoming_connector: np.ndarray | None = None
        incoming_is_tangent_u_turn = False
        while current_endpoint is not None:
            index = current_endpoint // 2
            side = current_endpoint % 2
            if index in connected_indices:
                break
            connected_indices.add(index)
            oriented_path = (
                np.asarray(paths[index][:, :2], dtype=np.float32)
                if side == 0
                else np.asarray(paths[index][::-1, :2], dtype=np.float32)
            )
            if not chain_points:
                if compensate_wall_seams:
                    start_tail = _solid_wall_seam_tail(
                        current_endpoint,
                        endpoint_points,
                        scan_levels,
                        path_lines,
                        boundary_rings,
                        spacing,
                        maximum_scan_spacing,
                        solid_bead_width,
                        wall_seam_clearance,
                        safe_geometry,
                        tolerance,
                        compensated_wall_gaps,
                        compensated_wall_lines,
                        accepted_connector_linework,
                        prepend=True,
                    )
                    if start_tail is not None:
                        chain_points.extend(start_tail)
                chain_points.extend(oriented_path if not chain_points else oriented_path[1:])
            elif (
                compensate_wall_seams
                and incoming_connector is not None
                and not incoming_is_tangent_u_turn
            ):
                dogleg = _solid_wall_seam_dogleg(
                    incoming_connector,
                    current_endpoint,
                    oriented_path,
                    chain_points,
                    paths,
                    path_lines,
                    endpoint_points,
                    scan_levels,
                    boundary_rings,
                    spacing,
                    maximum_scan_spacing,
                    solid_bead_width,
                    wall_seam_clearance,
                    safe_geometry,
                    tolerance,
                    compensated_wall_gaps,
                    compensated_wall_lines,
                    accepted_connector_linework,
                    smoothing_angle_degrees=solid_smoothing_angle_degrees,
                    smoothing_corner_cut=solid_smoothing_corner_cut,
                )
                if dogleg is None:
                    chain_points.extend(oriented_path[1:])
                else:
                    bridge, trimmed_path = dogleg
                    chain_points.extend(bridge[1:])
                    chain_points.extend(trimmed_path[1:])
            else:
                chain_points.extend(oriented_path[1:])

            exit_endpoint = 2 * index + (1 - side)
            connection = connector_by_endpoint.get(exit_endpoint)
            if connection is None:
                if compensate_wall_seams:
                    end_tail = _solid_wall_seam_tail(
                        exit_endpoint,
                        endpoint_points,
                        scan_levels,
                        path_lines,
                        boundary_rings,
                        spacing,
                        maximum_scan_spacing,
                        solid_bead_width,
                        wall_seam_clearance,
                        safe_geometry,
                        tolerance,
                        compensated_wall_gaps,
                        compensated_wall_lines,
                        accepted_connector_linework,
                        prepend=False,
                    )
                    if end_tail is not None:
                        chain_points.extend(end_tail[1:])
                current_endpoint = None
                continue
            next_endpoint, connector_points, is_tangent_u_turn = connection
            oriented_connector = (
                connector_points
                if _close(connector_points[0, :2], endpoint_points[exit_endpoint], tolerance)
                else connector_points[::-1].copy()
            )
            connector_entry_index = len(chain_points) - 1
            chain_points.extend(oriented_connector[1:])
            chain_connector_ranges.append(
                (connector_entry_index, len(chain_points) - 1)
            )
            incoming_connector = oriented_connector
            incoming_is_tangent_u_turn = is_tangent_u_turn
            current_endpoint = next_endpoint

        if len(chain_points) >= 2:
            connected_paths.extend(
                validated_chain_chunks(
                    np.asarray(chain_points, dtype=np.float32),
                    chain_connector_ranges,
                    component,
                )
            )

    for index, path in enumerate(paths):
        if index not in connected_indices:
            connected_paths.append(path)
    return connected_paths


@dataclass(frozen=True)
class _InfillBoundaryRing:
    line: LineString
    coordinates: np.ndarray
    cumulative_lengths: np.ndarray


def _infill_boundary_rings(geometry, tolerance: float) -> list[_InfillBoundaryRing]:
    rings: list[_InfillBoundaryRing] = []
    for polygon in _iter_polygons(geometry):
        exterior = LineString(polygon.exterior.coords)
        if exterior.length > tolerance:
            rings.append(_infill_boundary_ring(exterior))
        for interior in polygon.interiors:
            ring = LineString(interior.coords)
            if ring.length > tolerance:
                rings.append(_infill_boundary_ring(ring))
    return rings


def _infill_boundary_ring(line: LineString) -> _InfillBoundaryRing:
    coordinates = np.asarray(line.coords, dtype=np.float64)
    differences = np.diff(coordinates, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    return _InfillBoundaryRing(
        line=line,
        coordinates=coordinates,
        cumulative_lengths=np.concatenate(([0.0], np.cumsum(segment_lengths))),
    )


def _solid_wall_seam_tail(
    endpoint: int,
    endpoint_points: dict[int, np.ndarray],
    scan_levels: dict[int, float],
    path_lines: dict[int, LineString],
    boundary_rings: list[_InfillBoundaryRing],
    minimum_scan_spacing: float,
    maximum_scan_spacing: float | None,
    bead_width: float,
    wall_clearance: float,
    safe_geometry,
    tolerance: float,
    compensated_gaps: set[tuple[int, int]],
    compensated_lines: list[LineString],
    accepted_connector_linework,
    *,
    prepend: bool,
) -> np.ndarray | None:
    """Extend a free solid-hatch end only across its uncovered wall gap."""

    gap = _solid_wall_seam_gap_arc(
        endpoint,
        endpoint_points,
        scan_levels,
        boundary_rings,
        minimum_scan_spacing,
        maximum_scan_spacing,
        bead_width,
        wall_clearance,
        tolerance,
        compensated_gaps,
    )
    if gap is None:
        return None
    gap_arc, gap_key = gap
    prefix = _solid_wall_seam_gap_prefix(
        gap_arc,
        bead_width,
        wall_clearance,
        tolerance,
    )
    if prefix is None:
        return None
    prefix_line = LineString(prefix)
    incident_index = endpoint // 2
    if (
        not safe_geometry.covers(prefix_line)
        or _has_unexpected_linework_intersection(
            prefix_line,
            accepted_connector_linework,
            (),
            tolerance,
        )
        or any(
            _has_unexpected_linework_intersection(
                prefix_line,
                path_line,
                (prefix[0, :2],) if path_index == incident_index else (),
                tolerance,
            )
            for path_index, path_line in path_lines.items()
        )
        or any(
            not prefix_line.disjoint(existing)
            for existing in compensated_lines
        )
    ):
        return None
    compensated_gaps.add(gap_key)
    compensated_lines.append(prefix_line)
    return prefix[::-1].copy() if prepend else prefix


def _constant_radius_turn_points(
    start: np.ndarray,
    tangent: np.ndarray,
    radius: float,
    signed_angle: float,
    tolerance: float,
) -> np.ndarray | None:
    """Sample a true-radius planar turn from a point and unit tangent."""

    tangent = np.asarray(tangent[:2], dtype=np.float64)
    tangent_length = float(np.linalg.norm(tangent))
    if (
        tangent_length <= tolerance
        or radius <= tolerance
        or abs(signed_angle) <= tolerance
    ):
        return None
    tangent /= tangent_length
    left_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    turn_sign = 1.0 if signed_angle > 0.0 else -1.0
    # Keep sampled heading changes at or below ten degrees, matching the
    # normal fillet exporter while preserving the analytical radius.
    sample_count = max(
        2,
        int(math.ceil(abs(signed_angle) / math.radians(10.0))) + 1,
    )
    angles = np.linspace(0.0, signed_angle, sample_count)
    points = np.asarray(
        [
            np.asarray(start[:2], dtype=np.float64)
            + radius
            * turn_sign
            * (
                math.sin(float(angle)) * tangent
                + (1.0 - math.cos(float(angle))) * left_normal
            )
            for angle in angles
        ],
        dtype=np.float32,
    )
    points[0] = np.asarray(start[:2], dtype=np.float32)
    return points


def _smooth_solid_wall_seam_dogleg_return(
    prefix: np.ndarray,
    oriented_path: np.ndarray,
    target_radius: float,
    maximum_trim_distance: float,
    safe_geometry,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Replace a seam hairpin with a true-radius tangent return.

    A raw wall-seam dogleg follows the wall past the next hatch endpoint and
    then reverses through a 15--25 degree interior angle.  Treating the UI's
    smoothing value as a tangent cut leaves only a roughly 0.02 mm physical
    radius for a 2 mm bead.  Instead, continue the boundary tangent through a
    constant-radius 180-degree turn, add the analytically required straight,
    and finish with a second constant-radius turn tangent to the hatch.  The
    endpoint is solved on the original hatch ray, so no retrace is introduced.
    """

    if (
        prefix.shape[0] < 2
        or oriented_path.shape[0] < 2
        or target_radius <= tolerance
        or maximum_trim_distance <= tolerance
    ):
        return None

    boundary_vector = _last_nondegenerate_segment(prefix, tolerance)
    hatch_vector = _first_nondegenerate_segment(oriented_path, tolerance)
    if boundary_vector is None or hatch_vector is None:
        return None
    boundary_length = float(np.linalg.norm(boundary_vector))
    hatch_length = float(np.linalg.norm(hatch_vector))
    if min(boundary_length, hatch_length) <= tolerance:
        return None
    boundary_tangent = boundary_vector / boundary_length
    hatch_tangent = hatch_vector / hatch_length
    return_tangent = -boundary_tangent
    start = np.asarray(prefix[-1, :2], dtype=np.float64)
    hatch_start = np.asarray(oriented_path[0, :2], dtype=np.float64)

    # Prefer the requested physical radius.  Shrinking is solely a geometric
    # fallback; radii below half the request would again look like a pointed
    # tooth and are deliberately rejected by the caller's normal fallback.
    for radius_factor in (1.0, 0.85, 0.7, 0.5):
        radius = target_radius * radius_factor
        if radius <= tolerance:
            continue
        for turn_sign in (1.0, -1.0):
            first_angle = turn_sign * math.pi
            first_turn = _constant_radius_turn_points(
                start,
                boundary_tangent,
                radius,
                first_angle,
                tolerance,
            )
            if first_turn is None:
                continue

            cross = float(
                return_tangent[0] * hatch_tangent[1]
                - return_tangent[1] * hatch_tangent[0]
            )
            dot = float(np.clip(np.dot(return_tangent, hatch_tangent), -1.0, 1.0))
            second_angle = math.atan2(cross, dot)
            if abs(second_angle) <= tolerance:
                second_displacement = np.zeros(2, dtype=np.float64)
            else:
                unit_left = np.asarray(
                    [-return_tangent[1], return_tangent[0]],
                    dtype=np.float64,
                )
                second_sign = 1.0 if second_angle > 0.0 else -1.0
                second_displacement = radius * second_sign * (
                    math.sin(second_angle) * return_tangent
                    + (1.0 - math.cos(second_angle)) * unit_left
                )

            first_end = np.asarray(first_turn[-1], dtype=np.float64)
            second_base = first_end + second_displacement
            # second_base + straight_length * return_tangent must equal
            # hatch_start + trim_distance * hatch_tangent.
            system = np.column_stack((return_tangent, -hatch_tangent))
            determinant = float(np.linalg.det(system))
            if abs(determinant) <= tolerance:
                continue
            straight_length, trim_distance = np.linalg.solve(
                system,
                hatch_start - second_base,
            )
            straight_length = float(straight_length)
            trim_distance = float(trim_distance)
            if (
                straight_length < -tolerance
                or trim_distance <= tolerance
                or trim_distance > maximum_trim_distance + tolerance
            ):
                continue

            trimmed_path = _trim_open_path_start(
                oriented_path,
                trim_distance,
                tolerance,
            )
            if trimmed_path is None:
                continue
            straight_end = first_end + max(0.0, straight_length) * return_tangent
            pieces: list[np.ndarray] = [
                np.asarray(prefix[:, :2], dtype=np.float32),
                first_turn[1:],
            ]
            if straight_length > tolerance:
                pieces.append(np.asarray([straight_end], dtype=np.float32))
            if abs(second_angle) > tolerance:
                second_turn = _constant_radius_turn_points(
                    straight_end,
                    return_tangent,
                    radius,
                    second_angle,
                    tolerance,
                )
                if second_turn is None:
                    continue
                pieces.append(second_turn[1:])
            pieces.append(np.asarray([trimmed_path[0, :2]], dtype=np.float32))
            bridge = _dedupe_consecutive(
                np.vstack(pieces).astype(np.float32),
                tolerance,
            )
            if bridge.shape[0] < 4:
                continue
            bridge_line = LineString(bridge[:, :2])
            if not bridge_line.is_simple or not safe_geometry.covers(bridge_line):
                continue
            return bridge, trimmed_path, radius
    return None


def _smooth_solid_wall_seam_dogleg_entry(
    assembled_prefix: np.ndarray,
    bridge: np.ndarray,
    target_radius: float,
    angle_threshold_degrees: float,
    safe_geometry,
    tolerance: float,
) -> tuple[np.ndarray, float] | None:
    """Add a C1 entry loop when a dogleg starts at a convex void corner."""

    if (
        assembled_prefix.shape[0] < 2
        or bridge.shape[0] < 2
        or target_radius <= tolerance
    ):
        return None
    start = np.asarray(bridge[0, :2], dtype=np.float64)
    end = np.asarray(bridge[1, :2], dtype=np.float64)
    incoming = start - np.asarray(assembled_prefix[-2, :2], dtype=np.float64)
    outgoing = end - start
    incoming_length = float(np.linalg.norm(incoming))
    outgoing_length = float(np.linalg.norm(outgoing))
    if min(incoming_length, outgoing_length) <= tolerance:
        return None
    incoming /= incoming_length
    outgoing /= outgoing_length
    interior_angle = math.degrees(
        math.acos(float(np.clip(np.dot(-incoming, outgoing), -1.0, 1.0)))
    )
    if interior_angle >= angle_threshold_degrees - 1e-6:
        return np.asarray(bridge[:, :2], dtype=np.float32), math.inf

    chord_length = float(np.linalg.norm(end - start))
    minimum_radius = target_radius * 0.45
    sample_count = 65
    parameters = np.linspace(0.0, 1.0, sample_count)[:, None]
    candidates: list[tuple[float, float, np.ndarray]] = []
    for incoming_factor in (0.3, 0.45, 0.6, 0.75, 1.0):
        for outgoing_factor in (0.15, 0.25, 0.35, 0.5):
            controls = np.asarray(
                [
                    start,
                    start + incoming * chord_length * incoming_factor,
                    end - outgoing * chord_length * outgoing_factor,
                    end,
                ],
                dtype=np.float64,
            )
            one_minus = 1.0 - parameters
            curve = (
                one_minus**3 * controls[0]
                + 3.0 * one_minus**2 * parameters * controls[1]
                + 3.0 * one_minus * parameters**2 * controls[2]
                + parameters**3 * controls[3]
            )
            first_derivative = (
                3.0 * one_minus**2 * (controls[1] - controls[0])
                + 6.0
                * one_minus
                * parameters
                * (controls[2] - controls[1])
                + 3.0 * parameters**2 * (controls[3] - controls[2])
            )
            second_derivative = (
                6.0
                * one_minus
                * (controls[2] - 2.0 * controls[1] + controls[0])
                + 6.0
                * parameters
                * (controls[3] - 2.0 * controls[2] + controls[1])
            )
            cross = np.abs(
                first_derivative[:, 0] * second_derivative[:, 1]
                - first_derivative[:, 1] * second_derivative[:, 0]
            )
            speed = np.linalg.norm(first_derivative, axis=1)
            radii = np.divide(
                speed**3,
                cross,
                out=np.full(sample_count, np.inf, dtype=np.float64),
                where=cross > tolerance * tolerance,
            )
            actual_radius = float(np.min(radii))
            if actual_radius < minimum_radius - tolerance:
                continue
            curve = np.asarray(curve, dtype=np.float32)
            curve[0] = np.asarray(start, dtype=np.float32)
            curve[-1] = np.asarray(end, dtype=np.float32)
            proposal = _dedupe_consecutive(
                np.vstack((curve, bridge[2:, :2])).astype(np.float32),
                tolerance,
            )
            local_chain = _dedupe_consecutive(
                np.vstack((assembled_prefix[-2:, :2], proposal[1:, :2])).astype(
                    np.float32
                ),
                tolerance,
            )
            proposal_line = LineString(proposal[:, :2])
            if (
                not proposal_line.is_simple
                or not LineString(local_chain[:, :2]).is_simple
                or not safe_geometry.covers(proposal_line)
            ):
                continue
            candidates.append(
                (actual_radius, -float(LineString(curve).length), proposal)
            )
    if not candidates:
        return None
    actual_radius, _, proposal = max(candidates, key=lambda item: (item[0], item[1]))
    return proposal, actual_radius


def _solid_wall_seam_dogleg(
    incoming_connector: np.ndarray,
    endpoint: int,
    oriented_path: np.ndarray,
    assembled_prefix: list[np.ndarray],
    paths: list[np.ndarray],
    path_lines: dict[int, LineString],
    endpoint_points: dict[int, np.ndarray],
    scan_levels: dict[int, float],
    boundary_rings: list[_InfillBoundaryRing],
    minimum_scan_spacing: float,
    maximum_scan_spacing: float | None,
    bead_width: float,
    wall_clearance: float,
    safe_geometry,
    tolerance: float,
    compensated_gaps: set[tuple[int, int]],
    compensated_lines: list[LineString],
    accepted_connector_linework,
    *,
    smoothing_angle_degrees: float | None = None,
    smoothing_corner_cut: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fold a short wall-gap correction into an existing solid-fill turn."""

    if oriented_path.shape[0] < 2:
        return None
    gap = _solid_wall_seam_gap_arc(
        endpoint,
        endpoint_points,
        scan_levels,
        boundary_rings,
        minimum_scan_spacing,
        maximum_scan_spacing,
        bead_width,
        wall_clearance,
        tolerance,
        compensated_gaps,
        incoming_connector=incoming_connector,
    )
    if gap is None:
        return None
    gap_arc, gap_key = gap
    prefix = _solid_wall_seam_gap_prefix(
        gap_arc,
        bead_width,
        wall_clearance,
        tolerance,
    )
    if prefix is None:
        return None

    bead_radius = bead_width * 0.5
    path_length = _open_path_length(oriented_path)
    trim_distance = min(bead_radius, path_length * 0.25)
    if trim_distance <= tolerance:
        return None
    trimmed_path = _trim_open_path_start(oriented_path, trim_distance, tolerance)
    if trimmed_path is None:
        return None

    smoothed_return = (
        _smooth_solid_wall_seam_dogleg_return(
            prefix,
            oriented_path,
            smoothing_corner_cut,
            min(bead_width * 1.5, path_length * 0.25),
            safe_geometry,
            tolerance,
        )
        if smoothing_corner_cut is not None and smoothing_corner_cut > tolerance
        else None
    )
    if smoothed_return is not None:
        bridge, trimmed_path, _ = smoothed_return
    else:
        return_point = trimmed_path[0, :2]
        bridge = _dedupe_consecutive(
            np.vstack((prefix, np.asarray(return_point, dtype=np.float32))),
            tolerance,
        )
    if smoothing_corner_cut is not None and smoothing_corner_cut > tolerance:
        smoothed_entry = _smooth_solid_wall_seam_dogleg_entry(
            np.asarray(assembled_prefix, dtype=np.float32),
            bridge,
            smoothing_corner_cut,
            (
                DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES
                if smoothing_angle_degrees is None
                else smoothing_angle_degrees
            ),
            safe_geometry,
            tolerance,
        )
        if smoothed_entry is not None:
            bridge, _ = smoothed_entry
    return_point = trimmed_path[0, :2]
    if bridge.shape[0] < 3:
        return None
    bridge_line = LineString(bridge)
    prefix_line = LineString(prefix)
    if (
        not bridge_line.is_simple
        or not safe_geometry.covers(bridge_line)
        or _has_unexpected_linework_intersection(
            bridge_line,
            accepted_connector_linework,
            (bridge[0, :2],),
            tolerance,
        )
        or _has_unexpected_linework_intersection(
            bridge_line,
            path_lines[endpoint // 2],
            (bridge[0, :2], bridge[-1, :2]),
            tolerance,
        )
        or any(
            not bridge_line.disjoint(existing)
            for existing in compensated_lines
        )
    ):
        return None

    boundary_vector = _last_nondegenerate_segment(prefix, tolerance)
    return_vector = np.asarray(return_point - prefix[-1, :2], dtype=np.float64)
    hatch_vector = _first_nondegenerate_segment(trimmed_path, tolerance)
    if boundary_vector is None or hatch_vector is None:
        return None
    if (
        _acute_angle_degrees(boundary_vector, return_vector, tolerance) < 10.0
        or _acute_angle_degrees(return_vector, hatch_vector, tolerance) < 10.0
    ):
        return None

    return_chord = LineString(
        [
            (float(prefix[-1, 0]), float(prefix[-1, 1])),
            (float(return_point[0]), float(return_point[1])),
        ]
    )
    incident_index = endpoint // 2
    for other_index, other_line in path_lines.items():
        if other_index == incident_index:
            continue
        if (
            _has_unexpected_linework_intersection(
                bridge_line,
                other_line,
                (),
                tolerance,
            )
            or return_chord.distance(other_line) <= tolerance * 2.0
        ):
            return None

    candidate_chain = _dedupe_consecutive(
        np.vstack(
            (
                np.asarray(assembled_prefix, dtype=np.float32),
                bridge[1:],
                trimmed_path[1:],
            )
        ),
        tolerance * 2.0,
    )
    if candidate_chain.shape[0] >= 2 and not LineString(candidate_chain).is_simple:
        return None

    original_trim = LineString(
        [
            (float(oriented_path[0, 0]), float(oriented_path[0, 1])),
            (float(return_point[0]), float(return_point[1])),
        ]
    )
    deposited = bridge_line.buffer(
        max(bead_radius - tolerance * 2.0, bead_radius * 0.95),
        cap_style="round",
        join_style="round",
    )
    if original_trim.difference(deposited).length > tolerance * 10.0:
        return None
    if not _solid_wall_seam_dogleg_is_finishable(
        candidate_chain,
        bridge,
        np.asarray(assembled_prefix, dtype=np.float32),
        trimmed_path,
        safe_geometry,
        bead_width,
        smoothing_angle_degrees,
        smoothing_corner_cut,
        tolerance,
    ):
        return None
    compensated_gaps.add(gap_key)
    compensated_lines.append(bridge_line)
    return bridge, trimmed_path


def _solid_wall_seam_dogleg_is_finishable(
    candidate_chain: np.ndarray,
    bridge: np.ndarray,
    assembled_prefix: np.ndarray,
    trimmed_path: np.ndarray,
    safe_geometry,
    bead_width: float,
    smoothing_angle_degrees: float | None,
    smoothing_corner_cut: float | None,
    tolerance: float,
) -> bool:
    """Reject a seam detour only when the final corner pipeline cannot fix it."""

    angle_threshold = (
        DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES
        if smoothing_angle_degrees is None
        else smoothing_angle_degrees
    )
    corner_cut = (
        min(
            bead_width * DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR,
            bead_width * 0.15,
        )
        if smoothing_corner_cut is None
        else max(0.0, smoothing_corner_cut)
    )
    initial_corner_cut = min(corner_cut, bead_width * 0.04)
    effective_span = max(tolerance * 20.0, bead_width * 0.005)
    context_distance = max(
        bead_width * 2.0,
        corner_cut * 4.0,
        effective_span * 10.0,
    )
    before = _solid_wall_seam_trial_context(
        assembled_prefix,
        context_distance,
        tolerance,
        from_end=True,
    )
    after = _solid_wall_seam_trial_context(
        trimmed_path,
        context_distance,
        tolerance,
        from_end=False,
    )
    local_candidate = _dedupe_consecutive(
        np.vstack((before, bridge[1:], after[1:])).astype(np.float32),
        tolerance * 2.0,
    )
    bridge_line = LineString(bridge[:, :2])
    # The dogleg builder may already have produced an analytical tangent
    # return with the requested physical radius.  Do not run that valid arc
    # through the generic corner fitter a second time: re-fitting samples of
    # an existing arc is precisely what can create a short backwards hook.
    if not _solid_wall_seam_trial_fails(
        [local_candidate],
        bridge_line,
        corner_cut,
        angle_threshold,
        effective_span,
        tolerance,
    ):
        return True
    local_trial = _solid_wall_seam_trial_finish(
        local_candidate,
        safe_geometry,
        initial_corner_cut,
        corner_cut,
        angle_threshold,
        effective_span,
        tolerance,
    )
    if not _solid_wall_seam_trial_fails(
        local_trial,
        bridge_line,
        corner_cut,
        angle_threshold,
        effective_span,
        tolerance,
    ):
        return True

    # A clipped context is intentionally conservative and can make a valid
    # fillet look impossible at one of its artificial ends.  Confirm a local
    # failure on the complete component assembled so far before dropping a
    # coverage-compensating dogleg.
    full_trial = _solid_wall_seam_trial_finish(
        candidate_chain,
        safe_geometry,
        initial_corner_cut,
        corner_cut,
        angle_threshold,
        effective_span,
        tolerance,
    )
    return not _solid_wall_seam_trial_fails(
        full_trial,
        bridge_line,
        corner_cut,
        angle_threshold,
        effective_span,
        tolerance,
    )


def _solid_wall_seam_trial_context(
    path: np.ndarray,
    context_distance: float,
    tolerance: float,
    *,
    from_end: bool,
) -> np.ndarray:
    points = np.asarray(path[:, :2], dtype=np.float32)
    if points.shape[0] < 2:
        return points
    working = points[::-1].copy() if from_end else points
    if _open_path_length(working) > context_distance + tolerance:
        clipped = _open_path_prefix(working, context_distance, tolerance)
        if clipped is not None:
            working = clipped
    return working[::-1].copy() if from_end else working


def _solid_wall_seam_trial_finish(
    path: np.ndarray,
    safe_geometry,
    initial_corner_cut: float,
    corner_cut: float,
    angle_threshold_degrees: float,
    effective_span: float,
    tolerance: float,
) -> list[np.ndarray]:
    cleaned = _remove_smoothing_micro_segments(
        path,
        safe_geometry,
        effective_span,
        tolerance,
    )
    initially_smoothed = [cleaned]
    if initial_corner_cut > tolerance:
        initially_smoothed = _smooth_resin_infill_paths(
            initially_smoothed,
            safe_geometry,
            initial_corner_cut,
            angle_threshold_degrees,
            tolerance,
            cut_fraction=0.3,
        )
    final_inputs = [
        _remove_smoothing_micro_segments(
            candidate,
            safe_geometry,
            effective_span,
            tolerance,
        )
        for candidate in initially_smoothed
    ]
    if corner_cut <= tolerance:
        return final_inputs
    finally_smoothed = _smooth_resin_infill_paths(
        final_inputs,
        safe_geometry,
        corner_cut,
        angle_threshold_degrees,
        tolerance,
        cut_fraction=0.3,
    )
    return [
        _smooth_effective_path_corners(
            candidate,
            safe_geometry,
            corner_cut,
            angle_threshold_degrees,
            effective_span,
            tolerance,
        )
        for candidate in finally_smoothed
    ]


def _solid_wall_seam_trial_fails(
    paths: list[np.ndarray],
    bridge_line: LineString,
    corner_cut: float,
    angle_threshold_degrees: float,
    effective_span: float,
    tolerance: float,
) -> bool:
    audit_distance = corner_cut + effective_span + tolerance * 10.0
    for path in paths:
        if path.shape[0] < 2 or not LineString(path[:, :2]).is_simple:
            return True
        for _, index, _, _ in _effective_path_corner_candidates(
            path,
            effective_span,
            angle_threshold_degrees,
            tolerance,
        ):
            point = Point(float(path[index, 0]), float(path[index, 1]))
            if bridge_line.distance(point) <= audit_distance:
                return True
    return False


def _has_unexpected_linework_intersection(
    candidate: LineString,
    existing,
    allowed_points: tuple[np.ndarray, ...],
    tolerance: float,
) -> bool:
    if existing.is_empty or candidate.disjoint(existing):
        return False
    intersection = candidate.intersection(existing)
    if not allowed_points:
        return not intersection.is_empty
    allowed = unary_union(
        [
            Point(float(point[0]), float(point[1])).buffer(tolerance * 10.0)
            for point in allowed_points
        ]
    )
    return not intersection.difference(allowed).is_empty


def _solid_wall_seam_gap_arc(
    endpoint: int,
    endpoint_points: dict[int, np.ndarray],
    scan_levels: dict[int, float],
    boundary_rings: list[_InfillBoundaryRing],
    minimum_scan_spacing: float,
    maximum_scan_spacing: float | None,
    bead_width: float,
    wall_clearance: float,
    tolerance: float,
    compensated_gaps: set[tuple[int, int]],
    *,
    incoming_connector: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    point = endpoint_points[endpoint]
    point_geometry = Point(float(point[0]), float(point[1]))
    snap_tolerance = max(tolerance * 20.0, min(bead_width * 0.02, 0.05))
    minimum_delta = minimum_scan_spacing - max(
        minimum_scan_spacing * 0.02,
        tolerance * 20.0,
    )
    maximum_delta = (
        maximum_scan_spacing
        if maximum_scan_spacing is not None
        else minimum_scan_spacing * 1.1
    ) + max(minimum_scan_spacing * 0.02, tolerance * 20.0)

    best: tuple[float, np.ndarray, tuple[int, int]] | None = None
    for ring in boundary_rings:
        if ring.line.distance(point_geometry) > snap_tolerance:
            continue
        direction = _solid_wall_seam_ring_direction(
            ring,
            incoming_connector,
            point,
            snap_tolerance,
        )
        if incoming_connector is not None and direction is None:
            continue
        directions = (direction,) if direction is not None else (True, False)
        point_distance = float(ring.line.project(point_geometry))
        total_length = float(ring.cumulative_lengths[-1])
        for forward in directions:
            for candidate, candidate_point in endpoint_points.items():
                if candidate == endpoint or candidate // 2 == endpoint // 2:
                    continue
                gap_key = tuple(sorted((endpoint, candidate)))
                if gap_key in compensated_gaps:
                    continue
                level_delta = abs(
                    scan_levels[candidate // 2] - scan_levels[endpoint // 2]
                )
                if not (minimum_delta <= level_delta <= maximum_delta):
                    continue
                candidate_geometry = Point(
                    float(candidate_point[0]),
                    float(candidate_point[1]),
                )
                if ring.line.distance(candidate_geometry) > snap_tolerance:
                    continue
                candidate_distance = float(ring.line.project(candidate_geometry))
                travel = (
                    (candidate_distance - point_distance) % total_length
                    if forward
                    else (point_distance - candidate_distance) % total_length
                )
                if not (
                    tolerance < travel <= max(bead_width, maximum_delta) * 8.0
                ):
                    continue
                arc = (
                    _forward_infill_ring_arc(
                        ring,
                        point_distance,
                        candidate_distance,
                        tolerance,
                    )
                    if forward
                    else _forward_infill_ring_arc(
                        ring,
                        candidate_distance,
                        point_distance,
                        tolerance,
                    )[::-1].copy()
                )
                arc = _dedupe_consecutive(arc, tolerance)
                arc_length = _open_path_length(arc)
                if arc_length <= tolerance:
                    continue
                if best is None or arc_length < best[0]:
                    best = (arc_length, arc, gap_key)
    return None if best is None else (best[1], best[2])


def _solid_wall_seam_ring_direction(
    ring: _InfillBoundaryRing,
    incoming_connector: np.ndarray | None,
    endpoint: np.ndarray,
    snap_tolerance: float,
) -> bool | None:
    if incoming_connector is None or incoming_connector.shape[0] < 2:
        return None
    previous = incoming_connector[-2, :2]
    previous_geometry = Point(float(previous[0]), float(previous[1]))
    if ring.line.distance(previous_geometry) > snap_tolerance:
        return None
    total_length = float(ring.cumulative_lengths[-1])
    previous_distance = float(ring.line.project(previous_geometry))
    endpoint_distance = float(
        ring.line.project(Point(float(endpoint[0]), float(endpoint[1])))
    )
    forward_travel = (endpoint_distance - previous_distance) % total_length
    backward_travel = (previous_distance - endpoint_distance) % total_length
    return forward_travel <= backward_travel


def _solid_wall_seam_gap_prefix(
    gap_arc: np.ndarray,
    bead_width: float,
    wall_clearance: float,
    tolerance: float,
) -> np.ndarray | None:
    bead_radius = bead_width * 0.5
    seam_normal_offset = max(0.0, wall_clearance - bead_radius)
    if seam_normal_offset >= bead_radius - tolerance:
        return None
    cap_half_width = math.sqrt(
        max(0.0, bead_radius * bead_radius - seam_normal_offset * seam_normal_offset)
    )
    gap_length = _open_path_length(gap_arc)
    correction_length = gap_length - 2.0 * cap_half_width
    numerical_margin = max(tolerance * 20.0, bead_width * 1e-4)
    if correction_length <= numerical_margin:
        return None
    correction_length = min(
        gap_length - numerical_margin,
        correction_length + numerical_margin,
    )
    return _open_path_prefix(gap_arc, correction_length, tolerance)


def _open_path_prefix(
    path: np.ndarray,
    target_length: float,
    tolerance: float,
) -> np.ndarray | None:
    if path.shape[0] < 2 or target_length <= tolerance:
        return None
    points = [np.asarray(path[0, :2], dtype=np.float32)]
    remaining = target_length
    for start, end in zip(path[:-1, :2], path[1:, :2]):
        delta = np.asarray(end - start, dtype=np.float64)
        segment_length = float(np.linalg.norm(delta))
        if segment_length <= tolerance:
            continue
        if remaining >= segment_length - tolerance:
            points.append(np.asarray(end, dtype=np.float32))
            remaining -= segment_length
            if remaining <= tolerance:
                break
            continue
        points.append(
            np.asarray(start, dtype=np.float64)
            + delta * (remaining / segment_length)
        )
        remaining = 0.0
        break
    if remaining > tolerance or len(points) < 2:
        return None
    return _dedupe_consecutive(np.asarray(points, dtype=np.float32), tolerance)


def _trim_open_path_start(
    path: np.ndarray,
    trim_length: float,
    tolerance: float,
) -> np.ndarray | None:
    if path.shape[0] < 2 or trim_length <= tolerance:
        return None
    remaining = trim_length
    for index, (start, end) in enumerate(zip(path[:-1, :2], path[1:, :2])):
        delta = np.asarray(end - start, dtype=np.float64)
        segment_length = float(np.linalg.norm(delta))
        if segment_length <= tolerance:
            continue
        if remaining >= segment_length - tolerance:
            remaining -= segment_length
            continue
        first = np.asarray(start, dtype=np.float64) + delta * (remaining / segment_length)
        return _dedupe_consecutive(
            np.vstack((np.asarray(first, dtype=np.float32), path[index + 1 :, :2])),
            tolerance,
        )
    return None


def _last_nondegenerate_segment(
    path: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    for start, end in reversed(list(zip(path[:-1, :2], path[1:, :2]))):
        delta = np.asarray(end - start, dtype=np.float64)
        if float(np.linalg.norm(delta)) > tolerance:
            return delta
    return None


def _first_nondegenerate_segment(
    path: np.ndarray,
    tolerance: float,
) -> np.ndarray | None:
    for start, end in zip(path[:-1, :2], path[1:, :2]):
        delta = np.asarray(end - start, dtype=np.float64)
        if float(np.linalg.norm(delta)) > tolerance:
            return delta
    return None


def _acute_angle_degrees(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> float:
    first_length = float(np.linalg.norm(first))
    second_length = float(np.linalg.norm(second))
    if first_length <= tolerance or second_length <= tolerance:
        return 0.0
    cosine = abs(float(np.dot(first, second)) / (first_length * second_length))
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _infill_scan_levels(
    paths: list[np.ndarray],
    open_indices: list[int],
    tolerance: float,
) -> dict[int, float]:
    direction: np.ndarray | None = None
    longest = 0.0
    for index in open_indices:
        candidate = np.asarray(paths[index][-1, :2] - paths[index][0, :2], dtype=np.float64)
        length = float(np.linalg.norm(candidate))
        if length > longest:
            direction = candidate / length
            longest = length
    if direction is None or longest <= tolerance:
        return {index: float(index) for index in open_indices}
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    return {
        index: float(
            np.dot(
                (paths[index][0, :2] + paths[index][-1, :2]) * 0.5,
                normal,
            )
        )
        for index in open_indices
    }


def _infill_endpoint_connector(
    first_point: np.ndarray,
    second_point: np.ndarray,
    boundary_rings: list[_InfillBoundaryRing],
    safe_geometry,
    spacing: float,
    tolerance: float,
) -> np.ndarray | None:
    snap_tolerance = max(tolerance * 20.0, min(spacing * 0.02, 0.05))
    first = Point(float(first_point[0]), float(first_point[1]))
    second = Point(float(second_point[0]), float(second_point[1]))
    boundary_candidates: list[np.ndarray] = []
    for ring in boundary_rings:
        if ring.line.distance(first) > snap_tolerance or ring.line.distance(second) > snap_tolerance:
            continue
        connector = _shortest_infill_ring_arc(
            ring,
            first_point,
            second_point,
            tolerance,
        )
        connector_line = LineString(
            [(float(point[0]), float(point[1])) for point in connector]
        )
        direct_distance = float(np.linalg.norm(second_point - first_point))
        maximum_length = max(spacing * 8.0, direct_distance * 1.5)
        if connector_line.length <= maximum_length and safe_geometry.covers(connector_line):
            boundary_candidates.append(connector)
    if boundary_candidates:
        return min(boundary_candidates, key=_open_path_length)

    direct_distance = float(np.linalg.norm(second_point - first_point))
    if not (tolerance < direct_distance <= max(spacing * 1.5, tolerance * 10.0)):
        return None
    direct = np.asarray([first_point, second_point], dtype=np.float32)
    direct_line = LineString([(float(point[0]), float(point[1])) for point in direct])
    return direct if safe_geometry.covers(direct_line) else None


def _shortest_infill_ring_arc(
    ring: _InfillBoundaryRing,
    first_point: np.ndarray,
    second_point: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    first_distance = float(
        ring.line.project(Point(float(first_point[0]), float(first_point[1])))
    )
    second_distance = float(
        ring.line.project(Point(float(second_point[0]), float(second_point[1])))
    )
    forward = _forward_infill_ring_arc(ring, first_distance, second_distance, tolerance)
    backward = _forward_infill_ring_arc(ring, second_distance, first_distance, tolerance)[::-1].copy()
    arc = forward if _open_path_length(forward) <= _open_path_length(backward) else backward
    return _dedupe_consecutive(
        np.vstack(
            [
                np.asarray(first_point[:2], dtype=np.float32),
                arc,
                np.asarray(second_point[:2], dtype=np.float32),
            ]
        ),
        tolerance,
    )


def _forward_infill_ring_arc(
    ring: _InfillBoundaryRing,
    start_distance: float,
    end_distance: float,
    tolerance: float,
) -> np.ndarray:
    total_length = float(ring.cumulative_lengths[-1])
    travel = (end_distance - start_distance) % total_length
    target_end = start_distance + travel
    vertex_distances = ring.cumulative_lengths[1:]
    vertices = ring.coordinates[1:]
    if target_end > total_length:
        vertex_distances = np.concatenate((vertex_distances, vertex_distances + total_length))
        vertices = np.vstack((vertices, vertices))
    mask = (
        (vertex_distances > start_distance + tolerance)
        & (vertex_distances < target_end - tolerance)
    )
    start_point = np.asarray(ring.line.interpolate(start_distance).coords[0], dtype=np.float32)
    end_point = np.asarray(ring.line.interpolate(end_distance).coords[0], dtype=np.float32)
    return _dedupe_consecutive(
        np.vstack((start_point, vertices[mask], end_point)).astype(np.float32),
        tolerance,
    )


def _connect_concentric_infill_paths(
    paths: list[np.ndarray],
    geometry,
    spacing: float,
    minimum_clearance: float,
    tolerance: float,
) -> list[np.ndarray]:
    """Join adjacent closed rings without retracing either ring."""

    closed_indices = [
        index
        for index, path in enumerate(paths)
        if path.shape[0] >= 3 and _is_closed_path(path, tolerance)
    ]
    if len(closed_indices) < 2:
        return paths

    path_lines = {
        index: LineString(
            [(float(point[0]), float(point[1])) for point in paths[index][:, :2]]
        )
        for index in closed_indices
    }
    safe_geometry = geometry.buffer(max(tolerance * 10.0, 1e-7), join_style="round")
    unused = set(closed_indices)
    connected_indices: set[int] = set()
    connected_paths: list[np.ndarray] = []

    while unused:
        current_index = min(unused)
        unused.remove(current_index)
        connected_indices.add(current_index)
        chain = np.asarray(paths[current_index][:, :2], dtype=np.float32).copy()

        while unused:
            current_point = np.asarray(chain[-1, :2], dtype=np.float32)
            candidates: list[tuple[float, int, np.ndarray, np.ndarray]] = []
            for candidate_index in sorted(unused):
                candidate_line = path_lines[candidate_index]
                anchor_distance = candidate_line.project(
                    Point(float(current_point[0]), float(current_point[1]))
                )
                anchor = np.asarray(
                    candidate_line.interpolate(anchor_distance).coords[0],
                    dtype=np.float32,
                )
                connector_points = np.asarray([current_point, anchor], dtype=np.float32)
                connector_length = _open_path_length(connector_points)
                if not (
                    tolerance
                    < connector_length
                    <= max(spacing * 1.6, minimum_clearance * 1.6, tolerance * 10.0)
                ):
                    continue
                connector = LineString(
                    [(float(point[0]), float(point[1])) for point in connector_points]
                )
                if not safe_geometry.covers(connector):
                    continue
                if not _centerline_connector_is_clear(
                    connector,
                    path_lines,
                    {
                        current_index: Point(float(current_point[0]), float(current_point[1])),
                        candidate_index: Point(float(anchor[0]), float(anchor[1])),
                    },
                    [],
                    tolerance,
                    minimum_clearance,
                ):
                    continue
                rerooted = _reroot_closed_path(
                    paths[candidate_index],
                    anchor,
                    tolerance,
                )
                candidates.append(
                    (connector_length, candidate_index, connector_points, rerooted)
                )

            if not candidates:
                break
            _, candidate_index, connector_points, rerooted = min(
                candidates,
                key=lambda item: (item[0], item[1]),
            )
            chain = _dedupe_consecutive(
                np.vstack((chain, connector_points[1:], rerooted[1:])),
                tolerance,
            )
            unused.remove(candidate_index)
            connected_indices.add(candidate_index)
            current_index = candidate_index

        connected_paths.append(chain)

    for index, path in enumerate(paths):
        if index not in connected_indices:
            connected_paths.append(path)
    return connected_paths


def _reroot_closed_path(
    path: np.ndarray,
    anchor: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    points = _dedupe_consecutive(np.asarray(path[:, :2], dtype=np.float32), tolerance)
    if not _is_closed_path(points, tolerance):
        return points
    open_points = points[:-1]
    if open_points.shape[0] < 2:
        return points

    ring_points = np.vstack((open_points, open_points[0]))
    segment_lengths = np.linalg.norm(np.diff(ring_points, axis=0), axis=1)
    ring = LineString(
        [(float(point[0]), float(point[1])) for point in ring_points]
    )
    anchor_distance = float(ring.project(Point(float(anchor[0]), float(anchor[1]))))
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    segment_index = min(
        int(np.searchsorted(cumulative, anchor_distance, side="right") - 1),
        open_points.shape[0] - 1,
    )
    projected_anchor = np.asarray(
        ring.interpolate(anchor_distance).coords[0],
        dtype=np.float32,
    )
    rerooted = np.vstack(
        (
            projected_anchor,
            open_points[segment_index + 1 :],
            open_points[: segment_index + 1],
            projected_anchor,
        )
    )
    return _dedupe_consecutive(rerooted, tolerance)
def _connect_resin_infill_paths(
    paths: list[np.ndarray],
    geometry,
    spacing: float,
    tolerance: float,
    *,
    minimum_clearance: float = 0.0,
) -> list[np.ndarray]:
    """Chain safe open infill paths using an endpoint graph.

    This adapts the chaining part of PrusaSlicer's ``Fill::connect_infill``
    for the path-only contract. Perimeter hooks and extrusion-dependent
    decisions are intentionally omitted.
    """
    if len(paths) < 2 or spacing <= 0 or geometry.is_empty:
        return paths

    open_indices = [
        index
        for index, path in enumerate(paths)
        if path.shape[0] >= 2 and not _is_closed_path(path, tolerance)
    ]
    if len(open_indices) < 2:
        return paths

    safe_geometry = geometry.buffer(max(tolerance * 10.0, 1e-7), join_style="round")
    path_lines = {
        index: LineString(
            [(float(point[0]), float(point[1])) for point in path[:, :2]]
        )
        for index, path in enumerate(paths)
        if path.shape[0] >= 2
    }
    max_connector_length = max(spacing * 1.5, tolerance * 10.0)
    endpoint_points = {
        2 * index + side: np.asarray(
            paths[index][0 if side == 0 else -1, :2], dtype=np.float32
        )
        for index in open_indices
        for side in (0, 1)
    }

    candidates: list[tuple[float, int, int]] = []
    for first_position, first_index in enumerate(open_indices):
        for second_index in open_indices[first_position + 1 :]:
            for first_side in (0, 1):
                first_endpoint = 2 * first_index + first_side
                first_point = endpoint_points[first_endpoint]
                for second_side in (0, 1):
                    second_endpoint = 2 * second_index + second_side
                    distance = float(
                        np.linalg.norm(first_point - endpoint_points[second_endpoint])
                    )
                    if tolerance < distance <= max_connector_length:
                        candidates.append((distance, first_endpoint, second_endpoint))
    if not candidates:
        return paths
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    parent = {index: index for index in open_indices}
    endpoint_used: set[int] = set()

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            next_index = parent[index]
            parent[index] = root
            index = next_index
        return root

    accepted: list[tuple[int, int, np.ndarray]] = []
    connector_by_endpoint: dict[int, tuple[int, np.ndarray]] = {}

    for distance, first_endpoint, second_endpoint in candidates:
        if first_endpoint in endpoint_used or second_endpoint in endpoint_used:
            continue
        first_index = first_endpoint // 2
        second_index = second_endpoint // 2
        if find(first_index) == find(second_index):
            # Do not close a chain into a loop. Closed contours are kept as
            # their own paths because the output has no travel/extrusion data.
            continue

        first_point = endpoint_points[first_endpoint]
        second_point = endpoint_points[second_endpoint]
        connector = LineString(
            [
                (float(first_point[0]), float(first_point[1])),
                (float(second_point[0]), float(second_point[1])),
            ]
        )
        if not safe_geometry.covers(connector):
            continue
        if not _resin_connector_is_clear(
            connector,
            paths,
            path_lines,
            first_endpoint,
            second_endpoint,
            accepted,
            tolerance,
            minimum_clearance=minimum_clearance,
        ):
            continue

        endpoint_used.update((first_endpoint, second_endpoint))
        connector_points = np.asarray([first_point, second_point], dtype=np.float32)
        accepted.append((first_endpoint, second_endpoint, connector_points))
        connector_by_endpoint[first_endpoint] = (second_endpoint, connector_points)
        connector_by_endpoint[second_endpoint] = (first_endpoint, connector_points)
        parent[find(first_index)] = find(second_index)

    if not accepted:
        return paths

    components: dict[int, list[int]] = defaultdict(list)
    for index in open_indices:
        components.setdefault(find(index), []).append(index)
    component_starts = sorted(components.values(), key=lambda indexes: min(indexes))
    connected_paths: list[np.ndarray] = []
    connected_indices: set[int] = set()

    for component in component_starts:
        start_endpoint = next(
            (
                endpoint
                for index in component
                for endpoint in (2 * index, 2 * index + 1)
                if endpoint not in connector_by_endpoint
            ),
            None,
        )
        if start_endpoint is None:
            continue

        chain_points: list[np.ndarray] = []
        current_endpoint = start_endpoint
        while current_endpoint is not None:
            index = current_endpoint // 2
            side = current_endpoint % 2
            if index in connected_indices:
                break
            connected_indices.add(index)
            path = np.asarray(paths[index][:, :2], dtype=np.float32)
            oriented = path if side == 0 else path[::-1].copy()
            if not chain_points:
                chain_points.extend(oriented)
            else:
                chain_points.extend(oriented[1:])

            mate_endpoint = 2 * index + (1 - side)
            connection = connector_by_endpoint.get(mate_endpoint)
            if connection is None:
                current_endpoint = None
                continue
            next_endpoint, connector_points = connection
            if np.allclose(connector_points[0], endpoint_points[mate_endpoint]):
                oriented_connector = connector_points
            else:
                oriented_connector = connector_points[::-1].copy()
            chain_points.extend(oriented_connector[1:])
            current_endpoint = next_endpoint

        if len(chain_points) >= 2:
            connected_paths.append(np.asarray(chain_points, dtype=np.float32))

    # Preserve any closed paths and any malformed/singleton paths untouched.
    for index, path in enumerate(paths):
        if index not in connected_indices:
            connected_paths.append(path)
    return connected_paths


def _resin_connector_is_clear(
    connector: LineString,
    paths: list[np.ndarray],
    path_lines: dict[int, LineString],
    first_endpoint: int,
    second_endpoint: int,
    accepted: list[tuple[int, int, np.ndarray]],
    tolerance: float,
    *,
    minimum_clearance: float = 0.0,
    maximum_overlap_spacing: float | None = None,
    bead_width: float | None = None,
    safe_geometry=None,
    smoothing_angle_degrees: float | None = None,
    smoothing_corner_cut: float | None = None,
) -> bool:
    first_index = first_endpoint // 2
    second_index = second_endpoint // 2
    allowed_intersections = {
        first_index: Point(
            tuple(
                float(value)
                for value in paths[first_index][0 if first_endpoint % 2 == 0 else -1, :2]
            )
        ),
        second_index: Point(
            tuple(
                float(value)
                for value in paths[second_index][0 if second_endpoint % 2 == 0 else -1, :2]
            )
        ),
    }
    centerline_is_clear = _centerline_connector_is_clear(
        connector,
        path_lines,
        allowed_intersections,
        accepted,
        tolerance,
        minimum_clearance,
    )
    if not centerline_is_clear:
        return False
    if maximum_overlap_spacing is None:
        return True
    if bead_width is None:
        raise ValueError("bead_width is required for strict connector spacing")
    first_path = np.asarray(paths[first_index][:, :2], dtype=np.float32)
    if first_endpoint % 2 == 0:
        first_path = first_path[::-1].copy()
    second_path = np.asarray(paths[second_index][:, :2], dtype=np.float32)
    if second_endpoint % 2 == 1:
        second_path = second_path[::-1].copy()
    connector_path = np.asarray(connector.coords, dtype=np.float32)
    if not _close(connector_path[0], first_path[-1], tolerance):
        connector_path = connector_path[::-1].copy()
    candidate = _dedupe_consecutive(
        np.vstack(
            (
                first_path,
                connector_path[1:],
                second_path[1:],
            )
        ).astype(np.float32),
        tolerance,
    )
    candidate_paths = [candidate]
    if (
        safe_geometry is not None
        and smoothing_corner_cut is not None
        and smoothing_corner_cut > tolerance
    ):
        candidate_paths = _smooth_resin_infill_paths(
            candidate_paths,
            safe_geometry,
            smoothing_corner_cut,
            (
                DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES
                if smoothing_angle_degrees is None
                else smoothing_angle_degrees
            ),
            tolerance,
            cut_fraction=0.3,
        )
    return _solid_fill_spacing_postcondition(
        candidate_paths,
        Polygon(),
        maximum_overlap_spacing,
        tolerance,
        bead_width=bead_width,
        allow_boundary_bridges=True,
    )


def _centerline_connector_is_clear(
    connector: LineString,
    path_lines: dict[int, LineString],
    allowed_intersections: dict[int, Point],
    accepted: list[tuple[int, int, np.ndarray]],
    tolerance: float,
    minimum_clearance: float,
) -> bool:
    endpoint_buffer = max(tolerance * 10.0, 1e-7)
    bounds_padding = max(endpoint_buffer, minimum_clearance)
    connector_bounds = connector.bounds

    for index, path_line in path_lines.items():
        if not _bounds_overlap(connector_bounds, path_line.bounds, bounds_padding):
            continue
        intersection = connector.intersection(path_line)
        if not intersection.is_empty:
            if index not in allowed_intersections:
                return False
            residual = intersection.difference(
                allowed_intersections[index].buffer(endpoint_buffer, join_style="round")
            )
            if not residual.is_empty:
                return False

        if minimum_clearance > tolerance:
            clearance_line = path_line
            if index in allowed_intersections:
                # Consecutive segments intentionally share material at their
                # turn.  Ignore a short two-pitch neighborhood, then enforce
                # normal clearance so a shallow connector cannot run alongside
                # its incident path and create an extended overfill strip.
                clearance_line = path_line.difference(
                    allowed_intersections[index].buffer(
                        minimum_clearance * 2.0,
                        join_style="round",
                    )
                )
            if (
                not clearance_line.is_empty
                and connector.distance(clearance_line) < minimum_clearance - tolerance
            ):
                return False

    for _, _, existing_connector in accepted:
        existing_line = LineString(
            [(float(point[0]), float(point[1])) for point in existing_connector[:, :2]]
        )
        if not _bounds_overlap(connector_bounds, existing_line.bounds, bounds_padding):
            continue
        if not connector.disjoint(existing_line):
            return False
        if (
            minimum_clearance > tolerance
            and connector.distance(existing_line) < minimum_clearance - tolerance
        ):
            return False
    return True


def _bounds_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    padding: float,
) -> bool:
    return not (
        first[2] < second[0] - padding
        or second[2] < first[0] - padding
        or first[3] < second[1] - padding
        or second[3] < first[1] - padding
    )


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
    for index in range(1, points.shape[0] - 1):
        previous_point = points[index - 1]
        current_point = points[index]
        next_point = points[index + 1]
        rounded = _safe_rounded_corner_points(
            previous_point,
            current_point,
            next_point,
            max_radius,
            angle_threshold_degrees,
            tolerance,
            cut_fraction,
            safe_geometry,
        )
        if rounded is not None:
            result.extend(rounded)
            continue

        # A failed fillet is not a reason to stop extrusion.  The original
        # corner is already inside the safe centerline corridor, so retain the
        # sharp turn and keep the path continuous.
        result.append(current_point)

    result.append(points[-1])
    continuous = _dedupe_consecutive(np.asarray(result, dtype=np.float32), tolerance)
    if (
        continuous.shape[0] >= 2
        and LineString(points).is_simple
        and not LineString(continuous).is_simple
    ):
        # Individually valid fillets can still cross one another when two
        # non-adjacent turns are close.  Never trade a safe continuous toolpath
        # for a locally rounded but globally self-intersecting one.
        return [points]
    return [continuous] if continuous.shape[0] >= 2 else []


def _smooth_path_corners(
    path: np.ndarray,
    max_radius: float,
    angle_threshold_degrees: float,
    tolerance: float,
    safe_geometry=None,
    cut_fraction: float = 0.35,
) -> np.ndarray:
    points = _dedupe_consecutive(np.asarray(path[:, :2], dtype=np.float32), tolerance)
    original_points = points.copy()
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
        rounded = _safe_rounded_corner_points(
            previous_point,
            current_point,
            next_point,
            max_radius,
            angle_threshold_degrees,
            tolerance,
            cut_fraction,
            safe_geometry,
        )
        if rounded is None:
            result.append(current_point)
        else:
            result.extend(rounded)

    if not closed:
        result.append(points[-1])
    elif result:
        result.append(result[0])

    smoothed = _dedupe_consecutive(np.asarray(result, dtype=np.float32), tolerance)
    if (
        smoothed.shape[0] >= 2
        and LineString(original_points).is_simple
        and not LineString(smoothed).is_simple
    ):
        return original_points
    return smoothed


def _safe_rounded_corner_points(
    previous_point: np.ndarray,
    current_point: np.ndarray,
    next_point: np.ndarray,
    max_radius: float,
    angle_threshold_degrees: float,
    tolerance: float,
    cut_fraction: float,
    safe_geometry,
) -> list[np.ndarray] | None:
    """Fit the largest permitted fillet, shrinking only when a wall blocks it."""

    radius_factors = (1.0,) if safe_geometry is None else (1.0, 0.75, 0.5, 0.25, 0.1, 0.05)
    for radius_factor in radius_factors:
        rounded = _rounded_corner_points(
            previous_point,
            current_point,
            next_point,
            max_radius * radius_factor,
            angle_threshold_degrees,
            tolerance,
            cut_fraction,
        )
        if rounded is None:
            continue
        if safe_geometry is None or safe_geometry.covers(
            LineString([(float(point[0]), float(point[1])) for point in rounded])
        ):
            return rounded
    return None


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

    half_angle = math.radians(angle_degrees) * 0.5
    tangent_half = math.tan(half_angle)
    cosine_half = math.cos(half_angle)
    if tangent_half <= tolerance or abs(cosine_half) <= tolerance:
        return None

    # ``max_radius`` is a physical centreline radius, matching the UI label.
    # The former implementation used it directly as the tangent cut.  At a
    # 45-degree interior hairpin that produced an actual radius of only
    # ``max_radius * tan(22.5°)`` and left visibly pointed teeth even though
    # the trajectory contained many tiny arc samples.
    cut_fraction = min(max(float(cut_fraction), 0.05), 0.8)
    maximum_radius_cut = max_radius / tangent_half
    cut_distance = min(
        maximum_radius_cut,
        incoming_length * cut_fraction,
        outgoing_length * cut_fraction,
    )
    if cut_distance <= tolerance:
        return None

    start = current_point + incoming_unit * cut_distance
    end = current_point + outgoing_unit * cut_distance
    bisector = incoming_unit + outgoing_unit
    bisector_length = float(np.linalg.norm(bisector))
    if bisector_length <= tolerance:
        return None

    center = current_point + (bisector / bisector_length) * (cut_distance / cosine_half)
    start_angle = math.atan2(float(start[1] - center[1]), float(start[0] - center[0]))
    end_angle = math.atan2(float(end[1] - center[1]), float(end[0] - center[0]))
    delta = (end_angle - start_angle + math.pi) % (2.0 * math.pi) - math.pi
    # Keep the exported polyline's heading increments below 10 degrees.  The
    # old five-point minimum could turn a mathematically round but tiny fillet
    # back into a 30-degree sampled corner in the robot trajectory.
    steps = max(5, int(math.ceil(abs(delta) / (math.pi / 18.0))) + 1)

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

    if not config.print_perimeters:
        # Treat contour-free infill like the old inner-contour region instead
        # of forcing every hatch turn onto the exact STL boundary.  The first
        # half-bead keeps the emitted bead inside the part; one nominal pitch
        # provides a smooth, independent safety band at U-shaped and rounded
        # transitions without using or printing a contour ring.
        path_spacing = _resin_path_spacing(
            config.line_width,
            config.infill_overlap,
        )
        offset = config.line_width * 0.5 + path_spacing
        infill_surface = geometry.buffer(-offset, join_style="round")
        if not infill_surface.is_valid:
            infill_surface = infill_surface.buffer(0)
        return infill_surface

    perimeter_path_spacing = _resin_path_spacing(
        config.line_width,
        config.infill_overlap,
    )
    contour_infill_spacing = _resin_contour_infill_spacing(config)
    last_perimeter_centerline = config.line_width * 0.5 + (
        config.perimeter_count - 1
    ) * perimeter_path_spacing
    # Keep the already-good nominal perimeter placement unchanged.  The first
    # infill centreline is separated from the innermost perimeter by its own
    # configurable seam pitch derived from the independently measured,
    # pressure-flattened footprint. The infill-run overlap remains a separate
    # control; nominal process line_width remains unchanged.
    centerline_inset = last_perimeter_centerline + contour_infill_spacing
    return _libslic3r_offset_geometry(
        geometry,
        -centerline_inset,
        config.tolerance,
    )


def _libslic3r_fill_surface_overlap_offset(
    line_width: float,
    overlap_percent: float,
) -> float:
    """Move an inner bead edge to the safe infill-centerline boundary.

    The line center must remain half a physical bead inside the free surface,
    then may move back toward the perimeter by the requested overlap.  Using
    half of an already overlap-reduced pitch here counts overlap twice.
    """

    return _resin_overlap_width(line_width, overlap_percent) - 0.5 * line_width


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


def _last_perimeter_linework(
    geometry,
    line_width: float,
    path_spacing: float,
    perimeter_count: int,
    tolerance: float,
):
    """Return only the innermost configured perimeter centerline rings."""

    offset_distance = line_width * 0.5 + (perimeter_count - 1) * path_spacing
    offset_geometry = _libslic3r_offset_geometry(
        geometry,
        -offset_distance,
        tolerance,
    )
    rings: list[LineString] = []
    for polygon in _iter_polygons(offset_geometry):
        exterior = LineString(polygon.exterior.coords)
        if exterior.length > tolerance:
            rings.append(exterior)
        for interior in polygon.interiors:
            ring = LineString(interior.coords)
            if ring.length > tolerance:
                rings.append(ring)
    return unary_union(rings) if rings else GeometryCollection()


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


def _fast_concentric_infill_geometry(
    geometry,
    line_width: float,
    tolerance: float,
    *,
    residual_gap_tolerance: float = CONCENTRIC_RESIDUAL_GAP_TOLERANCE_MM,
    minimum_path_length: float = CONCENTRIC_MINIMUM_PATH_LENGTH_MM,
) -> list[np.ndarray]:
    """Fill with direct measured-width offsets and only essential centerlines.

    Concentric mode intentionally favors coverage and print speed over the
    strict non-local overlap contract used by Zigzag. Uniform rings are spaced
    by one measured flattened bead width, so their nominal overlap is zero.
    Residual components are measured after applying that physical bead width;
    a centered supplement is added only while an uncovered gap wider than the
    configured tolerance remains. Degenerate rings and supplements shorter
    than the configured path-length threshold are omitted entirely.
    """

    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if residual_gap_tolerance < 0:
        raise ValueError("residual_gap_tolerance must be non-negative")
    if minimum_path_length < 0:
        raise ValueError("minimum_path_length must be non-negative")
    if geometry.is_empty:
        return []

    max_offset = _max_geometry_offset(geometry, tolerance)
    offsets: list[float] = [0.0]
    offset = line_width
    while offset <= max_offset + tolerance:
        offsets.append(offset)
        offset += line_width

    paths: list[np.ndarray] = []
    for offset in offsets:
        paths.extend(_concentric_paths_at_offset(geometry, offset, tolerance))
    paths = [
        path
        for path in paths
        if path.shape[0] >= 2
        and _open_path_length(path) + tolerance >= minimum_path_length
    ]

    # Topology changes can leave several local cores even though the source is
    # one polygon. Repeatedly cover the widest point of every visible residual
    # with the shortest useful centerline stroke. Stop when the physical gap
    # criterion is met or every remaining correction would be shorter than the
    # minimum printable path threshold.
    for _ in range(64):
        coverage = _round_bead_coverage(paths, line_width * 0.5)
        residual = geometry.difference(coverage)
        residual_regions = [
            polygon
            for polygon in _iter_polygons(residual)
            if _maximum_inscribed_diameter(polygon, tolerance)
            > residual_gap_tolerance + tolerance
        ]
        if not residual_regions:
            break

        added: list[np.ndarray] = []
        existing_linework = unary_union(
            [LineString(path[:, :2]) for path in paths if path.shape[0] >= 2]
        )
        for polygon in residual_regions:
            candidate = _concentric_residual_supplement(
                polygon,
                line_width,
                tolerance,
                minimum_path_length=minimum_path_length,
            )
            if candidate is None:
                continue
            candidate_line = LineString(candidate[:, :2])
            if (
                not existing_linework.is_empty
                and candidate_line.difference(
                    existing_linework.buffer(tolerance * 2.0)
                ).length
                <= tolerance
            ):
                continue
            added.append(candidate)
            existing_linework = unary_union((existing_linework, candidate_line))
        if not added:
            break
        paths.extend(added)

    return [
        path
        for path in paths
        if _open_path_length(path) + tolerance >= minimum_path_length
    ]


def _concentric_residual_supplement(
    polygon,
    line_width: float,
    tolerance: float,
    *,
    minimum_path_length: float,
) -> np.ndarray | None:
    """Return a minimum-length stroke through a residual's widest point."""

    circle = maximum_inscribed_circle(polygon, tolerance=tolerance)
    if circle.is_empty or len(circle.coords) < 2:
        return None
    center = np.asarray(circle.coords[0], dtype=np.float64)
    nearest_boundary = np.asarray(circle.coords[1], dtype=np.float64)
    radius_vector = nearest_boundary - center
    radius = float(np.linalg.norm(radius_vector))
    if radius <= tolerance:
        return None

    # The nearest-boundary radius points across the narrow direction. Its
    # perpendicular follows the residual's locally long direction and avoids
    # wasting path length across already-covered neighboring rings.
    direction = np.asarray(
        [-radius_vector[1], radius_vector[0]],
        dtype=np.float64,
    ) / radius
    min_x, min_y, max_x, max_y = polygon.bounds
    probe_half_length = math.hypot(max_x - min_x, max_y - min_y) + line_width
    probe = LineString(
        [
            center - direction * probe_half_length,
            center + direction * probe_half_length,
        ]
    )
    segments = _extract_line_segments(polygon.intersection(probe), tolerance)
    if not segments:
        return None
    center_point = Point(float(center[0]), float(center[1]))
    segment = min(
        segments,
        key=lambda item: (float(item.distance(center_point)), -float(item.length)),
    )
    segment_length = float(segment.length)
    if segment_length <= tolerance:
        return None

    # Round end caps cover half a bead beyond each endpoint. Remove that
    # already-covered length from the stroke. Point-like cores whose required
    # move is below the explicit fragment threshold are omitted rather than
    # exported as visible dots; elongated residuals retain their longer chord.
    minimum_stroke = max(tolerance * 4.0, line_width * 0.001)
    stroke_length = min(
        segment_length,
        max(minimum_stroke, segment_length - line_width),
    )
    if stroke_length + tolerance < minimum_path_length:
        return None
    center_distance = float(segment.project(center_point))
    start_distance = min(
        max(0.0, center_distance - stroke_length * 0.5),
        max(0.0, segment_length - stroke_length),
    )
    end_distance = start_distance + stroke_length
    start = np.asarray(segment.interpolate(start_distance).coords[0], dtype=np.float32)
    end = np.asarray(segment.interpolate(end_distance).coords[0], dtype=np.float32)
    path = np.asarray([start, end], dtype=np.float32)
    return path if _open_path_length(path) > tolerance else None


def _concentric_infill_geometry(
    geometry,
    line_width: float,
    path_spacing: float,
    tolerance: float,
    *,
    minimum_spacing: float | None = None,
) -> list[np.ndarray]:
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if path_spacing <= 0:
        raise ValueError("path_spacing must be positive")
    if geometry.is_empty:
        return []

    max_offset = _max_geometry_offset(geometry, tolerance)
    offsets = _uniform_concentric_offsets(
        max_offset,
        line_width,
        path_spacing,
        minimum_spacing=minimum_spacing,
    )
    paths: list[np.ndarray] = []
    for offset in offsets:
        paths.extend(_concentric_paths_at_offset(geometry, offset, tolerance))
    paths = _filter_concentric_paths_by_spacing(
        paths,
        path_spacing,
        tolerance,
        minimum_spacing=minimum_spacing,
        bead_width=line_width,
    )
    return paths


def _uniform_concentric_offsets(
    max_offset: float,
    line_width: float,
    path_spacing: float,
    *,
    minimum_spacing: float | None = None,
) -> list[float]:
    # ``geometry`` is a centerline-safe corridor, so its boundary is the first
    # valid concentric centerline.  Older code applied another half-width inset
    # here, giving concentric a different wall-overlap meaning from all other
    # patterns.
    offsets: list[float] = [0.0]
    offset = path_spacing
    while offset <= max_offset + 1e-9:
        offsets.append(offset)
        offset += path_spacing

    residual = max_offset - offsets[-1]
    if minimum_spacing is not None:
        # The maximum offset is the geometric collapse point.  Even when its
        # nominal distance from the preceding ring is large enough, the tiny
        # residual ring can fold back on itself and print two nearly coincident
        # long sides.  Strict measured-width mode therefore keeps only the
        # uniform sequence and conservatively leaves the centre residual empty.
        append_residual = False
    else:
        append_residual = residual > line_width * 0.2
    if append_residual:
        offsets.append(max_offset)
    return offsets


def _filter_concentric_paths_by_spacing(
    paths: list[np.ndarray],
    path_spacing: float,
    tolerance: float,
    *,
    minimum_spacing: float | None = None,
    bead_width: float | None = None,
) -> list[np.ndarray]:
    accepted: list[np.ndarray] = []
    accepted_lines: list[LineString] = []
    # GEOS approximates round joins with chords, so two exact offset rings may
    # measure a fraction of a percent closer than their requested pitch.  Keep
    # a tight numerical allowance without accepting a genuinely under-spaced
    # ring (for example 1.7 mm at a requested 1.8 mm pitch).
    if minimum_spacing is None:
        spacing_tolerance = max(tolerance * 100.0, path_spacing * 0.001)
        required_spacing = path_spacing - spacing_tolerance
        comparison_epsilon = tolerance
    else:
        required_spacing = minimum_spacing
        # The strict threshold is a physical contract, not a geometry cleanup
        # tolerance.  Admit only floating-point noise measured in sub-microns.
        comparison_epsilon = max(1e-7, required_spacing * 1e-7)

    for path in paths:
        if path.shape[0] < 2:
            continue
        line = LineString([(float(point[0]), float(point[1])) for point in path])
        if line.is_empty:
            continue
        if minimum_spacing is not None and not _solid_fill_spacing_postcondition(
            [path],
            GeometryCollection(),
            required_spacing,
            tolerance,
            bead_width=bead_width,
        ):
            # A single offset ring can fold through a narrow neck and create
            # two long, nearly coincident sides even though it is topologically
            # simple.  Dropping that whole ring is the conservative alternative
            # to locally doubling the measured bead dose.
            continue
        if any(
            line.distance(existing) < required_spacing - comparison_epsilon
            for existing in accepted_lines
        ):
            continue
        accepted.append(path)
        accepted_lines.append(line)
    return accepted


def _concentric_paths_at_offset(
    geometry,
    offset_distance: float,
    tolerance: float,
) -> list[np.ndarray]:
    offset_geometry = (
        geometry
        if offset_distance <= tolerance
        else geometry.buffer(
            -offset_distance,
            join_style="round",
            quad_segs=32,
        )
    )
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
    if not paths:
        # A sparse pitch may place the global scan grid entirely outside a
        # small island.  Keep one centered stroke so non-zero density never
        # silently turns into an empty layer.
        scan_y = (min_y + max_y) * 0.5
        line = LineString([(min_x - padding, scan_y), (max_x + padding, scan_y)])
        for segment in _extract_line_segments(rotated.intersection(line), tolerance):
            restored = affinity.rotate(
                segment,
                angle_degrees,
                origin=(0, 0),
                use_radians=False,
            )
            path = _dedupe_consecutive(
                np.asarray(
                    [[float(x), float(y)] for x, y in restored.coords],
                    dtype=np.float32,
                ),
                tolerance,
            )
            if path.shape[0] >= 2 and np.linalg.norm(path[-1] - path[0]) > tolerance:
                paths.append(path)
    return paths


def _solid_zigzag_infill_paths(
    geometry,
    spacing: float,
    line_width: float,
    angle_degrees: float,
    minimum_clearance: float,
    tolerance: float,
    *,
    smoothing_angle_degrees: float = DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES,
    smoothing_corner_cut: float | None = None,
    enforce_maximum_overlap: bool = False,
    maximum_overlap_spacing: float | None = None,
    connect_adjacent: bool = True,
    follow_boundaries: bool = True,
) -> list[np.ndarray]:
    """Plan full-density hatch lines with bounded bead-spacing correction.

    Sparse infill intentionally shares a global phase so successive layers are
    stable.  At full density that phase can leave almost an entire pitch beside
    a wall.  Each printable island therefore uses an unchanged, centered pitch
    by default.  When long boundaries are parallel to the hatch, their levels
    are safe phase anchors only if every intervening band can be divided close
    to the requested overlap pitch.  A small expansion-only adjustment may
    reduce overlap beside an anchored boundary, but centre-lines are never
    compressed below the requested pitch.  A corridor narrower than one pitch
    still receives exactly one centered stroke.
    """

    if spacing <= 0:
        raise ValueError("infill spacing must be positive")
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if geometry.is_empty:
        return []

    spacing_adjustment = _solid_spacing_adjustment_limit(spacing, line_width)
    # Coverage phasing may leave a slightly wider band, but it must never
    # compress parallel centre-lines below the requested overlap pitch.
    minimum_scan_spacing = max(
        tolerance,
        spacing if enforce_maximum_overlap else spacing - spacing_adjustment,
    )
    maximum_scan_spacing = min(line_width, spacing + spacing_adjustment)
    def paths_at_phase(phase_fraction: float) -> list[np.ndarray]:
        phase_paths: list[np.ndarray] = []
        for polygon in _iter_polygons(geometry):
            rotated = affinity.rotate(
                polygon,
                -angle_degrees,
                origin=(0, 0),
                use_radians=False,
            )
            min_x, min_y, max_x, max_y = rotated.bounds
            scan_levels = _solid_zigzag_scan_levels(
                rotated,
                spacing,
                line_width,
                tolerance,
                enforce_maximum_overlap=enforce_maximum_overlap,
            )
            if abs(phase_fraction) > tolerance:
                phase_offset = spacing * phase_fraction
                scan_levels = [
                    level + phase_offset
                    for level in scan_levels
                    if min_y - tolerance
                    <= level + phase_offset
                    <= max_y + tolerance
                ]
            padding = spacing * 2.0

            for scan_y in scan_levels:
                line = LineString(
                    [(min_x - padding, scan_y), (max_x + padding, scan_y)]
                )
                intersection = rotated.intersection(line)
                segments = list(_extract_line_segments(intersection, tolerance))
                if len(segments) > 1:
                    # GEOS may split a scanline into touching collinear pieces
                    # when it lands exactly on a hole vertex. Joining only
                    # pieces that already touch removes artificial stops
                    # without bridging a real void.
                    intersection = linemerge(unary_union(segments))
                    segments = list(_extract_line_segments(intersection, tolerance))
                for segment in segments:
                    restored = affinity.rotate(
                        segment,
                        angle_degrees,
                        origin=(0, 0),
                        use_radians=False,
                    )
                    path = _dedupe_consecutive(
                        np.asarray(
                            [[float(x), float(y)] for x, y in restored.coords],
                            dtype=np.float32,
                        ),
                        tolerance,
                    )
                    if (
                        path.shape[0] >= 2
                        and np.linalg.norm(path[-1] - path[0]) > tolerance
                    ):
                        phase_paths.append(path)
        if enforce_maximum_overlap:
            minimum_fragment_length = spacing * 0.25
            phase_paths = [
                path
                for path in phase_paths
                if _open_path_length(path) >= minimum_fragment_length
            ]
        return phase_paths

    phase_paths: dict[float, list[np.ndarray]] = {0.0: paths_at_phase(0.0)}

    def cached_paths_at_phase(phase_fraction: float) -> list[np.ndarray]:
        if phase_fraction not in phase_paths:
            phase_paths[phase_fraction] = paths_at_phase(phase_fraction)
        return phase_paths[phase_fraction]

    baseline_paths = phase_paths[0.0]
    if not connect_adjacent:
        return baseline_paths

    def connect_phase(paths: list[np.ndarray]) -> list[np.ndarray]:
        return _connect_zigzag_infill_paths(
            paths,
            geometry,
            minimum_scan_spacing,
            min(minimum_clearance, minimum_scan_spacing),
            tolerance,
            maximum_spacing=maximum_scan_spacing,
            solid_bead_width=line_width,
            wall_seam_clearance=(
                line_width if enforce_maximum_overlap else minimum_clearance
            ),
            solid_smoothing_angle_degrees=smoothing_angle_degrees,
            solid_smoothing_corner_cut=(
                min(
                    line_width * DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR,
                    line_width * 0.15,
                )
                if smoothing_corner_cut is None
                else smoothing_corner_cut
            ),
            maximum_connector_overlap_spacing=(
                maximum_overlap_spacing
                if enforce_maximum_overlap
                else None
            ),
            follow_boundaries=follow_boundaries,
        )

    baseline_connected = connect_phase(baseline_paths)
    if not enforce_maximum_overlap:
        return baseline_connected

    def coverage_metrics(paths: list[np.ndarray]) -> tuple[float, float]:
        coverage = _round_bead_coverage(paths, line_width * 0.5)
        uncovered = geometry.difference(coverage)
        return (
            float(uncovered.area),
            (
                0.0
                if uncovered.is_empty
                else _maximum_inscribed_diameter(uncovered, tolerance)
            ),
        )

    baseline_area, baseline_diameter = coverage_metrics(baseline_connected)
    area_allowance = max(
        tolerance * max(float(geometry.boundary.length), 1.0),
        line_width * line_width * 1e-4,
    )
    diameter_allowance = max(tolerance * 20.0, line_width * 0.005)
    short_threshold = max(spacing * 2.0, line_width * 2.0)

    def continuity_score(paths: list[np.ndarray]) -> tuple[int, int, float]:
        lengths = [_open_path_length(path) for path in paths]
        return (
            len(paths),
            sum(length < short_threshold for length in lengths),
            sum(max(0.0, short_threshold - length) for length in lengths),
        )

    def coverage_is_allowed(paths: list[np.ndarray]) -> bool:
        uncovered_area, uncovered_diameter = coverage_metrics(paths)
        return bool(
            uncovered_area <= baseline_area + area_allowance
            and uncovered_diameter <= baseline_diameter + diameter_allowance
        )

    # A centred phase can graze a curved/diagonal boundary and produce a
    # sub-bead sliver that has no usable continuation. A half-pitch phase keeps
    # the same inter-line spacing and dose, but often moves those intersections
    # onto printable spans. When multiple trails remain, sample four coarse
    # phases; only if the best coarse result still contains a short fragment do
    # we sample their midpoints. Every candidate keeps the requested pitch and
    # must match the centred baseline's physical bead coverage before its
    # continuity score is considered. This is a topology optimization, not a
    # spacing or boundary relaxation.
    eligible = [baseline_connected]
    half_pitch_connected = connect_phase(cached_paths_at_phase(0.5))
    if coverage_is_allowed(half_pitch_connected):
        eligible.append(half_pitch_connected)

    best = min(eligible, key=continuity_score)
    if len(best) > 1:
        for phase_fraction in (0.1, 0.2, 0.3, 0.4):
            candidate = connect_phase(cached_paths_at_phase(phase_fraction))
            if coverage_is_allowed(candidate):
                eligible.append(candidate)

    best = min(eligible, key=continuity_score)
    if any(_open_path_length(path) < short_threshold for path in best):
        for phase_fraction in (0.15, 0.25, 0.35, 0.45):
            candidate = connect_phase(cached_paths_at_phase(phase_fraction))
            if coverage_is_allowed(candidate):
                eligible.append(candidate)

    return min(eligible, key=continuity_score)


def _solid_spacing_adjustment_limit(spacing: float, line_width: float) -> float:
    """Return the maximum symmetric solid-pitch correction.

    The value is now an expansion-only phase allowance.  At 2.2 mm measured
    footprint and 10% overlap the requested pitch is 1.98 mm; anchored bands
    may relax toward 2.09 mm, but may not compress below 1.98 mm.  This keeps
    local material contact at or below the UI overlap instead of silently
    creating a denser strip.
    """

    available_overlap = max(0.0, line_width - spacing)
    return max(
        0.0,
        min(
            available_overlap * 0.5,
            line_width * 0.05,
            spacing * 0.25,
        ),
    )


def _solid_zigzag_scan_levels(
    rotated_polygon: Polygon,
    spacing: float,
    line_width: float,
    tolerance: float,
    *,
    enforce_maximum_overlap: bool = False,
) -> list[float]:
    min_y = float(rotated_polygon.bounds[1])
    max_y = float(rotated_polygon.bounds[3])
    centered = _centered_scan_levels(min_y, max_y, spacing, tolerance)
    if max_y - min_y < spacing - tolerance:
        return centered

    anchors = _parallel_boundary_scan_anchors(
        rotated_polygon,
        line_width,
        tolerance,
    )
    anchor_tolerance = max(tolerance * 20.0, line_width * 1e-4)
    if not anchors:
        return centered

    anchors_minimum = abs(anchors[0] - min_y) <= anchor_tolerance
    anchors_maximum = abs(anchors[-1] - max_y) <= anchor_tolerance
    spacing_adjustment = _solid_spacing_adjustment_limit(spacing, line_width)
    minimum_scan_spacing = max(
        tolerance,
        spacing if enforce_maximum_overlap else spacing - spacing_adjustment,
    )
    maximum_scan_spacing = min(line_width, spacing + spacing_adjustment)
    if anchors_minimum and anchors_maximum:
        anchored = _scan_levels_between_anchors(
            anchors,
            spacing,
            minimum_scan_spacing,
            maximum_scan_spacing,
            tolerance,
        )
        if anchored is not None:
            return anchored
        return centered
    if anchors_minimum:
        return _one_sided_anchored_scan_levels(
            anchors,
            min_y,
            max_y,
            spacing,
            minimum_scan_spacing,
            maximum_scan_spacing,
            tolerance,
            anchor_from_minimum=True,
        )
    if anchors_maximum:
        return _one_sided_anchored_scan_levels(
            anchors,
            min_y,
            max_y,
            spacing,
            minimum_scan_spacing,
            maximum_scan_spacing,
            tolerance,
            anchor_from_minimum=False,
        )
    return centered


def _scan_levels_between_anchors(
    anchors: list[float],
    spacing: float,
    minimum_scan_spacing: float,
    maximum_scan_spacing: float,
    tolerance: float,
) -> list[float] | None:
    levels: list[float] = [anchors[0]]
    for lower, upper in zip(anchors[:-1], anchors[1:]):
        interval = _scan_levels_for_anchored_interval(
            lower,
            upper,
            spacing,
            minimum_scan_spacing,
            maximum_scan_spacing,
            tolerance,
        )
        if interval is None:
            return None
        levels.extend(interval)
    return levels


def _scan_levels_for_anchored_interval(
    lower: float,
    upper: float,
    spacing: float,
    minimum_scan_spacing: float,
    maximum_scan_spacing: float,
    tolerance: float,
) -> list[float] | None:
    span = upper - lower
    interval_count = max(1, math.floor(span / spacing + 0.5))
    local_spacing = span / interval_count
    if (
        local_spacing < minimum_scan_spacing - tolerance
        or local_spacing > maximum_scan_spacing + tolerance
    ):
        return None
    levels = [
        lower + local_spacing * index
        for index in range(1, interval_count + 1)
    ]
    # Repeated floating-point addition can put the mathematically identical
    # final level a few ulps outside the polygon.  GEOS then returns an empty
    # boundary intersection and silently drops a complete solid-fill row.
    levels[-1] = upper
    return levels


def _one_sided_anchored_scan_levels(
    anchors: list[float],
    min_y: float,
    max_y: float,
    spacing: float,
    minimum_scan_spacing: float,
    maximum_scan_spacing: float,
    tolerance: float,
    *,
    anchor_from_minimum: bool,
) -> list[float]:
    ordered = anchors if anchor_from_minimum else [-value for value in reversed(anchors)]
    transformed_max = max_y if anchor_from_minimum else -min_y
    levels: list[float] = [ordered[0]]
    current = ordered[0]

    for anchor in ordered[1:]:
        interval = _scan_levels_for_anchored_interval(
            current,
            anchor,
            spacing,
            minimum_scan_spacing,
            maximum_scan_spacing,
            tolerance,
        )
        if interval is None:
            continue
        levels.extend(interval)
        current = anchor

    next_level = current + spacing
    while next_level < transformed_max - tolerance:
        levels.append(next_level)
        next_level += spacing
    if abs(next_level - transformed_max) <= max(tolerance * 20.0, spacing * 1e-6):
        levels.append(transformed_max)

    if anchor_from_minimum:
        return levels
    return sorted(-value for value in levels)


def _centered_scan_levels(
    min_y: float,
    max_y: float,
    spacing: float,
    tolerance: float,
) -> list[float]:
    span = max_y - min_y
    interval_count = max(0, math.floor((span + tolerance) / spacing))
    residual = max(0.0, span - interval_count * spacing)
    first_scan_y = min_y + residual * 0.5
    levels = [first_scan_y + index * spacing for index in range(interval_count + 1)]
    snap_tolerance = max(tolerance * 20.0, spacing * 1e-6)
    if levels and abs(levels[0] - min_y) <= snap_tolerance:
        levels[0] = min_y
    if levels and abs(levels[-1] - max_y) <= snap_tolerance:
        levels[-1] = max_y
    return levels


def _parallel_boundary_scan_anchors(
    polygon: Polygon,
    line_width: float,
    tolerance: float,
) -> list[float]:
    anchors: list[float] = []
    maximum_vertical_delta = max(tolerance * 20.0, line_width * 1e-5)
    for ring in (polygon.exterior, *polygon.interiors):
        coordinates = list(ring.coords)
        for start, end in zip(coordinates[:-1], coordinates[1:]):
            dx = float(end[0] - start[0])
            dy = float(end[1] - start[1])
            if math.hypot(dx, dy) < line_width - tolerance:
                continue
            if abs(dy) <= maximum_vertical_delta:
                anchors.append((float(start[1]) + float(end[1])) * 0.5)

    if not anchors:
        return []
    anchors.sort()
    cluster_tolerance = max(tolerance * 20.0, line_width * 1e-4)
    clustered: list[float] = []
    cluster: list[float] = []
    for anchor in anchors:
        if cluster and anchor - cluster[-1] > cluster_tolerance:
            clustered.append(sum(cluster) / len(cluster))
            cluster = []
        cluster.append(anchor)
    if cluster:
        clustered.append(sum(cluster) / len(cluster))
    return clustered


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
    return _edge_disjoint_graph_trails(edges, coordinates, tolerance)


def _multi_axis_lattice_infill_geometry(
    geometry,
    spacing: float,
    angles: tuple[float, ...],
    tolerance: float,
) -> list[np.ndarray]:
    """Node a multi-axis lattice and cover every edge with minimum trails."""

    candidate_paths: list[np.ndarray] = []
    for angle in angles:
        candidate_paths.extend(
            _zigzag_infill_geometry(geometry, spacing, angle, tolerance)
        )
    if not candidate_paths:
        return []

    noded = unary_union(
        [
            LineString([(float(point[0]), float(point[1])) for point in path])
            for path in candidate_paths
            if path.shape[0] >= 2
        ]
    )
    edges, coordinates = _unique_line_graph_edges(noded, tolerance)
    return _edge_disjoint_graph_trails(edges, coordinates, tolerance)


def _merge_collinear_triangular_edges(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    tolerance: float,
) -> list[np.ndarray]:
    """Merge noded lattice edges without joining different directions.

    The lattice is noded at every crossing so overlaps can be removed. Those
    graph edges are not separate printing paths, however, so merge the pieces
    back along each of the three original lattice directions.
    """
    direction_lines: dict[int, list[LineString]] = {0: [], 60: [], 120: []}
    for start, end in edges:
        start_point = coordinates[start]
        end_point = coordinates[end]
        angle = math.degrees(
            math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        ) % 180.0
        direction = min(
            direction_lines,
            key=lambda expected: min(
                abs(angle - expected),
                abs(angle - expected - 180.0),
                abs(angle - expected + 180.0),
            ),
        )
        direction_lines[direction].append(
            LineString([start_point, end_point])
        )

    merged_paths: list[np.ndarray] = []
    for direction in (0, 60, 120):
        if not direction_lines[direction]:
            continue
        merged = linemerge(unary_union(direction_lines[direction]))
        for segment in _extract_line_segments(merged, tolerance):
            coords = list(segment.coords)
            path = _dedupe_consecutive(
                np.asarray([[float(x), float(y)] for x, y in coords], dtype=np.float32),
                tolerance,
            )
            if path.shape[0] >= 2 and np.linalg.norm(path[-1] - path[0]) > tolerance:
                merged_paths.append(path)
    return merged_paths


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


def _edge_disjoint_graph_trails(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    tolerance: float,
) -> list[np.ndarray]:
    """Return a minimum-count edge-disjoint trail cover for each component.

    Odd vertices are paired with virtual edges, an Euler circuit is computed,
    and the circuit is cut at those virtual edges.  Virtual edges are never
    printed, so every real lattice edge is emitted exactly once without the
    retracing that would otherwise create material piles.
    """

    if not edges:
        return []

    vertex_edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for edge_index, (start, end) in enumerate(edges):
        vertex_edges[start].append(edge_index)
        vertex_edges[end].append(edge_index)

    remaining_edges = set(range(len(edges)))
    trails: list[np.ndarray] = []
    while remaining_edges:
        seed_edge = min(remaining_edges)
        component_edges: set[int] = set()
        pending_vertices = [edges[seed_edge][0]]
        visited_vertices: set[tuple[int, int]] = set()
        while pending_vertices:
            vertex = pending_vertices.pop()
            if vertex in visited_vertices:
                continue
            visited_vertices.add(vertex)
            for edge_index in vertex_edges[vertex]:
                component_edges.add(edge_index)
                start, end = edges[edge_index]
                pending_vertices.append(end if start == vertex else start)
        remaining_edges.difference_update(component_edges)
        trails.extend(
            _edge_component_trails(
                [edges[index] for index in sorted(component_edges)],
                coordinates,
                tolerance,
            )
        )
    return trails


def _edge_component_trails(
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    coordinates: dict[tuple[int, int], tuple[float, float]],
    tolerance: float,
) -> list[np.ndarray]:
    degree: dict[tuple[int, int], int] = defaultdict(int)
    for start, end in edges:
        degree[start] += 1
        degree[end] += 1

    unpaired = {vertex for vertex, value in degree.items() if value % 2 == 1}
    virtual_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    while unpaired:
        first = min(unpaired)
        unpaired.remove(first)
        second = min(
            unpaired,
            key=lambda vertex: (
                math.dist(coordinates[first], coordinates[vertex]),
                vertex,
            ),
        )
        unpaired.remove(second)
        virtual_edges.append((first, second))

    augmented = [(start, end, False) for start, end in edges]
    augmented.extend((start, end, True) for start, end in virtual_edges)
    adjacency: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    for edge_index, (start, end, _) in enumerate(augmented):
        adjacency[start].append((edge_index, end))
        adjacency[end].append((edge_index, start))
    for entries in adjacency.values():
        entries.sort(reverse=True)

    start_vertex = min(adjacency)
    vertex_stack = [start_vertex]
    edge_stack: list[int] = []
    used_edges: set[int] = set()
    circuit_vertices: list[tuple[int, int]] = []
    circuit_edges: list[int] = []
    while vertex_stack:
        vertex = vertex_stack[-1]
        while adjacency[vertex] and adjacency[vertex][-1][0] in used_edges:
            adjacency[vertex].pop()
        if adjacency[vertex]:
            edge_index, neighbor = adjacency[vertex].pop()
            if edge_index in used_edges:
                continue
            used_edges.add(edge_index)
            vertex_stack.append(neighbor)
            edge_stack.append(edge_index)
            continue
        circuit_vertices.append(vertex_stack.pop())
        if edge_stack:
            circuit_edges.append(edge_stack.pop())

    ordered_vertices = list(reversed(circuit_vertices))
    ordered_edges = list(reversed(circuit_edges))
    edge_sequence = [
        (edge_index, ordered_vertices[index], ordered_vertices[index + 1])
        for index, edge_index in enumerate(ordered_edges)
    ]
    virtual_ids = {
        index for index, (_, _, is_virtual) in enumerate(augmented) if is_virtual
    }
    if not virtual_ids:
        points = np.asarray(
            [coordinates[vertex] for vertex in ordered_vertices],
            dtype=np.float32,
        )
        points = _dedupe_consecutive(points, tolerance)
        return [points] if points.shape[0] >= 2 else []

    first_virtual = next(
        index for index, (edge_index, _, _) in enumerate(edge_sequence) if edge_index in virtual_ids
    )
    edge_sequence = edge_sequence[first_virtual + 1 :] + edge_sequence[: first_virtual + 1]
    component_trails: list[np.ndarray] = []
    current_vertices: list[tuple[int, int]] = [edge_sequence[0][1]]
    for edge_index, _, end in edge_sequence:
        if edge_index in virtual_ids:
            if len(current_vertices) >= 2:
                path = _dedupe_consecutive(
                    np.asarray([coordinates[vertex] for vertex in current_vertices], dtype=np.float32),
                    tolerance,
                )
                if path.shape[0] >= 2:
                    component_trails.append(path)
            current_vertices = [end]
            continue
        current_vertices.append(end)
    if len(current_vertices) >= 2:
        path = _dedupe_consecutive(
            np.asarray([coordinates[vertex] for vertex in current_vertices], dtype=np.float32),
            tolerance,
        )
        if path.shape[0] >= 2:
            component_trails.append(path)
    return component_trails


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
    # The path-only format has no per-segment flow control.  Split the total
    # material-length budget evenly between directions; treating each axis as
    # an independent coverage probability prints roughly 2x/3x material for
    # full-density grid/triangle patterns.
    return path_spacing * axis_count / density


def _infill_geometry_inset(config: SliceConfig) -> float:
    perimeter_path_spacing = _resin_path_spacing(
        config.line_width,
        config.infill_overlap,
    )
    last_perimeter_centerline = config.line_width * 0.5 + (
        config.perimeter_count - 1
    ) * perimeter_path_spacing
    return last_perimeter_centerline + _resin_planning_path_spacing(config)


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
