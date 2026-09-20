"""A small local web server for interactively inspecting surface equations."""

from __future__ import annotations

import json
import math
import os
import secrets
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..conformal_lattice.contracts import (
    CONFORMAL_LATTICE_SPEC_V1,
    double_sine_source_sha256,
    generated_surface_source_sha256,
    load_conformal_lattice_spec,
)
from .model import DoubleSineSurface
from .stl_domain import STLProjectionDomain, stl_projection_domain_from_bytes
from ..surface_mapper.progression import LayerProgression


DEFAULT_PREVIEW_WIDTH_MM = 120.0
DEFAULT_PREVIEW_HEIGHT_MM = 100.0
DEFAULT_PREVIEW_SAMPLES = 49
MAX_PREVIEW_SAMPLES = 120
MAX_CONFORMAL_SAMPLES = 512
MAX_STL_BYTES = 64 * 1024 * 1024
MAX_DESIGNER_STATE_BYTES = 32 * 1024
CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM = 0.5
SURFACE_PREVIEW_API_VERSION = "surface_preview_v2"


def _designer_state_path() -> Path:
    """Return the per-user, persistent design-state location."""

    root = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    return root / "KukaSlicer" / "conformal_designer_state_v1.json"


def _load_designer_state(state_path: Path) -> dict[str, str | bool]:
    """Load one small, user-editable designer state without trusting its shape."""

    try:
        if not state_path.is_file() or state_path.stat().st_size > MAX_DESIGNER_STATE_BYTES:
            return {}
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict) or len(raw) > 64:
        return {}
    state: dict[str, str | bool] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.replace("_", "").isalnum() or len(key) > 80:
            return {}
        if isinstance(value, (str, bool)):
            state[key] = value
        else:
            return {}
    return state


def _save_designer_state(state_path: Path, state: dict[str, str | bool]) -> None:
    """Persist a validated small state atomically for the next server launch."""

    if len(state) > 64:
        raise ValueError("designer state has too many fields")
    for key, value in state.items():
        if not key.replace("_", "").isalnum() or len(key) > 80 or not isinstance(value, (str, bool)):
            raise ValueError("designer state contains an invalid field")
    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_DESIGNER_STATE_BYTES:
        raise ValueError("designer state is too large")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_suffix(".tmp")
    temporary_path.write_text(encoded, encoding="utf-8")
    temporary_path.replace(state_path)


def _git_revision() -> str:
    """Return a local revision label without making the preview depend on Git."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=1,
        ).strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


SURFACE_PREVIEW_GIT_REVISION = _git_revision()


def _query_float(
    params: dict[str, list[str]],
    name: str,
    default: float,
    *,
    positive: bool = False,
) -> float:
    raw = params.get(name, [str(default)])[0]
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _query_bool(params: dict[str, list[str]], name: str, default: bool) -> bool:
    raw = params.get(name, ["true" if default else "false"])[0]
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "on", "yes"}:
        return True
    if normalized in {"0", "false", "off", "no"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _query_phase_radians(
    params: dict[str, list[str]],
    *,
    pi_multiple_name: str,
    legacy_radians_name: str,
) -> float:
    """Read a designer-facing π multiple while preserving the rad contract.

    The current designer submits the short, human-friendly multiple of π.
    Existing query clients and all exported JSON remain in radians, so a
    legacy ``*_rad`` value is still accepted when the new UI value is absent.
    """

    if pi_multiple_name in params:
        return math.pi * _query_float(params, pi_multiple_name, 0.0)
    return _query_float(params, legacy_radians_name, 0.0)


def _query_samples(params: dict[str, list[str]]) -> int:
    raw = params.get("samples", [str(DEFAULT_PREVIEW_SAMPLES)])[0]
    try:
        samples = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples must be an integer") from exc
    if not 8 <= samples <= MAX_PREVIEW_SAMPLES:
        raise ValueError(f"samples must be in the range [8, {MAX_PREVIEW_SAMPLES}]")
    return samples


def _normalized_positive_phase(phase_rad: float) -> float:
    """Express an analytically equivalent phase in [0, 2π)."""

    return phase_rad % (2.0 * math.pi)


def _query_positive_half_integer(params: dict[str, list[str]], name: str, default: float) -> float:
    """Read a positive half-integer wave count for the tensile phase rule."""

    value = _query_float(params, name, default, positive=True)
    if not math.isclose(value - 0.5, round(value - 0.5), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must be a positive half-integer (0.5, 1.5, 2.5, ...)")
    return value


def _surface_from_query(
    params: dict[str, list[str]], *, x_extent_mm: float, y_extent_mm: float
) -> tuple[DoubleSineSurface, dict[str, object]]:
    """Resolve either manual parameters or the tensile, size-normalised rule.

    The tensile rule uses wave counts rather than fixed wavelengths.  It keeps
    the double-sine geometry comparable when a specimen's gauge dimensions
    change, while placing a positive stationary peak at the gauge centre and
    a zero-height contour at each gauge boundary.  It is a geometry rule, not
    an inspection-point or bending-load alignment rule.
    """

    mode = params.get("surface_parameter_mode", ["manual_wavelength_phase"])[0]
    amplitude_mm = _query_float(params, "amplitude_mm", 0.8)
    z_reference_mm = _query_float(params, "z_reference_mm", 0.0)
    if mode == "tensile_centered_wave_count":
        wave_count_x = _query_positive_half_integer(params, "wave_count_x", 1.5)
        wave_count_y = _query_positive_half_integer(params, "wave_count_y", 1.5)
        wavelength_x_mm = x_extent_mm / wave_count_x
        wavelength_y_mm = y_extent_mm / wave_count_y
        phase_x_rad = _normalized_positive_phase(math.pi / 2.0 - math.pi * wave_count_x)
        phase_y_rad = _normalized_positive_phase(math.pi / 2.0 - math.pi * wave_count_y)
        return (
            DoubleSineSurface(
                amplitude_mm=amplitude_mm,
                wavelength_x_mm=wavelength_x_mm,
                wavelength_y_mm=wavelength_y_mm,
                phase_x_rad=phase_x_rad,
                phase_y_rad=phase_y_rad,
                z_reference_mm=z_reference_mm,
            ),
            {
                "mode": mode,
                "wave_count_x": wave_count_x,
                "wave_count_y": wave_count_y,
                "phase_policy": "gauge_center_positive_peak_and_boundary_zero",
            },
        )
    if mode != "manual_wavelength_phase":
        raise ValueError("surface_parameter_mode must be tensile_centered_wave_count or manual_wavelength_phase")
    return (
        DoubleSineSurface(
            amplitude_mm=amplitude_mm,
            wavelength_x_mm=_query_float(params, "wavelength_x_mm", 40.0, positive=True),
            wavelength_y_mm=_query_float(params, "wavelength_y_mm", 50.0, positive=True),
            phase_x_rad=_query_phase_radians(
                params,
                pi_multiple_name="phase_x_pi",
                legacy_radians_name="phase_x_rad",
            ),
            phase_y_rad=_query_phase_radians(
                params,
                pi_multiple_name="phase_y_pi",
                legacy_radians_name="phase_y_rad",
            ),
            z_reference_mm=z_reference_mm,
        ),
        {"mode": mode, "phase_policy": "manual"},
    )


def _query_nonnegative_int(
    params: dict[str, list[str]], name: str, default: int, *, minimum: int = 0, maximum: int | None = None
) -> int:
    raw = params.get(name, [str(default)])[0]
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        upper = f", {maximum}" if maximum is not None else ""
        raise ValueError(f"{name} must be in the range [{minimum}{upper}]")
    return value


def _resolve_surface_progression_start(
    params: dict[str, list[str]], *, logical_layer_count: int
) -> tuple[int, int]:
    """Resolve the UI's physical first-curved-layer wording to legacy indices.

    The shared mapper contract continues to use ``surface_start_layer`` as a
    zero-based, zero-alpha boundary layer.  The design UI submits an explicit
    semantic marker, under which the same numeric input means the one-based
    physical layer that first has ``alpha > 0``.  Keeping the conversion here
    prevents the user-facing wording from leaking into mapper/Core contracts.
    """

    if logical_layer_count < 1:
        raise ValueError("logical_layer_count must be positive")
    semantics = params.get("surface_start_layer_semantics", ["legacy_zero_alpha_zero_based"])[0]
    if semantics == "first_nonzero_curvature_physical":
        first_curved_layer = _query_nonnegative_int(
            params, "surface_start_layer", 3, minimum=2
        )
        legacy_start_layer = first_curved_layer - 2
    elif semantics == "legacy_zero_alpha_zero_based":
        legacy_start_layer = _query_nonnegative_int(params, "surface_start_layer", 3)
        first_curved_layer = legacy_start_layer + 2
    else:
        raise ValueError("surface_start_layer_semantics is unsupported")

    if legacy_start_layer > (logical_layer_count - 1) // 2:
        raise ValueError("surface_start_layer must leave a symmetric curved region inside the final physical height")
    return legacy_start_layer, first_curved_layer


def _inspection_point(
    params: dict[str, list[str]],
    surface: DoubleSineSurface,
    *,
    x_bounds_mm: tuple[float, float],
    y_bounds_mm: tuple[float, float],
) -> dict[str, float]:
    """Return one inspectable target-surface point in the exported XY frame."""

    x_min_mm, x_max_mm = x_bounds_mm
    y_min_mm, y_max_mm = y_bounds_mm
    x_mm = _query_float(params, "check_x_mm", (x_min_mm + x_max_mm) / 2.0)
    y_mm = _query_float(params, "check_y_mm", (y_min_mm + y_max_mm) / 2.0)
    if not x_min_mm <= x_mm <= x_max_mm or not y_min_mm <= y_mm <= y_max_mm:
        raise ValueError("check point must lie inside the preview XY bounds")
    phase_x = (2.0 * math.pi * x_mm) / surface.wavelength_x_mm + surface.phase_x_rad
    phase_y = (2.0 * math.pi * y_mm) / surface.wavelength_y_mm + surface.phase_y_rad
    kx = 2.0 * math.pi / surface.wavelength_x_mm
    ky = 2.0 * math.pi / surface.wavelength_y_mm
    sin_x, cos_x = math.sin(phase_x), math.cos(phase_x)
    sin_y, cos_y = math.sin(phase_y), math.cos(phase_y)
    fx = surface.amplitude_mm * kx * cos_x * sin_y
    fy = surface.amplitude_mm * ky * sin_x * cos_y
    fxx = -surface.amplitude_mm * kx * kx * sin_x * sin_y
    fyy = -surface.amplitude_mm * ky * ky * sin_x * sin_y
    fxy = surface.amplitude_mm * kx * ky * cos_x * cos_y
    denominator = 2.0 * (1.0 + fx * fx + fy * fy) ** 1.5
    mean_curvature = (
        (1.0 + fy * fy) * fxx - 2.0 * fx * fy * fxy + (1.0 + fx * fx) * fyy
    ) / denominator
    return {
        "x_mm": x_mm,
        "y_mm": y_mm,
        "height_mm": float(surface.height(x_mm, y_mm)),
        "slope": math.hypot(fx, fy),
        "mean_curvature_per_mm": mean_curvature,
    }


def _conformal_solid_stack_payload(
    params: dict[str, list[str]],
    surface: DoubleSineSurface,
    *,
    x_samples_mm: list[float],
    section_y_mm: float,
) -> dict[str, object] | None:
    """Sample the existing smoothstep stack on a selected XZ cut."""

    if "part_height_mm" not in params and "surface_start_layer" not in params:
        return None
    final_height_mm = _query_float(params, "part_height_mm", 10.0, positive=True)
    layer_count = int(math.ceil(final_height_mm / CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM))
    start_layer, first_curved_layer = _resolve_surface_progression_start(
        params, logical_layer_count=layer_count
    )
    progression = LayerProgression(start_layer, layer_count - 1)
    layer_thicknesses = [CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM] * layer_count
    layer_thicknesses[-1] = final_height_mm - CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM * (layer_count - 1)
    base_z_by_layer: list[float] = []
    accumulated = 0.0
    for thickness in layer_thicknesses:
        base_z_by_layer.append(accumulated + thickness * 0.5)
        accumulated += thickness
    y_mm = section_y_mm
    target_displacements = [
        float(surface.height(x_mm, y_mm)) - surface.z_reference_mm
        for x_mm in x_samples_mm
    ]
    layers = []
    for index, base_z_mm in enumerate(base_z_by_layer):
        alpha = progression.alpha(index)
        layers.append(
            {
                "index": index,
                "alpha": alpha,
                "base_z_mm": base_z_mm,
                "xz_points": [
                    [x_mm, base_z_mm + surface.z_reference_mm + alpha * height_mm]
                    for x_mm, height_mm in zip(x_samples_mm, target_displacements, strict=True)
                ],
            }
        )
    return {
        "format": "conformal_solid_stack_preview_v1",
        "reference_layer_height_mm": CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM,
        "final_height_mm": final_height_mm,
        "section_y_mm": y_mm,
        "surface_start_layer": start_layer,
        "first_nonzero_curvature_layer_physical": first_curved_layer,
        "surface_start_layer_semantics": "first_nonzero_curvature_physical",
        "surface_return_layer": progression.surface_return_layer,
        "peak_layer_indices": list(progression.peak_layers),
        # The lower of the one/two complete-curvature layers is the stable
        # three-dimensional representative.  It is a physical layer centre,
        # not the Z=0 surface-definition reference plane.
        "representative_peak_layer_index": progression.peak_layers[0],
        "layers": layers,
    }


def surface_payload(
    params: dict[str, list[str]],
    domain: STLProjectionDomain | None = None,
    *,
    include_projection_geometry: bool = True,
    rectangle_origin_lower_left: bool = False,
) -> dict[str, object]:
    """Create browser-safe sampled geometry from URL query parameters.

    ``rectangle_origin_lower_left`` is specific to the STL-free conformal
    designer.  Its exported part contract fixes the rectangle at
    ``[0, 0] → [length, width]``; the preview must evaluate the same points
    rather than the legacy centered demonstration grid.
    """

    width_mm = domain.width_mm if domain else _query_float(
        params, "width_mm", DEFAULT_PREVIEW_WIDTH_MM, positive=True
    )
    height_mm = domain.height_mm if domain else _query_float(
        params, "height_mm", DEFAULT_PREVIEW_HEIGHT_MM, positive=True
    )
    surface, surface_parameterization = _surface_from_query(
        params,
        x_extent_mm=width_mm,
        y_extent_mm=height_mm,
    )
    samples = _query_samples(params)
    grid = surface.sample_grid(
        width_mm=width_mm,
        height_mm=height_mm,
        samples=samples,
        x_min_mm=0.0 if domain or rectangle_origin_lower_left else None,
        y_min_mm=0.0 if domain or rectangle_origin_lower_left else None,
    )
    grid_payload: dict[str, object] = {"x": grid.x.tolist(), "y": grid.y.tolist(), "z": grid.z.tolist()}
    if domain:
        center_x = (grid.x[:-1, :-1] + grid.x[:-1, 1:] + grid.x[1:, 1:] + grid.x[1:, :-1]) / 4.0
        center_y = (grid.y[:-1, :-1] + grid.y[:-1, 1:] + grid.y[1:, 1:] + grid.y[1:, :-1]) / 4.0
        grid_payload["material_mask"] = domain.material_mask(center_x, center_y).tolist()
    lower_left_origin = domain is not None or rectangle_origin_lower_left
    origin_label = (
        "stl_projection_lower_left"
        if domain is not None
        else "rectangle_lower_left"
        if lower_left_origin
        else "centered_preview"
    )
    xy_bounds_mm = (
        [0.0, 0.0, width_mm, height_mm]
        if lower_left_origin
        else [-width_mm / 2.0, -height_mm / 2.0, width_mm / 2.0, height_mm / 2.0]
    )
    x_bounds = (0.0, width_mm) if lower_left_origin else (-width_mm / 2.0, width_mm / 2.0)
    y_bounds = (0.0, height_mm) if lower_left_origin else (-height_mm / 2.0, height_mm / 2.0)
    inspection_enabled = _query_bool(params, "inspection_enabled", True)
    inspection = (
        _inspection_point(
            params,
            surface,
            x_bounds_mm=x_bounds,
            y_bounds_mm=y_bounds,
        )
        if inspection_enabled
        else None
    )
    section_y_mm = inspection["y_mm"] if inspection is not None else (y_bounds[0] + y_bounds[1]) / 2.0
    return {
        "preview_version": SURFACE_PREVIEW_API_VERSION,
        "export_version": CONFORMAL_LATTICE_SPEC_V1,
        "git_revision": SURFACE_PREVIEW_GIT_REVISION,
        "coordinate_system": {
            "origin_label": origin_label,
            "xy_bounds_mm": xy_bounds_mm,
        },
        "surface": {
            "type": "double_sine_product",
            "amplitude_mm": surface.amplitude_mm,
            "wavelength_x_mm": surface.wavelength_x_mm,
            "wavelength_y_mm": surface.wavelength_y_mm,
            "phase_x_rad": surface.phase_x_rad,
            "phase_y_rad": surface.phase_y_rad,
            "z_reference_mm": surface.z_reference_mm,
        },
        "surface_parameterization": surface_parameterization,
        "domain": {
            "width_mm": width_mm,
            "height_mm": height_mm,
            "samples": samples,
            "mode": "stl_projection" if domain else "rectangle",
            "projection": domain.preview_payload(include_polygons=include_projection_geometry) if domain else None,
        },
        "statistics": grid.summary(),
        "grid": grid_payload,
        "inspection_point": inspection,
        "solid_stack": _conformal_solid_stack_payload(
            params,
            surface,
            x_samples_mm=[float(value) for value in grid.x[0]],
            section_y_mm=section_y_mm,
        ),
    }


def graded_surface_config_payload(
    params: dict[str, list[str]], domain: STLProjectionDomain
) -> dict[str, object]:
    """Build the portable geometry-only sidecar consumed by the mapper."""

    surface = surface_payload(params, domain)["surface"]
    projection = domain.preview_payload()
    return {
        "format": "graded_surface_v1",
        "units": "mm",
        "coordinate_system": {
            "plane": "XY",
            "build_axis": "Z",
            "origin": "stl_xy_min",
            "source_build_axis": domain.build_axis,
        },
        "domain": {
            "mode": "stl_projection",
            "boundary": "stl_outer_contour",
            "edge_policy": "closed_perimeter",
            "source": {
                "file_name": projection["file_name"],
                "sha256": projection["sha256"],
                "triangle_count": projection["triangle_count"],
                "xy_bounds_mm": projection["source_xy_bounds_mm"],
                "projection_layer_height_mm": projection["projection_layer_height_mm"],
            },
        },
        "surface": surface,
    }


def conformal_lattice_config_payload(params: dict[str, list[str]]) -> dict[str, object]:
    """Build the STL-free rectangular conformal-design contract."""

    return _rectangular_lattice_config_payload(params, source_provider="double_sine")


def planar_lattice_config_payload(params: dict[str, list[str]]) -> dict[str, object]:
    """Build the flat-print contract without retaining double-sine parameters."""

    return _rectangular_lattice_config_payload(params, source_provider="planar")


def _rectangular_lattice_config_payload(
    params: dict[str, list[str]],
    *,
    source_provider: str,
) -> dict[str, object]:
    if source_provider not in {"double_sine", "planar"}:
        raise ValueError("source_provider must be double_sine or planar")
    length_mm = _query_float(params, "part_length_mm", 150.0, positive=True)
    width_mm = _query_float(params, "part_width_mm", 50.0, positive=True)
    final_height_mm = _query_float(params, "part_height_mm", 10.0, positive=True)
    grip_end_length_mm = _query_float(params, "grip_end_length_mm", 0.0)
    if grip_end_length_mm < 0.0:
        raise ValueError("grip_end_length_mm must be non-negative")
    if 2.0 * grip_end_length_mm >= length_mm:
        raise ValueError("two grip_end_length_mm regions must leave a positive honeycomb working length")
    # The design page describes surface morphology, not process settings.
    # Keep one stable reference for validating a layer-index start value; the
    # actual physical layer height is supplied later by the slicer/Core UI.
    layer_height_mm = CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM
    wall_width_mm = _query_float(params, "wall_width_mm", 2.0, positive=True)
    wall_bead_count = round(wall_width_mm / 2.0)
    if wall_bead_count < 1 or not math.isclose(wall_width_mm, 2.0 * wall_bead_count, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("wall_width_mm must be a positive integer multiple of the fixed 2 mm resin nozzle width")
    base_cell_size_mm = _query_float(params, "base_cell_size_mm", 5.0, positive=True)
    nominal_fill = (2.0 * wall_width_mm) / (math.sqrt(3.0) * base_cell_size_mm)
    if nominal_fill >= 1.0:
        raise ValueError(
            "wall_width_mm and base_cell_size_mm produce a nominal fill ratio of at least 1; "
            "reduce wall_width_mm or increase base_cell_size_mm"
        )
    boundary_mode = params.get("boundary_mode", ["clip"])[0]
    if boundary_mode not in {"clip", "inset"}:
        raise ValueError("boundary_mode must be clip or inset")
    phase_origin = [
        _query_float(params, "phase_origin_x_mm", 0.0),
        _query_float(params, "phase_origin_y_mm", 0.0),
    ]
    load_line_alignment_enabled = _query_bool(params, "align_load_line", False)
    honeycomb_align_x = _query_bool(params, "honeycomb_align_x", False)
    honeycomb_align_y = _query_bool(params, "honeycomb_align_y", False)
    honeycomb_align_x_mm = _query_float(
        params, "honeycomb_align_x_mm", length_mm / 2.0
    )
    honeycomb_align_y_mm = _query_float(
        params, "honeycomb_align_y_mm", width_mm / 2.0
    )
    if not 0.0 <= honeycomb_align_x_mm <= length_mm:
        raise ValueError("honeycomb_align_x_mm must lie within the part X range")
    if not 0.0 <= honeycomb_align_y_mm <= width_mm:
        raise ValueError("honeycomb_align_y_mm must lie within the part Y range")
    if load_line_alignment_enabled and (honeycomb_align_x or honeycomb_align_y):
        raise ValueError("three-point-bending load-line alignment and honeycomb X/Y alignment cannot be enabled together")
    orientation_angle_deg = 0.0 if (load_line_alignment_enabled or honeycomb_align_x or honeycomb_align_y) else _query_float(
        params, "orientation_angle_deg", 0.0
    )
    random_seed = _query_nonnegative_int(params, "random_seed", 0)
    if source_provider == "double_sine":
        surface_params = {**params, "width_mm": [str(length_mm)], "height_mm": [str(width_mm)]}
        surface = surface_payload(
            surface_params,
            include_projection_geometry=False,
            rectangle_origin_lower_left=True,
        )["surface"]
        logical_layer_count = math.ceil(final_height_mm / layer_height_mm)
        surface_start_layer, first_curved_layer = _resolve_surface_progression_start(
            params, logical_layer_count=logical_layer_count
        )
        samples_x = _query_nonnegative_int(
            params, "samples_x", DEFAULT_PREVIEW_SAMPLES, minimum=2, maximum=MAX_CONFORMAL_SAMPLES
        )
        samples_y = _query_nonnegative_int(
            params, "samples_y", DEFAULT_PREVIEW_SAMPLES, minimum=2, maximum=MAX_CONFORMAL_SAMPLES
        )
        source_surface: dict[str, object] = {
            "provider": "double_sine",
            "source_file": "generated://double-sine-rectangular-part",
            "domain": "outer_boundary_only",
            "double_sine": {
                **surface,
                "xy_bounds_mm": [0.0, 0.0, length_mm, width_mm],
                "samples": [samples_x, samples_y],
            },
        }
        source_surface["sha256"] = double_sine_source_sha256(source_surface)
        layer_embedding: dict[str, object] = {
            "mode": "symmetric_shape_morphing",
            "transition": "smoothstep",
            "surface_start_layer": surface_start_layer,
            "surface_start_layer_semantics": "legacy_zero_alpha_zero_based",
            "first_nonzero_curvature_layer_physical": first_curved_layer,
        }
    else:
        source_surface = {
            "provider": "planar",
            "source_file": "generated://planar-rectangular-part",
            "domain": "outer_boundary_only",
            "xy_bounds_mm": [0.0, 0.0, length_mm, width_mm],
        }
        source_surface["sha256"] = generated_surface_source_sha256(source_surface)
        layer_embedding = {"mode": "planar_stack"}
    config: dict[str, object] = {
        "format": CONFORMAL_LATTICE_SPEC_V1,
        "units": "mm",
        "source_surface": source_surface,
        "part": {
            "boundary": "rectangle",
            "length_mm": length_mm,
            "width_mm": width_mm,
            "final_height_mm": final_height_mm,
        },
        "manufacturing": {
            "layer_height_mm": layer_height_mm,
            "nominal_bead_width_mm": 2.0,
        },
        "parameterization": {
            "method": "lscm",
            "anchor_strategy": "farthest_boundary_pair",
            "seam_strategy": "none",
        },
        "lattice": {
            "family": "triangular_dual_hex",
            "wall_width_mm": wall_width_mm,
            "wall_bead_count": wall_bead_count,
            "base_cell_size_mm": base_cell_size_mm,
            "boundary_mode": boundary_mode,
            "phase_origin": phase_origin,
            "boundary_phase_policy": "auto_avoid_outer_boundary_coincidence",
            "load_line_alignment": {
                "enabled": load_line_alignment_enabled,
                "axis": "x",
                "position": "part_length_midplane",
                "feature": "wall",
            },
            "honeycomb_feature_alignment": {
                "align_x": honeycomb_align_x,
                "align_y": honeycomb_align_y,
                "target_x_mm": honeycomb_align_x_mm,
                "target_y_mm": honeycomb_align_y_mm,
                "x_feature": "y_directed_wall",
                "y_feature": "inclined_edge_zigzag_centerline",
                "scope": "center_features_only",
            },
        },
        "fill_field": {"mode": "fixed_cell_size", "drivers": []},
        "orientation_field": {"mode": "global_axis", "angle_deg": orientation_angle_deg, "constraints": []},
        "layer_embedding": layer_embedding,
        "quality_limits": {},
        "random_seed": random_seed,
    }
    if grip_end_length_mm > 0.0:
        # A pure geometry range: the executable zigzag process settings are
        # intentionally supplied later by the resin path planner/Core preset.
        config["part"]["symmetric_grip_end_length_mm"] = grip_end_length_mm  # type: ignore[index]
    load_conformal_lattice_spec(config)
    return config


def run_surface_preview_server(host: str, port: int) -> None:
    """Start the independent local surface-preview server."""

    server = ThreadingHTTPServer((host, port), SurfacePreviewHandler)
    server.preview_domains = {}
    server.designer_state_path = _designer_state_path()
    print(f"KUKA surface preview running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()


class SurfacePreviewHandler(BaseHTTPRequestHandler):
    """Serve the future-embeddable web shell and sampled surface data."""

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(surface_preview_html())
            return
        if parsed.path == "/api/surface":
            try:
                params = parse_qs(parsed.query)
                domain = self._domain_from_params(params)
                payload = surface_payload(
                    params,
                    domain,
                    include_projection_geometry=params.get("compact", [""])[0] != "1",
                    rectangle_origin_lower_left=domain is None,
                )
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True, **payload})
            return
        if parsed.path == "/api/designer-state":
            self._send_json({"ok": True, "state": _load_designer_state(self.server.designer_state_path)})
            return
        if parsed.path == "/api/export-surface-config":
            try:
                params = parse_qs(parsed.query)
                config = graded_surface_config_payload(
                    params, self._domain_from_params(params, required=True)
                )
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                config,
                attachment_name="graded_surface_v1.json",
            )
            return
        if parsed.path == "/api/export-conformal-lattice-config":
            try:
                params = parse_qs(parsed.query)
                config = conformal_lattice_config_payload(params)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                config,
                attachment_name="conformal_lattice_spec_v1.json",
            )
            return
        if parsed.path == "/api/export-planar-lattice-config":
            try:
                params = parse_qs(parsed.query)
                config = planar_lattice_config_payload(params)
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json(
                config,
                attachment_name="planar_honeycomb_spec_v1.json",
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/designer-state":
            try:
                raw_length = self.headers.get("Content-Length")
                if raw_length is None:
                    raise ValueError("designer state requires Content-Length")
                content_length = int(raw_length)
                if not 0 < content_length <= MAX_DESIGNER_STATE_BYTES:
                    raise ValueError("designer state must be between 1 byte and 32 KB")
                raw_state = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not isinstance(raw_state, dict):
                    raise ValueError("designer state must be an object")
                state = dict(raw_state)
                _save_designer_state(self.server.designer_state_path, state)
            except (UnicodeDecodeError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True})
            return
        if parsed.path != "/api/stl-domain":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            params = parse_qs(parsed.query)
            build_axis = params.get("build_axis", ["z"])[0]
            if build_axis not in ("x", "y", "z"):
                raise ValueError("build_axis must be x, y, or z")
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise ValueError("STL upload requires Content-Length")
            content_length = int(raw_length)
            if not 0 < content_length <= MAX_STL_BYTES:
                raise ValueError(f"STL must be between 1 byte and {MAX_STL_BYTES // 1024 // 1024} MB")
            data = self.rfile.read(content_length)
            file_name = unquote(self.headers.get("X-STL-File-Name", "model.stl"))
            domain = stl_projection_domain_from_bytes(
                data,
                file_name=file_name,
                build_axis=build_axis,
            )
        except (TypeError, ValueError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        domain_id = secrets.token_urlsafe(18)
        self.server.preview_domains[domain_id] = domain
        self._send_json({"ok": True, "domain_id": domain_id, "projection": domain.preview_payload()})

    def _domain_from_params(
        self, params: dict[str, list[str]], *, required: bool = False
    ) -> STLProjectionDomain | None:
        domain_id = params.get("domain_id", [""])[0]
        if not domain_id:
            if required:
                raise ValueError("import an STL before exporting a surface configuration")
            return None
        domain = self.server.preview_domains.get(domain_id)
        if domain is None:
            raise ValueError("the imported STL is no longer available; import it again")
        return domain

    def _send_html(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(
        self,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
        *,
        attachment_name: str | None = None,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if attachment_name:
            self.send_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def surface_preview_html() -> str:
    """Return the self-contained browser shell without third-party dependencies."""

    return r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KUKA 曲面预览器</title>
  <style>
    :root { color-scheme: light; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; color: #152033; background: #f4f7fb; }
    * { box-sizing: border-box; }
    body { margin: 0; min-width: 320px; }
    main { max-width: 1260px; margin: 0 auto; padding: 24px; }
    header { margin-bottom: 18px; }
    h1 { margin: 0; font-size: clamp(22px, 3vw, 32px); letter-spacing: -.02em; }
    header p { color: #526074; margin: 8px 0 0; line-height: 1.55; }
    .workspace { display: grid; grid-template-columns: minmax(260px, 340px) minmax(0, 1fr); gap: 18px; align-items: start; }
    .panel { background: #fff; border: 1px solid #dbe3ef; border-radius: 14px; box-shadow: 0 10px 30px rgba(32, 52, 82, .07); }
    .controls { padding: 18px; }
    .controls h2, .preview h2 { margin: 0 0 14px; font-size: 16px; }
    .field { display: grid; grid-template-columns: 1fr 112px; align-items: center; gap: 10px; margin: 10px 0; }
    label { font-size: 13px; color: #38475d; }
    input { width: 100%; border: 1px solid #bdcadb; border-radius: 7px; padding: 8px; color: #142238; font: inherit; }
    input[type="checkbox"] { width: 18px; height: 18px; justify-self: start; padding: 0; }
    input:disabled { background: #f0f4f8; color: #7b8798; cursor: not-allowed; }
    input:focus { outline: 3px solid rgba(36, 122, 207, .18); border-color: #247acf; }
    .divider { height: 1px; background: #e6ecf4; margin: 17px 0; }
    button { width: 100%; border: 0; border-radius: 8px; padding: 10px 12px; background: #126fd1; color: white; font: 600 14px inherit; cursor: pointer; }
    button:hover { background: #075eaf; }
    button:disabled { background: #9ba9ba; cursor: not-allowed; }
    button.secondary { background: #eaf2fb; color: #0b5da9; margin-top: 8px; }
    button.secondary:hover { background: #dcebf9; }
    select { width: 100%; border: 1px solid #bdcadb; border-radius: 7px; padding: 8px; color: #142238; font: inherit; background: #fff; }
    .fileInput { margin: 8px 0 0; font-size: 12px; }
    .modelMeta { min-height: 18px; margin: 9px 0 0; color: #526074; font-size: 12px; line-height: 1.5; word-break: break-word; }
    .hint { margin: 12px 0 0; color: #66758b; font-size: 12px; line-height: 1.55; }
    .designSummary { margin: 10px 0 0; padding: 9px 10px; border: 1px solid #d9e5f4; border-radius: 8px; background: #f5f9fd; color: #40516a; font-size: 12px; line-height: 1.55; }
    .designSummary.error { border-color: #f0c5c2; background: #fff7f6; color: #a52a21; }
    details.advanced { margin-top: 12px; color: #40516a; font-size: 13px; }
    details.advanced summary { cursor: pointer; color: #2e405a; font-weight: 600; }
    .advancedBody { padding-top: 4px; }
    .preview { overflow: hidden; }
    .previewHead { padding: 18px 18px 0; display: flex; justify-content: space-between; gap: 12px; align-items: start; }
    .stats { display: flex; flex-wrap: wrap; gap: 7px; justify-content: end; }
    .stat { border: 1px solid #dae4f1; border-radius: 999px; padding: 4px 8px; color: #40516a; font-size: 12px; white-space: nowrap; }
    canvas { display: block; width: 100%; height: min(65vh, 620px); min-height: 400px; background: linear-gradient(180deg, #fbfdff 0%, #eef4fa 100%); touch-action: none; cursor: grab; }
    canvas.isDragging { cursor: grabbing; }
    .navigationHint { margin: 0; padding: 10px 18px 14px; color: #66758b; font-size: 12px; line-height: 1.5; border-top: 1px solid #edf1f6; }
    .status { min-height: 18px; padding: 0 18px 16px; color: #68778c; font-size: 12px; }
    .status.error { color: #b42318; }
    @media (max-width: 820px) { main { padding: 14px; } .workspace { grid-template-columns: 1fr; } canvas { min-height: 330px; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>蜂窝网格共形设计器</h1>
      <p>在矩形实体上定义双正弦共形曲面与六边形格栅；默认采用拉伸标距段策略，导出 JSON 后回到主切片器生成路径与送入 Core。</p>
    </header>
    <section class="workspace">
      <form class="panel controls" id="surfaceForm">
        <h2>矩形实体</h2>
        <div class="field"><label for="part_length_mm">零件长度 X（mm）</label><input id="part_length_mm" type="number" min="0.001" step="1" value="150"></div>
        <div class="field"><label for="part_width_mm">零件宽度 Y（mm）</label><input id="part_width_mm" type="number" min="0.001" step="1" value="50"></div>
        <div class="field"><label for="part_height_mm">最终物理高度 Z（mm）</label><input id="part_height_mm" type="number" min="0.001" step="0.1" value="10"></div>
        <div class="field"><label for="grip_end_length_mm">每端夹持区 X（mm）</label><input id="grip_end_length_mm" type="number" min="0" step="0.5" value="25" aria-describedby="gripLengthHint"></div>
        <p class="hint" id="gripLengthHint">两端采用相同长度；蜂窝工作段为 X 总长 − 2 × 每端夹持区。曲面仍按完整零件 X/Y 范围计算，不会因夹持区而改变波长、相位或曲率。</p>
        <p class="modelMeta" id="modelMeta">外边界固定为矩形；新共形流程不读取 STL，也不继承 STL 中的蜂窝孔壁。</p>
        <div class="divider"></div>
        <h2>曲面参数</h2>
        <div class="field"><label for="surface_parameter_mode">曲面参数策略</label><select id="surface_parameter_mode"><option value="tensile_centered_wave_count" selected>拉伸：试样中心对称波数</option><option value="manual_wavelength_phase">手动：波长与相位</option></select></div>
        <div class="field"><label for="amplitude_mm">幅值 A（mm）</label><input id="amplitude_mm" type="number" step="0.01" value="1.5"></div>
        <div id="tensileWaveFields">
          <div class="field"><label for="wave_count_x">X 向波数 nx</label><input id="wave_count_x" type="number" min="0.5" step="1" value="1.5"></div>
          <div class="field"><label for="wave_count_y">Y 向波数 ny</label><input id="wave_count_y" type="number" min="0.5" step="1" value="1.5"></div>
          <button type="button" class="secondary" id="applyTensilePreset">应用拉伸中间参数组</button>
          <p class="hint" id="tensileWaveHint">仅允许 0.5、1.5、2.5… 等半整数波数。波数按完整试样 X/Y 尺寸归一化：自动计算 λx、λy 与相位，使试样中心为正峰，四周边界回到 H=0；改变矩形尺寸不会改变无量纲曲面构型。</p>
        </div>
        <div id="manualSurfaceFields" hidden>
          <div class="field"><label for="wavelength_x_mm">X 波长 λx（mm）</label><input id="wavelength_x_mm" type="number" min="0.001" step="0.1" value="100"></div>
          <div class="field"><label for="wavelength_y_mm">Y 波长 λy（mm）</label><input id="wavelength_y_mm" type="number" min="0.001" step="0.1" value="66.667"></div>
          <div class="field"><label for="phase_x_pi">X 相位 φx（π）</label><input id="phase_x_pi" type="number" step="0.25" value="1" aria-describedby="phasePiHint"></div>
          <div class="field"><label for="phase_y_pi">Y 相位 φy（π）</label><input id="phase_y_pi" type="number" step="0.25" value="1" aria-describedby="phasePiHint"></div>
          <p class="hint" id="phasePiHint">输入 π 的倍数：1 表示 π，0.5 表示 π/2，1.5 表示 3π/2；导出的设计 JSON 仍以 rad 保存。</p>
        </div>
        <div class="field"><label for="z_reference_mm">Z 基准（mm）</label><input id="z_reference_mm" type="number" step="0.01" value="0"></div>
        <div class="divider"></div>
        <h2>弯曲专用检验（可选）</h2>
        <div class="field"><label for="inspection_enabled">显示弯曲检验点</label><input id="inspection_enabled" type="checkbox"></div>
        <div id="inspectionPointFields" hidden>
          <div class="field"><label for="check_x_mm">检验点 X（mm）</label><input id="check_x_mm" type="number" min="0" step="0.1" value="75" aria-describedby="checkPointHint"></div>
          <div class="field"><label for="check_y_mm">检验点 Y（mm）</label><input id="check_y_mm" type="number" min="0" step="0.1" value="50" aria-describedby="checkPointHint"></div>
          <p class="hint" id="checkPointHint">仅在第三章三点弯曲时启用。启用后可查看任意点的 H、坡度和平均曲率；尺寸变化时，超出矩形范围的坐标会自动收回到范围内。</p>
        </div>
        <div class="divider"></div>
        <h2>连续路径蜂窝（预览）</h2>
        <div class="field"><label for="base_cell_size_mm">目标六边形边长（mm）</label><input id="base_cell_size_mm" type="number" min="0.001" step="0.01" value="10"></div>
        <p class="hint">以目标边长为唯一蜂窝几何参数。一个完整黄色孔洞中心固定在蜂窝工作区中心；300 × 300 mm 母板只用于向外铺展，再按当前工作区逐边裁剪。裁剪窗已扣除外矩形轮廓和夹持分界树脂带的半宽，因此边界允许出现截断六边形，但不会穿入树脂轮廓。</p>
        <p class="hint">红线为 2 mm 连续纤维的中心线预览：先沿黄色孔洞之间可容纳纤维的 X 向材料通道绕行，再在左右夹持区保持当前 Y 高度直线延伸到零件边界。绿色点为起点、深红点为终点；纤维只在试样端部切断。</p>
        <!-- Kept only so the still-supported legacy JSON form remains readable while
             the new continuous-course topology is preview-only. -->
        <div hidden aria-hidden="true">
          <input id="wall_width_mm" value="2">
          <input id="orientation_angle_deg" value="0">
          <input id="align_load_line" type="checkbox">
          <input id="honeycomb_align_x" type="checkbox">
          <input id="honeycomb_align_x_mm" value="75">
          <input id="honeycomb_align_y" type="checkbox">
          <input id="honeycomb_align_y_mm" value="25">
          <button type="button" id="centreHoneycombAlignment"></button>
          <span id="loadLineAlignmentHint"></span><span id="honeycombAlignmentHint"></span>
        </div>
        <div class="designSummary" id="latticeDesignSummary" aria-live="polite"></div>
        <div class="designSummary" id="latticeLengthSummary" aria-live="polite">连续路径总长将在曲面预览更新后显示。</div>
        <p class="hint">长度是平面预览中每条完整连续路径的累加，不包含层数和曲面映射造成的弧长变化；后续接入路径内核时会重新以实际三维长度计算挤出量。</p>
        <div class="divider"></div>
        <h2>对称层间渐变</h2>
        <div class="field"><label for="surface_start_layer">首个非零曲率层（物理层）</label><input id="surface_start_layer" type="number" min="2" step="1" value="3"></div>
        <p class="hint">以自下而上、从 1 开始计数。填 3 表示第 1–2 层为平面，第 3 层首次出现非零曲率；连续纤维可在第 2 层树脂完成后铺设。导出仍保留旧映射器所需的零基边界层索引。</p>
        <div class="designSummary" id="layerProgressionSummary" aria-live="polite"></div>
        <details class="advanced">
          <summary>高级参数（共形计算）</summary>
          <div class="advancedBody">
            <div class="field"><label for="samples_x">曲面采样 X</label><input id="samples_x" type="number" min="2" max="512" step="1" value="49"></div>
            <div class="field"><label for="samples_y">曲面采样 Y</label><input id="samples_y" type="number" min="2" max="512" step="1" value="49"></div>
            <div class="field"><label for="boundary_mode">边界策略</label><select id="boundary_mode"><option value="clip" selected>裁剪至矩形</option><option value="inset">向内缩进</option></select></div>
            <div class="field"><label for="random_seed">随机种子</label><input id="random_seed" type="number" min="0" step="1" value="0"></div>
            <div class="field"><label for="samples">预览网格密度</label><input id="samples" type="number" min="8" max="120" step="1" value="49"></div>
            <p class="hint">曲面采样 X/Y 参与共形计算；预览网格密度只影响本页显示。格栅相位由导出流程在实际曲面相位域自动避让，避免蜂窝墙与矩形外边界重合；参数化固定使用 LSCM、最远边界锚点和无切缝。</p>
          </div>
        </details>
        <div class="divider"></div>
        <h2>下一步</h2>
        <button type="button" id="exportConformalConfig">导出连续路径 JSON</button>
        <button type="button" class="secondary" id="exportPlanarConfig">导出平面蜂窝结构 JSON</button>
        <button type="button" class="secondary" id="reset">恢复示例参数</button>
        <p class="hint">曲面 JSON 包含双正弦参数；平面 JSON 只保留当前零件尺寸、分区和蜂窝路径/形状样式。两者都可直接导入主切片器，并复用同一套树脂、可选纤维及 Core 工艺参数。</p>
      </form>
      <section class="panel preview">
        <div class="previewHead"><h2 id="previewTitle">α=1 完整曲率层（物理 Z）</h2><div class="stats" id="stats"></div></div>
        <div class="field"><label for="previewMode">预览模式</label><select id="previewMode"><option value="surface" selected>α=1 完整曲率层（物理 Z）</option><option value="solid_xz">实体层叠 / XZ 剖面</option></select></div>
        <div class="field"><label for="surfaceZScale">三维视觉 Z 放大</label><select id="surfaceZScale"><option value="1">真实比例 ×1</option><option value="3">形态观察 ×3</option><option value="5" selected>形态观察 ×5</option><option value="10">形态观察 ×10</option></select></div>
        <div class="field"><label for="sectionZScale">XZ 剖面视觉 Z 放大</label><select id="sectionZScale"><option value="1">真实比例 ×1</option><option value="3" selected>辅助观察 ×3</option><option value="5">辅助观察 ×5</option></select></div>
        <p class="hint">视觉 Z 放大只影响画布，不改变参数、检验值、导出的 JSON 或实际零件尺寸。XZ 剖面采用统一 X/Z 比例后再按所选倍率放大 Z，避免隐藏的纵向拉伸。</p>
        <canvas id="canvas" aria-label="双正弦曲面预览"></canvas>
        <p class="navigationHint">左键拖拽旋转；中键拖拽平移；右键上下拖拽缩放；滚轮缩放；双击恢复视角。</p>
        <div class="status" id="status">正在生成曲面…</div>
      </section>
    </section>
  </main>
  <script>
    const surfaceIds = ['surface_parameter_mode', 'amplitude_mm', 'wave_count_x', 'wave_count_y', 'wavelength_x_mm', 'wavelength_y_mm', 'phase_x_pi', 'phase_y_pi', 'z_reference_mm', 'inspection_enabled', 'check_x_mm', 'check_y_mm', 'samples'];
    const mappingReferenceLayerHeightMm = 0.5;
    const conformalDesignIds = ['part_length_mm', 'part_width_mm', 'part_height_mm', 'grip_end_length_mm', 'wall_width_mm', 'base_cell_size_mm', 'orientation_angle_deg', 'align_load_line', 'honeycomb_align_x', 'honeycomb_align_x_mm', 'honeycomb_align_y', 'honeycomb_align_y_mm', 'surface_start_layer', 'samples_x', 'samples_y', 'boundary_mode', 'random_seed'];
    const canvas = document.getElementById('canvas');
    const statusEl = document.getElementById('status');
    const statsEl = document.getElementById('stats');
    const exportConformalConfigButton = document.getElementById('exportConformalConfig');
    const exportPlanarConfigButton = document.getElementById('exportPlanarConfig');
    const previewMode = document.getElementById('previewMode');
    const surfaceZScale = document.getElementById('surfaceZScale');
    const sectionZScale = document.getElementById('sectionZScale');
    const previewTitle = document.getElementById('previewTitle');
    const designerStateKey = 'kuka-slicer.conformal-designer-state.v1';
    const latticePreviewLimit = 1600;
    const persistedInputIds = [...new Set([
      ...surfaceIds,
      ...conformalDesignIds,
      'surfaceZScale',
      'sectionZScale',
      'previewMode',
    ])];
    let payload = null;
    let queued = 0;
    let latticePreviewCache = null;
    let continuousCoursePreviewCache = null;
    let designerStateDirty = false;
    let persistentStateSaveTimer = null;
    const initialView = { yaw: -42 * Math.PI / 180, pitch: 54 * Math.PI / 180, zoom: 1, panX: 0, panY: 0 };
    const view = { ...initialView };
    let drag = null;

    function positiveNumber(id) {
      const value = Number(document.getElementById(id).value);
      return Number.isFinite(value) && value > 0 ? value : null;
    }

    function nonNegativeNumber(id) {
      const value = Number(document.getElementById(id).value);
      return Number.isFinite(value) && value >= 0 ? value : null;
    }

    function honeycombActiveXBounds() {
      const length = positiveNumber('part_length_mm');
      const grip = nonNegativeNumber('grip_end_length_mm');
      if (length === null || grip === null || 2 * grip >= length) return null;
      return [grip, length - grip];
    }

    function nonNegativeInteger(id) {
      const value = Number(document.getElementById(id).value);
      return Number.isInteger(value) && value >= 0 ? value : null;
    }

    function currentDesignerState() {
      return Object.fromEntries(persistedInputIds.map((id) => {
        const element = document.getElementById(id);
        return [id, element.type === 'checkbox' ? element.checked : element.value];
      }));
    }

    function applyDesignerState(state) {
      if (!state || typeof state !== 'object') return false;
      const needsTensileMigration = !Object.prototype.hasOwnProperty.call(state, 'surface_parameter_mode');
      persistedInputIds.forEach((id) => {
        const element = document.getElementById(id);
        if (element.type === 'checkbox' && typeof state[id] === 'boolean') element.checked = state[id];
        else if (typeof state[id] === 'string') element.value = state[id];
      });
      if (needsTensileMigration) {
        document.getElementById('surface_parameter_mode').value = 'tensile_centered_wave_count';
        document.getElementById('wave_count_x').value = 1.5;
        document.getElementById('wave_count_y').value = 1.5;
        document.getElementById('inspection_enabled').checked = false;
        document.getElementById('align_load_line').checked = false;
      }
      return true;
    }

    function saveDesignerState() {
      const state = currentDesignerState();
      designerStateDirty = true;
      try {
        localStorage.setItem(designerStateKey, JSON.stringify(state));
      } catch (_) {
        // The per-user state file below remains available when browser storage is not.
      }
      clearTimeout(persistentStateSaveTimer);
      persistentStateSaveTimer = setTimeout(() => {
        fetch('/api/designer-state', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(state),
        }).catch(() => {});
      }, 250);
    }

    function restoreDesignerState() {
      try {
        applyDesignerState(JSON.parse(localStorage.getItem(designerStateKey) || 'null'));
      } catch (_) {
        // Ignore malformed or unavailable browser-local state.
      }
    }

    async function restorePersistentDesignerState() {
      try {
        const response = await fetch('/api/designer-state');
        const result = await response.json();
        if (!designerStateDirty && response.ok && result.ok) applyDesignerState(result.state);
      } catch (_) {
        // Browser-local state remains available when the local state file cannot be read.
      }
    }

    function syncSurfaceParameterControls() {
      const tensileMode = document.getElementById('surface_parameter_mode').value === 'tensile_centered_wave_count';
      document.getElementById('tensileWaveFields').hidden = !tensileMode;
      document.getElementById('manualSurfaceFields').hidden = tensileMode;
      ['wave_count_x', 'wave_count_y'].forEach((id) => { document.getElementById(id).disabled = !tensileMode; });
      ['wavelength_x_mm', 'wavelength_y_mm', 'phase_x_pi', 'phase_y_pi'].forEach((id) => { document.getElementById(id).disabled = tensileMode; });
      updateTensileWaveHint();
    }

    function updateTensileWaveHint() {
      const hint = document.getElementById('tensileWaveHint');
      const length = positiveNumber('part_length_mm');
      const width = positiveNumber('part_width_mm');
      const nx = positiveNumber('wave_count_x');
      const ny = positiveNumber('wave_count_y');
      const isHalfInteger = (waves) => Math.abs((waves - 0.5) - Math.round(waves - 0.5)) < 1e-9;
      if (length === null || width === null || nx === null || ny === null || !isHalfInteger(nx) || !isHalfInteger(ny)) {
        hint.textContent = '请输入正的试样尺寸，以及 0.5、1.5、2.5… 等半整数 X/Y 波数，以自动换算波长和相位。';
        return;
      }
      const phasePi = (waves) => ((0.5 - waves) % 2 + 2) % 2;
      hint.textContent = `当前换算：λx=${(length / nx).toFixed(3)} mm，λy=${(width / ny).toFixed(3)} mm，φx=${phasePi(nx).toFixed(3)}π，φy=${phasePi(ny).toFixed(3)}π。中心为正峰，边界 H=0；改变矩形尺寸时保持 nx、ny 不变即可保持同类构型。`;
    }

    function clampInspectionPointToPartBounds() {
      if (!document.getElementById('inspection_enabled').checked) return;
      const length = positiveNumber('part_length_mm');
      const width = positiveNumber('part_width_mm');
      if (length === null || width === null) return;
      const xInput = document.getElementById('check_x_mm');
      const yInput = document.getElementById('check_y_mm');
      const x = Number(xInput.value);
      const y = Number(yInput.value);
      xInput.value = Number.isFinite(x) ? Math.min(length, Math.max(0, x)) : length / 2;
      yInput.value = Number.isFinite(y) ? Math.min(width, Math.max(0, y)) : width / 2;
    }

    function syncInspectionPointControls() {
      const enabled = document.getElementById('inspection_enabled').checked;
      const fields = document.getElementById('inspectionPointFields');
      fields.hidden = !enabled;
      ['check_x_mm', 'check_y_mm'].forEach((id) => { document.getElementById(id).disabled = !enabled; });
      if (enabled) clampInspectionPointToPartBounds();
    }

    function updateConformalDesignSummary() {
      const cellSize = positiveNumber('base_cell_size_mm');
      const activeXBounds = honeycombActiveXBounds();
      const latticeSummary = document.getElementById('latticeDesignSummary');
      if (activeXBounds === null) {
        latticeSummary.className = 'designSummary error';
        latticeSummary.textContent = '每端夹持区必须为非负数，且两端夹持区之和必须小于零件长度，才能留下蜂窝工作段。';
      } else if (cellSize === null) {
        latticeSummary.className = 'designSummary error';
        latticeSummary.textContent = '目标六边形边长必须为正数。';
      } else {
        latticeSummary.className = 'designSummary';
        const workingLength = activeXBounds[1] - activeXBounds[0];
        const width = positiveNumber('part_width_mm');
        const estimatedCourses = width === null ? '?' : Math.max(1, Math.floor(width / (Math.sqrt(3) * cellSize)));
        latticeSummary.textContent = `连续路径工作段：X=${activeXBounds[0].toFixed(2)}–${activeXBounds[1].toFixed(2)} mm（长 ${workingLength.toFixed(2)} mm）；目标边长 ${cellSize.toFixed(2)} mm；预计约 ${estimatedCourses} 条 X 向长路径。路径数由完整单元能否落入矩形决定，不以蜂窝边数计。`;
      }

      const firstCurvedLayer = nonNegativeInteger('surface_start_layer');
      const samplesX = nonNegativeInteger('samples_x');
      const samplesY = nonNegativeInteger('samples_y');
      const partHeight = positiveNumber('part_height_mm');
      const progressionSummary = document.getElementById('layerProgressionSummary');
      if (firstCurvedLayer === null || firstCurvedLayer < 2) {
        progressionSummary.className = 'designSummary error';
        progressionSummary.textContent = '首个非零曲率层必须是大于等于 2 的物理层号。';
      } else if (partHeight === null) {
        progressionSummary.className = 'designSummary error';
        progressionSummary.textContent = '最终物理高度必须是正数。';
      } else if (samplesX === null || samplesX < 2 || samplesY === null || samplesY < 2) {
        progressionSummary.className = 'designSummary error';
        progressionSummary.textContent = '曲面采样 X 和 Y 都必须是不小于 2 的整数。';
      } else {
        const layerCount = Math.ceil(partHeight / mappingReferenceLayerHeightMm);
        const maxLegacyStart = Math.floor((layerCount - 1) / 2);
        const maxFirstCurvedLayer = maxLegacyStart + 2;
        if (firstCurvedLayer > maxFirstCurvedLayer) {
          progressionSummary.className = 'designSummary error';
          progressionSummary.textContent = `当前高度与参考层高共得到 ${layerCount} 个物理层；首个非零曲率层不能大于 ${maxFirstCurvedLayer}。`;
        } else {
          const legacyStartLayer = firstCurvedLayer - 2;
          const returnLayerPhysical = layerCount - legacyStartLayer;
          const peakLayers = layerCount % 2 === 1 ? `${Math.floor(layerCount / 2) + 1}` : `${layerCount / 2}、${layerCount / 2 + 1}`;
          progressionSummary.className = 'designSummary';
          progressionSummary.textContent = `映射参考层数：${layerCount}；首个非零曲率层：第 ${firstCurvedLayer} 层；对称回落至平面：第 ${returnLayerPhysical} 层；完整曲率层：第 ${peakLayers} 层；共形采样：${samplesX} × ${samplesY}。实际切片层高在主界面 Core 工艺参数中设置。`;
        }
      }
    }

    function parameters() {
      const query = new URLSearchParams();
      surfaceIds.forEach((id) => {
        const element = document.getElementById(id);
        query.set(id, element.type === 'checkbox' ? String(element.checked) : element.value);
      });
      query.set('width_mm', document.getElementById('part_length_mm').value);
      query.set('height_mm', document.getElementById('part_width_mm').value);
      query.set('part_height_mm', document.getElementById('part_height_mm').value);
      query.set('surface_start_layer', document.getElementById('surface_start_layer').value);
      query.set('surface_start_layer_semantics', 'first_nonzero_curvature_physical');
      return query;
    }

    function conformalParameters() {
      const query = parameters();
      conformalDesignIds.forEach((id) => {
        const element = document.getElementById(id);
        query.set(id, element.type === 'checkbox' ? String(element.checked) : element.value);
      });
      return query;
    }

    function syncLoadLineAlignmentControls() {
      const enabled = document.getElementById('align_load_line').checked;
      const orientation = document.getElementById('orientation_angle_deg');
      const hint = document.getElementById('loadLineAlignmentHint');
      if (enabled) {
        document.getElementById('honeycomb_align_x').checked = false;
        document.getElementById('honeycomb_align_y').checked = false;
      }
      syncHoneycombAlignmentControls();
      const honeycombAligned = document.getElementById('honeycomb_align_x').checked || document.getElementById('honeycomb_align_y').checked;
      if (enabled || honeycombAligned) orientation.value = 0;
      orientation.disabled = enabled || honeycombAligned;
      hint.textContent = enabled
        ? '已开启：自动在零件长度中面定位一条沿 Y 的蜂窝壁；该选项只服务于第三章三点弯曲，不改变双正弦曲面参数。'
        : '默认关闭：使用全局格栅方向角；拉伸试验不需要加载线蜂窝壁对齐。';
    }

    function syncHoneycombAlignmentControls() {
      const alignX = document.getElementById('honeycomb_align_x').checked;
      const alignY = document.getElementById('honeycomb_align_y').checked;
      document.getElementById('honeycomb_align_x_mm').disabled = !alignX;
      document.getElementById('honeycomb_align_y_mm').disabled = !alignY;
      const hint = document.getElementById('honeycombAlignmentHint');
      hint.textContent = alignX || alignY
        ? '已开启中心特征对齐：X 放置沿 Y 蜂窝壁，Y 放置纤维所用斜边锯齿链的几何中线；链会在目标 Y 上下交替，并非水平墙。格栅方向固定为 0°；这不代表裁切后的整张网格严格镜像对称。'
        : '已关闭中心特征对齐：使用手动/自动避边相位。可保留任意格栅方向，但连续纤维路径不再保证围绕零件中线布置。';
    }

    function colour(fraction, lighting = 1) {
      const value = Math.max(0, Math.min(1, fraction));
      const luminance = Math.max(28, Math.min(91, (88 - value * 38) * lighting));
      return `hsl(204, 68%, ${luminance}%)`;
    }

    function surfaceLighting(x, y) {
      const surface = payload.surface;
      const xPhase = (2 * Math.PI * x) / surface.wavelength_x_mm + surface.phase_x_rad;
      const yPhase = (2 * Math.PI * y) / surface.wavelength_y_mm + surface.phase_y_rad;
      const dx = surface.amplitude_mm * (2 * Math.PI / surface.wavelength_x_mm) * Math.cos(xPhase) * Math.sin(yPhase);
      const dy = surface.amplitude_mm * (2 * Math.PI / surface.wavelength_y_mm) * Math.sin(xPhase) * Math.cos(yPhase);
      const normalLength = Math.hypot(dx, dy, 1);
      const normal = [-dx / normalLength, -dy / normalLength, 1 / normalLength];
      const light = [-0.38, -0.46, 0.8];
      const diffuse = Math.max(0, normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2]);
      return 0.62 + 0.48 * diffuse;
    }

    function rotateVector(x, y, angle) {
      return [x * Math.cos(angle) - y * Math.sin(angle), x * Math.sin(angle) + y * Math.cos(angle)];
    }

    function clipSegmentToBounds(start, end, bounds) {
      const [xMin, yMin, xMax, yMax] = bounds;
      const dx = end[0] - start[0];
      const dy = end[1] - start[1];
      const limits = [
        [-dx, start[0] - xMin], [dx, xMax - start[0]],
        [-dy, start[1] - yMin], [dy, yMax - start[1]],
      ];
      let lower = 0;
      let upper = 1;
      for (const [p, q] of limits) {
        if (Math.abs(p) < 1e-12) {
          if (q < 0) return null;
          continue;
        }
        const ratio = q / p;
        if (p < 0) lower = Math.max(lower, ratio);
        else upper = Math.min(upper, ratio);
        if (lower > upper) return null;
      }
      return [
        [start[0] + lower * dx, start[1] + lower * dy],
        [start[0] + upper * dx, start[1] + upper * dy],
      ];
    }

    function latticePreviewParameters() {
      const edgeLength = positiveNumber('base_cell_size_mm');
      const wallWidth = positiveNumber('wall_width_mm');
      if (edgeLength === null || wallWidth === null || !payload) return null;
      const partBounds = payload.coordinate_system.xy_bounds_mm;
      const activeXBounds = honeycombActiveXBounds();
      if (activeXBounds === null) return null;
      const bounds = [activeXBounds[0], partBounds[1], activeXBounds[1], partBounds[3]];
      const loadAligned = document.getElementById('align_load_line').checked;
      const alignX = document.getElementById('honeycomb_align_x').checked;
      const alignY = document.getElementById('honeycomb_align_y').checked;
      const aligned = loadAligned || alignX || alignY;
      const angle = aligned ? 0 : Number(document.getElementById('orientation_angle_deg').value) * Math.PI / 180;
      if (!Number.isFinite(angle)) return null;
      if (loadAligned || alignX || alignY) {
        const loadCenter = [(partBounds[0] + partBounds[2]) * 0.5, (partBounds[1] + partBounds[3]) * 0.5];
        // In the preview's pointy-top hex lattice, a right vertical wall is
        // sqrt(3)/2 * a from its cell centre.  Put its midpoint at the part
        // centre so the visible guide and exported semantic request agree.
        const targetX = loadAligned ? loadCenter[0] : Number(document.getElementById('honeycomb_align_x_mm').value);
        const targetY = loadAligned ? loadCenter[1] : Number(document.getElementById('honeycomb_align_y_mm').value);
        const phaseSeed = [0.37, 0.23];
        const fallbackOrigin = [
          bounds[0] + Math.sqrt(3.0) * edgeLength * (phaseSeed[0] + 0.5 * phaseSeed[1]),
          bounds[1] + 1.5 * edgeLength * phaseSeed[1],
        ];
        const origin = [
          (loadAligned || alignX) ? targetX - Math.sqrt(3.0) * edgeLength * 0.5 : fallbackOrigin[0],
          // A left-to-right honeycomb course is an inclined-edge zigzag. Its
          // two vertex rows sit at centre-line ±a/4, so the underlying cell
          // centre must be 3a/4 below the requested course centre-line.
          (loadAligned || alignY) ? targetY - 0.75 * edgeLength : fallbackOrigin[1],
        ];
        const boundaryMode = document.getElementById('boundary_mode').value;
        return { edgeLength, wallWidth, angle, origin, bounds, partBounds, boundaryMode, aligned };
      }
      // The production pipeline chooses the final offset in the solved phase
      // domain.  This inexpensive canvas equivalent keeps the initial lattice
      // away from the rectangular axes as the requested cell size changes.
      const phaseSeed = [0.37, 0.23];
      const localOrigin = [
        Math.sqrt(3.0) * edgeLength * (phaseSeed[0] + 0.5 * phaseSeed[1]),
        1.5 * edgeLength * phaseSeed[1],
      ];
      const rotatedOrigin = rotateVector(localOrigin[0], localOrigin[1], angle);
      const origin = [bounds[0] + rotatedOrigin[0], bounds[1] + rotatedOrigin[1]];
      const boundaryMode = document.getElementById('boundary_mode').value;
      return { edgeLength, wallWidth, angle, origin, bounds, partBounds, boundaryMode, aligned };
    }

    function latticePreviewSegments() {
      const settings = latticePreviewParameters();
      if (!settings) return { segments: [], sampled: false, edgeLength: 0, wallWidth: 0 };
      const { edgeLength, wallWidth, angle, origin, bounds, partBounds, boundaryMode } = settings;
      const key = JSON.stringify({ edgeLength, wallWidth, angle, origin, bounds, partBounds, boundaryMode });
      if (latticePreviewCache?.key === key) return latticePreviewCache.value;
      const area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
      const exactCellEstimate = area / (1.5 * Math.sqrt(3.0) * edgeLength * edgeLength);
      const previewEdgeLength = exactCellEstimate > latticePreviewLimit
        ? edgeLength * Math.sqrt(exactCellEstimate / latticePreviewLimit)
        : edgeLength;
      const sampled = previewEdgeLength > edgeLength * (1 + 1e-9);
      const inverseAngle = -angle;
      const localCorners = [
        [bounds[0], bounds[1]], [bounds[0], bounds[3]], [bounds[2], bounds[1]], [bounds[2], bounds[3]],
      ].map(([x, y]) => rotateVector(x - origin[0], y - origin[1], inverseAngle));
      const localX = localCorners.map((point) => point[0]);
      const localY = localCorners.map((point) => point[1]);
      const radius = previewEdgeLength;
      const centerStepX = Math.sqrt(3.0) * radius;
      const centerStepY = 1.5 * radius;
      const jMin = Math.floor(Math.min(...localY) / centerStepY) - 3;
      const jMax = Math.ceil(Math.max(...localY) / centerStepY) + 3;
      const segments = [];
      const seen = new Set();
      for (let j = jMin; j <= jMax; j += 1) {
        const iMin = Math.floor(Math.min(...localX) / centerStepX - 0.5 * j) - 3;
        const iMax = Math.ceil(Math.max(...localX) / centerStepX - 0.5 * j) + 3;
        for (let i = iMin; i <= iMax; i += 1) {
          const localCenter = [centerStepX * (i + 0.5 * j), centerStepY * j];
          const rotatedCenter = rotateVector(localCenter[0], localCenter[1], angle);
          const center = [origin[0] + rotatedCenter[0], origin[1] + rotatedCenter[1]];
          const vertices = Array.from({ length: 6 }, (_, index) => {
            const vertex = rotateVector(radius * Math.cos(Math.PI / 6 + index * Math.PI / 3), radius * Math.sin(Math.PI / 6 + index * Math.PI / 3), angle);
            return [center[0] + vertex[0], center[1] + vertex[1]];
          });
          if (boundaryMode === 'inset' && !vertices.every(([x, y]) => x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3])) continue;
          vertices.forEach((start, index) => {
            const clipped = clipSegmentToBounds(start, vertices[(index + 1) % vertices.length], bounds);
            if (!clipped) return;
            const keyPart = (point) => `${point[0].toFixed(4)},${point[1].toFixed(4)}`;
            const first = keyPart(clipped[0]);
            const second = keyPart(clipped[1]);
            const edgeKey = first < second ? `${first}|${second}` : `${second}|${first}`;
            if (seen.has(edgeKey)) return;
            seen.add(edgeKey);
            segments.push(clipped);
          });
        }
      }
      const totalWallLengthMm = segments.reduce(
        (total, [start, end]) => total + Math.hypot(end[0] - start[0], end[1] - start[1]), 0
      );
      const value = { segments, sampled, edgeLength, previewEdgeLength, wallWidth, totalWallLengthMm, activeXBounds: [bounds[0], bounds[2]] };
      latticePreviewCache = { key, value };
      return value;
    }

    function continuousCoursePreview() {
      const fiberTowWidthMm = 2.0;
      const edgeLength = positiveNumber('base_cell_size_mm');
      const activeXBounds = honeycombActiveXBounds();
      if (!payload || edgeLength === null || activeXBounds === null) {
        return { courses: [], pores: [], edgeLength: 0, fiberTowWidthMm, totalLengthMm: 0, activeXBounds: [0, 0] };
      }
      const partBounds = payload.coordinate_system.xy_bounds_mm;
      const bounds = [activeXBounds[0], partBounds[1], activeXBounds[1], partBounds[3]];
      // Both the global rectangular perimeter and the two grip separators are
      // planned on their nominal resin-bead centre lines.  A pore may reach
      // only the material's *inner* edge, hence the half-bead inset here.
      const resinContourWidthMm = positiveNumber('wall_width_mm') ?? 2.0;
      const contourInnerInsetMm = resinContourWidthMm * 0.5;
      const poreClipBounds = [
        bounds[0] + contourInnerInsetMm,
        bounds[1] + contourInnerInsetMm,
        bounds[2] - contourInnerInsetMm,
        bounds[3] - contourInnerInsetMm,
      ];
      const key = JSON.stringify({ edgeLength, bounds, resinContourWidthMm });
      if (continuousCoursePreviewCache?.key === key) return continuousCoursePreviewCache.value;
      if (poreClipBounds[0] >= poreClipBounds[2] || poreClipBounds[1] >= poreClipBounds[3]) {
        return { courses: [], pores: [], poreClipBounds, resinContourWidthMm, edgeLength, fiberTowWidthMm, totalLengthMm: 0, activeXBounds };
      }

      // The yellow geometry is primary: it is a regular, two-phase pore
      // lattice.  Main rows have a 4 mm horizontal opening; the void between
      // adjacent main cells is occupied by a same-size interleaved pore.  Its
      // facing inclined sides are exactly 2 mm apart, so they carry one tow.
      const singleWallMm = fiberTowWidthMm;
      const doubleWallMm = fiberTowWidthMm * 2.0;
      const xMid = (bounds[0] + bounds[2]) * 0.5;
      const yMid = (bounds[1] + bounds[3]) * 0.5;
      // The 300 mm parent is only an oversized source for clipping.  Its
      // phase is translated for every cell size so that one *complete* pore
      // centre always coincides with the tensile working-region centre.  This
      // keeps cell size separate from pore/curvature/load-axis registration.
      const parentHalfSpanMm = 150.0;
      const parentBounds = [
        xMid - parentHalfSpanMm,
        yMid - parentHalfSpanMm,
        xMid + parentHalfSpanMm,
        yMid + parentHalfSpanMm,
      ];
      const parentXMid = xMid;
      const parentYMid = yMid;
      const hexHalfHeight = Math.sqrt(3.0) * edgeLength * 0.5;
      const rowPitch = hexHalfHeight * 2.0 + doubleWallMm;
      const referenceHexagon = (centerX, centerY) => [
        [centerX - edgeLength, centerY],
        [centerX - edgeLength * 0.5, centerY + hexHalfHeight],
        [centerX + edgeLength * 0.5, centerY + hexHalfHeight],
        [centerX + edgeLength, centerY],
        [centerX + edgeLength * 0.5, centerY - hexHalfHeight],
        [centerX - edgeLength * 0.5, centerY - hexHalfHeight],
      ];
      const signedArea = (start, end, point) => (
        (end[0] - start[0]) * (point[1] - start[1])
        - (end[1] - start[1]) * (point[0] - start[0])
      );
      // Build the parent as horizontal main rows.  Consecutive main rows
      // leave an exact 2w opening.  The half-row is filled with interleaved
      // pores, producing the single-width inclined channels requested by the
      // fibre topology.
      const sidePortInsetX = fiberTowWidthMm / (2.0 * Math.sqrt(3.0));
      // The interleaved centre lies halfway through a 2w horizontal opening.
      // The face-normal projection of the 2 mm diagonal channel is w, hence
      // its X phase is 1.5s + w/sqrt(3), not 1.5s + sqrt(3)w (the latter
      // would make the apparent diagonal opening 4 mm wide).
      const diagonalColumnOffset = edgeLength * 1.5 + fiberTowWidthMm / Math.sqrt(3.0);
      const diagonalRowOffset = hexHalfHeight + fiberTowWidthMm;
      // A main row and its interleaved row are separated by a single 2 mm
      // diagonal channel.  Doubling the X offset leaves a same-size hexagon
      // in every alternating void, including the one marked in the review.
      const columnPitch = diagonalColumnOffset * 2.0;
      const pores = [];
      const poreRecords = [];
      const columnMinimum = Math.ceil((parentBounds[0] - edgeLength - parentXMid) / columnPitch);
      const columnMaximum = Math.floor((parentBounds[2] + edgeLength - parentXMid) / columnPitch);
      const parentRowOriginY = parentYMid - rowPitch * 0.5;
      const parentRowMinimum = Math.ceil((parentBounds[1] - hexHalfHeight - parentRowOriginY) / rowPitch);
      const parentRowMaximum = Math.floor((parentBounds[3] + hexHalfHeight - parentRowOriginY) / rowPitch);
      for (let row = parentRowMinimum; row <= parentRowMaximum; row += 1) {
        const centerY = parentRowOriginY + row * rowPitch;
        for (let column = columnMinimum - 1; column <= columnMaximum + 1; column += 1) {
          // Every row uses the same X phase.  Thus the upper/lower yellow
          // horizontal sides of a pore column have identical endpoints: the
          // corresponding cells remain the same size instead of forming the
          // stagger-induced oversized voids visible in the previous preview.
          const centerX = parentXMid + column * columnPitch;
          const vertices = referenceHexagon(centerX, centerY);
          const record = { id: `${row}:${column}`, row, column, center: [centerX, centerY], vertices };
          poreRecords.push(record);
          pores.push(vertices);
        }
        // Fill every alternating void in the parent lattice.  Do not make a
        // boundary exception here: both yellow pores and fibre centre-lines
        // are generated on this full parent and clipped only afterwards.
        const interleavedCenterY = centerY + diagonalRowOffset;
        for (let column = columnMinimum - 1; column <= columnMaximum + 1; column += 1) {
          const centerX = parentXMid + column * columnPitch + diagonalColumnOffset;
          const vertices = referenceHexagon(centerX, interleavedCenterY);
          const record = {
            id: `${row + 0.5}:${column}`,
            row: row + 0.5,
            column,
            center: [centerX, interleavedCenterY],
            vertices,
          };
          poreRecords.push(record);
          pores.push(vertices);
        }
      }

      // Superseded experimental analytic-offset implementation follows in a
      // disabled block; it is retained only so this rollback stays local.
      // The red paths are derived directly from the yellow pores.  Offset a
      // pore edge by one half tow width: two pores separated by 4 mm leave two
      // distinct 2 mm centreline rails, whereas pores separated by 2 mm
      // produce the same (single) offset rail.  This is the uniform
      // mixed-wall construction; it deliberately contains no raster search.
      /* Prior free-space corridor search retained only for comparison during
         this replacement; it is intentionally not called.
      const courses = [];
      const fiberRadiusMm = fiberTowWidthMm * 0.5;
      const courseBounds = [
        activeXBounds[0],
        partBounds[1] + fiberRadiusMm,
        activeXBounds[1],
        partBounds[3] - fiberRadiusMm,
      ];
      const clipCourseSegment = (start, end, clipBounds) => {
        const dx = end[0] - start[0];
        const dy = end[1] - start[1];
        let lower = 0.0;
        let upper = 1.0;
        const tests = [
          [-dx, start[0] - clipBounds[0]], [dx, clipBounds[2] - start[0]],
          [-dy, start[1] - clipBounds[1]], [dy, clipBounds[3] - start[1]],
        ];
        for (const [p, q] of tests) {
          if (Math.abs(p) < 1e-12) {
            if (q < 0) return null;
            continue;
          }
          const ratio = q / p;
          if (p < 0) lower = Math.max(lower, ratio);
          else upper = Math.min(upper, ratio);
          if (lower > upper) return null;
        }
        return [
          [start[0] + lower * dx, start[1] + lower * dy],
          [start[0] + upper * dx, start[1] + upper * dy],
        ];
      };
      const cross = (first, second) => first[0] * second[1] - first[1] * second[0];
      const lineIntersection = (firstStart, firstEnd, secondStart, secondEnd) => {
        const firstVector = [firstEnd[0] - firstStart[0], firstEnd[1] - firstStart[1]];
        const secondVector = [secondEnd[0] - secondStart[0], secondEnd[1] - secondStart[1]];
        const denominator = cross(firstVector, secondVector);
        if (Math.abs(denominator) < 1e-9) return null;
        const delta = [secondStart[0] - firstStart[0], secondStart[1] - firstStart[1]];
        const t = cross(delta, secondVector) / denominator;
        return [firstStart[0] + firstVector[0] * t, firstStart[1] + firstVector[1] * t];
      };
      const outwardOffsetPolygon = (polygon, offsetMm) => {
        const shiftedEdges = polygon.map((start, index) => {
          const end = polygon[(index + 1) % polygon.length];
          const dx = end[0] - start[0];
          const dy = end[1] - start[1];
          const length = Math.hypot(dx, dy);
          // referenceHexagon is clockwise, hence its left normal is outward.
          const normal = [-dy / length, dx / length];
          return [
            [start[0] + normal[0] * offsetMm, start[1] + normal[1] * offsetMm],
            [end[0] + normal[0] * offsetMm, end[1] + normal[1] * offsetMm],
          ];
        });
        return shiftedEdges.map((edge, index) => (
          lineIntersection(shiftedEdges[(index + shiftedEdges.length - 1) % shiftedEdges.length][0], shiftedEdges[(index + shiftedEdges.length - 1) % shiftedEdges.length][1], edge[0], edge[1]) || edge[0]
        ));
      };
      const pointKey = (point) => `${point[0].toFixed(6)},${point[1].toFixed(6)}`;
      const edgeKey = (start, end) => {
        const first = pointKey(start);
        const second = pointKey(end);
        return first < second ? `${first}|${second}` : `${second}|${first}`;
      };
      const nodes = new Map();
      const edges = [];
      const uniqueEdges = new Set();
      const addNode = (point) => {
        const key = pointKey(point);
        if (!nodes.has(key)) nodes.set(key, { point, edgeIds: [] });
        return key;
      };
      pores.forEach((pore) => {
        const offsetPore = outwardOffsetPolygon(pore, fiberRadiusMm);
        offsetPore.forEach((start, index) => {
          const clipped = clipCourseSegment(start, offsetPore[(index + 1) % offsetPore.length], courseBounds);
          if (!clipped || Math.hypot(clipped[1][0] - clipped[0][0], clipped[1][1] - clipped[0][1]) < 1e-7) return;
          const key = edgeKey(clipped[0], clipped[1]);
          if (uniqueEdges.has(key)) return;
          uniqueEdges.add(key);
          const startKey = addNode(clipped[0]);
          const endKey = addNode(clipped[1]);
          const id = edges.length;
          edges.push({ startKey, endKey });
          nodes.get(startKey).edgeIds.push(id);
          nodes.get(endKey).edgeIds.push(id);
        });
      });
      const otherKey = (edge, nodeKey) => edge.startKey === nodeKey ? edge.endKey : edge.startKey;
      const simplifyExactCourse = (points) => points.filter((point, index) => {
        if (index === 0 || index === points.length - 1) return true;
        const previous = points[index - 1];
        const next = points[index + 1];
        return Math.abs(signedArea(previous, point, next)) > 1e-7;
      });
      const unusedEdges = new Set(edges.map((_, index) => index));
      const traceForwardCourse = (startKey, firstEdge) => {
        const points = [nodes.get(startKey).point];
        let currentKey = startKey;
        let edgeId = firstEdge;
        const usedByCourse = [];
        while (edgeId !== undefined && !usedByCourse.includes(edgeId)) {
          usedByCourse.push(edgeId);
          const nextKey = otherKey(edges[edgeId], currentKey);
          points.push(nodes.get(nextKey).point);
          currentKey = nextKey;
          if (reachesRight(nodes.get(currentKey).point)) break;
          const current = nodes.get(currentKey).point;
          const candidates = nodes.get(currentKey).edgeIds
            .filter((candidate) => candidate !== edgeId && unusedEdges.has(candidate) && !usedByCourse.includes(candidate))
            .map((candidate) => {
              const next = nodes.get(otherKey(edges[candidate], currentKey)).point;
              const dx = next[0] - current[0];
              const dy = next[1] - current[1];
              return { candidate, dx, directionX: dx / Math.hypot(dx, dy) };
            })
            .filter(({ dx }) => dx > 1e-7)
            .sort((first, second) => second.directionX - first.directionX || second.dx - first.dx);
          edgeId = candidates[0]?.candidate;
        }
        return { points: simplifyExactCourse(points), edgeIds: usedByCourse };
      };
      const reachesLeft = (point) => Math.abs(point[0] - activeXBounds[0]) < 1e-6;
      const reachesRight = (point) => Math.abs(point[0] - activeXBounds[1]) < 1e-6;
      const addThroughGripCourse = (corePoints) => {
        if (corePoints.length < 2) return;
        let points = corePoints;
        if (reachesRight(points[0]) && reachesLeft(points[points.length - 1])) points = points.slice().reverse();
        if (!reachesLeft(points[0]) || !reachesRight(points[points.length - 1])) return;
        courses.push({
          points: [[partBounds[0], points[0][1]], ...points, [partBounds[2], points[points.length - 1][1]]],
          railOffsetsMm: [0],
        });
      };
      // Begin at clipped left ports and follow only the next analytic segment
      // with a positive X component.  At a mixed-wall junction this pairs
      // each rail with its forward diagonal continuation, so distinct fibres
      // never merge.  A completed route then claims its own segments.
      [...nodes.entries()]
        .filter(([nodeKey]) => reachesLeft(nodes.get(nodeKey).point))
        .sort(([, first], [, second]) => first.point[1] - second.point[1])
        .forEach(([nodeKey, node]) => {
          node.edgeIds.forEach((edgeId) => {
            if (!unusedEdges.has(edgeId)) return;
            const trace = traceForwardCourse(nodeKey, edgeId);
            if (!reachesRight(trace.points.at(-1))) return;
            trace.edgeIds.forEach((usedEdgeId) => unusedEdges.delete(usedEdgeId));
            addThroughGripCourse(trace.points);
          });
        });
      */
      // Roll back to the prior corridor-course preview.  Its source points
      // are sampled in the actual pore gaps, then simplified to long courses;
      // it retains all viable X-through channels instead of discarding them
      // while trying to resolve mixed-wall graph branches.
      /*
      const courses = [];
      const extendThroughGripRegions = (corePoints) => {
        if (corePoints.length < 2) return [];
        const first = corePoints[0];
        const last = corePoints[corePoints.length - 1];
        return [[partBounds[0], first[1]], ...corePoints, [partBounds[2], last[1]]];
      };
      const fiberRadiusMm = fiberTowWidthMm * 0.5;
      const routeClearanceMm = fiberRadiusMm - 1e-6;
      const gridStepMm = 0.5;
      const routePores = pores.filter((pore) => {
        const xs = pore.map((point) => point[0]);
        const ys = pore.map((point) => point[1]);
        return Math.max(...xs) >= activeXBounds[0] - fiberRadiusMm
          && Math.min(...xs) <= activeXBounds[1] + fiberRadiusMm
          && Math.max(...ys) >= poreClipBounds[1] - fiberRadiusMm
          && Math.min(...ys) <= poreClipBounds[3] + fiberRadiusMm;
      });
      const pointToSegmentDistance = (point, start, end) => {
        const dx = end[0] - start[0];
        const dy = end[1] - start[1];
        const lengthSquared = dx * dx + dy * dy;
        if (lengthSquared <= 1e-12) return Math.hypot(point[0] - start[0], point[1] - start[1]);
        const t = Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared));
        return Math.hypot(point[0] - (start[0] + dx * t), point[1] - (start[1] + dy * t));
      };
      const pointOnSegment = (point, start, end) => {
        if (Math.abs(signedArea(start, end, point)) > 1e-9) return false;
        return point[0] >= Math.min(start[0], end[0]) - 1e-9
          && point[0] <= Math.max(start[0], end[0]) + 1e-9
          && point[1] >= Math.min(start[1], end[1]) - 1e-9
          && point[1] <= Math.max(start[1], end[1]) + 1e-9;
      };
      const pointInsidePolygon = (point, polygon) => {
        let inside = false;
        polygon.forEach((start, index) => {
          const end = polygon[(index + 1) % polygon.length];
          if (pointOnSegment(point, start, end)) inside = true;
          if ((start[1] > point[1]) !== (end[1] > point[1])) {
            const crossingX = (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0];
            if (point[0] < crossingX) inside = !inside;
          }
        });
        return inside;
      };
      const poreClearance = (point) => {
        let clearance = Number.POSITIVE_INFINITY;
        for (const pore of routePores) {
          if (pointInsidePolygon(point, pore)) return -Number.POSITIVE_INFINITY;
          pore.forEach((start, index) => {
            clearance = Math.min(clearance, pointToSegmentDistance(point, start, pore[(index + 1) % pore.length]));
          });
        }
        return clearance;
      };
      const segmentsIntersect = (firstStart, firstEnd, secondStart, secondEnd) => {
        const firstSecondStart = signedArea(firstStart, firstEnd, secondStart);
        const firstSecondEnd = signedArea(firstStart, firstEnd, secondEnd);
        const secondFirstStart = signedArea(secondStart, secondEnd, firstStart);
        const secondFirstEnd = signedArea(secondStart, secondEnd, firstEnd);
        if ((firstSecondStart > 1e-9 && firstSecondEnd < -1e-9 || firstSecondStart < -1e-9 && firstSecondEnd > 1e-9)
          && (secondFirstStart > 1e-9 && secondFirstEnd < -1e-9 || secondFirstStart < -1e-9 && secondFirstEnd > 1e-9)) return true;
        return pointOnSegment(secondStart, firstStart, firstEnd)
          || pointOnSegment(secondEnd, firstStart, firstEnd)
          || pointOnSegment(firstStart, secondStart, secondEnd)
          || pointOnSegment(firstEnd, secondStart, secondEnd);
      };
      const segmentClearance = (start, end) => {
        let clearance = Number.POSITIVE_INFINITY;
        for (const pore of routePores) {
          if (pointInsidePolygon(start, pore) || pointInsidePolygon(end, pore)) return -Number.POSITIVE_INFINITY;
          pore.forEach((edgeStart, index) => {
            const edgeEnd = pore[(index + 1) % pore.length];
            if (segmentsIntersect(start, end, edgeStart, edgeEnd)) {
              clearance = 0.0;
              return;
            }
            clearance = Math.min(
              clearance,
              pointToSegmentDistance(start, edgeStart, edgeEnd),
              pointToSegmentDistance(end, edgeStart, edgeEnd),
              pointToSegmentDistance(edgeStart, start, end),
              pointToSegmentDistance(edgeEnd, start, end),
            );
          });
          if (clearance < routeClearanceMm) return clearance;
        }
        return clearance;
      };
      const simplifyCorridorCourse = (points) => {
        if (points.length < 3) return points;
        const simplified = [points[0]];
        let current = 0;
        while (current < points.length - 1) {
          let next = points.length - 1;
          while (next > current + 1 && segmentClearance(points[current], points[next]) < routeClearanceMm) next -= 1;
          simplified.push(points[next]);
          current = next;
        }
        return simplified;
      };
      const xCount = Math.max(2, Math.round((activeXBounds[1] - activeXBounds[0]) / gridStepMm) + 1);
      const yCount = Math.max(2, Math.round((poreClipBounds[3] - poreClipBounds[1]) / gridStepMm) + 1);
      const xCoordinates = Array.from({ length: xCount }, (_, index) => (
        activeXBounds[0] + (activeXBounds[1] - activeXBounds[0]) * index / (xCount - 1)
      ));
      const yCoordinates = Array.from({ length: yCount }, (_, index) => (
        poreClipBounds[1] + (poreClipBounds[3] - poreClipBounds[1]) * index / (yCount - 1)
      ));
      const clearanceGrid = xCoordinates.map((xCoordinate) => yCoordinates.map((yCoordinate) => poreClearance([xCoordinate, yCoordinate])));
      const middleXIndex = xCoordinates.reduce((bestIndex, xCoordinate, index) => (
        Math.abs(xCoordinate - xMid) < Math.abs(xCoordinates[bestIndex] - xMid) ? index : bestIndex
      ), 0);
      const freeAtMiddle = clearanceGrid[middleXIndex].map((clearance) => clearance >= routeClearanceMm);
      const seedRows = [];
      for (let first = 0; first < yCount;) {
        if (!freeAtMiddle[first]) { first += 1; continue; }
        let last = first;
        while (last + 1 < yCount && freeAtMiddle[last + 1]) last += 1;
        if (yCoordinates[last] - yCoordinates[first] >= fiberTowWidthMm - 1e-9) {
          seedRows.push(first, last);
        } else {
          seedRows.push(Math.round((first + last) * 0.5));
        }
        first = last + 1;
      }
      const traceCorridorToSide = (seedRow, direction) => {
        const xIndices = [];
        for (let index = middleXIndex; index >= 0 && index < xCount; index += direction) xIndices.push(index);
        let previousCosts = new Float64Array(yCount);
        previousCosts.fill(Number.POSITIVE_INFINITY);
        previousCosts[seedRow] = 0.0;
        const parents = [];
        for (let step = 1; step < xIndices.length; step += 1) {
          const currentCosts = new Float64Array(yCount);
          currentCosts.fill(Number.POSITIVE_INFINITY);
          const parent = new Int16Array(yCount);
          parent.fill(-1);
          for (let row = 0; row < yCount; row += 1) {
            const clearance = clearanceGrid[xIndices[step]][row];
            if (clearance < routeClearanceMm) continue;
            for (let previousRow = Math.max(0, row - 3); previousRow <= Math.min(yCount - 1, row + 3); previousRow += 1) {
              if (!Number.isFinite(previousCosts[previousRow])) continue;
              const rowChange = row - previousRow;
              const centreBias = (yCoordinates[row] - yCoordinates[seedRow]) / fiberTowWidthMm;
              const cost = previousCosts[previousRow] + rowChange * rowChange * 0.09 + centreBias * centreBias * 0.002 - Math.min(clearance, fiberTowWidthMm * 2.0) * 0.02;
              if (cost < currentCosts[row]) { currentCosts[row] = cost; parent[row] = previousRow; }
            }
          }
          if (![...currentCosts].some(Number.isFinite)) return null;
          parents.push(parent);
          previousCosts = currentCosts;
        }
        let finalRow = 0;
        for (let row = 1; row < yCount; row += 1) if (previousCosts[row] < previousCosts[finalRow]) finalRow = row;
        if (!Number.isFinite(previousCosts[finalRow])) return null;
        const rows = Array(xIndices.length);
        rows[rows.length - 1] = finalRow;
        for (let step = parents.length - 1; step >= 0; step -= 1) rows[step] = parents[step][rows[step + 1]];
        return xIndices.map((xIndex, step) => [xCoordinates[xIndex], yCoordinates[rows[step]]]);
      };
      [...new Set(seedRows)].filter((row) => clearanceGrid[middleXIndex][row] >= routeClearanceMm).forEach((seedRow) => {
        const left = traceCorridorToSide(seedRow, -1);
        const right = traceCorridorToSide(seedRow, 1);
        if (!left || !right) return;
        const points = extendThroughGripRegions(simplifyCorridorCourse([...left.slice().reverse(), ...right.slice(1)]));
        if (points.length >= 2) courses.push({ points, railOffsetsMm: [0] });
      });
      */
      // Build the requested paths directly from the yellow parent lattice.
      // Each 4 mm horizontal opening starts two lanes.  Those lanes travel on
      // opposite sides of the interleaved pore in that opening: each inclined
      // 2 mm channel therefore has capacity one, and no route search or
      // nearest-edge snapping is involved.
      const courses = [];
      const fiberRadiusMm = fiberTowWidthMm * 0.5;
      const appendDistinctPoint = (points, point) => {
        const previous = points.at(-1);
        if (!previous || Math.hypot(point[0] - previous[0], point[1] - previous[1]) > 1e-7) points.push(point);
      };
      const simplifyMonotoneCourse = (points) => points.filter((point, index) => {
        if (index === 0 || index === points.length - 1) return true;
        const previous = points[index - 1];
        const next = points[index + 1];
        return Math.abs(signedArea(previous, point, next)) > 1e-7;
      });
      // Build the directed
      // opening topology.  Each complete 2w horizontal opening owns exactly
      // two lanes: the upper offset of its lower pore row and the lower offset
      // of its upper pore row.  Their corners are intersections of the
      // analytic side-offset supports, not sampled or snapped points.
      const recordsByRow = new Map();
      poreRecords.forEach((record) => {
        if (!recordsByRow.has(record.row)) recordsByRow.set(record.row, []);
        recordsByRow.get(record.row).push(record);
      });
      recordsByRow.forEach((records) => records.sort((first, second) => first.center[0] - second.center[0]));
      const offsetHalfPore = (record, side) => {
        const [centerX, centerY] = record.center;
        const verticalSign = side === 'upper' ? 1.0 : -1.0;
        const portY = centerY + verticalSign * fiberRadiusMm;
        const flatY = centerY + verticalSign * (hexHalfHeight + fiberRadiusMm);
        return [
          [centerX - edgeLength - sidePortInsetX, portY],
          [centerX - edgeLength * 0.5 - sidePortInsetX, flatY],
          [centerX + edgeLength * 0.5 + sidePortInsetX, flatY],
          [centerX + edgeLength + sidePortInsetX, portY],
        ];
      };
      const analyticRowCourse = (row, side) => {
        const records = recordsByRow.get(row) ?? [];
        const sidePaths = records.map((record) => offsetHalfPore(record, side));
        if (!sidePaths.length) return [];
        // Keep the complete parent track.  The renderer clips each resulting
        // segment to the real rectangular part, so no boundary repair segment
        // is invented after a pore has been cut.
        const sourcePoints = [sidePaths[0][0]];
        sidePaths.forEach((path) => {
          appendDistinctPoint(sourcePoints, path[0]);
          appendDistinctPoint(sourcePoints, path[1]);
          appendDistinctPoint(sourcePoints, path[2]);
          appendDistinctPoint(sourcePoints, path[3]);
        });
        return simplifyMonotoneCourse(sourcePoints);
      };
      // The parent route remains one analytical line for design purposes, but
      // every rectangle-clipped run is one manufacturing path.  Keeping this
      // split here makes the designer use the same endpoint semantics as the
      // Core SourceJob: no invisible connector is invented across a cut.
      const clippedCourseFragments = (points) => {
        const clipSegment = (start, end) => {
          const dx = end[0] - start[0];
          const dy = end[1] - start[1];
          let lower = 0.0;
          let upper = 1.0;
          for (const [p, q] of [
            [-dx, start[0] - bounds[0]], [dx, bounds[2] - start[0]],
            [-dy, start[1] - bounds[1]], [dy, bounds[3] - start[1]],
          ]) {
            if (Math.abs(p) < 1e-12) {
              if (q < 0) return null;
              continue;
            }
            const ratio = q / p;
            if (p < 0) lower = Math.max(lower, ratio);
            else upper = Math.min(upper, ratio);
            if (lower > upper) return null;
          }
          return [[start[0] + lower * dx, start[1] + lower * dy], [start[0] + upper * dx, start[1] + upper * dy]];
        };
        const fragments = [];
        let fragment = [];
        points.slice(1).forEach((end, index) => {
          const segment = clipSegment(points[index], end);
          if (!segment) {
            if (fragment.length > 1) fragments.push(fragment);
            fragment = [];
            return;
          }
          if (!fragment.length || Math.hypot(fragment.at(-1)[0] - segment[0][0], fragment.at(-1)[1] - segment[0][1]) > 1e-7) {
            if (fragment.length > 1) fragments.push(fragment);
            fragment = [segment[0]];
          }
          if (Math.hypot(fragment.at(-1)[0] - segment[1][0], fragment.at(-1)[1] - segment[1][1]) > 1e-7) fragment.push(segment[1]);
        });
        if (fragment.length > 1) fragments.push(fragment);
        // A corner-only diagonal tip is not a usable pore-to-pore channel.
        // A printable boundary fragment must retain a finite horizontal
        // support.  This is intentionally the same filter as Core's
        // continuous_course planner, so the displayed red paths are exactly
        // the independent manufacturing paths handed to Core.
        return fragments.filter((fragment) => fragment.slice(1).some((point, index) => (
          Math.abs(point[1] - fragment[index][1]) <= 1e-7
          && point[0] - fragment[index][0] > 1e-7
        )));
      };
      const mainRows = [...recordsByRow.keys()]
        .filter((row) => Math.abs(row - Math.round(row)) < 1e-7)
        .sort((first, second) => first - second);
      mainRows.slice(1).forEach((upperRow, index) => {
        const lowerRow = mainRows[index];
        const lowerCenterY = recordsByRow.get(lowerRow)[0].center[1];
        const upperCenterY = recordsByRow.get(upperRow)[0].center[1];
        const gapLowerY = lowerCenterY + hexHalfHeight;
        const gapUpperY = upperCenterY - hexHalfHeight;
        if (gapLowerY < poreClipBounds[1] - 1e-7 || gapUpperY > poreClipBounds[3] + 1e-7) return;
        const lowerPoints = analyticRowCourse(lowerRow + 0.5, 'lower');
        const upperPoints = analyticRowCourse(lowerRow + 0.5, 'upper');
        courses.push({
          points: lowerPoints,
          fragments: clippedCourseFragments(lowerPoints),
          railOffsetsMm: [0],
          opening: { lowerRow, upperRow, lane: 'lower', capacity: 2 },
        });
        courses.push({
          points: upperPoints,
          fragments: clippedCourseFragments(upperPoints),
          railOffsetsMm: [0],
          opening: { lowerRow, upperRow, lane: 'upper', capacity: 2 },
        });
      });
      const courseLength = (course) => course.fragments.flatMap((fragment) => fragment.slice(1).map((point, index) => [fragment[index], point])).reduce(
        (length, [start, end]) => length + Math.hypot(end[0] - start[0], end[1] - start[1]), 0
      );
      const totalLengthMm = courses.reduce((total, course) => total + courseLength(course) * course.railOffsetsMm.length, 0);
      const value = {
        courses,
        pores,
        poreClipBounds,
        resinContourWidthMm,
        fiberTowWidthMm,
        edgeLength,
        totalLengthMm,
        activeXBounds,
        courseClipBounds: bounds,
        latticeAnchorMm: [xMid, yMid],
        parentBounds,
        parentRowOriginY,
        rowPitch,
        columnPitch,
      };
      continuousCoursePreviewCache = { key, value };
      return value;
    }

    function offsetContinuousCourse(course, offsetMm) {
      if (course.length < 2 || Math.abs(offsetMm) < 1e-9) return course.map((point) => [...point]);
      const unitNormal = (start, end) => {
        const dx = end[0] - start[0];
        const dy = end[1] - start[1];
        const length = Math.hypot(dx, dy);
        return length <= 1e-9 ? [0, 0] : [-dy / length, dx / length];
      };
      const result = [];
      course.forEach((point, index) => {
        const previousNormal = index === 0
          ? unitNormal(course[0], course[1])
          : unitNormal(course[index - 1], point);
        const nextNormal = index === course.length - 1
          ? previousNormal
          : unitNormal(point, course[index + 1]);
        const miterX = previousNormal[0] + nextNormal[0];
        const miterY = previousNormal[1] + nextNormal[1];
        const miterLength = Math.hypot(miterX, miterY);
        if (miterLength <= 1e-8) {
          // A reference-course U-turn has no finite miter.  Keep its two
          // sharp rail endpoints explicitly rather than inventing an arc.
          result.push([point[0] + previousNormal[0] * offsetMm, point[1] + previousNormal[1] * offsetMm]);
          result.push([point[0] + nextNormal[0] * offsetMm, point[1] + nextNormal[1] * offsetMm]);
          return;
        }
        const unitMiter = [miterX / miterLength, miterY / miterLength];
        const denominator = Math.abs(unitMiter[0] * nextNormal[0] + unitMiter[1] * nextNormal[1]);
        const distance = Math.min(Math.abs(offsetMm) * 4, Math.abs(offsetMm) / Math.max(denominator, 1e-6));
        const direction = offsetMm < 0 ? -1 : 1;
        result.push([point[0] + unitMiter[0] * distance * direction, point[1] + unitMiter[1] * distance * direction]);
      });
      return result;
    }

    function appendProjectedContinuousCourse(ctx, course, layer, zMid, yaw, pitch, scale, cx, cy) {
      course.forEach((point, index) => {
        const projected = project(point[0], point[1], physicalLayerZ(heightAt(point[0], point[1]), layer) - zMid, yaw, pitch, scale, cx, cy);
        if (index === 0) ctx.moveTo(projected.x, projected.y);
        else ctx.lineTo(projected.x, projected.y);
      });
    }

    function appendProjectedClippedContinuousCourse(ctx, course, clipBounds, layer, zMid, yaw, pitch, scale, cx, cy) {
      course.slice(1).forEach((end, index) => {
        const segment = clipSegmentToBounds(course[index], end, clipBounds);
        if (!segment) return;
        const projectedStart = project(segment[0][0], segment[0][1], physicalLayerZ(heightAt(segment[0][0], segment[0][1]), layer) - zMid, yaw, pitch, scale, cx, cy);
        const projectedEnd = project(segment[1][0], segment[1][1], physicalLayerZ(heightAt(segment[1][0], segment[1][1]), layer) - zMid, yaw, pitch, scale, cx, cy);
        ctx.moveTo(projectedStart.x, projectedStart.y);
        ctx.lineTo(projectedEnd.x, projectedEnd.y);
      });
    }

    function appendProjectedClosedPore(ctx, pore, layer, zMid, yaw, pitch, scale, cx, cy) {
      pore.forEach((point, index) => {
        const projected = project(point[0], point[1], physicalLayerZ(heightAt(point[0], point[1]), layer) - zMid, yaw, pitch, scale, cx, cy);
        if (index === 0) ctx.moveTo(projected.x, projected.y);
        else ctx.lineTo(projected.x, projected.y);
      });
      ctx.closePath();
    }

    function appendProjectedClippedPore(ctx, pore, clipBounds, layer, zMid, yaw, pitch, scale, cx, cy) {
      pore.forEach((start, index) => {
        const end = pore[(index + 1) % pore.length];
        const segment = clipSegmentToBounds(start, end, clipBounds);
        if (!segment) return;
        const projectedStart = project(segment[0][0], segment[0][1], physicalLayerZ(heightAt(segment[0][0], segment[0][1]), layer) - zMid, yaw, pitch, scale, cx, cy);
        const projectedEnd = project(segment[1][0], segment[1][1], physicalLayerZ(heightAt(segment[1][0], segment[1][1]), layer) - zMid, yaw, pitch, scale, cx, cy);
        ctx.moveTo(projectedStart.x, projectedStart.y);
        ctx.lineTo(projectedEnd.x, projectedEnd.y);
      });
    }

    function updateLatticeLengthSummary() {
      const summary = document.getElementById('latticeLengthSummary');
      if (!payload) {
        summary.textContent = '连续路径总长将在曲面预览更新后显示。';
        return;
      }
      const courses = continuousCoursePreview();
      if (!courses.courses.length) {
        summary.textContent = '当前尺寸与目标边长不能容纳完整的连续路径单元。请减小目标边长或增大工作段。';
      } else {
        const fragmentCount = courses.courses.reduce((count, course) => count + course.fragments.length, 0);
        summary.textContent = `当前平面连续路径：中心孔锚定在 (${courses.latticeAnchorMm[0].toFixed(2)}, ${courses.latticeAnchorMm[1].toFixed(2)}) mm；黄色孔洞由该锚点的 300 × 300 mm 母板裁切，并在外矩形轮廓和夹持分界树脂带内侧截断（树脂轮廓宽 ${courses.resinContourWidthMm.toFixed(2)} mm）。${courses.courses.length} 条基础长路径裁为 ${fragmentCount} 条独立制造路径；每个截断首尾均按独立树脂/纤维路径处理。预计纤维总长 ${courses.totalLengthMm.toFixed(2)} mm。`;
      }
    }

    function project(x, y, z, yaw, pitch, scale, cx, cy) {
      const bounds = payload.coordinate_system.xy_bounds_mm;
      const centeredX = x - (bounds[0] + bounds[2]) * 0.5;
      const centeredY = y - (bounds[1] + bounds[3]) * 0.5;
      const displayZ = z * Number(surfaceZScale.value);
      const xr = centeredX * Math.cos(yaw) - centeredY * Math.sin(yaw);
      const yr = centeredX * Math.sin(yaw) + centeredY * Math.cos(yaw);
      // Canvas Y grows downwards.  This conventional pitch rotation keeps a
      // positive physical Z axis visually upwards.
      const yp = yr * Math.cos(pitch) + displayZ * Math.sin(pitch);
      const depth = -yr * Math.sin(pitch) + displayZ * Math.cos(pitch);
      return { x: cx + xr * scale, y: cy - yp * scale, depth };
    }

    function heightAt(x, y) {
      const surface = payload.surface;
      return surface.z_reference_mm + surface.amplitude_mm
        * Math.sin((2 * Math.PI * x) / surface.wavelength_x_mm + surface.phase_x_rad)
        * Math.sin((2 * Math.PI * y) / surface.wavelength_y_mm + surface.phase_y_rad);
    }

    function physicalPreviewLayer() {
      const stack = payload.solid_stack;
      if (!stack) return { index: 0, alpha: 1, base_z_mm: 0 };
      const index = Number.isInteger(stack.representative_peak_layer_index)
        ? stack.representative_peak_layer_index
        : stack.peak_layer_indices[0];
      return stack.layers[index];
    }

    function physicalLayerZ(targetZ, layer) {
      return layer.base_z_mm + payload.surface.z_reference_mm
        + layer.alpha * (targetZ - payload.surface.z_reference_mm);
    }

    function appendProjectedRing(ctx, ring, layer, zMid, yaw, pitch, scale, cx, cy) {
      if (ring.length < 2) return;
      const first = project(ring[0][0], ring[0][1], physicalLayerZ(heightAt(ring[0][0], ring[0][1]), layer) - zMid, yaw, pitch, scale, cx, cy);
      ctx.moveTo(first.x, first.y);
      ring.slice(1).forEach(([x, y]) => {
        const point = project(x, y, physicalLayerZ(heightAt(x, y), layer) - zMid, yaw, pitch, scale, cx, cy);
        ctx.lineTo(point.x, point.y);
      });
      ctx.closePath();
    }

    function clipToProjection(ctx, projection, layer, zMid, yaw, pitch, scale, cx, cy) {
      ctx.beginPath();
      projection.polygons.forEach((polygon) => {
        appendProjectedRing(ctx, polygon.outer, layer, zMid, yaw, pitch, scale, cx, cy);
        polygon.holes.forEach((ring) => appendProjectedRing(ctx, ring, layer, zMid, yaw, pitch, scale, cx, cy));
      });
      ctx.clip('evenodd');
    }

    function drawProjectionBoundaries(ctx, projection, layer, zMid, yaw, pitch, scale, cx, cy) {
      if (!projection) return;
      ctx.strokeStyle = 'rgba(12, 44, 82, .94)';
      ctx.lineWidth = 1.2;
      projection.polygons.forEach((polygon) => {
        const rings = drag ? [polygon.outer] : [polygon.outer, ...polygon.holes];
        rings.forEach((ring) => {
          ctx.beginPath();
          appendProjectedRing(ctx, ring, layer, zMid, yaw, pitch, scale, cx, cy);
          ctx.stroke();
        });
      });
    }

    function drawInspectionMarker(ctx, layer, zMid, yaw, pitch, scale, cx, cy) {
      const point = payload.inspection_point;
      if (!point) return;
      const projected = project(point.x_mm, point.y_mm, physicalLayerZ(point.height_mm, layer) - zMid, yaw, pitch, scale, cx, cy);
      ctx.beginPath();
      ctx.arc(projected.x, projected.y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#d14322';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.fillStyle = '#7a271a';
      ctx.font = '12px Segoe UI, Microsoft YaHei, sans-serif';
      ctx.fillText(`检验点 (${point.x_mm.toFixed(1)}, ${point.y_mm.toFixed(1)})`, projected.x + 7, projected.y - 7);
    }

    function drawSurfaceReferenceFrame(ctx, zMid, yaw, pitch, scale, cx, cy) {
      const bounds = payload.coordinate_system.xy_bounds_mm;
      const baseZ = 0;
      const corners = [
        [bounds[0], bounds[1]], [bounds[2], bounds[1]], [bounds[2], bounds[3]], [bounds[0], bounds[3]],
      ].map(([x, y]) => project(x, y, baseZ - zMid, yaw, pitch, scale, cx, cy));
      ctx.save();
      ctx.fillStyle = 'rgba(216, 230, 239, .48)';
      ctx.strokeStyle = 'rgba(112, 142, 159, .65)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y);
      corners.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      const origin = project(bounds[0], bounds[1], baseZ - zMid, yaw, pitch, scale, cx, cy);
      const axisLength = Math.min(30, Math.max(10, Math.min(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.22));
      const axes = [
        { end: project(bounds[0] + axisLength, bounds[1], baseZ - zMid, yaw, pitch, scale, cx, cy), color: '#b91c1c', label: 'X' },
        { end: project(bounds[0], bounds[1] + axisLength, baseZ - zMid, yaw, pitch, scale, cx, cy), color: '#0f766e', label: 'Y' },
        { end: project(bounds[0], bounds[1], baseZ + axisLength - zMid, yaw, pitch, scale, cx, cy), color: '#1d4ed8', label: 'Z' },
      ];
      ctx.setLineDash([]);
      ctx.font = '600 11px Segoe UI, Microsoft YaHei, sans-serif';
      axes.forEach((axis) => {
        ctx.strokeStyle = axis.color;
        ctx.fillStyle = axis.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(origin.x, origin.y);
        ctx.lineTo(axis.end.x, axis.end.y);
        ctx.stroke();
        ctx.fillText(axis.label, axis.end.x + 6, axis.end.y - 4);
      });
      ctx.restore();
    }

    function drawSurfaceGuideMesh(ctx, x, y, z, layer, zMid, yaw, pitch, scale, cx, cy) {
      const rowStride = Math.max(1, Math.floor((z.length - 1) / 8));
      const colStride = Math.max(1, Math.floor((z[0].length - 1) / 8));
      const centerRow = Math.floor((z.length - 1) * 0.5);
      const centerCol = Math.floor((z[0].length - 1) * 0.5);
      const drawCurve = (points, emphasis) => {
        ctx.beginPath();
        points.forEach(([xMm, yMm, zMm], index) => {
          const point = project(xMm, yMm, physicalLayerZ(zMm, layer) - zMid, yaw, pitch, scale, cx, cy);
          if (index === 0) ctx.moveTo(point.x, point.y);
          else ctx.lineTo(point.x, point.y);
        });
        ctx.strokeStyle = emphasis ? 'rgba(27, 73, 105, .75)' : 'rgba(44, 103, 139, .3)';
        ctx.lineWidth = emphasis ? 1.05 : 0.55;
        ctx.stroke();
      };
      ctx.save();
      for (let row = 0; row < z.length; row += rowStride) {
        drawCurve(x[row].map((xMm, col) => [xMm, y[row][col], z[row][col]]), row === centerRow);
      }
      if ((z.length - 1) % rowStride !== 0) drawCurve(x[z.length - 1].map((xMm, col) => [xMm, y[z.length - 1][col], z[z.length - 1][col]]), false);
      for (let col = 0; col < z[0].length; col += colStride) {
        drawCurve(z.map((row, rowIndex) => [x[rowIndex][col], y[rowIndex][col], row[col]]), col === centerCol);
      }
      if ((z[0].length - 1) % colStride !== 0) drawCurve(z.map((row, rowIndex) => [x[rowIndex][z[0].length - 1], y[rowIndex][z[0].length - 1], row[z[0].length - 1]]), false);
      ctx.restore();
    }

    function drawLatticePreview(ctx, layer, zMid, yaw, pitch, scale, cx, cy) {
      const coursePreview = continuousCoursePreview();
      if (!coursePreview.pores.length && !coursePreview.courses.length) return;
      ctx.save();
      // Yellow is the clear-pore skeleton.  It is a geometric reference only,
      // not an added printable path.  Its source is the fixed 300 x 300 mm
      // parent lattice; only its individual line segments are clipped here.
      ctx.beginPath();
      coursePreview.pores.forEach((pore) => appendProjectedClippedPore(ctx, pore, coursePreview.poreClipBounds, layer, zMid, yaw, pitch, scale, cx, cy));
      ctx.strokeStyle = 'rgba(255, 191, 0, .98)';
      ctx.lineWidth = Math.max(1.0, Math.min(2.4, scale * 0.18));
      ctx.lineJoin = 'miter';
      ctx.lineCap = 'butt';
      ctx.stroke();
      // Red is a centreline-only view.  Single-track and double-track bands
      // are explicit properties of each course, never a blanket offset of an
      // old honeycomb skeleton.  The double band uses centres +/-1 mm.
      coursePreview.courses.forEach((course) => {
        ctx.beginPath();
        course.railOffsetsMm.forEach((offsetMm) => {
          course.fragments.forEach((fragment) => appendProjectedContinuousCourse(
            ctx,
            offsetContinuousCourse(fragment, offsetMm),
            layer,
            zMid,
            yaw,
            pitch,
            scale,
            cx,
            cy,
          ));
        });
        ctx.strokeStyle = 'rgba(213, 42, 51, .96)';
        ctx.lineWidth = Math.max(0.9, Math.min(1.8, scale * 0.13));
        ctx.lineJoin = 'miter';
        ctx.miterLimit = 4;
        ctx.lineCap = 'butt';
        ctx.stroke();
      });
      coursePreview.courses.forEach((course) => {
        course.railOffsetsMm.forEach((offsetMm) => {
          course.fragments.forEach((fragment) => {
            const track = offsetContinuousCourse(fragment, offsetMm);
            if (track.length < 2) return;
            [track[0], track.at(-1)].forEach((point, index) => {
            const projected = project(point[0], point[1], physicalLayerZ(heightAt(point[0], point[1]), layer) - zMid, yaw, pitch, scale, cx, cy);
            ctx.beginPath();
            ctx.arc(projected.x, projected.y, 2.7, 0, 2 * Math.PI);
            ctx.fillStyle = index === 0 ? '#16856e' : '#8d1b26';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 0.7;
            ctx.stroke();
          });
          });
        });
      });
      ctx.restore();
      ctx.fillStyle = 'rgba(132, 25, 37, .88)';
      ctx.font = '12px Segoe UI, Microsoft YaHei, sans-serif';
      ctx.fillText(`连续路径蜂窝：中心孔锚定 (${coursePreview.latticeAnchorMm[0].toFixed(2)}, ${coursePreview.latticeAnchorMm[1].toFixed(2)})；黄色由 300 × 300 mm 母板逐边裁切；红色为孔间通道中心线，裁断后每段独立制造；目标边长 ${coursePreview.edgeLength.toFixed(2)} mm；α=${layer.alpha.toFixed(2)}，物理层 ${layer.index + 1}`, 14, 20);
    }

    function renderSolidStack(ctx, width, height) {
      const stack = payload.solid_stack;
      if (!stack) return;
      const layers = stack.layers;
      const xBounds = payload.coordinate_system.xy_bounds_mm;
      const xMin = xBounds[0];
      const xMax = xBounds[2];
      const zValues = [0, ...layers.flatMap((layer) => layer.xz_points.map((point) => point[1]))];
      const zMin = Math.min(...zValues);
      const zMax = Math.max(...zValues);
      const margin = { left: 54, right: 20, top: 28, bottom: 42 };
      const plotWidth = Math.max(1, width - margin.left - margin.right);
      const plotHeight = Math.max(1, height - margin.top - margin.bottom);
      const visualZScale = Number(sectionZScale.value);
      const zMid = (zMin + zMax) * 0.5;
      const displayZMin = zMid + (zMin - zMid) * visualZScale;
      const displayZMax = zMid + (zMax - zMid) * visualZScale;
      const uniformScale = Math.min(
        plotWidth / Math.max(xMax - xMin, 1e-9),
        plotHeight / Math.max(displayZMax - displayZMin, 1e-9),
      );
      const renderedWidth = (xMax - xMin) * uniformScale;
      const renderedHeight = (displayZMax - displayZMin) * uniformScale;
      const offsetX = margin.left + (plotWidth - renderedWidth) * 0.5;
      const offsetY = margin.top + (plotHeight - renderedHeight) * 0.5;
      const mapX = (x) => offsetX + (x - xMin) * uniformScale;
      const mapZ = (z) => offsetY + renderedHeight - (zMid + (z - zMid) * visualZScale - displayZMin) * uniformScale;
      ctx.strokeStyle = '#a9b9ca';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(offsetX, offsetY);
      ctx.lineTo(offsetX, offsetY + renderedHeight);
      ctx.lineTo(offsetX + renderedWidth, offsetY + renderedHeight);
      ctx.stroke();
      ctx.save();
      ctx.strokeStyle = 'rgba(29, 78, 216, .65)';
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(offsetX, mapZ(0));
      ctx.lineTo(offsetX + renderedWidth, mapZ(0));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(29, 78, 216, .82)';
      ctx.font = '600 11px Segoe UI, Microsoft YaHei, sans-serif';
      ctx.fillText('Z=0 基准面', offsetX + 5, mapZ(0) - 5);
      ctx.restore();
      layers.forEach((layer) => {
        ctx.beginPath();
        layer.xz_points.forEach(([x, z], index) => {
          if (index === 0) ctx.moveTo(mapX(x), mapZ(z));
          else ctx.lineTo(mapX(x), mapZ(z));
        });
        ctx.strokeStyle = `hsla(207, 74%, ${35 + layer.alpha * 28}%, ${0.2 + layer.alpha * 0.75})`;
        ctx.lineWidth = layer.index === stack.representative_peak_layer_index ? 2.7 : layer.alpha >= 0.999 ? 2.2 : 1.15;
        ctx.stroke();
        if (payload.inspection_point) {
          const markerZ = layer.base_z_mm + payload.surface.z_reference_mm
            + layer.alpha * (payload.inspection_point.height_mm - payload.surface.z_reference_mm);
          ctx.beginPath();
          ctx.arc(mapX(payload.inspection_point.x_mm), mapZ(markerZ), 2.5, 0, 2 * Math.PI);
          ctx.fillStyle = '#d14322';
          ctx.fill();
        }
      });
      ctx.fillStyle = 'rgba(21,32,51,.78)';
      ctx.font = '12px Segoe UI, Microsoft YaHei, sans-serif';
      ctx.fillText(`XZ 剖面：Y = ${stack.section_y_mm.toFixed(2)} mm；完整曲率层 = L${stack.representative_peak_layer_index + 1}；参考层高 ${stack.reference_layer_height_mm.toFixed(2)} mm；视觉 Z ×${visualZScale}；α 为旧版对称 smoothstep`, margin.left, 17);
      ctx.fillText(`X：${xMin.toFixed(1)} ～ ${xMax.toFixed(1)} mm`, offsetX, height - 16);
      ctx.fillText(`物理 Z：${zMin.toFixed(2)} ～ ${zMax.toFixed(2)} mm`, width - 178, height - 16);
    }

    function render() {
      if (!payload) return;
      const rect = canvas.getBoundingClientRect();
      const pixelRatio = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * pixelRatio));
      canvas.height = Math.max(1, Math.round(rect.height * pixelRatio));
      const ctx = canvas.getContext('2d');
      ctx.scale(pixelRatio, pixelRatio);
      const width = rect.width;
      const height = rect.height;
      ctx.clearRect(0, 0, width, height);
      if (previewMode.value === 'solid_xz' && payload.solid_stack) {
        renderSolidStack(ctx, width, height);
        return;
      }
      const { x, y, z } = payload.grid;
      const stats = payload.statistics;
      const projection = payload.domain.projection;
      const materialMask = payload.grid.material_mask;
      const { yaw, pitch } = view;
      const layer = physicalPreviewLayer();
      const physicalZMin = physicalLayerZ(stats.z_min_mm, layer);
      const physicalZMax = physicalLayerZ(stats.z_max_mm, layer);
      const zMin = Math.min(0, physicalZMin);
      const zMax = Math.max(0, physicalZMax);
      const span = Math.max(payload.domain.width_mm, payload.domain.height_mm, (zMax - zMin) * Number(surfaceZScale.value), 1);
      const scale = Math.min(width, height) * 0.72 * view.zoom / span;
      const cx = width / 2 + view.panX;
      const cy = height / 2 + 8 + view.panY;
      const zMid = (zMin + zMax) / 2;
      const exactProjectionClip = Boolean(projection && !drag);
      const cells = [];
      for (let row = 0; row < z.length - 1; row += 1) {
        for (let col = 0; col < z[row].length - 1; col += 1) {
          if (!exactProjectionClip && materialMask && !materialMask[row][col]) continue;
          const p = [
            project(x[row][col], y[row][col], physicalLayerZ(z[row][col], layer) - zMid, yaw, pitch, scale, cx, cy),
            project(x[row][col + 1], y[row][col + 1], physicalLayerZ(z[row][col + 1], layer) - zMid, yaw, pitch, scale, cx, cy),
            project(x[row + 1][col + 1], y[row + 1][col + 1], physicalLayerZ(z[row + 1][col + 1], layer) - zMid, yaw, pitch, scale, cx, cy),
            project(x[row + 1][col], y[row + 1][col], physicalLayerZ(z[row + 1][col], layer) - zMid, yaw, pitch, scale, cx, cy),
          ];
          const averageZ = (z[row][col] + z[row][col + 1] + z[row + 1][col + 1] + z[row + 1][col]) / 4;
          const centerX = (x[row][col] + x[row][col + 1] + x[row + 1][col + 1] + x[row + 1][col]) / 4;
          const centerY = (y[row][col] + y[row][col + 1] + y[row + 1][col + 1] + y[row + 1][col]) / 4;
          cells.push({ p, depth: p.reduce((sum, point) => sum + point.depth, 0) / 4, averageZ, centerX, centerY });
        }
      }
      cells.sort((a, b) => a.depth - b.depth);
      drawSurfaceReferenceFrame(ctx, zMid, yaw, pitch, scale, cx, cy);
      if (exactProjectionClip) {
        ctx.save();
        clipToProjection(ctx, projection, layer, zMid, yaw, pitch, scale, cx, cy);
      }
      ctx.save();
      ctx.globalAlpha = 0.72;
      cells.forEach((cell) => {
        const fraction = stats.z_range_mm === 0 ? 0.5 : (cell.averageZ - stats.z_min_mm) / stats.z_range_mm;
        ctx.beginPath();
        ctx.moveTo(cell.p[0].x, cell.p[0].y);
        cell.p.slice(1).forEach((point) => ctx.lineTo(point.x, point.y));
        ctx.closePath();
        ctx.fillStyle = colour(fraction, surfaceLighting(cell.centerX, cell.centerY));
        ctx.fill();
      });
      ctx.restore();
      if (exactProjectionClip) ctx.restore();
      drawProjectionBoundaries(ctx, projection, layer, zMid, yaw, pitch, scale, cx, cy);
      drawSurfaceGuideMesh(ctx, x, y, z, layer, zMid, yaw, pitch, scale, cx, cy);
      drawLatticePreview(ctx, layer, zMid, yaw, pitch, scale, cx, cy);
      drawInspectionMarker(ctx, layer, zMid, yaw, pitch, scale, cx, cy);
      ctx.fillStyle = 'rgba(21,32,51,.68)';
      ctx.font = '12px Segoe UI, Microsoft YaHei, sans-serif';
      const originLabel = payload.domain.mode === 'rectangle'
        ? '矩形左下角 (0, 0)'
        : 'STL 投影左下基准 (0, 0)';
      ctx.fillText(`X / Y：mm，原点：${originLabel}；Z=0 为零件底面；展示 L${layer.index + 1}（α=${layer.alpha.toFixed(2)}，中心 Z=${layer.base_z_mm.toFixed(2)} mm）；视觉 Z ×${surfaceZScale.value}`, 14, height - 16);
    }

    function showStats(data) {
      const statistics = data.statistics;
      const point = data.inspection_point;
      const coordinateSystem = data.coordinate_system;
      const values = [
        `Z：${statistics.z_min_mm.toFixed(3)} ～ ${statistics.z_max_mm.toFixed(3)} mm`,
        `起伏：${statistics.z_range_mm.toFixed(3)} mm`,
        `最大坡度：${statistics.max_slope.toFixed(3)}`,
        `坐标：${coordinateSystem.origin_label}；范围：[${coordinateSystem.xy_bounds_mm.join(', ')}] mm`,
        `预览：${data.preview_version}；导出：${data.export_version}；Git：${data.git_revision}`,
      ];
      if (data.surface_parameterization.mode === 'tensile_centered_wave_count') {
        values.splice(3, 0, `拉伸波数：nx=${data.surface_parameterization.wave_count_x.toFixed(2)}；ny=${data.surface_parameterization.wave_count_y.toFixed(2)}`);
      }
      if (point) {
        values.splice(3, 0, `检验点 H：${point.height_mm.toFixed(3)} mm；坡度：${point.slope.toFixed(4)}；平均曲率（有符号）：${point.mean_curvature_per_mm.toFixed(5)} 1/mm`);
      }
      statsEl.replaceChildren(...values.map((value) => {
        const item = document.createElement('span');
        item.className = 'stat';
        item.textContent = value;
        return item;
      }));
    }

    async function refresh() {
      const sequence = ++queued;
      statusEl.className = 'status';
      statusEl.textContent = '正在更新曲面…';
      try {
        const response = await fetch(`/api/surface?${parameters().toString()}`);
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || '无法生成曲面');
        if (sequence !== queued) return;
        payload = result;
        showStats(result);
        updateLatticeLengthSummary();
        render();
        statusEl.textContent = '已更新：当前预览对应固定矩形外边界的双正弦承载曲面。';
      } catch (error) {
        if (sequence !== queued) return;
        statusEl.className = 'status error';
        statusEl.textContent = error.message;
      }
    }

    let timer = null;
    function scheduleRefresh() { clearTimeout(timer); timer = setTimeout(refresh, 120); }
    function invalidateLatticePreview() {
      latticePreviewCache = null;
      continuousCoursePreviewCache = null;
    }
    surfaceIds.forEach((id) => document.getElementById(id).addEventListener('input', () => {
      updateTensileWaveHint();
      saveDesignerState();
      scheduleRefresh();
    }));
    ['part_length_mm', 'part_width_mm', 'part_height_mm', 'surface_start_layer'].forEach((id) => document.getElementById(id).addEventListener('input', () => {
      syncInspectionPointControls();
      updateTensileWaveHint();
      saveDesignerState();
      invalidateLatticePreview();
      scheduleRefresh();
    }));
    conformalDesignIds.forEach((id) => document.getElementById(id).addEventListener('input', () => {
      saveDesignerState();
      updateConformalDesignSummary();
    }));
    document.getElementById('align_load_line').addEventListener('change', () => {
      syncLoadLineAlignmentControls();
      saveDesignerState();
      invalidateLatticePreview();
      updateConformalDesignSummary();
      updateLatticeLengthSummary();
      if (payload) render();
    });
    ['honeycomb_align_x', 'honeycomb_align_y'].forEach((id) => document.getElementById(id).addEventListener('change', () => {
      if (document.getElementById(id).checked) document.getElementById('align_load_line').checked = false;
      syncLoadLineAlignmentControls();
      saveDesignerState();
      invalidateLatticePreview();
      updateConformalDesignSummary();
      updateLatticeLengthSummary();
      if (payload) render();
    }));
    document.getElementById('centreHoneycombAlignment').addEventListener('click', () => {
      const length = positiveNumber('part_length_mm');
      const width = positiveNumber('part_width_mm');
      if (length === null || width === null) return;
      document.getElementById('honeycomb_align_x_mm').value = (length / 2).toFixed(3);
      document.getElementById('honeycomb_align_y_mm').value = (width / 2).toFixed(3);
      saveDesignerState();
      invalidateLatticePreview();
      updateLatticeLengthSummary();
      if (payload) render();
    });
    document.getElementById('surface_parameter_mode').addEventListener('change', () => {
      syncSurfaceParameterControls();
      saveDesignerState();
      scheduleRefresh();
    });
    document.getElementById('inspection_enabled').addEventListener('change', () => {
      syncInspectionPointControls();
      saveDesignerState();
      scheduleRefresh();
    });
    document.getElementById('applyTensilePreset').addEventListener('click', () => {
      document.getElementById('surface_parameter_mode').value = 'tensile_centered_wave_count';
      document.getElementById('amplitude_mm').value = 1.5;
      document.getElementById('wave_count_x').value = 1.5;
      document.getElementById('wave_count_y').value = 1.5;
      document.getElementById('inspection_enabled').checked = false;
      document.getElementById('align_load_line').checked = false;
      syncSurfaceParameterControls();
      syncInspectionPointControls();
      syncLoadLineAlignmentControls();
      invalidateLatticePreview();
      saveDesignerState();
      updateConformalDesignSummary();
      scheduleRefresh();
    });
    ['grip_end_length_mm', 'wall_width_mm', 'base_cell_size_mm', 'orientation_angle_deg', 'honeycomb_align_x_mm', 'honeycomb_align_y_mm'].forEach((id) => document.getElementById(id).addEventListener('input', () => {
      invalidateLatticePreview();
      updateLatticeLengthSummary();
      if (payload) render();
    }));
    document.getElementById('boundary_mode').addEventListener('change', () => {
      saveDesignerState();
      invalidateLatticePreview();
      updateLatticeLengthSummary();
      if (payload) render();
    });
    previewMode.addEventListener('change', () => {
      saveDesignerState();
      previewTitle.textContent = previewMode.value === 'solid_xz' ? '实体层叠 / XZ 剖面' : 'α=1 完整曲率层（物理 Z）';
      render();
    });
    surfaceZScale.addEventListener('change', () => { saveDesignerState(); render(); });
    sectionZScale.addEventListener('change', () => { saveDesignerState(); render(); });
    exportConformalConfigButton.addEventListener('click', async () => {
      try {
        const response = await fetch(`/api/export-conformal-lattice-config?${conformalParameters().toString()}`);
        if (!response.ok) {
          const result = await response.json();
          throw new Error(result.error || '无法导出共形格栅配置');
        }
        const blob = await response.blob();
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'conformal_lattice_spec_v1.json';
        link.click();
        URL.revokeObjectURL(link.href);
        statusEl.className = 'status';
        statusEl.textContent = '已导出连续路径设计 JSON；回到主切片器导入该文件以生成正式路径。';
      } catch (error) {
        statusEl.className = 'status error';
        statusEl.textContent = error.message;
      }
    });
    exportPlanarConfigButton.addEventListener('click', async () => {
      try {
        const response = await fetch(`/api/export-planar-lattice-config?${conformalParameters().toString()}`);
        if (!response.ok) {
          const result = await response.json();
          throw new Error(result.error || '无法导出平面蜂窝结构配置');
        }
        const blob = await response.blob();
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'planar_honeycomb_spec_v1.json';
        link.click();
        URL.revokeObjectURL(link.href);
        statusEl.className = 'status';
        statusEl.textContent = '已导出平面蜂窝结构 JSON；文件不包含双正弦曲面参数，可在主切片器中按平面路径生成 Core NPZ。';
      } catch (error) {
        statusEl.className = 'status error';
        statusEl.textContent = error.message;
      }
    });
    canvas.addEventListener('contextmenu', (event) => event.preventDefault());
    canvas.addEventListener('pointerdown', (event) => {
      const mode = event.button === 0 ? 'rotate' : event.button === 1 ? 'pan' : 'zoom';
      drag = { pointerId: event.pointerId, mode, x: event.clientX, y: event.clientY, zoom: view.zoom };
      canvas.setPointerCapture(event.pointerId);
      canvas.classList.add('isDragging');
      event.preventDefault();
    });
    canvas.addEventListener('pointermove', (event) => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (drag.mode === 'rotate') {
        view.yaw += dx * 0.012;
        view.pitch = Math.max(0.08, Math.min(Math.PI / 2 - 0.08, view.pitch - dy * 0.012));
      } else if (drag.mode === 'pan') {
        view.panX += dx;
        view.panY += dy;
      } else {
        view.zoom = Math.max(0.2, Math.min(5, drag.zoom * Math.exp(-dy * 0.012)));
      }
      drag.x = event.clientX;
      drag.y = event.clientY;
      if (drag.mode === 'zoom') drag.zoom = view.zoom;
      render();
    });
    function endDrag(event) {
      if (!drag || event.pointerId !== drag.pointerId) return;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      drag = null;
      canvas.classList.remove('isDragging');
      render();
    }
    canvas.addEventListener('pointerup', endDrag);
    canvas.addEventListener('pointercancel', endDrag);
    canvas.addEventListener('lostpointercapture', endDrag);
    window.addEventListener('pointerup', endDrag);
    canvas.addEventListener('wheel', (event) => {
      view.zoom = Math.max(0.2, Math.min(5, view.zoom * Math.exp(-event.deltaY * 0.0012)));
      render();
      event.preventDefault();
    }, { passive: false });
    canvas.addEventListener('dblclick', () => {
      Object.assign(view, initialView);
      render();
    });
    document.getElementById('reset').addEventListener('click', () => {
      const defaults = { part_length_mm: 150, part_width_mm: 50, part_height_mm: 10, grip_end_length_mm: 25, surface_parameter_mode: 'tensile_centered_wave_count', amplitude_mm: 1.5, wave_count_x: 1.5, wave_count_y: 1.5, wavelength_x_mm: 100, wavelength_y_mm: 33.333, phase_x_pi: 1, phase_y_pi: 1, z_reference_mm: 0, inspection_enabled: false, check_x_mm: 75, check_y_mm: 25, wall_width_mm: 2, base_cell_size_mm: 10, orientation_angle_deg: 0, honeycomb_align_x: false, honeycomb_align_x_mm: 75, honeycomb_align_y: false, honeycomb_align_y_mm: 25, surface_start_layer: 3, samples_x: 49, samples_y: 49, boundary_mode: 'clip', random_seed: 0, samples: 49, surfaceZScale: 5, sectionZScale: 3, previewMode: 'surface' };
      Object.entries(defaults).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element.type === 'checkbox') element.checked = value;
        else element.value = value;
      });
      document.getElementById('align_load_line').checked = false;
      syncSurfaceParameterControls();
      syncInspectionPointControls();
      syncLoadLineAlignmentControls();
      invalidateLatticePreview();
      saveDesignerState();
      updateConformalDesignSummary();
      refresh();
    });
    window.addEventListener('resize', render);
    async function initialiseDesigner() {
      restoreDesignerState();
      await restorePersistentDesignerState();
      syncSurfaceParameterControls();
      syncInspectionPointControls();
      syncLoadLineAlignmentControls();
      updateConformalDesignSummary();
      refresh();
    }
    initialiseDesigner();
  </script>
</body>
</html>'''
