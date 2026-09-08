"""A small local web server for interactively inspecting surface equations."""

from __future__ import annotations

import json
import math
import secrets
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..conformal_lattice.contracts import (
    CONFORMAL_LATTICE_SPEC_V1,
    double_sine_source_sha256,
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
CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM = 0.5
SURFACE_PREVIEW_API_VERSION = "surface_preview_v2"


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
    inspection: dict[str, float],
) -> dict[str, object] | None:
    """Sample the existing smoothstep stack on the inspection-point XZ cut."""

    if "part_height_mm" not in params and "surface_start_layer" not in params:
        return None
    final_height_mm = _query_float(params, "part_height_mm", 10.0, positive=True)
    start_layer = _query_nonnegative_int(params, "surface_start_layer", 3)
    layer_count = int(math.ceil(final_height_mm / CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM))
    progression = LayerProgression(start_layer, layer_count - 1)
    layer_thicknesses = [CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM] * layer_count
    layer_thicknesses[-1] = final_height_mm - CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM * (layer_count - 1)
    base_z_by_layer: list[float] = []
    accumulated = 0.0
    for thickness in layer_thicknesses:
        base_z_by_layer.append(accumulated + thickness * 0.5)
        accumulated += thickness
    y_mm = inspection["y_mm"]
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

    surface = DoubleSineSurface(
        amplitude_mm=_query_float(params, "amplitude_mm", 0.8),
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
        z_reference_mm=_query_float(params, "z_reference_mm", 0.0),
    )
    width_mm = domain.width_mm if domain else _query_float(
        params, "width_mm", DEFAULT_PREVIEW_WIDTH_MM, positive=True
    )
    height_mm = domain.height_mm if domain else _query_float(
        params, "height_mm", DEFAULT_PREVIEW_HEIGHT_MM, positive=True
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
    inspection = _inspection_point(
        params,
        surface,
        x_bounds_mm=x_bounds,
        y_bounds_mm=y_bounds,
    )
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
            inspection=inspection,
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

    length_mm = _query_float(params, "part_length_mm", 150.0, positive=True)
    width_mm = _query_float(params, "part_width_mm", 100.0, positive=True)
    final_height_mm = _query_float(params, "part_height_mm", 10.0, positive=True)
    # The design page describes surface morphology, not process settings.
    # Keep one stable reference for validating a layer-index start value; the
    # actual physical layer height is supplied later by the slicer/Core UI.
    layer_height_mm = CONFORMAL_MAPPING_REFERENCE_LAYER_HEIGHT_MM
    surface_params = {**params, "width_mm": [str(length_mm)], "height_mm": [str(width_mm)]}
    surface = surface_payload(
        surface_params,
        include_projection_geometry=False,
        rectangle_origin_lower_left=True,
    )["surface"]
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
    surface_start_layer = _query_nonnegative_int(params, "surface_start_layer", 3)
    logical_layer_count = math.ceil(final_height_mm / layer_height_mm)
    if surface_start_layer > (logical_layer_count - 1) // 2:
        raise ValueError("surface_start_layer must leave a symmetric curved region inside the final physical height")
    samples_x = _query_nonnegative_int(
        params, "samples_x", DEFAULT_PREVIEW_SAMPLES, minimum=2, maximum=MAX_CONFORMAL_SAMPLES
    )
    samples_y = _query_nonnegative_int(
        params, "samples_y", DEFAULT_PREVIEW_SAMPLES, minimum=2, maximum=MAX_CONFORMAL_SAMPLES
    )
    boundary_mode = params.get("boundary_mode", ["clip"])[0]
    if boundary_mode not in {"clip", "inset"}:
        raise ValueError("boundary_mode must be clip or inset")
    phase_origin = [
        _query_float(params, "phase_origin_x_mm", 0.0),
        _query_float(params, "phase_origin_y_mm", 0.0),
    ]
    orientation_angle_deg = _query_float(params, "orientation_angle_deg", 0.0)
    random_seed = _query_nonnegative_int(params, "random_seed", 0)
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
        },
        "fill_field": {"mode": "fixed_cell_size", "drivers": []},
        "orientation_field": {"mode": "global_axis", "angle_deg": orientation_angle_deg, "constraints": []},
        "layer_embedding": {
            "mode": "symmetric_shape_morphing",
            "transition": "smoothstep",
            "surface_start_layer": surface_start_layer,
        },
        "quality_limits": {},
        "random_seed": random_seed,
    }
    load_conformal_lattice_spec(config)
    return config


def run_surface_preview_server(host: str, port: int) -> None:
    """Start the independent local surface-preview server."""

    server = ThreadingHTTPServer((host, port), SurfacePreviewHandler)
    server.preview_domains = {}
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
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
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
      <p>在固定矩形实体上定义双正弦承载曲面和六边形格栅；导出 JSON 后回到主切片器生成路径与送入 Core。</p>
    </header>
    <section class="workspace">
      <form class="panel controls" id="surfaceForm">
        <h2>矩形实体</h2>
        <div class="field"><label for="part_length_mm">零件长度 X（mm）</label><input id="part_length_mm" type="number" min="0.001" step="1" value="150"></div>
        <div class="field"><label for="part_width_mm">零件宽度 Y（mm）</label><input id="part_width_mm" type="number" min="0.001" step="1" value="100"></div>
        <div class="field"><label for="part_height_mm">最终物理高度 Z（mm）</label><input id="part_height_mm" type="number" min="0.001" step="0.1" value="10"></div>
        <p class="modelMeta" id="modelMeta">外边界固定为矩形；新共形流程不读取 STL，也不继承 STL 中的蜂窝孔壁。</p>
        <div class="divider"></div>
        <h2>曲面参数</h2>
        <div class="field"><label for="amplitude_mm">幅值 A（mm）</label><input id="amplitude_mm" type="number" step="0.01" value="1.5"></div>
        <div class="field"><label for="wavelength_x_mm">X 波长 λx（mm）</label><input id="wavelength_x_mm" type="number" min="0.001" step="0.1" value="100"></div>
        <div class="field"><label for="wavelength_y_mm">Y 波长 λy（mm）</label><input id="wavelength_y_mm" type="number" min="0.001" step="0.1" value="200"></div>
        <div class="field"><label for="phase_x_pi">X 相位 φx（π）</label><input id="phase_x_pi" type="number" step="0.25" value="1" aria-describedby="phasePiHint"></div>
        <div class="field"><label for="phase_y_pi">Y 相位 φy（π）</label><input id="phase_y_pi" type="number" step="0.25" value="0" aria-describedby="phasePiHint"></div>
        <p class="hint" id="phasePiHint">输入 π 的倍数：1 表示 π，0.5 表示 π/2，1.5 表示 3π/2；导出的设计 JSON 仍以 rad 保存。</p>
        <div class="field"><label for="z_reference_mm">Z 基准（mm）</label><input id="z_reference_mm" type="number" step="0.01" value="0"></div>
        <div class="field"><label for="check_x_mm">检验点 X（mm）</label><input id="check_x_mm" type="number" min="0" step="0.1" value="75" aria-describedby="checkPointHint"></div>
        <div class="field"><label for="check_y_mm">检验点 Y（mm）</label><input id="check_y_mm" type="number" min="0" step="0.1" value="50" aria-describedby="checkPointHint"></div>
        <p class="hint" id="checkPointHint">默认检验零件中心 (75, 50)。预览会标出该点，并显示目标曲面的 H、坡度和平均曲率。</p>
        <div class="divider"></div>
        <h2>固定六边形格栅</h2>
        <div class="field"><label for="wall_width_mm">设计墙宽（mm）</label><input id="wall_width_mm" type="number" min="2" step="2" value="2"></div>
        <div class="field"><label for="base_cell_size_mm">目标六边形边长（mm）</label><input id="base_cell_size_mm" type="number" min="0.001" step="0.01" value="5"></div>
        <div class="field"><label for="orientation_angle_deg">全局格栅方向角（°）</label><input id="orientation_angle_deg" type="number" step="1" value="0"></div>
        <p class="hint">喷嘴基准线宽固定为 2 mm。墙宽只能填 2、4、6… mm；4 mm 代表后续由两条 2 mm 沉积道组成。目标边长沿承载曲面测量。</p>
        <div class="designSummary" id="latticeDesignSummary" aria-live="polite"></div>
        <div class="divider"></div>
        <h2>对称层间渐变</h2>
        <div class="field"><label for="surface_start_layer">曲面起始层</label><input id="surface_start_layer" type="number" min="0" step="1" value="3"></div>
        <p class="hint">沿用旧版语义：起始层本身保持平面，下一层才开始增大曲率。</p>
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
        <button type="button" id="exportConformalConfig">导出共形蜂窝设计 JSON</button>
        <button type="button" class="secondary" id="reset">恢复示例参数</button>
        <p class="hint">方程：H(x,y)=A·sin(2πx/λx+φx)·sin(2πy/λy+φy)+Zref。导出文件为 <code>conformal_lattice_spec_v1.json</code>，请在主切片器中导入该文件。</p>
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
    const surfaceIds = ['amplitude_mm', 'wavelength_x_mm', 'wavelength_y_mm', 'phase_x_pi', 'phase_y_pi', 'z_reference_mm', 'check_x_mm', 'check_y_mm', 'samples'];
    const mappingReferenceLayerHeightMm = 0.5;
    const conformalDesignIds = ['part_length_mm', 'part_width_mm', 'part_height_mm', 'wall_width_mm', 'base_cell_size_mm', 'orientation_angle_deg', 'surface_start_layer', 'samples_x', 'samples_y', 'boundary_mode', 'random_seed'];
    const canvas = document.getElementById('canvas');
    const statusEl = document.getElementById('status');
    const statsEl = document.getElementById('stats');
    const exportConformalConfigButton = document.getElementById('exportConformalConfig');
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
    const initialView = { yaw: -42 * Math.PI / 180, pitch: 54 * Math.PI / 180, zoom: 1, panX: 0, panY: 0 };
    const view = { ...initialView };
    let drag = null;

    function positiveNumber(id) {
      const value = Number(document.getElementById(id).value);
      return Number.isFinite(value) && value > 0 ? value : null;
    }

    function nonNegativeInteger(id) {
      const value = Number(document.getElementById(id).value);
      return Number.isInteger(value) && value >= 0 ? value : null;
    }

    function saveDesignerState() {
      try {
        const state = Object.fromEntries(persistedInputIds.map((id) => [id, document.getElementById(id).value]));
        localStorage.setItem(designerStateKey, JSON.stringify(state));
      } catch (_) {
        // Local preview remains usable when browser storage is unavailable.
      }
    }

    function restoreDesignerState() {
      try {
        const state = JSON.parse(localStorage.getItem(designerStateKey) || 'null');
        if (!state || typeof state !== 'object') return;
        persistedInputIds.forEach((id) => {
          if (typeof state[id] === 'string') document.getElementById(id).value = state[id];
        });
      } catch (_) {
        // Ignore malformed or unavailable browser-local state.
      }
    }

    function updateConformalDesignSummary() {
      const wallWidth = positiveNumber('wall_width_mm');
      const cellSize = positiveNumber('base_cell_size_mm');
      const latticeSummary = document.getElementById('latticeDesignSummary');
      const beadCount = wallWidth === null ? null : Math.round(wallWidth / 2);
      const isNozzleMultiple = beadCount !== null && beadCount >= 1 && Math.abs(wallWidth - 2 * beadCount) < 1e-9;
      if (wallWidth === null || cellSize === null) {
        latticeSummary.className = 'designSummary error';
        latticeSummary.textContent = '设计墙宽和目标六边形边长都必须是正数。';
      } else if (!isNozzleMultiple) {
        latticeSummary.className = 'designSummary error';
        latticeSummary.textContent = '设计墙宽必须是 2 mm 喷嘴基准线宽的正整数倍，例如 2、4、6。';
      } else {
        const nominalFill = (2 * wallWidth) / (Math.sqrt(3) * cellSize);
        if (nominalFill >= 1) {
          latticeSummary.className = 'designSummary error';
          latticeSummary.textContent = `名义填充率为 ${(nominalFill * 100).toFixed(1)}%，必须小于 100%。请减小墙宽或增大单元边长。`;
        } else {
          latticeSummary.className = 'designSummary';
          latticeSummary.textContent = `名义填充率：${(nominalFill * 100).toFixed(1)}%；墙体将规划为 ${beadCount} 条 2 mm 沉积道。实际填充率以生成几何测量结果为准。`;
        }
      }

      const startLayer = nonNegativeInteger('surface_start_layer');
      const samplesX = nonNegativeInteger('samples_x');
      const samplesY = nonNegativeInteger('samples_y');
      const partHeight = positiveNumber('part_height_mm');
      const progressionSummary = document.getElementById('layerProgressionSummary');
      if (startLayer === null) {
        progressionSummary.className = 'designSummary error';
        progressionSummary.textContent = '曲面起始层必须是非负整数。';
      } else if (partHeight === null) {
        progressionSummary.className = 'designSummary error';
        progressionSummary.textContent = '最终物理高度必须是正数。';
      } else if (samplesX === null || samplesX < 2 || samplesY === null || samplesY < 2) {
        progressionSummary.className = 'designSummary error';
        progressionSummary.textContent = '曲面采样 X 和 Y 都必须是不小于 2 的整数。';
      } else {
        const layerCount = Math.ceil(partHeight / mappingReferenceLayerHeightMm);
        const maxStart = Math.floor((layerCount - 1) / 2);
        if (startLayer > maxStart) {
          progressionSummary.className = 'designSummary error';
          progressionSummary.textContent = `当前高度与层高共得到 ${layerCount} 个逻辑层；曲面起始层不能大于 ${maxStart}。`;
        } else {
          const returnLayer = layerCount - 1 - startLayer;
          const peakLayers = layerCount % 2 === 1 ? `${Math.floor(layerCount / 2)}` : `${layerCount / 2 - 1}、${layerCount / 2}`;
          progressionSummary.className = 'designSummary';
          progressionSummary.textContent = `映射参考层数：${layerCount}；曲面起始层：${startLayer}；镜像回落层：${returnLayer}；完整曲率层：${peakLayers}；共形采样：${samplesX} × ${samplesY}。实际切片层高在主界面 Core 工艺参数中设置。`;
        }
      }
    }

    function parameters() {
      const query = new URLSearchParams();
      surfaceIds.forEach((id) => query.set(id, document.getElementById(id).value));
      query.set('width_mm', document.getElementById('part_length_mm').value);
      query.set('height_mm', document.getElementById('part_width_mm').value);
      query.set('part_height_mm', document.getElementById('part_height_mm').value);
      query.set('surface_start_layer', document.getElementById('surface_start_layer').value);
      return query;
    }

    function conformalParameters() {
      const query = parameters();
      conformalDesignIds.forEach((id) => query.set(id, document.getElementById(id).value));
      return query;
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
      const bounds = payload.coordinate_system.xy_bounds_mm;
      const angle = Number(document.getElementById('orientation_angle_deg').value) * Math.PI / 180;
      if (!Number.isFinite(angle)) return null;
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
      return { edgeLength, wallWidth, angle, origin, bounds, boundaryMode };
    }

    function latticePreviewSegments() {
      const settings = latticePreviewParameters();
      if (!settings) return { segments: [], sampled: false, edgeLength: 0, wallWidth: 0 };
      const { edgeLength, wallWidth, angle, origin, bounds, boundaryMode } = settings;
      const key = JSON.stringify({ edgeLength, wallWidth, angle, origin, bounds, boundaryMode });
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
      const value = { segments, sampled, edgeLength, previewEdgeLength, wallWidth };
      latticePreviewCache = { key, value };
      return value;
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
      const lattice = latticePreviewSegments();
      if (!lattice.segments.length) return;
      const wavelength = Math.min(payload.surface.wavelength_x_mm, payload.surface.wavelength_y_mm);
      const subdivisions = drag ? 1 : Math.max(2, Math.min(6, Math.ceil(lattice.previewEdgeLength / Math.max(wavelength / 8, 0.1))));
      ctx.save();
      ctx.beginPath();
      lattice.segments.forEach(([start, end]) => {
        for (let index = 0; index <= subdivisions; index += 1) {
          const ratio = index / subdivisions;
          const x = start[0] + (end[0] - start[0]) * ratio;
          const y = start[1] + (end[1] - start[1]) * ratio;
          const point = project(x, y, physicalLayerZ(heightAt(x, y), layer) - zMid, yaw, pitch, scale, cx, cy);
          if (index === 0) ctx.moveTo(point.x, point.y);
          else ctx.lineTo(point.x, point.y);
        }
      });
      ctx.strokeStyle = 'rgba(7, 91, 76, .88)';
      ctx.lineWidth = Math.max(0.8, Math.min(12, lattice.wallWidth * scale * 0.72));
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = 'rgba(7, 71, 62, .78)';
      ctx.font = '12px Segoe UI, Microsoft YaHei, sans-serif';
      const sampling = lattice.sampled ? `；显示降采样边长 ${lattice.previewEdgeLength.toFixed(2)} mm` : '';
      ctx.fillText(`蜂窝格栅：墙宽 ${lattice.wallWidth.toFixed(2)} mm；目标边长 ${lattice.edgeLength.toFixed(2)} mm；α=${layer.alpha.toFixed(2)}，物理层 ${layer.index + 1}${sampling}`, 14, 20);
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
        const markerZ = layer.base_z_mm + payload.surface.z_reference_mm
          + layer.alpha * (payload.inspection_point.height_mm - payload.surface.z_reference_mm);
        ctx.beginPath();
        ctx.arc(mapX(payload.inspection_point.x_mm), mapZ(markerZ), 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = '#d14322';
        ctx.fill();
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
        `检验点 H：${point.height_mm.toFixed(3)} mm；坡度：${point.slope.toFixed(4)}；平均曲率（有符号）：${point.mean_curvature_per_mm.toFixed(5)} 1/mm`,
        `坐标：${coordinateSystem.origin_label}；范围：[${coordinateSystem.xy_bounds_mm.join(', ')}] mm`,
        `预览：${data.preview_version}；导出：${data.export_version}；Git：${data.git_revision}`,
      ];
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
    function invalidateLatticePreview() { latticePreviewCache = null; }
    surfaceIds.forEach((id) => document.getElementById(id).addEventListener('input', () => {
      saveDesignerState();
      scheduleRefresh();
    }));
    ['part_length_mm', 'part_width_mm', 'part_height_mm', 'surface_start_layer'].forEach((id) => document.getElementById(id).addEventListener('input', () => {
      saveDesignerState();
      invalidateLatticePreview();
      scheduleRefresh();
    }));
    conformalDesignIds.forEach((id) => document.getElementById(id).addEventListener('input', () => {
      saveDesignerState();
      updateConformalDesignSummary();
    }));
    ['wall_width_mm', 'base_cell_size_mm', 'orientation_angle_deg'].forEach((id) => document.getElementById(id).addEventListener('input', () => {
      invalidateLatticePreview();
      if (payload) render();
    }));
    document.getElementById('boundary_mode').addEventListener('change', () => {
      saveDesignerState();
      invalidateLatticePreview();
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
        statusEl.textContent = '已导出共形蜂窝设计 JSON；回到主切片器导入该文件以生成路径。';
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
      const defaults = { part_length_mm: 150, part_width_mm: 100, part_height_mm: 10, amplitude_mm: 1.5, wavelength_x_mm: 100, wavelength_y_mm: 200, phase_x_pi: 1, phase_y_pi: 0, z_reference_mm: 0, check_x_mm: 75, check_y_mm: 50, wall_width_mm: 2, base_cell_size_mm: 5, orientation_angle_deg: 0, surface_start_layer: 3, samples_x: 49, samples_y: 49, boundary_mode: 'clip', random_seed: 0, samples: 49, surfaceZScale: 5, sectionZScale: 3, previewMode: 'surface' };
      Object.entries(defaults).forEach(([id, value]) => {
        document.getElementById(id).value = value;
      });
      invalidateLatticePreview();
      saveDesignerState();
      updateConformalDesignSummary();
      refresh();
    });
    window.addEventListener('resize', render);
    restoreDesignerState();
    updateConformalDesignSummary();
    refresh();
  </script>
</body>
</html>'''
