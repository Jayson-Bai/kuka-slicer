from __future__ import annotations

from datetime import datetime
from email import policy
from email.parser import BytesParser
import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np

from .external_npz import MaterialPaths, write_external_source_npz
from .slicer import (
    DEFAULT_FIBER_LINE_WIDTH_MM,
    DEFAULT_FIBER_LAYER_HEIGHT_MM,
    DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES,
    DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR,
    DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
    DEFAULT_RESIN_INFILL_OVERLAP_PERCENT,
    DEFAULT_RESIN_LAYER_HEIGHT_MM,
    DEFAULT_RESIN_LINE_WIDTH_MM,
    PySLMConfig,
    RaftLayerConfig,
    SliceConfig,
    _intersect_mesh_at_z,
    _layer_z_values,
    _stitch_segments,
    add_raft_to_job,
    normalize_job_xy_origin,
    optimize_open_path_travel,
    orient_mesh_for_build_axis,
    recommended_geometry_tolerance,
    recommended_pyslm_strategy_defaults,
    slice_mesh_to_job,
    _smooth_path_corners,
)
from .stl_io import load_stl


def run_ui_server(host: str, port: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    class SlicerUiHandler(_SlicerUiHandler):
        server_output_dir = output_dir.resolve()

    server = ThreadingHTTPServer((host, port), SlicerUiHandler)
    print(f"KUKA slicer UI running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")
    finally:
        server.server_close()


class _SlicerUiHandler(BaseHTTPRequestHandler):
    server_output_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_index_html())
            return
        if parsed.path.startswith("/outputs/"):
            self._send_output_file(parsed.path.removeprefix("/outputs/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/slice":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            result = self._handle_slice(parsed.query)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"ok": True, **result})

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _handle_slice(self, query: str) -> dict[str, object]:
        params, files = self._read_slice_request(query)
        filename = _safe_filename(params.get("filename", ["input.stl"])[0])
        layer_height = _float_param(
            params,
            "layer_height",
            DEFAULT_RESIN_LAYER_HEIGHT_MM,
        )
        line_width = _float_param(params, "line_width", DEFAULT_RESIN_LINE_WIDTH_MM)
        pyslm_strategy_defaults = recommended_pyslm_strategy_defaults(layer_height, line_width)
        requested_build_axis = params.get("build_axis", ["auto"])[0]
        z_min = _optional_float_param(params, "z_min")
        z_max = _optional_float_param(params, "z_max")
        tolerance = _optional_float_param(params, "tolerance")
        if tolerance is None:
            tolerance = recommended_geometry_tolerance(layer_height, line_width)
        perimeter_count = _int_param(params, "perimeter_count", 2)
        smoothing_angle = _float_param(
            params,
            "smoothing_angle",
            DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES,
        )
        smoothing_radius_factor = _float_param(
            params,
            "smoothing_radius_factor",
            DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR,
        )
        infill_density = _float_param(
            params,
            "infill_density",
            DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
        )
        infill_overlap = _float_param(
            params,
            "infill_overlap",
            DEFAULT_RESIN_INFILL_OVERLAP_PERCENT,
        )
        slicing_kernel = params.get("slicing_kernel", ["legacy"])[0]
        pyslm_config = PySLMConfig(
            hatcher=params.get("pyslm_hatcher", ["basic"])[0],  # type: ignore[arg-type]
            hatch_angle=_optional_float_param(params, "pyslm_hatch_angle"),
            layer_angle_increment=_float_param(params, "pyslm_layer_angle_increment", 0.0),
            hatch_distance=_optional_float_param(params, "pyslm_hatch_distance"),
            contour_offset=_optional_float_param(params, "pyslm_contour_offset"),
            spot_compensation=_optional_float_param(params, "pyslm_spot_compensation"),
            volume_offset_hatch=_optional_float_param(params, "pyslm_volume_offset_hatch"),
            num_outer_contours=_optional_int_param(params, "pyslm_num_outer_contours"),
            num_inner_contours=_optional_int_param(params, "pyslm_num_inner_contours"),
            scan_contour_first=_bool_param(params, "pyslm_scan_contour_first", True),
            hatch_sort=params.get("pyslm_hatch_sort", ["none"])[0],  # type: ignore[arg-type]
            stripe_width=_float_param(
                params,
                "pyslm_stripe_width",
                pyslm_strategy_defaults.width,
            ),
            stripe_overlap=_float_param(
                params,
                "pyslm_stripe_overlap",
                pyslm_strategy_defaults.overlap,
            ),
            stripe_offset=_float_param(
                params,
                "pyslm_stripe_offset",
                pyslm_strategy_defaults.offset,
            ),
            island_width=_float_param(
                params,
                "pyslm_island_width",
                pyslm_strategy_defaults.width,
            ),
            island_overlap=_float_param(
                params,
                "pyslm_island_overlap",
                pyslm_strategy_defaults.overlap,
            ),
            island_offset=_float_param(
                params,
                "pyslm_island_offset",
                pyslm_strategy_defaults.offset,
            ),
            fix_polygons=_bool_param(params, "pyslm_fix_polygons", True),
            simplification_factor=_optional_float_param(params, "pyslm_simplification_factor"),
            simplification_preserve_topology=_bool_param(
                params,
                "pyslm_simplification_preserve_topology",
                True,
            ),
            simplification_mode=params.get("pyslm_simplification_mode", ["absolute"])[0],  # type: ignore[arg-type]
        )
        infill_pattern = params.get("infill_pattern", ["rectilinear"])[0]
        curve_mode = params.get("curve_mode", ["flat"])[0]
        curve_amplitude = _float_param(params, "curve_amplitude", 0.0)
        curve_period = _float_param(params, "curve_period", 50.0)
        raft_layer_count = _int_param(params, "raft_layer_count", 2)
        raft_top_gap = _float_param(params, "raft_top_gap", 0.0)
        raft_layers = _raft_layers_from_params(params, raft_layer_count)

        stl_upload = files.get("stl_file")
        if stl_upload is None:
            raise ValueError("missing STL payload")
        stl_filename, stl_bytes = stl_upload
        filename = _safe_filename(stl_filename or filename)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        job_dir = self.server_output_dir / stamp
        job_dir.mkdir(parents=True, exist_ok=True)
        stl_path = job_dir / filename
        npz_path = job_dir / f"{Path(filename).stem}_source.npz"
        stl_path.write_bytes(stl_bytes)

        fiber_json_name = None
        fiber_template_paths = []
        if "fiber_json" in files:
            fiber_filename, fiber_bytes = files["fiber_json"]
            fiber_json_name = _safe_filename(fiber_filename or "fiber_paths.json")
            fiber_json_path = job_dir / fiber_json_name
            fiber_json_path.write_bytes(fiber_bytes)
            fiber_template_paths = load_fiber_template_json(fiber_json_path)

        mesh = load_stl(stl_path)
        build_axis = resolve_build_axis(mesh, requested_build_axis)
        config = SliceConfig(
            material="R",
            layer_height=layer_height,
            line_width=line_width,
            z_min=z_min,
            z_max=z_max,
            tolerance=tolerance,
            build_axis=build_axis,  # type: ignore[arg-type]
            curve_mode=curve_mode,  # type: ignore[arg-type]
            curve_amplitude=curve_amplitude,
            curve_period=curve_period,
            infill_pattern=infill_pattern,  # type: ignore[arg-type]
            infill_density=infill_density,
            infill_overlap=infill_overlap,
            slicing_kernel=slicing_kernel,  # type: ignore[arg-type]
            pyslm=pyslm_config,
            perimeter_count=perimeter_count,
            smoothing_angle=smoothing_angle,
            smoothing_radius_factor=smoothing_radius_factor,
        )
        job = slice_mesh_to_job(mesh, config)
        fiber_preview_paths = {}
        if fiber_template_paths:
            fiber_preview_paths = expand_fiber_template_for_resin_layers(job, fiber_template_paths)
            merge_fiber_paths_into_job(job, fiber_preview_paths)
        if raft_layers:
            z_shift = add_raft_to_job(job, mesh, config, raft_layers, raft_top_gap)
            fiber_preview_paths = _shift_fiber_preview_paths(
                fiber_preview_paths,
                len(raft_layers),
                z_shift,
            )
        normalize_job_xy_origin(job)
        fiber_preview_paths = _fiber_preview_paths_from_job(job)
        write_external_source_npz(job, npz_path)

        path_count = sum(len(group.paths) for group in job.material_paths)
        preview = _preview_payload(mesh, config, job, fiber_preview_paths)
        recommendation = _triangle_infill_recommendation(mesh, config, job)
        return {
            "download_url": f"/outputs/{quote(stamp)}/{quote(npz_path.name)}",
            "filename": npz_path.name,
            "layers": len(preview["layers"]),
            "paths": path_count,
            "preview": preview,
            "recommendation": recommendation,
            "fiber_json": fiber_json_name,
            "build_axis": build_axis,
            "slicing_kernel": config.slicing_kernel,
        }

    def _read_slice_request(
        self, query: str
    ) -> tuple[dict[str, list[str]], dict[str, tuple[str | None, bytes]]]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("missing request body")
        body = self.rfile.read(content_length)
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            return _parse_multipart_form(content_type, body)

        params = parse_qs(query)
        filename = params.get("filename", ["input.stl"])[0]
        return params, {"stl_file": (filename, body)}

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_output_file(self, relative_url_path: str) -> None:
        parts = [unquote(part) for part in relative_url_path.split("/") if part]
        target = self.server_output_dir.joinpath(*parts).resolve()
        if not str(target).startswith(str(self.server_output_dir)) or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{html.escape(target.name)}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _float_param(params: dict[str, list[str]], name: str, default: float) -> float:
    raw = params.get(name, [str(default)])[0]
    return float(raw if raw != "" else default)


def _optional_float_param(params: dict[str, list[str]], name: str) -> float | None:
    raw = params.get(name, [""])[0].strip()
    return None if raw == "" else float(raw)


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    raw = params.get(name, [str(default)])[0]
    return int(raw if raw != "" else default)


def _optional_int_param(params: dict[str, list[str]], name: str) -> int | None:
    raw = params.get(name, [""])[0].strip()
    return None if raw == "" else int(raw)


def _bool_param(params: dict[str, list[str]], name: str, default: bool) -> bool:
    raw = params.get(name, [str(default).lower()])[0].strip().lower()
    return raw in ("1", "true", "yes", "on")


def _raft_layers_from_params(
    params: dict[str, list[str]],
    layer_count: int,
) -> list[RaftLayerConfig]:
    if layer_count <= 0:
        return []
    offsets = _float_list_param(params, "raft_offsets", "15,10", layer_count)
    heights = _float_list_param(params, "raft_layer_heights", DEFAULT_RESIN_LAYER_HEIGHT_MM, layer_count)
    densities = _float_list_param(params, "raft_infill_densities", "100,70", layer_count)
    patterns = _string_list_param(params, "raft_infill_patterns", "zigzag,zigzag", layer_count)
    return [
        RaftLayerConfig(
            outward_offset=offsets[index],
            layer_height=heights[index],
            infill_density=densities[index],
            infill_pattern=patterns[index],  # type: ignore[arg-type]
        )
        for index in range(layer_count)
    ]


def _float_list_param(
    params: dict[str, list[str]],
    name: str,
    default: float | str,
    layer_count: int,
) -> list[float]:
    raw = params.get(name, [str(default)])[0].strip()
    fallback = float(default) if isinstance(default, (float, int)) else 0.0
    values = [float(part.strip()) for part in raw.split(",") if part.strip()] if raw else [fallback]
    if len(values) == 1:
        return values * layer_count
    if len(values) != layer_count:
        raise ValueError(f"{name} must contain either 1 value or {layer_count} comma-separated values")
    return values


def _string_list_param(
    params: dict[str, list[str]],
    name: str,
    default: str,
    layer_count: int,
) -> list[str]:
    raw = params.get(name, [default])[0].strip()
    values = [part.strip() for part in raw.split(",") if part.strip()] if raw else []
    aliases = {
        "concentric": "concentric",
        "zigzag": "zigzag",
        "同心轮廓": "concentric",
        "同心轮廓填充": "concentric",
        "之字形": "zigzag",
        "之字形填充": "zigzag",
    }
    values = [aliases.get(value, value) for value in values]
    if len(values) == 1:
        values *= layer_count
    if len(values) != layer_count:
        raise ValueError(f"{name} must contain either 1 value or {layer_count} comma-separated values")
    unsupported = [value for value in values if value not in ("concentric", "zigzag")]
    if unsupported:
        raise ValueError(f"{name} contains unsupported pattern: {unsupported[0]}")
    return values


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(filename).name)
    return cleaned or "input.stl"


def _parse_multipart_form(
    content_type: str, body: bytes
) -> tuple[dict[str, list[str]], dict[str, tuple[str | None, bytes]]]:
    message_bytes = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body
    )
    message = BytesParser(policy=policy.default).parsebytes(message_bytes)
    params: dict[str, list[str]] = {}
    files: dict[str, tuple[str | None, bytes]] = {}

    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if "form-data" not in disposition:
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_param("filename", header="content-disposition")
        payload = part.get_payload(decode=True) or b""
        if filename:
            if payload:
                files[name] = (filename, payload)
        else:
            params.setdefault(name, []).append(payload.decode("utf-8"))

    return params, files


def resolve_build_axis(mesh, requested_axis: str) -> str:
    if requested_axis in ("x", "y", "z"):
        return requested_axis
    if requested_axis != "auto":
        raise ValueError("build_axis must be auto, x, y, or z")

    points = mesh.triangles.reshape(-1, 3)
    size = points.max(axis=0) - points.min(axis=0)
    axis_index = int(np.argmin(size))
    return ("x", "y", "z")[axis_index]


def load_fiber_template_json(json_path: Path) -> list[list[list[float]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if len(data) != 1:
            raise ValueError("fiber JSON object must contain exactly one path-family key")
        data = next(iter(data.values()))
    if not isinstance(data, list):
        raise ValueError("fiber JSON must be a list of paths or an object containing one path list")

    paths: list[list[list[float]]] = []
    for path_index, raw_path in enumerate(data):
        if not isinstance(raw_path, list):
            raise ValueError(f"fiber JSON path {path_index} must be a list")
        path: list[list[float]] = []
        for point_index, raw_point in enumerate(raw_path):
            if isinstance(raw_point, dict):
                if "x" not in raw_point or "y" not in raw_point:
                    raise ValueError(f"fiber JSON path {path_index} point {point_index} must contain x and y")
                x = float(raw_point["x"])
                y = float(raw_point["y"])
            elif isinstance(raw_point, list | tuple) and len(raw_point) >= 2:
                x = float(raw_point[0])
                y = float(raw_point[1])
            else:
                raise ValueError(f"fiber JSON path {path_index} point {point_index} has unsupported format")
            path.append([x, y, 0.0])
        if len(path) >= 2:
            paths.append(path)

    if not paths:
        raise ValueError("fiber JSON contains no valid paths")
    return paths


def expand_fiber_template_for_resin_layers(
    job, template_paths: list[list[list[float]]]
) -> dict[int, list[list[list[float]]]]:
    resin_groups = [group for group in job.material_paths if group.material == "R"]
    resin_groups.sort(key=lambda group: group.layer_index)
    paths_by_layer: dict[int, list[list[list[float]]]] = {}

    # Fiber is printed between resin layers; the final resin layer is a cap.
    for group in resin_groups[:-1]:
        z = _group_layer_z(group) + DEFAULT_FIBER_LAYER_HEIGHT_MM
        layer_paths = []
        for template_path in template_paths:
            layer_paths.append(_smooth_fiber_template_path(template_path, z))
        layer_paths = [
            path.tolist()
            for path in optimize_open_path_travel(
                [np.asarray(path, dtype=np.float32) for path in layer_paths]
            )
        ]
        paths_by_layer[group.layer_index] = layer_paths
    return paths_by_layer


def _smooth_fiber_template_path(template_path: list[list[float]], z: float) -> list[list[float]]:
    path = np.asarray([[float(x), float(y), z] for x, y, _ in template_path], dtype=np.float32)
    smoothed_xy = _smooth_path_corners(
        path,
        DEFAULT_FIBER_LINE_WIDTH_MM * DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR,
        DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES,
        1e-5,
    )
    return [[float(x), float(y), z] for x, y in smoothed_xy[:, :2]]


def _shift_fiber_preview_paths(
    paths_by_layer: dict[int, list[list[list[float]]]],
    layer_offset: int,
    z_shift: float,
) -> dict[int, list[list[list[float]]]]:
    shifted: dict[int, list[list[list[float]]]] = {}
    for layer_index, paths in paths_by_layer.items():
        shifted[layer_index + layer_offset] = [
            [[float(x), float(y), float(z) + z_shift] for x, y, z in path]
            for path in paths
        ]
    return shifted


def _group_layer_z(group) -> float:
    for path in group.paths:
        if len(path) > 0:
            return float(path[0][2])
    return float(group.layer_index)


def merge_fiber_paths_into_job(job, fiber_paths_by_layer: dict[int, list[list[list[float]]]]) -> None:
    existing = {(group.layer_index, group.material) for group in job.material_paths}
    for layer_index in sorted(fiber_paths_by_layer):
        if (layer_index, "F") in existing:
            continue
        paths = [np.asarray(path, dtype=np.float32) for path in fiber_paths_by_layer[layer_index]]
        if paths:
            job.material_paths.append(MaterialPaths(layer_index, "F", paths))
    job.material_paths.sort(key=lambda group: (group.layer_index, 0 if group.material == "R" else 1))


def _fiber_preview_paths_from_job(job) -> dict[int, list[list[list[float]]]]:
    paths_by_layer: dict[int, list[list[list[float]]]] = {}
    for group in job.material_paths:
        if group.material != "F":
            continue
        paths_by_layer.setdefault(group.layer_index, []).extend(
            [
                [[float(point[0]), float(point[1]), float(point[2])] for point in path]
                for path in group.paths
            ]
        )
    return paths_by_layer


def _preview_payload(
    mesh, config: SliceConfig, job, fiber_paths_by_layer: dict[int, list[list[list[float]]]] | None = None
) -> dict[str, object]:
    layers_by_index: dict[int, dict[str, object]] = {}
    bounds = {
        "min_x": None,
        "max_x": None,
        "min_y": None,
        "max_y": None,
        "min_z": None,
        "max_z": None,
    }
    fiber_paths_by_layer = fiber_paths_by_layer or {}
    resin_roles_by_layer = (
        job.meta.get("path_roles", {}).get("R", {})
        if isinstance(job.meta.get("path_roles", {}), dict)
        else {}
    )
    groups_by_layer: dict[int, dict[str, list]] = {}
    for group in job.material_paths:
        groups_by_layer.setdefault(group.layer_index, {}).setdefault(group.material, []).append(group)

    layer_indices = {
        group.layer_index for group in job.material_paths
    } | set(fiber_paths_by_layer)

    for layer_index in sorted(layer_indices):
        infill_paths: list[list[list[float]]] = []
        resin_paths: list[dict[str, object]] = []
        group_resin_index = 0
        for group in groups_by_layer.get(layer_index, {}).get("R", []):
            layer_roles = resin_roles_by_layer.get(str(layer_index), [])
            for path in group.paths:
                points = _simplify_preview_path(
                    [[float(point[0]), float(point[1]), float(point[2])] for point in path],
                    max_points=2000,
                )
                role = (
                    layer_roles[group_resin_index]
                    if isinstance(layer_roles, list) and group_resin_index < len(layer_roles)
                    else None
                )
                group_resin_index += 1
                if role in ("outer_contour", "inner_contour", "infill"):
                    resin_paths.append({"role": role, "points": points})
                    if role == "infill":
                        infill_paths.append(points)
                elif path.shape[0] > 2:
                    resin_paths.append({"role": "outer_contour", "points": points})
                else:
                    resin_paths.append({"role": "infill", "points": points})
                    infill_paths.append(points)
                for x, y, z in points:
                    _expand_bounds(bounds, x, y, z)

        outer_contours: list[list[list[float]]] = []
        inner_contours: list[list[list[float]]] = []
        for entry in resin_paths:
            if entry["role"] == "inner_contour":
                inner_contours.append(entry["points"])
            elif entry["role"] == "outer_contour":
                outer_contours.append(entry["points"])

        serialized_fiber_paths = [
            _simplify_preview_path(path, max_points=2000)
            for path in fiber_paths_by_layer.get(layer_index, [])
        ]
        if not serialized_fiber_paths:
            for group in groups_by_layer.get(layer_index, {}).get("F", []):
                serialized_fiber_paths.extend(
                    _simplify_preview_path(
                        [[float(point[0]), float(point[1]), float(point[2])] for point in path],
                        max_points=2000,
                    )
                    for path in group.paths
                )
        for fiber_path in serialized_fiber_paths:
            for x, y, z in fiber_path:
                _expand_bounds(bounds, x, y, z)

        layers_by_index[layer_index] = {
            "index": layer_index,
            "outer_contours": outer_contours,
            "inner_contours": inner_contours,
            "infill_paths": infill_paths,
            "resin_paths": resin_paths,
            "paths": [entry["points"] for entry in resin_paths],
            "fiber_paths": serialized_fiber_paths,
        }

    return {
        "bounds": bounds,
        "line_widths": {
            "resin": float(config.line_width),
            "fiber": DEFAULT_FIBER_LINE_WIDTH_MM,
        },
        "layers": list(layers_by_index.values()),
    }


def _triangle_infill_recommendation(mesh, config: SliceConfig, current_job) -> dict[str, object] | None:
    if config.infill_pattern != "triangles":
        return None

    current_density = float(config.infill_density)
    current_max_paths = _max_resin_infill_paths_per_layer(current_job)
    if current_density < 40.0:
        message = (
            "当前三角填充率偏低，三角形容易不成形。建议优先尝试 50%-70%。"
        )
        recommended_density = 50.0
    elif current_density > 75.0:
        message = "当前三角填充率较高，路径会明显增多。若允许更大间隙，可尝试 70%。"
        recommended_density = 70.0
    else:
        message = f"当前三角填充率可用；当前每层最多 {current_max_paths} 条填充路径。"
        recommended_density = current_density
    return {
        "recommended_density": recommended_density,
        "current_density": current_density,
        "current_max_infill_paths": current_max_paths,
        "recommended_max_infill_paths": current_max_paths,
        "message": message,
    }


def _max_resin_infill_paths_per_layer(job) -> int:
    roles_by_layer = job.meta.get("path_roles", {}).get("R", {})
    max_count = 0
    for group in job.material_paths:
        if group.material != "R":
            continue
        roles = roles_by_layer.get(str(group.layer_index), [])
        if isinstance(roles, list):
            max_count = max(max_count, sum(1 for role in roles if role == "infill"))
    return max_count


def _classify_contours(contours: list[np.ndarray]) -> list[str]:
    roles: list[str] = []
    for index, contour in enumerate(contours):
        if contour.shape[0] < 3:
            roles.append("outer_contour")
            continue
        centroid = np.mean(contour[:, :2], axis=0)
        containing_count = 0
        for other_index, other in enumerate(contours):
            if other_index == index or other.shape[0] < 3:
                continue
            if _point_in_polygon(float(centroid[0]), float(centroid[1]), other):
                containing_count += 1
        roles.append("inner_contour" if containing_count % 2 == 1 else "outer_contour")
    return roles


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


def _simplify_preview_path(
    points: list[list[float]], max_points: int
) -> list[list[float]]:
    if len(points) <= max_points:
        return points
    if max_points < 2:
        return points[:max_points]
    step = (len(points) - 1) / (max_points - 1)
    simplified = [points[round(index * step)] for index in range(max_points)]
    simplified[-1] = points[-1]
    return simplified


def _load_fiber_preview_paths(npz_path: Path) -> dict[int, list[list[list[float]]]]:
    paths_by_layer: dict[int, list[list[list[float]]]] = {}
    key_pattern = re.compile(r"^layer_(\d{4})_F$")
    with np.load(npz_path, allow_pickle=False) as archive:
        for key in archive.files:
            match = key_pattern.match(key)
            if not match:
                continue
            layer_index = int(match.group(1))
            array = np.asarray(archive[key], dtype=np.float32)
            if array.ndim != 3 or array.shape[2] not in (3, 6):
                raise ValueError(f"fiber NPZ key {key} must be a 3D array with 3 or 6 columns")
            layer_paths: list[list[list[float]]] = []
            for raw_path in array:
                valid_rows = []
                for row in raw_path:
                    nan_mask = np.isnan(row)
                    if nan_mask.all():
                        continue
                    if nan_mask.any():
                        raise ValueError(f"fiber NPZ key {key} contains partial-NaN padding")
                    valid_rows.append(row[:3])
                if len(valid_rows) >= 2:
                    layer_paths.append(
                        [[float(point[0]), float(point[1]), float(point[2])] for point in valid_rows]
                    )
            if layer_paths:
                paths_by_layer[layer_index] = layer_paths
    return paths_by_layer


def _slice_contours_for_preview(mesh, config: SliceConfig) -> dict[int, list]:
    contours_by_layer = {}
    z_values = _layer_z_values(mesh, config)
    for layer_index, base_z in enumerate(z_values):
        segments = _intersect_mesh_at_z(mesh.triangles, float(base_z), config.tolerance)
        contours_2d = _stitch_segments(segments, config.tolerance)
        contours_3d = []
        for contour in contours_2d:
            if contour.shape[0] < 2:
                continue
            points = []
            for x, y in contour:
                points.append([float(x), float(y), float(base_z)])
            contours_3d.append(points)
        if contours_3d:
            contours_by_layer[layer_index] = contours_3d
    return contours_by_layer


def _expand_bounds(bounds: dict[str, float | None], x: float, y: float, z: float) -> None:
    values = {
        "min_x": x,
        "max_x": x,
        "min_y": y,
        "max_y": y,
        "min_z": z,
        "max_z": z,
    }
    for key, value in values.items():
        if bounds[key] is None:
            bounds[key] = value
    bounds["min_x"] = min(bounds["min_x"], x)
    bounds["max_x"] = max(bounds["max_x"], x)
    bounds["min_y"] = min(bounds["min_y"], y)
    bounds["max_y"] = max(bounds["max_y"], y)
    bounds["min_z"] = min(bounds["min_z"], z)
    bounds["max_z"] = max(bounds["max_z"], z)


def _index_html() -> str:
    pyslm_defaults = PySLMConfig()
    pyslm_strategy_defaults = recommended_pyslm_strategy_defaults(
        DEFAULT_RESIN_LAYER_HEIGHT_MM,
        DEFAULT_RESIN_LINE_WIDTH_MM,
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KUKA Slicer</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5c6972;
      --line: #d8dde2;
      --panel: #f7f9fb;
      --accent: #0b6bcb;
      --accent-dark: #084f96;
      --ok: #16754c;
      --error: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      height: 64px;
      display: flex;
      align-items: center;
      padding: 0 28px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px;
      display: grid;
      grid-template-columns: minmax(320px, 440px) 1fr;
      gap: 28px;
    }}
    section {{
      min-width: 0;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 18px;
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 15px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .subhead {{
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }}
    .formSection {{
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
    }}
    .formSection:first-of-type {{
      margin-top: 0;
      padding-top: 0;
      border-top: 0;
    }}
    .formSection h3 {{
      margin: 0 0 4px;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0;
      color: var(--ink);
    }}
    label {{
      display: block;
      margin: 14px 0 6px;
      font-size: 13px;
      color: var(--muted);
    }}
    input, select {{
      width: 100%;
      min-height: 38px;
      border: 1px solid #bcc5cd;
      border-radius: 6px;
      padding: 7px 10px;
      font: inherit;
      background: #ffffff;
      color: var(--ink);
    }}
    input[type="file"] {{
      padding: 8px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .actions {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin-top: 18px;
    }}
    button {{
      min-height: 40px;
      border: 0;
      border-radius: 6px;
      padding: 0 16px;
      font: inherit;
      font-weight: 650;
      color: #ffffff;
      background: var(--accent);
      cursor: pointer;
    }}
    button:hover {{ background: var(--accent-dark); }}
    button:disabled {{ opacity: 0.55; cursor: wait; }}
    .status {{
      min-height: 20px;
      font-size: 13px;
      color: var(--muted);
    }}
    .status.ok {{ color: var(--ok); }}
    .status.error {{ color: var(--error); }}
    .notice {{
      margin-top: 8px;
      min-height: 18px;
      font-size: 13px;
      color: var(--muted);
    }}
    .notice.warning {{ color: #9a5b00; }}
    .advancedSettings {{
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }}
    .advancedSettings summary {{
      color: var(--muted);
      cursor: pointer;
      font-size: 13px;
      font-weight: 650;
      user-select: none;
    }}
    .advancedSettings[open] summary {{
      margin-bottom: 8px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: #ffffff;
    }}
    .metric span {{
      display: block;
      font-size: 12px;
      color: var(--muted);
    }}
    .metric strong {{
      display: block;
      margin-top: 6px;
      font-size: 22px;
      line-height: 1.1;
    }}
    .download {{
      margin-top: 16px;
      display: none;
      color: var(--accent-dark);
      font-weight: 650;
      text-decoration: none;
    }}
    .download.visible {{ display: inline-block; }}
    .viewerControls {{
      margin-top: 16px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }}
    .viewerControls label {{
      margin-top: 0;
    }}
    .rangeRow {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
    }}
    .rangeRow input {{
      padding: 0;
    }}
    .rangeRow output {{
      min-width: 48px;
      text-align: right;
      font-size: 13px;
      color: var(--muted);
    }}
    .legend {{
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px 18px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }}
    .legendItem {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
    }}
    .swatch {{
      width: 18px;
      height: 3px;
      border-radius: 999px;
      display: inline-block;
    }}
    .outerSwatch {{ background: #146c43; }}
    .innerSwatch {{ background: #7b2cbf; }}
    .infillSwatch {{ background: #0b6bcb; }}
    .fiberSwatch {{ background: #e66f00; }}
    .viewOptions {{
      margin-top: 12px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }}
    .viewOptions label {{
      margin: 0;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
    }}
    .viewOptions input {{
      width: auto;
      min-height: 0;
      padding: 0;
    }}
    .checkboxLabel {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .checkboxLabel input {{
      width: auto;
      min-height: 0;
      padding: 0;
    }}
    .preview {{
      margin-top: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      height: 360px;
      background:
        linear-gradient(90deg, rgba(23,32,38,.08) 1px, transparent 1px),
        linear-gradient(0deg, rgba(23,32,38,.08) 1px, transparent 1px),
        #ffffff;
      background-size: 28px 28px;
      position: relative;
      overflow: visible;
    }}
    .preview canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    @media (max-width: 820px) {{
      main {{ grid-template-columns: 1fr; padding: 18px; }}
      .summary {{ grid-template-columns: 1fr; }}
      .viewerControls {{ grid-template-columns: 1fr; }}
      header {{ padding: 0 18px; }}
    }}
  </style>
</head>
<body>
  <header><h1>KUKA Slicer</h1></header>
  <main>
    <section class="panel">
      <h2>树脂切片</h2>
      <form id="sliceForm">
        <div class="formSection">
          <h3>输入文件</h3>
          <label for="stlFile">STL 文件</label>
          <input id="stlFile" name="stlFile" type="file" accept=".stl" required>

          <label for="fiberJsonFile">纤维路径 JSON</label>
          <input id="fiberJsonFile" name="fiberJsonFile" type="file" accept=".json,application/json">
          <div id="fiberNotice" class="notice"></div>
        </div>

        <div class="formSection">
          <h3>模型与分层</h3>
          <div class="grid">
            <div>
              <label for="layerHeight">树脂层高 mm</label>
              <input id="layerHeight" name="layerHeight" type="number" min="0.001" step="0.001" value="{DEFAULT_RESIN_LAYER_HEIGHT_MM}">
            </div>
            <div>
              <label for="buildAxis">层高方向</label>
              <select id="buildAxis" name="buildAxis">
                <option value="auto" selected>自动</option>
                <option value="y">Y 轴</option>
                <option value="z">Z 轴</option>
                <option value="x">X 轴</option>
              </select>
            </div>
          </div>
          <div class="grid">
            <div>
              <label for="zMin">起始 Z mm</label>
              <input id="zMin" name="zMin" type="number" step="0.001" placeholder="自动">
            </div>
            <div>
              <label for="zMax">结束 Z mm</label>
              <input id="zMax" name="zMax" type="number" step="0.001" placeholder="自动">
            </div>
          </div>
          <label for="tolerance">几何容差 mm</label>
          <input id="tolerance" name="tolerance" type="number" min="0.000001" step="0.000001" placeholder="自动">
        </div>

        <div class="formSection">
          <h3>树脂路径内核</h3>
          <div class="grid">
            <div>
              <label for="lineWidth">树脂线宽 mm</label>
              <input id="lineWidth" name="lineWidth" type="number" min="0.001" step="0.001" value="{DEFAULT_RESIN_LINE_WIDTH_MM}">
            </div>
            <div>
              <label for="perimeterCount">边界圈数</label>
              <input id="perimeterCount" name="perimeterCount" type="number" min="1" step="1" value="2">
            </div>
          </div>

          <label for="slicingKernel">切片内核</label>
          <select id="slicingKernel" name="slicingKernel">
            <option value="legacy" selected>原始内核（稳定）</option>
            <option value="pyslm">PySLM（实验）</option>
          </select>

          <div id="pyslmNativeSettings" hidden>
            <h3>PySLM 原生扫描参数</h3>
            <div class="grid">
              <div>
                <label for="pyslmHatcher" title="决定 PySLM 使用基础、条带或岛状扫描组织方式。">PySLM 填充策略</label>
                <select id="pyslmHatcher" name="pyslmHatcher">
                  <option value="basic" selected>基础直线填充</option>
                  <option value="stripe">条带填充</option>
                  <option value="island">岛状填充</option>
                  <option value="basic_island">基础岛状填充</option>
                </select>
              </div>
              <div>
                <label for="pyslmHatchSort">扫描线排序</label>
                <select id="pyslmHatchSort" name="pyslmHatchSort">
                  <option value="none" selected>PySLM 默认</option>
                  <option value="alternate">交替方向</option>
                  <option value="unidirectional">单向扫描</option>
                  <option value="linear">线性排序</option>
                  <option value="directional">方向交替</option>
                </select>
              </div>
            </div>
            <div class="grid">
              <div>
                <label for="pyslmHatchAngle">填充角度 °</label>
                <input id="pyslmHatchAngle" name="pyslmHatchAngle" type="number" min="-180" max="180" step="0.1" placeholder="自动">
              </div>
              <div>
                <label for="pyslmLayerAngleIncrement">层间角度增量 °</label>
                <input id="pyslmLayerAngleIncrement" name="pyslmLayerAngleIncrement" type="number" step="0.1" value="{pyslm_defaults.layer_angle_increment:g}">
              </div>
            </div>
            <div class="grid">
              <div>
                <label for="pyslmHatchDistance">填充线间距 mm</label>
                <input id="pyslmHatchDistance" name="pyslmHatchDistance" type="number" min="0" step="0.001" placeholder="自动">
              </div>
              <div>
                <label for="pyslmContourOffset">轮廓偏移 mm</label>
                <input id="pyslmContourOffset" name="pyslmContourOffset" type="number" min="0" step="0.001" placeholder="自动">
              </div>
            </div>
            <div class="grid">
              <div>
                <label for="pyslmSpotCompensation">光斑补偿 mm</label>
                <input id="pyslmSpotCompensation" name="pyslmSpotCompensation" type="number" min="0" step="0.001" placeholder="自动">
              </div>
              <div>
                <label for="pyslmVolumeOffset">体积填充偏移 mm</label>
                <input id="pyslmVolumeOffset" name="pyslmVolumeOffset" type="number" step="0.001" placeholder="自动">
              </div>
            </div>
            <div class="grid">
              <div>
                <label for="pyslmOuterContours">外轮廓数量</label>
                <input id="pyslmOuterContours" name="pyslmOuterContours" type="number" min="0" step="1" placeholder="自动">
              </div>
              <div>
                <label for="pyslmInnerContours">内轮廓数量</label>
                <input id="pyslmInnerContours" name="pyslmInnerContours" type="number" min="0" step="1" placeholder="自动">
              </div>
            </div>
            <details id="pyslmPatternSettings" class="advancedSettings">
              <summary>条带/岛状参数（自动）</summary>
              <label class="checkboxLabel" title="启用后根据树脂层高和线宽重新计算下面的推荐值。"><input id="pyslmPatternAuto" type="checkbox" checked> 自动设置条带/岛状参数</label>
              <div class="grid">
              <div>
                <label for="pyslmStripeWidth">条带宽度 mm</label>
                <input id="pyslmStripeWidth" name="pyslmStripeWidth" type="number" min="0.001" step="0.1" value="{pyslm_strategy_defaults.width:g}">
              </div>
              <div>
                <label for="pyslmStripeOverlap">条带重叠 mm</label>
                <input id="pyslmStripeOverlap" name="pyslmStripeOverlap" type="number" min="0" step="0.1" value="{pyslm_strategy_defaults.overlap:g}">
              </div>
              </div>
              <div class="grid">
              <div>
                <label for="pyslmStripeOffset">条带平移系数</label>
                <input id="pyslmStripeOffset" name="pyslmStripeOffset" type="number" min="0" step="0.05" value="{pyslm_strategy_defaults.offset:g}">
              </div>
              <div>
                <label for="pyslmIslandWidth">岛状宽度 mm</label>
                <input id="pyslmIslandWidth" name="pyslmIslandWidth" type="number" min="0.001" step="0.1" value="{pyslm_strategy_defaults.width:g}">
              </div>
              </div>
              <div class="grid">
              <div>
                <label for="pyslmIslandOverlap">岛状重叠 mm</label>
                <input id="pyslmIslandOverlap" name="pyslmIslandOverlap" type="number" min="0" step="0.1" value="{pyslm_strategy_defaults.overlap:g}">
              </div>
              <div>
                <label for="pyslmIslandOffset">岛状平移系数</label>
                <input id="pyslmIslandOffset" name="pyslmIslandOffset" type="number" min="0" step="0.05" value="{pyslm_strategy_defaults.offset:g}">
              </div>
              </div>
            </details>
            <div class="grid">
              <div>
                <label for="pyslmSimplificationFactor">切层边界简化 mm</label>
                <input id="pyslmSimplificationFactor" name="pyslmSimplificationFactor" type="number" min="0" step="0.001" placeholder="关闭">
              </div>
              <div>
                <label for="pyslmSimplificationMode">简化模式</label>
                <select id="pyslmSimplificationMode" name="pyslmSimplificationMode">
                  <option value="absolute" selected>绝对距离</option>
                  <option value="line">按线宽</option>
                </select>
              </div>
            </div>
            <label class="checkboxLabel"><input id="pyslmScanContourFirst" type="checkbox" checked> 轮廓优先扫描</label>
            <label class="checkboxLabel"><input id="pyslmFixPolygons" type="checkbox" checked> 修复切层多边形</label>
            <label class="checkboxLabel"><input id="pyslmSimplificationPreserveTopology" type="checkbox" checked> 保持拓扑结构</label>
          </div>

          <div id="legacyInfillControl">
          <label for="infillPattern" title="仅原始内核使用；PySLM 模式由上方的原生填充策略决定。">原始内核填充路径</label>
          <select id="infillPattern" name="infillPattern">
            <option value="none">仅轮廓</option>
            <option value="rectilinear">交替直线填充</option>
            <option value="aligned_rectilinear">对齐直线填充</option>
            <option value="line">单向线填充</option>
            <option value="grid">网格填充</option>
            <option value="triangles">三角形填充</option>
            <option value="gyroid">陀螺曲线填充</option>
            <option value="concentric">同心轮廓填充</option>
            <option value="zigzag">之字形填充</option>
          </select>
          </div>

          <div class="grid">
            <div>
              <label for="infillDensity">填充率 %</label>
              <input id="infillDensity" name="infillDensity" type="number" min="0" max="100" step="1" value="{DEFAULT_RESIN_INFILL_DENSITY_PERCENT:g}">
            </div>
            <div>
              <label for="infillOverlap">填充搭边 %</label>
              <input id="infillOverlap" name="infillOverlap" type="number" min="0" max="99" step="1" value="{DEFAULT_RESIN_INFILL_OVERLAP_PERCENT:g}">
            </div>
          </div>

          <div class="grid">
            <div>
              <label for="smoothingAngle">平滑角阈值 °</label>
              <input id="smoothingAngle" name="smoothingAngle" type="number" min="1" max="179" step="1" value="{DEFAULT_RESIN_SMOOTHING_ANGLE_DEGREES:g}">
            </div>
            <div>
              <label for="smoothingRadiusFactor">平滑半径系数</label>
              <input id="smoothingRadiusFactor" name="smoothingRadiusFactor" type="number" min="0" step="0.01" value="{DEFAULT_RESIN_SMOOTHING_RADIUS_FACTOR:g}">
            </div>
          </div>

        </div>

        <div class="formSection">
          <h3>筏板</h3>
        <div class="grid">
          <div>
            <label for="raftLayerCount">筏板层数</label>
            <input id="raftLayerCount" name="raftLayerCount" type="number" min="0" step="1" value="2">
          </div>
          <div>
            <label for="raftTopGap">筏板顶层间隙 mm</label>
            <input id="raftTopGap" name="raftTopGap" type="number" min="0" step="0.001" value="0">
          </div>
        </div>

        <label for="raftOffsets">每层外扩距离 mm</label>
        <input id="raftOffsets" name="raftOffsets" type="text" value="15,10" placeholder="单值或逗号分隔，例如 8,6,4">

        <label for="raftLayerHeights">每层筏板层高 mm</label>
        <input id="raftLayerHeights" name="raftLayerHeights" type="text" value="0.5" placeholder="单值或逗号分隔，例如 0.3,0.25,0.2">

        <label for="raftInfillDensities">每层筏板填充率 %</label>
        <input id="raftInfillDensities" name="raftInfillDensities" type="text" value="100,70" placeholder="单值或逗号分隔，例如 80,70,60">

        <label for="raftInfillPatterns">每层筏板填充策略</label>
        <input id="raftInfillPatterns" name="raftInfillPatterns" type="text" value="之字形,之字形" placeholder="同心轮廓或之字形，单值或逗号分隔">
        </div>

        <div class="formSection">
          <h3>曲面 Z</h3>
        <label for="curveMode">Z 模式</label>
        <select id="curveMode" name="curveMode">
          <option value="flat">平面层</option>
          <option value="sinusoidal">正弦曲面</option>
        </select>

        <div class="grid">
          <div>
            <label for="curveAmplitude">曲面幅值 mm</label>
            <input id="curveAmplitude" name="curveAmplitude" type="number" step="0.001" value="0">
          </div>
          <div>
            <label for="curvePeriod">曲面周期 mm</label>
            <input id="curvePeriod" name="curvePeriod" type="number" min="0.001" step="0.001" value="50">
          </div>
        </div>
        </div>

        <div class="actions">
          <button id="sliceButton" type="submit">生成 NPZ</button>
          <span id="status" class="status"></span>
        </div>
      </form>
    </section>

    <section>
      <div class="summary">
        <div class="metric"><span>层数</span><strong id="layers">-</strong></div>
        <div class="metric"><span>路径数</span><strong id="paths">-</strong></div>
        <div class="metric"><span>输出</span><strong id="outputName">-</strong></div>
      </div>
      <a id="download" class="download" href="#">下载 NPZ</a>
      <div class="viewerControls">
        <div>
          <label for="layerSlider">层</label>
          <div class="rangeRow">
            <input id="layerSlider" type="range" min="0" max="0" value="0" disabled>
            <output id="layerLabel">-</output>
          </div>
        </div>
        <div>
          <label for="resinPathSlider">树脂路径进度</label>
          <div class="rangeRow">
            <input id="resinPathSlider" type="range" min="0" max="0" value="0" disabled>
            <output id="resinPathLabel">-</output>
          </div>
        </div>
        <div>
          <label for="fiberPathSlider">纤维路径进度</label>
          <div class="rangeRow">
            <input id="fiberPathSlider" type="range" min="0" max="0" value="0" disabled>
            <output id="fiberPathLabel">-</output>
          </div>
        </div>
      </div>
      <div class="legend" aria-label="预览图例">
        <span class="legendItem"><span class="swatch outerSwatch"></span>外轮廓</span>
        <span class="legendItem"><span class="swatch innerSwatch"></span>内轮廓</span>
        <span class="legendItem"><span class="swatch infillSwatch"></span>树脂填充</span>
        <span class="legendItem"><span class="swatch fiberSwatch"></span>纤维路径</span>
      </div>
      <div class="viewOptions" aria-label="显示选项">
        <label><input id="showLineWidth" type="checkbox">按线宽显示</label>
        <label><input id="showPathPoints" type="checkbox">显示当前路径点</label>
        <label><input id="showDirection" type="checkbox" checked>显示打印方向</label>
      </div>
      <div class="preview" aria-hidden="true"><canvas id="previewCanvas"></canvas></div>
    </section>
  </main>

  <script>
    const form = document.getElementById('sliceForm');
    const button = document.getElementById('sliceButton');
    const statusEl = document.getElementById('status');
    const downloadEl = document.getElementById('download');
    const layersEl = document.getElementById('layers');
    const pathsEl = document.getElementById('paths');
    const outputNameEl = document.getElementById('outputName');
    const previewCanvas = document.getElementById('previewCanvas');
    const layerSlider = document.getElementById('layerSlider');
    const resinPathSlider = document.getElementById('resinPathSlider');
    const fiberPathSlider = document.getElementById('fiberPathSlider');
    const layerLabel = document.getElementById('layerLabel');
    const resinPathLabel = document.getElementById('resinPathLabel');
    const fiberPathLabel = document.getElementById('fiberPathLabel');
    const stlFileInput = document.getElementById('stlFile');
    const fiberJsonInput = document.getElementById('fiberJsonFile');
    const fiberNotice = document.getElementById('fiberNotice');
    const showLineWidthInput = document.getElementById('showLineWidth');
    const showPathPointsInput = document.getElementById('showPathPoints');
    const showDirectionInput = document.getElementById('showDirection');
    const slicingKernelInput = document.getElementById('slicingKernel');
    const layerHeightInput = document.getElementById('layerHeight');
    const lineWidthInput = document.getElementById('lineWidth');
    const legacyInfillControl = document.getElementById('legacyInfillControl');
    const infillPatternInput = document.getElementById('infillPattern');
    const pyslmNativeSettings = document.getElementById('pyslmNativeSettings');
    const pyslmHatcherInput = document.getElementById('pyslmHatcher');
    const pyslmPatternAutoInput = document.getElementById('pyslmPatternAuto');
    const stripeParameterIds = ['pyslmStripeWidth', 'pyslmStripeOverlap', 'pyslmStripeOffset'];
    const islandParameterIds = ['pyslmIslandWidth', 'pyslmIslandOverlap', 'pyslmIslandOffset'];
    const pyslmNativePatterns = new Set(['none', 'line', 'aligned_rectilinear', 'rectilinear', 'zigzag']);
    const pyslmSettingsIds = [
      'pyslmHatcher', 'pyslmHatchSort', 'pyslmHatchAngle', 'pyslmLayerAngleIncrement',
      'pyslmHatchDistance', 'pyslmContourOffset', 'pyslmSpotCompensation',
      'pyslmVolumeOffset', 'pyslmOuterContours', 'pyslmInnerContours',
      'pyslmStripeWidth', 'pyslmStripeOverlap', 'pyslmStripeOffset',
      'pyslmIslandWidth', 'pyslmIslandOverlap', 'pyslmIslandOffset',
      'pyslmSimplificationFactor', 'pyslmSimplificationMode',
      'pyslmScanContourFirst', 'pyslmFixPolygons', 'pyslmSimplificationPreserveTopology'
    ];
    let previewData = null;
    function updatePyslmStrategyDefaults() {{
      if (!pyslmPatternAutoInput.checked) return;
      const layerHeight = Number(layerHeightInput.value);
      const lineWidth = Number(lineWidthInput.value);
      if (!(layerHeight > 0) || !(lineWidth > 0)) return;
      const width = Math.max(lineWidth * 5.0, layerHeight * 10.0);
      const overlap = Math.min(0.1, lineWidth * 0.05, layerHeight * 0.2);
      for (const id of ['pyslmStripeWidth', 'pyslmIslandWidth']) {{
        document.getElementById(id).value = width.toFixed(3).replace(/\\.0+$/, '').replace(/(\\.\\d*?)0+$/, '$1');
      }}
      for (const id of ['pyslmStripeOverlap', 'pyslmIslandOverlap']) {{
        document.getElementById(id).value = overlap.toFixed(3).replace(/\\.0+$/, '').replace(/(\\.\\d*?)0+$/, '$1');
      }}
      for (const id of ['pyslmStripeOffset', 'pyslmIslandOffset']) {{
        document.getElementById(id).value = '0.5';
      }}
    }}
    function syncKernelControls() {{
      const isPyslm = slicingKernelInput.value === 'pyslm';
      pyslmNativeSettings.hidden = !isPyslm;
      legacyInfillControl.hidden = isPyslm;
      infillPatternInput.disabled = isPyslm;
      for (const option of infillPatternInput.options) {{
        option.disabled = isPyslm && !pyslmNativePatterns.has(option.value);
      }}
      if (isPyslm && !pyslmNativePatterns.has(infillPatternInput.value)) {{
        infillPatternInput.value = 'rectilinear';
      }}
      for (const id of pyslmSettingsIds) {{
        document.getElementById(id).disabled = !isPyslm;
      }}
      pyslmPatternAutoInput.disabled = !isPyslm;
      updatePyslmStrategyDefaults();
      const strategy = pyslmHatcherInput.value;
      const stripeEnabled = isPyslm && strategy === 'stripe';
      const islandEnabled = isPyslm && (strategy === 'island' || strategy === 'basic_island');
      for (const id of stripeParameterIds) {{
        document.getElementById(id).disabled = !stripeEnabled || pyslmPatternAutoInput.checked;
      }}
      for (const id of islandParameterIds) {{
        document.getElementById(id).disabled = !islandEnabled || pyslmPatternAutoInput.checked;
      }}
    }}
    slicingKernelInput.addEventListener('change', syncKernelControls);
    infillPatternInput.addEventListener('change', syncKernelControls);
    pyslmHatcherInput.addEventListener('change', syncKernelControls);
    pyslmPatternAutoInput.addEventListener('change', syncKernelControls);
    layerHeightInput.addEventListener('input', syncKernelControls);
    lineWidthInput.addEventListener('input', syncKernelControls);
    syncKernelControls();
    fiberNotice.textContent = 'JSON 中的单层纤维路径会复制到每个树脂层，最后一层树脂封顶不打印纤维。';

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      const file = stlFileInput.files[0];
      const fiberFile = fiberJsonInput.files[0];
      if (!file) return;

      button.disabled = true;
      button.textContent = '处理中';
      statusEl.textContent = '处理中';
      statusEl.className = 'status';
      downloadEl.className = 'download';

      const formData = new FormData();
      formData.append('stl_file', file, file.name);
      if (fiberFile) formData.append('fiber_json', fiberFile, fiberFile.name);
      formData.append('filename', file.name);
      formData.append('layer_height', document.getElementById('layerHeight').value);
      formData.append('line_width', document.getElementById('lineWidth').value);
      formData.append('build_axis', document.getElementById('buildAxis').value);
      formData.append('z_min', document.getElementById('zMin').value);
      formData.append('z_max', document.getElementById('zMax').value);
      formData.append('tolerance', document.getElementById('tolerance').value);
      formData.append('perimeter_count', document.getElementById('perimeterCount').value);
      formData.append('infill_pattern', document.getElementById('infillPattern').value);
      formData.append('infill_density', document.getElementById('infillDensity').value);
      formData.append('infill_overlap', document.getElementById('infillOverlap').value);
      formData.append('slicing_kernel', document.getElementById('slicingKernel').value);
      formData.append('pyslm_hatcher', document.getElementById('pyslmHatcher').value);
      formData.append('pyslm_hatch_sort', document.getElementById('pyslmHatchSort').value);
      formData.append('pyslm_hatch_angle', document.getElementById('pyslmHatchAngle').value);
      formData.append('pyslm_layer_angle_increment', document.getElementById('pyslmLayerAngleIncrement').value);
      formData.append('pyslm_hatch_distance', document.getElementById('pyslmHatchDistance').value);
      formData.append('pyslm_contour_offset', document.getElementById('pyslmContourOffset').value);
      formData.append('pyslm_spot_compensation', document.getElementById('pyslmSpotCompensation').value);
      formData.append('pyslm_volume_offset_hatch', document.getElementById('pyslmVolumeOffset').value);
      formData.append('pyslm_num_outer_contours', document.getElementById('pyslmOuterContours').value);
      formData.append('pyslm_num_inner_contours', document.getElementById('pyslmInnerContours').value);
      formData.append('pyslm_stripe_width', document.getElementById('pyslmStripeWidth').value);
      formData.append('pyslm_stripe_overlap', document.getElementById('pyslmStripeOverlap').value);
      formData.append('pyslm_stripe_offset', document.getElementById('pyslmStripeOffset').value);
      formData.append('pyslm_island_width', document.getElementById('pyslmIslandWidth').value);
      formData.append('pyslm_island_overlap', document.getElementById('pyslmIslandOverlap').value);
      formData.append('pyslm_island_offset', document.getElementById('pyslmIslandOffset').value);
      formData.append('pyslm_fix_polygons', document.getElementById('pyslmFixPolygons').checked ? 'true' : 'false');
      formData.append('pyslm_scan_contour_first', document.getElementById('pyslmScanContourFirst').checked ? 'true' : 'false');
      formData.append('pyslm_simplification_factor', document.getElementById('pyslmSimplificationFactor').value);
      formData.append('pyslm_simplification_mode', document.getElementById('pyslmSimplificationMode').value);
      formData.append('pyslm_simplification_preserve_topology', document.getElementById('pyslmSimplificationPreserveTopology').checked ? 'true' : 'false');
      formData.append('smoothing_angle', document.getElementById('smoothingAngle').value);
      formData.append('smoothing_radius_factor', document.getElementById('smoothingRadiusFactor').value);
      formData.append('raft_layer_count', document.getElementById('raftLayerCount').value);
      formData.append('raft_top_gap', document.getElementById('raftTopGap').value);
      formData.append('raft_offsets', document.getElementById('raftOffsets').value);
      formData.append('raft_layer_heights', document.getElementById('raftLayerHeights').value);
      formData.append('raft_infill_densities', document.getElementById('raftInfillDensities').value);
      formData.append('raft_infill_patterns', document.getElementById('raftInfillPatterns').value);
      formData.append('curve_mode', document.getElementById('curveMode').value);
      formData.append('curve_amplitude', document.getElementById('curveAmplitude').value);
      formData.append('curve_period', document.getElementById('curvePeriod').value);

      try {{
        const response = await fetch('/slice', {{
          method: 'POST',
          body: formData
        }});
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || '切片失败');

        layersEl.textContent = result.layers;
        pathsEl.textContent = result.paths;
        outputNameEl.textContent = result.filename;
        previewData = result.preview;
        configureViewer();
        drawPreview();
        downloadEl.href = result.download_url;
        downloadEl.textContent = '下载 ' + result.filename;
        downloadEl.className = 'download visible';
        statusEl.textContent = result.recommendation?.message
          ? '完成。' + result.recommendation.message
          : '完成';
        statusEl.className = 'status ok';
      }} catch (error) {{
        statusEl.textContent = error.message;
        statusEl.className = 'status error';
      }} finally {{
        button.disabled = false;
        button.textContent = '生成 NPZ';
      }}
    }});

    function configureViewer() {{
      const layers = previewData?.layers || [];
      layerSlider.disabled = layers.length === 0;
      layerSlider.max = Math.max(0, layers.length - 1);
      layerSlider.value = layers.length ? 0 : 0;
      updatePathSlider();
    }}

    function updatePathSlider() {{
      const layer = currentLayer();
      const resinPathCount = layer ? layer.paths.length : 0;
      const fiberPathCount = layer ? (layer.fiber_paths || []).length : 0;
      resinPathSlider.disabled = resinPathCount === 0;
      resinPathSlider.max = resinPathCount;
      resinPathSlider.value = resinPathCount;
      fiberPathSlider.disabled = fiberPathCount === 0;
      fiberPathSlider.max = fiberPathCount;
      fiberPathSlider.value = fiberPathCount;
      updateViewerLabels();
    }}

    function currentLayer() {{
      const layers = previewData?.layers || [];
      return layers[Number(layerSlider.value)] || null;
    }}

    function updateViewerLabels() {{
      const layer = currentLayer();
      const layerCount = previewData?.layers?.length || 0;
      const resinPathCount = layer ? layer.paths.length : 0;
      const fiberPathCount = layer ? (layer.fiber_paths || []).length : 0;
      layerLabel.textContent = layer ? `${{Number(layerSlider.value) + 1}} / ${{layerCount}}` : '-';
      resinPathLabel.textContent = resinPathCount ? `${{resinPathSlider.value}} / ${{resinPathCount}}` : '-';
      fiberPathLabel.textContent = fiberPathCount ? `${{fiberPathSlider.value}} / ${{fiberPathCount}}` : '-';
    }}

    function drawPreview() {{
      const canvas = previewCanvas;
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * scale));
      canvas.height = Math.max(1, Math.floor(rect.height * scale));
      const ctx = canvas.getContext('2d');
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const layer = currentLayer();
      if (!previewData || !layer) {{
        drawEmptyPreview(ctx, rect.width, rect.height);
        updateViewerLabels();
        return;
      }}

      const bounds = previewData.bounds;
      const minX = bounds.min_x;
      const maxX = bounds.max_x;
      const minY = bounds.min_y;
      const maxY = bounds.max_y;
      if ([minX, maxX, minY, maxY].some((value) => value === null || value === undefined)) {{
        drawEmptyPreview(ctx, rect.width, rect.height);
        updateViewerLabels();
        return;
      }}
      const spanX = Math.max(0.001, maxX - minX);
      const spanY = Math.max(0.001, maxY - minY);
      const lineWidths = previewData.line_widths || {{ resin: 2.0, fiber: 1.0 }};
      const usePhysicalWidth = showLineWidthInput.checked;
      const previewLineWidth = Math.max(Number(lineWidths.resin || 2.0), Number(lineWidths.fiber || 1.0));
      const initialFit = Math.min((rect.width - 48) / spanX, (rect.height - 48) / spanY);
      const margin = usePhysicalWidth
        ? Math.max(24, previewLineWidth * initialFit * 0.5 + 10)
        : 24;
      const x = margin;
      const y = margin;
      const w = Math.max(1, rect.width - margin * 2);
      const h = Math.max(1, rect.height - margin * 2);
      const fit = Math.min(w / spanX, h / spanY);
      const offsetX = x + (w - spanX * fit) / 2;
      const offsetY = y + (h - spanY * fit) / 2;
      const resinStrokeWidth = usePhysicalWidth ? Math.max(1.2, Number(lineWidths.resin || 2.0) * fit) : 1.7;
      const fiberStrokeWidth = usePhysicalWidth ? Math.max(1.0, Number(lineWidths.fiber || 1.0) * fit) : 2.0;

      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      function project(point) {{
        return [
          offsetX + (point[0] - minX) * fit,
          offsetY + (maxY - point[1]) * fit
        ];
      }}

      const resinEntries = layer.resin_paths || (layer.paths || []).map((points) => ({{ role: 'infill', points }}));
      const visiblePaths = Math.min(Number(resinPathSlider.value), resinEntries.length);
      let currentResinPath = null;
      let currentResinColor = '#0b6bcb';
      for (let pathIndex = 0; pathIndex < resinEntries.length; pathIndex++) {{
        const entry = resinEntries[pathIndex];
        const path = entry.points || entry;
        if (!path || path.length < 2) continue;
        const isContour = entry.role === 'outer_contour' || entry.role === 'inner_contour';
        if (!isContour && pathIndex >= visiblePaths) continue;
        if (entry.role === 'outer_contour') {{
          ctx.strokeStyle = '#146c43';
          if (pathIndex === visiblePaths - 1) currentResinColor = '#146c43';
          ctx.lineWidth = resinStrokeWidth;
        }} else if (entry.role === 'inner_contour') {{
          ctx.strokeStyle = '#7b2cbf';
          if (pathIndex === visiblePaths - 1) currentResinColor = '#7b2cbf';
          ctx.lineWidth = resinStrokeWidth;
        }} else {{
          ctx.strokeStyle = '#0b6bcb';
          if (pathIndex === visiblePaths - 1) currentResinColor = '#0b6bcb';
          ctx.lineWidth = resinStrokeWidth;
        }}
        if (pathIndex === visiblePaths - 1) currentResinPath = path;
        const first = project(path[0]);
        ctx.beginPath();
        ctx.moveTo(first[0], first[1]);
        for (let pointIndex = 1; pointIndex < path.length; pointIndex++) {{
          const point = project(path[pointIndex]);
          ctx.lineTo(point[0], point[1]);
        }}
        const last = path[path.length - 1];
        if (path.length > 2 && Math.abs(last[0] - path[0][0]) < 0.001 && Math.abs(last[1] - path[0][1]) < 0.001) {{
          ctx.closePath();
        }}
        ctx.stroke();
      }}

      const fiberPaths = layer.fiber_paths || [];
      const visibleFiberPaths = Math.min(Number(fiberPathSlider.value), fiberPaths.length);
      let currentFiberPath = null;
      for (let fiberIndex = 0; fiberIndex < visibleFiberPaths; fiberIndex++) {{
        const path = fiberPaths[fiberIndex];
        if (!path || path.length < 2) continue;
        if (fiberIndex === visibleFiberPaths - 1) currentFiberPath = path;
        ctx.strokeStyle = '#e66f00';
        ctx.lineWidth = fiberStrokeWidth;
        const first = project(path[0]);
        ctx.beginPath();
        ctx.moveTo(first[0], first[1]);
        for (let pointIndex = 1; pointIndex < path.length; pointIndex++) {{
          const point = project(path[pointIndex]);
          ctx.lineTo(point[0], point[1]);
        }}
        ctx.stroke();
      }}
      if (showPathPointsInput.checked) {{
        drawPathPoints(ctx, currentResinPath, currentResinColor, project);
        drawPathPoints(ctx, currentFiberPath, '#e66f00', project);
      }}
      if (showDirectionInput.checked) {{
        drawDirection(ctx, currentResinPath, currentResinColor, project);
        drawDirection(ctx, currentFiberPath, '#e66f00', project);
      }}
      ctx.restore();
      updateViewerLabels();
    }}

    function drawPathPoints(ctx, path, color, project) {{
      if (!path || path.length < 1) return;
      ctx.save();
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      const sampleSpacing = 12;
      let lastDrawn = null;

      function drawPoint(point, radius) {{
        if (lastDrawn && Math.hypot(point[0] - lastDrawn[0], point[1] - lastDrawn[1]) < 4) return;
        ctx.beginPath();
        ctx.arc(point[0], point[1], radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        lastDrawn = point;
      }}

      const first = project(path[0]);
      drawPoint(first, 4.5);
      for (let index = 0; index < path.length - 1; index++) {{
        const a = project(path[index]);
        const b = project(path[index + 1]);
        const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
        const sampleCount = Math.max(1, Math.floor(length / sampleSpacing));
        for (let sampleIndex = 1; sampleIndex <= sampleCount; sampleIndex++) {{
          const t = sampleIndex / sampleCount;
          drawPoint([
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t
          ], 3.0);
        }}
      }}
      ctx.restore();
    }}

    function drawDirection(ctx, path, color, project) {{
      if (!path || path.length < 2) return;
      let totalLength = 0;
      const segments = [];
      for (let index = 0; index < path.length - 1; index++) {{
        const a = project(path[index]);
        const b = project(path[index + 1]);
        const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
        if (length > 0.5) segments.push({{ a, b, length, startLength: totalLength }});
        totalLength += length;
      }}
      if (!segments.length || totalLength <= 0) return;
      const target = totalLength * 0.5;
      const segment = segments.find((item) => item.startLength + item.length >= target) || segments[0];
      const angle = Math.atan2(segment.b[1] - segment.a[1], segment.b[0] - segment.a[0]);
      const arrowLength = Math.min(56, Math.max(28, segment.length * 0.8));
      const centerOffset = Math.min(Math.max(target - segment.startLength, 0), segment.length);
      const center = [
        segment.a[0] + Math.cos(angle) * centerOffset,
        segment.a[1] + Math.sin(angle) * centerOffset
      ];
      const start = [
        center[0] - Math.cos(angle) * arrowLength * 0.45,
        center[1] - Math.sin(angle) * arrowLength * 0.45
      ];
      const tip = [
        start[0] + Math.cos(angle) * arrowLength,
        start[1] + Math.sin(angle) * arrowLength
      ];
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 9;
      ctx.beginPath();
      ctx.moveTo(start[0], start[1]);
      ctx.lineTo(tip[0], tip[1]);
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.moveTo(tip[0], tip[1]);
      ctx.lineTo(tip[0] - Math.cos(angle - 0.58) * 17, tip[1] - Math.sin(angle - 0.58) * 17);
      ctx.lineTo(tip[0] - Math.cos(angle + 0.58) * 17, tip[1] - Math.sin(angle + 0.58) * 17);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(start[0], start[1]);
      ctx.lineTo(tip[0], tip[1]);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(tip[0], tip[1]);
      ctx.lineTo(tip[0] - Math.cos(angle - 0.58) * 14, tip[1] - Math.sin(angle - 0.58) * 14);
      ctx.lineTo(tip[0] - Math.cos(angle + 0.58) * 14, tip[1] - Math.sin(angle + 0.58) * 14);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }}

    function drawEmptyPreview(ctx, width, height) {{
    }}

    layerSlider.addEventListener('input', () => {{
      updatePathSlider();
      drawPreview();
    }});
    resinPathSlider.addEventListener('input', drawPreview);
    fiberPathSlider.addEventListener('input', drawPreview);
    showLineWidthInput.addEventListener('change', drawPreview);
    showPathPointsInput.addEventListener('change', drawPreview);
    showDirectionInput.addEventListener('change', drawPreview);
    window.addEventListener('resize', drawPreview);
    drawPreview();
  </script>
</body>
</html>"""
