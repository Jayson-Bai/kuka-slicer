from __future__ import annotations

from datetime import datetime
from dataclasses import asdict, replace
from email import policy
from email.parser import BytesParser
import html
import importlib
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from uuid import uuid4
import zipfile

import numpy as np

from .external_npz import (
    ExternalSourceJob,
    MaterialPaths,
    TravelPaths,
    write_external_source_npz,
)
from .cpu_limiter import limit_slicer_task
from .fiber_travel import plan_fiber_interpath_travels
from .gcode_legacy_postprocess import apply_legacy_resin_optimization
from .honeycomb_pathing import HoneycombPathingConfig

from .slicer import (
    DEFAULT_FIBER_LINE_WIDTH_MM,
    DEFAULT_FIBER_LAYER_HEIGHT_MM,
    DEFAULT_RESIN_CONTOUR_INFILL_OVERLAP_PERCENT,
    DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
    DEFAULT_RESIN_LAYER_HEIGHT_MM,
    DEFAULT_RESIN_LINE_WIDTH_MM,
    DEFAULT_RESIN_PLANNING_LINE_WIDTH_MM,
    DEFAULT_PRUSA_RAFT_CONTACT_DISTANCE_MM,
    DEFAULT_PRUSA_RAFT_EXPANSION_MM,
    DEFAULT_PRUSA_RAFT_FIRST_LAYER_DENSITY_PERCENT,
    DEFAULT_PRUSA_RAFT_FIRST_LAYER_EXPANSION_MM,
    DEFAULT_PRUSA_RAFT_LAYER_COUNT,
    DEFAULT_RAFT_LAYER_COUNT,
    DEFAULT_RAFT_OUTWARD_OFFSETS_MM,
    DEFAULT_RAFT_TOP_GAP_MM,
    PySLMConfig,
    PrusaGeometryConfig,
    PrusaRaftConfig,
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
)
from .stl_io import load_stl
from .surface_peak_collision import check_peak_surface_collision


MAX_SURFACE_PREVIEW_NPZ_BYTES = 256 * 1024 * 1024
PRINTHEAD_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "printhead"

DEFAULT_UI_RESIN_INFILL_OVERLAP_PERCENT = 0.0


class FiberTemplatePaths(list[list[list[float]]]):
    """Fiber geometry together with the declared XY coordinate semantics."""

    def __init__(
        self,
        paths: list[list[list[float]]],
        *,
        coordinate_system: str | None,
    ) -> None:
        super().__init__(paths)
        self.coordinate_system = coordinate_system


def _surface_preview_picker_state_path(output_dir: Path) -> Path:
    return output_dir / ".surface_preview_picker.json"


def _core_preview_picker_state_path(output_dir: Path) -> Path:
    return output_dir / ".core_preview_picker.json"


def _load_surface_preview_last_directory(state_path: Path) -> Path | None:
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
        directory = raw.get("last_directory") if isinstance(raw, dict) else None
        candidate = Path(directory) if isinstance(directory, str) else None
        return candidate if candidate is not None and candidate.is_dir() else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _save_surface_preview_last_directory(state_path: Path, directory: Path) -> None:
    try:
        state_path.write_text(
            json.dumps({"last_directory": str(directory)}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        # Directory recall improves this local UI command but must never make
        # an otherwise valid NPZ preview fail to load.
        pass


def _choose_mapped_surface_npz_file(initial_directory: Path | None) -> Path | None:
    """Open the Windows picker for legacy mapped or conformal-path NPZ files."""

    if sys.platform != "win32":
        raise RuntimeError("native surface/conformal-NPZ picker is available only on Windows")
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("无法加载 Windows 原生文件选择组件") from exc

    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        selected = filedialog.askopenfilename(
            parent=root,
            title="选择映射曲面或共形格栅 NPZ",
            initialdir=(
                str(initial_directory)
                if initial_directory is not None and initial_directory.is_dir()
                else None
            ),
            filetypes=[("映射曲面或共形格栅 NPZ", "*.npz"), ("所有文件", "*.*")],
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def _choose_final_core_npz_file(initial_directory: Path | None) -> Path | None:
    """Open the Windows picker for a regular final Core trajectory NPZ."""

    if sys.platform != "win32":
        raise RuntimeError("native Core-NPZ picker is available only on Windows")
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("无法加载 Windows 原生文件选择组件") from exc

    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes("-topmost", True)
        root.update()
        selected = filedialog.askopenfilename(
            parent=root,
            title="选择 Core 导出 NPZ",
            initialdir=(
                str(initial_directory)
                if initial_directory is not None and initial_directory.is_dir()
                else None
            ),
            filetypes=[("Core 导出 NPZ", "*.npz"), ("所有文件", "*.*")],
        )
    finally:
        root.destroy()
    return Path(selected) if selected else None


def _offline_planner_data_root() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "packages"
        / "offline_path_planner"
        / "data"
    )


def _core_print_params_path() -> Path:
    return _offline_planner_data_root() / "external_npz_preprocessor" / "print_params.json"


def _prusa_params_path() -> Path:
    return _offline_planner_data_root() / "external_npz_preprocessor" / "prusa_params.json"


def _core_output_download_path(core_npz_path: Path) -> Path:
    """Return one downloadable artifact for a core export.

    The core writer renames a one-part export to the requested NPZ path.  If a
    large export is split, bundle its parts and sidecars so the browser does
    not silently download only the first trajectory fragment.
    """

    if core_npz_path.is_file():
        return core_npz_path

    parts = sorted(core_npz_path.parent.glob(f"{core_npz_path.stem}_part*.npz"))
    if not parts:
        raise FileNotFoundError(f"core NPZ output was not written: {core_npz_path}")

    package_path = core_npz_path.with_name(f"{core_npz_path.stem}_package.zip")
    sidecars = [
        core_npz_path.with_name(f"{core_npz_path.stem}.offset.json"),
        core_npz_path.with_name(f"{core_npz_path.stem}.timing.json"),
    ]
    # Core parts are already ``np.savez_compressed`` archives.  Deflating them
    # a second time burns CPU for negligible size reduction, so the outer ZIP
    # only bundles the exact same files for browser download.
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for candidate in [*parts, *sidecars]:
            if candidate.is_file():
                archive.write(candidate, arcname=candidate.name)
    return package_path


def _final_core_npz_parts(core_npz_path: Path) -> list[Path]:
    """Return the exact final Core files that the runtime would load."""

    if core_npz_path.is_file():
        part_match = re.fullmatch(r"(.+)_part\d+", core_npz_path.stem)
        if part_match is not None:
            parts = sorted(core_npz_path.parent.glob(f"{part_match.group(1)}_part*.npz"))
            if parts:
                return parts
        return [core_npz_path]
    parts = sorted(core_npz_path.parent.glob(f"{core_npz_path.stem}_part*.npz"))
    if not parts:
        raise FileNotFoundError(f"final Core NPZ was not written: {core_npz_path}")
    return parts


def _core_move_type_codes(data) -> set[int]:
    """Return vocabulary codes that represent deposited Core trajectory rows."""

    keys_name = "move_type_vocab_keys"
    values_name = "move_type_vocab_vals"
    if keys_name in data.files and values_name in data.files:
        def _name(value) -> str:
            raw = value.item() if hasattr(value, "item") else value
            return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

        return {
            int(value)
            for key, value in zip(data[keys_name], data[values_name])
            if _name(key).upper() in {"PRINT", "PRINT_FIT"}
        }
    # Legacy Core files use these stable numeric values when no vocabulary is
    # present.  Keep the fallback local to preview decoding.
    return {1, 3}


def _use_native_prusa_gcode_for_core(
    config: SliceConfig,
    native_gcode: bytes | str | None,
) -> bool:
    """Return whether Core may consume the native Prusa G-code chain.

    The project-owned G-code postprocess applies Legacy infill continuity and,
    when selected, Brim one-stroke continuity to the parsed ``SourceJob``.
    Therefore ordinary Prusa jobs, including Brim jobs, retain G-code as the
    sole Core input representation.  Honeycomb remains a separate planner.
    """

    return (
        config.slicing_kernel == "prusa"
        and not config.honeycomb_pathing.enabled
        and isinstance(native_gcode, (bytes, str))
    )


def _planning_mesh_for_gcode_source(mesh, config: SliceConfig):
    """Return geometry in the same frame as translated native Prusa G-code.

    The Prusa backend slices an oriented mesh and exposes the inverse of its
    temporary bed placement as ``native_gcode_translation_mm``.  Once that
    translation has been applied, SourceJob coordinates are in the oriented
    model frame.  Legacy avoidance geometry must use that frame as well.
    """

    return orient_mesh_for_build_axis(mesh, config.build_axis)


def _preview_payload_from_core_source_job(mesh, config: SliceConfig, source_job) -> dict[str, object]:
    """Render the exact G-code SourceJob that is handed to Core."""

    material_paths = []
    travel_paths = []
    for layer in source_job.layers:
        if layer.resin_paths:
            material_paths.append(
                MaterialPaths(
                    layer.index,
                    "R",
                    [path.points for path in layer.resin_paths],
                    [path.extrusion for path in layer.resin_paths],
                )
            )
        if layer.fiber_paths:
            material_paths.append(
                MaterialPaths(
                    layer.index,
                    "F",
                    [path.points for path in layer.fiber_paths],
                    [path.extrusion for path in layer.fiber_paths],
                )
            )
        if layer.travel_paths:
            travel_paths.append(TravelPaths(layer.index, [path.points for path in layer.travel_paths]))
    return _preview_payload(
        mesh,
        config,
        ExternalSourceJob(material_paths=material_paths, travel_paths=travel_paths, meta=source_job.meta),
    )


def _preview_payload_from_final_core_npz(
    core_npz_path: Path,
    config: SliceConfig,
) -> dict[str, object]:
    """Build the browser payload from the final Core NPZ, never from its source.

    Only non-event rows are considered because the runtime queue likewise does
    not emit trajectory points for event rows.  The browser receives direct
    samples of those final rows; it performs no Core-like fitting, smoothing,
    offsetting, or interpolation.  Stationary process rows (prime/retract/
    reset at a fixed XYZ) remain in the NPZ for runtime timing, but are not
    spatial paths and therefore are not rendered as deposition points.
    """

    entries_by_layer: dict[int, list[dict[str, object]]] = {}
    bounds = {
        "min_x": None,
        "max_x": None,
        "min_y": None,
        "max_y": None,
        "min_z": None,
        "max_z": None,
    }
    has_curved_deposition = False
    has_tool_orientation = False
    order = 0

    for path in _final_core_npz_parts(core_npz_path):
        with np.load(path, allow_pickle=False) as data:
            required = {"x", "y", "z", "tool_id", "move_type"}
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(f"final Core NPZ is missing fields: {', '.join(missing)}")

            x = np.asarray(data["x"], dtype=np.float64)
            y = np.asarray(data["y"], dtype=np.float64)
            z = np.asarray(data["z"], dtype=np.float64)
            count = len(x)
            if count == 0:
                continue
            tool_id = np.asarray(data["tool_id"], dtype=np.int64)
            move_type = np.asarray(data["move_type"], dtype=np.int64)
            event_flag = (
                np.asarray(data["event_flag"], dtype=np.int64)
                if "event_flag" in data.files
                else np.zeros(count, dtype=np.int64)
            )
            layer = np.asarray(
                data["preview_layer_index"]
                if "preview_layer_index" in data.files
                else data["layer_index"]
                if "layer_index" in data.files
                else np.zeros(count, dtype=np.int64),
                dtype=np.int64,
            )
            path_id = (
                np.asarray(data["path_id"], dtype=np.int64)
                if "path_id" in data.files
                else np.zeros(count, dtype=np.int64)
            )
            path_end = (
                np.asarray(data["path_end_flag"], dtype=np.int64)
                if "path_end_flag" in data.files
                else np.zeros(count, dtype=np.int64)
            )
            a = np.asarray(data["a"], dtype=np.float64) if "a" in data.files else None
            b = np.asarray(data["b"], dtype=np.float64) if "b" in data.files else None
            c = np.asarray(data["c"], dtype=np.float64) if "c" in data.files else None

            valid = (
                (event_flag != 1)
                & np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(z)
            )
            if not np.any(valid):
                continue

            valid_x, valid_y, valid_z = x[valid], y[valid], z[valid]
            for key, value in (
                ("min_x", valid_x.min()), ("max_x", valid_x.max()),
                ("min_y", valid_y.min()), ("max_y", valid_y.max()),
                ("min_z", valid_z.min()), ("max_z", valid_z.max()),
            ):
                bounds[key] = float(value) if bounds[key] is None else (
                    min(float(bounds[key]), float(value)) if key.startswith("min")
                    else max(float(bounds[key]), float(value))
                )

            include_abc = (
                a is not None
                and b is not None
                and c is not None
                and bool(np.any(np.abs(a[valid]) > 1e-9)
                         or np.any(np.abs(b[valid]) > 1e-9)
                         or np.any(np.abs(c[valid]) > 1e-9))
            )
            has_tool_orientation = has_tool_orientation or include_abc
            print_codes = _core_move_type_codes(data)
            is_print = np.isin(move_type, list(print_codes))

            indices = np.flatnonzero(valid)
            start = 0
            while start < len(indices):
                first = int(indices[start])
                is_fiber = bool(is_print[first] and tool_id[first] == 1)
                is_resin = bool(is_print[first] and not is_fiber)
                role = "fiber" if is_fiber else "final_resin" if is_resin else "travel"
                end = start + 1
                while end < len(indices):
                    previous = int(indices[end - 1])
                    current = int(indices[end])
                    current_is_fiber = bool(is_print[current] and tool_id[current] == 1)
                    current_is_resin = bool(is_print[current] and not current_is_fiber)
                    current_role = (
                        "fiber" if current_is_fiber
                        else "final_resin" if current_is_resin
                        else "travel"
                    )
                    same_path = path_id[current] == path_id[first]
                    # Final NPZ path IDs describe Core command boundaries,
                    # not a discontinuity in the RSI point sequence.  For
                    # adjacent non-event Travel rows, join the browser path
                    # across those metadata boundaries while retaining every
                    # final XYZABC point.  This is intentionally display-only:
                    # no avoidance waypoint, sequence value, or exported NPZ
                    # row is changed.
                    adjacent_travel = role == "travel" and current_role == "travel"
                    if (
                        current != previous + 1
                        or layer[current] != layer[first]
                        or current_role != role
                        or tool_id[current] != tool_id[first]
                        or (
                            not adjacent_travel
                            and (not same_path or path_end[previous] == 1)
                        )
                    ):
                        break
                    end += 1

                segment_indices = indices[start:end]
                is_stationary_process = role != "travel" and bool(
                    np.ptp(x[segment_indices]) <= 1e-9
                    and np.ptp(y[segment_indices]) <= 1e-9
                    and np.ptp(z[segment_indices]) <= 1e-9
                )
                point_columns = [x[segment_indices], y[segment_indices], z[segment_indices]]
                if include_abc:
                    point_columns.extend((a[segment_indices], b[segment_indices], c[segment_indices]))
                # For diagnosis the browser receives the complete final Core
                # point sequence.  No display-side point decimation is allowed:
                # a sharp corner must be attributable to the NPZ itself.
                points = np.column_stack(point_columns).tolist()
                if points and not is_stationary_process:
                    if role != "travel" and float(np.ptp(z[segment_indices])) > 1e-7:
                        has_curved_deposition = True
                    entries_by_layer.setdefault(int(layer[first]), []).append(
                        {
                            "kind": "deposit" if role != "travel" else "travel",
                            "role": role,
                            "points": points,
                            "order": order,
                        }
                    )
                    order += 1
                start = end

    all_entries = [entry for entries in entries_by_layer.values() for entry in entries]
    if not all_entries:
        raise ValueError("final Core NPZ contains no displayable trajectory rows")
    layers = []
    for layer_index in sorted(entries_by_layer):
        entries = entries_by_layer[layer_index]
        resin_paths = [
            {"role": entry["role"], "points": entry["points"]}
            for entry in entries if entry["role"] == "final_resin"
        ]
        fiber_paths = [
            entry["points"] for entry in entries if entry["role"] == "fiber"
        ]
        travel_paths = [
            entry["points"] for entry in entries if entry["role"] == "travel"
        ]
        layers.append(
            {
                "index": layer_index,
                "resin_paths": resin_paths,
                "fiber_paths": fiber_paths,
                "travel_paths": travel_paths,
                "motion_paths": entries,
            }
        )

    planning_line_width = (
        config.line_width
        if config.slicing_kernel != "legacy" or config.planning_line_width is None
        else config.planning_line_width
    )
    return {
        "bounds": bounds,
        "origin": [0.0, 0.0],
        "geometry_mode": "surface_3d" if has_curved_deposition else "planar_2d",
        "tool_orientation": {
            "available": has_tool_orientation,
            "fallback": "calibrated_flat_downward",
        },
        "line_widths": {
            "resin": float(planning_line_width),
            "resin_nominal": float(config.line_width),
            "fiber": DEFAULT_FIBER_LINE_WIDTH_MM,
        },
        "preview_source": "final_core_npz",
        "layers": layers,
    }


def _preview_position(point, xy_offset: tuple[float, float] = (0.0, 0.0)) -> list[float]:
    return [
        float(point.x) + float(xy_offset[0]),
        float(point.y) + float(xy_offset[1]),
        float(point.z),
    ]


def _core_preview_overlay_from_commands(
    commands,
    *,
    xy_offset: tuple[float, float] = (0.0, 0.0),
) -> dict[str, object]:
    """Collect Core-only geometry that is absent from the Prusa preview."""

    overlay: dict[str, object] = {
        "origin": [0.0, 0.0],
        "primeline_paths": [],
        "core_travel_paths": [],
        "layer_lift_paths": [],
        "sequence": [],
    }
    primeline_paths = overlay["primeline_paths"]
    core_travel_paths = overlay["core_travel_paths"]
    layer_lift_paths = overlay["layer_lift_paths"]
    sequence = overlay["sequence"]

    resin_base_counts: dict[int, int] = {}
    previous_prusa_travel_layer: int | None = None
    for command in commands:
        layer = int(getattr(command, "layer", 0))
        raw = getattr(command, "raw", None)
        subtype = getattr(command, "subtype", None)
        if raw in {"external_npz_start_xy_travel", "external_npz_prusa_travel"}:
            if raw == "external_npz_start_xy_travel" or previous_prusa_travel_layer != layer:
                resin_base_counts[layer] = resin_base_counts.get(layer, 0) + 1
            previous_prusa_travel_layer = layer if raw == "external_npz_prusa_travel" else None
        elif getattr(command, "type", None) == "PRINT":
            previous_prusa_travel_layer = None
            if subtype != "FIBER_PRINT" and raw != "external_npz_primeline":
                resin_base_counts[layer] = resin_base_counts.get(layer, 0) + 1
        else:
            previous_prusa_travel_layer = None

    source_index: dict[int, int] = {}
    fiber_index: dict[int, int] = {}
    previous_prusa_travel_layer: int | None = None
    sequence_order = 0

    def display_index(layer: int) -> int:
        if fiber_index.get(layer, 0) > 0:
            return resin_base_counts.get(layer, 0) + fiber_index[layer]
        return source_index.get(layer, 0)

    for command in commands:
        raw = getattr(command, "raw", None)
        layer = int(getattr(command, "layer", 0))
        subtype = getattr(command, "subtype", None)
        if raw == "external_npz_primeline":
            points = [
                _preview_position(getattr(command, "start_pos"), xy_offset),
                *[
                    _preview_position(point, xy_offset)
                    for point in getattr(command, "control_points", [])
                ],
            ]
            if len(points) >= 2:
                item = {"layer": layer, "points": points}
                primeline_paths.append(item)
                sequence.append(
                    {
                        "layer": layer,
                        "kind": "deposit",
                        "role": "primeline",
                        "points": points,
                        "anchor": display_index(layer),
                        "order": sequence_order,
                    }
                )
        elif raw in {"external_npz_travel", "external_npz_layer_lift"}:
            points = [
                _preview_position(getattr(command, "start_pos"), xy_offset),
                _preview_position(getattr(command, "pos"), xy_offset),
            ]
            if raw == "external_npz_layer_lift":
                layer_lift_paths.append({"layer": layer, "points": points})
                role = "layer_lift"
            else:
                core_travel_paths.append({"layer": layer, "points": points})
                role = "core_travel"
            sequence.append(
                {
                    "layer": layer,
                    "kind": "travel",
                    "role": role,
                    "points": points,
                    "anchor": display_index(layer),
                    "order": sequence_order,
                }
            )
        elif raw in {"external_npz_start_xy_travel", "external_npz_prusa_travel"}:
            if raw == "external_npz_start_xy_travel" or previous_prusa_travel_layer != layer:
                source_index[layer] = source_index.get(layer, 0) + 1
            previous_prusa_travel_layer = layer if raw == "external_npz_prusa_travel" else None
        elif getattr(command, "type", None) == "PRINT":
            previous_prusa_travel_layer = None
            if subtype == "FIBER_PRINT":
                fiber_index[layer] = fiber_index.get(layer, 0) + 1
            elif raw != "external_npz_primeline":
                source_index[layer] = source_index.get(layer, 0) + 1
        else:
            previous_prusa_travel_layer = None
        sequence_order += 1
    return overlay


def _core_preview_xy_offset(job, core_params) -> tuple[float, float]:
    """Map Core command coordinates back into the Prusa preview frame.

    The external converter normalizes material coordinates by the minimum XY
    of all material paths before generating Core commands.  The browser keeps
    showing the Prusa/source frame, so Core-only overlay geometry needs the
    inverse translation for display.
    """

    min_x = float("inf")
    min_y = float("inf")
    for group in getattr(job, "material_paths", []):
        for path in getattr(group, "paths", []):
            points = np.asarray(path, dtype=np.float32)
            if points.size == 0:
                continue
            finite = points[np.isfinite(points[:, :2]).all(axis=1)]
            if finite.size == 0:
                continue
            min_x = min(min_x, float(np.min(finite[:, 0])))
            min_y = min(min_y, float(np.min(finite[:, 1])))

    if not np.isfinite(min_x) or not np.isfinite(min_y):
        return (0.0, 0.0)
    return (
        min_x - float(getattr(core_params, "start_x_mm", 0.0)),
        min_y - float(getattr(core_params, "start_y_mm", 0.0)),
    )


def _ensure_offline_planner_import_paths() -> None:
    """Make the checked-out offline planner packages available to the UI.

    The standalone slicer package intentionally does not statically import the
    offline planner package.  The UI still needs the repository checkout when
    both projects are run directly without installing the ROS-style packages.
    """

    package_root = (
        Path(__file__).resolve().parent.parent
        / "packages"
        / "offline_path_planner"
        / "src"
        / "my_project"
    )
    for package_name in (
        "external_npz_preprocessor",
        "path_processing_core",
        "gcode_planner",
    ):
        package_path = package_root / package_name
        if package_path.is_dir() and str(package_path) not in sys.path:
            sys.path.insert(0, str(package_path))


def _load_core_print_params():
    """Load the persisted process_core defaults used by the integrated UI."""

    _ensure_offline_planner_import_paths()
    module = importlib.import_module("external_npz_preprocessor.param_config")
    return module.load_print_params(_core_print_params_path())


def _load_prusa_params() -> dict[str, object]:
    """Load the integrated Prusa UI defaults, tolerating an absent/old file."""

    path = _prusa_params_path()
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    values = raw.get("params", raw) if isinstance(raw, dict) else {}
    return values if isinstance(values, dict) else {}


def _save_prusa_params(values: dict[str, object]) -> Path:
    path = _prusa_params_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "kuka_slicer_prusa_params",
        "version": 1,
        "params": values,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _coerce_form_params(values: dict[str, object]) -> dict[str, list[str]]:
    """Convert JSON scalar values into the parser's form-style mapping."""

    return {key: [str(value).lower() if isinstance(value, bool) else str(value)] for key, value in values.items()}


_PRUSA_FLOAT_KEYS = {
    "layer_height", "first_layer_height", "line_width", "prusa_start_x_mm",
    "prusa_start_y_mm", "prusa_infill_density", "prusa_contour_infill_overlap",
    "prusa_raft_expansion", "prusa_raft_first_layer_density",
    "prusa_raft_first_layer_expansion", "prusa_raft_contact_distance",
    "prusa_raft_contact_layer_height", "prusa_raft_contact_density",
    "prusa_raft_contact_extrusion_width",
    "prusa_brim_width", "prusa_brim_separation",
    "prusa_external_perimeter_width", "prusa_perimeter_width", "prusa_infill_width",
    "prusa_xy_size_compensation", "prusa_elephant_foot_compensation",
    "prusa_infill_anchor", "prusa_infill_anchor_max", "prusa_avoid_crossing_max_detour",
}
_PRUSA_INT_KEYS = {
    "prusa_perimeter_count", "prusa_raft_layers",
}
_PRUSA_BOOL_KEYS = {
    "prusa_print_perimeters", "prusa_raft_enabled", "prusa_raft_auto_contact",
    "prusa_gap_fill_enabled", "prusa_brim_enabled", "prusa_brim_one_stroke",
    "honeycomb_centerline_enabled",
}
_PRUSA_NULLABLE_FLOAT_KEYS = {
    "prusa_external_perimeter_width", "prusa_perimeter_width", "prusa_infill_width",
    "prusa_infill_anchor", "prusa_infill_anchor_max",
}


def _normalize_prusa_params(values: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in values.items():
        if key in _PRUSA_NULLABLE_FLOAT_KEYS and value in (None, ""):
            normalized[key] = None
        elif key in _PRUSA_FLOAT_KEYS:
            normalized[key] = float(value)
        elif key in _PRUSA_INT_KEYS:
            normalized[key] = int(float(value))
        elif key in _PRUSA_BOOL_KEYS:
            normalized[key] = (
                value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
            )
        else:
            normalized[key] = value
    return normalized


def run_ui_server(host: str, port: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    class SlicerUiHandler(_SlicerUiHandler):
        server_output_dir = output_dir.resolve()
        slice_jobs: dict[str, dict[str, object]] = {}
        slice_jobs_lock = threading.Lock()
        tool_launchers: dict[str, subprocess.Popen[bytes]] = {}
        tool_launchers_lock = threading.Lock()
        surface_preview_picker_lock = threading.Lock()
        surface_preview_picker_state_path = _surface_preview_picker_state_path(server_output_dir)
        surface_preview_last_directory = _load_surface_preview_last_directory(
            surface_preview_picker_state_path
        )
        core_preview_picker_state_path = _core_preview_picker_state_path(server_output_dir)
        core_preview_last_directory = _load_surface_preview_last_directory(
            core_preview_picker_state_path
        )

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
    slice_jobs: dict[str, dict[str, object]] = {}
    slice_jobs_lock = threading.Lock()
    tool_launchers: dict[str, subprocess.Popen[bytes]] = {}
    tool_launchers_lock = threading.Lock()
    surface_preview_picker_lock = threading.Lock()
    surface_preview_last_directory: Path | None = None
    surface_preview_picker_state_path: Path | None = None
    surface_preview_selected_path: Path | None = None
    core_preview_last_directory: Path | None = None
    core_preview_picker_state_path: Path | None = None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(_index_html())
            return
        if parsed.path == "/slice-status":
            self._send_slice_status(parse_qs(parsed.query))
            return
        if parsed.path.startswith("/outputs/"):
            self._send_output_file(parsed.path.removeprefix("/outputs/"))
            return
        if parsed.path.startswith("/assets/printhead/"):
            self._send_printhead_asset(parsed.path.removeprefix("/assets/printhead/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/launch-tool":
            self._launch_tool(parse_qs(parsed.query))
            return
        if parsed.path == "/ui-settings":
            try:
                self._save_ui_settings()
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"ok": True})
            return
        if parsed.path == "/choose-surface-npz-preview":
            try:
                self._choose_surface_npz_preview()
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/choose-core-npz-preview":
            try:
                self._choose_core_npz_preview()
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/check-surface-npz-collision":
            try:
                selected = type(self).surface_preview_selected_path
                if selected is None:
                    raise ValueError("请选择本地映射 NPZ 后再执行碰撞检查")
                self._send_json(check_peak_surface_collision(selected))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path == "/preview-source-npz":
            try:
                _, files = self._read_slice_request(parsed.query)
                source_upload = files.get("source_npz")
                if source_upload is None:
                    raise ValueError("missing mapped source NPZ payload")
                source_name, source_bytes = source_upload
                if len(source_bytes) > MAX_SURFACE_PREVIEW_NPZ_BYTES:
                    raise ValueError("mapped source NPZ exceeds the 256 MB preview limit")
                self._send_json(
                    {
                        "ok": True,
                        "preview": _preview_payload_from_source_npz(
                            source_bytes,
                            _safe_filename(source_name or "curved.npz"),
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if parsed.path != "/slice":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            request_data = self._read_slice_request(parsed.query)
            job_id = self._start_slice_job(parsed.query, request_data)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(
            {
                "ok": True,
                "job_id": job_id,
                "state": "queued",
                "progress": 0,
                "message": "已接收任务，等待处理",
            }
        )

    def _choose_surface_npz_preview(self) -> None:
        """Choose and load a legacy mapped or conformal-path NPZ."""

        handler_type = type(self)
        with handler_type.surface_preview_picker_lock:
            selected = _choose_mapped_surface_npz_file(handler_type.surface_preview_last_directory)
        if selected is None:
            self._send_json({"ok": True, "cancelled": True})
            return
        selected = selected.resolve()
        if selected.suffix.lower() != ".npz":
            raise ValueError("请选择 .npz 映射曲面或共形格栅文件")
        if not selected.is_file():
            raise ValueError("所选曲面或共形格栅 NPZ 不存在")
        if selected.stat().st_size > MAX_SURFACE_PREVIEW_NPZ_BYTES:
            raise ValueError("mapped source NPZ exceeds the 256 MB preview limit")
        handler_type.surface_preview_last_directory = selected.parent
        handler_type.surface_preview_selected_path = selected
        state_path = handler_type.surface_preview_picker_state_path
        if state_path is not None:
            _save_surface_preview_last_directory(state_path, selected.parent)
        preview = _preview_payload_from_source_npz(selected.read_bytes(), selected.name)
        self._send_json(
            {
                "ok": True,
                "file_name": selected.name,
                "collision_check_available": preview.get("preview_source") != "conformal_lattice_external_source_npz",
                "preview": preview,
            }
        )

    def _choose_core_npz_preview(self) -> None:
        """Choose a final Core NPZ and render its exported trajectory directly."""

        handler_type = type(self)
        with handler_type.surface_preview_picker_lock:
            selected = _choose_final_core_npz_file(handler_type.core_preview_last_directory)
        if selected is None:
            self._send_json({"ok": True, "cancelled": True})
            return
        selected = selected.resolve()
        if selected.suffix.lower() != ".npz":
            raise ValueError("请选择 .npz Core 导出文件")
        if not selected.is_file():
            raise ValueError("所选 Core NPZ 不存在")
        core_parts = _final_core_npz_parts(selected)
        total_size = sum(part.stat().st_size for part in core_parts)
        if total_size > MAX_SURFACE_PREVIEW_NPZ_BYTES:
            raise ValueError("Core NPZ exceeds the 256 MB preview limit")
        handler_type.core_preview_last_directory = selected.parent
        state_path = handler_type.core_preview_picker_state_path
        if state_path is not None:
            _save_surface_preview_last_directory(state_path, selected.parent)
        self._send_json(
            {
                "ok": True,
                "file_name": selected.name,
                "preview": _preview_payload_from_final_core_npz(
                    selected,
                    SliceConfig(line_width=2.0),
                ),
            }
        )

    def _launch_tool(self, params: dict[str, list[str]]) -> None:
        tool = params.get("tool", [""])[0]
        if tool not in {"surface-preview", "surface-map"}:
            self._send_json({"ok": False, "error": "unknown local tool"}, HTTPStatus.BAD_REQUEST)
            return
        with self.tool_launchers_lock:
            existing = self.tool_launchers.get(tool)
            if existing is not None and existing.poll() is None:
                self._send_json({"ok": True, "already_running": True})
                return
            from .app_session import spawn_app_session

            process = spawn_app_session(tool)
            self.tool_launchers[tool] = process
        watcher = threading.Thread(
            target=self._clear_finished_tool,
            args=(tool, process),
            daemon=True,
            name=f"slicer-tool-{tool}",
        )
        watcher.start()
        self._send_json({"ok": True, "already_running": False})

    def _clear_finished_tool(self, tool: str, process: subprocess.Popen[bytes]) -> None:
        process.wait()
        with self.tool_launchers_lock:
            if self.tool_launchers.get(tool) is process:
                self.tool_launchers.pop(tool, None)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _start_slice_job(
        self,
        query: str,
        request_data: tuple[dict[str, list[str]], dict[str, tuple[str | None, bytes]]],
    ) -> str:
        job_id = uuid4().hex
        self._update_slice_job(
            job_id,
            state="queued",
            progress=0,
            message="已接收任务，等待处理",
            elapsed_s=0.0,
        )
        worker = threading.Thread(
            target=self._run_slice_job,
            args=(job_id, query, request_data),
            daemon=True,
            name=f"slicer-ui-{job_id[:8]}",
        )
        worker.start()
        return job_id

    def _run_slice_job(
        self,
        job_id: str,
        query: str,
        request_data: tuple[dict[str, list[str]], dict[str, tuple[str | None, bytes]]],
    ) -> None:
        started_at = time.perf_counter()

        def update_progress(progress: int, message: str) -> None:
            elapsed_s = time.perf_counter() - started_at
            self._update_slice_job(
                job_id,
                state="running",
                progress=max(0, min(99, int(progress))),
                message=message,
                elapsed_s=elapsed_s,
            )

        try:
            # Limit the whole UI task, including native slicing and the Core
            # trajectory export.  Nested slice_mesh_to_job calls reuse this
            # re-entrant guard.
            with limit_slicer_task() as cpu_limit:
                result = self._handle_slice(
                    query,
                    request_data=request_data,
                    progress_callback=update_progress,
                )
            result["cpu_limit"] = cpu_limit.to_metadata()
            elapsed_s = time.perf_counter() - started_at
            self._update_slice_job(
                job_id,
                state="complete",
                progress=100,
                message="导出完成",
                elapsed_s=elapsed_s,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_s = time.perf_counter() - started_at
            self._update_slice_job(
                job_id,
                state="error",
                progress=0,
                message="导出失败",
                elapsed_s=elapsed_s,
                error=str(exc),
            )

    def _update_slice_job(self, job_id: str, **updates: object) -> None:
        cls = type(self)
        with cls.slice_jobs_lock:
            job = cls.slice_jobs.setdefault(job_id, {})
            job.update(updates)

    def _send_slice_status(self, query: dict[str, list[str]]) -> None:
        job_id = query.get("job_id", [""])[0].strip()
        if not job_id:
            self._send_json(
                {"ok": False, "error": "missing job_id"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        cls = type(self)
        with cls.slice_jobs_lock:
            job = dict(cls.slice_jobs.get(job_id, {}))
        if not job:
            self._send_json(
                {"ok": False, "error": "slice job not found"},
                HTTPStatus.NOT_FOUND,
            )
            return
        self._send_json({"ok": True, "job_id": job_id, **job})

    def _handle_slice(
        self,
        query: str,
        *,
        request_data=None,
        progress_callback=None,
    ) -> dict[str, object]:
        params, files = request_data or self._read_slice_request(query)

        def progress(value: int, message: str) -> None:
            if progress_callback is not None:
                progress_callback(value, message)

        progress(2, "正在读取模型和切片参数")
        filename = _safe_filename(params.get("filename", ["input.stl"])[0])
        slicing_kernel = params.get("slicing_kernel", ["prusa"])[0]
        if slicing_kernel not in {"prusa", "legacy", "pyslm"}:
            raise ValueError("slicing_kernel must be prusa, legacy, or pyslm")
        shared = _parse_shared_slice_config(params, slicing_kernel)

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
        fiber_template_paths: list[list[list[float]]] = []
        if "fiber_json" in files:
            fiber_filename, fiber_bytes = files["fiber_json"]
            fiber_json_name = _safe_filename(fiber_filename or "fiber_paths.json")
            fiber_json_path = job_dir / fiber_json_name
            fiber_json_path.write_bytes(fiber_bytes)
            fiber_template_paths = load_fiber_template_json(fiber_json_path)

        mesh = load_stl(stl_path)
        progress(8, "正在准备 Prusa 路径")
        build_axis = resolve_build_axis(mesh, shared["requested_build_axis"])
        if slicing_kernel == "prusa":
            config = _parse_prusa_slice_config(params, shared, build_axis)
            raft_layers: list[RaftLayerConfig] = []
        elif slicing_kernel == "legacy":
            config = _parse_legacy_slice_config(params, shared, build_axis)
            raft_layers = _raft_layers_from_params(params)
        else:
            config = _parse_pyslm_slice_config(params, shared, build_axis)
            raft_layers = []
        job = slice_mesh_to_job(mesh, config)
        core_source_job = None
        native_gcode_path = None
        source_gcode_module = None
        if _use_native_prusa_gcode_for_core(config, job.native_gcode):
            _ensure_offline_planner_import_paths()
            source_gcode_module = importlib.import_module(
                "external_npz_preprocessor.source_gcode"
            )
            native_gcode_path = job_dir / f"{Path(filename).stem}_prusa.gcode"
            native_gcode_path.write_bytes(
                job.native_gcode
                if isinstance(job.native_gcode, bytes)
                else job.native_gcode.encode("utf-8")
            )
            core_source_job = source_gcode_module.translate_source_job(
                source_gcode_module.load_source_gcode(native_gcode_path),
                job.native_gcode_translation_mm or (0.0, 0.0, 0.0),
            )
            core_source_job = apply_legacy_resin_optimization(
                core_source_job,
                _planning_mesh_for_gcode_source(mesh, config),
                config,
            )
        progress(35, "Prusa 路径生成完成，正在保留原始预览")
        resolved_config = _resolved_slice_config(config)
        slicing_meta = job.meta.get("slicing")
        if isinstance(slicing_meta, dict):
            slicing_meta["resolved_config"] = resolved_config
        fiber_preview_paths = {}
        fiber_travel_paths = {}
        if fiber_template_paths:
            fiber_template_paths = align_fiber_template_paths_to_resin(
                job,
                fiber_template_paths,
            )
            fiber_preview_paths = expand_fiber_template_for_resin_layers(
                job, fiber_template_paths
            )
            fiber_travel_paths = plan_fiber_interpath_travels(
                mesh,
                config,
                fiber_preview_paths,
                reference_z_by_layer=job.meta.get("fiber_interpath_reference_z_mm"),
            )
            merge_fiber_paths_into_job(
                job,
                fiber_preview_paths,
                fiber_travel_paths,
            )
            if core_source_job is not None and source_gcode_module is not None:
                core_source_job = source_gcode_module.with_fiber_paths(
                    core_source_job,
                    fiber_preview_paths,
                    fiber_travel_paths_by_layer=fiber_travel_paths,
                )
        if raft_layers:
            z_shift = add_raft_to_job(job, mesh, config, raft_layers, DEFAULT_RAFT_TOP_GAP_MM)
            fiber_preview_paths = _shift_fiber_preview_paths(
                fiber_preview_paths,
                len(raft_layers),
                z_shift,
            )
        normalize_job_xy_origin(
            job,
            target_xy=(float(config.start_x_mm), float(config.start_y_mm)),
            reference_material="R",
        )
        primeline_enabled = _bool_param(params, "core_primeline_enabled", True)
        if slicing_kernel == "prusa":
            _prepend_prusa_startup_travel(
                job,
                start_xy=(float(config.start_x_mm), float(config.start_y_mm)),
                primeline_enabled=primeline_enabled,
                primeline_xy=(
                    _float_param(params, "core_primeline_x_mm", 0.0),
                    _float_param(params, "core_primeline_y_mm", -10.0),
                ),
            )
            if core_source_job is not None and source_gcode_module is not None:
                core_source_job = source_gcode_module.prepend_prusa_startup_travel(
                    core_source_job,
                    start_xy=(float(config.start_x_mm), float(config.start_y_mm)),
                    primeline_enabled=primeline_enabled,
                    primeline_xy=(
                        _float_param(params, "core_primeline_x_mm", 0.0),
                        _float_param(params, "core_primeline_y_mm", -10.0),
                    ),
                )
        fiber_preview_paths = _fiber_preview_paths_from_job(job)
        if core_source_job is None:
            write_external_source_npz(job, npz_path)
        progress(45, "Prusa 路径生成完成，正在交给 path_processing_core")

        path_count = sum(len(group.paths) for group in job.material_paths)
        recommendation = _triangle_infill_recommendation(mesh, config, job)
        slicing_metadata = job.meta.get("slicing", {})
        if not isinstance(slicing_metadata, dict):
            slicing_metadata = {}

        _ensure_offline_planner_import_paths()
        export_runner = importlib.import_module(
            "external_npz_preprocessor.export_runner"
        )
        process_params_module = importlib.import_module(
            "external_npz_preprocessor.process_params"
        )
        core_params = _parse_core_process_params(params, process_params_module)

        core_npz_path = job_dir / f"{Path(filename).stem}_core.npz"

        def core_progress(ratio: float) -> None:
            bounded = max(0.0, min(1.0, float(ratio)))
            progress(
                50 + int(bounded * 45),
                "正在执行 path_processing_core 并写出系统 NPZ",
            )

        if core_source_job is None:
            core_stats = export_runner.convert_external_npz(
                npz_path,
                core_npz_path,
                core_params,
                progress_callback=core_progress,
                chunk_size=5_000_000,
            )
        else:
            core_stats = export_runner.convert_source_job(
                core_source_job,
                source_path=native_gcode_path,
                output_path=core_npz_path,
                params=core_params,
                progress_callback=core_progress,
                chunk_size=5_000_000,
            )
        progress(98, "正在完成系统 NPZ 和时间元数据写入")
        # The browser is a source-trajectory inspector: it must show the
        # exact geometry written to the NPZ handed into Core, not Core's
        # fitted/resampled output.  In particular, a multi-waypoint travel
        # remains visibly routed around its avoidance vertices while Core
        # applies one zero-speed-endpoint profile to that complete route.
        preview = (
            _preview_payload_from_core_source_job(mesh, config, core_source_job)
            if core_source_job is not None
            else _preview_payload(mesh, config, job, fiber_preview_paths)
        )
        download_path = _core_output_download_path(core_npz_path)
        return {
            "download_url": f"/outputs/{quote(stamp)}/{quote(download_path.name)}",
            "filename": download_path.name,
            "layers": len(preview["layers"]),
            "paths": path_count,
            "preview": preview,
            "recommendation": recommendation,
            "fiber_json": fiber_json_name,
            "build_axis": build_axis,
            "slicing_kernel": config.slicing_kernel,
            "resolved_config": resolved_config,
            "effective_infill_pattern": slicing_metadata.get(
                "effective_infill_pattern",
                config.infill_pattern,
            ),
            "infill_pattern_execution": slicing_metadata.get(
                "infill_pattern_execution"
            ),
            "planning_line_width": float(
                config.line_width
                if (
                    config.slicing_kernel != "legacy"
                    or config.planning_line_width is None
                )
                else config.planning_line_width
            ),
            "core_export_seconds": float(core_stats.get("total_s", 0.0)),
            "core_rows": int(core_stats.get("rows", 0)),
            "core_parts": int(core_stats.get("parts", 0)),
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

    def _save_ui_settings(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("missing settings payload")
        raw = json.loads(self.rfile.read(content_length).decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("settings payload must be an object")

        core_values = raw.get("core", {})
        if isinstance(core_values, dict):
            _ensure_offline_planner_import_paths()
            process_module = importlib.import_module(
                "external_npz_preprocessor.process_params"
            )
            param_config = importlib.import_module(
                "external_npz_preprocessor.param_config"
            )
            core_form_values = _coerce_form_params(core_values)
            params = _parse_core_process_params(core_form_values, process_module)
            prusa_values = raw.get("prusa", {})
            if isinstance(prusa_values, dict):
                existing = _load_core_print_params()
                start_x = prusa_values.get("prusa_start_x_mm", existing.start_x_mm)
                start_y = prusa_values.get("prusa_start_y_mm", existing.start_y_mm)
                if start_x not in (None, "") and start_y not in (None, ""):
                    params = replace(
                        params,
                        start_x_mm=float(start_x),
                        start_y_mm=float(start_y),
                    )
            param_config.save_print_params(params, _core_print_params_path())

        prusa_values = raw.get("prusa", {})
        if isinstance(prusa_values, dict):
            current = _load_prusa_params()
            current.update(_normalize_prusa_params(prusa_values))
            _save_prusa_params(current)

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

    def _send_printhead_asset(self, asset_name: str) -> None:
        allowed_types = {
            ".glb": "model/gltf-binary",
            ".json": "application/json; charset=utf-8",
        }
        target = (PRINTHEAD_ASSET_DIR / unquote(asset_name)).resolve()
        if (
            target.parent != PRINTHEAD_ASSET_DIR.resolve()
            or target.suffix.lower() not in allowed_types
            or not target.is_file()
        ):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", allowed_types[target.suffix.lower()])
        cache_control = "no-cache" if target.suffix.lower() == ".json" else "public, max-age=3600"
        self.send_header("Cache-Control", cache_control)
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


def _parse_shared_slice_config(
    params: dict[str, list[str]],
    slicing_kernel: str,
) -> dict[str, Any]:
    """Read only inputs shared by every supported slicing backend."""

    prusa_defaults = _load_prusa_params()
    layer_height = _float_param(
        params,
        "layer_height",
        float(prusa_defaults.get("layer_height", DEFAULT_RESIN_LAYER_HEIGHT_MM)),
    )
    line_width = _float_param(
        params,
        "line_width",
        float(prusa_defaults.get("line_width", DEFAULT_RESIN_LINE_WIDTH_MM)),
    )
    tolerance = _optional_float_param(params, "tolerance")
    return {
        "slicing_kernel": slicing_kernel,
        "layer_height": layer_height,
        "first_layer_height": _float_param(
            params,
            "first_layer_height",
            float(prusa_defaults.get("first_layer_height", layer_height)),
        ),
        "line_width": line_width,
        "requested_build_axis": params.get(
            "build_axis", [str(prusa_defaults.get("build_axis", "auto"))]
        )[0],
        "z_min": _optional_float_param(params, "z_min"),
        "z_max": _optional_float_param(params, "z_max"),
        "tolerance": (
            tolerance
            if tolerance is not None
            else recommended_geometry_tolerance(layer_height, line_width)
        ),
    }


def _base_slice_config_fields(
    shared: dict[str, Any],
    build_axis: str,
) -> dict[str, Any]:
    return {
        "material": "R",
        "layer_height": shared["layer_height"],
        "first_layer_height": shared["first_layer_height"],
        "line_width": shared["line_width"],
        "z_min": shared["z_min"],
        "z_max": shared["z_max"],
        "tolerance": shared["tolerance"],
        "build_axis": build_axis,
    }


def _parse_core_process_params(
    params: dict[str, list[str]], module, defaults=None
) -> object:
    """Build process_core parameters from the dedicated KUKA settings panel."""

    defaults = defaults or _load_core_print_params()
    resin_defaults = defaults.resin
    fiber_defaults = defaults.fiber
    export_defaults = defaults.export
    resin = module.ResinProcessParams(
        layer_height_mm=_float_param(params, "core_resin_layer_height", resin_defaults.layer_height_mm),
        extrusion_scale=_float_param(params, "core_resin_extrusion_scale", resin_defaults.extrusion_scale),
        feed_mm_s=_float_param(params, "core_resin_feed", resin_defaults.feed_mm_s),
        first_layer_feed_mm_s=_float_param(params, "core_resin_first_layer_feed", resin_defaults.first_layer_feed_mm_s),
        temperature_c=_float_param(params, "core_resin_temp", resin_defaults.temperature_c),
        # Cooling is a fixed process requirement; it is intentionally not a UI option.
        fan_enabled=True,
        prime_length_mm=_float_param(params, "core_resin_prime_length", resin_defaults.prime_length_mm),
        prime_speed_mm_s=_float_param(params, "core_resin_prime_speed", resin_defaults.prime_speed_mm_s),
        retract_length_mm=_float_param(params, "core_resin_retract_length", resin_defaults.retract_length_mm),
        retract_speed_mm_s=_float_param(params, "core_resin_retract_speed", resin_defaults.retract_speed_mm_s),
        e_per_mm_override=_optional_float_param(params, "core_resin_e_override"),
    )
    fiber = module.FiberProcessParams(
        layer_height_mm=_float_param(params, "core_fiber_layer_height", fiber_defaults.layer_height_mm),
        extrusion_scale=_float_param(params, "core_fiber_extrusion_scale", fiber_defaults.extrusion_scale),
        feed_mm_s=_float_param(params, "core_fiber_feed", fiber_defaults.feed_mm_s),
        first_layer_feed_mm_s=_float_param(params, "core_fiber_first_layer_feed", fiber_defaults.first_layer_feed_mm_s),
        temperature_c=_float_param(params, "core_fiber_temp", fiber_defaults.temperature_c),
        fan_enabled=True,
        prime_length_mm=_float_param(params, "core_fiber_prime_length", fiber_defaults.prime_length_mm),
        prime_speed_mm_s=_float_param(params, "core_fiber_prime_speed", fiber_defaults.prime_speed_mm_s),
        retract_length_mm=_float_param(params, "core_fiber_retract_length", fiber_defaults.retract_length_mm),
        retract_speed_mm_s=_float_param(params, "core_fiber_retract_speed", fiber_defaults.retract_speed_mm_s),
        start_accel_s=_float_param(params, "core_fiber_start_accel", fiber_defaults.start_accel_s),
    )
    export = module.CoreExportParams(
        fiber_x_print_compensation_mm=_optional_float_param(params, "core_fiber_offset_x") if "core_fiber_offset_x" in params else export_defaults.fiber_x_print_compensation_mm,
        fiber_y_print_compensation_mm=_optional_float_param(params, "core_fiber_offset_y") if "core_fiber_offset_y" in params else export_defaults.fiber_y_print_compensation_mm,
        fiber_z_print_compensation_mm=_optional_float_param(params, "core_fiber_offset_z") if "core_fiber_offset_z" in params else export_defaults.fiber_z_print_compensation_mm,
        resin_z_print_compensation_mm=_optional_float_param(params, "core_resin_z_comp") if "core_resin_z_comp" in params else export_defaults.resin_z_print_compensation_mm,
        enable_extrude_wait=_bool_param(params, "core_enable_extrude_wait", export_defaults.enable_extrude_wait),
        enable_travel_extrude_overlap=_bool_param(
            params, "core_enable_travel_extrude_overlap", export_defaults.enable_travel_extrude_overlap
        ),
        initial_tool_id=_int_param(params, "core_initial_tool", export_defaults.initial_tool_id),
        tool_change_safe_lift_mm=_float_param(params, "core_tool_safe_lift", export_defaults.tool_change_safe_lift_mm),
        cut_lift_mm=_float_param(params, "core_cut_lift", export_defaults.cut_lift_mm),
        cut_wait_s=_float_param(params, "core_cut_wait", export_defaults.cut_wait_s),
        fiber_retract_length_mm=_optional_float_param(params, "core_fiber_retract_override") if "core_fiber_retract_override" in params else export_defaults.fiber_retract_length_mm,
        external_npz_cut_absolute_e=_bool_param(
            params, "core_external_npz_cut_absolute_e", export_defaults.external_npz_cut_absolute_e
        ),
    )
    return module.ProcessParams(
        resin=resin,
        fiber=fiber,
        travel_feed_mm_s=_float_param(params, "core_travel_feed", defaults.travel_feed_mm_s),
        first_layer_travel_feed_mm_s=_float_param(
            params, "core_first_layer_travel_feed", defaults.first_layer_travel_feed_mm_s
        ),
        default_a=_float_param(params, "core_default_a", defaults.default_a),
        default_b=_float_param(params, "core_default_b", defaults.default_b),
        default_c=_float_param(params, "core_default_c", defaults.default_c),
        # The Prusa UI normalizes the source job to this placement before it
        # is written. Pass the same placement into Core so the converter's
        # source-minimum normalization preserves the requested machine frame
        # instead of translating the material paths back to (0, 0).
        start_x_mm=_float_param(params, "prusa_start_x_mm", defaults.start_x_mm),
        start_y_mm=_float_param(params, "prusa_start_y_mm", defaults.start_y_mm),
        primeline_enabled=_bool_param(params, "core_primeline_enabled", defaults.primeline_enabled),
        primeline_x_mm=_float_param(params, "core_primeline_x_mm", defaults.primeline_x_mm),
        primeline_y_mm=_float_param(params, "core_primeline_y_mm", defaults.primeline_y_mm),
        primeline_length_mm=_float_param(params, "core_primeline_length", defaults.primeline_length_mm),
        prime_settle_s=_float_param(params, "core_prime_settle", defaults.prime_settle_s),
        dt=_float_param(params, "core_dt", defaults.dt),
        corner_angle_deg=_float_param(params, "core_corner_angle", defaults.corner_angle_deg),
        corner_retreat_ratio=_float_param(params, "core_corner_retreat_ratio", defaults.corner_retreat_ratio),
        spline_max_error_mm=_float_param(params, "core_spline_max_error", defaults.spline_max_error_mm),
        spline_max_angle_deg=_float_param(params, "core_spline_max_angle", defaults.spline_max_angle_deg),
        source_merge_distance_mm=_float_param(
            params, "core_source_merge_distance", defaults.source_merge_distance_mm
        ),
        corner_retreat_max_mm=_float_param(params, "core_corner_retreat_max", defaults.corner_retreat_max_mm),
        corner_blend_segments=_int_param(params, "core_corner_blend_segments", defaults.corner_blend_segments),
        density=_int_param(params, "core_density", defaults.density),
        degree=_int_param(params, "core_degree", defaults.degree),
        max_fit_points_per_segment=_int_param(params, "core_max_fit_points", defaults.max_fit_points_per_segment),
        export=export,
    )


def _prepend_prusa_startup_travel(
    job,
    *,
    start_xy: tuple[float, float],
    primeline_enabled: bool,
    primeline_xy: tuple[float, float],
) -> None:
    """Add the user-visible origin-to-first-motion travel to the Prusa source job."""

    if not job.material_paths:
        return
    first_layer_index = min(group.layer_index for group in job.material_paths)
    first_layer_travel = next(
        (group for group in job.travel_paths if group.layer_index == first_layer_index),
        None,
    )
    first_material = next(
        (group for group in job.material_paths if group.layer_index == first_layer_index),
        None,
    )
    if first_material is None or not first_material.paths:
        return
    first_z = float(first_material.paths[0][0, 2])
    if primeline_enabled:
        target = np.array(
            [
                float(start_xy[0]) + float(primeline_xy[0]),
                float(start_xy[1]) + float(primeline_xy[1]),
                first_z,
            ],
            dtype=np.float64,
        )
    elif first_layer_travel is not None and first_layer_travel.paths:
        target = np.asarray(first_layer_travel.paths[0][0, :3], dtype=np.float64)
    else:
        target = np.asarray(first_material.paths[0][0, :3], dtype=np.float64)
    start = np.array([0.0, 0.0, target[2]], dtype=np.float64)
    if float(np.linalg.norm(target - start)) <= 1e-7:
        return
    startup = np.vstack((start, target))
    if first_layer_travel is None:
        first_layer_travel = TravelPaths(first_layer_index, [])
        job.travel_paths.insert(0, first_layer_travel)
    first_layer_travel.paths.insert(0, startup)
    job.meta["startup_travel_count"] = 1
    job.meta["startup_travel_source_frame"] = "normalized_prusa"
    motion_order = job.meta.get("motion_order")
    if isinstance(motion_order, dict):
        records = motion_order.get(str(first_layer_index), [])
        if isinstance(records, list):
            shifted = []
            for record in records:
                if isinstance(record, dict) and record.get("kind") == "travel":
                    shifted.append({**record, "index": int(record.get("index", 0)) + 1})
                else:
                    shifted.append(record)
            motion_order[str(first_layer_index)] = [
                {"kind": "travel", "index": 0},
                *shifted,
            ]


def _parse_prusa_slice_config(
    params: dict[str, list[str]],
    shared: dict[str, Any],
    build_axis: str,
) -> SliceConfig:
    """Build Prusa configuration without reading Legacy or PySLM fields."""

    saved = _load_prusa_params()
    saved_params = {
        key: [
            ""
            if value is None
            else str(value).lower()
            if isinstance(value, bool)
            else str(value)
        ]
        for key, value in saved.items()
    }
    saved_params.update(params)
    params = saved_params
    raft_enabled = _bool_param(params, "prusa_raft_enabled", False)
    raft_contact_auto = _bool_param(params, "prusa_raft_auto_contact", True)
    prusa_raft = (
        PrusaRaftConfig(
            layer_count=_int_param(
                params, "prusa_raft_layers", DEFAULT_PRUSA_RAFT_LAYER_COUNT
            ),
            expansion=_float_param(
                params, "prusa_raft_expansion", DEFAULT_PRUSA_RAFT_EXPANSION_MM
            ),
            first_layer_density=_float_param(
                params,
                "prusa_raft_first_layer_density",
                DEFAULT_PRUSA_RAFT_FIRST_LAYER_DENSITY_PERCENT,
            ),
            first_layer_expansion=_float_param(
                params,
                "prusa_raft_first_layer_expansion",
                DEFAULT_PRUSA_RAFT_FIRST_LAYER_EXPANSION_MM,
            ),
            contact_distance=_float_param(
                params,
                "prusa_raft_contact_distance",
                DEFAULT_PRUSA_RAFT_CONTACT_DISTANCE_MM,
            ),
            contact_auto=raft_contact_auto,
            contact_layer_height=_float_param(
                params, "prusa_raft_contact_layer_height", 0.75
            ),
            contact_density=_float_param(
                params, "prusa_raft_contact_density", 100.0
            ),
            contact_extrusion_width=_float_param(
                params, "prusa_raft_contact_extrusion_width", 1.5
            ),
        )
        if raft_enabled
        else PrusaRaftConfig()
    )
    prusa_geometry = PrusaGeometryConfig(
        perimeter_generator=params.get("prusa_perimeter_generator", ["arachne"])[0],  # type: ignore[arg-type]
        gap_fill_enabled=_bool_param(params, "prusa_gap_fill_enabled", True),
        infill_anchor=_optional_float_param(params, "prusa_infill_anchor"),
        infill_anchor_max=_optional_float_param(params, "prusa_infill_anchor_max"),
        external_perimeter_width=_optional_float_param(
            params, "prusa_external_perimeter_width"
        ),
        perimeter_width=_optional_float_param(params, "prusa_perimeter_width"),
        infill_width=_optional_float_param(params, "prusa_infill_width"),
        xy_size_compensation=_float_param(params, "prusa_xy_size_compensation", 0.0),
        elephant_foot_compensation=_float_param(
            params, "prusa_elephant_foot_compensation", 0.0
        ),
        avoid_crossing_max_detour=_float_param(
            params, "prusa_avoid_crossing_max_detour", 0.0
        ),
        seam_position=params.get("prusa_seam_position", ["random"])[0],  # type: ignore[arg-type]
    )
    brim_enabled = _bool_param(params, "prusa_brim_enabled", False)
    return SliceConfig(
        **_base_slice_config_fields(shared, build_axis),
        slicing_kernel="prusa",
        curve_mode="flat",
        infill_pattern=params.get("prusa_infill_pattern", ["zigzag_horizontal"])[0],  # type: ignore[arg-type]
        infill_density=_float_param(
            params,
            "prusa_infill_density",
            DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
        ),
        contour_infill_overlap=_float_param(
            params,
            "prusa_contour_infill_overlap",
            DEFAULT_RESIN_CONTOUR_INFILL_OVERLAP_PERCENT,
        ),
        perimeter_count=_int_param(params, "prusa_perimeter_count", 2),
        print_perimeters=_bool_param(params, "prusa_print_perimeters", True),
        prusa_raft=prusa_raft,
        prusa_geometry=prusa_geometry,
        honeycomb_pathing=HoneycombPathingConfig(
            enabled=_bool_param(params, "honeycomb_centerline_enabled", False),
            # Saved UI states may still contain one of the removed topology
            # names.  The only supported mode is deliberately selected here
            # instead of allowing an old setting to make slicing fail.
            topology="macro_partition_zero_e",
        ),
        brim_enabled=brim_enabled,
        brim_width_mm=_float_param(params, "prusa_brim_width", 5.0),
        brim_type=params.get("prusa_brim_type", ["outer_only"])[0],  # type: ignore[arg-type]
        brim_separation_mm=_float_param(params, "prusa_brim_separation", 0.0),
        brim_one_stroke=_bool_param(params, "prusa_brim_one_stroke", False),
        start_x_mm=_float_param(params, "prusa_start_x_mm", 0.0),
        start_y_mm=_float_param(params, "prusa_start_y_mm", 0.0),
    )


def _parse_legacy_slice_config(
    params: dict[str, list[str]],
    shared: dict[str, Any],
    build_axis: str,
) -> SliceConfig:
    """Build the existing project-native configuration from Legacy fields."""

    return SliceConfig(
        **_base_slice_config_fields(shared, build_axis),
        slicing_kernel="legacy",
        planning_line_width=_float_param(
            params,
            "planning_line_width",
            DEFAULT_RESIN_PLANNING_LINE_WIDTH_MM,
        ),
        curve_mode=params.get("curve_mode", ["flat"])[0],  # type: ignore[arg-type]
        curve_amplitude=_float_param(params, "curve_amplitude", 0.0),
        curve_period=_float_param(params, "curve_period", 50.0),
        infill_pattern=params.get("infill_pattern", ["zigzag_horizontal"])[0],  # type: ignore[arg-type]
        infill_density=_float_param(
            params,
            "infill_density",
            DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
        ),
        infill_overlap=_float_param(
            params,
            "infill_overlap",
            DEFAULT_UI_RESIN_INFILL_OVERLAP_PERCENT,
        ),
        contour_infill_overlap=_float_param(
            params,
            "contour_infill_overlap",
            DEFAULT_RESIN_CONTOUR_INFILL_OVERLAP_PERCENT,
        ),
        triangle_path_optimization=_bool_param(
            params,
            "triangle_path_optimization",
            True,
        ),
        zigzag_path_optimization=_bool_param(
            params,
            "zigzag_path_optimization",
            True,
        ),
        perimeter_count=_int_param(params, "perimeter_count", 2),
        print_perimeters=_bool_param(params, "print_perimeters", True),
    )


def _parse_pyslm_slice_config(
    params: dict[str, list[str]],
    shared: dict[str, Any],
    build_axis: str,
) -> SliceConfig:
    """Build PySLM configuration without accepting Legacy path-planner fields."""

    strategy_defaults = recommended_pyslm_strategy_defaults(
        shared["layer_height"],
        shared["line_width"],
    )
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
        stripe_width=_float_param(params, "pyslm_stripe_width", strategy_defaults.width),
        stripe_overlap=_float_param(params, "pyslm_stripe_overlap", strategy_defaults.overlap),
        stripe_offset=_float_param(params, "pyslm_stripe_offset", strategy_defaults.offset),
        island_width=_float_param(params, "pyslm_island_width", strategy_defaults.width),
        island_overlap=_float_param(params, "pyslm_island_overlap", strategy_defaults.overlap),
        island_offset=_float_param(params, "pyslm_island_offset", strategy_defaults.offset),
        fix_polygons=_bool_param(params, "pyslm_fix_polygons", True),
        simplification_factor=_optional_float_param(params, "pyslm_simplification_factor"),
        simplification_preserve_topology=_bool_param(
            params,
            "pyslm_simplification_preserve_topology",
            True,
        ),
        simplification_mode=params.get("pyslm_simplification_mode", ["absolute"])[0],  # type: ignore[arg-type]
    )
    return SliceConfig(
        **_base_slice_config_fields(shared, build_axis),
        slicing_kernel="pyslm",
        curve_mode="flat",
        infill_pattern=params.get("pyslm_infill_pattern", ["zigzag_horizontal"])[0],  # type: ignore[arg-type]
        infill_density=_float_param(
            params,
            "pyslm_infill_density",
            DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
        ),
        infill_overlap=_float_param(
            params,
            "pyslm_infill_overlap",
            DEFAULT_UI_RESIN_INFILL_OVERLAP_PERCENT,
        ),
        contour_infill_overlap=_float_param(
            params,
            "pyslm_contour_infill_overlap",
            DEFAULT_RESIN_CONTOUR_INFILL_OVERLAP_PERCENT,
        ),
        perimeter_count=_int_param(params, "pyslm_perimeter_count", 2),
        print_perimeters=_bool_param(params, "pyslm_print_perimeters", True),
        pyslm=pyslm_config,
    )


def _resolved_slice_config(config: SliceConfig) -> dict[str, object]:
    """Small auditable record of exactly the settings passed to one backend."""

    resolved: dict[str, object] = {
        "kernel": config.slicing_kernel,
        "layer_height": config.layer_height,
        "first_layer_height": config.first_layer_height,
        "line_width": config.line_width,
        "build_axis": config.build_axis,
        "z_min": config.z_min,
        "z_max": config.z_max,
        "start_x_mm": config.start_x_mm,
        "start_y_mm": config.start_y_mm,
        "tolerance": config.tolerance,
        "perimeter_count": config.perimeter_count,
        "print_perimeters": config.print_perimeters,
        "infill_pattern": config.infill_pattern,
        "infill_density": config.infill_density,
        "contour_infill_overlap": config.contour_infill_overlap,
    }
    if config.slicing_kernel == "prusa":
        resolved.update(
            path_backend="prusa_fff",
            prusa_perimeter_infill_overlap=config.contour_infill_overlap,
            prusa_raft=config.prusa_raft.to_metadata(),
            prusa_brim={
                "enabled": config.brim_enabled,
                "width": config.brim_width_mm,
                "type": config.brim_type,
                "separation": config.brim_separation_mm,
                "one_stroke": config.brim_one_stroke,
            },
            prusa_geometry=config.prusa_geometry.to_metadata(),
        )
    elif config.slicing_kernel == "legacy":
        resolved.update(
            path_backend="legacy",
            planning_line_width=config.planning_line_width,
            infill_overlap=config.infill_overlap,
            triangle_path_optimization=config.triangle_path_optimization,
            zigzag_path_optimization=config.zigzag_path_optimization,
            curve_mode=config.curve_mode,
        )
    else:
        resolved.update(
            path_backend="pyslm",
            infill_overlap=config.infill_overlap,
            pyslm_hatcher=config.pyslm.hatcher,
            pyslm_hatch_sort=config.pyslm.hatch_sort,
        )
    return resolved


def _raft_layers_from_params(params: dict[str, list[str]]) -> list[RaftLayerConfig]:
    if not _bool_param(params, "print_raft", True):
        return []
    layer_count = DEFAULT_RAFT_LAYER_COUNT
    default_offsets = ",".join(f"{offset:g}" for offset in DEFAULT_RAFT_OUTWARD_OFFSETS_MM)
    offsets = _float_list_param(params, "raft_offsets", default_offsets, layer_count)
    return [
        RaftLayerConfig(
            outward_offset=offsets[index],
            layer_height=DEFAULT_RESIN_LAYER_HEIGHT_MM,
            infill_density=DEFAULT_RESIN_INFILL_DENSITY_PERCENT,
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
    coordinate_system: str | None = None
    if isinstance(data, dict):
        raw_coordinate_system = data.get("coordinate_system")
        if raw_coordinate_system is not None:
            if not isinstance(raw_coordinate_system, str):
                raise ValueError("fiber JSON coordinate_system must be a string")
            coordinate_system = raw_coordinate_system
        # Canonical fiber paths use one record per trajectory under ``paths``.
        if "paths" in data:
            data = data["paths"]
        # Optimized results contain metadata and merged final paths.  They are
        # kept compatible as geometry-only input.
        elif "final_paths" in data:
            data = data["final_paths"]
        else:
            # Keep accepting the original compact path-family format, where
            # every top-level key is a family and its value is a path list.
            merged_paths: list[object] = []
            for family_name, family_paths in data.items():
                if not isinstance(family_paths, list):
                    raise ValueError(
                        f"fiber JSON path family {family_name!r} must be a list"
                    )
                merged_paths.extend(family_paths)
            data = merged_paths
    if not isinstance(data, list):
        raise ValueError("fiber JSON paths/final_paths must be a list")

    paths: list[list[list[float]]] = []
    for path_index, raw_path in enumerate(data):
        if isinstance(raw_path, dict):
            raw_path = raw_path.get("points")
            if raw_path is None:
                raise ValueError(
                    f"fiber JSON path {path_index} must contain a points list"
                )
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
    return FiberTemplatePaths(paths, coordinate_system=coordinate_system)


def align_fiber_template_paths_to_resin(
    job,
    template_paths: list[list[list[float]]],
) -> list[list[list[float]]]:
    """Map declared fiber XY coordinates into the unnormalized resin frame.

    ``project_default`` is the coordinate convention emitted by the fiber
    planner: (0, 0) is the center of the STL build plane.  Slicer paths retain
    the STL's local XY translation until the later UI placement step, so map
    that origin to the resin bounding-box center before fibers are injected.
    The later normalization then applies the existing UI start/travel offset
    equally to both materials.
    """

    coordinate_system = getattr(template_paths, "coordinate_system", None)
    if coordinate_system is None or coordinate_system == "slicer_xy":
        return template_paths
    if coordinate_system != "project_default":
        raise ValueError(
            "fiber JSON coordinate_system must be project_default or slicer_xy"
        )

    resin_paths = [
        np.asarray(path, dtype=np.float64)
        for group in job.material_paths
        if group.material == "R"
        for path in group.paths
        if np.asarray(path).size
    ]
    if not resin_paths:
        raise ValueError("cannot align project_default fiber JSON without resin paths")
    resin_points = np.vstack(resin_paths)
    translation_x = float((np.min(resin_points[:, 0]) + np.max(resin_points[:, 0])) * 0.5)
    translation_y = float((np.min(resin_points[:, 1]) + np.max(resin_points[:, 1])) * 0.5)
    aligned_paths = [
        [[float(x) + translation_x, float(y) + translation_y, float(z)] for x, y, z in path]
        for path in template_paths
    ]
    job.meta["fiber_coordinate_alignment"] = {
        "source_coordinate_system": coordinate_system,
        "reference": "resin_xy_bounds_center",
        "translation_x_mm": translation_x,
        "translation_y_mm": translation_y,
    }
    return aligned_paths


def expand_fiber_template_for_resin_layers(
    job, template_paths: list[list[list[float]]]
) -> dict[int, list[list[list[float]]]]:
    resin_groups = [group for group in job.material_paths if group.material == "R"]
    resin_groups.sort(key=lambda group: group.layer_index)
    paths_by_layer: dict[int, list[list[list[float]]]] = {}

    slicing_metadata = job.meta.get("slicing")
    raft_layer_count = 0
    if isinstance(slicing_metadata, dict):
        prusa_raft = slicing_metadata.get("prusa_raft")
        if isinstance(prusa_raft, dict):
            raw_count = prusa_raft.get("layer_count", 0)
            if isinstance(raw_count, int) and raw_count > 0:
                raft_layer_count = raw_count
    part_resin_groups = resin_groups[raft_layer_count:]
    fiber_reference_z_by_layer = {
        int(group.layer_index): _group_layer_z(group)
        for group in part_resin_groups
    }

    # A brim is printed on the first part resin layer, but fiber should start
    # only above the following resin layer.  Keep the normal resin/fiber
    # schedule otherwise: skipping this one fiber layer also removes its
    # 0.1 mm contribution from all subsequent absolute Z values.
    first_part_has_brim = False
    if part_resin_groups:
        roles_by_layer = job.meta.get("path_roles", {}).get("R", {})
        if isinstance(roles_by_layer, dict):
            roles = roles_by_layer.get(str(part_resin_groups[0].layer_index), [])
            first_part_has_brim = isinstance(roles, list) and "brim" in roles
    skipped_fiber_layers = 1 if first_part_has_brim else 0

    # The fiber is physically printed between resin layers.  Include its
    # thickness in the exported Z schedule instead of letting every resin
    # layer continue to use the original resin-only grid.  Prusa travel paths
    # are part of the same physical layer geometry: apply the identical shift
    # below so their endpoints remain continuous with the deposited paths.
    fiber_layer_height = DEFAULT_FIBER_LAYER_HEIGHT_MM
    z_offset_by_resin_layer: dict[int, float] = {}
    for layer_order, group in enumerate(part_resin_groups):
        z_offset = max(0, layer_order - skipped_fiber_layers) * fiber_layer_height
        z_offset_by_resin_layer[int(group.layer_index)] = z_offset
        if z_offset == 0.0:
            continue
        group.paths = [
            np.asarray(path, dtype=np.float64).copy()
            for path in group.paths
        ]
        for path in group.paths:
            path[:, 2] += z_offset

    # A travel group is emitted by Prusa in the motion order of its matching
    # resin layer.  Once a fiber layer has been inserted below that resin
    # layer, retaining Prusa's original Z would leave an artificial vertical
    # gap immediately before the next deposited path.  Keep the native XY
    # route intact and translate only Z by that layer's accumulated fiber
    # thickness.
    for group in job.travel_paths:
        z_offset = z_offset_by_resin_layer.get(int(group.layer_index), 0.0)
        if z_offset == 0.0:
            continue
        group.paths = [
            np.asarray(path, dtype=np.float64).copy()
            for path in group.paths
        ]
        for path in group.paths:
            path[:, 2] += z_offset

    if isinstance(slicing_metadata, dict) and part_resin_groups:
        z_max = slicing_metadata.get("z_max")
        if isinstance(z_max, (int, float)):
            inserted_fiber_layers = max(0, len(part_resin_groups) - 1 - skipped_fiber_layers)
            slicing_metadata["z_max"] = float(z_max) + inserted_fiber_layers * fiber_layer_height
        slicing_metadata["fiber_layer_height_applied_mm"] = fiber_layer_height
        slicing_metadata["fiber_layers_skipped_for_brim"] = skipped_fiber_layers

    # The physical fiber Z accumulates earlier fiber courses.  Routing must
    # inspect the same unshifted STL section that produced the resin layer,
    # while the connector itself retains its raised output Z.
    job.meta["fiber_interpath_reference_z_mm"] = {
        str(layer_index): float(z)
        for layer_index, z in fiber_reference_z_by_layer.items()
    }

    # Fiber is printed between resin layers; the final resin layer is a cap.
    for group in part_resin_groups[skipped_fiber_layers:-1]:
        z = _group_layer_z(group) + fiber_layer_height
        layer_paths = []
        for template_path in template_paths:
            layer_paths.append(_fiber_template_path_at_z(template_path, z))
        layer_paths = [
            path.tolist()
            for path in optimize_open_path_travel(
                [np.asarray(path, dtype=np.float64) for path in layer_paths]
            )
        ]
        paths_by_layer[group.layer_index] = layer_paths
    return paths_by_layer


def _fiber_template_path_at_z(
    template_path: list[list[float]],
    z: float,
) -> list[list[float]]:
    return [[float(x), float(y), z] for x, y, _ in template_path]


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


def merge_fiber_paths_into_job(
    job,
    fiber_paths_by_layer: dict[int, list[list[list[float]]]],
    fiber_travel_paths_by_layer: dict[int, list[np.ndarray]] | None = None,
) -> None:
    """Attach fiber deposition and only its explicit interpath travels.

    Native resin paths and their Prusa travel order are copied verbatim.  The
    appended records describe the UI-planned fiber sequence exclusively, so
    the pre-Core source NPZ and the native-G-code Core adapter share identical
    fiber travel geometry without changing any resin motion.
    """

    fiber_travel_paths_by_layer = fiber_travel_paths_by_layer or {}
    existing = {(group.layer_index, group.material) for group in job.material_paths}
    for layer_index in sorted(fiber_paths_by_layer):
        if (layer_index, "F") in existing:
            continue
        paths = [np.asarray(path, dtype=np.float64) for path in fiber_paths_by_layer[layer_index]]
        connector_paths = [
            np.asarray(path, dtype=np.float64)
            for path in fiber_travel_paths_by_layer.get(layer_index, [])
        ]
        if len(connector_paths) != max(0, len(paths) - 1):
            raise ValueError(
                f"fiber layer {layer_index} needs {max(0, len(paths) - 1)} interpath travels, "
                f"received {len(connector_paths)}"
            )
        if paths:
            job.material_paths.append(MaterialPaths(layer_index, "F", paths))
            travel_group = next(
                (group for group in job.travel_paths if group.layer_index == layer_index),
                None,
            )
            if travel_group is None:
                travel_group = TravelPaths(layer_index, [])
                job.travel_paths.append(travel_group)
            first_travel_index = len(travel_group.paths)
            travel_group.paths.extend(connector_paths)
            motion_root = job.meta.get("motion_order")
            if isinstance(motion_root, dict):
                records = motion_root.setdefault(str(layer_index), [])
                if isinstance(records, list):
                    for fiber_index in range(len(paths)):
                        if fiber_index:
                            records.append(
                                {
                                    "kind": "fiber_travel",
                                    "index": first_travel_index + fiber_index - 1,
                                }
                            )
                        records.append({"kind": "fiber_deposit", "index": fiber_index})
    job.material_paths.sort(key=lambda group: (group.layer_index, 0 if group.material == "R" else 1))
    job.travel_paths.sort(key=lambda group: group.layer_index)


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
    mesh,
    config: SliceConfig,
    job,
    fiber_paths_by_layer: dict[int, list[list[list[float]]]] | None = None,
    *,
    hide_initial_prusa_travel: bool = False,
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
    has_curved_deposition = False
    has_tool_orientation = False
    resin_roles_by_layer = (
        job.meta.get("path_roles", {}).get("R", {})
        if isinstance(job.meta.get("path_roles", {}), dict)
        else {}
    )
    motion_order_by_layer = (
        job.meta.get("motion_order", {})
        if isinstance(job.meta.get("motion_order", {}), dict)
        else {}
    )
    groups_by_layer: dict[int, dict[str, list]] = {}
    for group in job.material_paths:
        groups_by_layer.setdefault(group.layer_index, {}).setdefault(group.material, []).append(group)
    travel_groups_by_layer: dict[int, list] = {}
    for group in job.travel_paths:
        travel_groups_by_layer.setdefault(group.layer_index, []).append(group)

    layer_indices = {
        group.layer_index for group in job.material_paths
    } | set(fiber_paths_by_layer) | set(travel_groups_by_layer)
    first_material_layer = min(layer_indices) if layer_indices else None
    startup_travel_count = 0
    try:
        startup_travel_count = max(0, int(job.meta.get("startup_travel_count", 0)))
    except (TypeError, ValueError):
        startup_travel_count = 0

    for layer_index in sorted(layer_indices):
        resin_paths: list[dict[str, object]] = []
        preview_resin_indices: dict[int, list[int]] = {}
        group_resin_index = 0
        for group in groups_by_layer.get(layer_index, {}).get("R", []):
            layer_roles = resin_roles_by_layer.get(str(layer_index), [])
            for path_index, path in enumerate(group.paths):
                raw_points = [_serialize_preview_point(point) for point in path]
                has_curved_deposition = has_curved_deposition or _preview_path_varies_in_z(raw_points)
                has_tool_orientation = has_tool_orientation or _preview_path_has_orientation(raw_points)
                raw_extrusion = (
                    group.extrusion[path_index]
                    if group.extrusion is not None and path_index < len(group.extrusion)
                    else None
                )
                source_resin_index = group_resin_index
                role = (
                    layer_roles[source_resin_index]
                    if isinstance(layer_roles, list) and source_resin_index < len(layer_roles)
                    else None
                )
                group_resin_index += 1
                preview_role = role if role in ("outer_contour", "inner_contour", "infill", "raft", "brim") else (
                    "outer_contour" if path.shape[0] > 2 else "infill"
                )
                # Rendering must never uniformly decimate a long polyline:
                # that joins unrelated honeycomb nodes with false chords.
                # Honeycomb macro partitions must remain one logical preview
                # path so the slider and playback show the actual full
                # execution flow.  Other very long paths keep their bounded
                # transport chunks to protect general preview responsiveness.
                preview_chunk_size = 16_000 if role == "honeycomb_wall" else 2_000
                chunk_indices: list[int] = []
                for points, extrusion in _preview_path_chunks(
                    raw_points,
                    raw_extrusion,
                    max_points=preview_chunk_size,
                ):
                    entry: dict[str, object] = {"role": preview_role, "points": points}
                    if extrusion is not None:
                        entry["extrusion"] = extrusion
                    chunk_indices.append(len(resin_paths))
                    resin_paths.append(entry)
                    for point in points:
                        _expand_bounds(bounds, point[0], point[1], point[2])
                preview_resin_indices[source_resin_index] = chunk_indices

        # The preview is a source-NPZ geometry inspector.  Never decimate a
        # deposited or travel route: a skipped point can turn a hole-avoiding
        # path into a false chord on the Canvas.
        serialized_fiber_paths = [
            [list(point) for point in path]
            for path in fiber_paths_by_layer.get(layer_index, [])
        ]
        if not serialized_fiber_paths:
            for group in groups_by_layer.get(layer_index, {}).get("F", []):
                serialized_fiber_paths.extend(
                    [_serialize_preview_point(point) for point in path]
                    for path in group.paths
                )
        for fiber_path in serialized_fiber_paths:
            has_curved_deposition = has_curved_deposition or _preview_path_varies_in_z(fiber_path)
            has_tool_orientation = has_tool_orientation or _preview_path_has_orientation(fiber_path)
            for point in fiber_path:
                _expand_bounds(bounds, point[0], point[1], point[2])

        serialized_travel_paths: list[list[list[float]]] = []
        for group in travel_groups_by_layer.get(layer_index, []):
            serialized_travel_paths.extend(
                [_serialize_preview_point(point) for point in path]
                for path in group.paths
            )
        for travel_path in serialized_travel_paths:
            for point in travel_path:
                _expand_bounds(bounds, point[0], point[1], point[2])

        motion_paths: list[dict[str, object]] = []
        motion_order = motion_order_by_layer.get(str(layer_index), [])
        if isinstance(motion_order, list):
            for motion in motion_order:
                if not isinstance(motion, dict):
                    continue
                kind = motion.get("kind")
                index = motion.get("index")
                if not isinstance(index, int):
                    continue
                if kind == "deposit" and index in preview_resin_indices:
                    for preview_index in preview_resin_indices[index]:
                        source = resin_paths[preview_index]
                        motion_paths.append(
                            {
                                "kind": "deposit",
                                "role": source["role"],
                                "points": source["points"],
                                "extrusion": source.get("extrusion"),
                            }
                        )
                elif kind == "fiber_deposit" and 0 <= index < len(serialized_fiber_paths):
                    motion_paths.append(
                        {
                            "kind": "deposit",
                            "role": "fiber",
                            "points": serialized_fiber_paths[index],
                        }
                    )
                elif kind == "travel" and 0 <= index < len(serialized_travel_paths):
                    if (
                        hide_initial_prusa_travel
                        and layer_index == first_material_layer
                        and index == startup_travel_count
                    ):
                        continue
                    motion_paths.append(
                        {"kind": "travel", "points": serialized_travel_paths[index]}
                    )
                elif kind == "fiber_travel" and 0 <= index < len(serialized_travel_paths):
                    motion_paths.append(
                        {
                            "kind": "travel",
                            "role": "travel",
                            "points": serialized_travel_paths[index],
                        }
                    )
        if not motion_paths:
            motion_paths.extend(
                {
                    "kind": "deposit",
                    "role": path["role"],
                    "points": path["points"],
                    "extrusion": path.get("extrusion"),
                }
                for path in resin_paths
            )
            for travel_index, path in enumerate(serialized_travel_paths):
                if (
                    hide_initial_prusa_travel
                    and layer_index == first_material_layer
                    and travel_index == startup_travel_count
                ):
                    continue
                motion_paths.append({"kind": "travel", "points": path})

        layers_by_index[layer_index] = {
            "index": layer_index,
            "resin_paths": resin_paths,
            "fiber_paths": serialized_fiber_paths,
            "travel_paths": serialized_travel_paths,
            "motion_paths": motion_paths,
        }

    planning_line_width = (
        config.line_width
        if config.slicing_kernel != "legacy" or config.planning_line_width is None
        else config.planning_line_width
    )
    return {
        "bounds": bounds,
        "origin": [0.0, 0.0],
        # A flat job may span many layers in Z.  The view changes only when a
        # depositing path itself varies in Z, which is the signature of a
        # mapped surface rather than ordinary planar slicing.
        "geometry_mode": "surface_3d" if has_curved_deposition else "planar_2d",
        "tool_orientation": {
            "available": has_tool_orientation,
            "fallback": "calibrated_flat_downward",
        },
        "line_widths": {
            "resin": float(planning_line_width),
            "resin_nominal": float(config.line_width),
            "fiber": DEFAULT_FIBER_LINE_WIDTH_MM,
        },
        "preview_source": "pre_core_source_npz",
        "layers": list(layers_by_index.values()),
    }


def _preview_payload_from_source_npz(source_bytes: bytes, source_name: str) -> dict[str, object]:
    """Adapt a legacy mapped or conformal external NPZ to the main preview schema."""

    # The mapper contract remains the authority for validation.  This adapter
    # only removes NaN padding and forwards XYZABC to the shared Canvas view;
    # it does not perform mapping, interpolation, or any Core processing.
    from .surface_mapper.contracts import read_source_npz

    source = read_source_npz(source_bytes, source_name=source_name)
    material_paths: list[MaterialPaths] = []
    travel_paths: list[TravelPaths] = []
    key_pattern = re.compile(r"^layer_(\d{4,})_([RFT])$")
    for key in source.path_keys:
        match = key_pattern.match(key)
        if match is None:
            continue
        layer_index = int(match.group(1))
        material = match.group(2)
        paths = [
            np.asarray(path[np.isfinite(path[:, 0])], dtype=np.float64)
            for path in np.asarray(source.arrays[key])
            if np.isfinite(path[:, 0]).any()
        ]
        if material == "T":
            if paths:
                travel_paths.append(TravelPaths(layer_index, paths))
            continue
        extrusion_key = f"{key}_E"
        extrusion = None
        if extrusion_key in source.arrays:
            extrusion = [
                np.asarray(values[np.isfinite(values)], dtype=np.float64)
                for values in np.asarray(source.arrays[extrusion_key])
                if np.isfinite(values).any()
            ]
        if paths:
            material_paths.append(MaterialPaths(layer_index, material, paths, extrusion))

    slicing_meta = source.meta.get("slicing")
    resolved_config = slicing_meta.get("resolved_config", {}) if isinstance(slicing_meta, dict) else {}
    line_width = resolved_config.get("line_width", DEFAULT_RESIN_LINE_WIDTH_MM)
    try:
        config = SliceConfig(line_width=float(line_width))
    except (TypeError, ValueError):
        config = SliceConfig(line_width=DEFAULT_RESIN_LINE_WIDTH_MM)
    preview = _preview_payload(
        None,
        config,
        ExternalSourceJob(
            material_paths=material_paths,
            travel_paths=travel_paths,
            meta=source.meta,
        ),
    )
    bridge = source.meta.get("conformal_lattice_path_bridge")
    if isinstance(bridge, dict):
        preview["preview_source"] = "conformal_lattice_external_source_npz"
        preview["conformal_lattice"] = {
            "edge_count_per_layer": int(bridge.get("edge_count_per_layer", len(material_paths[0].paths) if material_paths else 0)),
            "path_order": bridge.get("path_order"),
            "uses_existing_main_canvas": True,
            "planning_line_width_mm": float(config.line_width),
        }
    return preview


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


def _preview_path_chunks(
    points: list[list[float]],
    extrusion_values,
    *,
    max_points: int,
) -> list[tuple[list[list[float]], list[float] | None]]:
    """Split a display-only path without dropping or reordering source points."""

    if max_points < 2:
        raise ValueError("preview chunk size must be at least two")
    values = (
        [float(value) for value in extrusion_values]
        if extrusion_values is not None and len(extrusion_values) == len(points)
        else None
    )
    chunks = []
    start = 0
    while start < len(points):
        end = min(start + max_points, len(points))
        chunks.append((points[start:end], None if values is None else values[start:end]))
        if end == len(points):
            break
        start = end - 1
    return chunks


def _serialize_preview_point(point: object) -> list[float]:
    """Keep XYZABC when a mapped source path supplies a KUKA orientation."""

    values = [float(point[index]) for index in range(3)]
    try:
        orientation = [float(point[index]) for index in range(3, 6)]
    except (IndexError, TypeError):
        return values
    if all(np.isfinite(value) for value in orientation):
        values.extend(orientation)
    return values


def _preview_path_varies_in_z(points: list[list[float]]) -> bool:
    if len(points) < 2:
        return False
    z_values = [point[2] for point in points]
    return max(z_values) - min(z_values) > 1e-5


def _preview_path_has_orientation(points: list[list[float]]) -> bool:
    return any(len(point) >= 6 for point in points)


def _load_fiber_preview_paths(npz_path: Path) -> dict[int, list[list[list[float]]]]:
    paths_by_layer: dict[int, list[list[list[float]]]] = {}
    key_pattern = re.compile(r"^layer_(\d{4})_F$")
    with np.load(npz_path, allow_pickle=False) as archive:
        for key in archive.files:
            match = key_pattern.match(key)
            if not match:
                continue
            layer_index = int(match.group(1))
            array = np.asarray(archive[key])
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
    core_defaults = _load_core_print_params()
    prusa_saved = _load_prusa_params()

    def prusa_value(name: str, fallback: object) -> object:
        return prusa_saved.get(name, fallback)

    def prusa_num(name: str, fallback: float) -> str:
        value = prusa_value(name, fallback)
        return "" if value is None or value == "" else f"{float(value):g}"

    def prusa_checked(name: str, fallback: bool) -> str:
        return " checked" if bool(prusa_value(name, fallback)) else ""

    def prusa_selected(name: str, fallback: str, option: str) -> str:
        return " selected" if str(prusa_value(name, fallback)) == option else ""

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
  <title>机械臂空间复合材料增材制造系统切片器</title>
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
      --space-1: 4px;
      --space-2: 8px;
      --space-3: 12px;
      --space-4: 16px;
      --space-5: 20px;
      --space-6: 24px;
      --radius-sm: 6px;
      --radius-md: 8px;
      --control-height: 40px;
      --z-tooltip: 10;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: #f5f7f9;
      line-height: 1.5;
    }}
    header {{
      min-height: 68px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-4);
      padding: 14px max(20px, calc((100vw - 960px) / 2));
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      line-height: 1.35;
      letter-spacing: 0;
      text-wrap: balance;
    }}
    main {{
      max-width: 960px;
      margin: 0 auto;
      padding: 28px 20px 36px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
      align-items: start;
    }}
    section {{
      min-width: 0;
    }}
    .resultsColumn {{
      min-width: 0;
      position: static;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #ffffff;
      padding: var(--space-5);
      box-shadow: 0 2px 8px rgba(23, 32, 38, 0.04);
    }}
    h2 {{
      margin: 0 0 var(--space-5);
      font-size: 16px;
      font-weight: 650;
      letter-spacing: 0;
      line-height: 1.35;
      text-wrap: balance;
    }}
    .subhead {{
      margin-top: var(--space-5);
      padding-top: var(--space-4);
      border-top: 1px solid var(--line);
    }}
    .formSection {{
      margin-top: 0;
      padding: var(--space-2);
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      background: #fbfcfd;
    }}
    .formSection:first-of-type {{
      margin-top: 0;
      padding-top: var(--space-2);
    }}
    #sliceForm {{
      display: grid;
      grid-template-columns: 1fr;
      gap: var(--space-1);
      align-items: start;
    }}
    #sliceForm > .formSection {{
      grid-column: 1 / -1;
    }}
    .formSection h3 {{
      margin: 0;
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0;
      color: var(--ink);
      line-height: 1.4;
      text-wrap: balance;
    }}
    label {{
      display: block;
      margin: var(--space-3) 0 var(--space-1);
      font-size: 13px;
      color: var(--muted);
      line-height: 1.35;
      text-wrap: pretty;
    }}
    .inputBand label,
    .kernelBand label,
    .processBand label {{
      margin-top: var(--space-1);
    }}
    .labelWithHelp {{
      display: flex;
      align-items: center;
      gap: var(--space-2);
    }}
    .helpTip {{
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 16px;
      height: 16px;
      border: 1px solid #8aa0b8;
      border-radius: 50%;
      color: #47627e;
      font-size: 11px;
      font-weight: 700;
      cursor: help;
      outline: none;
    }}
    .helpTip::after {{
      content: attr(data-tooltip);
      position: absolute;
      left: 50%;
      bottom: calc(100% + 8px);
      z-index: var(--z-tooltip);
      width: min(360px, 75vw);
      padding: 9px 11px;
      border-radius: 6px;
      background: #243447;
      color: #fff;
      font-size: 12px;
      font-weight: 400;
      line-height: 1.5;
      white-space: normal;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
      opacity: 0;
      visibility: hidden;
      transform: translate(-50%, 3px);
      transition: opacity 0.12s ease-out, transform 0.12s ease-out;
      pointer-events: none;
    }}
    .helpTip:hover::after,
    .helpTip:focus::after {{
      opacity: 1;
      visibility: visible;
      transform: translate(-50%, 0);
    }}
    .tooltipLabel[data-tooltip] {{
      position: relative;
      width: fit-content;
      max-width: 100%;
      cursor: help;
      text-decoration: underline dotted rgba(92, 105, 114, 0.7);
      text-underline-offset: 3px;
    }}
    .tooltipLabel[data-tooltip]::after {{
      content: attr(data-tooltip);
      position: absolute;
      z-index: var(--z-tooltip);
      top: calc(100% + 6px);
      left: 0;
      width: max-content;
      max-width: min(340px, calc(100vw - 48px));
      padding: 8px 10px;
      border: 1px solid #27313a;
      background: #ffffff;
      color: var(--ink);
      box-shadow: 0 4px 12px rgba(23, 32, 38, 0.16);
      font-size: 13px;
      font-weight: 400;
      line-height: 1.45;
      text-decoration: none;
      white-space: normal;
      pointer-events: none;
      visibility: hidden;
      opacity: 0;
      transform: translateY(-2px);
      transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
    }}
    .tooltipLabel[data-tooltip]:hover::after,
    .tooltipLabel[data-tooltip]:focus-within::after {{
      visibility: visible;
      opacity: 1;
      transform: translateY(0);
    }}
    input:not([type="checkbox"]):not([type="range"]), select {{
      width: 100%;
      height: var(--control-height);
      min-height: var(--control-height);
      border: 1px solid #bcc5cd;
      border-radius: var(--radius-sm);
      padding: 7px 10px;
      font: inherit;
      background: #ffffff;
      color: var(--ink);
    }}
    input:focus-visible, select:focus-visible, button:focus-visible, a:focus-visible, summary:focus-visible {{
      outline: 3px solid rgba(11, 107, 203, 0.24);
      outline-offset: 2px;
      border-color: var(--accent);
    }}
    .surfaceTools {{
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2);
      justify-content: flex-end;
    }}
    .surfaceToolButton {{
      min-height: 34px;
      padding: 6px 10px;
      border: 1px solid #8aa0b8;
      border-radius: var(--radius-sm);
      color: #084f96;
      background: #ffffff;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }}
      .surfaceToolButton:hover {{ background: #eef6ff; }}
      .surfaceToolButton:disabled {{ cursor: wait; opacity: 0.7; }}
      .surfaceCollisionResult {{
        flex: 1 1 100%;
        min-height: 18px;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.35;
      }}
      .surfaceCollisionResult.ok {{ color: var(--ok); }}
      .surfaceCollisionResult.error {{ color: var(--error); }}
    .inputBand input[type="file"] {{
      padding: 0;
      line-height: calc(var(--control-height) - 2px);
    }}
    input[type="file"]::file-selector-button {{
      box-sizing: border-box;
      height: 100%;
      margin: 0 10px 0 0;
      border: 0;
      border-right: 1px solid #bcc5cd;
      border-radius: 0;
      padding: 0 12px;
      font: inherit;
      background: var(--panel);
      color: var(--ink);
      cursor: pointer;
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: var(--space-2);
    }}
    .bandGrid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: var(--space-2);
      align-items: start;
    }}
    .inputBand .bandGrid {{
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) 94px 94px 140px;
    }}
    .sectionTitleRow {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-3);
    }}
    .sectionTitleRow .fiberNotice {{
      flex: 1 1 auto;
      min-width: 0;
      margin: 0;
      text-align: right;
    }}
    .prusaSettingsGrid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
      gap: var(--space-2);
      align-items: end;
    }}
    .prusaSettingsGrid > * {{ grid-column: auto; }}
    .prusaFeatureToggle {{
      display: flex;
      align-items: center;
      min-height: var(--control-height);
      margin-top: var(--space-2);
    }}
    .prusaSubSettings {{
      margin-top: var(--space-2);
      padding: var(--space-2);
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #fbfcfd;
    }}
    .prusaAdvancedGroup + .prusaAdvancedGroup {{
      margin-top: var(--space-3);
      padding-top: var(--space-3);
      border-top: 1px solid var(--line);
    }}
    .prusaAdvancedGroup h4 {{
      margin: 0 0 var(--space-2);
      font-size: 13px;
      color: var(--ink);
    }}
    input[type="number"] {{
      appearance: auto;
      -moz-appearance: auto;
    }}
    input[type="number"]::-webkit-inner-spin-button,
    input[type="number"]::-webkit-outer-spin-button {{
      margin: 0;
      -webkit-appearance: auto;
    }}
    .magnitudeInputWrap {{
      display: block;
      width: 100%;
      min-width: 0;
    }}
    .magnitudeInputWrap > input[type="number"] {{
      padding-right: 8px;
    }}
    .magnitudeSpinButtons {{
      display: none;
    }}
    .coreParameterSubgroup {{
      margin-top: var(--space-2);
    }}
    .coreParameterSubgroup:first-of-type {{
      margin-top: 0;
    }}
    .coreParameterSubgroup h5 {{
      margin: 0 0 var(--space-1);
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
    }}
    .processCoreSettings .prusaSettingsGrid {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    .coreMaterialColumns {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-3);
      align-items: start;
    }}
    .coreMaterialColumns > .prusaAdvancedGroup + .prusaAdvancedGroup {{
      margin-top: 0;
      padding-top: 0;
      padding-left: var(--space-3);
      border-top: 0;
      border-left: 1px solid var(--line);
    }}
    .coreMaterialColumns > .prusaAdvancedGroup {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      column-gap: var(--space-3);
      align-items: start;
    }}
    .coreMaterialColumns > .prusaAdvancedGroup > h4 {{
      grid-column: 1 / -1;
    }}
    .coreMaterialColumns > .prusaAdvancedGroup > .coreParameterSubgroup:nth-of-type(1),
    .coreMaterialColumns > .prusaAdvancedGroup > .coreParameterSubgroup:nth-of-type(4) {{
      grid-column: 1 / -1;
    }}
    .coreMaterialColumns > .prusaAdvancedGroup > .coreParameterSubgroup:nth-of-type(1) .prusaSettingsGrid,
    .coreMaterialColumns > .prusaAdvancedGroup > .coreParameterSubgroup:nth-of-type(4) .prusaSettingsGrid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .coreMaterialColumns > .prusaAdvancedGroup > .coreParameterSubgroup:nth-of-type(2) .prusaSettingsGrid,
    .coreMaterialColumns > .prusaAdvancedGroup > .coreParameterSubgroup:nth-of-type(3) .prusaSettingsGrid {{
      grid-template-columns: 1fr;
    }}
    .coreTravelPanel {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      column-gap: var(--space-3);
      align-items: start;
    }}
    .coreTravelPanel > h4 {{
      grid-column: 1 / -1;
    }}
    .coreTravelPanel > .coreParameterSubgroup:nth-of-type(3) {{
      grid-column: 1 / -1;
    }}
    .coreTravelPanel > .coreParameterSubgroup:nth-of-type(1) .prusaSettingsGrid,
    .coreTravelPanel > .coreParameterSubgroup:nth-of-type(2) .prusaSettingsGrid {{
      grid-template-columns: 1fr;
    }}
    .coreTravelPanel > .coreParameterSubgroup:nth-of-type(3) .prusaSettingsGrid {{
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .fieldGroup {{ min-width: 0; }}
    .inputBand .bandGrid > .fieldGroup {{ grid-column: auto; }}
    .inputBand .bandGrid > .layerAdvanced {{ grid-column: 1 / -1; }}
    .compactDimensionField > label {{ white-space: nowrap; }}
    .prusaQuickField {{ min-width: 0; }}
    .prusaQuickField[hidden] {{ display: none; }}
    .prusaQuickField .checkboxLabel {{ margin: 0; min-height: var(--control-height); display: flex; align-items: center; }}
    .span-1 {{ grid-column: span 1; }}
    .span-2 {{ grid-column: span 2; }}
    .span-3 {{ grid-column: span 3; }}
    .span-4 {{ grid-column: span 4; }}
    .span-5 {{ grid-column: span 5; }}
    .span-6 {{ grid-column: span 6; }}
    .span-8 {{ grid-column: span 8; }}
    .span-9 {{ grid-column: span 9; }}
    .span-12 {{ grid-column: 1 / -1; }}
    .compactGrid {{
      gap: var(--space-2);
    }}
    .compactOptions {{
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2) var(--space-4);
      align-items: center;
      align-content: center;
      align-self: end;
      height: var(--control-height);
      min-height: var(--control-height);
      padding-top: 0;
    }}
    .compactOptions .checkboxLabel {{
      margin: 0;
    }}
    .formSection > .grid,
    .formSection > .advancedSettings {{
      margin-top: var(--space-2);
    }}
    .bandGrid > .advancedSettings {{
      margin-top: var(--space-3);
    }}
    .kernelBand > .bandGrid:not(:first-of-type),
    .kernelBand > #prusaNativeSettings:not([hidden]),
    .kernelBand > #legacyNativeSettings:not([hidden]),
    .kernelBand > #pyslmNativeSettings:not([hidden]) {{
      margin-top: var(--space-2);
    }}
    .readOnlySummary {{
      display: block;
      min-height: var(--control-height);
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      background: #f6f8fb;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .processGroup {{
      min-width: 0;
      align-self: stretch;
    }}
    .processGroup + .processGroup {{
      padding-left: var(--space-3);
      border-left: 1px solid var(--line);
    }}
    .processGroup h3 {{
      margin: 0;
    }}
    .processTitleRow {{
      min-height: 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-2);
      margin-bottom: var(--space-1);
    }}
    .processTitleRow .checkboxLabel {{
      margin: 0;
    }}
    .curveFields {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: var(--space-2);
    }}
    .processBand .actions {{
      display: flex;
      flex-direction: row;
      align-items: center;
      align-self: stretch;
      justify-content: flex-end;
      gap: var(--space-2);
      margin: 0;
      min-height: 0;
      grid-column: 1 / -1;
    }}
    .processBand > .bandGrid {{
      align-items: stretch;
    }}
    .processBand .actions button {{
      width: auto;
      flex: 0 0 auto;
    }}
    .processBand .status:empty {{
      display: none;
    }}
    .actions {{
      display: flex;
      align-items: center;
      gap: var(--space-3);
      margin-top: var(--space-6);
      flex-wrap: wrap;
    }}
    .exportActionRow {{
      display: flex;
      align-items: center;
      gap: var(--space-3);
      width: auto;
      flex: 1 1 auto;
      min-width: 0;
      flex-wrap: nowrap;
    }}
    button {{
      min-height: var(--control-height);
      height: var(--control-height);
      border: 0;
      border-radius: var(--radius-sm);
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
      white-space: nowrap;
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
    .notice:empty {{ display: none; }}
    .advancedSettings {{
      margin-top: var(--space-4);
      padding-top: var(--space-3);
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
      margin-bottom: var(--space-3);
    }}
    .advancedSettings:not([open]) {{
      padding-top: var(--space-2);
    }}
    .advancedPopupHost {{
      position: relative;
    }}
    .advancedPopupHost#coreProcessSettings {{
      position: static;
    }}
    .advancedPopupTrigger {{
      display: inline-flex;
      align-items: center;
      gap: var(--space-2);
      min-height: 32px;
      height: auto;
      padding: 5px 9px;
      border: 1px solid #c5d5e1;
      border-radius: var(--radius-sm);
      color: var(--accent-dark);
      background: #f1f7fb;
      font-size: 13px;
      font-weight: 650;
      text-align: left;
    }}
    .advancedPopupTrigger::after {{
      content: '打开窗口';
      margin-left: var(--space-1);
      font-size: 11px;
      font-weight: 600;
    }}
    .advancedPopupTrigger:hover {{
      color: var(--accent-dark);
      background: #e7f2fb;
    }}
    .advancedPopupHost > summary {{
      display: inline-flex;
      align-items: center;
      gap: var(--space-2);
      min-height: 32px;
      padding: 5px 9px;
      border: 1px solid #c5d5e1;
      border-radius: var(--radius-sm);
      color: var(--accent-dark);
      background: #f1f7fb;
    }}
    .advancedPopupHost > summary::marker {{
      content: '';
    }}
    .advancedPopupHost > summary::after {{
      content: '打开设置';
      font-size: 11px;
      font-weight: 600;
    }}
    .advancedPopupHost > summary:hover {{
      background: #e7f2fb;
    }}
    .advancedPopup {{
      position: fixed;
      z-index: 1100;
      left: 50%;
      top: 50%;
      display: none;
      flex-direction: column;
      width: min(900px, calc(100vw - 24px));
      min-width: min(320px, calc(100vw - 24px));
      height: min(72vh, 680px);
      min-height: 180px;
      max-width: calc(100vw - 24px);
      max-height: calc(100vh - 24px);
      resize: both;
      overflow: hidden;
      transform: translate(-50%, -50%);
      border: 1px solid #8fa0ad;
      border-radius: var(--radius-md);
      background: #ffffff;
      box-shadow: 0 14px 36px rgba(23, 32, 38, 0.24);
    }}
    .advancedPopup.visible {{
      display: flex;
    }}
    .advancedPopup.coreAdvancedPopup {{
      position: absolute;
      height: auto;
      max-height: none;
      overflow: visible;
      transform: none;
    }}
    .advancedPopupHeader {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-3);
      flex: 0 0 auto;
      min-height: 42px;
      padding: 8px 12px;
      border-bottom: 1px solid var(--line);
      background: #f7fafc;
      cursor: move;
      user-select: none;
    }}
    .advancedPopupTitle {{
      min-width: 0;
      overflow: hidden;
      color: var(--ink);
      font-size: 14px;
      font-weight: 700;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .advancedPopupClose {{
      width: 30px;
      min-width: 30px;
      height: 30px;
      min-height: 30px;
      padding: 0;
      border: 1px solid #bcc5cd;
      border-radius: 5px;
      color: var(--ink);
      background: #ffffff;
      font-size: 18px;
      line-height: 1;
    }}
    .advancedPopupClose:hover {{
      color: var(--ink);
      background: #edf3f7;
    }}
    .advancedPopupBody {{
      min-height: 0;
      overflow: auto;
      padding: var(--space-4);
    }}
    .coreAdvancedPopup .advancedPopupBody {{
      overflow: visible;
      padding: 10px 12px;
    }}
    .coreAdvancedPopup .coreMaterialColumns {{
      gap: 8px;
    }}
    .coreAdvancedPopup .coreMaterialColumns > .prusaAdvancedGroup {{
      column-gap: 8px;
    }}
    .coreAdvancedPopup .coreTravelPanel {{
      column-gap: 8px;
    }}
    .coreAdvancedPopup .prusaAdvancedGroup h4 {{
      margin-bottom: 4px;
    }}
    .coreAdvancedPopup .coreParameterSubgroup {{
      margin-top: 5px;
    }}
    .coreAdvancedPopup .coreParameterSubgroup h5 {{
      margin-bottom: 2px;
    }}
    .coreAdvancedPopup .prusaSettingsGrid {{
      gap: 4px;
    }}
    .coreAdvancedPopup label {{
      margin-top: 2px;
      margin-bottom: 2px;
      line-height: 1.2;
    }}
    .coreAdvancedPopup input:not([type="checkbox"]):not([type="range"]),
    .coreAdvancedPopup select {{
      height: 34px;
      min-height: 34px;
      padding: 5px 8px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: var(--space-3);
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: var(--space-4);
      background: #ffffff;
      min-width: 0;
    }}
    .metric span {{
      display: block;
      font-size: 12px;
      color: var(--muted);
    }}
    .metric strong {{
      display: block;
      margin-top: var(--space-2);
      font-size: 21px;
      line-height: 1.1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    .download {{
      margin-top: var(--space-4);
      display: none;
      color: var(--accent-dark);
      font-weight: 650;
      text-decoration: none;
    }}
    .download.visible {{ display: inline-block; }}
    .exportProgress {{
      display: none;
      align-items: center;
      gap: var(--space-2);
      min-width: 0;
      flex: 0 1 auto;
      color: var(--muted);
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }}
    .exportProgress.visible {{ display: inline-flex; }}
    .exportProgressHeader {{
      display: flex;
      gap: var(--space-1);
      white-space: nowrap;
    }}
    .exportProgress progress {{
      width: 120px;
      height: 8px;
      accent-color: var(--accent);
    }}
    .exportElapsed {{
      white-space: nowrap;
      color: var(--muted);
    }}
    .viewerControls {{
      margin-top: var(--space-5);
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: var(--space-4);
    }}
    .viewerControls label {{
      margin-top: 0;
    }}
    .rangeRow {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: var(--space-3);
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
    .pathPlaybackControls {{
      margin-top: var(--space-2);
      display: grid;
      grid-template-columns: auto auto minmax(110px, 1fr) auto;
      gap: var(--space-2);
      align-items: center;
    }}
    .pathPlaybackControls button {{
      min-height: 30px;
      height: 30px;
      padding: 0 10px;
      font-size: 13px;
    }}
    .pathPlaybackControls label {{
      margin: 0;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .pathPlaybackControls input {{
      min-width: 0;
      padding: 0;
    }}
    .legend {{
      margin-top: var(--space-4);
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-3) var(--space-5);
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }}
    .legendItem {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      margin: 0;
      color: var(--muted);
      cursor: pointer;
    }}
    .legendItem input {{
      width: auto;
      min-height: 0;
      padding: 0;
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
    .raftSwatch {{ background: #7f5539; }}
    .fiberSwatch {{ background: #e66f00; }}
    .travelSwatch {{
      height: 4px;
      background: repeating-linear-gradient(90deg, #526f8c 0 7px, transparent 7px 12px);
    }}
    .coreTravelSwatch {{
      height: 4px;
      background: repeating-linear-gradient(90deg, #c2410c 0 5px, transparent 5px 9px);
    }}
    .primelineSwatch {{ background: #b91c1c; }}
    .originSwatch {{
      width: 10px;
      height: 10px;
      border: 2px solid #b91c1c;
      background: #ffffff;
      transform: rotate(45deg);
    }}
    .viewOptions {{
      margin-top: var(--space-3);
      display: flex;
      flex-wrap: wrap;
      gap: var(--space-2) var(--space-4);
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
    .extrusionColorLegend {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      font-size: 12px;
    }}
    .extrusionColorRamp {{
      width: 82px;
      height: 8px;
      border: 1px solid rgba(23, 32, 38, 0.18);
      border-radius: 999px;
      background: linear-gradient(90deg, #1e40af, #0f96a0, #f59e0b, #dc2626);
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
    input[type="checkbox"] {{
      width: 14px;
      height: 14px;
      min-height: 14px;
      margin: 0;
      padding: 0;
      flex: 0 0 auto;
      accent-color: var(--accent);
    }}
    .preview {{
      margin-top: var(--space-5);
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      height: min(58vh, 560px);
      min-height: 420px;
      background: #ffffff;
      position: relative;
      overflow: hidden;
      cursor: grab;
      user-select: none;
      touch-action: none;
      box-shadow: 0 2px 8px rgba(23, 32, 38, 0.04);
    }}
    .preview.dragging {{
      cursor: grabbing;
    }}
    .preview canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    @media (max-width: 820px) {{
      main {{ padding: 24px 18px 32px; }}
      .bandGrid {{ grid-template-columns: repeat(6, minmax(0, 1fr)); }}
      .inputBand .bandGrid {{ grid-template-columns: repeat(6, minmax(0, 1fr)); }}
      .inputBand .bandGrid > .fieldGroup {{ grid-column: span 3; }}
      .inputBand .fiberNotice,
      .inputBand .layerAdvanced {{ grid-column: 1 / -1; }}
      .kernelBand > .bandGrid > .span-3 {{ grid-column: span 3; }}
      .kernelBand > .bandGrid > .span-4 {{ grid-column: span 2; }}
      .kernelBand > .bandGrid > .span-12,
      .kernelBand .span-8 {{ grid-column: 1 / -1; }}
      .processBand > .bandGrid > .span-9 {{ grid-column: 1 / -1; }}
      .processBand .processGroup {{ grid-column: 1 / -1; }}
      .processCoreSettings .prusaSettingsGrid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .coreMaterialColumns {{
        grid-template-columns: 1fr;
      }}
      .coreMaterialColumns > .prusaAdvancedGroup + .prusaAdvancedGroup {{
        margin-top: var(--space-3);
        padding-top: var(--space-3);
        padding-left: 0;
        border-top: 1px solid var(--line);
        border-left: 0;
      }}
      .coreMaterialColumns > .prusaAdvancedGroup {{
        display: block;
      }}
      .coreTravelPanel {{
        display: block;
      }}
      .processGroup + .processGroup {{
        padding-top: var(--space-3);
        padding-left: 0;
        border-top: 1px solid var(--line);
        border-left: 0;
      }}
      .summary {{ grid-template-columns: 1fr; }}
      .viewerControls {{ grid-template-columns: 1fr; }}
      header {{ padding: 14px 18px; }}
      .preview {{ height: 460px; min-height: 420px; }}
    }}
    @media (max-width: 520px) {{
      main {{ padding: 20px 12px 28px; }}
      .panel {{ padding: var(--space-4); }}
      .bandGrid {{ grid-template-columns: 1fr; }}
      .inputBand .bandGrid {{ grid-template-columns: 1fr; }}
      .span-2, .span-3, .span-4, .span-5, .span-6, .span-8, .span-9, .span-12,
      .inputBand .bandGrid > .fieldGroup,
      .inputBand .fiberNotice,
      .inputBand .layerAdvanced,
      .kernelBand > .bandGrid > .span-3,
      .kernelBand > .bandGrid > .span-4 {{ grid-column: auto; }}
      .sectionTitleRow {{ align-items: flex-start; flex-direction: column; gap: var(--space-1); }}
      .sectionTitleRow .fiberNotice {{ text-align: left; }}
      .grid {{ grid-template-columns: 1fr; }}
      .curveFields {{ grid-template-columns: 1fr; }}
      .processCoreSettings .prusaSettingsGrid {{
        grid-template-columns: 1fr;
      }}
      h1 {{ font-size: 18px; }}
      header {{ align-items: flex-start; flex-direction: column; }}
      .surfaceTools {{ justify-content: flex-start; }}
      .preview {{ height: 380px; min-height: 340px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>机械臂空间复合材料增材制造系统切片器</h1>
    <div class="surfaceTools" aria-label="曲面工具">
      <button id="surfacePreviewButton" class="surfaceToolButton" type="button">启动曲面预览器</button>
      <button id="surfaceMapperButton" class="surfaceToolButton" type="button">启动曲面映射器</button>
      <button id="coreNpzPreviewButton" class="surfaceToolButton" type="button">导入 Core NPZ 预览</button>
      <button id="surfaceNpzPreviewButton" class="surfaceToolButton" type="button">导入曲面/共形 NPZ 预览</button>
      <button id="surfaceNpzCollisionButton" class="surfaceToolButton" type="button" disabled>检查当前 NPZ 碰撞</button>
      <output id="surfaceNpzCollisionResult" class="surfaceCollisionResult" aria-live="polite">请先导入本地映射 NPZ。</output>
      <input id="surfaceNpzInput" type="file" accept=".npz,application/octet-stream" hidden>
    </div>
  </header>
  <main>
    <section class="resultsColumn">
      <div class="summary">
        <div class="metric"><span>层数</span><strong id="layers">-</strong></div>
        <div class="metric"><span>输出</span><strong id="outputName">-</strong></div>
        <div class="metric"><span>实际填充策略</span><strong id="executedInfillPattern">-</strong></div>
      </div>
      <a id="download" class="download" href="#">下载 Core NPZ</a>
      <div class="viewerControls">
        <div>
          <label for="layerSlider">层</label>
          <div class="rangeRow">
            <input id="layerSlider" type="range" min="0" max="0" value="0" disabled>
            <output id="layerLabel">-</output>
          </div>
        </div>
        <div id="pathProgressControl">
          <label for="pathProgressSlider">所选路径进度</label>
          <div class="rangeRow">
            <input id="pathProgressSlider" type="range" min="0" max="0" value="0" disabled>
            <output id="pathProgressLabel">-</output>
          </div>
          <div id="pathPlaybackControl" class="pathPlaybackControls" hidden>
            <button id="playCurrentPath" type="button" disabled aria-pressed="false">播放当前路径</button>
            <label for="pathPlaybackRate">播放速率</label>
            <input id="pathPlaybackRate" type="range" min="0" max="1" step="0.05" value="1" aria-label="当前路径播放速率">
            <output id="pathPlaybackRateLabel">1.00</output>
          </div>
        </div>
      </div>
      <div class="legend" aria-label="预览图例">
        <label class="legendItem"><input id="showOuterContour" type="checkbox" checked><span class="swatch outerSwatch"></span>外轮廓</label>
        <label class="legendItem"><input id="showInnerContour" type="checkbox" checked><span class="swatch innerSwatch"></span>内轮廓</label>
        <label class="legendItem"><input id="showResinInfill" type="checkbox" checked><span class="swatch infillSwatch"></span>树脂填充</label>
        <label class="legendItem"><input id="showRaftPaths" type="checkbox" checked><span class="swatch raftSwatch"></span>Prusa 筏层</label>
        <label class="legendItem"><input id="showFiberPaths" type="checkbox" checked><span class="swatch fiberSwatch"></span>纤维路径</label>
        <label class="legendItem"><input id="showTravelPaths" type="checkbox" checked><span class="swatch travelSwatch"></span>空移 Travel</label>
        <label class="legendItem"><input id="showCoreTravelPaths" type="checkbox" checked><span class="swatch coreTravelSwatch"></span>Core 转场空走</label>
        <label class="legendItem"><input id="showPrimeline" type="checkbox" checked><span class="swatch primelineSwatch"></span>Core Primeline</label>
        <span class="legendItem"><span class="swatch originSwatch"></span>打印平面原点 (0, 0)</span>
      </div>
      <div class="viewOptions" aria-label="显示选项">
        <label title="开启后同时绘制当前层及此前各层的完整路径；每层保留自身实际 Z 高度。"><input id="showLayerOverlay" type="checkbox">叠加层显示</label>
        <label title="仅改变预览笔触宽度，不改变轨迹中心线或挤出倍率"><input id="showLineWidth" type="checkbox">按实际规划线宽显示（当前 <span id="previewLineWidthValue">2.2 mm</span>）</label>
        <label title="仅对包含 Prusa E 数据的树脂路径按绝对单位长度挤出量着色；关闭时保持轮廓/填充原有配色。"><input id="showExtrusion" type="checkbox">显示绝对挤出量（E/mm）</label>
        <span id="extrusionColorLegend" class="extrusionColorLegend" hidden>0.00 E/mm <span class="extrusionColorRamp"></span> 0.50 E/mm</span>
        <span title="连续蜂窝路径中，灰蓝虚线只移动喷头，不增加 E。">灰蓝虚线：零挤出安全连接</span>
        <label><input id="showPathPoints" type="checkbox">显示当前路径点</label>
        <label><input id="showDirection" type="checkbox" checked><span id="showDirectionLabel">显示打印方向</span></label>
        <span id="printSizeLabel">打印范围 -</span>
      </div>
      <div id="previewSurface" class="preview"><canvas id="previewCanvas" title="滚轮缩放；鼠标左键、右键或中键拖动视图"></canvas></div>
    </section>

    <section class="panel">
      <h2>模型切片</h2>
      <form id="sliceForm">
        <div class="formSection inputBand" data-layout-band="input-layer">
          <div class="sectionTitleRow">
            <h3>输入与分层</h3>
            <div id="fiberNotice" class="notice fiberNotice"></div>
          </div>
          <div class="bandGrid">
            <div class="fieldGroup span-4">
              <label for="stlFile">STL 文件</label>
              <input id="stlFile" name="stlFile" type="file" accept=".stl" required>
            </div>
            <div class="fieldGroup span-4">
              <label for="fiberJsonFile">纤维路径 JSON</label>
              <input id="fiberJsonFile" name="fiberJsonFile" type="file" accept=".json,application/json">
            </div>
            <div class="fieldGroup compactDimensionField">
              <label for="layerHeight">树脂层高 mm</label>
              <input id="layerHeight" name="layerHeight" type="number" min="0.001" step="0.001" value="{prusa_num('layer_height', core_defaults.resin.layer_height_mm)}">
            </div>
            <div class="fieldGroup compactDimensionField">
              <label for="firstLayerHeight">首层层高 mm</label>
              <input id="firstLayerHeight" name="firstLayerHeight" type="number" min="0.001" step="0.001" value="{prusa_num('first_layer_height', core_defaults.resin.layer_height_mm)}">
            </div>
            <div class="fieldGroup compactDimensionField">
              <label for="buildAxis">层高方向</label>
              <select id="buildAxis" name="buildAxis">
                <option value="auto" selected>自动</option>
                <option value="y">Y 轴</option>
                <option value="z">Z 轴</option>
                <option value="x">X 轴</option>
              </select>
            </div>
          </div>
        </div>

        <div class="formSection kernelBand" data-layout-band="kernel">
          <h3>树脂路径内核</h3>
          <div class="bandGrid">
            <div class="fieldGroup span-3">
              <label for="lineWidth">名义树脂线宽 mm</label>
              <input id="lineWidth" name="lineWidth" type="number" min="0.001" step="0.001" value="{prusa_num('line_width', DEFAULT_RESIN_LINE_WIDTH_MM)}">
            </div>
            <div class="fieldGroup span-3">
              <label for="slicingKernel">切片内核</label>
              <select id="slicingKernel" name="slicingKernel">
                <option value="prusa" selected>Prusa（推荐）</option>
                <option value="legacy">Legacy（兼容 / 实验）</option>
                <option value="pyslm">PySLM（实验）</option>
              </select>
            </div>
            <div class="fieldGroup span-3 prusaQuickField">
              <label for="prusaInfillPattern" class="tooltipLabel" data-tooltip="Prusa 内部填充的几何图案和主方向。">Prusa 填充图案</label>
              <select id="prusaInfillPattern">
                <option value="zigzag_horizontal" selected>横向 Zigzag</option>
                <option value="zigzag_vertical">竖向 Zigzag</option>
                <option value="zigzag_plus45">+45° Zigzag</option>
                <option value="zigzag_minus45">-45° Zigzag</option>
                <option value="isotropic">各向同性填充</option>
                <option value="triangles">三角填充</option>
                <option value="concentric">同心轮廓填充</option>
              </select>
            </div>
            <div class="compactOptions span-3 prusaQuickField">
              <label for="prusaPrintPerimeters" class="tooltipLabel checkboxLabel" data-tooltip="生成模型的外轮廓与内轮廓。"><input id="prusaPrintPerimeters" type="checkbox" checked> 打印内外轮廓</label>
            </div>
          </div>

          <div id="prusaNativeSettings">
            <h3>Prusa 原生路径参数</h3>
            <p class="notice">由 Prusa 完整生成轮廓、填充与空移；不使用项目原生的连续路径、三角形或之字形优化。</p>
            <div class="prusaSettingsGrid">
              <div class="fieldGroup">
                <label for="prusaStartX">零件起始 Travel X mm</label>
                <input id="prusaStartX" name="prusaStartX" type="number" step="0.001" value="{prusa_num('prusa_start_x_mm', core_defaults.start_x_mm)}">
              </div>
              <div class="fieldGroup">
                <label for="prusaStartY">零件起始 Travel Y mm</label>
                <input id="prusaStartY" name="prusaStartY" type="number" step="0.001" value="{prusa_num('prusa_start_y_mm', core_defaults.start_y_mm)}">
              </div>
              <div class="fieldGroup">
                <label for="prusaPerimeterCount" class="tooltipLabel" data-tooltip="零件外轮廓和内轮廓的总圈数。更多圈数会提升边缘强度与尺寸稳定性，但会增加打印时间。">边界圈数</label>
                <input id="prusaPerimeterCount" type="number" min="1" step="1" value="{prusa_num('prusa_perimeter_count', 2)}">
              </div>
              <div class="fieldGroup">
                <label for="prusaInfillDensity" class="tooltipLabel" data-tooltip="模型内部由填充覆盖的比例。100% 为实心；较低数值可缩短时间并降低材料用量。">填充率 %</label>
                <input id="prusaInfillDensity" type="number" min="0" max="100" step="1" value="{prusa_num('prusa_infill_density', DEFAULT_RESIN_INFILL_DENSITY_PERCENT)}">
              </div>
              <div class="fieldGroup">
                <label for="prusaContourInfillOverlap" class="tooltipLabel" data-tooltip="填充与最内侧轮廓的重叠比例。适当搭边能避免两者之间留缝；过大可能造成局部过挤。">轮廓与填充搭边 %</label>
                <input id="prusaContourInfillOverlap" type="number" min="0" max="99" step="0.1" value="{prusa_num('prusa_contour_infill_overlap', DEFAULT_RESIN_CONTOUR_INFILL_OVERLAP_PERCENT)}">
              </div>
            </div>
            <div class="prusaFeatureToggle">
              <label for="prusaRaftEnabled" class="tooltipLabel checkboxLabel" data-tooltip="在零件底部生成可剥离的 Prusa 筏层，以改善首层附着与底面稳定性；不会额外生成悬垂支撑。"><input id="prusaRaftEnabled" type="checkbox"> 启用可剥离 Prusa 筏层</label>
            </div>
            <div id="prusaRaftSettings" class="prusaSettingsGrid prusaSubSettings" hidden>
              <div class="fieldGroup fieldGroupWide">
                <label for="prusaRaftAutoContact" class="tooltipLabel checkboxLabel" data-tooltip="勾选时由 Prusa 自动计算接触筏层的层高、填充和线宽；取消后使用下面的手动测试参数。"><input id="prusaRaftAutoContact" type="checkbox" checked> 自动计算接触筏层</label>
              </div>
              <div class="fieldGroup">
                <label for="prusaRaftLayers" class="tooltipLabel" data-tooltip="筏层的打印层数。层数越多，筏层越稳固，但材料和时间也会增加。">筏层数</label>
                <input id="prusaRaftLayers" type="number" min="1" step="1" value="{DEFAULT_PRUSA_RAFT_LAYER_COUNT}">
              </div>
              <div class="fieldGroup">
                <label for="prusaRaftExpansion" class="tooltipLabel" data-tooltip="筏层相对模型轮廓向外扩展的距离，用于增加与打印平台的接触面积。">筏层外扩 mm</label>
                <input id="prusaRaftExpansion" type="number" min="0" step="0.1" value="{DEFAULT_PRUSA_RAFT_EXPANSION_MM:g}">
              </div>
              <div class="fieldGroup">
                <label for="prusaRaftFirstLayerDensity" class="tooltipLabel" data-tooltip="筏层第一层的填充密度。较高密度通常能提升与打印平台的附着。">首层密度 %</label>
                <input id="prusaRaftFirstLayerDensity" type="number" min="10" max="100" step="1" value="{DEFAULT_PRUSA_RAFT_FIRST_LAYER_DENSITY_PERCENT:g}">
              </div>
              <div class="fieldGroup">
                <label for="prusaRaftFirstLayerExpansion" class="tooltipLabel" data-tooltip="仅对筏层第一层增加的额外外扩距离，可进一步提高底部附着面积。">首层额外外扩 mm</label>
                <input id="prusaRaftFirstLayerExpansion" type="number" min="0" step="0.1" value="{DEFAULT_PRUSA_RAFT_FIRST_LAYER_EXPANSION_MM:g}">
              </div>
              <div class="fieldGroup">
                <label for="prusaRaftContactDistance" class="tooltipLabel" data-tooltip="零件与筏层之间的 Z 向间隙。数值越大越易剥离，但底面支撑越弱；建议从 0.25 mm 开始实测。">可剥离间隙 mm</label>
                <input id="prusaRaftContactDistance" type="number" min="0" step="0.01" value="{DEFAULT_PRUSA_RAFT_CONTACT_DISTANCE_MM:g}">
              </div>
              <div class="fieldGroup">
                <label for="prusaRaftContactLayerHeight" class="tooltipLabel" data-tooltip="仅在取消自动计算时生效。接触筏层是靠近零件的一层，建议先测试 0.5、0.75 和 1.0 mm。">接触层高度 mm</label>
                <input id="prusaRaftContactLayerHeight" type="number" min="0.1" max="2" step="0.05" value="0.75">
              </div>
              <div class="fieldGroup">
                <label for="prusaRaftContactDensity" class="tooltipLabel" data-tooltip="仅在取消自动计算时生效。接触层填充密度，建议保持 100% 以优先保证与下层筏层的接触。">接触层密度 %</label>
                <input id="prusaRaftContactDensity" type="number" min="10" max="100" step="1" value="100">
              </div>
              <div class="fieldGroup">
                <label for="prusaRaftContactExtrusionWidth" class="tooltipLabel" data-tooltip="仅在取消自动计算时生效。只改变接触筏层线宽，不改变零件打印线宽；建议先测试 1.0~1.5 mm。">接触层线宽 mm</label>
                <input id="prusaRaftContactExtrusionWidth" type="number" min="0.1" max="4" step="0.1" value="1.5">
              </div>
            </div>
            <div class="prusaFeatureToggle">
              <label for="prusaSkirtEnabled" class="tooltipLabel checkboxLabel" data-tooltip="在打印开始时生成 Prusa 裙边。开启后可直接调整裙边圈数、距离、高度和最小长度；默认关闭。"><input id="prusaSkirtEnabled" type="checkbox"> 启用 Prusa 裙边</label>
            </div>
            <div id="prusaSkirtSettings" class="prusaSettingsGrid prusaSubSettings" hidden>
              <div class="fieldGroup">
                <label for="prusaSkirtLoops">裙边圈数</label>
                <input id="prusaSkirtLoops" type="number" min="0" step="1" value="1">
              </div>
              <div class="fieldGroup">
                <label for="prusaSkirtDistance">裙边距离 mm</label>
                <input id="prusaSkirtDistance" type="number" min="0" step="0.1" value="6">
              </div>
              <div class="fieldGroup">
                <label for="prusaSkirtHeight">裙边高度 层</label>
                <input id="prusaSkirtHeight" type="number" min="0" step="1" value="1">
              </div>
              <div class="fieldGroup">
                <label for="prusaMinSkirtLength">最小裙边长度 mm</label>
                <input id="prusaMinSkirtLength" type="number" min="0" step="1" value="10">
              </div>
            </div>
            <div class="prusaFeatureToggle">
              <label for="prusaBrimEnabled" class="tooltipLabel checkboxLabel" data-tooltip="在首层生成 Prusa Brim，用于增加底部附着面积；默认关闭。"><input id="prusaBrimEnabled" type="checkbox"> 启用 Prusa Brim</label>
            </div>
            <div id="prusaBrimSettings" class="prusaSettingsGrid prusaSubSettings" hidden>
              <div class="fieldGroup">
                <label for="prusaBrimWidth">Brim 宽度 mm</label>
                <input id="prusaBrimWidth" type="number" min="0" step="0.1" value="5">
              </div>
              <div class="fieldGroup">
                <label for="prusaBrimType">Brim 类型</label>
                <select id="prusaBrimType">
                  <option value="outer_only" selected>仅外侧</option>
                  <option value="outer_and_inner">外侧和内侧</option>
                  <option value="no_brim">不生成</option>
                </select>
              </div>
              <div class="fieldGroup">
                <label for="prusaBrimSeparation">Brim 分离间隙 mm</label>
                <input id="prusaBrimSeparation" type="number" min="0" step="0.1" value="0">
              </div>
              <label for="prusaBrimOneStroke" class="tooltipLabel checkboxLabel" data-tooltip="尝试复用 Core 的安全边界连接策略，将 Prusa Brim 连接为一条连续挤出路径；无法安全连接时保留原生多路径。"><input id="prusaBrimOneStroke" type="checkbox"> Brim 一笔画</label>
            </div>
            <div class="prusaFeatureToggle">
              <label for="honeycombCenterlineEnabled" class="tooltipLabel checkboxLabel" data-tooltip="附加于完整 Prusa 切片之后：每层先打印正式 150×100 外框，再生成原始 STL 孔壁的蜂窝路径。每个宏观分区内以不跨孔的零挤出安全换段连接，分区之间采用最短安全空走；所有沉积蜂窝壁均不重走。区内连接转角不超过 90°，三岔节点在一个线宽内渐降/渐升挤出。启用后 Core 使用该附加路径，不使用原生 Prusa G-code。"><input id="honeycombCenterlineEnabled" type="checkbox"{prusa_checked('honeycomb_centerline_enabled', False)}> 蜂窝连续路径（每层外框）</label>
              <div class="fieldGroup">
                <label for="honeycombTopology" class="tooltipLabel" data-tooltip="从原始 STL 的蜂窝孔壁生成不重走轨迹；通过多起点安全连接搜索，优先减少宏观分区数。分区内以不跨孔、转角不超过 90° 的零挤出安全换段连接；分区之间为最短安全空走。三岔节点以 1/3 挤出量平滑过渡。">蜂窝拓扑</label>
                <select id="honeycombTopology">
                  <option value="macro_partition_zero_e"{prusa_selected('honeycomb_topology', 'macro_partition_zero_e', 'macro_partition_zero_e')}>原始 STL 孔壁（最少宏观分区，区内零挤出安全换段）</option>
                </select>
              </div>
            </div>
            <details id="prusaAdvancedSettings" class="advancedSettings">
              <summary>Prusa 高级几何与路径设置</summary>
              <p class="notice">空白宽度和锚定长度保持 Prusa 默认值；其余项仅作用于 Prusa 内核。</p>
              <section class="prusaAdvancedGroup">
                <h4>轮廓与尺寸</h4>
                <div class="prusaSettingsGrid">
                  <div class="fieldGroup">
                  <label for="prusaPerimeterGenerator" class="tooltipLabel" data-tooltip="Arachne 会通过变线宽填补薄壁和窄区；Classic 保持更固定的线宽，结果更可预测。">轮廓生成器</label>
                  <select id="prusaPerimeterGenerator">
                    <option value="arachne" selected>Arachne（变线宽）</option>
                    <option value="classic">Classic（固定线宽）</option>
                  </select>
                  </div>
                  <div class="compactOptions">
                    <label for="prusaGapFillEnabled" class="tooltipLabel checkboxLabel" data-tooltip="让 Prusa 在常规轮廓无法覆盖的窄缝中生成额外填充路径。关闭可减少细小短路径。"><input id="prusaGapFillEnabled" type="checkbox" checked> 启用 Gap fill</label>
                  </div>
                  <div class="fieldGroup">
                  <label for="prusaExternalPerimeterWidth" class="tooltipLabel" data-tooltip="最外侧轮廓的目标线宽，直接影响外观、尺寸精度与边缘强度。留空使用 Prusa 默认。">外轮廓线宽 mm</label>
                  <input id="prusaExternalPerimeterWidth" type="number" min="0.001" step="0.001" placeholder="Prusa 默认">
                  </div>
                  <div class="fieldGroup">
                  <label for="prusaPerimeterWidth" class="tooltipLabel" data-tooltip="内侧轮廓的目标线宽。留空时由 Prusa 根据名义线宽和层高自动计算。">内轮廓线宽 mm</label>
                  <input id="prusaPerimeterWidth" type="number" min="0.001" step="0.001" placeholder="Prusa 默认">
                  </div>
                  <div class="fieldGroup">
                  <label for="prusaInfillWidth" class="tooltipLabel" data-tooltip="内部填充的目标线宽。可单独调整填充效率与实体内部的覆盖效果；留空使用 Prusa 默认。">填充线宽 mm</label>
                  <input id="prusaInfillWidth" type="number" min="0.001" step="0.001" placeholder="Prusa 默认">
                  </div>
                  <div class="fieldGroup">
                  <label for="prusaXySizeCompensation" class="tooltipLabel" data-tooltip="整体补偿模型的 XY 尺寸：正值向外扩张，负值向内收缩。用于校准材料收缩或实际线宽偏差。">XY 尺寸补偿 mm</label>
                  <input id="prusaXySizeCompensation" type="number" step="0.01" value="0">
                  </div>
                  <div class="fieldGroup">
                  <label for="prusaElephantFootCompensation" class="tooltipLabel" data-tooltip="削减首层外缘因挤压变宽产生的象脚。只在首层尺寸偏大或边缘鼓起时增加。">象脚补偿 mm</label>
                  <input id="prusaElephantFootCompensation" type="number" min="0" step="0.01" value="0">
                  </div>
                </div>
              </section>
              <section class="prusaAdvancedGroup">
                <h4>填充连接</h4>
                <div class="prusaSettingsGrid">
                  <div class="fieldGroup">
                    <label for="prusaInfillAnchor" class="tooltipLabel" data-tooltip="填充连接到轮廓时采用的目标锚定长度。较大可提高连接可靠性，但会让轮廓附近堆料更多；留空使用 Prusa 默认。">填充锚定长度 mm</label>
                    <input id="prusaInfillAnchor" type="number" min="0" step="0.1" placeholder="Prusa 默认">
                  </div>
                  <div class="fieldGroup">
                    <label for="prusaInfillAnchorMax" class="tooltipLabel" data-tooltip="限制填充连接轮廓时允许使用的最大锚定长度，避免为连接而沿轮廓走得过远；留空使用 Prusa 默认。">最大锚定长度 mm</label>
                    <input id="prusaInfillAnchorMax" type="number" min="0" step="0.1" placeholder="Prusa 默认">
                  </div>
                </div>
              </section>
              <section class="prusaAdvancedGroup">
                <h4>空移与接缝</h4>
                <div class="prusaSettingsGrid">
                  <div class="fieldGroup">
                  <label for="prusaAvoidCrossingMaxDetour" class="tooltipLabel" data-tooltip="Prusa 为避免空移跨越轮廓时允许的最大绕行距离。0 表示不限制；较小值可减少绕行，但可能更常跨越轮廓。">空移最大绕行 mm</label>
                  <input id="prusaAvoidCrossingMaxDetour" type="number" min="0" step="0.1" value="0">
                  </div>
                  <div class="fieldGroup">
                  <label for="prusaSeamPosition" class="tooltipLabel" data-tooltip="每层轮廓起止点的布置策略。对齐会形成一条固定接缝；最近减少空移；后侧尝试隐藏在模型后方；随机会分散接缝。">Z 接缝位置</label>
                  <select id="prusaSeamPosition">
                    <option value="aligned">对齐</option>
                    <option value="nearest">最近</option>
                    <option value="rear">后侧</option>
                    <option value="random" selected>随机</option>
                  </select>
                  </div>
                </div>
              </section>
            </details>
          </div>

          <div id="pyslmNativeSettings" hidden>
            <h3>PySLM 原生扫描参数</h3>
            <div class="grid">
              <div>
                <label for="pyslmInfillPattern">基础填充方向</label>
                <select id="pyslmInfillPattern">
                  <option value="zigzag_horizontal" selected>横向 Zigzag</option>
                  <option value="zigzag_vertical">竖向 Zigzag</option>
                  <option value="zigzag_plus45">+45° Zigzag</option>
                  <option value="zigzag_minus45">-45° Zigzag</option>
                  <option value="isotropic">各向同性填充</option>
                </select>
              </div>
              <div>
                <label for="pyslmInfillDensity">填充率 %</label>
                <input id="pyslmInfillDensity" type="number" min="0" max="100" step="1" value="{DEFAULT_RESIN_INFILL_DENSITY_PERCENT:g}">
              </div>
            </div>
            <div class="grid">
              <div>
                <label for="pyslmPerimeterCount">边界圈数</label>
                <input id="pyslmPerimeterCount" type="number" min="1" step="1" value="2">
              </div>
              <div>
                <label for="pyslmContourInfillOverlap">轮廓与填充搭边 %</label>
                <input id="pyslmContourInfillOverlap" type="number" min="0" max="99" step="0.1" value="{DEFAULT_RESIN_CONTOUR_INFILL_OVERLAP_PERCENT:g}">
              </div>
            </div>
            <div class="grid">
              <div>
                <label for="pyslmInfillOverlap">填充线间搭边 %</label>
                <input id="pyslmInfillOverlap" type="number" min="0" max="99" step="0.1" value="{DEFAULT_UI_RESIN_INFILL_OVERLAP_PERCENT:g}">
              </div>
              <label class="checkboxLabel"><input id="pyslmPrintPerimeters" type="checkbox" checked> 打印内外轮廓</label>
            </div>
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
                  <option value="bound">按边界尺度</option>
                </select>
              </div>
            </div>
            <label class="checkboxLabel"><input id="pyslmScanContourFirst" type="checkbox" checked> 轮廓优先扫描</label>
            <label class="checkboxLabel"><input id="pyslmFixPolygons" type="checkbox" checked> 修复切层多边形</label>
            <label class="checkboxLabel"><input id="pyslmSimplificationPreserveTopology" type="checkbox" checked> 保持拓扑结构</label>
          </div>

          <div id="legacyNativeSettings" hidden>
          <h3>Legacy 兼容 / 实验参数</h3>
          <div class="bandGrid compactGrid">
            <div class="fieldGroup span-4">
              <label class="labelWithHelp" for="planningLineWidth">实测压平线宽 mm
                <span class="helpTip" tabindex="0" data-tooltip="仅项目原生内核使用：用于树脂轨迹间距、搭边计算和实际铺展预览；不会修改 NPZ 中的名义线宽，也不会改变挤出倍率。">?</span>
              </label>
              <input id="planningLineWidth" name="planningLineWidth" type="number" min="0.001" step="0.001" value="{DEFAULT_RESIN_PLANNING_LINE_WIDTH_MM}">
            </div>
            <div class="fieldGroup span-4">
              <label for="perimeterCount">边界圈数</label>
              <input id="perimeterCount" name="perimeterCount" type="number" min="1" step="1" value="2">
            </div>
          </div>
          <div id="legacyInfillControl" class="bandGrid compactGrid">
            <div class="fieldGroup span-4">
              <label for="infillPattern" title="各向同性填充使用固定的 45°、0°、-45°、90° 循环；Prusa 会按此顺序写入原生逐层填充角。">树脂填充路径</label>
              <select id="infillPattern" name="infillPattern">
                <option value="zigzag_horizontal" selected>横向 Zigzag</option>
                <option value="zigzag_vertical">竖向 Zigzag</option>
                <option value="zigzag_plus45">+45° Zigzag</option>
                <option value="zigzag_minus45">-45° Zigzag</option>
                <option value="isotropic">各向同性填充</option>
                <option value="triangles">三角填充</option>
                <option value="concentric">同心轮廓填充</option>
              </select>
              <div id="infillSafetyNote" class="notice warning"></div>
            </div>
            <div class="compactOptions span-8">
              <label class="checkboxLabel" title="关闭时保留现有填充几何与路径规划，仅不输出内外轮廓路径"><input id="printPerimeters" type="checkbox" checked> 打印内外轮廓</label>
              <label class="checkboxLabel" title="仅项目原生内核使用；Prusa 使用自身的路径排序与连接策略。"><input id="trianglePathOptimization" type="checkbox" checked> 三角形填充路径优化</label>
              <label class="checkboxLabel" title="仅项目原生内核使用；Prusa 使用自身的路径排序与连接策略。"><input id="zigzagPathOptimization" type="checkbox" checked> 之字形填充路径优化</label>
            </div>
          </div>

          <div class="bandGrid">
            <div class="fieldGroup span-4">
              <label for="infillDensity">填充率 %</label>
              <input id="infillDensity" name="infillDensity" type="number" min="0" max="100" step="1" value="{DEFAULT_RESIN_INFILL_DENSITY_PERCENT:g}">
            </div>
            <div class="fieldGroup span-4">
              <label class="labelWithHelp" for="infillOverlap">填充线间搭边 %
                <span class="helpTip" tabindex="0" data-tooltip="仅项目原生与 PySLM 使用。Prusa 的实体填充线距由线宽和填充率决定，没有相同的独立参数。">?</span>
              </label>
              <input id="infillOverlap" name="infillOverlap" type="number" min="0" max="99" step="0.1" value="{DEFAULT_UI_RESIN_INFILL_OVERLAP_PERCENT:g}">
            </div>
            <div class="fieldGroup span-4">
              <label class="labelWithHelp" for="contourInfillOverlap">轮廓与填充搭边 %
                <span class="helpTip" tabindex="0" data-tooltip="Prusa 会映射为原生 infill_overlap（填充与最内层轮廓的搭边）；不改变填充线之间的线距。">?</span>
              </label>
              <input id="contourInfillOverlap" name="contourInfillOverlap" type="number" min="0" max="99" step="0.1" value="{DEFAULT_RESIN_CONTOUR_INFILL_OVERLAP_PERCENT:g}">
            </div>
          </div>
          </div>

        </div>

        <details id="coreProcessSettings" class="advancedSettings processCoreSettings">
          <summary>process_core 处理参数（不属于 Prusa 切片内核）</summary>
          <p class="notice">这些参数在路径进入 process_core 后生效；Prusa 区域只负责几何、层高、打印路径和 Prusa 空走。</p>

          <div class="coreMaterialColumns">
          <section class="prusaAdvancedGroup">
            <h4>树脂</h4>
            <div class="coreParameterSubgroup">
              <h5>基础参数</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreResinLayerHeight">层高 mm</label><input id="coreResinLayerHeight" type="number" min="0.001" step="0.001" value="0.5"></div>
                <div class="fieldGroup"><label for="coreResinExtrusionScale">挤出倍率</label><input id="coreResinExtrusionScale" type="number" min="0" step="0.001" value="1"></div>
                <div class="fieldGroup"><label for="coreResinTemp">温度 °C</label><input id="coreResinTemp" type="number" min="0" max="500" step="1" value="250"></div>
                <div class="fieldGroup"><label for="coreResinEOverride">E/mm 覆盖值</label><input id="coreResinEOverride" type="number" min="0" step="0.001" placeholder="使用默认计算"></div>
              </div>
            </div>
            <div class="coreParameterSubgroup">
              <h5>打印速度</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreResinFeed">常规打印速度 mm/s</label><input id="coreResinFeed" type="number" min="0.001" step="0.001" value="10"></div>
              </div>
            </div>
            <div class="coreParameterSubgroup">
              <h5>首层参数</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreResinFirstLayerFeed">首层打印速度 mm/s</label><input id="coreResinFirstLayerFeed" type="number" min="0.001" step="0.001" value="10"></div>
              </div>
            </div>
            <div class="coreParameterSubgroup">
              <h5>预挤出 / 回抽</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreResinPrimeLength">预挤出长度 mm</label><input id="coreResinPrimeLength" type="number" min="0" step="0.001" value="18"></div>
                <div class="fieldGroup"><label for="coreResinPrimeSpeed">预挤出速度 mm/s</label><input id="coreResinPrimeSpeed" type="number" min="0.001" step="0.001" value="15"></div>
                <div class="fieldGroup"><label for="coreResinRetractLength">回抽长度 mm</label><input id="coreResinRetractLength" type="number" min="0" step="0.001" value="15"></div>
                <div class="fieldGroup"><label for="coreResinRetractSpeed">回抽速度 mm/s</label><input id="coreResinRetractSpeed" type="number" min="0.001" step="0.001" value="30"></div>
              </div>
            </div>
          </section>

          <section class="prusaAdvancedGroup">
            <h4>纤维</h4>
            <div class="coreParameterSubgroup">
              <h5>基础参数</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreFiberLayerHeight">层高 mm</label><input id="coreFiberLayerHeight" type="number" min="0.001" step="0.001" value="0.1"></div>
                <div class="fieldGroup"><label for="coreFiberExtrusionScale">挤出倍率</label><input id="coreFiberExtrusionScale" type="number" min="0" step="0.001" value="1"></div>
                <div class="fieldGroup"><label for="coreFiberTemp">温度 °C</label><input id="coreFiberTemp" type="number" min="0" max="500" step="1" value="250"></div>
                <div class="fieldGroup"><label for="coreFiberStartAccel">起步加速时间 s</label><input id="coreFiberStartAccel" type="number" min="0.001" step="0.001" value="2"></div>
              </div>
            </div>
            <div class="coreParameterSubgroup">
              <h5>打印速度</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreFiberFeed">常规打印速度 mm/s</label><input id="coreFiberFeed" type="number" min="0.001" step="0.001" value="10"></div>
              </div>
            </div>
            <div class="coreParameterSubgroup">
              <h5>首层参数</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreFiberFirstLayerFeed">首层打印速度 mm/s</label><input id="coreFiberFirstLayerFeed" type="number" min="0.001" step="0.001" value="10"></div>
              </div>
            </div>
            <div class="coreParameterSubgroup">
              <h5>预挤出 / 回抽</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreFiberPrimeLength">预挤出长度 mm</label><input id="coreFiberPrimeLength" type="number" min="0" step="0.001" value="12"></div>
                <div class="fieldGroup"><label for="coreFiberPrimeSpeed">预挤出速度 mm/s</label><input id="coreFiberPrimeSpeed" type="number" min="0.001" step="0.001" value="5"></div>
                <div class="fieldGroup"><label for="coreFiberRetractLength">回抽长度 mm</label><input id="coreFiberRetractLength" type="number" min="0" step="0.001" value="10"></div>
                <div class="fieldGroup"><label for="coreFiberRetractSpeed">回抽速度 mm/s</label><input id="coreFiberRetractSpeed" type="number" min="0.001" step="0.001" value="5"></div>
              </div>
            </div>
          </section>
          </div>

          <section class="prusaAdvancedGroup coreTravelPanel">
            <h4>空走与起始姿态</h4>
            <div class="coreParameterSubgroup">
              <h5>空走速度</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreTravelFeed">常规空走速度 mm/s</label><input id="coreTravelFeed" type="number" min="0.001" step="0.001" value="10"></div>
              </div>
            </div>
            <div class="coreParameterSubgroup">
              <h5>首层参数（旧输入兼容）</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="coreFirstLayerTravelFeed">首层空走速度 mm/s</label><input id="coreFirstLayerTravelFeed" type="number" min="0.001" step="0.001" value="10"></div>
              </div>
            </div>
            <div class="coreParameterSubgroup">
              <h5>起始姿态与等待</h5>
              <div class="prusaSettingsGrid">
                <div class="fieldGroup"><label for="corePrimeSettle">预挤出稳定等待 s</label><input id="corePrimeSettle" type="number" min="0" step="0.001" value="0.5"></div>
                <div class="fieldGroup"><label for="coreDefaultA">默认 A °</label><input id="coreDefaultA" type="number" min="-360" max="360" step="0.001" value="0"></div>
                <div class="fieldGroup"><label for="coreDefaultB">默认 B °</label><input id="coreDefaultB" type="number" min="-360" max="360" step="0.001" value="0"></div>
                <div class="fieldGroup"><label for="coreDefaultC">默认 C °</label><input id="coreDefaultC" type="number" min="-360" max="360" step="0.001" value="0"></div>
              </div>
            </div>
          </section>

          <section class="prusaAdvancedGroup">
            <h4>Primeline</h4>
            <p class="notice">仅在勾选后启用下方的 Primeline 起点和长度参数；预览和导出会按命令顺序显示。</p>
            <div class="prusaSettingsGrid">
              <div class="fieldGroup compactOptions"><label class="checkboxLabel" for="corePrimelineEnabled"><input id="corePrimelineEnabled" type="checkbox" checked> 打印 Primeline</label></div>
              <div class="fieldGroup"><label for="corePrimelineX">起点 X mm</label><input id="corePrimelineX" type="number" step="0.001" value="0" disabled></div>
              <div class="fieldGroup"><label for="corePrimelineY">起点 Y mm</label><input id="corePrimelineY" type="number" step="0.001" value="-10" disabled></div>
              <div class="fieldGroup"><label for="corePrimelineLength">长度 mm</label><input id="corePrimelineLength" type="number" min="0" step="0.001" value="100" disabled></div>
            </div>
          </section>

          <section class="prusaAdvancedGroup">
            <h4>路径平滑与七阶采样</h4>
            <div class="prusaSettingsGrid">
              <div class="fieldGroup"><label for="coreDt">采样周期 dt s</label><input id="coreDt" type="number" min="0.0001" step="0.0001" value="0.004"></div>
              <div class="fieldGroup"><label for="coreCornerAngle">拐角角度阈值 °</label><input id="coreCornerAngle" type="number" min="0" step="0.1" value="45"></div>
              <div class="fieldGroup"><label for="coreCornerRetreatRatio">拐角回退比例</label><input id="coreCornerRetreatRatio" type="number" min="0" max="1" step="0.001" value="0.65"></div>
              <div class="fieldGroup"><label for="coreSplineMaxError">Spline 最大误差 mm</label><input id="coreSplineMaxError" type="number" min="0" step="0.001" value="0.1"></div>
              <div class="fieldGroup"><label for="coreSplineMaxAngle">Spline 最大转角 °</label><input id="coreSplineMaxAngle" type="number" min="0" step="0.1" value="45"></div>
              <div class="fieldGroup"><label for="coreSourceMergeDistance">源段合并距离 mm</label><input id="coreSourceMergeDistance" type="number" min="0" step="0.001" value="0.04"></div>
              <div class="fieldGroup"><label for="coreCornerRetreatMax">拐角最大回退 mm</label><input id="coreCornerRetreatMax" type="number" min="0" step="0.001" value="0.4"></div>
              <div class="fieldGroup"><label for="coreCornerBlendSegments">拐角平滑段数</label><input id="coreCornerBlendSegments" type="number" min="2" step="1" value="8"></div>
              <div class="fieldGroup"><label for="coreDensity">Spline 密度</label><input id="coreDensity" type="number" min="0" step="1" value="0"></div>
              <div class="fieldGroup"><label for="coreDegree">Spline 阶数</label><input id="coreDegree" type="number" min="1" step="1" value="3"></div>
              <div class="fieldGroup"><label for="coreMaxFitPoints">单段最大拟合点数</label><input id="coreMaxFitPoints" type="number" min="2" step="1" value="20000"></div>
            </div>
          </section>

          <section class="prusaAdvancedGroup">
            <h4>现场材料补偿与工具偏置</h4>
            <p class="notice">按材料区分的现场校准参数。它们在 process_core 命令层注入，随后统一经过七阶参数化。</p>
            <div class="prusaSettingsGrid">
              <div class="fieldGroup"><label for="coreFiberOffsetX">纤维头 X 偏置 mm</label><input id="coreFiberOffsetX" type="number" step="0.001" placeholder="读取校准文件"></div>
              <div class="fieldGroup"><label for="coreFiberOffsetY">纤维头 Y 偏置 mm</label><input id="coreFiberOffsetY" type="number" step="0.001" placeholder="读取校准文件"></div>
              <div class="fieldGroup"><label for="coreFiberOffsetZ">纤维头 Z 偏置 mm</label><input id="coreFiberOffsetZ" type="number" step="0.001" placeholder="读取校准文件"></div>
              <div class="fieldGroup"><label for="coreResinZComp">树脂 Z 补偿 mm</label><input id="coreResinZComp" type="number" step="0.001" placeholder="读取校准文件"></div>
              <div class="fieldGroup"><label for="coreFiberRetractOverride">纤维回抽覆盖 mm</label><input id="coreFiberRetractOverride" type="number" min="0" step="0.001" placeholder="使用纤维工艺参数"></div>
            </div>
          </section>

          <section class="prusaAdvancedGroup">
            <h4>现场动作与导出注入</h4>
            <p class="notice">机械臂换头、CUT 和挤出等待参数保留在 process_core 中，供现场调试；导出时与轨迹统一注入。</p>
            <div class="prusaSettingsGrid">
              <div class="fieldGroup"><label for="coreToolSafeLift">换头安全抬升 mm</label><input id="coreToolSafeLift" type="number" min="0" step="0.001" value="20"></div>
              <div class="fieldGroup"><label for="coreCutLift">CUT 抬升 mm</label><input id="coreCutLift" type="number" min="0" step="0.001" value="20"></div>
              <div class="fieldGroup"><label for="coreCutWait">CUT 等待 s</label><input id="coreCutWait" type="number" min="0" step="0.001" value="15"></div>
              <div class="fieldGroup"><label for="coreInitialTool">初始工具 ID</label><input id="coreInitialTool" type="number" min="0" step="1" value="2"></div>
              <div class="fieldGroup compactOptions"><label class="checkboxLabel" for="coreEnableExtrudeWait"><input id="coreEnableExtrudeWait" type="checkbox" checked> 启用挤出等待</label></div>
              <div class="fieldGroup compactOptions"><label class="checkboxLabel" for="coreTravelExtrudeOverlap"><input id="coreTravelExtrudeOverlap" type="checkbox" checked> 允许等待叠加到空走</label></div>
              <div class="fieldGroup compactOptions"><label class="checkboxLabel" for="coreCutAbsoluteE"><input id="coreCutAbsoluteE" type="checkbox" checked> CUT 使用绝对 E</label></div>
            </div>
          </section>
        </details>

        <div class="formSection processBand" data-layout-band="process-action">
          <div class="bandGrid">
            <div id="legacyProcessSettings" class="span-9" hidden>
            <div class="bandGrid">
            <div class="processGroup span-4">
              <div class="processTitleRow">
                <h3>筏板</h3>
                <label class="checkboxLabel" title="关闭后不生成筏板层，零件层也不会为筏板向上平移"><input id="printRaft" type="checkbox" checked> 打印筏板</label>
              </div>
              <div class="grid">
                <div>
                  <label for="raftBottomOffset" title="底层筏板外扩距离 mm">底层外扩 mm</label>
                  <input id="raftBottomOffset" name="raftBottomOffset" type="number" min="0" step="0.1" value="{DEFAULT_RAFT_OUTWARD_OFFSETS_MM[0]:g}">
                </div>
                <div>
                  <label for="raftSecondOffset" title="第 2 层筏板打印外扩距离 mm">第 2 层外扩 mm</label>
                  <input id="raftSecondOffset" name="raftSecondOffset" type="number" min="0" step="0.1" value="{DEFAULT_RAFT_OUTWARD_OFFSETS_MM[1]:g}">
                </div>
              </div>
            </div>
            <div class="processGroup span-5">
              <div class="processTitleRow"><h3>曲面 Z</h3></div>
              <div class="curveFields">
                <div class="fieldGroup">
                  <label for="curveMode">Z 模式</label>
                  <select id="curveMode" name="curveMode">
                    <option value="flat">平面层</option>
                    <option value="sinusoidal">正弦曲面</option>
                  </select>
                </div>
                <div class="fieldGroup">
                  <label for="curveAmplitude" title="曲面幅值 mm">幅值 mm</label>
                  <input id="curveAmplitude" name="curveAmplitude" type="number" step="0.001" value="0">
                </div>
                <div class="fieldGroup">
                  <label for="curvePeriod" title="曲面周期 mm">周期 mm</label>
                  <input id="curvePeriod" name="curvePeriod" type="number" min="0.001" step="0.001" value="50">
                </div>
              </div>
            </div>
            </div>
            </div>
            <div class="actions span-3">
              <div class="exportActionRow">
                <button id="sliceButton" type="submit">生成并导出 Core NPZ</button>
                <div id="exportProgress" class="exportProgress" aria-live="polite">
                <div class="exportProgressHeader">
                  <span id="exportProgressMessage">等待处理</span>
                  <strong id="exportProgressValue">0%</strong>
                </div>
                <progress id="exportProgressBar" max="100" value="0"></progress>
                <span id="exportElapsed" class="exportElapsed">已用时 0.0 秒</span>
                </div>
                <span id="status" class="status" aria-live="polite"></span>
              </div>
            </div>
          </div>
        </div>
      </form>
    </section>

  </main>

  <script>
    const form = document.getElementById('sliceForm');
    const button = document.getElementById('sliceButton');
    const surfaceToolButtons = {{
      'surface-preview': document.getElementById('surfacePreviewButton'),
      'surface-map': document.getElementById('surfaceMapperButton')
    }};
    const surfaceNpzPreviewButton = document.getElementById('surfaceNpzPreviewButton');
    const coreNpzPreviewButton = document.getElementById('coreNpzPreviewButton');
    const surfaceNpzCollisionButton = document.getElementById('surfaceNpzCollisionButton');
    const surfaceNpzCollisionResult = document.getElementById('surfaceNpzCollisionResult');
    const surfaceNpzInput = document.getElementById('surfaceNpzInput');
    const statusEl = document.getElementById('status');
    async function launchSurfaceTool(tool) {{
      const toolButton = surfaceToolButtons[tool];
      const originalLabel = toolButton.textContent;
      toolButton.disabled = true;
      toolButton.textContent = '正在启动…';
      try {{
        const response = await fetch('/launch-tool?tool=' + encodeURIComponent(tool), {{ method: 'POST' }});
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || '启动失败');
        statusEl.className = 'status ok';
        statusEl.textContent = result.already_running
          ? '该曲面工具窗口已打开。'
          : '曲面工具已在独立窗口打开；关闭该窗口后服务会自动停止。';
      }} catch (error) {{
        statusEl.className = 'status error';
        statusEl.textContent = '无法启动曲面工具：' + error.message;
      }} finally {{
        toolButton.disabled = false;
        toolButton.textContent = originalLabel;
      }}
    }}
    surfaceToolButtons['surface-preview'].addEventListener('click', () => launchSurfaceTool('surface-preview'));
    surfaceToolButtons['surface-map'].addEventListener('click', () => launchSurfaceTool('surface-map'));
    function applyMappedSurfacePreview(preview, fileName, collisionCheckAvailable = false) {{
      const isConformalLattice = preview?.preview_source === 'conformal_lattice_external_source_npz';
      previewData = preview;
      configureViewer();
      layersEl.textContent = String(previewData.layers?.length || 0);
      outputNameEl.textContent = fileName;
      downloadEl.removeAttribute('href');
      downloadEl.textContent = '已载入曲面预览（未导出）';
      statusEl.className = 'status ok';
      statusEl.textContent = isConformalLattice
        ? '已载入共形格栅 NPZ：复用主三维预览的图层、路径播放和打印头显示。'
        : '已载入映射曲面 NPZ：左键旋转，箭头尖端为当前打印点。';
      surfaceNpzCollisionButton.disabled = !collisionCheckAvailable;
      surfaceNpzCollisionResult.className = 'surfaceCollisionResult';
      surfaceNpzCollisionResult.textContent = isConformalLattice
        ? '共形格栅结构边预览不适用旧版峰值曲率碰撞检查。'
        : collisionCheckAvailable
        ? '已就绪：可检查峰值曲率层的加热块实体碰撞。'
        : '浏览器上传的 NPZ 仅供预览；请通过“导入曲面/共形 NPZ 预览”选择本地文件后检查。';
      drawPreview();
    }}
    function applyFinalCorePreview(preview, fileName) {{
      previewData = preview;
      configureViewer();
      layersEl.textContent = String(previewData.layers?.length || 0);
      outputNameEl.textContent = fileName;
      executedInfillPatternEl.textContent = '最终 Core 轨迹';
      downloadEl.removeAttribute('href');
      downloadEl.textContent = '已载入 Core 轨迹预览（未导出）';
      statusEl.className = 'status ok';
      statusEl.textContent = '已载入 Core NPZ：预览显示最终导出的 print 与 travel 实际采样轨迹。';
      surfaceNpzCollisionButton.disabled = true;
      surfaceNpzCollisionResult.className = 'surfaceCollisionResult';
      surfaceNpzCollisionResult.textContent = '当前为普通 Core NPZ 预览，不适用曲面碰撞检查。';
      drawPreview();
    }}
    async function loadMappedSurfaceNpz(file) {{
      if (!file) return;
      try {{
        const payload = new FormData();
        payload.append('source_npz', file, file.name);
        const response = await fetch('/preview-source-npz', {{ method: 'POST', body: payload }});
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || '曲面 NPZ 预览载入失败');
        applyMappedSurfacePreview(result.preview, file.name, false);
      }} catch (error) {{
        statusEl.className = 'status error';
        statusEl.textContent = '无法载入曲面 NPZ：' + error.message;
      }} finally {{
        surfaceNpzInput.value = '';
      }}
    }}
    surfaceNpzPreviewButton.addEventListener('click', async () => {{
      const originalLabel = surfaceNpzPreviewButton.textContent;
      surfaceNpzPreviewButton.disabled = true;
      surfaceNpzPreviewButton.textContent = '正在选择文件…';
      try {{
        const response = await fetch('/choose-surface-npz-preview', {{ method: 'POST' }});
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || '曲面 NPZ 选择失败');
        if (!result.cancelled) applyMappedSurfacePreview(result.preview, result.file_name, result.collision_check_available === true);
      }} catch (error) {{
        statusEl.className = 'status error';
        statusEl.textContent = '无法选择曲面 NPZ：' + error.message;
      }} finally {{
        surfaceNpzPreviewButton.disabled = false;
        surfaceNpzPreviewButton.textContent = originalLabel;
      }}
    }});
    coreNpzPreviewButton.addEventListener('click', async () => {{
      const originalLabel = coreNpzPreviewButton.textContent;
      coreNpzPreviewButton.disabled = true;
      coreNpzPreviewButton.textContent = '正在选择文件…';
      try {{
        const response = await fetch('/choose-core-npz-preview', {{ method: 'POST' }});
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || 'Core NPZ 选择失败');
        if (!result.cancelled) applyFinalCorePreview(result.preview, result.file_name);
      }} catch (error) {{
        statusEl.className = 'status error';
        statusEl.textContent = '无法载入 Core NPZ：' + error.message;
      }} finally {{
        coreNpzPreviewButton.disabled = false;
        coreNpzPreviewButton.textContent = originalLabel;
      }}
    }});
    surfaceNpzCollisionButton.addEventListener('click', async () => {{
      const originalLabel = surfaceNpzCollisionButton.textContent;
      surfaceNpzCollisionButton.disabled = true;
      surfaceNpzCollisionButton.textContent = '正在检查峰值层…';
      surfaceNpzCollisionResult.className = 'surfaceCollisionResult';
      surfaceNpzCollisionResult.textContent = '正在检查峰值曲率层，请稍候…';
      statusEl.className = 'status';
      statusEl.textContent = '正在以加热块实体和蜂窝 STL 孔洞截面检查最大曲率打印层…';
      try {{
        const response = await fetch('/check-surface-npz-collision', {{ method: 'POST' }});
        const result = await response.json();
        if (!response.ok || !result.ok) throw new Error(result.error || '碰撞检查失败');
        if (result.passed) {{
          statusEl.className = 'status ok';
          const clearance = Number(result.minimum_sampled_clearance_mm);
          const clearanceText = Number.isFinite(clearance) ? clearance.toFixed(3) + ' mm' : '无有效候选点';
          const coarsePitch = Number(result.coarse_sampling_pitch_mm);
          const finePitch = Number(result.refinement_sampling_pitch_mm);
          const refinedPoses = Number(result.refinement?.tested_pose_count);
          const refinementText = Number.isFinite(finePitch) && Number.isFinite(refinedPoses)
            ? `；${{coarsePitch.toFixed(1)}} mm 全路径初筛后，以 ${{finePitch.toFixed(1)}} mm 复核最小净空附近 ${{refinedPoses}} 个姿态`
            : '';
          const message = `通过：峰值层 ${{result.peak_layers.join('、')}} 未发现加热块实体相交；最小采样净空 ${{clearanceText}}（最终采样 ${{result.sampling_pitch_mm}} mm）${{refinementText}}。`;
          statusEl.textContent = message;
          surfaceNpzCollisionResult.className = 'surfaceCollisionResult ok';
          surfaceNpzCollisionResult.textContent = message;
        }} else {{
          const hit = result.collision;
          statusEl.className = 'status error';
          statusEl.textContent = `检测到碰撞：第 ${{hit.layer}} 层，峰值层路径姿态 #${{hit.pose_index + 1}}。已停止检查；加热块实体与蜂窝材料区域相交。`;
          surfaceNpzCollisionResult.className = 'surfaceCollisionResult error';
          surfaceNpzCollisionResult.textContent = statusEl.textContent;
        }}
      }} catch (error) {{
        statusEl.className = 'status error';
        statusEl.textContent = '碰撞检查无法完成：' + error.message;
        surfaceNpzCollisionResult.className = 'surfaceCollisionResult error';
        surfaceNpzCollisionResult.textContent = statusEl.textContent;
      }} finally {{
        surfaceNpzCollisionButton.disabled = false;
        surfaceNpzCollisionButton.textContent = originalLabel;
      }}
    }});
    surfaceNpzInput.addEventListener('change', async () => {{
      await loadMappedSurfaceNpz(surfaceNpzInput.files?.[0]);
    }});
    const exportProgressEl = document.getElementById('exportProgress');
    const exportProgressBarEl = document.getElementById('exportProgressBar');
    const exportProgressMessageEl = document.getElementById('exportProgressMessage');
    const exportProgressValueEl = document.getElementById('exportProgressValue');
    const exportElapsedEl = document.getElementById('exportElapsed');
    const downloadEl = document.getElementById('download');
    const layersEl = document.getElementById('layers');
    const outputNameEl = document.getElementById('outputName');
    const executedInfillPatternEl = document.getElementById('executedInfillPattern');
    const previewSurface = document.getElementById('previewSurface');
    const previewCanvas = document.getElementById('previewCanvas');
    const layerSlider = document.getElementById('layerSlider');
    const pathProgressControl = document.getElementById('pathProgressControl');
    const pathProgressSlider = document.getElementById('pathProgressSlider');
    const layerLabel = document.getElementById('layerLabel');
    const pathProgressLabel = document.getElementById('pathProgressLabel');
    const pathPlaybackControl = document.getElementById('pathPlaybackControl');
    const playCurrentPathButton = document.getElementById('playCurrentPath');
    const pathPlaybackRateInput = document.getElementById('pathPlaybackRate');
    const pathPlaybackRateLabel = document.getElementById('pathPlaybackRateLabel');
    const printSizeLabel = document.getElementById('printSizeLabel');
    const stlFileInput = document.getElementById('stlFile');
    const fiberJsonInput = document.getElementById('fiberJsonFile');
    const fiberNotice = document.getElementById('fiberNotice');
    const showLayerOverlayInput = document.getElementById('showLayerOverlay');
    const surfaceCurvatureTextCache = new WeakMap();
    let historicalOverlayCache = {{ preview: null, key: '', entries: [] }};
    let pendingPreviewFrame = null;
    const pathPlayback = {{
      entry: null,
      timeline: null,
      distanceMm: 0,
      running: false,
      frame: null,
      previousTimestamp: null,
    }};
    const PLAYBACK_MAX_SPEED_MM_PER_S = 8;
    const showLineWidthInput = document.getElementById('showLineWidth');
    const showExtrusionInput = document.getElementById('showExtrusion');
    const extrusionColorLegend = document.getElementById('extrusionColorLegend');
    const previewLineWidthValueEl = document.getElementById('previewLineWidthValue');
    const showPathPointsInput = document.getElementById('showPathPoints');
    const showDirectionInput = document.getElementById('showDirection');
    const showDirectionLabel = document.getElementById('showDirectionLabel');
    const showOuterContourInput = document.getElementById('showOuterContour');
    const showInnerContourInput = document.getElementById('showInnerContour');
    const showResinInfillInput = document.getElementById('showResinInfill');
    const showRaftPathsInput = document.getElementById('showRaftPaths');
    const showFiberPathsInput = document.getElementById('showFiberPaths');
    const showTravelPathsInput = document.getElementById('showTravelPaths');
    const showCoreTravelPathsInput = document.getElementById('showCoreTravelPaths');
    const showPrimelineInput = document.getElementById('showPrimeline');
    const slicingKernelInput = document.getElementById('slicingKernel');
    const layerHeightInput = document.getElementById('layerHeight');
    const firstLayerHeightInput = document.getElementById('firstLayerHeight');
    const lineWidthInput = document.getElementById('lineWidth');
    const planningLineWidthInput = document.getElementById('planningLineWidth');
    const infillOverlapInput = document.getElementById('infillOverlap');
    const trianglePathOptimizationInput = document.getElementById('trianglePathOptimization');
    const zigzagPathOptimizationInput = document.getElementById('zigzagPathOptimization');
    const printRaftInput = document.getElementById('printRaft');
    const raftBottomOffsetInput = document.getElementById('raftBottomOffset');
    const raftSecondOffsetInput = document.getElementById('raftSecondOffset');
    const prusaNativeSettings = document.getElementById('prusaNativeSettings');
    document.getElementById('prusaSkirtEnabled')?.closest('.prusaFeatureToggle')?.remove();
    document.getElementById('prusaSkirtSettings')?.remove();
    const prusaQuickFields = document.querySelectorAll('.prusaQuickField');
    const corePrimelineEnabledInput = document.getElementById('corePrimelineEnabled');
    const corePrimelineParameterIds = ['corePrimelineX', 'corePrimelineY', 'corePrimelineLength'];
    const prusaRaftEnabledInput = document.getElementById('prusaRaftEnabled');
    const prusaRaftAutoContactInput = document.getElementById('prusaRaftAutoContact');
    const prusaRaftSettings = document.getElementById('prusaRaftSettings');
    const prusaRaftSettingIds = [
      'prusaRaftLayers', 'prusaRaftExpansion', 'prusaRaftFirstLayerDensity',
      'prusaRaftFirstLayerExpansion', 'prusaRaftContactDistance',
      'prusaRaftContactLayerHeight', 'prusaRaftContactDensity',
      'prusaRaftContactExtrusionWidth'
    ];
    const prusaBrimEnabledInput = document.getElementById('prusaBrimEnabled');
    const prusaBrimSettings = document.getElementById('prusaBrimSettings');
    const prusaBrimSettingIds = [
      'prusaBrimWidth', 'prusaBrimType', 'prusaBrimSeparation', 'prusaBrimOneStroke'
    ];
    const legacyNativeSettings = document.getElementById('legacyNativeSettings');
    const legacyProcessSettings = document.getElementById('legacyProcessSettings');
    const legacyInfillControl = document.getElementById('legacyInfillControl');
    const infillPatternInput = document.getElementById('infillPattern');
    const infillSafetyNote = document.getElementById('infillSafetyNote');
    const pyslmNativeSettings = document.getElementById('pyslmNativeSettings');
    const pyslmHatcherInput = document.getElementById('pyslmHatcher');
    const pyslmPatternAutoInput = document.getElementById('pyslmPatternAuto');
    const stripeParameterIds = ['pyslmStripeWidth', 'pyslmStripeOverlap', 'pyslmStripeOffset'];
    const islandParameterIds = ['pyslmIslandWidth', 'pyslmIslandOverlap', 'pyslmIslandOffset'];
    const pyslmSupportedPatterns = new Set(['zigzag_horizontal', 'zigzag_vertical', 'zigzag_plus45', 'zigzag_minus45', 'isotropic']);
    const strictLayeredFallbackPatterns = {{
      grid: '严格实测线宽模式下，同层交叉会改为按层 0°/90° 单向之字形。',
      triangles: '严格实测线宽模式下，同层交叉会改为按层 0°/60°/120° 单向之字形。',
      gyroid: '严格实测线宽模式下，曲线局部近接会改为按层 45°/-45° 单向之字形。'
    }};
    const pyslmSettingsIds = [
      'pyslmHatcher', 'pyslmHatchSort', 'pyslmHatchAngle', 'pyslmLayerAngleIncrement',
      'pyslmHatchDistance', 'pyslmContourOffset', 'pyslmSpotCompensation',
      'pyslmVolumeOffset', 'pyslmOuterContours', 'pyslmInnerContours',
      'pyslmStripeWidth', 'pyslmStripeOverlap', 'pyslmStripeOffset',
      'pyslmIslandWidth', 'pyslmIslandOverlap', 'pyslmIslandOffset',
      'pyslmSimplificationFactor', 'pyslmSimplificationMode',
      'pyslmScanContourFirst', 'pyslmFixPolygons', 'pyslmSimplificationPreserveTopology'
    ];
    const initialCoreParams = {json.dumps(asdict(core_defaults), ensure_ascii=False)};
    const initialPrusaParams = {json.dumps(prusa_saved, ensure_ascii=False)};
    let previewData = null;
    let settingsSaveTimer = null;
    const coreNumberSettings = [
      ['coreResinLayerHeight', 'core_resin_layer_height', 'resin', 'layer_height_mm'],
      ['coreResinExtrusionScale', 'core_resin_extrusion_scale', 'resin', 'extrusion_scale'],
      ['coreResinFeed', 'core_resin_feed', 'resin', 'feed_mm_s'],
      ['coreResinFirstLayerFeed', 'core_resin_first_layer_feed', 'resin', 'first_layer_feed_mm_s'],
      ['coreResinTemp', 'core_resin_temp', 'resin', 'temperature_c'],
      ['coreResinPrimeLength', 'core_resin_prime_length', 'resin', 'prime_length_mm'],
      ['coreResinPrimeSpeed', 'core_resin_prime_speed', 'resin', 'prime_speed_mm_s'],
      ['coreResinRetractLength', 'core_resin_retract_length', 'resin', 'retract_length_mm'],
      ['coreResinRetractSpeed', 'core_resin_retract_speed', 'resin', 'retract_speed_mm_s'],
      ['coreResinEOverride', 'core_resin_e_override', 'resin', 'e_per_mm_override'],
      ['coreFiberLayerHeight', 'core_fiber_layer_height', 'fiber', 'layer_height_mm'],
      ['coreFiberExtrusionScale', 'core_fiber_extrusion_scale', 'fiber', 'extrusion_scale'],
      ['coreFiberFeed', 'core_fiber_feed', 'fiber', 'feed_mm_s'],
      ['coreFiberFirstLayerFeed', 'core_fiber_first_layer_feed', 'fiber', 'first_layer_feed_mm_s'],
      ['coreFiberTemp', 'core_fiber_temp', 'fiber', 'temperature_c'],
      ['coreFiberPrimeLength', 'core_fiber_prime_length', 'fiber', 'prime_length_mm'],
      ['coreFiberPrimeSpeed', 'core_fiber_prime_speed', 'fiber', 'prime_speed_mm_s'],
      ['coreFiberRetractLength', 'core_fiber_retract_length', 'fiber', 'retract_length_mm'],
      ['coreFiberRetractSpeed', 'core_fiber_retract_speed', 'fiber', 'retract_speed_mm_s'],
      ['coreFiberStartAccel', 'core_fiber_start_accel', 'fiber', 'start_accel_s'],
      ['coreTravelFeed', 'core_travel_feed', 'root', 'travel_feed_mm_s'],
      ['coreFirstLayerTravelFeed', 'core_first_layer_travel_feed', 'root', 'first_layer_travel_feed_mm_s'],
      ['corePrimeSettle', 'core_prime_settle', 'root', 'prime_settle_s'],
      ['coreDefaultA', 'core_default_a', 'root', 'default_a'],
      ['coreDefaultB', 'core_default_b', 'root', 'default_b'],
      ['coreDefaultC', 'core_default_c', 'root', 'default_c'],
      ['corePrimelineX', 'core_primeline_x_mm', 'root', 'primeline_x_mm'],
      ['corePrimelineY', 'core_primeline_y_mm', 'root', 'primeline_y_mm'],
      ['corePrimelineLength', 'core_primeline_length', 'root', 'primeline_length_mm'],
      ['coreDt', 'core_dt', 'root', 'dt'],
      ['coreCornerAngle', 'core_corner_angle', 'root', 'corner_angle_deg'],
      ['coreCornerRetreatRatio', 'core_corner_retreat_ratio', 'root', 'corner_retreat_ratio'],
      ['coreSplineMaxError', 'core_spline_max_error', 'root', 'spline_max_error_mm'],
      ['coreSplineMaxAngle', 'core_spline_max_angle', 'root', 'spline_max_angle_deg'],
      ['coreSourceMergeDistance', 'core_source_merge_distance', 'root', 'source_merge_distance_mm'],
      ['coreCornerRetreatMax', 'core_corner_retreat_max', 'root', 'corner_retreat_max_mm'],
      ['coreCornerBlendSegments', 'core_corner_blend_segments', 'root', 'corner_blend_segments'],
      ['coreDensity', 'core_density', 'root', 'density'],
      ['coreDegree', 'core_degree', 'root', 'degree'],
      ['coreMaxFitPoints', 'core_max_fit_points', 'root', 'max_fit_points_per_segment'],
      ['coreFiberOffsetX', 'core_fiber_offset_x', 'export', 'fiber_x_print_compensation_mm'],
      ['coreFiberOffsetY', 'core_fiber_offset_y', 'export', 'fiber_y_print_compensation_mm'],
      ['coreFiberOffsetZ', 'core_fiber_offset_z', 'export', 'fiber_z_print_compensation_mm'],
      ['coreResinZComp', 'core_resin_z_comp', 'export', 'resin_z_print_compensation_mm'],
      ['coreToolSafeLift', 'core_tool_safe_lift', 'export', 'tool_change_safe_lift_mm'],
      ['coreCutLift', 'core_cut_lift', 'export', 'cut_lift_mm'],
      ['coreCutWait', 'core_cut_wait', 'export', 'cut_wait_s'],
      ['coreFiberRetractOverride', 'core_fiber_retract_override', 'export', 'fiber_retract_length_mm'],
      ['coreInitialTool', 'core_initial_tool', 'export', 'initial_tool_id']
    ];
    const coreBooleanSettings = [
      ['corePrimelineEnabled', 'core_primeline_enabled', 'root', 'primeline_enabled'],
      ['coreEnableExtrudeWait', 'core_enable_extrude_wait', 'export', 'enable_extrude_wait'],
      ['coreTravelExtrudeOverlap', 'core_enable_travel_extrude_overlap', 'export', 'enable_travel_extrude_overlap'],
      ['coreCutAbsoluteE', 'core_external_npz_cut_absolute_e', 'export', 'external_npz_cut_absolute_e']
    ];
    const prusaNumberSettings = [
      ['layerHeight', 'layer_height'], ['firstLayerHeight', 'first_layer_height'],
      ['lineWidth', 'line_width'], ['prusaStartX', 'prusa_start_x_mm'],
      ['prusaStartY', 'prusa_start_y_mm'], ['prusaPerimeterCount', 'prusa_perimeter_count'],
      ['prusaInfillDensity', 'prusa_infill_density'], ['prusaContourInfillOverlap', 'prusa_contour_infill_overlap'],
      ['prusaRaftLayers', 'prusa_raft_layers'], ['prusaRaftExpansion', 'prusa_raft_expansion'],
      ['prusaRaftFirstLayerDensity', 'prusa_raft_first_layer_density'],
      ['prusaRaftFirstLayerExpansion', 'prusa_raft_first_layer_expansion'],
      ['prusaRaftContactDistance', 'prusa_raft_contact_distance'],
      ['prusaRaftContactLayerHeight', 'prusa_raft_contact_layer_height'],
      ['prusaRaftContactDensity', 'prusa_raft_contact_density'],
      ['prusaRaftContactExtrusionWidth', 'prusa_raft_contact_extrusion_width'],
      ['prusaBrimWidth', 'prusa_brim_width'], ['prusaBrimSeparation', 'prusa_brim_separation'],
      ['prusaExternalPerimeterWidth', 'prusa_external_perimeter_width'],
      ['prusaPerimeterWidth', 'prusa_perimeter_width'], ['prusaInfillWidth', 'prusa_infill_width'],
      ['prusaXySizeCompensation', 'prusa_xy_size_compensation'],
      ['prusaElephantFootCompensation', 'prusa_elephant_foot_compensation'],
      ['prusaInfillAnchor', 'prusa_infill_anchor'], ['prusaInfillAnchorMax', 'prusa_infill_anchor_max'],
      ['prusaAvoidCrossingMaxDetour', 'prusa_avoid_crossing_max_detour']
    ];
    const prusaBooleanSettings = [
      ['prusaPrintPerimeters', 'prusa_print_perimeters'], ['prusaRaftEnabled', 'prusa_raft_enabled'],
      ['prusaRaftAutoContact', 'prusa_raft_auto_contact'],
      ['prusaGapFillEnabled', 'prusa_gap_fill_enabled'],
      ['prusaBrimEnabled', 'prusa_brim_enabled'], ['prusaBrimOneStroke', 'prusa_brim_one_stroke'],
      ['honeycombCenterlineEnabled', 'honeycomb_centerline_enabled']
    ];
    const prusaSelectSettings = [
      ['slicingKernel', 'slicing_kernel'], ['buildAxis', 'build_axis'],
      ['prusaInfillPattern', 'prusa_infill_pattern'], ['prusaPerimeterGenerator', 'prusa_perimeter_generator'],
      ['prusaSeamPosition', 'prusa_seam_position'], ['prusaBrimType', 'prusa_brim_type'],
      ['honeycombTopology', 'honeycomb_topology']
    ];
    function setInitialValue(id, value) {{
      const input = document.getElementById(id);
      if (!input || value === null || value === undefined || value === '') return;
      const normalized = String(value);
      if (input instanceof HTMLSelectElement && !Array.from(input.options).some((option) => option.value === normalized)) return;
      input.value = normalized;
    }}
    function applyInitialSavedSettings() {{
      const core = initialCoreParams || {{}};
      for (const [id, unused, group, key] of coreNumberSettings) {{
        const source = group === 'root' ? core : core[group];
        if (source) setInitialValue(id, source[key]);
      }}
      for (const [id, unused, group, key] of coreBooleanSettings) {{
        const input = document.getElementById(id);
        const source = group === 'root' ? core : core[group];
        if (input && source && source[key] !== undefined) input.checked = Boolean(source[key]);
      }}
      setInitialValue('prusaStartX', initialPrusaParams.prusa_start_x_mm ?? core.start_x_mm);
      setInitialValue('prusaStartY', initialPrusaParams.prusa_start_y_mm ?? core.start_y_mm);
      setInitialValue('layerHeight', initialPrusaParams.layer_height ?? core.resin?.layer_height_mm);
      setInitialValue('firstLayerHeight', initialPrusaParams.first_layer_height ?? core.resin?.layer_height_mm);
      for (const [id, key] of prusaNumberSettings) setInitialValue(id, initialPrusaParams[key]);
      for (const [id, key] of prusaSelectSettings) setInitialValue(id, initialPrusaParams[key]);
      for (const [id, key] of prusaBooleanSettings) {{
        const input = document.getElementById(id);
        if (input && initialPrusaParams[key] !== undefined) input.checked = Boolean(initialPrusaParams[key]);
      }}
    }}
    function collectPersistentSettings() {{
      const core = {{}};
      for (const [id, name] of coreNumberSettings) {{
        const input = document.getElementById(id);
        if (input) core[name] = input.value;
      }}
      for (const [id, name] of coreBooleanSettings) {{
        const input = document.getElementById(id);
        if (input) core[name] = input.checked;
      }}
      const prusa = {{}};
      for (const [id, name] of prusaNumberSettings) {{
        const input = document.getElementById(id);
        if (input) prusa[name] = input.value;
      }}
      for (const [id, name] of prusaBooleanSettings) {{
        const input = document.getElementById(id);
        if (input) prusa[name] = input.checked;
      }}
      for (const [id, name] of prusaSelectSettings) {{
        const input = document.getElementById(id);
        if (input) prusa[name] = input.value;
      }}
      return {{ core, prusa }};
    }}
    function scheduleSettingsSave() {{
      window.clearTimeout(settingsSaveTimer);
      settingsSaveTimer = window.setTimeout(() => {{
        fetch('/ui-settings', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(collectPersistentSettings())
        }}).catch(() => {{}});
      }}, 120);
    }}
    function installSettingsPersistence() {{
      const ids = [
        ...coreNumberSettings.map(([id]) => id), ...coreBooleanSettings.map(([id]) => id),
        ...prusaNumberSettings.map(([id]) => id), ...prusaBooleanSettings.map(([id]) => id),
        ...prusaSelectSettings.map(([id]) => id)
      ];
      for (const id of ids) {{
        const input = document.getElementById(id);
        if (input) {{ input.addEventListener('input', scheduleSettingsSave); input.addEventListener('change', scheduleSettingsSave); }}
      }}
    }}
    function magnitudeStepForValue(value) {{
      const magnitude = Math.abs(Number(value));
      if (!Number.isFinite(magnitude) || magnitude === 0) return null;
      return magnitude < 1 ? 10 ** Math.floor(Math.log10(magnitude)) : 1;
    }}
    function decimalPlaces(valueText) {{
      const text = String(valueText ?? '').trim().toLowerCase();
      if (!text || !Number.isFinite(Number(text))) return 0;
      const [coefficient, exponentText] = text.split('e');
      const coefficientPlaces = (coefficient.split('.')[1] || '').length;
      const exponent = Number(exponentText || 0);
      return Math.max(0, coefficientPlaces - exponent);
    }}
    function installMagnitudeNumberStepping() {{
      for (const input of document.querySelectorAll('input[type="number"]')) {{
        if (input.dataset.magnitudeSpinnerInstalled === 'true') continue;
        input.dataset.magnitudeSpinnerInstalled = 'true';
        input.dataset.magnitudeLastValue = input.value;
        const adjustValue = (targetInput, direction) => {{
          const current = Number.parseFloat(targetInput.value);
          const step = magnitudeStepForValue(current);
          if (!Number.isFinite(current) || step === null || !Number.isFinite(direction)) return;
          const requested = current + (direction > 0 ? step : -step);
          let next = requested;
          const min = targetInput.min.trim() === '' ? null : Number(targetInput.min);
          const max = targetInput.max.trim() === '' ? null : Number(targetInput.max);
          if (min !== null && Number.isFinite(min)) next = Math.max(min, next);
          if (max !== null && Number.isFinite(max)) next = Math.min(max, next);
          const clamped = next !== requested;
          const places = Math.max(
            decimalPlaces(String(current)),
            decimalPlaces(String(step)),
            clamped ? decimalPlaces(String(next)) : 0,
          );
          targetInput.value = String(Number(next.toFixed(places)));
          targetInput.dataset.magnitudeLastValue = targetInput.value;
          targetInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
          targetInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
        }};
        const wrapper = document.createElement('span');
        wrapper.className = 'magnitudeInputWrap';
        input.parentNode.insertBefore(wrapper, input);
        wrapper.appendChild(input);
        const buttons = document.createElement('span');
        buttons.className = 'magnitudeSpinButtons';
        const makeButton = (symbol, direction, label) => {{
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'magnitudeSpinButton';
          button.textContent = symbol;
          button.dataset.magnitudeDirection = String(direction);
          button.setAttribute('aria-label', label);
          button.addEventListener('pointerdown', (event) => event.preventDefault());
          button.addEventListener('click', (event) => {{
            event.preventDefault();
            event.stopPropagation();
            const targetInput = event.currentTarget.closest('.magnitudeInputWrap')?.querySelector('input[type="number"]');
            if (!targetInput || targetInput.disabled) return;
            adjustValue(targetInput, Number(event.currentTarget.dataset.magnitudeDirection));
          }});
          buttons.appendChild(button);
        }};
        makeButton('▲', 1, '增加当前数量级');
        makeButton('▼', -1, '减少当前数量级');
        wrapper.appendChild(buttons);
        // Keep the native spinner appearance while preserving the existing
        // magnitude-based increment/decrement behavior.
        input.addEventListener('pointerdown', (event) => {{
          if (input.disabled || event.button !== 0) return;
          const rect = input.getBoundingClientRect();
          const localX = event.clientX - rect.left;
          const localY = event.clientY - rect.top;
          const spinnerWidth = Math.min(24, Math.max(16, rect.height * 0.9));
          if (
            !Number.isFinite(localX) || !Number.isFinite(localY)
            || localX < rect.width - spinnerWidth
            || localY < 0 || localY > rect.height
          ) return;
          event.preventDefault();
          event.stopPropagation();
          adjustValue(input, localY < rect.height / 2 ? 1 : -1);
        }});
        input.addEventListener('keydown', (event) => {{
          if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
          event.preventDefault();
          adjustValue(input, event.key === 'ArrowUp' ? 1 : -1);
        }});
        input.addEventListener('input', () => {{
          input.dataset.magnitudeLastValue = input.value;
        }});
        input.addEventListener('change', () => {{
          input.dataset.magnitudeLastValue = input.value;
        }});
      }}
    }}
    function installAdvancedPopups() {{
      const advancedIds = ['prusaAdvancedSettings', 'coreProcessSettings'];
      let activePopup = null;
      let activeSummary = null;
      let dragState = null;

      const clampPopup = (popup) => {{
        const rect = popup.getBoundingClientRect();
        const left = Math.max(12, Math.min(popup.offsetLeft, window.innerWidth - rect.width - 12));
        const isCorePopup = popup.classList.contains('coreAdvancedPopup');
        const top = isCorePopup
          ? Math.max(12, popup.offsetTop)
          : Math.max(12, Math.min(popup.offsetTop, window.innerHeight - rect.height - 12));
        popup.style.left = left + 'px';
        popup.style.top = top + 'px';
      }};
      const closePopup = (popup, trigger) => {{
        popup.classList.remove('visible');
        popup.setAttribute('aria-hidden', 'true');
        trigger.setAttribute('aria-expanded', 'false');
        if (activePopup === popup) {{
          activePopup = null;
          activeSummary = null;
        }}
      }};
      const openPopup = (popup, trigger) => {{
        if (activePopup && activePopup !== popup && activeSummary) closePopup(activePopup, activeSummary);
        activePopup = popup;
        activeSummary = trigger;
        const isCorePopup = popup.closest('#coreProcessSettings') !== null;
        const widthLimit = isCorePopup ? 1180 : 900;
        const heightLimit = isCorePopup ? 960 : 680;
        const width = Math.min(widthLimit, window.innerWidth - 24);
        const height = Math.min(heightLimit, window.innerHeight - 24);
        popup.classList.toggle('coreAdvancedPopup', isCorePopup);
        popup.style.width = Math.max(320, width) + 'px';
        popup.style.height = isCorePopup ? 'auto' : Math.max(180, height) + 'px';
        popup.style.maxHeight = isCorePopup ? 'none' : '';
        popup.style.left = isCorePopup
          ? Math.max(12, window.scrollX + (window.innerWidth - width) / 2) + 'px'
          : Math.max(12, (window.innerWidth - width) / 2) + 'px';
        popup.style.top = isCorePopup
          ? Math.max(12, window.scrollY + 12) + 'px'
          : Math.max(12, (window.innerHeight - height) / 2) + 'px';
        popup.style.transform = 'none';
        popup.classList.add('visible');
        popup.setAttribute('aria-hidden', 'false');
        trigger.setAttribute('aria-expanded', 'true');
        clampPopup(popup);
      }};

      for (const id of advancedIds) {{
        const host = document.getElementById(id);
        if (!host) continue;
        const summary = host.querySelector(':scope > summary');
        if (!summary) continue;
        const trigger = document.createElement('button');
        trigger.type = 'button';
        trigger.className = 'advancedPopupTrigger';
        trigger.textContent = summary.textContent.trim();
        trigger.setAttribute('aria-haspopup', 'dialog');
        trigger.setAttribute('aria-expanded', 'false');
        summary.hidden = true;
        host.insertBefore(trigger, summary);
        const body = document.createElement('div');
        body.className = 'advancedPopupBody';
        while (summary.nextSibling) body.appendChild(summary.nextSibling);
        const popup = document.createElement('div');
        popup.className = 'advancedPopup';
        popup.setAttribute('role', 'dialog');
        popup.setAttribute('aria-modal', 'false');
        popup.setAttribute('aria-hidden', 'true');
        const header = document.createElement('div');
        header.className = 'advancedPopupHeader';
        const title = document.createElement('span');
        title.className = 'advancedPopupTitle';
        title.textContent = summary.textContent.trim();
        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'advancedPopupClose';
        closeButton.setAttribute('aria-label', '关闭');
        closeButton.textContent = '×';
        header.append(title, closeButton);
        popup.append(header, body);
        host.appendChild(popup);
        host.open = true;
        trigger.addEventListener('click', () => {{
          if (popup.classList.contains('visible')) closePopup(popup, trigger);
          else openPopup(popup, trigger);
        }});
        closeButton.addEventListener('click', () => closePopup(popup, trigger));
        header.addEventListener('pointerdown', (event) => {{
          if (event.target === closeButton) return;
          dragState = {{
            popup,
            pointerId: event.pointerId,
            startX: event.clientX,
            startY: event.clientY,
            left: popup.offsetLeft,
            top: popup.offsetTop
          }};
          header.setPointerCapture(event.pointerId);
        }});
        header.addEventListener('pointermove', (event) => {{
          if (!dragState || dragState.popup !== popup || dragState.pointerId !== event.pointerId) return;
          popup.style.left = dragState.left + event.clientX - dragState.startX + 'px';
          popup.style.top = dragState.top + event.clientY - dragState.startY + 'px';
          clampPopup(popup);
        }});
        const finishDrag = (event) => {{
          if (dragState?.popup === popup && dragState.pointerId === event.pointerId) dragState = null;
        }};
        header.addEventListener('pointerup', finishDrag);
        header.addEventListener('pointercancel', finishDrag);
      }}
      document.addEventListener('pointerdown', (event) => {{
        if (activePopup && !activePopup.contains(event.target) && event.target !== activeSummary) {{
          closePopup(activePopup, activeSummary);
        }}
      }});
      document.addEventListener('keydown', (event) => {{
        if (event.key === 'Escape' && activePopup && activeSummary) {{
          event.preventDefault();
          closePopup(activePopup, activeSummary);
          activeSummary.focus();
        }}
      }});
      window.addEventListener('resize', () => {{
        if (activePopup) clampPopup(activePopup);
      }});
    }}
    function installPopupSelects() {{
      const selects = Array.from(document.querySelectorAll('select'));
      if (!selects.length) return;

      const layer = document.createElement('div');
      layer.className = 'selectPopupLayer';
      layer.hidden = true;
      layer.innerHTML = `
        <div id="selectPopup" class="selectPopup" role="dialog" aria-modal="true" hidden>
          <div class="selectPopupHeader">
            <span class="selectPopupTitle"></span>
            <button type="button" class="selectPopupClose" aria-label="关闭">×</button>
          </div>
          <div class="selectPopupOptions" role="listbox"></div>
        </div>`;
      document.body.appendChild(layer);

      const popup = layer.querySelector('.selectPopup');
      const title = layer.querySelector('.selectPopupTitle');
      const closeButton = layer.querySelector('.selectPopupClose');
      const optionsPanel = layer.querySelector('.selectPopupOptions');
      let activeSelect = null;
      let activeTrigger = null;
      let dragState = null;

      const close = () => {{
        if (activeTrigger) activeTrigger.setAttribute('aria-expanded', 'false');
        activeSelect = null;
        activeTrigger = null;
        popup.hidden = true;
        layer.hidden = true;
        layer.style.pointerEvents = 'none';
      }};
      const updateTrigger = (select, trigger) => {{
        const selected = select.options[select.selectedIndex];
        trigger.querySelector('.popupSelectLabel').textContent = selected?.textContent || '';
        trigger.disabled = select.disabled;
      }};
      const clampPopup = () => {{
        if (popup.hidden) return;
        const width = popup.getBoundingClientRect().width;
        const height = popup.getBoundingClientRect().height;
        const left = Math.max(12, Math.min(Number.parseFloat(popup.style.left) || 12, window.innerWidth - width - 12));
        const top = Math.max(12, Math.min(Number.parseFloat(popup.style.top) || 12, window.innerHeight - height - 12));
        popup.style.left = left + 'px';
        popup.style.top = top + 'px';
      }};
      const open = (select, trigger) => {{
        if (select.disabled) return;
        if (activeSelect === select) {{
          close();
          return;
        }}
        activeSelect = select;
        activeTrigger = trigger;
        title.textContent = select.dataset.popupTitle || select.id || '选择';
        optionsPanel.innerHTML = '';
        for (const option of select.options) {{
          const optionButton = document.createElement('button');
          optionButton.type = 'button';
          optionButton.className = 'selectPopupOption';
          optionButton.textContent = option.textContent;
          optionButton.disabled = option.disabled;
          optionButton.setAttribute('role', 'option');
          optionButton.setAttribute('aria-selected', option.value === select.value ? 'true' : 'false');
          optionButton.addEventListener('click', () => {{
            select.value = option.value;
            select.dispatchEvent(new Event('change', {{ bubbles: true }}));
            updateTrigger(select, trigger);
            close();
            trigger.focus();
          }});
          optionsPanel.appendChild(optionButton);
        }}
        const triggerRect = trigger.getBoundingClientRect();
        const width = Math.min(Math.max(triggerRect.width, 240), Math.max(240, window.innerWidth - 24));
        popup.style.width = width + 'px';
        popup.style.left = Math.max(12, Math.min(triggerRect.left, window.innerWidth - width - 12)) + 'px';
        popup.style.top = Math.max(12, Math.min(triggerRect.bottom + 8, window.innerHeight - 220)) + 'px';
        layer.hidden = false;
        layer.style.pointerEvents = 'auto';
        popup.hidden = false;
        clampPopup();
        optionsPanel.querySelector('[aria-selected="true"]')?.focus();
      }};

      closeButton.addEventListener('click', close);
      layer.addEventListener('pointerdown', (event) => {{
        if (event.target === layer) close();
      }});
      document.addEventListener('keydown', (event) => {{
        if (event.key === 'Escape' && !popup.hidden) {{
          event.preventDefault();
          close();
        }}
      }});
      window.addEventListener('resize', clampPopup);
      const header = layer.querySelector('.selectPopupHeader');
      header.addEventListener('pointerdown', (event) => {{
        if (event.target === closeButton || popup.hidden) return;
        dragState = {{
          pointerId: event.pointerId,
          startX: event.clientX,
          startY: event.clientY,
          left: popup.offsetLeft,
          top: popup.offsetTop
        }};
        header.setPointerCapture(event.pointerId);
      }});
      header.addEventListener('pointermove', (event) => {{
        if (!dragState || dragState.pointerId !== event.pointerId) return;
        popup.style.left = dragState.left + event.clientX - dragState.startX + 'px';
        popup.style.top = dragState.top + event.clientY - dragState.startY + 'px';
        clampPopup();
      }});
      const finishDrag = (event) => {{
        if (dragState?.pointerId === event.pointerId) dragState = null;
      }};
      header.addEventListener('pointerup', finishDrag);
      header.addEventListener('pointercancel', finishDrag);

      for (const select of selects) {{
        const wrapper = document.createElement('div');
        wrapper.className = 'popupSelect';
        const trigger = document.createElement('button');
        trigger.type = 'button';
        trigger.className = 'popupSelectTrigger';
        trigger.setAttribute('aria-haspopup', 'dialog');
        trigger.setAttribute('aria-expanded', 'false');
        trigger.setAttribute('aria-controls', 'selectPopup');
        const label = document.createElement('span');
        label.className = 'popupSelectLabel';
        trigger.appendChild(label);
        select.classList.add('popupSelectNative');
        select.setAttribute('aria-hidden', 'true');
        select.tabIndex = -1;
        const labelElement = document.querySelector('label[for="' + select.id + '"]');
        select.dataset.popupTitle = labelElement?.textContent?.trim() || select.id || '选择';
        select.parentNode.insertBefore(wrapper, select);
        wrapper.appendChild(select);
        wrapper.appendChild(trigger);
        updateTrigger(select, trigger);
        select.addEventListener('change', () => updateTrigger(select, trigger));
        const observer = new MutationObserver(() => updateTrigger(select, trigger));
        observer.observe(select, {{ attributes: true, attributeFilter: ['disabled'] }});
        trigger.addEventListener('click', () => open(select, trigger));
        trigger.addEventListener('keydown', (event) => {{
          if (event.key === 'Enter' || event.key === ' ' || event.key === 'ArrowDown') {{
            event.preventDefault();
            open(select, trigger);
          }}
        }});
      }}
    }}
    const viewerState = {{
      zoom: 1.0,
      centerX: null,
      centerY: null,
      centerZ: null,
      surfaceYaw: -0.72,
      surfacePitch: -0.58,
      surfacePanX: 0,
      surfacePanY: 0,
      dragging: false,
      dragMode: null,
      pointerId: null,
      lastX: 0,
      lastY: 0
    }};
    const printHeadAsset = {{ data: null, error: null }};
    fetch('/assets/printhead/printhead_interference_check.preview.json?v=mesh-preflight-v6')
      .then((response) => {{
        if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
        return response.json();
      }})
      .then((data) => {{
        printHeadAsset.data = data;
        drawPreview();
      }})
      .catch((error) => {{
        printHeadAsset.error = error;
        console.warn('Printhead preview asset unavailable:', error);
      }});
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
      const isLegacy = slicingKernelInput.value === 'legacy';
      const setPanelState = (panel, active) => {{
        panel.hidden = !active;
        for (const control of panel.querySelectorAll('input, select, textarea')) {{
          control.disabled = !active;
        }}
      }};
      setPanelState(prusaNativeSettings, !isPyslm && !isLegacy);
      for (const field of prusaQuickFields) {{
        field.hidden = isPyslm || isLegacy;
        for (const control of field.querySelectorAll('input, select, textarea')) {{
          control.disabled = isPyslm || isLegacy;
        }}
      }}
      setPanelState(legacyNativeSettings, isLegacy);
      setPanelState(legacyProcessSettings, isLegacy);
      setPanelState(pyslmNativeSettings, isPyslm);
      raftBottomOffsetInput.disabled = !isLegacy || !printRaftInput.checked;
      raftSecondOffsetInput.disabled = !isLegacy || !printRaftInput.checked;
      const prusaRaftEnabled = !isPyslm && !isLegacy && prusaRaftEnabledInput.checked;
      prusaRaftSettings.hidden = !prusaRaftEnabled;
      for (const id of prusaRaftSettingIds) {{
        document.getElementById(id).disabled = !prusaRaftEnabled;
      }}
      prusaRaftAutoContactInput.disabled = !prusaRaftEnabled;
      const manualContactEnabled = prusaRaftEnabled && !prusaRaftAutoContactInput.checked;
      for (const id of ['prusaRaftContactLayerHeight', 'prusaRaftContactDensity', 'prusaRaftContactExtrusionWidth']) {{
        document.getElementById(id).disabled = !manualContactEnabled;
      }}
      const prusaBrimEnabled = !isPyslm && !isLegacy && prusaBrimEnabledInput.checked;
      prusaBrimSettings.hidden = !prusaBrimEnabled;
      for (const id of prusaBrimSettingIds) {{
        document.getElementById(id).disabled = !prusaBrimEnabled;
      }}
      const primelineEnabled = corePrimelineEnabledInput.checked;
      for (const id of corePrimelineParameterIds) {{
        document.getElementById(id).disabled = !primelineEnabled;
      }}
      infillSafetyNote.textContent = isLegacy
        ? (strictLayeredFallbackPatterns[infillPatternInput.value] || '')
        : '';
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
      updatePreviewLineWidthValue();
    }}
    function updatePreviewLineWidthValue() {{
      const previewWidth = Number(previewData?.line_widths?.resin);
      const configuredWidth = slicingKernelInput.value === 'legacy'
        ? Number(planningLineWidthInput.value)
        : Number(lineWidthInput.value);
      const width = Number.isFinite(previewWidth) && previewWidth > 0
        ? previewWidth
        : configuredWidth;
      previewLineWidthValueEl.textContent = Number.isFinite(width) && width > 0
        ? String(Number(width.toFixed(3))) + ' mm'
        : '-';
    }}
    slicingKernelInput.addEventListener('change', syncKernelControls);
    infillPatternInput.addEventListener('change', syncKernelControls);
    pyslmHatcherInput.addEventListener('change', syncKernelControls);
    pyslmPatternAutoInput.addEventListener('change', syncKernelControls);
    layerHeightInput.addEventListener('input', syncKernelControls);
    lineWidthInput.addEventListener('input', syncKernelControls);
    planningLineWidthInput.addEventListener('input', updatePreviewLineWidthValue);
    printRaftInput.addEventListener('change', () => {{
      const disabled = !printRaftInput.checked || slicingKernelInput.value !== 'legacy';
      raftBottomOffsetInput.disabled = disabled;
      raftSecondOffsetInput.disabled = disabled;
    }});
    prusaRaftEnabledInput.addEventListener('change', syncKernelControls);
    prusaRaftAutoContactInput.addEventListener('change', syncKernelControls);
    prusaBrimEnabledInput.addEventListener('change', syncKernelControls);
    corePrimelineEnabledInput.addEventListener('change', syncKernelControls);
    applyInitialSavedSettings();
    installAdvancedPopups();
    syncKernelControls();
    installMagnitudeNumberStepping();
    installSettingsPersistence();
    fiberNotice.textContent = 'JSON 中的单层纤维路径会复制到每个树脂层，纤维层高 0.1 mm 会计入后续树脂层 Z 位置，最后一层树脂封顶不打印纤维。';

    function updateExportProgress(job) {{
      const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
      exportProgressEl.classList.add('visible');
      exportProgressBarEl.value = progress;
      exportProgressValueEl.textContent = progress + '%';
      exportProgressMessageEl.textContent = job.state === 'complete'
        ? '完成'
        : job.state === 'error'
          ? '失败'
          : progress > 0
            ? '处理中'
            : '准备中';
      const elapsed = Number(job.elapsed_s);
      exportElapsedEl.textContent = '已用时 ' + (Number.isFinite(elapsed) ? elapsed.toFixed(1) : '0.0') + ' 秒';
    }}

    async function waitForSliceJob(jobId) {{
      while (true) {{
        const response = await fetch('/slice-status?job_id=' + encodeURIComponent(jobId), {{
          cache: 'no-store'
        }});
        const job = await response.json();
        if (!response.ok || !job.ok) throw new Error(job.error || '无法读取导出进度');
        updateExportProgress(job);
        if (job.state === 'complete') return job.result;
        if (job.state === 'error') throw new Error(job.error || '导出失败');
        await new Promise(resolve => setTimeout(resolve, 250));
      }}
    }}

    form.addEventListener('submit', async (event) => {{
      event.preventDefault();
      const file = stlFileInput.files[0];
      const fiberFile = fiberJsonInput.files[0];
      if (!file) return;

      button.disabled = true;
      button.textContent = '处理中…';
      statusEl.textContent = '处理中';
      statusEl.className = 'status';
      downloadEl.className = 'download';
      updateExportProgress({{ progress: 0, message: '正在提交任务', elapsed_s: 0 }});

      const formData = new FormData();
      formData.append('stl_file', file, file.name);
      if (fiberFile) formData.append('fiber_json', fiberFile, fiberFile.name);
      formData.append('filename', file.name);
      formData.append('layer_height', document.getElementById('layerHeight').value);
      formData.append('first_layer_height', document.getElementById('firstLayerHeight').value);
      formData.append('line_width', document.getElementById('lineWidth').value);
      formData.append('build_axis', document.getElementById('buildAxis').value);
       formData.append('prusa_start_x_mm', document.getElementById('prusaStartX').value);
       formData.append('prusa_start_y_mm', document.getElementById('prusaStartY').value);
       const coreFieldIds = [
         'coreResinLayerHeight', 'coreResinExtrusionScale', 'coreResinFeed',
         'coreResinFirstLayerFeed', 'coreResinTemp', 'coreResinPrimeLength',
         'coreResinPrimeSpeed', 'coreResinRetractLength', 'coreResinRetractSpeed',
         'coreResinEOverride', 'coreFiberLayerHeight', 'coreFiberExtrusionScale',
         'coreFiberFeed', 'coreFiberFirstLayerFeed', 'coreFiberTemp',
         'coreFiberPrimeLength', 'coreFiberPrimeSpeed', 'coreFiberRetractLength',
         'coreFiberRetractSpeed', 'coreFiberStartAccel', 'coreTravelFeed',
         'coreFirstLayerTravelFeed', 'corePrimeSettle', 'coreDefaultA',
         'coreDefaultB', 'coreDefaultC', 'corePrimelineX', 'corePrimelineY',
         'corePrimelineLength', 'coreDt', 'coreCornerAngle',
         'coreCornerRetreatRatio', 'coreSplineMaxError', 'coreSplineMaxAngle',
         'coreSourceMergeDistance', 'coreCornerRetreatMax', 'coreCornerBlendSegments',
         'coreDensity', 'coreDegree', 'coreMaxFitPoints', 'coreFiberOffsetX',
         'coreFiberOffsetY', 'coreFiberOffsetZ', 'coreResinZComp',
         'coreToolSafeLift', 'coreCutLift', 'coreCutWait', 'coreFiberRetractOverride',
         'coreInitialTool'
       ];
       for (const id of coreFieldIds) {{
         const input = document.getElementById(id);
         const name = 'core_' + id.slice(4).replace(/[A-Z]/g, m => '_' + m.toLowerCase());
         formData.append(name, input.value);
       }}
       formData.append('core_primeline_enabled', corePrimelineEnabledInput.checked ? 'true' : 'false');
       for (const [id, name] of [
         ['coreEnableExtrudeWait', 'core_enable_extrude_wait'],
         ['coreTravelExtrudeOverlap', 'core_enable_travel_extrude_overlap'],
         ['coreCutAbsoluteE', 'core_external_npz_cut_absolute_e']
       ]) {{
         formData.append(name, document.getElementById(id).checked ? 'true' : 'false');
       }}
      formData.append('slicing_kernel', slicingKernelInput.value);
      if (slicingKernelInput.value === 'prusa') {{
        formData.append('prusa_perimeter_count', document.getElementById('prusaPerimeterCount').value);
        formData.append('prusa_print_perimeters', document.getElementById('prusaPrintPerimeters').checked ? 'true' : 'false');
        formData.append('honeycomb_centerline_enabled', document.getElementById('honeycombCenterlineEnabled').checked ? 'true' : 'false');
        formData.append('honeycomb_topology', document.getElementById('honeycombTopology').value);
        formData.append('prusa_infill_pattern', document.getElementById('prusaInfillPattern').value);
        formData.append('prusa_infill_density', document.getElementById('prusaInfillDensity').value);
        formData.append('prusa_contour_infill_overlap', document.getElementById('prusaContourInfillOverlap').value);
        formData.append('prusa_raft_enabled', prusaRaftEnabledInput.checked ? 'true' : 'false');
        formData.append('prusa_raft_layers', document.getElementById('prusaRaftLayers').value);
        formData.append('prusa_raft_expansion', document.getElementById('prusaRaftExpansion').value);
        formData.append('prusa_raft_first_layer_density', document.getElementById('prusaRaftFirstLayerDensity').value);
        formData.append('prusa_raft_first_layer_expansion', document.getElementById('prusaRaftFirstLayerExpansion').value);
        formData.append('prusa_raft_contact_distance', document.getElementById('prusaRaftContactDistance').value);
        formData.append('prusa_raft_auto_contact', prusaRaftAutoContactInput.checked ? 'true' : 'false');
        formData.append('prusa_raft_contact_layer_height', document.getElementById('prusaRaftContactLayerHeight').value);
        formData.append('prusa_raft_contact_density', document.getElementById('prusaRaftContactDensity').value);
        formData.append('prusa_raft_contact_extrusion_width', document.getElementById('prusaRaftContactExtrusionWidth').value);
        formData.append('prusa_brim_enabled', prusaBrimEnabledInput.checked ? 'true' : 'false');
        formData.append('prusa_brim_width', document.getElementById('prusaBrimWidth').value);
        formData.append('prusa_brim_type', document.getElementById('prusaBrimType').value);
        formData.append('prusa_brim_separation', document.getElementById('prusaBrimSeparation').value);
        formData.append('prusa_brim_one_stroke', document.getElementById('prusaBrimOneStroke').checked ? 'true' : 'false');
        formData.append('prusa_perimeter_generator', document.getElementById('prusaPerimeterGenerator').value);
        formData.append('prusa_gap_fill_enabled', document.getElementById('prusaGapFillEnabled').checked ? 'true' : 'false');
        formData.append('prusa_infill_anchor', document.getElementById('prusaInfillAnchor').value);
        formData.append('prusa_infill_anchor_max', document.getElementById('prusaInfillAnchorMax').value);
        formData.append('prusa_external_perimeter_width', document.getElementById('prusaExternalPerimeterWidth').value);
        formData.append('prusa_perimeter_width', document.getElementById('prusaPerimeterWidth').value);
        formData.append('prusa_infill_width', document.getElementById('prusaInfillWidth').value);
        formData.append('prusa_xy_size_compensation', document.getElementById('prusaXySizeCompensation').value);
        formData.append('prusa_elephant_foot_compensation', document.getElementById('prusaElephantFootCompensation').value);
        formData.append('prusa_avoid_crossing_max_detour', document.getElementById('prusaAvoidCrossingMaxDetour').value);
        formData.append('prusa_seam_position', document.getElementById('prusaSeamPosition').value);
      }} else if (slicingKernelInput.value === 'legacy') {{
        formData.append('planning_line_width', document.getElementById('planningLineWidth').value);
        formData.append('perimeter_count', document.getElementById('perimeterCount').value);
        formData.append('print_perimeters', document.getElementById('printPerimeters').checked ? 'true' : 'false');
        formData.append('infill_pattern', document.getElementById('infillPattern').value);
        formData.append('infill_density', document.getElementById('infillDensity').value);
        formData.append('infill_overlap', document.getElementById('infillOverlap').value);
        formData.append('contour_infill_overlap', document.getElementById('contourInfillOverlap').value);
        formData.append('triangle_path_optimization', trianglePathOptimizationInput.checked ? 'true' : 'false');
        formData.append('zigzag_path_optimization', zigzagPathOptimizationInput.checked ? 'true' : 'false');
        formData.append('print_raft', printRaftInput.checked ? 'true' : 'false');
        formData.append('raft_offsets', raftBottomOffsetInput.value + ',' + raftSecondOffsetInput.value);
        formData.append('curve_mode', document.getElementById('curveMode').value);
        formData.append('curve_amplitude', document.getElementById('curveAmplitude').value);
        formData.append('curve_period', document.getElementById('curvePeriod').value);
      }} else {{
        formData.append('pyslm_infill_pattern', document.getElementById('pyslmInfillPattern').value);
        formData.append('pyslm_infill_density', document.getElementById('pyslmInfillDensity').value);
        formData.append('pyslm_perimeter_count', document.getElementById('pyslmPerimeterCount').value);
        formData.append('pyslm_print_perimeters', document.getElementById('pyslmPrintPerimeters').checked ? 'true' : 'false');
        formData.append('pyslm_infill_overlap', document.getElementById('pyslmInfillOverlap').value);
        formData.append('pyslm_contour_infill_overlap', document.getElementById('pyslmContourInfillOverlap').value);
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
      }}

      try {{
        const response = await fetch('/slice', {{
          method: 'POST',
          body: formData
        }});
        const queued = await response.json();
        if (!response.ok || !queued.ok) throw new Error(queued.error || 'export job submission failed');
        const result = await waitForSliceJob(queued.job_id);

        layersEl.textContent = result.layers;
        outputNameEl.textContent = result.filename;
        const patternExecution = result.infill_pattern_execution;
        if (patternExecution?.applied) {{
          const angles = (patternExecution.angle_schedule_degrees || []).join('° / ');
          executedInfillPatternEl.textContent = '安全分层单向之字形'
            + (angles ? '（' + angles + '°）' : '');
        }} else {{
          executedInfillPatternEl.textContent = String(result.effective_infill_pattern || '-');
        }}
        previewData = result.preview;
        updatePreviewLineWidthValue();
        configureViewer();
        drawPreview();
        downloadEl.href = result.download_url;
        downloadEl.textContent = '下载 ' + result.filename;
        downloadEl.className = 'download visible';
        const previewLabel = result.preview?.preview_source === 'pre_core_source_npz'
          ? '完成（预览：送入 Core 前的源 NPZ；travel 保留全部避障点，连续段仅在首尾速度为 0）。'
          : result.preview?.preview_source === 'final_core_npz'
            ? '完成（预览：最终 Core NPZ）。'
            : '完成。';
        statusEl.textContent = result.recommendation?.message
          ? previewLabel + result.recommendation.message
          : previewLabel;
        statusEl.className = 'status ok';
        const coreSeconds = Number(result.core_export_seconds);
        if (Number.isFinite(coreSeconds)) {{
          exportElapsedEl.textContent = 'core 最终 NPZ 处理耗时 ' + coreSeconds.toFixed(1) + ' 秒';
        }}
      }} catch (error) {{
        statusEl.textContent = error.message;
        statusEl.className = 'status error';
      }} finally {{
        button.disabled = false;
        button.textContent = '生成并导出 Core NPZ';
      }}
    }});

    function configureViewer() {{
      const layers = previewData?.layers || [];
      stopPathPlayback();
      layerSlider.disabled = layers.length === 0;
      layerSlider.max = Math.max(0, layers.length - 1);
      layerSlider.value = 0;
      resetPreviewView();
      const isSurface = isSurfacePreview();
      showDirectionLabel.textContent = isSurface ? '显示路径方向/打印头' : '显示打印方向';
      previewCanvas.title = isSurface
        ? '滚轮缩放；左键旋转；右键或中键平移；双击复位'
        : '滚轮缩放；鼠标左键、右键或中键拖动视图；双击复位';
      updatePrintSizeLabel();
      updatePathPlaybackRateLabel();
      updatePathSlider(true);
    }}

    function resetPreviewView() {{
      viewerState.zoom = 1.0;
      viewerState.centerX = null;
      viewerState.centerY = null;
      viewerState.centerZ = null;
      viewerState.surfaceYaw = -0.72;
      viewerState.surfacePitch = -0.58;
      viewerState.surfacePanX = 0;
      viewerState.surfacePanY = 0;
    }}

    function currentLayer() {{
      const layers = previewData?.layers || [];
      return layers[Number(layerSlider.value)] || null;
    }}

    function isSurfacePreview() {{
      return previewData?.geometry_mode === 'surface_3d';
    }}

    function roleIsSelected(role) {{
      if (role === 'outer_contour') return showOuterContourInput.checked;
      if (role === 'inner_contour') return showInnerContourInput.checked;
      if (role === 'raft') return showRaftPathsInput.checked;
      if (role === 'fiber') return showFiberPathsInput.checked;
      if (role === 'travel') return showTravelPathsInput.checked;
      if (role === 'core_travel' || role === 'layer_lift') return showCoreTravelPathsInput.checked;
      if (role === 'primeline') return showPrimelineInput.checked;
      return showResinInfillInput.checked;
    }}

    function selectedPrintEntries(layer) {{
      if (!layer) return [];
      const entries = [];
      const motionEntries = Array.isArray(layer.motion_paths)
        ? layer.motion_paths
        : [
          ...(layer.resin_paths || (layer.paths || []).map((points) => ({{ role: 'infill', points }})))
            .map((entry) => ({{ ...entry, kind: 'deposit' }})),
          ...(layer.travel_paths || []).map((points) => ({{ kind: 'travel', points }}))
        ];
      for (const rawEntry of motionEntries) {{
        const kind = rawEntry.kind === 'travel' ? 'travel' : 'deposit';
        const role = kind === 'travel' ? 'travel' : (rawEntry.role || 'infill');
        const points = rawEntry.points || rawEntry;
        if (points && points.length >= 1) {{
          entries.push({{ kind, role, points, extrusion: rawEntry.extrusion || null }});
        }}
      }}
      const hasOrderedFiber = motionEntries.some((entry) =>
        entry?.kind !== 'travel' && entry?.role === 'fiber'
      );
      if (!hasOrderedFiber) {{
        for (const points of layer.fiber_paths || []) {{
          if (points && points.length >= 1) entries.push({{ kind: 'deposit', role: 'fiber', points }});
        }}
      }}
      const additions = (previewData?.core_overlay?.sequence || [])
        .filter((entry) => Number(entry.layer) === Number(layer.index))
        // A layer lift is a Z-only move.  In this top-view preview its two
        // endpoints have the same XY and therefore collapse to a misleading
        // one-point "path".  Keep it in the Core NPZ/overlay metadata, but do
        // not put it into the selectable 2D path sequence.
        .filter((entry) => entry.role !== 'layer_lift')
        .sort((left, right) => Number(left.order) - Number(right.order));
      if (!additions.length) {{
        return entries.filter((entry) => roleIsSelected(entry.role));
      }}
      const orderedEntries = [];
      let baseIndex = 0;
      for (const addition of additions) {{
        const anchor = Math.max(
          0,
          Math.min(entries.length, Number.isFinite(Number(addition.anchor))
            ? Number(addition.anchor)
            : entries.length)
        );
        while (baseIndex < anchor && baseIndex < entries.length) {{
          orderedEntries.push(entries[baseIndex++]);
        }}
        const points = addition.points;
        if (points && points.length >= 1) {{
          orderedEntries.push({{
            kind: addition.kind === 'travel' ? 'travel' : 'deposit',
            role: addition.role || 'travel',
            points,
            extrusion: null,
          }});
        }}
      }}
      while (baseIndex < entries.length) orderedEntries.push(entries[baseIndex++]);
      return orderedEntries.filter((entry) => roleIsSelected(entry.role));
    }}

    function historicalOverlayEntries() {{
      const layers = previewData?.layers || [];
      const currentLayerPosition = Number(layerSlider.value);
      const visibilityKey = [
        showOuterContourInput,
        showInnerContourInput,
        showResinInfillInput,
        showRaftPathsInput,
        showFiberPathsInput,
        showTravelPathsInput,
        showCoreTravelPathsInput,
        showPrimelineInput,
      ].map((input) => input.checked ? '1' : '0').join('');
      const key = `${{currentLayerPosition}}:${{visibilityKey}}`;
      if (historicalOverlayCache.preview === previewData && historicalOverlayCache.key === key) {{
        return historicalOverlayCache.entries;
      }}
      const entries = [];
      for (let layerPosition = 0; layerPosition < currentLayerPosition; layerPosition++) {{
        entries.push(...selectedPrintEntries(layers[layerPosition]));
      }}
      historicalOverlayCache = {{ preview: previewData, key, entries }};
      return entries;
    }}

    function currentPlaybackEntry() {{
      const entries = selectedPrintEntries(currentLayer());
      const visibleCount = Math.min(Number(pathProgressSlider.value), entries.length);
      for (let index = visibleCount - 1; index >= 0; index--) {{
        const entry = entries[index];
        if (entry.kind === 'deposit' && entry.points?.length >= 2) return entry;
      }}
      return null;
    }}

    function buildPathPlaybackTimeline(path, extrusion) {{
      const cumulative = [0];
      for (let index = 1; index < path.length; index++) {{
        const previous = path[index - 1];
        const point = path[index];
        cumulative.push(cumulative[index - 1] + Math.hypot(
          Number(point[0]) - Number(previous[0]),
          Number(point[1]) - Number(previous[1]),
          Number(point[2]) - Number(previous[2]),
        ));
      }}
      const eProfile = Array.isArray(extrusion) && extrusion.length === path.length
        ? extrusion.map(Number)
        : null;
      return {{ path, eProfile, cumulative, totalMm: cumulative[cumulative.length - 1] }};
    }}

    function stopPathPlayback({{ keepArrow = false }} = {{}}) {{
      if (pathPlayback.frame !== null) cancelAnimationFrame(pathPlayback.frame);
      pathPlayback.frame = null;
      pathPlayback.running = false;
      pathPlayback.previousTimestamp = null;
      if (!keepArrow) {{
        pathPlayback.entry = null;
        pathPlayback.timeline = null;
        pathPlayback.distanceMm = 0;
      }}
      playCurrentPathButton.textContent = '播放当前路径';
      playCurrentPathButton.setAttribute('aria-pressed', 'false');
    }}

    function updatePathPlaybackControl() {{
      const entry = currentPlaybackEntry();
      const selectedEntryChanged = pathPlayback.entry && (
        !entry
        || pathPlayback.entry.points !== entry.points
        || pathPlayback.entry.role !== entry.role
      );
      if (selectedEntryChanged) stopPathPlayback();
      pathPlaybackControl.hidden = !entry;
      playCurrentPathButton.disabled = !entry;
      pathPlaybackRateInput.disabled = !entry;
      if (!entry) stopPathPlayback();
    }}

    function updatePathPlaybackRateLabel() {{
      pathPlaybackRateLabel.textContent = Number(pathPlaybackRateInput.value).toFixed(2);
    }}

    function updatePathSlider(resetToEnd = false) {{
      const pathCount = selectedPrintEntries(currentLayer()).length;
      const previousValue = Number(pathProgressSlider.value);
      const previousMax = Number(pathProgressSlider.max);
      const wasAtEnd = previousValue >= previousMax;
      pathProgressSlider.disabled = pathCount === 0;
      pathProgressControl.hidden = pathCount === 0;
      pathProgressSlider.max = pathCount;
      pathProgressSlider.value = resetToEnd || wasAtEnd
        ? pathCount
        : Math.min(previousValue, pathCount);
      updatePathPlaybackControl();
      updateViewerLabels();
    }}

    function updateViewerLabels() {{
      const layer = currentLayer();
      const layerCount = previewData?.layers?.length || 0;
      const pathCount = selectedPrintEntries(layer).length;
      layerLabel.textContent = layer ? `${{Number(layerSlider.value) + 1}} / ${{layerCount}}` : '-';
      pathProgressLabel.textContent = pathCount
        ? `${{pathProgressSlider.value}} / ${{pathCount}}`
        : '-';
      updatePathPlaybackControl();
    }}

    function updatePrintSizeLabel() {{
      const bounds = previewData?.bounds;
      if (!bounds || [bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y]
        .some((value) => value === null || value === undefined)) {{
        printSizeLabel.textContent = '打印范围 -';
        return;
      }}
      const width = Math.max(0, Number(bounds.max_x) - Number(bounds.min_x));
      const height = Math.max(0, Number(bounds.max_y) - Number(bounds.min_y));
      printSizeLabel.textContent = `打印范围 X ${{formatDimension(width)}} mm × Y ${{formatDimension(height)}} mm`;
    }}

    function formatDimension(value) {{
      const decimals = value >= 100 ? 1 : 2;
      return Number(value.toFixed(decimals)).toString();
    }}

    function buildViewport(rect, bounds) {{
      const plot = {{
        left: 56,
        top: 18,
        right: Math.max(80, rect.width - 18),
        bottom: Math.max(80, rect.height - 38)
      }};
      plot.width = Math.max(1, plot.right - plot.left);
      plot.height = Math.max(1, plot.bottom - plot.top);
      const spanX = Math.max(0.001, Number(bounds.max_x) - Number(bounds.min_x));
      const spanY = Math.max(0.001, Number(bounds.max_y) - Number(bounds.min_y));
      const baseScale = Math.max(
        1e-6,
        Math.min(
          Math.max(1, plot.width - 24) / spanX,
          Math.max(1, plot.height - 24) / spanY
        )
      );
      if (viewerState.centerX === null || viewerState.centerY === null) {{
        viewerState.centerX = (Number(bounds.min_x) + Number(bounds.max_x)) * 0.5;
        viewerState.centerY = (Number(bounds.min_y) + Number(bounds.max_y)) * 0.5;
      }}
      const pixelsPerMm = baseScale * viewerState.zoom;
      const plotCenterX = (plot.left + plot.right) * 0.5;
      const plotCenterY = (plot.top + plot.bottom) * 0.5;
      const project = (point) => [
        plotCenterX + (Number(point[0]) - viewerState.centerX) * pixelsPerMm,
        plotCenterY - (Number(point[1]) - viewerState.centerY) * pixelsPerMm
      ];
      const unproject = (x, y) => [
        viewerState.centerX + (x - plotCenterX) / pixelsPerMm,
        viewerState.centerY - (y - plotCenterY) / pixelsPerMm
      ];
      return {{ plot, baseScale, pixelsPerMm, plotCenterX, plotCenterY, project, unproject }};
    }}

    function buildSurfaceViewport(rect, bounds) {{
      const plot = {{
        left: 40,
        top: 18,
        right: Math.max(80, rect.width - 18),
        bottom: Math.max(80, rect.height - 28)
      }};
      plot.width = Math.max(1, plot.right - plot.left);
      plot.height = Math.max(1, plot.bottom - plot.top);
      const minimum = [Number(bounds.min_x), Number(bounds.min_y), Number(bounds.min_z)];
      const maximum = [Number(bounds.max_x), Number(bounds.max_y), Number(bounds.max_z)];
      if (viewerState.centerX === null || viewerState.centerY === null || viewerState.centerZ === null) {{
        viewerState.centerX = (minimum[0] + maximum[0]) * 0.5;
        viewerState.centerY = (minimum[1] + maximum[1]) * 0.5;
        viewerState.centerZ = (minimum[2] + maximum[2]) * 0.5;
      }}
      const yaw = viewerState.surfaceYaw;
      const pitch = viewerState.surfacePitch;
      const cosYaw = Math.cos(yaw);
      const sinYaw = Math.sin(yaw);
      const cosPitch = Math.cos(pitch);
      const sinPitch = Math.sin(pitch);
      const rotate = (point) => {{
        const x = Number(point[0]) - viewerState.centerX;
        const y = Number(point[1]) - viewerState.centerY;
        const z = Number(point[2]) - viewerState.centerZ;
        const yawX = cosYaw * x - sinYaw * y;
        const yawY = sinYaw * x + cosYaw * y;
        return [
          yawX,
          cosPitch * yawY - sinPitch * z,
          sinPitch * yawY + cosPitch * z,
        ];
      }};
      // Keep the camera distance independent of yaw and pitch.  Fitting the
      // rotated bounding box here made the preview appear to zoom in or out
      // whenever the user turned the model.  The unrotated 3D bounding-sphere
      // diameter safely contains every orientation while leaving zoom solely
      // under explicit user control.
      const partSpan = Math.max(
        0.001,
        Math.hypot(
          maximum[0] - minimum[0],
          maximum[1] - minimum[1],
          maximum[2] - minimum[2]
        )
      );
      const headBoxSize = printHeadAsset.data?.model_bounds?.size_mm || [0, 0, 0];
      const headReach = Math.max(0, ...headBoxSize.map(Number));
      const modelSpan = partSpan + headReach;
      const baseScale = Math.max(
        1e-6,
        Math.min(
          Math.max(1, plot.width - 36) / modelSpan,
          Math.max(1, plot.height - 36) / modelSpan
        )
      );
      const pixelsPerMm = baseScale * viewerState.zoom;
      const plotCenterX = (plot.left + plot.right) * 0.5;
      const plotCenterY = (plot.top + plot.bottom) * 0.5;
      const project = (point) => {{
        const rotated = rotate(point);
        return [
          plotCenterX + rotated[0] * pixelsPerMm + viewerState.surfacePanX,
          plotCenterY - rotated[1] * pixelsPerMm + viewerState.surfacePanY,
          rotated[2],
        ];
      }};
      return {{ plot, baseScale, pixelsPerMm, plotCenterX, plotCenterY, project, minimum, maximum, isSurface: true }};
    }}

    function drawSurfaceReference(ctx, viewport) {{
      const {{ plot, minimum, maximum, project }} = viewport;
      const z = minimum[2];
      const corners = [
        [minimum[0], minimum[1], z],
        [maximum[0], minimum[1], z],
        [maximum[0], maximum[1], z],
        [minimum[0], maximum[1], z],
      ].map(project);
      ctx.fillStyle = 'rgba(232, 239, 244, 0.64)';
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let index = 1; index < corners.length; index++) ctx.lineTo(corners[index][0], corners[index][1]);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = '#d5dbe0';
      ctx.lineWidth = 1;
      ctx.stroke();
      const axisOrigin = [minimum[0], minimum[1], z];
      const axisLength = Math.max(8, Math.min(30, Math.max(
        maximum[0] - minimum[0],
        maximum[1] - minimum[1],
        maximum[2] - minimum[2]
      ) * 0.18));
      const axes = [
        {{ end: [axisOrigin[0] + axisLength, axisOrigin[1], axisOrigin[2]], color: '#b91c1c', label: 'X' }},
        {{ end: [axisOrigin[0], axisOrigin[1] + axisLength, axisOrigin[2]], color: '#0f766e', label: 'Y' }},
        {{ end: [axisOrigin[0], axisOrigin[1], axisOrigin[2] + axisLength], color: '#1d4ed8', label: 'Z' }},
      ];
      const start = project(axisOrigin);
      ctx.font = '600 11px Segoe UI, Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (const axis of axes) {{
        const end = project(axis.end);
        ctx.strokeStyle = axis.color;
        ctx.fillStyle = axis.color;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(start[0], start[1]);
        ctx.lineTo(end[0], end[1]);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(end[0], end[1], 2.8, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(axis.label, end[0] + 9, end[1] - 6);
      }}
      ctx.strokeStyle = '#aeb8c0';
      ctx.lineWidth = 1;
      ctx.strokeRect(plot.left + 0.5, plot.top + 0.5, plot.width - 1, plot.height - 1);
      ctx.fillStyle = '#5c6972';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'top';
      ctx.fillText('三维曲面预览 · 左键旋转', plot.right - 7, plot.top + 7);
    }}

    function surfaceLayerCurvatureText(layer) {{
      // This is the geometric curvature of the deposited 3D centerlines, not
      // a fit or a remapping result.  It intentionally reads the raw preview
      // layer, so visibility and path-progress controls cannot change it.
      if (layer && surfaceCurvatureTextCache.has(layer)) {{
        return surfaceCurvatureTextCache.get(layer);
      }}
      const paths = [
        ...(layer?.resin_paths || []).map((entry) => entry.points),
        ...(layer?.fiber_paths || []),
      ].filter((path) => Array.isArray(path) && path.length >= 3);
      let curvatureCount = 0;
      let curvatureTotal = 0;
      let minimumCurvature = Infinity;
      let maximumCurvature = -Infinity;
      for (const path of paths) {{
        for (let index = 1; index < path.length - 1; index++) {{
          const a = path[index - 1];
          const b = path[index];
          const c = path[index + 1];
          const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
          const bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
          const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
          const abLength = Math.hypot(...ab);
          const bcLength = Math.hypot(...bc);
          const acLength = Math.hypot(...ac);
          if (abLength <= 1e-9 || bcLength <= 1e-9 || acLength <= 1e-9) continue;
          const crossLength = Math.hypot(
            ab[1] * bc[2] - ab[2] * bc[1],
            ab[2] * bc[0] - ab[0] * bc[2],
            ab[0] * bc[1] - ab[1] * bc[0],
          );
          const curvature = 2 * crossLength / (abLength * bcLength * acLength);
          if (Number.isFinite(curvature)) {{
            curvatureCount += 1;
            curvatureTotal += curvature;
            minimumCurvature = Math.min(minimumCurvature, curvature);
            maximumCurvature = Math.max(maximumCurvature, curvature);
          }}
        }}
      }}
      const displayLayer = Number(layerSlider.value) + 1;
      const text = curvatureCount
        ? `第${{displayLayer}}层 · κ min/avg/max: ${{minimumCurvature.toFixed(4)}} / ${{(curvatureTotal / curvatureCount).toFixed(4)}} / ${{maximumCurvature.toFixed(4)}} mm⁻¹`
        : `第${{displayLayer}}层 · κ = 0.0000 mm⁻¹（直线段/采样不足）`;
      if (layer) surfaceCurvatureTextCache.set(layer, text);
      return text;
    }}

    function drawSurfaceLayerCurvature(ctx, viewport, layer) {{
      const text = surfaceLayerCurvatureText(layer);
      const {{ plot }} = viewport;
      ctx.save();
      ctx.font = '600 11px Segoe UI, Arial, sans-serif';
      const x = plot.left + 8;
      const y = plot.top + 8;
      const width = Math.min(plot.width - 16, ctx.measureText(text).width + 16);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.88)';
      ctx.fillRect(x - 5, y - 4, width, 21);
      ctx.strokeStyle = 'rgba(174, 184, 192, 0.88)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x - 4.5, y - 3.5, width - 1, 20);
      ctx.fillStyle = '#31414d';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x, y + 6);
      ctx.restore();
    }}

    function niceGridStep(pixelsPerMm) {{
      const targetWorldStep = 72 / Math.max(pixelsPerMm, 1e-9);
      const exponent = Math.floor(Math.log10(targetWorldStep));
      const magnitude = 10 ** exponent;
      const normalized = targetWorldStep / magnitude;
      const factor = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
      return factor * magnitude;
    }}

    function forEachGridValue(minimum, maximum, step, callback) {{
      const first = Math.ceil((minimum - step * 1e-8) / step);
      const last = Math.floor((maximum + step * 1e-8) / step);
      for (let index = first; index <= last && index - first < 1200; index++) {{
        callback(index * step);
      }}
    }}

    function formatRulerValue(value, step) {{
      const decimals = step < 0.1 ? 2 : step < 1 ? 1 : 0;
      const normalized = Math.abs(value) < step * 1e-6 ? 0 : value;
      return normalized.toFixed(decimals);
    }}

    function drawMeasurementGrid(ctx, viewport, rect) {{
      const {{ plot, project, unproject, pixelsPerMm }} = viewport;
      const topLeft = unproject(plot.left, plot.top);
      const bottomRight = unproject(plot.right, plot.bottom);
      const minVisibleX = Math.min(topLeft[0], bottomRight[0]);
      const maxVisibleX = Math.max(topLeft[0], bottomRight[0]);
      const minVisibleY = Math.min(topLeft[1], bottomRight[1]);
      const maxVisibleY = Math.max(topLeft[1], bottomRight[1]);
      const majorStep = niceGridStep(pixelsPerMm);
      const minorStep = majorStep / 5;

      ctx.fillStyle = '#ffffff';
      ctx.fillRect(plot.left, plot.top, plot.width, plot.height);
      ctx.save();
      ctx.beginPath();
      ctx.rect(plot.left, plot.top, plot.width, plot.height);
      ctx.clip();

      ctx.strokeStyle = '#edf0f2';
      ctx.lineWidth = 1;
      forEachGridValue(minVisibleX, maxVisibleX, minorStep, (value) => {{
        const x = project([value, 0])[0];
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.bottom);
        ctx.stroke();
      }});
      forEachGridValue(minVisibleY, maxVisibleY, minorStep, (value) => {{
        const y = project([0, value])[1];
        ctx.beginPath();
        ctx.moveTo(plot.left, y);
        ctx.lineTo(plot.right, y);
        ctx.stroke();
      }});

      ctx.strokeStyle = '#d5dbe0';
      forEachGridValue(minVisibleX, maxVisibleX, majorStep, (value) => {{
        const x = project([value, 0])[0];
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.bottom);
        ctx.stroke();
      }});
      forEachGridValue(minVisibleY, maxVisibleY, majorStep, (value) => {{
        const y = project([0, value])[1];
        ctx.beginPath();
        ctx.moveTo(plot.left, y);
        ctx.lineTo(plot.right, y);
        ctx.stroke();
      }});
      ctx.restore();

      ctx.strokeStyle = '#aeb8c0';
      ctx.lineWidth = 1;
      ctx.strokeRect(plot.left + 0.5, plot.top + 0.5, plot.width - 1, plot.height - 1);
      ctx.fillStyle = '#5c6972';
      ctx.font = '11px Segoe UI, Arial, sans-serif';
      ctx.textBaseline = 'middle';
      ctx.textAlign = 'center';
      forEachGridValue(minVisibleX, maxVisibleX, majorStep, (value) => {{
        const x = project([value, 0])[0];
        if (x > plot.left + 16 && x < plot.right - 58) {{
          ctx.fillText(formatRulerValue(value, majorStep), x, plot.bottom + 18);
        }}
      }});
      ctx.textAlign = 'right';
      forEachGridValue(minVisibleY, maxVisibleY, majorStep, (value) => {{
        const y = project([0, value])[1];
        if (y > plot.top + 8 && y < plot.bottom - 8) {{
          ctx.fillText(formatRulerValue(value, majorStep), plot.left - 8, y);
        }}
      }});
      ctx.fillStyle = '#172026';
      ctx.font = '600 11px Segoe UI, Arial, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText('X (mm)', plot.right, plot.bottom + 18);
      ctx.save();
      ctx.translate(12, (plot.top + plot.bottom) * 0.5);
      ctx.rotate(-Math.PI * 0.5);
      ctx.textAlign = 'center';
      ctx.fillText('Y (mm)', 0, 0);
      ctx.restore();
    }}

    function pathColor(role) {{
      if (role === 'outer_contour') return '#146c43';
      if (role === 'inner_contour') return '#7b2cbf';
      if (role === 'raft') return '#7f5539';
      if (role === 'fiber') return '#e66f00';
      if (role === 'travel') return '#526f8c';
      if (role === 'primeline') return '#b91c1c';
      return '#0b6bcb';
    }}

    function coreOverlayPaths(key, layer) {{
      const layerIndex = Number(layer?.index);
      return (previewData?.core_overlay?.[key] || [])
        .filter((entry) => Number(entry.layer) === layerIndex)
        .map((entry) => entry.points)
        .filter((points) => Array.isArray(points) && points.length >= 2);
    }}

    function drawOverlayPath(ctx, path, project) {{
      const first = project(path[0]);
      ctx.beginPath();
      ctx.moveTo(first[0], first[1]);
      for (let index = 1; index < path.length; index++) {{
        const point = project(path[index]);
        ctx.lineTo(point[0], point[1]);
      }}
      ctx.stroke();
    }}

    function drawOriginMarker(ctx, viewport) {{
      const origin = previewData?.origin || [0, 0];
      const point = viewport.project(origin);
      const {{ plot }} = viewport;
      if (
        point[0] < plot.left - 10 || point[0] > plot.right + 10
        || point[1] < plot.top - 10 || point[1] > plot.bottom + 10
      ) return;
      ctx.save();
      ctx.strokeStyle = '#b91c1c';
      ctx.fillStyle = '#b91c1c';
      ctx.lineWidth = 2;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(point[0] - 7, point[1]);
      ctx.lineTo(point[0] + 7, point[1]);
      ctx.moveTo(point[0], point[1] - 7);
      ctx.lineTo(point[0], point[1] + 7);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(point[0], point[1], 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = '600 11px Segoe UI, Arial, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText('(0, 0)', point[0] + 9, point[1] - 8);
      ctx.restore();
    }}

    function extrusionDensity(path, extrusion, segmentIndex) {{
      if (!Array.isArray(extrusion) || extrusion.length !== path.length) return null;
      const deltaE = Number(extrusion[segmentIndex + 1]) - Number(extrusion[segmentIndex]);
      const dx = Number(path[segmentIndex + 1][0]) - Number(path[segmentIndex][0]);
      const dy = Number(path[segmentIndex + 1][1]) - Number(path[segmentIndex][1]);
      const dz = Number(path[segmentIndex + 1][2]) - Number(path[segmentIndex][2]);
      const distance = Math.hypot(dx, dy, dz);
      if (!Number.isFinite(deltaE) || !Number.isFinite(distance) || distance <= 1e-9) return null;
      return Math.max(0, deltaE) / distance;
    }}

    // Use one physical E/mm domain for every layer and path.  A percentile
    // range derived from the current layer makes a tiny difference between
    // Brim/contour/infill look like a large flow change.
    const ABSOLUTE_EXTRUSION_COLOR_RANGE = Object.freeze({{ low: 0.0, high: 0.5 }});

    function extrusionDensityRange(entries) {{
      for (const entry of entries) {{
        for (let index = 0; index < entry.points.length - 1; index++) {{
          const density = extrusionDensity(entry.points, entry.extrusion, index);
          if (density !== null && density >= 0) return ABSOLUTE_EXTRUSION_COLOR_RANGE;
        }}
      }}
      return null;
    }}

    function extrusionColorForSegment(density, range) {{
      const t = Math.max(0, Math.min(1, (density - range.low) / (range.high - range.low)));
      const stops = [
        [30, 64, 175],
        [15, 150, 160],
        [245, 158, 11],
        [220, 38, 38]
      ];
      const scaled = t * (stops.length - 1);
      const lower = Math.min(stops.length - 2, Math.floor(scaled));
      const fraction = scaled - lower;
      const start = stops[lower];
      const end = stops[lower + 1];
      const channel = (index) => Math.round(start[index] + (end[index] - start[index]) * fraction);
      return `rgb(${{channel(0)}}, ${{channel(1)}}, ${{channel(2)}})`;
    }}

    function drawPreviewNow() {{
      const canvas = previewCanvas;
      const rect = canvas.getBoundingClientRect();
      const deviceScale = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.floor(rect.width * deviceScale));
      canvas.height = Math.max(1, Math.floor(rect.height * deviceScale));
      const ctx = canvas.getContext('2d');
      ctx.setTransform(deviceScale, 0, 0, deviceScale, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);

      const layer = currentLayer();
      const bounds = previewData?.bounds;
      if (!layer || !bounds || [bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y]
        .some((value) => value === null || value === undefined)) {{
        extrusionColorLegend.hidden = true;
        drawEmptyPreview(ctx, rect.width, rect.height);
        updateViewerLabels();
        return;
      }}

      const surfacePreview = isSurfacePreview();
      const viewport = surfacePreview
        ? buildSurfaceViewport(rect, bounds)
        : buildViewport(rect, bounds);
      if (surfacePreview) {{
        drawSurfaceReference(ctx, viewport);
      }} else {{
        drawMeasurementGrid(ctx, viewport, rect);
      }}
      const entries = selectedPrintEntries(layer);
      const visibleCount = Math.min(Number(pathProgressSlider.value), entries.length);
      const currentEntry = visibleCount > 0 ? entries[visibleCount - 1] : null;
      const lineWidths = previewData.line_widths || {{ resin: 2.0, fiber: 1.0 }};
      const usePhysicalWidth = showLineWidthInput.checked;
    // Keep the color scale stable while the path-progress slider reveals more
    // paths.  Deriving it from only the visible prefix would recolor every
    // previously drawn path whenever a new path changes the min/max range.
    const extrusionRange = showExtrusionInput.checked && visibleCount > 0
      ? extrusionDensityRange(entries)
      : null;
      extrusionColorLegend.hidden = extrusionRange === null;

      function drawPath(path) {{
        const first = viewport.project(path[0]);
        if (path.length === 1) {{
          ctx.beginPath();
          ctx.arc(first[0], first[1], Math.max(1.2, ctx.lineWidth * 0.5), 0, Math.PI * 2);
          ctx.fillStyle = ctx.strokeStyle;
          ctx.fill();
          return;
        }}
        ctx.beginPath();
        ctx.moveTo(first[0], first[1]);
        for (let pointIndex = 1; pointIndex < path.length; pointIndex++) {{
          const point = viewport.project(path[pointIndex]);
          ctx.lineTo(point[0], point[1]);
        }}
        const last = path[path.length - 1];
        if (path.length > 2
          && Math.abs(last[0] - path[0][0]) < 0.001
          && Math.abs(last[1] - path[0][1]) < 0.001) {{
          ctx.closePath();
        }}
        ctx.stroke();
      }}

      function drawExtrusionPath(path, extrusion, fallbackColor) {{
        for (let pointIndex = 0; pointIndex < path.length - 1; pointIndex++) {{
          const deltaE = Number(extrusion[pointIndex + 1]) - Number(extrusion[pointIndex]);
          const zeroExtrusion = Number.isFinite(deltaE) && Math.abs(deltaE) <= 1e-9;
          const density = extrusionDensity(path, extrusion, pointIndex);
          const activeLineWidth = ctx.lineWidth;
          const activeAlpha = ctx.globalAlpha;
          if (zeroExtrusion) {{
            // A continuous honeycomb motion has connector segments with
            // exactly constant E.  Draw them distinctly even when the
            // extrusion heat map is disabled, so they cannot be mistaken for
            // deposited walls in the preview.
            ctx.strokeStyle = '#526f8c';
            ctx.globalAlpha = activeAlpha * 0.95;
            ctx.lineWidth = Math.min(activeLineWidth, 1.5);
            ctx.setLineDash([7, 5]);
          }} else if (density === null || extrusionRange === null) {{
            ctx.strokeStyle = fallbackColor;
          }} else {{
            ctx.strokeStyle = extrusionColorForSegment(density, extrusionRange);
          }}
          const first = viewport.project(path[pointIndex]);
          const last = viewport.project(path[pointIndex + 1]);
          ctx.beginPath();
          ctx.moveTo(first[0], first[1]);
          ctx.lineTo(last[0], last[1]);
          ctx.stroke();
          if (zeroExtrusion) {{
            ctx.setLineDash([]);
            ctx.lineWidth = activeLineWidth;
            ctx.globalAlpha = activeAlpha;
          }}
        }}
      }}

      function drawEntry(entry, opacity = 1) {{
        ctx.save();
        ctx.globalAlpha = opacity;
        if (entry.kind === 'travel') {{
          ctx.strokeStyle = entry.role === 'core_travel'
            ? '#c2410c'
            : entry.role === 'layer_lift'
              ? '#f59e0b'
              : '#526f8c';
          ctx.globalAlpha = opacity * 0.95;
          ctx.lineWidth = entry.role === 'layer_lift' ? 2.2 : entry.role === 'core_travel' ? 2.0 : 1.5;
          ctx.setLineDash(
            entry.role === 'layer_lift' ? [2, 3]
              : entry.role === 'core_travel' ? [5, 4]
              : [7, 5]
          );
          drawPath(entry.points);
          ctx.restore();
          return;
        }}
        const physicalWidth = entry.role === 'fiber'
          ? Number(lineWidths.fiber || 1.0)
          : Number(lineWidths.resin || 2.0);
        ctx.lineWidth = usePhysicalWidth
          ? Math.max(1.0, physicalWidth * viewport.pixelsPerMm)
          : entry.role === 'fiber' ? 2.0 : 1.7;
        if (
          entry.role !== 'fiber'
          && Array.isArray(entry.extrusion)
          && entry.extrusion.length === entry.points.length
        ) {{
          drawExtrusionPath(entry.points, entry.extrusion, pathColor(entry.role));
        }} else {{
          ctx.strokeStyle = pathColor(entry.role);
          drawPath(entry.points);
        }}
        ctx.restore();
      }}

      function drawHistoricalOverlay(entries) {{
        // Historical layers share one opacity and are immutable until the
        // layer or role filters change.  Batch their Canvas strokes by style
        // to avoid a beginPath/stroke pair for every individual path.
        const batches = new Map();
        const historicalPathStride = (path) => viewerState.dragging
          ? Math.max(1, Math.ceil(path.length / 240))
          : 1;
        const addToBatch = (key, style, path = null, segment = null) => {{
          let batch = batches.get(key);
          if (!batch) {{
            batch = {{ ...style, paths: [], segments: [] }};
            batches.set(key, batch);
          }}
          if (path) batch.paths.push({{ points: path, stride: historicalPathStride(path) }});
          if (segment) batch.segments.push(segment);
        }};
        const depositionStyle = (role) => {{
          const physicalWidth = role === 'fiber'
            ? Number(lineWidths.fiber || 1.0)
            : Number(lineWidths.resin || 2.0);
          return {{
            color: pathColor(role),
            width: usePhysicalWidth
              ? Math.max(1.0, physicalWidth * viewport.pixelsPerMm)
              : role === 'fiber' ? 2.0 : 1.7,
            dash: [],
          }};
        }};
        for (const entry of entries) {{
          if (entry.kind === 'travel') {{
            const style = entry.role === 'core_travel'
              ? {{ color: '#c2410c', width: 2.0, dash: [5, 4], alpha: 0.95 }}
              : entry.role === 'layer_lift'
                ? {{ color: '#f59e0b', width: 2.2, dash: [2, 3], alpha: 0.95 }}
                : {{ color: '#526f8c', width: 1.5, dash: [7, 5], alpha: 0.95 }};
            addToBatch(`travel:${{entry.role}}`, style, entry.points);
            continue;
          }}
          const fallback = depositionStyle(entry.role);
          const extrusion = entry.extrusion;
          // Rotating a dense overlay is projection-bound, not data-bound.
          // During the gesture, sample only historical paths and use their
          // role color; the complete E-aware view is restored on release.
          if (viewerState.dragging) {{
            addToBatch(`deposit:${{entry.role}}`, fallback, entry.points);
            continue;
          }}
          if (!Array.isArray(extrusion) || extrusion.length !== entry.points.length || entry.role === 'fiber') {{
            addToBatch(`deposit:${{entry.role}}`, fallback, entry.points);
            continue;
          }}
          for (let pointIndex = 0; pointIndex < entry.points.length - 1; pointIndex++) {{
            const deltaE = Number(extrusion[pointIndex + 1]) - Number(extrusion[pointIndex]);
            const zeroExtrusion = Number.isFinite(deltaE) && Math.abs(deltaE) <= 1e-9;
            if (zeroExtrusion) {{
              const connectorStyle = {{
                color: '#526f8c',
                width: Math.min(fallback.width, 1.5),
                dash: [7, 5],
                alpha: 0.95,
              }};
              addToBatch('connector', connectorStyle, null, [entry.points[pointIndex], entry.points[pointIndex + 1]]);
              continue;
            }}
            const density = extrusionDensity(entry.points, extrusion, pointIndex);
            const color = showExtrusionInput.checked && density !== null && extrusionRange !== null
              ? extrusionColorForSegment(density, extrusionRange)
              : fallback.color;
            addToBatch(
              `deposit-segment:${{fallback.width}}:${{color}}`,
              {{ color, width: fallback.width, dash: [] }},
              null,
              [entry.points[pointIndex], entry.points[pointIndex + 1]],
            );
          }}
        }}
        ctx.save();
        for (const batch of batches.values()) {{
          ctx.globalAlpha = 0.32 * (batch.alpha ?? 1);
          ctx.strokeStyle = batch.color;
          ctx.lineWidth = batch.width;
          ctx.setLineDash(batch.dash);
          ctx.beginPath();
          for (const {{ points: path, stride }} of batch.paths) {{
            const first = viewport.project(path[0]);
            ctx.moveTo(first[0], first[1]);
            for (let pointIndex = stride; pointIndex < path.length - 1; pointIndex += stride) {{
              const point = viewport.project(path[pointIndex]);
              ctx.lineTo(point[0], point[1]);
            }}
            if (path.length > 1) {{
              const last = viewport.project(path[path.length - 1]);
              ctx.lineTo(last[0], last[1]);
            }}
          }}
          for (const [start, end] of batch.segments) {{
            const projectedStart = viewport.project(start);
            const projectedEnd = viewport.project(end);
            ctx.moveTo(projectedStart[0], projectedStart[1]);
            ctx.lineTo(projectedEnd[0], projectedEnd[1]);
          }}
          ctx.stroke();
        }}
        ctx.restore();
      }}

      ctx.save();
      ctx.beginPath();
      ctx.rect(
        viewport.plot.left,
        viewport.plot.top,
        viewport.plot.width,
        viewport.plot.height
      );
      ctx.clip();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      if (showLayerOverlayInput.checked) {{
        drawHistoricalOverlay(historicalOverlayEntries());
      }}
      for (let index = 0; index < visibleCount; index++) {{
        drawEntry(entries[index]);
      }}
      if (currentEntry && showPathPointsInput.checked) {{
        drawPathPoints(ctx, currentEntry.points, pathColor(currentEntry.role), viewport.project);
      }}
      if (surfacePreview && showDirectionInput.checked && !pathPlayback.running) {{
        const currentDeposit = entries
          .slice(0, visibleCount)
          .reverse()
          .find((entry) => entry.kind === 'deposit' && entry.points?.length);
        if (currentDeposit) {{
          drawPrintHeadModel(
            ctx,
            currentDeposit.points[currentDeposit.points.length - 1],
            viewport,
          );
        }}
      }} else if (!surfacePreview && currentEntry && showDirectionInput.checked) {{
        drawDirection(ctx, currentEntry.points, pathColor(currentEntry.role), viewport.project);
      }}
      if (pathPlayback.running && pathPlayback.timeline) {{
        drawPlaybackPathArrow(
          ctx,
          pathPlayback.timeline,
          pathPlayback.distanceMm,
          viewport,
        );
      }}
      ctx.restore();
      if (surfacePreview) drawSurfaceLayerCurvature(ctx, viewport, layer);
      if (!surfacePreview) drawOriginMarker(ctx, viewport);
      updateViewerLabels();
    }}

    function drawPreview() {{
      if (pendingPreviewFrame !== null) return;
      pendingPreviewFrame = requestAnimationFrame(() => {{
        pendingPreviewFrame = null;
        drawPreviewNow();
      }});
    }}

    function playCurrentPath(timestamp) {{
      if (!pathPlayback.running || !pathPlayback.timeline) return;
      if (pathPlayback.previousTimestamp === null) pathPlayback.previousTimestamp = timestamp;
      const elapsedSeconds = Math.min(0.1, (timestamp - pathPlayback.previousTimestamp) / 1000);
      pathPlayback.previousTimestamp = timestamp;
      const rate = Math.max(0, Number(pathPlaybackRateInput.value));
      pathPlayback.distanceMm = Math.min(
        pathPlayback.timeline.totalMm,
        pathPlayback.distanceMm + elapsedSeconds * PLAYBACK_MAX_SPEED_MM_PER_S * rate,
      );
      drawPreviewNow();
      if (pathPlayback.distanceMm >= pathPlayback.timeline.totalMm || rate <= 0) {{
        stopPathPlayback();
        drawPreview();
        return;
      }}
      pathPlayback.frame = requestAnimationFrame(playCurrentPath);
    }}

    function startPathPlayback() {{
      const entry = currentPlaybackEntry();
      if (!entry) return;
      if (pathPlayback.running) {{
        stopPathPlayback();
        drawPreview();
        return;
      }}
      pathPlayback.entry = entry;
      pathPlayback.timeline = buildPathPlaybackTimeline(entry.points, entry.extrusion);
      pathPlayback.distanceMm = 0;
      pathPlayback.previousTimestamp = null;
      if (pathPlayback.timeline.totalMm <= 1e-9 || Number(pathPlaybackRateInput.value) <= 0) {{
        drawPreview();
        return;
      }}
      pathPlayback.running = true;
      playCurrentPathButton.textContent = '停止播放';
      playCurrentPathButton.setAttribute('aria-pressed', 'true');
      pathPlayback.frame = requestAnimationFrame(playCurrentPath);
    }}

    function drawPathPoints(ctx, path, color, project) {{
      if (!path || path.length < 1) return;
      ctx.save();
      ctx.fillStyle = color;
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

    function pathPointAtDistance(timeline, distanceMm) {{
      const distance = Math.max(0, Math.min(distanceMm, timeline.totalMm));
      const cumulative = timeline.cumulative;
      let index = 1;
      while (index < cumulative.length && cumulative[index] < distance) index++;
      const endIndex = Math.min(index, timeline.path.length - 1);
      const startIndex = Math.max(0, endIndex - 1);
      const startDistance = cumulative[startIndex];
      const endDistance = cumulative[endIndex];
      const segmentLength = endDistance - startDistance;
      const t = segmentLength > 1e-9 ? (distance - startDistance) / segmentLength : 0;
      const start = timeline.path[startIndex];
      const end = timeline.path[endIndex];
      return {{
        point: start.map((value, axis) => Number(value) + (Number(end[axis]) - Number(value)) * t),
        previous: start,
        ahead: end,
        segmentIndex: startIndex,
        atEnd: distance >= timeline.totalMm,
      }};
    }}

    function drawPlaybackPathArrow(
      ctx,
      timeline,
      distanceMm,
      viewport,
    ) {{
      const {{ point, previous, ahead, segmentIndex, atEnd }} = pathPointAtDistance(timeline, distanceMm);
      if (viewport?.isSurface) {{
        drawPrintHeadModel(ctx, point, viewport);
        return;
      }}
      const current = viewport.project(point);
      const neighboringPoint = viewport.project(atEnd ? previous : ahead);
      const angle = atEnd
        ? Math.atan2(current[1] - neighboringPoint[1], current[0] - neighboringPoint[0])
        : Math.atan2(neighboringPoint[1] - current[1], neighboringPoint[0] - current[0]);
      const arrowLength = 22;
      const tip = [
        current[0] + Math.cos(angle) * arrowLength * 0.55,
        current[1] + Math.sin(angle) * arrowLength * 0.55,
      ];
      const tail = [
        current[0] - Math.cos(angle) * arrowLength * 0.45,
        current[1] - Math.sin(angle) * arrowLength * 0.45,
      ];
      const deltaE = timeline.eProfile
        ? Number(timeline.eProfile[segmentIndex + 1]) - Number(timeline.eProfile[segmentIndex])
        : Number.NaN;
      const color = Number.isFinite(deltaE) && deltaE <= 1e-9 ? '#2563eb' : '#dc2626';
      ctx.save();
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 4;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(tail[0], tail[1]);
      ctx.lineTo(tip[0], tip[1]);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(tip[0], tip[1]);
      ctx.lineTo(tip[0] - Math.cos(angle - 0.58) * 11, tip[1] - Math.sin(angle - 0.58) * 11);
      ctx.lineTo(tip[0] - Math.cos(angle + 0.58) * 11, tip[1] - Math.sin(angle + 0.58) * 11);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }}

    function normalizeVector(vector) {{
      const length = Math.hypot(vector[0], vector[1], vector[2]);
      return length > 1e-9 ? vector.map((value) => value / length) : [0, 0, -1];
    }}

    function crossVector(left, right) {{
      return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
      ];
    }}

    function kukaToolDirection(point) {{
      if (!Array.isArray(point) || point.length < 6) {{
        return {{ direction: [0, 0, -1], exact: false }};
      }}
      const [a, b, c] = point.slice(3, 6).map(Number);
      if (![a, b, c].every(Number.isFinite)) {{
        return {{ direction: [0, 0, -1], exact: false }};
      }}
      // Surface mapping defines ABC as Rz(A) * Ry(B) * Rx(C), relative to
      // the calibrated flat pose whose +X_TOOL work axis is -Z_BASE.
      const radians = Math.PI / 180;
      const cosA = Math.cos(a * radians);
      const sinA = Math.sin(a * radians);
      const cosB = Math.cos(b * radians);
      const sinB = Math.sin(b * radians);
      const cosC = Math.cos(c * radians);
      const sinC = Math.sin(c * radians);
      const afterX = [0, sinC, -cosC];
      const afterY = [
        cosB * afterX[0] + sinB * afterX[2],
        afterX[1],
        -sinB * afterX[0] + cosB * afterX[2],
      ];
      return {{
        direction: normalizeVector([
          cosA * afterY[0] - sinA * afterY[1],
          sinA * afterY[0] + cosA * afterY[1],
          afterY[2],
        ]),
        exact: true,
      }};
    }}

    function kukaToolFrame(point) {{
      const flatFrame = {{
        xAxis: [0, 0, -1],
        yAxis: [0, 1, 0],
        zAxis: [1, 0, 0],
        exact: false,
      }};
      if (!Array.isArray(point) || point.length < 6) return flatFrame;
      const [a, b, c] = point.slice(3, 6).map(Number);
      if (![a, b, c].every(Number.isFinite)) return flatFrame;
      const radians = Math.PI / 180;
      const cosA = Math.cos(a * radians);
      const sinA = Math.sin(a * radians);
      const cosB = Math.cos(b * radians);
      const sinB = Math.sin(b * radians);
      const cosC = Math.cos(c * radians);
      const sinC = Math.sin(c * radians);
      const rotate = (vector) => {{
        const afterX = [
          vector[0],
          cosC * vector[1] - sinC * vector[2],
          sinC * vector[1] + cosC * vector[2],
        ];
        const afterY = [
          cosB * afterX[0] + sinB * afterX[2],
          afterX[1],
          -sinB * afterX[0] + cosB * afterX[2],
        ];
        return normalizeVector([
          cosA * afterY[0] - sinA * afterY[1],
          sinA * afterY[0] + cosA * afterY[1],
          afterY[2],
        ]);
      }};
      return {{
        xAxis: rotate(flatFrame.xAxis),
        yAxis: rotate(flatFrame.yAxis),
        zAxis: rotate(flatFrame.zAxis),
        exact: true,
      }};
    }}

    function toolPointToBase(toolPoint, tcpPoint, frame) {{
      return [0, 1, 2].map((axis) => Number(tcpPoint[axis])
        + Number(toolPoint[0]) * frame.xAxis[axis]
        + Number(toolPoint[1]) * frame.yAxis[axis]
        + Number(toolPoint[2]) * frame.zAxis[axis]);
    }}

    function drawPrintHeadModel(ctx, point, viewport) {{
      const asset = printHeadAsset.data;
      if (!point || !viewport?.isSurface) return;
      if (!asset?.positions?.length || !asset?.triangles?.length) {{
        drawPrintHeadArrow(ctx, point, viewport);
        return;
      }}
      const frame = kukaToolFrame(point);
      const projected = asset.positions.map((toolPoint) =>
        viewport.project(toolPointToBase(toolPoint, point, frame))
      );
      const faces = [];
      for (let index = 0; index < asset.triangles.length; index++) {{
        const triangle = asset.triangles[index];
        const vertices = triangle.map((vertexIndex) => projected[vertexIndex]);
        const screenArea = (vertices[1][0] - vertices[0][0]) * (vertices[2][1] - vertices[0][1])
          - (vertices[1][1] - vertices[0][1]) * (vertices[2][0] - vertices[0][0]);
        if (Math.abs(screenArea) < 0.035) continue;
        faces.push({{
          vertices,
          depth: (vertices[0][2] + vertices[1][2] + vertices[2][2]) / 3,
          facing: screenArea,
        }});
      }}
      faces.sort((left, right) => left.depth - right.depth);
      ctx.save();
      ctx.lineJoin = 'round';
      for (const face of faces) {{
        const front = face.facing < 0;
        ctx.fillStyle = front ? 'rgba(91, 111, 124, 0.88)' : 'rgba(149, 163, 173, 0.62)';
        ctx.beginPath();
        ctx.moveTo(face.vertices[0][0], face.vertices[0][1]);
        ctx.lineTo(face.vertices[1][0], face.vertices[1][1]);
        ctx.lineTo(face.vertices[2][0], face.vertices[2][1]);
        ctx.closePath();
        ctx.fill();
      }}
      ctx.restore();
    }}

    function drawPrintHeadArrow(ctx, point, viewport) {{
      if (!point || !viewport?.isSurface) return;
      const {{ direction, exact }} = kukaToolDirection(point);
      const sceneSpan = Math.max(
        viewport.maximum[0] - viewport.minimum[0],
        viewport.maximum[1] - viewport.minimum[1],
        viewport.maximum[2] - viewport.minimum[2]
      );
      const length = Math.max(8, Math.min(28, sceneSpan * 0.16));
      const coneLength = length * 0.30;
      const shaftBase = point.slice(0, 3).map((value, index) => value - direction[index] * length);
      const coneBase = point.slice(0, 3).map((value, index) => value - direction[index] * coneLength);
      const side = normalizeVector(crossVector(
        direction,
        Math.abs(direction[2]) < 0.85 ? [0, 0, 1] : [0, 1, 0]
      ));
      const up = normalizeVector(crossVector(side, direction));
      const radius = Math.max(1.5, length * 0.12);
      const coneRing = Array.from({{ length: 6 }}, (_, index) => {{
        const angle = index * Math.PI * 2 / 6;
        return coneBase.map((value, axis) => value + radius * (
          side[axis] * Math.cos(angle) + up[axis] * Math.sin(angle)
        ));
      }});
      const faces = coneRing.map((vertex, index) => {{
        const next = coneRing[(index + 1) % coneRing.length];
        const projected = [viewport.project(point), viewport.project(vertex), viewport.project(next)];
        return {{ projected, depth: (projected[0][2] + projected[1][2] + projected[2][2]) / 3 }};
      }}).sort((left, right) => left.depth - right.depth);
      const projectedBase = viewport.project(shaftBase);
      const projectedConeBase = viewport.project(coneBase);
      const projectedTip = viewport.project(point);
      const color = exact ? '#0f4c81' : '#4b6475';
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.strokeStyle = color;
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(projectedBase[0], projectedBase[1]);
      ctx.lineTo(projectedConeBase[0], projectedConeBase[1]);
      ctx.stroke();
      for (const face of faces) {{
        ctx.fillStyle = exact ? 'rgba(15, 76, 129, 0.86)' : 'rgba(75, 100, 117, 0.82)';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(face.projected[0][0], face.projected[0][1]);
        ctx.lineTo(face.projected[1][0], face.projected[1][1]);
        ctx.lineTo(face.projected[2][0], face.projected[2][1]);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }}
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(projectedTip[0], projectedTip[1], 3.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.font = '600 11px Segoe UI, Arial, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillStyle = color;
      ctx.fillText(exact ? '打印头 +X_TOOL' : '打印头（默认朝下）', projectedBase[0] + 7, projectedBase[1] - 6);
      ctx.restore();
    }}

    function drawEmptyPreview(ctx, width, height) {{
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
    }}

    function currentViewportForInteraction() {{
      const bounds = previewData?.bounds;
      if (!currentLayer() || !bounds
        || [bounds.min_x, bounds.max_x, bounds.min_y, bounds.max_y]
          .some((value) => value === null || value === undefined)) {{
        return null;
      }}
      const rect = previewCanvas.getBoundingClientRect();
      return {{
        rect,
        viewport: isSurfacePreview()
          ? buildSurfaceViewport(rect, bounds)
          : buildViewport(rect, bounds),
      }};
    }}

    previewCanvas.addEventListener('wheel', (event) => {{
      const current = currentViewportForInteraction();
      if (!current) return;
      event.preventDefault();
      const x = event.clientX - current.rect.left;
      const y = event.clientY - current.rect.top;
      const {{ plot }} = current.viewport;
      if (x < plot.left || x > plot.right || y < plot.top || y > plot.bottom) return;
      const zoomFactor = Math.exp(-event.deltaY * 0.0015);
      const nextZoom = Math.min(40.0, Math.max(0.2, viewerState.zoom * zoomFactor));
      if (Math.abs(nextZoom - viewerState.zoom) < 1e-9) return;
      if (isSurfacePreview()) {{
        viewerState.zoom = nextZoom;
        drawPreview();
        return;
      }}
      const worldPoint = current.viewport.unproject(x, y);
      viewerState.zoom = nextZoom;
      const nextScale = current.viewport.baseScale * nextZoom;
      viewerState.centerX = worldPoint[0]
        - (x - current.viewport.plotCenterX) / nextScale;
      viewerState.centerY = worldPoint[1]
        + (y - current.viewport.plotCenterY) / nextScale;
      drawPreview();
    }}, {{ passive: false }});

    previewCanvas.addEventListener('pointerdown', (event) => {{
      if (![0, 1, 2].includes(event.button) || !currentViewportForInteraction()) return;
      event.preventDefault();
      viewerState.dragging = true;
      viewerState.dragMode = isSurfacePreview() && event.button === 0 ? 'rotate' : 'pan';
      viewerState.pointerId = event.pointerId;
      viewerState.lastX = event.clientX;
      viewerState.lastY = event.clientY;
      previewCanvas.setPointerCapture(event.pointerId);
      previewSurface.classList.add('dragging');
    }});

    previewCanvas.addEventListener('pointermove', (event) => {{
      if (!viewerState.dragging || viewerState.pointerId !== event.pointerId) return;
      const current = currentViewportForInteraction();
      if (!current) return;
      const deltaX = event.clientX - viewerState.lastX;
      const deltaY = event.clientY - viewerState.lastY;
      viewerState.lastX = event.clientX;
      viewerState.lastY = event.clientY;
      if (isSurfacePreview()) {{
        if (viewerState.dragMode === 'rotate') {{
          viewerState.surfaceYaw += deltaX * 0.009;
          viewerState.surfacePitch = Math.max(-1.35, Math.min(1.35,
            viewerState.surfacePitch + deltaY * 0.009
          ));
        }} else {{
          viewerState.surfacePanX += deltaX;
          viewerState.surfacePanY += deltaY;
        }}
      }} else {{
        viewerState.centerX -= deltaX / current.viewport.pixelsPerMm;
        viewerState.centerY += deltaY / current.viewport.pixelsPerMm;
      }}
      drawPreview();
    }});

    function finishPreviewDrag(event) {{
      if (!viewerState.dragging || viewerState.pointerId !== event.pointerId) return;
      viewerState.dragging = false;
      viewerState.dragMode = null;
      viewerState.pointerId = null;
      previewSurface.classList.remove('dragging');
      if (previewCanvas.hasPointerCapture(event.pointerId)) {{
        previewCanvas.releasePointerCapture(event.pointerId);
      }}
      drawPreview();
    }}

    previewCanvas.addEventListener('pointerup', finishPreviewDrag);
    previewCanvas.addEventListener('pointercancel', finishPreviewDrag);
    previewCanvas.addEventListener('contextmenu', (event) => event.preventDefault());
    previewCanvas.addEventListener('dblclick', () => {{
      resetPreviewView();
      drawPreview();
    }});

    layerSlider.addEventListener('input', () => {{
      updatePathSlider(true);
      drawPreview();
    }});
    pathProgressSlider.addEventListener('input', () => {{
      updatePathPlaybackControl();
      drawPreview();
    }});
    playCurrentPathButton.addEventListener('click', startPathPlayback);
    pathPlaybackRateInput.addEventListener('input', () => {{
      updatePathPlaybackRateLabel();
      if (pathPlayback.running && Number(pathPlaybackRateInput.value) <= 0) {{
        stopPathPlayback();
        drawPreview();
      }}
    }});
    for (const input of [
      showOuterContourInput,
      showInnerContourInput,
      showResinInfillInput,
      showRaftPathsInput,
      showFiberPathsInput,
      showTravelPathsInput
    ]) {{
      input.addEventListener('change', () => {{
        updatePathSlider();
        drawPreview();
      }});
    }}
    showCoreTravelPathsInput.addEventListener('change', drawPreview);
    showPrimelineInput.addEventListener('change', drawPreview);
    showLayerOverlayInput.addEventListener('change', drawPreview);
    showLineWidthInput.addEventListener('change', drawPreview);
    showExtrusionInput.addEventListener('change', drawPreview);
    showPathPointsInput.addEventListener('change', drawPreview);
    showDirectionInput.addEventListener('change', drawPreview);
    window.addEventListener('resize', drawPreview);
    drawPreview();
  </script>
</body>
</html>"""
