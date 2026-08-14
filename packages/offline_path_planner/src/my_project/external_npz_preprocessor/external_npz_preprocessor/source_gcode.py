"""Adapt native Prusa G-code into the external source-job representation.

The adapter intentionally mirrors ``native/prusa_bridge/prusa_bridge.cpp``:
only same-Z XY G0/G1 motion above Z=0 is retained, positive extrusion is
deposition, and Prusa's path/travel coalescing and cumulative-E convention are
preserved.  The resulting :class:`SourceJob` then uses the exact same Core
conversion path as ``external_layer_paths_v1`` NPZ input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from .source_npz import LayerPaths, MaterialPath, SourceJob, TravelPath


_EPS_SQUARED = 1e-14
_PLANAR_EPS = 1e-7
_DEPOSIT_EPS = 1e-12
_COMMAND_RE = re.compile(r"^\s*([GMgm])\s*([+-]?\d+)")
_PARAM_RE = re.compile(
    r"(?:^|\s)([A-Za-z])([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


@dataclass
class _MotionLayer:
    z: float
    paths: list[list[tuple[float, float, float]]] = field(default_factory=list)
    extrusion: list[list[float]] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    travel: list[list[tuple[float, float, float]]] = field(default_factory=list)
    motions: list[dict[str, int | str]] = field(default_factory=list)


def load_source_gcode(
    path: str | Path,
    default_abc: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SourceJob:
    """Load a native Prusa G-code file as a Core source job."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(str(source))
    # Native Prusa output can contain localized non-UTF-8 comments. G-code
    # commands are ASCII, so Latin-1 provides a byte-for-byte reversible view
    # while keeping the parser independent from the local code page.
    return source_job_from_gcode_lines(
        source.read_bytes().decode("latin-1").splitlines(),
        default_abc=default_abc,
    )


def source_job_from_gcode_lines(
    lines: list[str],
    *,
    default_abc: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SourceJob:
    """Convert G-code lines using the native bridge's source-motion rules."""

    position = [0.0, 0.0, 0.0]
    absolute_xyz = True
    absolute_e = True
    raw_e = 0.0
    cumulative_e = 0.0
    current_role = "other"
    layers_by_z: dict[int, _MotionLayer] = {}

    for raw_line in lines:
        line = raw_line
        comment_index = line.find(";")
        if comment_index >= 0:
            comment = line[comment_index + 1 :]
            if comment.upper().startswith("TYPE:"):
                current_role = _path_role_from_gcode_type(comment[5:])
            line = line[:comment_index]

        command = _parse_command(line)
        if command is None:
            continue
        letter, code = command
        params = _parse_params(line)
        if letter == "G" and code == 90:
            absolute_xyz = True
            continue
        if letter == "G" and code == 91:
            absolute_xyz = False
            continue
        if letter == "M" and code == 82:
            absolute_e = True
            continue
        if letter == "M" and code == 83:
            absolute_e = False
            continue

        has_x = "X" in params
        has_y = "Y" in params
        has_z = "Z" in params
        has_e = "E" in params
        if letter == "G" and code == 92:
            if has_x:
                position[0] = params["X"]
            if has_y:
                position[1] = params["Y"]
            if has_z:
                position[2] = params["Z"]
            if has_e:
                raw_e = params["E"]
            continue
        if letter != "G" or code not in (0, 1):
            continue

        before = tuple(position)
        for index, key in enumerate(("X", "Y", "Z")):
            if key in params:
                position[index] = params[key] if absolute_xyz else position[index] + params[key]
        next_raw_e = (
            (params["E"] if absolute_e else raw_e + params["E"])
            if has_e
            else raw_e
        )
        deposited = max(0.0, next_raw_e - raw_e)
        after = tuple(position)
        xy_move = (after[0] - before[0]) ** 2 + (after[1] - before[1]) ** 2 > _EPS_SQUARED
        planar = abs(after[2] - before[2]) <= _PLANAR_EPS
        if xy_move and planar and after[2] > 0.0:
            key = _native_layer_key(after[2])
            layer = layers_by_z.setdefault(key, _MotionLayer(z=after[2]))
            if deposited > _DEPOSIT_EPS:
                _append_deposit(
                    layer, before, after, cumulative_e, cumulative_e + deposited, current_role
                )
                cumulative_e += deposited
            else:
                _append_travel(layer, before, after)
        raw_e = next_raw_e

    layers: list[LayerPaths] = []
    motion_order: dict[str, list[dict[str, object]]] = {}
    path_roles: dict[str, dict[str, list[str]]] = {"R": {}}
    for logical_index, (_, source_layer) in enumerate(sorted(layers_by_z.items())):
        resin_paths = [
            MaterialPath(
                material="R",
                order=index,
                points=_with_abc(points, default_abc),
                extrusion=np.asarray(source_layer.extrusion[index], dtype=np.float64),
            )
            for index, points in enumerate(source_layer.paths)
        ]
        travel_paths = [
            TravelPath(order=index, points=_with_abc(points, default_abc))
            for index, points in enumerate(source_layer.travel)
        ]
        if resin_paths:
            path_roles["R"][str(logical_index)] = list(source_layer.roles)
        if source_layer.motions:
            motion_order[str(logical_index)] = [dict(record) for record in source_layer.motions]
        layers.append(
            LayerPaths(
                index=logical_index,
                resin_paths=resin_paths,
                travel_paths=travel_paths,
            )
        )

    if not any(layer.resin_paths for layer in layers):
        raise ValueError("G-code does not contain any planar positive-extrusion motion above Z=0")
    return SourceJob(
        meta={
            "format": "prusa_gcode_source_v1",
            "source_format": "prusa_native_gcode",
            "path_roles": path_roles,
            "motion_order": motion_order,
        },
        layers=layers,
    )


def with_fiber_paths(
    job: SourceJob,
    fiber_paths_by_layer: Mapping[int, Sequence[np.ndarray]],
    *,
    fiber_travel_paths_by_layer: Mapping[int, Sequence[np.ndarray]] | None = None,
    default_abc: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SourceJob:
    """Attach already-expanded fiber paths without changing G-code resin data.

    Fiber JSON has project-specific placement rules and is expanded before it
    reaches Core.  This narrow adapter accepts that resulting mapping and
    leaves all resin paths, native travel paths, E values and motion-order
    records from G-code untouched.  Optional fiber connectors are appended as
    explicit source travels and indexed separately so only the fiber-to-fiber
    boundaries consume them.
    """

    remaining = {int(index): paths for index, paths in fiber_paths_by_layer.items()}
    remaining_travels = {
        int(index): paths
        for index, paths in (fiber_travel_paths_by_layer or {}).items()
    }
    layers: list[LayerPaths] = []
    fiber_travel_indexes: dict[str, list[int]] = {}
    for layer in job.layers:
        source_paths = remaining.pop(layer.index, ())
        fiber_paths = [
            MaterialPath(
                material="F",
                order=order,
                points=_normalize_fiber_path(points, default_abc),
            )
            for order, points in enumerate(source_paths)
        ]
        source_travels = remaining_travels.pop(layer.index, ())
        expected_travel_count = max(0, len(fiber_paths) - 1)
        if len(source_travels) != expected_travel_count:
            raise ValueError(
                f"fiber layer {layer.index} needs {expected_travel_count} interpath travels, "
                f"received {len(source_travels)}"
            )
        travel_paths = list(layer.travel_paths)
        if source_travels:
            first_travel_index = len(travel_paths)
            travel_paths.extend(
                TravelPath(
                    order=first_travel_index + index,
                    points=_normalize_fiber_path(points, default_abc),
                )
                for index, points in enumerate(source_travels)
            )
            fiber_travel_indexes[str(layer.index)] = list(
                range(first_travel_index, first_travel_index + len(source_travels))
            )
        layers.append(
            LayerPaths(
                index=layer.index,
                resin_paths=list(layer.resin_paths),
                fiber_paths=fiber_paths,
                travel_paths=travel_paths,
            )
        )
    if remaining:
        unknown = ", ".join(str(index) for index in sorted(remaining))
        raise ValueError(f"fiber paths reference G-code layers that do not exist: {unknown}")
    if remaining_travels:
        unknown = ", ".join(str(index) for index in sorted(remaining_travels))
        raise ValueError(f"fiber travels reference G-code layers that do not exist: {unknown}")
    meta = dict(job.meta)
    meta["fiber_source"] = "expanded_external_fiber_paths"
    if fiber_travel_indexes:
        meta["fiber_travel_path_indexes"] = fiber_travel_indexes
    return SourceJob(meta=meta, layers=layers)


def translate_source_job(
    job: SourceJob,
    translation_xyz: tuple[float, float, float],
) -> SourceJob:
    """Translate parsed native G-code into the slicer's source coordinate frame."""

    translation = np.asarray(translation_xyz, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("G-code source translation must contain three finite values")

    def translated(points: np.ndarray) -> np.ndarray:
        result = np.asarray(points, dtype=np.float64).copy()
        result[:, :3] += translation
        return result

    layers = [
        LayerPaths(
            index=layer.index,
            resin_paths=[
                MaterialPath(path.material, path.order, translated(path.points), path.extrusion)
                for path in layer.resin_paths
            ],
            fiber_paths=[
                MaterialPath(path.material, path.order, translated(path.points), path.extrusion)
                for path in layer.fiber_paths
            ],
            travel_paths=[
                TravelPath(path.order, translated(path.points))
                for path in layer.travel_paths
            ],
        )
        for layer in job.layers
    ]
    meta = dict(job.meta)
    meta["native_gcode_source_translation_mm"] = translation.tolist()
    return SourceJob(meta=meta, layers=layers)


def prepend_prusa_startup_travel(
    job: SourceJob,
    *,
    start_xy: tuple[float, float],
    primeline_enabled: bool,
    primeline_xy: tuple[float, float],
) -> SourceJob:
    """Add the same origin-to-first-motion travel used by the Prusa UI path.

    The source points are expressed so ``source_job_to_parsed_commands()``
    maps them to the existing final-machine coordinates without first writing
    a normalized source NPZ.
    """

    first_layer = next(
        (layer for layer in job.layers if layer.resin_paths or layer.fiber_paths),
        None,
    )
    if first_layer is None:
        return job
    source_min = _source_xy_min(job)
    first_material = (first_layer.resin_paths or first_layer.fiber_paths)[0]
    first_z = float(first_material.points[0, 2])
    if primeline_enabled:
        target_xy = (
            source_min[0] + float(primeline_xy[0]),
            source_min[1] + float(primeline_xy[1]),
        )
    elif first_layer.travel_paths:
        target_xy = tuple(float(value) for value in first_layer.travel_paths[0].points[0, :2])
    else:
        target_xy = tuple(float(value) for value in first_material.points[0, :2])
    start = np.asarray(
        [
            source_min[0] - float(start_xy[0]),
            source_min[1] - float(start_xy[1]),
            first_z,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    target = np.asarray([target_xy[0], target_xy[1], first_z, 0.0, 0.0, 0.0], dtype=np.float64)
    if np.linalg.norm(target[:3] - start[:3]) <= 1e-7:
        return job

    layers: list[LayerPaths] = []
    for layer in job.layers:
        if layer.index != first_layer.index:
            layers.append(layer)
            continue
        travel_paths = [TravelPath(0, np.vstack((start, target)))]
        travel_paths.extend(
            TravelPath(index, path.points)
            for index, path in enumerate(layer.travel_paths, start=1)
        )
        layers.append(
            LayerPaths(
                index=layer.index,
                resin_paths=list(layer.resin_paths),
                fiber_paths=list(layer.fiber_paths),
                travel_paths=travel_paths,
            )
        )
    meta = dict(job.meta)
    motion_order = dict(meta.get("motion_order", {}))
    records = motion_order.get(str(first_layer.index), [])
    if isinstance(records, list):
        shifted = [
            {**record, "index": int(record.get("index", 0)) + 1}
            if isinstance(record, dict) and record.get("kind") == "travel"
            else record
            for record in records
        ]
        motion_order[str(first_layer.index)] = [{"kind": "travel", "index": 0}, *shifted]
    meta["motion_order"] = motion_order
    meta["startup_travel_count"] = 1
    meta["startup_travel_source_frame"] = "normalized_prusa"
    return SourceJob(meta=meta, layers=layers)


def _parse_command(line: str) -> tuple[str, int] | None:
    match = _COMMAND_RE.match(line)
    if match is None:
        return None
    return match.group(1).upper(), int(match.group(2))


def _parse_params(line: str) -> dict[str, float]:
    return {match.group(1).upper(): float(match.group(2)) for match in _PARAM_RE.finditer(line)}


def _native_layer_key(z: float) -> int:
    # ``std::llround`` used by the native bridge rounds positive halfway values
    # away from zero. Retained source layers always have z > 0.
    return int(math.floor(z * 1_000_000.0 + 0.5))


def _same_point(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    return sum((left - right) ** 2 for left, right in zip(a, b)) <= _EPS_SQUARED


def _append_deposit(
    layer: _MotionLayer,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    e_start: float,
    e_end: float,
    role: str,
) -> None:
    if layer.paths and layer.roles[-1] == role and _same_point(layer.paths[-1][-1], start):
        layer.paths[-1].append(end)
        layer.extrusion[-1].append(e_end)
        return
    layer.paths.append([start, end])
    layer.extrusion.append([e_start, e_end])
    layer.roles.append(role)
    layer.motions.append({"kind": "deposit", "index": len(layer.paths) - 1})


def _append_travel(
    layer: _MotionLayer,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
) -> None:
    if layer.travel and _same_point(layer.travel[-1][-1], start):
        layer.travel[-1].append(end)
        return
    layer.travel.append([start, end])
    layer.motions.append({"kind": "travel", "index": len(layer.travel) - 1})


def _with_abc(
    points: list[tuple[float, float, float]], default_abc: tuple[float, float, float]
) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    abc = np.tile(np.asarray(default_abc, dtype=np.float64), (len(points), 1))
    return np.hstack((xyz, abc))


def _normalize_fiber_path(
    raw_path: np.ndarray,
    default_abc: tuple[float, float, float],
) -> np.ndarray:
    points = np.asarray(raw_path, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] not in (3, 6):
        raise ValueError(
            f"fiber paths must be Nx3 or Nx6 with at least two points, got shape {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("fiber paths must contain only finite values")
    if points.shape[1] == 6:
        return points.copy()
    return _with_abc([tuple(row) for row in points], default_abc)


def _source_xy_min(job: SourceJob) -> tuple[float, float]:
    points = [
        path.points[:, :2]
        for layer in job.layers
        for path in [*layer.resin_paths, *layer.fiber_paths]
        if path.points.size
    ]
    if not points:
        return (0.0, 0.0)
    merged = np.vstack(points)
    return (float(np.min(merged[:, 0])), float(np.min(merged[:, 1])))


def _path_role_from_gcode_type(value: str) -> str:
    """Match the native bridge's role taxonomy and path-boundary decisions."""

    upper = value.upper()
    if "BRIM" in upper:
        return "brim"
    if "SUPPORT MATERIAL" in upper:
        return "raft"
    if "EXTERNAL PERIMETER" in upper:
        return "outer_contour"
    if "PERIMETER" in upper:
        return "inner_contour"
    if "INFILL" in upper:
        return "infill"
    return "other"
