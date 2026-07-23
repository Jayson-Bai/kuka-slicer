from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal

import numpy as np

Material = Literal["R", "F"]
DEFAULT_EXPORT_CHORD_TOLERANCE_MM = 0.05
SOURCE_NPZ_CONTRACT_ID = "external_layer_paths_v1"


@dataclass
class MaterialPaths:
    layer_index: int
    material: Material
    paths: list[np.ndarray] = field(default_factory=list)


@dataclass
class ExternalSourceJob:
    material_paths: list[MaterialPaths]
    meta: dict[str, object] = field(default_factory=dict)


def write_external_source_npz(job: ExternalSourceJob, output_path: str | Path) -> None:
    """Write the documented external source NPZ archive with G-code-like paths."""

    arrays: dict[str, np.ndarray] = {
        "meta": np.array(json.dumps(_defaulted_meta(job.meta), ensure_ascii=False))
    }
    valid_path_count = 0

    for group in job.material_paths:
        key = f"layer_{group.layer_index:04d}_{group.material}"
        arrays[key] = paths_to_padded_array(simplify_paths_for_export(group.paths))
        valid_path_count += len(group.paths)

    if valid_path_count == 0:
        raise ValueError("cannot write external source NPZ without any paths")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)


def simplify_job_paths_for_export(
    job: ExternalSourceJob,
    chord_tolerance: float = DEFAULT_EXPORT_CHORD_TOLERANCE_MM,
) -> None:
    """Simplify every path in place without changing path count or order."""

    for group in job.material_paths:
        group.paths = simplify_paths_for_export(group.paths, chord_tolerance)


def simplify_paths_for_export(
    paths: list[np.ndarray],
    chord_tolerance: float = DEFAULT_EXPORT_CHORD_TOLERANCE_MM,
) -> list[np.ndarray]:
    if not np.isfinite(chord_tolerance) or chord_tolerance <= 0:
        raise ValueError("chord_tolerance must be positive")
    return [simplify_path_for_export(path, chord_tolerance) for path in paths]


def simplify_path_for_export(
    path: np.ndarray,
    chord_tolerance: float = DEFAULT_EXPORT_CHORD_TOLERANCE_MM,
) -> np.ndarray:
    """Keep line endpoints and only the curve points needed for shape fidelity."""

    array = _normalize_path(path)
    if array.shape[0] <= 2:
        return array.copy()

    points = np.asarray(array[:, :3], dtype=np.float64)
    closure_tolerance = max(chord_tolerance * 1e-3, 1e-7)
    closed = (
        array.shape[0] > 3
        and float(np.linalg.norm(points[0] - points[-1])) <= closure_tolerance
    )
    if not closed:
        return array[_rdp_keep_indices(points, chord_tolerance)].copy()

    # A closed ring has coincident endpoints, so simplify it as two open arcs
    # split at the point farthest from the start. This preserves closure and
    # avoids treating the whole ring as a zero-length straight segment.
    split_index = int(np.argmax(np.linalg.norm(points[:-1] - points[0], axis=1)))
    if split_index <= 0 or split_index >= array.shape[0] - 1:
        split_index = (array.shape[0] - 1) // 2
    first_arc = array[: split_index + 1]
    second_arc = array[split_index:]
    first_keep = _rdp_keep_indices(first_arc[:, :3], chord_tolerance)
    second_keep = _rdp_keep_indices(second_arc[:, :3], chord_tolerance)
    simplified = np.vstack(
        (
            first_arc[first_keep][:-1],
            second_arc[second_keep],
        )
    ).astype(np.float32, copy=False)
    if simplified.shape[0] < 4:
        return array.copy()
    simplified[-1] = simplified[0]
    return simplified


def _rdp_keep_indices(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Return Ramer-Douglas-Peucker indices using XYZ chord deviation."""

    point_count = points.shape[0]
    if point_count <= 2:
        return np.arange(point_count, dtype=np.intp)

    keep = np.zeros(point_count, dtype=bool)
    keep[0] = True
    keep[-1] = True
    pending = [(0, point_count - 1)]
    while pending:
        start_index, end_index = pending.pop()
        if end_index <= start_index + 1:
            continue
        start = points[start_index]
        end = points[end_index]
        segment = end - start
        segment_length_squared = float(np.dot(segment, segment))
        interior = points[start_index + 1 : end_index]
        if segment_length_squared <= 1e-24:
            distances = np.linalg.norm(interior - start, axis=1)
        else:
            projections = np.clip(
                ((interior - start) @ segment) / segment_length_squared,
                0.0,
                1.0,
            )
            nearest = start + projections[:, None] * segment
            distances = np.linalg.norm(interior - nearest, axis=1)
        relative_index = int(np.argmax(distances))
        maximum_distance = float(distances[relative_index])
        if maximum_distance <= tolerance:
            continue
        split_index = start_index + 1 + relative_index
        keep[split_index] = True
        pending.append((start_index, split_index))
        pending.append((split_index, end_index))
    return np.flatnonzero(keep)


def paths_to_padded_array(paths: list[np.ndarray]) -> np.ndarray:
    normalized = [_normalize_path(path) for path in paths]
    if not normalized:
        return np.full((0, 0, 3), np.nan, dtype=np.float32)

    column_counts = {path.shape[1] for path in normalized}
    if len(column_counts) != 1:
        raise ValueError("all paths in one layer/material group must use the same column count")

    columns = column_counts.pop()
    max_points = max(path.shape[0] for path in normalized)
    result = np.full((len(normalized), max_points, columns), np.nan, dtype=np.float32)
    for index, path in enumerate(normalized):
        result[index, : path.shape[0], :] = path
    return result


def _normalize_path(path: np.ndarray) -> np.ndarray:
    array = np.asarray(path, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("path must be a two-dimensional array")
    if array.shape[0] < 2:
        raise ValueError("path must contain at least two points")
    if array.shape[1] not in (3, 6):
        raise ValueError("path columns must be 3 or 6")
    if np.isnan(array).any():
        raise ValueError("paths passed to writer must not contain NaN values")
    return array


def _defaulted_meta(meta: dict[str, object]) -> dict[str, object]:
    base: dict[str, object] = {
        "format": SOURCE_NPZ_CONTRACT_ID,
        "unit": "mm",
        "point_columns": ["x", "y", "z"],
        "materials": {"R": "resin", "F": "fiber"},
        "description": "Layer/material path arrays for external_npz_preprocessor",
        "path_sampling": {
            "method": "3d_chord_error",
            "chord_tolerance_mm": DEFAULT_EXPORT_CHORD_TOLERANCE_MM,
            "straight_segments": "endpoints_only",
            "preserves_path_count_and_order": True,
        },
    }
    base.update(meta)
    return base
