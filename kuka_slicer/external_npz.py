from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal

import numpy as np

Material = Literal["R", "F"]


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
    """Write the documented external source NPZ archive."""

    arrays: dict[str, np.ndarray] = {
        "meta": np.array(json.dumps(_defaulted_meta(job.meta), ensure_ascii=False))
    }
    valid_path_count = 0

    for group in job.material_paths:
        key = f"layer_{group.layer_index:04d}_{group.material}"
        arrays[key] = paths_to_padded_array(group.paths)
        valid_path_count += len(group.paths)

    if valid_path_count == 0:
        raise ValueError("cannot write external source NPZ without any paths")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)


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
        "format": "external_layer_paths_v1",
        "unit": "mm",
        "point_columns": ["x", "y", "z"],
        "materials": {"R": "resin", "F": "fiber"},
        "description": "Layer/material path arrays for external_npz_preprocessor",
    }
    base.update(meta)
    return base

