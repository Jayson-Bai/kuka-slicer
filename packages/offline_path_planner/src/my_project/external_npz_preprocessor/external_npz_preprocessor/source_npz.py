"""Read external layer/material path NPZ files."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_LAYER_KEY_RE = re.compile(r"^layer_(\d{4})_([RF])$")


@dataclass(frozen=True)
class MaterialPath:
    material: str
    order: int
    points: np.ndarray


@dataclass(frozen=True)
class LayerPaths:
    index: int
    resin_paths: list[MaterialPath] = field(default_factory=list)
    fiber_paths: list[MaterialPath] = field(default_factory=list)


@dataclass(frozen=True)
class SourceJob:
    meta: dict[str, Any]
    layers: list[LayerPaths]


def load_source_npz(path: str | Path, default_abc: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> SourceJob:
    """Load the external NPZ source contract.

    Expected keys are ``layer_0000_R`` and ``layer_0000_F``. Each value is a
    numeric 3D array shaped ``path_count x max_points x columns`` with NaN padded
    rows, or a legacy object array of path arrays. Each path must be Nx3 or Nx6
    and is normalized to Nx6 ``[x, y, z, a, b, c]``. Source Z is the trajectory
    geometry and is not overwritten by process layer-height parameters.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(str(source))

    with np.load(source, allow_pickle=True) as npz:
        meta = _read_meta(npz)
        layer_map: dict[int, dict[str, list[MaterialPath]]] = {}
        for key in sorted(npz.files):
            match = _LAYER_KEY_RE.match(key)
            if not match:
                continue
            layer_index = int(match.group(1))
            material = match.group(2)
            paths = _read_material_paths(npz[key], material, layer_index, default_abc)
            bucket = layer_map.setdefault(layer_index, {"R": [], "F": []})
            bucket[material].extend(paths)

    if not layer_map:
        raise ValueError("source NPZ does not contain any layer_0000_R or layer_0000_F keys")

    layers = [
        LayerPaths(index=idx, resin_paths=values["R"], fiber_paths=values["F"])
        for idx, values in sorted(layer_map.items())
    ]
    return SourceJob(meta=meta, layers=layers)


def _read_meta(npz) -> dict[str, Any]:
    if "meta" not in npz.files:
        return {}
    raw = npz["meta"]
    if raw.shape == ():
        text = str(raw.item())
    else:
        text = str(raw.tolist())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"meta must be a JSON string: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("meta JSON must be an object")
    return data


def _read_material_paths(
    raw_paths,
    material: str,
    layer_index: int,
    default_abc: tuple[float, float, float],
) -> list[MaterialPath]:
    paths: list[MaterialPath] = []
    for order, raw_path in enumerate(_iter_raw_paths(raw_paths)):
        points = _normalize_points(raw_path, default_abc)
        paths.append(MaterialPath(material=material, order=order, points=points))
    return paths


def _iter_raw_paths(raw_paths):
    arr = np.asarray(raw_paths)
    if arr.dtype == object:
        yield from list(raw_paths)
        return
    if arr.ndim != 3:
        raise ValueError(
            f"layer material arrays must be a 3D numeric array or legacy object array, got shape {arr.shape}"
        )
    for raw_path in arr:
        path = np.asarray(raw_path, dtype=np.float32)
        valid_rows = ~np.isnan(path).all(axis=1)
        path = path[valid_rows]
        if np.isnan(path).any():
            raise ValueError("path padding rows must be all-NaN; partial NaN rows are invalid")
        yield path


def _normalize_points(raw_path, default_abc: tuple[float, float, float]) -> np.ndarray:
    points = np.asarray(raw_path, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] not in (3, 6):
        raise ValueError(
            f"path arrays must be Nx3 or Nx6 with at least 2 rows, got shape {points.shape}"
        )
    if points.shape[1] == 6:
        return points.astype(np.float32, copy=False)
    abc = np.tile(np.asarray(default_abc, dtype=np.float32), (points.shape[0], 1))
    return np.hstack((points, abc)).astype(np.float32, copy=False)

