"""Read external layer/material path NPZ files."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_LAYER_KEY_RE = re.compile(r"^layer_(\d{4})_([RFT])$")
SOURCE_NPZ_CONTRACT_ID = "external_layer_paths_v1"
SUPPORTED_SOURCE_NPZ_CONTRACTS = frozenset({SOURCE_NPZ_CONTRACT_ID})


@dataclass(frozen=True)
class MaterialPath:
    material: str
    order: int
    points: np.ndarray
    # Optional cumulative source E value for every XYZ point.  The value is
    # intentionally kept in the source coordinate system; the converter
    # normalizes it per path because the runtime resets E at every path end.
    extrusion: np.ndarray | None = None


@dataclass(frozen=True)
class TravelPath:
    order: int
    points: np.ndarray


@dataclass(frozen=True)
class LayerPaths:
    index: int
    resin_paths: list[MaterialPath] = field(default_factory=list)
    fiber_paths: list[MaterialPath] = field(default_factory=list)
    travel_paths: list[TravelPath] = field(default_factory=list)


@dataclass(frozen=True)
class SourceJob:
    meta: dict[str, Any]
    layers: list[LayerPaths]


def load_source_npz(path: str | Path, default_abc: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> SourceJob:
    """Load the external NPZ source contract.

    Expected keys are ``layer_0000_R``, ``layer_0000_F`` and optionally
    ``layer_0000_T``. Each value is a
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
        _validate_source_contract(meta)
        layer_map: dict[int, dict[str, list[Any]]] = {}
        for key in sorted(npz.files):
            match = _LAYER_KEY_RE.match(key)
            if not match:
                continue
            layer_index = int(match.group(1))
            material = match.group(2)
            if material == "T":
                paths = _read_travel_paths(npz[key], default_abc)
                bucket = layer_map.setdefault(
                    layer_index, {"R": [], "F": [], "T": []}
                )
                bucket["T"].extend(paths)
                continue
            extrusion_key = f"{key}_E"
            raw_extrusion = npz[extrusion_key] if extrusion_key in npz.files else None
            paths = _read_material_paths(
                npz[key],
                material,
                layer_index,
                default_abc,
                raw_extrusion=raw_extrusion,
            )
            bucket = layer_map.setdefault(layer_index, {"R": [], "F": [], "T": []})
            bucket[material].extend(paths)

    if not layer_map:
        raise ValueError("source NPZ does not contain any layer_0000_R or layer_0000_F keys")

    layers = [
        LayerPaths(
            index=idx,
            resin_paths=values["R"],
            fiber_paths=values["F"],
            travel_paths=values["T"],
        )
        for idx, values in sorted(layer_map.items())
    ]
    return SourceJob(meta=meta, layers=layers)


def _validate_source_contract(meta: dict[str, Any]) -> None:
    """Accept the documented v1 contract and unversioned legacy source files."""
    contract_id = meta.get("format")
    if contract_id is None:
        return
    if (
        not isinstance(contract_id, str)
        or contract_id not in SUPPORTED_SOURCE_NPZ_CONTRACTS
    ):
        supported = ", ".join(sorted(SUPPORTED_SOURCE_NPZ_CONTRACTS))
        raise ValueError(
            f"unsupported source NPZ format {contract_id!r}; supported: {supported}"
        )


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
    *,
    raw_extrusion=None,
) -> list[MaterialPath]:
    paths: list[MaterialPath] = []
    records = (
        _iter_raw_path_pairs(raw_paths, raw_extrusion)
        if raw_extrusion is not None
        else ((raw_path, None) for raw_path in _iter_raw_paths(raw_paths))
    )
    for order, (raw_path, raw_e) in enumerate(records):
        points = _normalize_points(raw_path, default_abc)
        extrusion = _normalize_extrusion(raw_e, points.shape[0])
        paths.append(
            MaterialPath(
                material=material,
                order=order,
                points=points,
                extrusion=extrusion,
            )
        )
    return paths


def _read_travel_paths(
    raw_paths,
    default_abc: tuple[float, float, float],
) -> list[TravelPath]:
    return [
        TravelPath(order=order, points=_normalize_points(raw_path, default_abc))
        for order, raw_path in enumerate(_iter_raw_paths(raw_paths))
    ]


def _iter_raw_path_pairs(raw_paths, raw_extrusion):
    """Yield a path and its aligned cumulative E profile after unpadding."""

    paths = np.asarray(raw_paths)
    extrusion = np.asarray(raw_extrusion)
    if paths.dtype == object:
        if extrusion.dtype != object or len(paths) != len(extrusion):
            raise ValueError("legacy object paths and E arrays must have matching path counts")
        for raw_path, raw_e in zip(paths, extrusion):
            path = np.asarray(raw_path)
            e_values = np.asarray(raw_e)
            if path.ndim != 2 or e_values.ndim != 1 or path.shape[0] != e_values.shape[0]:
                raise ValueError("legacy object path and E arrays must have aligned point counts")
            yield path, e_values
        return

    if paths.ndim != 3:
        raise ValueError(
            f"layer material arrays must be a 3D numeric array or legacy object array, got shape {paths.shape}"
        )
    extrusion = np.asarray(extrusion)
    expected_shape = paths.shape[:2]
    if extrusion.ndim != 2 or extrusion.shape != expected_shape:
        raise ValueError(
            f"layer E arrays must be a 2D numeric array shaped {expected_shape}, got shape {extrusion.shape}"
        )

    for raw_path, raw_e in zip(paths, extrusion):
        path = np.asarray(raw_path)
        e_values = np.asarray(raw_e)
        valid_rows = ~np.isnan(path).all(axis=1)
        if np.isnan(path[valid_rows]).any():
            raise ValueError("path padding rows must be all-NaN; partial NaN rows are invalid")
        if np.isnan(e_values[valid_rows]).any():
            raise ValueError("E values for valid path points must not be NaN")
        if np.any(~valid_rows & ~np.isnan(e_values)):
            raise ValueError("E padding rows must be all-NaN")
        yield path[valid_rows], e_values[valid_rows]


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
        path = np.asarray(raw_path)
        valid_rows = ~np.isnan(path).all(axis=1)
        path = path[valid_rows]
        if np.isnan(path).any():
            raise ValueError("path padding rows must be all-NaN; partial NaN rows are invalid")
        yield path


def _normalize_points(raw_path, default_abc: tuple[float, float, float]) -> np.ndarray:
    raw = np.asarray(raw_path)
    dtype = _source_float_dtype(raw)
    points = np.asarray(raw, dtype=dtype)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] not in (3, 6):
        raise ValueError(
            f"path arrays must be Nx3 or Nx6 with at least 2 rows, got shape {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ValueError("valid path points must contain only finite values")
    if points.shape[1] == 6:
        return points.astype(dtype, copy=False)
    abc = np.tile(np.asarray(default_abc, dtype=dtype), (points.shape[0], 1))
    return np.hstack((points, abc)).astype(dtype, copy=False)


def _normalize_extrusion(raw_extrusion, point_count: int) -> np.ndarray | None:
    if raw_extrusion is None:
        return None
    extrusion = np.asarray(raw_extrusion)
    dtype = _source_float_dtype(extrusion)
    extrusion = extrusion.astype(dtype, copy=False)
    if extrusion.ndim != 1 or extrusion.shape[0] != point_count:
        raise ValueError(
            f"path E arrays must contain one finite value per point, got shape {extrusion.shape}"
        )
    if not np.isfinite(extrusion).all():
        raise ValueError("path E values must be finite")
    return extrusion.astype(dtype, copy=False)


def _source_float_dtype(array: np.ndarray) -> np.dtype:
    """Keep source float64 precision while accepting legacy float32 files."""

    if array.dtype == np.dtype(np.float64):
        return np.dtype(np.float64)
    if array.dtype == np.dtype(np.float32):
        return np.dtype(np.float32)
    raise ValueError(
        f"source numeric arrays must use float64 or legacy float32, got {array.dtype}"
    )

