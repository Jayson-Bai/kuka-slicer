"""Generate example source NPZ files for the external preprocessor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _padded_paths(paths: list[np.ndarray]) -> np.ndarray:
    max_points = max(np.asarray(path).shape[0] for path in paths)
    columns = np.asarray(paths[0]).shape[1]
    out = np.full((len(paths), max_points, columns), np.nan, dtype=np.float32)
    for idx, path in enumerate(paths):
        arr = np.asarray(path, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != columns:
            raise ValueError("all template paths must be 2D arrays with the same column count")
        out[idx, : arr.shape[0], :] = arr
    return out


def write_two_layer_template_npz(path: str | Path) -> Path:
    """Write a two-layer resin/fiber source NPZ template."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": "external_layer_paths_v1",
        "template": "two_layer_resin_fiber",
        "unit": "mm",
        "point_columns": ["x", "y", "z"],
        "materials": {"R": "resin", "F": "fiber"},
        "description": "Two layers; each layer contains ordered XYZ resin and fiber paths. Z comes from the source slicer."
    }

    layer_0000_R = _padded_paths([
        np.array([
            [0.0, 0.0, 0.50],
            [30.0, 0.0, 0.50],
            [30.0, 20.0, 0.50],
        ]),
        np.array([
            [5.0, 5.0, 0.50],
            [25.0, 5.0, 0.50],
        ]),
    ])
    layer_0000_F = _padded_paths([
        np.array([
            [2.0, 2.0, 0.60],
            [28.0, 18.0, 0.60],
        ]),
    ])
    layer_0001_R = _padded_paths([
        np.array([
            [0.0, 0.0, 1.00],
            [0.0, 20.0, 1.00],
            [30.0, 20.0, 1.00],
        ]),
    ])
    layer_0001_F = _padded_paths([
        np.array([
            [4.0, 18.0, 1.10],
            [16.0, 8.0, 1.10],
            [28.0, 18.0, 1.10],
        ]),
        np.array([
            [4.0, 4.0, 1.10],
            [28.0, 4.0, 1.10],
        ]),
    ])

    np.savez(
        target,
        meta=np.array(json.dumps(meta, ensure_ascii=False)),
        layer_0000_R=layer_0000_R,
        layer_0000_F=layer_0000_F,
        layer_0001_R=layer_0001_R,
        layer_0001_F=layer_0001_F,
    )
    return target
