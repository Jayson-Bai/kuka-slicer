"""Replace existing cumulative E arrays after a Z-only surface mapping.

The source NPZ contract stores cumulative E values on the same padded
``(path, point)`` grid as each material path.  Surface mapping changes the
three-dimensional distance of a depositing segment, therefore this module
scales only positive E increments by the curved-to-flat segment length ratio.
It deliberately does not invent E values for inputs which did not provide
them, and it preserves zero/negative increments used by retraction semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..surface_mapper.contracts import SourceNPZ


_EXTRUSION_KEY = re.compile(r"^(layer_\d{4,}_[RF])_E$")
_LENGTH_TOLERANCE_MM = 1e-12


@dataclass(frozen=True, slots=True)
class ExtrusionCompensationResult:
    """A compensated NPZ job and compact diagnostics for callers/UI."""

    source: "SourceNPZ"
    replaced_arrays: tuple[str, ...]
    positive_segment_count: int
    mean_length_ratio: float | None
    max_length_ratio: float | None


def compensate_extrusion(
    flat_source: "SourceNPZ", curved_source: "SourceNPZ"
) -> ExtrusionCompensationResult:
    """Return ``curved_source`` with every existing material E array replaced.

    The two jobs must be the same path job before/after geometry mapping: path
    keys, padded shapes, XY values, and finite-row layout must agree.  Only Z
    is allowed to differ.  This makes a same-name E replacement unambiguous.
    """

    arrays = {key: value.copy() for key, value in curved_source.arrays.items()}
    replaced: list[str] = []
    ratios: list[float] = []
    positive_segment_count = 0
    flat_e_keys = {key for key in flat_source.arrays if _EXTRUSION_KEY.match(key)}
    curved_e_keys = {key for key in curved_source.arrays if _EXTRUSION_KEY.match(key)}
    if flat_e_keys != curved_e_keys:
        raise ValueError("flat and curved NPZ must contain the same extrusion array keys")

    for key in sorted(flat_e_keys):
        match = _EXTRUSION_KEY.match(key)
        assert match is not None
        path_key = match.group(1)
        flat_path = _paired_path_array(flat_source, curved_source, path_key)
        curved_path = curved_source.arrays[path_key]
        flat_e = flat_source.arrays[key]
        curved_e = curved_source.arrays[key]
        _validate_extrusion_grid(key, flat_e, flat_path)
        _validate_extrusion_grid(key, curved_e, curved_path)
        _validate_same_extrusion_layout(key, flat_e, curved_e)

        compensated = np.array(curved_e, dtype=np.float64, copy=True)
        for path_index in range(flat_path.shape[0]):
            point_count = _point_count(flat_path[path_index])
            if point_count == 0:
                continue
            flat_points = flat_path[path_index, :point_count, :3]
            curved_points = curved_path[path_index, :point_count, :3]
            original_e = flat_e[path_index, :point_count]
            new_e = compensated[path_index, :point_count]
            new_e[0] = original_e[0]
            for point_index in range(1, point_count):
                delta_e = float(original_e[point_index] - original_e[point_index - 1])
                if delta_e <= 0.0:
                    new_e[point_index] = new_e[point_index - 1] + delta_e
                    continue
                flat_length = float(np.linalg.norm(flat_points[point_index] - flat_points[point_index - 1]))
                curved_length = float(np.linalg.norm(curved_points[point_index] - curved_points[point_index - 1]))
                if flat_length <= _LENGTH_TOLERANCE_MM:
                    if curved_length > _LENGTH_TOLERANCE_MM:
                        raise ValueError(
                            f"{key} path {path_index} segment {point_index} has positive E but zero flat length"
                        )
                    ratio = 1.0
                else:
                    ratio = curved_length / flat_length
                new_e[point_index] = new_e[point_index - 1] + delta_e * ratio
                ratios.append(ratio)
                positive_segment_count += 1
        arrays[key] = compensated
        replaced.append(key)

    meta = dict(curved_source.meta)
    mapping = meta.get("surface_mapping")
    if isinstance(mapping, dict):
        mapping = dict(mapping)
        mapping["extrusion"] = "arc_length_ratio_compensated"
        meta["surface_mapping"] = mapping
    meta["extrusion_compensation"] = {
        "format": "arc_length_ratio_v1",
        "formula": "positive_delta_E_curved = positive_delta_E_flat * (segment_length_curved_mm / segment_length_flat_mm)",
        "replaced_arrays": replaced,
        "zero_or_negative_delta_E": "preserved",
        "does_not_create_missing_E_arrays": True,
        "scope": "path_length_only; bead_cross_section_and_orientation_are_not_compensated",
    }
    compensated_source = type(curved_source)(
        arrays=arrays,
        meta=meta,
        source_name=curved_source.source_name,
    )
    return ExtrusionCompensationResult(
        source=compensated_source,
        replaced_arrays=tuple(replaced),
        positive_segment_count=positive_segment_count,
        mean_length_ratio=float(np.mean(ratios)) if ratios else None,
        max_length_ratio=float(np.max(ratios)) if ratios else None,
    )


def _paired_path_array(
    flat_source: "SourceNPZ", curved_source: "SourceNPZ", key: str
) -> np.ndarray:
    flat_path = flat_source.arrays.get(key)
    curved_path = curved_source.arrays.get(key)
    if flat_path is None or curved_path is None:
        raise ValueError(f"flat and curved NPZ must both contain path array {key}")
    if flat_path.shape != curved_path.shape:
        raise ValueError(f"flat and curved path array shapes differ for {key}")
    flat_valid = np.isfinite(flat_path[..., 0])
    curved_valid = np.isfinite(curved_path[..., 0])
    if not np.array_equal(flat_valid, curved_valid):
        raise ValueError(f"flat and curved NPZ use different padding for {key}")
    if not np.array_equal(flat_path[..., :2], curved_path[..., :2], equal_nan=True):
        raise ValueError(f"flat and curved NPZ differ in XY for {key}")
    return flat_path


def _validate_extrusion_grid(key: str, extrusion: np.ndarray, path: np.ndarray) -> None:
    expected = path.shape[:2]
    if extrusion.shape != expected:
        raise ValueError(f"{key} shape must match {key[:-2]} path and point dimensions")
    valid = np.isfinite(path[..., 0])
    if not np.array_equal(np.isfinite(extrusion), valid):
        raise ValueError(f"{key} finite values must match its path-point padding")


def _validate_same_extrusion_layout(key: str, flat_e: np.ndarray, curved_e: np.ndarray) -> None:
    if flat_e.shape != curved_e.shape:
        raise ValueError(f"flat and curved extrusion array shapes differ for {key}")
    if not np.array_equal(np.isfinite(flat_e), np.isfinite(curved_e)):
        raise ValueError(f"flat and curved NPZ use different extrusion padding for {key}")


def _point_count(path: np.ndarray) -> int:
    return int(np.isfinite(path[:, 0]).sum())
