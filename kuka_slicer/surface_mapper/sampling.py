"""Surface-aware input resampling owned by the surface mapper.

The Core deliberately owns trajectory fitting and time sampling.  This module
only adds geometric samples before mapping so each newly introduced position
receives an analytical surface normal instead of an interpolated ABC value.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

from .contracts import SourceNPZ


_MATERIAL_PATH_KEY = re.compile(r"^layer_\d{4,}_[RF]$")


@dataclass(frozen=True, slots=True)
class SurfaceSamplingConfig:
    """Sampling policy for deposited paths before surface mapping.

    ``max_segment_length_mm`` is evaluated in the original planar XY domain.
    ``None`` keeps the source point grid unchanged, which is useful for
    compatibility checks.  Travel paths are intentionally excluded: their
    surface-normal accuracy does not define deposited bead geometry and
    resampling them would needlessly inflate the Core input.
    """

    max_segment_length_mm: float | None = 1.5

    def __post_init__(self) -> None:
        if self.max_segment_length_mm is None:
            return
        value = float(self.max_segment_length_mm)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("max_segment_length_mm must be finite and > 0, or None")

    @property
    def enabled(self) -> bool:
        return self.max_segment_length_mm is not None


@dataclass(frozen=True, slots=True)
class SurfaceSamplingResult:
    """The resampled planar source and compact accounting for mapping metadata."""

    source: SourceNPZ
    material_point_count_before: int
    material_point_count_after: int
    resampled_path_count: int


def resample_material_paths(
    source: SourceNPZ, config: SurfaceSamplingConfig
) -> SurfaceSamplingResult:
    """Resample R/F path grids and their aligned cumulative E profiles.

    Newly inserted rows interpolate every planar source column and cumulative
    E by the same segment parameter.  The mapper subsequently replaces their
    Z and ABC from the analytical target surface; extrusion compensation then
    uses this resampled planar source as its matching flat reference.
    """

    before = _material_point_count(source.arrays)
    if not config.enabled:
        return SurfaceSamplingResult(source, before, before, 0)

    arrays = {key: np.array(value, copy=True) for key, value in source.arrays.items()}
    resampled_path_count = 0
    for key, raw_paths in tuple(arrays.items()):
        if not _MATERIAL_PATH_KEY.match(key):
            continue
        e_key = f"{key}_E"
        raw_e = arrays.get(e_key)
        sampled_paths, sampled_e, changed = _resample_path_grid(
            raw_paths,
            raw_e,
            max_segment_length_mm=float(config.max_segment_length_mm),
        )
        arrays[key] = sampled_paths
        if sampled_e is not None:
            arrays[e_key] = sampled_e
        resampled_path_count += changed

    sampled = SourceNPZ(
        arrays=arrays,
        meta=dict(source.meta),
        source_name=source.source_name,
    )
    return SurfaceSamplingResult(
        source=sampled,
        material_point_count_before=before,
        material_point_count_after=_material_point_count(sampled.arrays),
        resampled_path_count=resampled_path_count,
    )


def _resample_path_grid(raw_paths, raw_e, *, max_segment_length_mm: float):
    paths = np.asarray(raw_paths, dtype=np.float64)
    if paths.ndim != 3:
        raise ValueError("material path arrays must be three-dimensional")
    e_values = None if raw_e is None else np.asarray(raw_e, dtype=np.float64)
    if e_values is not None and e_values.shape != paths.shape[:2]:
        raise ValueError("material E arrays must match their path grids")

    sampled_paths: list[np.ndarray] = []
    sampled_e: list[np.ndarray] | None = [] if e_values is not None else None
    changed = 0
    for index, raw_path in enumerate(paths):
        valid = np.isfinite(raw_path[:, 0])
        point_count = int(valid.sum())
        path = raw_path[:point_count]
        profile = None if e_values is None else e_values[index, :point_count]
        result, result_e = _resample_one_path(path, profile, max_segment_length_mm)
        sampled_paths.append(result)
        if sampled_e is not None:
            assert result_e is not None
            sampled_e.append(result_e)
        if len(result) != len(path):
            changed += 1

    max_points = max((len(path) for path in sampled_paths), default=0)
    output = np.full((len(sampled_paths), max_points, paths.shape[2]), np.nan, dtype=np.float64)
    output_e = (
        np.full((len(sampled_paths), max_points), np.nan, dtype=np.float64)
        if sampled_e is not None
        else None
    )
    for index, path in enumerate(sampled_paths):
        output[index, : len(path)] = path
        if output_e is not None:
            output_e[index, : len(path)] = sampled_e[index]
    return output, output_e, changed


def _resample_one_path(path, e_profile, max_segment_length_mm: float):
    if len(path) < 2:
        return np.array(path, copy=True), None if e_profile is None else np.array(e_profile, copy=True)

    points = [np.array(path[0], copy=True)]
    e_values = None if e_profile is None else [float(e_profile[0])]
    for index in range(1, len(path)):
        start = path[index - 1]
        end = path[index]
        xy_length = float(np.linalg.norm(end[:2] - start[:2]))
        subdivisions = max(1, int(math.ceil(xy_length / max_segment_length_mm)))
        for step in range(1, subdivisions + 1):
            if step == subdivisions:
                point = np.array(end, copy=True)
                e_value = None if e_profile is None else float(e_profile[index])
            else:
                ratio = step / subdivisions
                point = start + ratio * (end - start)
                e_value = (
                    None
                    if e_profile is None
                    else float(e_profile[index - 1] + ratio * (e_profile[index] - e_profile[index - 1]))
                )
            points.append(point)
            if e_values is not None:
                assert e_value is not None
                e_values.append(e_value)
    return np.asarray(points, dtype=np.float64), None if e_values is None else np.asarray(e_values, dtype=np.float64)


def _material_point_count(arrays: dict[str, np.ndarray]) -> int:
    return sum(
        int(np.isfinite(np.asarray(value)[..., 0]).sum())
        for key, value in arrays.items()
        if _MATERIAL_PATH_KEY.match(key)
    )
