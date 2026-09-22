"""Brim generation for the rectangular planar and double-sine workflows.

The conformal workflow does not go through Prusa's STL slicer.  Its Brim must
therefore be added to the exact ``ExternalSourceJob`` which is handed to Core,
rather than being a UI-only Prusa option.  The first resin-layer outer
boundary is the authoritative part footprint for both planar and double-sine
designs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ..external_npz import ExternalSourceJob, MaterialPaths


_SUPPORTED_BRIM_TYPES = frozenset({"outer_only", "outer_and_inner", "no_brim"})


@dataclass(frozen=True, slots=True)
class ConformalBrimSettings:
    """Explicit settings shared by the main UI and the Core hand-off."""

    enabled: bool
    width_mm: float
    brim_type: str
    separation_mm: float
    one_stroke: bool
    line_width_mm: float
    tolerance_mm: float = 1e-5

    def __post_init__(self) -> None:
        if self.brim_type not in _SUPPORTED_BRIM_TYPES:
            raise ValueError(f"unsupported brim type: {self.brim_type!r}")
        for name, value in (
            ("width_mm", self.width_mm),
            ("separation_mm", self.separation_mm),
            ("line_width_mm", self.line_width_mm),
            ("tolerance_mm", self.tolerance_mm),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.line_width_mm <= 0.0 or self.tolerance_mm <= 0.0:
            raise ValueError("line_width_mm and tolerance_mm must be positive")


def apply_conformal_brim(
    job: ExternalSourceJob,
    settings: ConformalBrimSettings,
) -> dict[str, object]:
    """Prepend generated Brim paths to the first conformal resin layer.

    The output paths carry the same XYZABC boundary pose field and E/mm
    calibration as the source layer.  This makes the Core NPZ and its final
    preview authoritative: there is no separate display-only Brim geometry.
    """

    report: dict[str, object] = {
        "enabled": bool(settings.enabled and settings.brim_type != "no_brim"),
        "width_mm": float(settings.width_mm),
        "type": settings.brim_type,
        "separation_mm": float(settings.separation_mm),
        "one_stroke_requested": bool(settings.one_stroke),
        "line_width_mm": float(settings.line_width_mm),
        "path_count": 0,
        "one_stroke_applied": False,
    }
    if not report["enabled"] or settings.width_mm <= settings.tolerance_mm:
        _record_report(job, report)
        return report

    group, roles, outer_index = _first_outer_boundary(job)
    boundary = np.asarray(group.paths[outer_index], dtype=np.float64)
    rings_xy = _outer_brim_rings(boundary[:, :2], settings)
    if not rings_xy:
        _record_report(job, report)
        return report

    if settings.one_stroke and len(rings_xy) > 1:
        # Reuse the Prusa Brim safety connector.  It returns several paths
        # when connecting a component would leave the generated Brim corridor.
        from ..slicer import _connect_brim_paths_one_stroke

        connected = _connect_brim_paths_one_stroke(
            rings_xy,
            settings.line_width_mm,
            max(settings.tolerance_mm, 1e-6),
        )
        if connected and len(connected) < len(rings_xy):
            rings_xy = [np.asarray(path, dtype=np.float64) for path in connected]
            report["one_stroke_applied"] = len(rings_xy) == 1
            report["one_stroke_strategy"] = "prusa_safe_boundary_connector"
        elif len(_outer_spiral(rings_xy)) >= 2:
            # The rectangular conformal footprint is one exterior polygon.
            # Its nested offset rings can be joined radially inside the Brim
            # band without crossing the part or its separation gap.  This is
            # the deterministic equivalent of Prusa's single-stroke Brim
            # fallback when the generic boundary connector declines closed
            # loops.
            rings_xy = [_outer_spiral(rings_xy)]
            report["one_stroke_applied"] = True
            report["one_stroke_strategy"] = "conformal_outer_spiral"

    brim_paths = [_lift_xy_to_boundary(path, boundary) for path in rings_xy]
    extrusion_rate = _boundary_extrusion_rate(group, outer_index, boundary)
    brim_extrusion = [_extrusion_profile(path, extrusion_rate) for path in brim_paths]

    group.paths[outer_index:outer_index] = brim_paths
    if group.extrusion is None:
        group.extrusion = [
            _extrusion_profile(np.asarray(path, dtype=np.float64), extrusion_rate)
            for path in group.paths
        ]
    else:
        group.extrusion[outer_index:outer_index] = brim_extrusion
    roles[outer_index:outer_index] = ["brim"] * len(brim_paths)

    report.update(
        path_count=len(brim_paths),
        lane_count=_brim_lane_count(settings),
        one_stroke_applied=bool(report["one_stroke_applied"] or len(brim_paths) == 1),
        source_layer=int(group.layer_index),
        source_role="conformal_outer_boundary",
        geometry="outer-footprint offsets; XYZABC is projected from first-layer conformal boundary",
    )
    _record_report(job, report)
    return report


def _first_outer_boundary(
    job: ExternalSourceJob,
) -> tuple[MaterialPaths, list[str], int]:
    roles_root = job.meta.get("path_roles")
    if not isinstance(roles_root, dict) or not isinstance(roles_root.get("R"), dict):
        raise ValueError("conformal source job is missing resin path-role metadata")
    resin_roles = roles_root["R"]
    groups = sorted(
        (group for group in job.material_paths if group.material == "R" and group.paths),
        key=lambda group: int(group.layer_index),
    )
    for group in groups:
        roles = resin_roles.get(str(group.layer_index))
        if not isinstance(roles, list) or len(roles) != len(group.paths):
            raise ValueError("conformal source job has inconsistent resin path roles")
        for index, role in enumerate(roles):
            if role == "conformal_outer_boundary":
                return group, roles, index
    raise ValueError("conformal source job has no outer boundary for Brim generation")


def _outer_brim_rings(boundary_xy: np.ndarray, settings: ConformalBrimSettings) -> list[np.ndarray]:
    boundary = np.asarray(boundary_xy, dtype=np.float64)
    if boundary.ndim != 2 or boundary.shape[1] != 2 or len(boundary) < 4:
        raise ValueError("conformal outer boundary must be a closed XY polygon")
    polygon = Polygon(boundary)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise ValueError("conformal outer boundary cannot form a valid Brim footprint")

    result: list[np.ndarray] = []
    lane_count = _brim_lane_count(settings)
    lane_width = settings.width_mm / lane_count
    for lane in range(lane_count):
        offset = settings.separation_mm + (lane + 0.5) * lane_width
        expanded = polygon.buffer(offset, join_style="round")
        geometries = (
            list(expanded.geoms)
            if isinstance(expanded, MultiPolygon)
            else [expanded]
        )
        for geometry in geometries:
            if geometry.is_empty:
                continue
            coords = np.asarray(geometry.exterior.coords, dtype=np.float64)
            if len(coords) >= 4:
                result.append(coords)
    return result


def _brim_lane_count(settings: ConformalBrimSettings) -> int:
    return max(1, int(math.ceil(settings.width_mm / settings.line_width_mm)))


def _outer_spiral(rings_xy: list[np.ndarray]) -> np.ndarray:
    """Join nested exterior rings into one safe, outward-to-inward stroke."""

    rings = [
        np.asarray(ring, dtype=np.float64)
        for ring in reversed(rings_xy)
        if np.asarray(ring).ndim == 2 and len(ring) >= 4
    ]
    if not rings:
        return np.empty((0, 2), dtype=np.float64)

    stitched: list[np.ndarray] = []
    previous_end: np.ndarray | None = None
    for ring in rings:
        # Shapely exterior coordinates are closed.  Rotate the open portion
        # to the vertex nearest the preceding loop, then re-close it before
        # adding the radial link to the next, inner ring.
        open_ring = ring[:-1]
        start_index = (
            0
            if previous_end is None
            else int(np.argmin(np.linalg.norm(open_ring - previous_end, axis=1)))
        )
        rotated = np.vstack((open_ring[start_index:], open_ring[:start_index + 1]))
        if stitched:
            stitched.append(rotated[0])
        stitched.extend(rotated)
        previous_end = rotated[-1]
    return np.asarray(stitched, dtype=np.float64)


def _lift_xy_to_boundary(path_xy: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Attach XYZABC by nearest-segment projection onto a conformal boundary."""

    points = np.asarray(path_xy, dtype=np.float64)
    reference = np.asarray(boundary, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[1] < 6:
        raise ValueError("conformal boundary must provide XYZABC poses")
    starts = reference[:-1, :]
    ends = reference[1:, :]
    vectors = ends[:, :2] - starts[:, :2]
    squared_lengths = np.einsum("ij,ij->i", vectors, vectors)
    result = np.empty((len(points), 6), dtype=np.float64)
    for point_index, point in enumerate(points):
        relative = point - starts[:, :2]
        parameter = np.divide(
            np.einsum("ij,ij->i", relative, vectors),
            squared_lengths,
            out=np.zeros_like(squared_lengths),
            where=squared_lengths > 1e-12,
        )
        parameter = np.clip(parameter, 0.0, 1.0)
        projected = starts[:, :2] + vectors * parameter[:, None]
        nearest = int(np.argmin(np.einsum("ij,ij->i", projected - point, projected - point)))
        result[point_index, :2] = point
        result[point_index, 2:] = (
            starts[nearest, 2:] * (1.0 - parameter[nearest])
            + ends[nearest, 2:] * parameter[nearest]
        )
    return result


def _boundary_extrusion_rate(
    group: MaterialPaths,
    path_index: int,
    boundary: np.ndarray,
) -> float:
    if group.extrusion is not None and path_index < len(group.extrusion):
        extrusion = np.asarray(group.extrusion[path_index], dtype=np.float64)
        length = float(np.linalg.norm(np.diff(boundary[:, :3], axis=0), axis=1).sum())
        if len(extrusion) == len(boundary) and length > 1e-9:
            delta = float(extrusion[-1] - extrusion[0])
            if math.isfinite(delta) and delta > 0.0:
                return delta / length
    raise ValueError("conformal outer boundary must provide a positive calibrated extrusion profile")


def _extrusion_profile(path: np.ndarray, rate: float) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(np.asarray(path, dtype=np.float64)[:, :3], axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths * rate)))


def _record_report(job: ExternalSourceJob, report: dict[str, object]) -> None:
    slicing = job.meta.setdefault("slicing", {})
    if not isinstance(slicing, dict):
        raise ValueError("conformal source job slicing metadata must be an object")
    slicing["conformal_brim"] = report
