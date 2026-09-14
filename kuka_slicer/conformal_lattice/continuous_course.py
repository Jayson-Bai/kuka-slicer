"""Analytic continuous-course planning shared by the designer and Core hand-off.

The yellow honeycomb is a pore reference, not a deposited wall graph.  This
module produces the red X-progressing centre-lines in an oversized, centre
anchored parent lattice, then clips those lines to the actual working region.
It deliberately contains no graph search, nearest-edge snapping, or synthetic
boundary connector.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .contracts import ConformalLatticeSpec
from .layer_embedding import LayerEmbedding
from ..surface_preview.model import DoubleSineSurface


_TOW_WIDTH_MM = 2.0
_PARENT_HALF_SPAN_MM = 150.0
_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class ContinuousCoursePlan:
    """Immutable planar red-course fragments before any surface embedding.

    ``paths_xy`` deliberately contains printable clipped fragments, not their
    parent courses.  A boundary graze consisting only of a diagonal tip is
    not a channel: a retained fragment must include a finite horizontal
    pore-to-pore support.  Each retained item therefore becomes one
    independent SourceJob path and receives the normal Core travel/cut
    boundary at both ends.
    """

    paths_xy: tuple[np.ndarray, ...]
    course_bounds_mm: tuple[float, float, float, float]
    pore_clip_bounds_mm: tuple[float, float, float, float]
    parent_bounds_mm: tuple[float, float, float, float]
    lattice_anchor_mm: tuple[float, float]
    edge_length_mm: float
    row_pitch_mm: float
    column_pitch_mm: float


def build_continuous_course_plan(spec: ConformalLatticeSpec) -> ContinuousCoursePlan:
    """Build the same complete-parent, clipped X courses used by the preview.

    A complete horizontal opening supplies two directed rails.  Each rail runs
    on one analytic half-pore support, so every inclined 2 mm channel has one
    owner.  We only retain openings whose yellow horizontal boundaries fit
    inside the contour inset, exactly matching the visible-preview rule.
    """

    if not spec.part:
        raise ValueError("continuous course planning requires the rectangular part contract")
    length = float(spec.part["length_mm"])
    width = float(spec.part["width_mm"])
    grip = float(spec.part.get("symmetric_grip_end_length_mm", 0.0))
    if grip < 0.0 or 2.0 * grip >= length:
        raise ValueError("continuous course planning has no valid central working region")
    edge = float(spec.lattice["base_cell_size_mm"])
    contour_width = float(spec.lattice["wall_width_mm"])
    if edge <= 0.0 or contour_width <= 0.0:
        raise ValueError("continuous course planning requires positive cell and contour widths")

    bounds = (grip, 0.0, length - grip, width)
    inset = contour_width * 0.5
    pore_bounds = (bounds[0] + inset, bounds[1] + inset, bounds[2] - inset, bounds[3] - inset)
    if pore_bounds[0] >= pore_bounds[2] or pore_bounds[1] >= pore_bounds[3]:
        raise ValueError("continuous course working region is consumed by contour clearance")

    anchor_x = (bounds[0] + bounds[2]) * 0.5
    anchor_y = (bounds[1] + bounds[3]) * 0.5
    parent_bounds = (
        anchor_x - _PARENT_HALF_SPAN_MM,
        anchor_y - _PARENT_HALF_SPAN_MM,
        anchor_x + _PARENT_HALF_SPAN_MM,
        anchor_y + _PARENT_HALF_SPAN_MM,
    )
    half_height = math.sqrt(3.0) * edge * 0.5
    row_pitch = 2.0 * half_height + 2.0 * _TOW_WIDTH_MM
    side_port_inset_x = _TOW_WIDTH_MM / (2.0 * math.sqrt(3.0))
    diagonal_column_offset = edge * 1.5 + _TOW_WIDTH_MM / math.sqrt(3.0)
    column_pitch = 2.0 * diagonal_column_offset
    row_origin = anchor_y - row_pitch * 0.5
    column_min = math.ceil((parent_bounds[0] - edge - anchor_x) / column_pitch)
    column_max = math.floor((parent_bounds[2] + edge - anchor_x) / column_pitch)
    row_min = math.ceil((parent_bounds[1] - half_height - row_origin) / row_pitch)
    row_max = math.floor((parent_bounds[3] + half_height - row_origin) / row_pitch)

    def half_pore(row: float, side: str) -> np.ndarray:
        centre_y = row_origin + (math.floor(row) * row_pitch + (half_height + _TOW_WIDTH_MM if row % 1 else 0.0))
        vertical = 1.0 if side == "upper" else -1.0
        points: list[tuple[float, float]] = []
        for column in range(column_min - 1, column_max + 2):
            centre_x = anchor_x + column * column_pitch + (diagonal_column_offset if row % 1 else 0.0)
            section = (
                (centre_x - edge - side_port_inset_x, centre_y + vertical * (_TOW_WIDTH_MM * 0.5)),
                (centre_x - edge * 0.5 - side_port_inset_x, centre_y + vertical * (half_height + _TOW_WIDTH_MM * 0.5)),
                (centre_x + edge * 0.5 + side_port_inset_x, centre_y + vertical * (half_height + _TOW_WIDTH_MM * 0.5)),
                (centre_x + edge + side_port_inset_x, centre_y + vertical * (_TOW_WIDTH_MM * 0.5)),
            )
            for point in section:
                if not points or math.dist(points[-1], point) > _EPSILON:
                    points.append(point)
        return _simplify_collinear(np.asarray(points, dtype=np.float64))

    paths: list[np.ndarray] = []
    for lower_row in range(row_min, row_max):
        upper_row = lower_row + 1
        lower_centre_y = row_origin + lower_row * row_pitch
        upper_centre_y = row_origin + upper_row * row_pitch
        gap_lower_y = lower_centre_y + half_height
        gap_upper_y = upper_centre_y - half_height
        if gap_lower_y < pore_bounds[1] - _EPSILON or gap_upper_y > pore_bounds[3] + _EPSILON:
            continue
        interleaved_row = lower_row + 0.5
        for side in ("lower", "upper"):
            fragments = _clip_polyline_to_bounds(half_pore(interleaved_row, side), bounds)
            for fragment in fragments:
                simplified = _simplify_collinear(fragment)
                # A rectangle corner can graze just the final diagonal of a
                # parent route.  That sub-millimetre tip is neither a complete
                # horizontal opening nor a usable one-tow channel.  Keep only
                # fragments containing a finite horizontal support; this
                # yields the intended three upper and three lower boundary
                # courses for the reference layout, rather than four tips.
                if _has_horizontal_channel_support(simplified):
                    paths.append(simplified)
    if not paths:
        raise ValueError("current dimensions cannot contain a printable continuous-course fragment")
    return ContinuousCoursePlan(
        paths_xy=tuple(paths),
        course_bounds_mm=bounds,
        pore_clip_bounds_mm=pore_bounds,
        parent_bounds_mm=parent_bounds,
        lattice_anchor_mm=(anchor_x, anchor_y),
        edge_length_mm=edge,
        row_pitch_mm=row_pitch,
        column_pitch_mm=column_pitch,
    )


def embed_continuous_course_plan(
    plan: ContinuousCoursePlan,
    spec: ConformalLatticeSpec,
    layer_embedding: LayerEmbedding,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]:
    """Map already-final planar courses to every physical layer once."""

    source = spec.source_surface.get("double_sine")
    if not isinstance(source, dict):
        raise ValueError("continuous course embedding requires double-sine source metadata")
    surface = DoubleSineSurface(
        amplitude_mm=float(source["amplitude_mm"]),
        wavelength_x_mm=float(source["wavelength_x_mm"]),
        wavelength_y_mm=float(source["wavelength_y_mm"]),
        phase_x_rad=float(source["phase_x_rad"]),
        phase_y_rad=float(source["phase_y_rad"]),
        z_reference_mm=float(source["z_reference_mm"]),
    )
    alpha = np.asarray(layer_embedding.report.get("alpha_by_layer"), dtype=np.float64)
    base_z = np.asarray(layer_embedding.report.get("base_z_by_layer_mm"), dtype=np.float64)
    if alpha.ndim != 1 or alpha.shape != base_z.shape:
        raise ValueError("continuous course embedding is missing layer alpha/base-Z values")
    result: list[tuple[tuple[np.ndarray, np.ndarray], ...]] = []
    for value_alpha, value_base_z in zip(alpha, base_z):
        layer: list[tuple[np.ndarray, np.ndarray]] = []
        for xy in plan.paths_xy:
            height = np.asarray(surface.height(xy[:, 0], xy[:, 1]), dtype=np.float64)
            dz_dx, dz_dy = surface.gradient(xy[:, 0], xy[:, 1])
            xyz = np.column_stack((xy, value_base_z + value_alpha * (height - surface.z_reference_mm)))
            normals = np.column_stack((-value_alpha * dz_dx, -value_alpha * dz_dy, np.ones(len(xy))))
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            layer.append((xyz, normals))
        result.append(tuple(layer))
    return tuple(result)


def _simplify_collinear(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return points
    result = [points[0]]
    for index in range(1, len(points) - 1):
        before, current, after = result[-1], points[index], points[index + 1]
        cross = (current[0] - before[0]) * (after[1] - current[1]) - (current[1] - before[1]) * (after[0] - current[0])
        if abs(cross) > _EPSILON:
            result.append(current)
    result.append(points[-1])
    return np.asarray(result, dtype=np.float64)


def _has_horizontal_channel_support(points: np.ndarray) -> bool:
    """Whether a clipped route still owns a finite horizontal pore channel."""
    if len(points) < 2:
        return False
    deltas = np.diff(points, axis=0)
    return bool(np.any((np.abs(deltas[:, 1]) <= _EPSILON) & (deltas[:, 0] > _EPSILON)))


def _clip_polyline_to_bounds(points: np.ndarray, bounds: tuple[float, float, float, float]) -> list[np.ndarray]:
    fragments: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    for start, end in zip(points, points[1:]):
        clipped = _clip_segment_to_bounds(start, end, bounds)
        if clipped is None:
            if current:
                fragments.append(current)
                current = []
            continue
        first, second = clipped
        if not current or np.linalg.norm(current[-1] - first) > _EPSILON:
            if current:
                fragments.append(current)
            current = [first]
        current.append(second)
    if current:
        fragments.append(current)
    return [np.asarray(fragment, dtype=np.float64) for fragment in fragments]


def _clip_segment_to_bounds(start: np.ndarray, end: np.ndarray, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray] | None:
    delta = end - start
    lower, upper = 0.0, 1.0
    for p, q in ((-delta[0], start[0] - bounds[0]), (delta[0], bounds[2] - start[0]), (-delta[1], start[1] - bounds[1]), (delta[1], bounds[3] - start[1])):
        if abs(p) <= _EPSILON:
            if q < 0.0:
                return None
            continue
        ratio = q / p
        if p < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    return start + lower * delta, start + upper * delta
