"""Strict closed-hexagon, two-track honeycomb topology.

The geometry is deliberately different from an Euler trail over a thin
three-way wall graph.  Each void is represented by a regular, closed hexagon
whose centreline is printed once.  Two neighbouring void loops supply the two
parallel bead tracks of their shared web, so no physical centreline is ever
deposited twice.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class ClosedHexagonHoneycombPattern:
    """Regular closed hexagon loops fitted inside a rectangular envelope."""

    loops: tuple[np.ndarray, ...]
    cell_count: int
    row_count: int
    nominal_cell_side_mm: float
    centreline_hex_side_mm: float
    wall_track_spacing_mm: float
    wall_track_overlap_mm: float


def build_closed_hexagon_double_wall_honeycomb(
    bounds: tuple[float, float, float, float],
    *,
    cell_side_mm: float,
    line_width_mm: float,
) -> ClosedHexagonHoneycombPattern:
    """Build regular, individually closed honeycomb void boundaries.

    ``cell_side_mm`` is the side of the source honeycomb lattice measured to
    its ideal three-wall junction.  The generated loop is inset by half a
    nominal bead radius at every vertex.  Consequently, two adjacent loops
    form a two-track wall.  A small, intentional overlap (about 13.4% for a
    pointy hexagon) closes the three-way junction rather than leaving a pin
    hole there; it is geometric bead overlap, *not* a repeated deposition.
    """

    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    side = float(cell_side_mm)
    width = float(line_width_mm)
    if not math.isfinite(side) or side <= 0:
        raise ValueError("closed-hexagon honeycomb cell side must be positive")
    if not math.isfinite(width) or width <= 0:
        raise ValueError("closed-hexagon honeycomb line width must be positive")
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("closed-hexagon honeycomb bounds must have positive area")

    # For a pointy regular hexagon, bringing a vertex within one bead radius
    # of the ideal lattice junction closes the meeting of three printed walls.
    # It also leaves two centre lines across a shared rib, separated by
    # sqrt(3) * width / 2.  This is a controlled 13.4% overlap for equal
    # width beads and avoids both a gap at the junction and any retraced path.
    centreline_side = side - width * 0.5
    if centreline_side <= width * 0.5:
        raise ValueError(
            "closed-hexagon honeycomb cell side is too small for two wall tracks"
        )
    half_width = math.sqrt(3.0) * centreline_side / 2.0
    horizontal_pitch = math.sqrt(3.0) * side
    vertical_pitch = 1.5 * side
    edge_clearance = max(width * 1.1, side * 0.4)

    rows: list[list[tuple[float, float]]] = []
    row_index = 0
    center_y = min_y + edge_clearance + centreline_side
    while center_y + centreline_side <= max_y - edge_clearance + 1e-8:
        centers: list[tuple[float, float]] = []
        center_x = (
            min_x
            + edge_clearance
            + half_width
            + (horizontal_pitch * 0.5 if row_index % 2 else 0.0)
        )
        while center_x + half_width <= max_x - edge_clearance + 1e-8:
            centers.append((center_x, center_y))
            center_x += horizontal_pitch
        if centers:
            rows.append(centers)
        center_y += vertical_pitch
        row_index += 1
    if not rows:
        raise ValueError("frame is too small for the requested closed honeycomb")

    loops = tuple(
        _closed_pointy_hexagon(center_x, center_y, centreline_side)
        for centers in rows
        for center_x, center_y in centers
    )
    track_spacing = math.sqrt(3.0) * (side - centreline_side)
    return ClosedHexagonHoneycombPattern(
        loops=loops,
        cell_count=len(loops),
        row_count=len(rows),
        nominal_cell_side_mm=side,
        centreline_hex_side_mm=centreline_side,
        wall_track_spacing_mm=track_spacing,
        wall_track_overlap_mm=max(0.0, width - track_spacing),
    )


def _closed_pointy_hexagon(center_x: float, center_y: float, side: float) -> np.ndarray:
    """Return one exact regular pointy hexagon, including its closing point."""

    points = np.asarray(
        [
            (
                center_x + side * math.cos(math.pi / 2.0 + index * math.pi / 3.0),
                center_y + side * math.sin(math.pi / 2.0 + index * math.pi / 3.0),
            )
            for index in range(6)
        ],
        dtype=np.float64,
    )
    return np.vstack((points, points[0]))
