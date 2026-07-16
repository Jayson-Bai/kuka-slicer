# PrusaSlicer Core Migration Plan

This project keeps its KUKA-specific input/output contract while incrementally
adopting algorithms from the PrusaSlicer/libslic3r family.

## Non-Negotiable Local Interfaces

- Keep `ExternalSourceJob` as the internal handoff object.
- Keep external source NPZ keys in the documented `layer_0000_R/F` format.
- Keep explicit trajectory Z ownership, including non-flat Z projection.
- Keep resin/fiber material grouping and fiber template expansion.
- Keep raft generation as a first-class project feature.
- Export path centerlines only; do not add extrusion amount, volumetric flow,
  pressure, or bead-width columns unless the external NPZ contract changes.

## Migration Scope

The initial migration should target geometry and toolpath quality only:

- Slice contours and polygon repair.
- Perimeter offset generation.
- Infill region generation.
- Infill overlap handling.
- Bead-aware continuity planning for the path-only output contract. Connectors
  are emitted only as printable centerlines; there is no zero-material travel
  or extrusion-flow channel in the NPZ format.

The migration should not initially replace:

- CLI/UI request handling.
- NPZ serialization.
- Fiber JSON ingestion.
- Raft insertion and Z shifting.
- Preview payload format.

Algorithms that rely on extrusion quantity, variable-width bead accounting,
volumetric flow, support material accounting, or G-code-only semantics should be
skipped or reduced to path-centerline geometry before being adapted.

## Current Prusa-Kernel Path Invariants

### Bead width and overlap

- Centerline placement accounts for the full physical line width first and
  applies overlap second. The base pitch is
  `line_width * (1 - overlap_fraction)`; the default `2.0 mm` line and `10%`
  overlap therefore produce a `1.8 mm` pitch.
- The infill corridor starts from the physical inner edge of the innermost
  perimeter bead, moves inward by half a bead to place a centerline, then moves
  back toward that bead by only the requested overlap.
  The innermost perimeter centerline and the first infill or boundary-connector
  centerline consequently remain one base pitch apart.
- Do not replace the physical half-line-width term with half of an
  overlap-reduced spacing. That counts overlap twice and can produce a
  wall-parallel strip of excess material.
- Printed continuity connectors use the same bead-aware clearance model as the
  infill itself. Clearance from every non-incident path or connector is at
  least the base pitch; only the two incident strokes intentionally share a
  footprint at their common turn.

### Density and material allocation

- One-axis spacing is `base_pitch / density_fraction`.
- `concentric` uses that same density-adjusted spacing between rings; non-zero
  density is not silently treated as 100% density.
- Multi-axis patterns allocate one total material-length budget across their
  directions. With `N` directions, spacing is
  `base_pitch * N / density_fraction`; `grid` uses `N = 2` and `triangles` uses
  `N = 3`.
- `gyroid` uses the density-adjusted one-axis spacing and a calibrated
  wavelength factor of `2.35`.

### Continuity policy by pattern

- Single-axis patterns (`rectilinear`, `aligned_rectilinear`, `line`, and
  `zigzag`) connect adjacent scanlines with the shortest valid direct or
  boundary-following link inside the safe infill corridor.
- `grid` and `triangles` node all real crossings, remove duplicate edges, and
  emit a minimum edge-disjoint trail cover. Euler-only virtual edges split the
  walk but are never printed, so a real lattice edge is not retraced merely to
  obtain one path.
- `gyroid` may connect clipped curve trails by safe boundary links.
- `concentric` may reroot and connect adjacent rings so that each ring is
  printed continuously and without retracing.
- A connector must remain inside the bead-aware corridor and must not cross or
  crowd another non-incident printed centerline. When those checks fail, the
  output remains split. Print safety and pile-up avoidance take priority over a
  forced one-stroke path.

## Phases

1. Establish baseline fixtures for current resin and fiber behavior.
2. Add adapter seams around contour, perimeter, and infill generation.
3. Port or reimplement libslic3r-style perimeter and overlap behavior behind
   the existing `SliceConfig` API.
4. Port selected infill generators one pattern at a time.
5. Compare output against existing tests and selected real STL fixtures.
6. Remove duplicate legacy helpers only after behavior parity is proven.

## Acceptance Criteria

- Existing tests pass before and after each phase.
- Existing NPZ consumers can load generated files without format changes.
- New behavior is gated by tests for holes, concave regions, overlap spacing,
  thin features, and path self-intersections.
- The default overlap fixture verifies a `1.8 mm` centerline pitch for a
  `2.0 mm` line at `10%` overlap, including the last-perimeter-to-infill and
  last-perimeter-to-boundary-connector relationships.
- Density fixtures verify total material-length allocation for multi-axis
  patterns, the `2.35` gyroid calibration, and density-dependent concentric
  ring spacing.
- Continuity fixtures verify minimum graph trails or safe pattern-specific
  links without retraced edges, unsafe boundary excursions, or sub-pitch
  clearance to non-incident printed paths.
- No migration phase changes public CLI/UI options unless explicitly planned.
