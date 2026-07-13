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
- Independent path emission for the path-only output contract. Resin paths are
  not globally reordered or joined with travel/connector segments.

The migration should not initially replace:

- CLI/UI request handling.
- NPZ serialization.
- Fiber JSON ingestion.
- Raft insertion and Z shifting.
- Preview payload format.

Algorithms that rely on extrusion quantity, variable-width bead accounting,
volumetric flow, support material accounting, or G-code-only semantics should be
skipped or reduced to path-centerline geometry before being adapted.

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
- No migration phase changes public CLI/UI options unless explicitly planned.
