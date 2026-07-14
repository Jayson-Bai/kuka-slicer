# PySLM Kernel Architecture

## Versioning Policy

- `main` remains the stable release line using the `legacy` kernel by default.
- Experimental PySLM work is isolated behind `SliceConfig.slicing_kernel`.
- The CLI and web UI default to `legacy`; both expose an explicit kernel switch.
- The checkpoint before this branch is tagged as
  `checkpoint-before-pyslm-20260713-2145`.
- The PySLM dependency is optional under `.[pyslm]`; baseline installs and tests
  must not require it.

## Boundary

The stable project contract is unchanged:

- `ExternalSourceJob` remains the internal handoff object.
- NPZ output keeps the documented `layer_0000_R/F` keys.
- KUKA-specific material grouping, fiber expansion, raft insertion, build-axis
  remapping, XY normalization, and curved-Z projection stay outside PySLM.

PySLM is wrapped only as a geometry/toolpath provider:

- mesh sectioning through `Part.getVectorSlice()`;
- contour offsets and straight hatch vectors through `Hatcher`;
- conversion from PySLM `LayerGeometry` objects back into project-owned
  `float32` path arrays.

## Current PySLM Adapter Scope

PySLM-native path generation:

- `none`
- `line`
- `aligned_rectilinear`
- `rectilinear`
- cap-layer `zigzag` as hatch vectors

The following patterns remain available only through the standalone legacy
kernel:

- `grid`
- `triangles`
- `gyroid`
- `concentric`
- legacy smoothing and connector semantics remain project-owned
- full output parity for cap-layer zigzag connectivity

The PySLM adapter fails fast for the four legacy-only patterns instead of
silently mixing algorithms. This keeps the two kernels independently
selectable and makes output comparisons meaningful.

Native PySLM controls are grouped under `SliceConfig.pyslm` and include the
hatcher strategy, hatch angle, layer angle increment, hatch spacing, contour
offsets, spot compensation, volume offset, contour counts, scan ordering,
stripe/island dimensions, polygon repair, and boundary simplification.

The hatcher strategy is intentionally separate from the project's legacy
`infill_pattern`: the hatcher selects PySLM's scan organization, while the
legacy pattern selects project-owned fill algorithms. The web UI hides the
legacy pattern selector while PySLM is active. Stripe and island dimensions
have scale-aware UI/CLI defaults from `recommended_pyslm_strategy_defaults()`;
for the default resin process (0.5 mm layer height, 2 mm line width), these
are 10 mm width, 0.1 mm overlap, and a 0.5 hatch-spacing offset. The UI keeps
the controls collapsed and permits manual overrides by disabling automatic
mode.

## Native Visualization Boundary

PySLM provides `pyslm.visualise.plot()` and related Matplotlib helpers for
plotting contours, hatch vectors, point exposures, 3D layer position, arrows,
scan order, and color mapping. The current project UI keeps its browser canvas
preview and displays the same project-owned path arrays that are written to
NPZ; it does not embed a Matplotlib figure. This keeps preview and export on
the same `ExternalSourceJob` data contract.
