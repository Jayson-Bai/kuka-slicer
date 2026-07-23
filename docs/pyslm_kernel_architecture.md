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
- `zigzag` section geometry and contours through PySLM, with project-owned
  boundary-following one-stroke hatch connection
- `isotropic` section geometry and contours through PySLM, with the same
  project-owned zigzag core and a fixed `+45, 0, -45, 90` degree layer cycle

The following patterns remain available only through the standalone legacy
kernel:

- `grid`
- `triangles`
- `gyroid`
- `concentric`
- one-stroke connector semantics and NPZ chord-error sampling remain project-owned

The PySLM adapter fails fast for the four legacy-only patterns instead of
silently mixing algorithms. This keeps the two kernels independently
selectable and makes output comparisons meaningful.

Native PySLM controls are grouped under `SliceConfig.pyslm` and include the
hatcher strategy, hatch angle, layer angle increment, hatch spacing, contour
offsets, spot compensation, volume offset, contour counts, scan ordering,
stripe/island dimensions, polygon repair, and boundary simplification. The
supported simplification modes for pinned PythonSLM 0.6.1 are `absolute` and
boundary-scaled `bound`; its advertised `line` branch raises
`NotImplementedError` and is not exposed.

The hatcher strategy is intentionally separate from the project's
`infill_pattern`: the hatcher selects PySLM's scan organization, while the
pattern selects native or project-owned fill algorithms. The web UI keeps the
shared pattern selector visible for both kernels. Stripe and island dimensions
have scale-aware UI/CLI defaults from `recommended_pyslm_strategy_defaults()`;
for the default resin process (0.5 mm layer height, 2 mm line width), these
are 10 mm width, 0.1 mm overlap, and a 0.5 hatch-spacing offset. The UI keeps
the controls collapsed and permits manual overrides by disabling automatic
mode.

The default PySLM hatch-volume offset is measured from the innermost contour
centerline and therefore uses the full bead-aware centerline pitch. With the
default 2 mm line width and 10% overlap, the contour-to-hatch distance is
1.8 mm; this avoids treating the physical line radius as free space and
overfilling the wall-adjacent strip.

Project-owned `zigzag` and `isotropic` use the same physical infill corridor as
the Prusa planner. Explicit PySLM contour offsets, spot compensation, hatch
volume offset, or contour-count overrides are rejected for these two patterns;
otherwise PySLM's emitted innermost contour and the independently generated
infill could disagree and violate the centerline pitch.

Native PySLM patterns may retain contour and hatch-distance overrides on
ordinary middle layers. The fixed top and bottom caps are project-owned
zigzags, so those cap layers reset the overrides to bead-aware defaults and
remain full density. When an isotropic slice uses explicit `z_min`/`z_max`,
`z_min` is the lower print bound and the first layer plane is one configured
first-layer height above it (0.5 mm by default), preserving the required
`4N+2` schedule when the configured layer schedule is valid.

## Native Visualization Boundary

PySLM provides `pyslm.visualise.plot()` and related Matplotlib helpers for
plotting contours, hatch vectors, point exposures, 3D layer position, arrows,
scan order, and color mapping. The current project UI keeps its browser canvas
preview and displays the same project-owned path arrays that are written to
NPZ; it does not embed a Matplotlib figure. This keeps preview and export on
the same `ExternalSourceJob` data contract.
