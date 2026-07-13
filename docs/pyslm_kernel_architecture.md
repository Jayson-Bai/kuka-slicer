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

Supported first:

- `none`
- `line`
- `aligned_rectilinear`
- `rectilinear`
- cap-layer `zigzag` as hatch vectors

Not migrated yet:

- `grid`
- `triangles`
- `gyroid`
- `concentric`
- legacy smoothing and connector semantics
- full output parity for cap-layer zigzag connectivity

Unsupported PySLM patterns fail fast instead of silently falling back. This
keeps comparison results explicit and prevents mixed-kernel outputs from being
mistaken for PySLM parity.
