# Surface preview package

This package is the isolated first stage of the graded-curvature workflow.
It is intentionally separate from `kuka_slicer.slicer`, `prusa_backend`, and
the integrated slicer UI.

```text
model.py   -> pure double-sine height field and geometry diagnostics
server.py  -> local HTTP API and self-contained interactive browser shell
cli.py     -> `kuka-slicer surface-preview` entry point
```

The current version previews `graded_surface_v1`'s
`double_sine_product` surface and can import a local STL as its XY material
domain.  The preview extracts the same union of printable XY sections used by
the slicer, moves the local origin to `stl_xy_min`, clips the rendered surface
to that domain, and preserves its openings in the preview.

Use **导出曲面指导 JSON** after importing the final STL.  The downloaded
`graded_surface_v1.json` stores the surface parameters, progression and
printability limits, plus the source STL name, SHA-256 fingerprint, source
build axis and XY bounds.  It is the sidecar input for the future post-Prusa
surface mapper; it is not an STL replacement and does not yet change
toolpaths, calculate E, or write NPZ output.

Future integration should reuse `DoubleSineSurface`, `mesh_xy_projection()`
and the exported JSON:

1. call the same height field from the post-Prusa surface mapper;
2. validate the STL fingerprint/domain before mapping;
3. embed `surface_preview_html()` or request `/api/surface` from the main UI.

Keeping the mathematical model and the browser adapter here prevents those
future steps from coupling a reusable surface definition to the slicer core.
