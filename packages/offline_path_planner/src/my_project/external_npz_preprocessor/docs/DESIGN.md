# External NPZ Preprocessor Design

## Goal

This package converts an agreed external source NPZ into the system NPZ format consumed by `control_center`, while always going through `path_processing_core.npz_exporter.export_npz()`. The source NPZ carries trajectory geometry, including explicit Z; UI layer-height values are process references only.

## Package Boundary

- `source_npz.py`: reads and validates the external source NPZ contract.
- `process_params.py`: defines material process parameters and derives extrusion.
- `converter.py`: converts layer/material paths into `path_processing_core.types` parsed commands.
- `export_runner.py`: connects reader, converter, and `npz_exporter`.
- `cli.py`: batch conversion entry point.
- `ui.py` / `app.py`: simple desktop UI for selecting paths and parameters.

The package is intentionally separate from `gcode_planner`. It depends on `path_processing_core`, while `gcode_planner` keeps the GCode-specific input path, preview helpers, and legacy import wrappers. The formal-print UI can route either `.npz` source files to this package or `.gcode/.gc/.g` files to the GCode adapter; both converge in `path_processing_core`.

## Data Flow

```text
external source NPZ
-> load_source_npz()
-> SourceJob / LayerPaths / MaterialPath
-> source_job_to_parsed_commands()
-> MoveCommand / ToolChangeCommand / ResetECommand / MCommand
-> path_processing_core.npz_exporter.export_npz()
-> system-compatible NPZ
```

## Current Scope

The package supports:

- `layer_0000_R` / `layer_0000_F` source keys.
- Per-key numeric `float32` arrays shaped `[path_count, max_points, columns]`; short paths use all-NaN padding rows.
- Nx3 `[x, y, z]` source paths as the required base format.
- Nx6 `[x, y, z, a, b, c]` source paths when the source needs explicit pose values.
- Source-side Z is trajectory geometry and is never overwritten by UI layer-height values.
- Formal-print compatible input supports both agreed source `.npz` and legacy `.gcode/.gc/.g` files; this package owns only the `.npz` branch.
- Resin extrusion derived from fixed `2.0 mm` resin line width, resin layer height, resin extrusion scale, and fixed `1.75 mm` filament cross-section conversion.
- Fiber extrusion matched to path length by default through fiber extrusion scale.
- Fiber print curves may carry a curve-level seven-order start acceleration time; this is external-NPZ fiber-only metadata and does not change the global interpolator default.
- A basic Qt UI and CLI.

