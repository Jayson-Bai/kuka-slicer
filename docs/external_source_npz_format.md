# External Source NPZ Format

This repository writes the external source NPZ format consumed by:

```text
external_npz_preprocessor.source_npz.load_source_npz()
```

The produced archive is distinct from the system runtime NPZ written later by
`path_processing_core.npz_exporter.export_npz()`.

## Archive Keys

Layer/material arrays use keys:

```text
layer_0000_R
layer_0000_F
layer_0000_T
layer_0001_R
layer_0001_F
layer_0001_T
```

Keys must match:

```text
^layer_(\d{4})_([RFT])$
```

`R` means resin, `F` means fiber, and `T` means non-depositing travel paths
for that layer. `meta` is optional and stores a JSON string.

When `T` is present, `meta.motion_order` records the source resin travel and
deposit order. A Prusa-integrated source may also set
`meta.startup_travel_count` to identify origin-to-first-motion travel paths
at the beginning of the first layer. The loader keeps these paths separate
from material paths so the Core can use Prusa resin travel without changing
the legacy R/F-only fallback behavior.

For resin paths, an optional companion key may provide the cumulative Prusa
extrusion values:

```text
layer_0000_R_E
```

The `_E` array has the same path and point dimensions as its matching `R`
array, and contains one finite value per valid XYZ point. Padding positions
must be `NaN`. Values are interpreted as cumulative source E, so they must be
non-decreasing within each path. The converter subtracts the first value of
each path before handing it to the runtime; this removes only the global
offset needed by the runtime's per-path reset and preserves every local E
increment. If `_E` is absent, the existing uniform `e_per_mm` calculation is
used unchanged.

## Path Array Shape

Formal high-precision source arrays are numeric `float64` (`<f8`) tensors:

```text
[path_count, max_points_per_path, columns]
```

`columns` is `3` for `[x, y, z]` or `6` for `[x, y, z, a, b, c]`. Short paths
are padded with full `NaN` rows. Legacy `float32` files remain readable, but
their lost source precision cannot be restored.

## Padding

A row whose every column is `NaN` is padding. A row with only some `NaN` values
is invalid for downstream loading.

## Path Preservation

The source writer does not simplify, smooth, resample, reorder, merge, or split
paths. It preserves the original path order and point order. Smoothing and
seven-order sampling happen only in the downstream Core processing pipeline.

## Z Ownership

The source NPZ explicitly owns Z. This slicer writes the Z values that should be
used by later processing. Downstream UI layer-height fields are process
parameters and must not overwrite trajectory Z.
