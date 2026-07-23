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
layer_0001_R
layer_0001_F
```

Keys must match:

```text
^layer_(\d{4})_([RF])$
```

`R` means resin and `F` means fiber. `meta` is optional and stores a JSON
string.

## Path Array Shape

Arrays are numeric `float32` tensors:

```text
[path_count, max_points_per_path, columns]
```

`columns` is `3` for `[x, y, z]` or `6` for `[x, y, z, a, b, c]`. Short paths
are padded with full `NaN` rows.

## Padding

A row whose every column is `NaN` is padding. A row with only some `NaN` values
is invalid for downstream loading.

## Path Sampling

Exported paths use a `0.05 mm` three-dimensional chord-error tolerance. A
collinear run contains only its start and end rows. Curves and closed contours
retain the minimum sampled vertices needed to preserve their XYZ shape within
that tolerance. Simplification never changes path order, path count, or path
continuity.

## Z Ownership

The source NPZ explicitly owns Z. This slicer writes the Z values that should be
used by later processing. Downstream UI layer-height fields are process
parameters and must not overwrite trajectory Z.
