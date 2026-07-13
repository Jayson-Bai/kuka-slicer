# KUKA Surface Slicer

Python tooling for converting STL geometry into the external source NPZ format
expected by `external_npz_preprocessor.source_npz.load_source_npz()`.

The first supported pipeline is:

```text
STL mesh -> sliced layer paths -> external source NPZ
```

The generated NPZ uses numeric `float32` arrays with shape:

```text
[path_count, max_points_per_path, columns]
```

where `columns` is currently `3` (`x, y, z`). Padding rows are full `NaN`
rows.

## CLI

```powershell
python -m kuka_slicer slice input.stl output.npz
```

Default process parameters are currently hard-coded:

| Material | Layer height | Line width |
| --- | ---: | ---: |
| Resin `R` | `0.5 mm` | `2.0 mm` |
| Fiber `F` | `0.1 mm` | `1.0 mm` |

Resin infill uses a Cura-style default overlap of `10%`. With a `2.0 mm`
resin line width, the default 100% line infill center spacing is therefore
`1.8 mm`, and infill is allowed to overlap the innermost perimeter by
`0.2 mm`. Set `--infill-overlap 0` to restore geometric line-width spacing.

Useful options:

```powershell
python -m kuka_slicer slice input.stl output.npz `
  --layer-height 0.5 `
  --line-width 2.0 `
  --build-axis y `
  --infill-pattern lines_x `
  --infill-density 100 `
  --infill-overlap 10 `
  --z-min 0.2 `
  --z-max 10.0 `
  --material R `
  --curve sinusoidal `
  --curve-amplitude 0.2 `
  --curve-period 40.0
```

Available resin infill patterns:

| Pattern | Meaning |
| --- | --- |
| `lines_x` | Straight lines parallel to X |
| `lines_y` | Straight lines parallel to Y |
| `grid` | X/Y grid |
| `triangles` | Three-direction triangular grid |
| `gyroid` | Continuous gyroid-like isotropic curve fill |
| `diagonal` | 45 degree straight lines |
| `alternating_diagonal` | Alternating +45/-45 degree lines by layer |
| `contour_offset` | Concentric offset fill inside the perimeters |
| `contour` | Slice contours only |

`--infill-density` is a resin fill percentage from `0` to `100`. It controls
the generated path spacing together with resin line width and
`--infill-overlap`.
For `triangles`, density is converted across the three lattice directions, so
lower densities produce larger triangles rather than three over-dense line sets.
`gyroid` uses continuous clipped contour curves, which usually reduces resin
path start/stop count at high densities while keeping a more balanced direction
distribution than one-direction line fill.

`--build-axis` selects the STL source axis used as the layer-height direction.
For a disk/cylinder whose round face is in the `X-Z` plane and thickness is
along `Y`, use `--build-axis y`.

## UI

```powershell
python -m kuka_slicer ui
```

Open:

```text
http://127.0.0.1:8765
```

Generate the documented two-layer resin/fiber template:

```powershell
python -m kuka_slicer make-template data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz
```

Run tests:

```powershell
python -m pytest
```
