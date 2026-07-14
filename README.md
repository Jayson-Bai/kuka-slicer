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

Default process parameters are:

| Material | Layer height | Line width |
| --- | ---: | ---: |
| Resin `R` | `0.5 mm` | `2.0 mm` |
| Fiber `F` | `0.1 mm` | `1.0 mm` |

Resin infill uses a default overlap of `10%`. With a `2.0 mm` resin line width,
the default 100% line infill center spacing is therefore `1.8 mm`. The infill
surface is offset with the PrusaSlicer-style `overlap - 0.5 * spacing` rule.
Set `--infill-overlap 0` to remove intentional wall overlap.

Useful options:

```powershell
python -m kuka_slicer slice input.stl output.npz `
  --layer-height 0.5 `
  --line-width 2.0 `
  --build-axis y `
  --infill-pattern rectilinear `
  --infill-density 100 `
  --infill-overlap 10 `
  --perimeter-count 2 `
  --smoothing-angle 150 `
  --smoothing-radius-factor 0.35 `
  --z-min 0.2 `
  --z-max 10.0 `
  --material R `
  --curve sinusoidal `
  --curve-amplitude 0.2 `
  --curve-period 40.0
```

The default slicing kernel is the in-repository `legacy` path-only kernel. It
can also be selected explicitly for reproducible comparisons:

```powershell
python -m kuka_slicer slice input.stl output.npz --slicing-kernel legacy
```

An experimental PySLM adapter can be selected from the CLI:

```powershell
python -m kuka_slicer slice input.stl output.npz --slicing-kernel pyslm
```

Install the optional dependency set first:

```powershell
python -m pip install ".[pyslm]"
```

The PySLM adapter is an independent native path: it uses PySLM for contour
generation and hatch generation, while keeping the same `ExternalSourceJob` and
NPZ handoff contract. Its native pattern set is `none`, `line`,
`aligned_rectilinear`, `rectilinear`, and cap-layer `zigzag`. `grid`,
`triangles`, `gyroid`, and `concentric` remain available in the standalone
`legacy` kernel and are rejected explicitly by the PySLM kernel.

The PySLM hatcher strategy is separate from the legacy `infill-pattern` option:
the former selects PySLM's scan organization (`Hatcher`, `StripeHatcher`, or
an island strategy), while the latter is only used by the legacy kernel. The
UI hides the legacy control when PySLM is selected. Stripe/island width,
overlap, and offset use scale-aware defaults; for the default resin process
(0.5 mm layer height and 2 mm line width), the defaults are 10 mm, 0.1 mm,
and 0.5 hatch-spacing units. The UI keeps these controls collapsed and lets
the user switch off automatic values before editing them.

PySLM-native controls are available through `--pyslm-*` CLI options and the UI:
Hatcher/StripeHatcher/IslandHatcher/BasicIslandHatcher, hatch angle and layer
angle increment, hatch distance, contour and spot offsets, volume offset,
contour counts, scan ordering, stripe/island dimensions, polygon repair, and
boundary simplification. Keep the standalone `legacy` kernel as the release
baseline until output parity is proven with fixtures.

Example native PySLM configuration:

```powershell
python -m kuka_slicer slice input.stl output.npz `
  --slicing-kernel pyslm `
  --infill-pattern rectilinear `
  --pyslm-hatcher stripe `
  --pyslm-hatch-angle 0 `
  --pyslm-layer-angle-increment 67 `
  --pyslm-hatch-distance 1.8 `
  --pyslm-hatch-sort alternate `
  --pyslm-stripe-width 10.0 `
  --pyslm-stripe-overlap 0.1
```

Available resin infill patterns use PrusaSlicer-style names for path-only
centerline generation:

| Pattern | Meaning |
| --- | --- |
| `rectilinear` | Alternating rectilinear lines by layer |
| `aligned_rectilinear` | Fixed-direction rectilinear lines |
| `line` | One-direction line fill |
| `grid` | Two-direction grid |
| `triangles` | Three-direction triangular grid |
| `gyroid` | Continuous gyroid-like curve fill |
| `concentric` | Concentric offset loops inside the perimeters |
| `zigzag` | Zig-zag rectilinear line segments without cross-path connectors |
| `none` | Internal option for perimeter-only output |

`--infill-density` is a resin fill percentage from `0` to `100`. It controls
the generated path spacing together with resin line width and
`--infill-overlap`.
For `triangles`, density is converted across the three lattice directions, so
lower densities produce larger triangles rather than three over-dense line sets.
Legacy triangle infill enables endpoint-based path optimization by default. It
first reverses or reorders open triangle paths, then merges consecutive paths
whose endpoints already coincide, and finally applies the existing smoothing
optimization. A small automatic endpoint tolerance of up to `0.1 mm` is also
applied after smoothing to coalesce numerical gaps below the safe threshold;
it never adds long connector segments. Disable it with
`--no-triangle-path-optimization` or the UI checkbox when the original path
order is required. This option is ignored by the PySLM kernel.
`gyroid` uses continuous clipped contour curves, which usually reduces resin
path start/stop count at high densities while keeping a more balanced direction
distribution than one-direction line fill.
Only path centerlines are exported. PrusaSlicer behaviors that depend on
extrusion amount, variable bead width, volumetric flow, or support/tree-specific
material accounting are not serialized into this NPZ format.

`--build-axis` selects the STL source axis used as the layer-height direction.
For a disk/cylinder whose round face is in the `X-Z` plane and thickness is
along `Y`, use `--build-axis y`.

In the web UI, geometric tolerance is an advanced numeric robustness value, not
a print compensation distance. Leaving it blank uses
`min(layer_height, line_width) * 0.001`, clamped to `0.00001..0.01 mm`.

## UI

```powershell
python -m kuka_slicer ui
```

Open:

```text
http://127.0.0.1:8765
```

The UI groups adjustable inputs into:

| Group | Parameters |
| --- | --- |
| Input files | STL upload, optional single-layer fiber JSON |
| Model and layers | layer height, build axis, optional `z_min`/`z_max`, geometric tolerance |
| Resin path kernel | slicing kernel (`Legacy` or `PySLM`), line width, perimeter count, infill pattern, density, overlap, triangle path optimization, smoothing, PySLM native settings |
| Raft | layer count, top gap, per-layer offsets, per-layer heights, per-layer densities, legacy per-layer infill patterns |
| Curved Z | flat/sinusoidal mode, amplitude, period |

Generate the documented two-layer resin/fiber template:

```powershell
python -m kuka_slicer make-template data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz
```

Run tests:

```powershell
python -m pytest
```
