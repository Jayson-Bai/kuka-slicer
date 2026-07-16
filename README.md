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

Resin infill uses a default overlap of `10%`. The planner accounts for the
physical bead width before applying that overlap. With a `2.0 mm` resin line
width, the overlap is `0.2 mm` and the default centerline pitch is therefore
`2.0 - 0.2 = 1.8 mm`. The first infill centerline, including a
boundary-following continuity link, stays one pitch inward from the innermost
(last) perimeter centerline. From the physical inner edge of the last `2.0 mm`
perimeter bead, the safe infill centerline moves inward by half a bead and then
back toward the wall by only the requested overlap. This avoids ignoring half
of the printed bead or counting overlap twice, either of which can create
wall-parallel overfill and material pile-up. Set `--infill-overlap 0` to remove
intentional wall overlap.

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

Available resin infill patterns use PrusaSlicer-style names for path-only
centerline generation:

| Pattern | Meaning |
| --- | --- |
| `rectilinear` | Alternating single-axis scanlines, joined along the safe infill boundary where possible |
| `aligned_rectilinear` | Fixed-direction scanlines with the same boundary-following joins |
| `line` | One-direction scanlines with the same boundary-following joins |
| `grid` | Two-axis noded lattice covered by a minimum number of edge-disjoint trails |
| `triangles` | Three-axis noded lattice covered by a minimum number of edge-disjoint trails |
| `gyroid` | Gyroid-like curves with boundary-safe trail connections and a calibrated `2.35` wavelength factor |
| `concentric` | Density-spaced offset loops with safe connections between adjacent rings |
| `zigzag` | Alternating scanlines with adjacent, boundary-following continuity links |
| `none` | Internal option for perimeter-only output |

`--infill-density` is a resin fill percentage from `0` to `100`. It controls
the generated path spacing together with resin line width and
`--infill-overlap`. For one-axis patterns and `concentric`, the density-adjusted
spacing is `centerline_pitch / density_fraction`; consequently, concentric ring
spacing increases as density decreases. For multi-axis patterns, density is a
total material-length budget shared across all directions, not a separate full
budget for each direction. Their spacing is
`centerline_pitch * axis_count / density_fraction`, so the default `1.8 mm`
pitch at 100% density gives `3.6 mm` per grid direction and `5.4 mm` per
triangle direction. This prevents two- and three-axis patterns from depositing
roughly two or three times the requested material.

`grid` and `triangles` are noded at crossings and use a graph trail cover that
prints every real lattice edge once. Virtual edges used to construct the Euler
walk are never printed, which minimizes starts without retracing and piling
material at an existing edge. `gyroid` derives its wavelength as `2.35` times
the density-adjusted line spacing and connects clipped curves only through
boundary-safe links. `concentric` uses the requested density for ring spacing
and tries to continue directly from one adjacent ring to the next without
reprinting either ring.

Continuity links are printed centerlines, not zero-material travel moves. A link
is accepted only when it stays inside the bead-aware safe infill corridor,
avoids non-incident paths and existing links, and preserves at least the base
centerline pitch where clearance is required. If no safe link exists, the
planner keeps separate paths instead of forcing a one-stroke result. Boundary
safety and avoidance of material pile-up take priority over eliminating every
start/stop.

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
| Resin path kernel | line width, perimeter count, PrusaSlicer-style infill pattern, density, overlap, smoothing |
| Raft | layer count, top gap, per-layer offsets, per-layer heights, per-layer densities |
| Curved Z | flat/sinusoidal mode, amplitude, period |

Generate the documented two-layer resin/fiber template:

```powershell
python -m kuka_slicer make-template data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz
```

Run tests:

```powershell
python -m pytest
```
