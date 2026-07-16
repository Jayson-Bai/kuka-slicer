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

The PySLM adapter keeps the same `ExternalSourceJob` and NPZ handoff contract.
Its native hatch patterns are `none`, `line`, `aligned_rectilinear`, and
`rectilinear`. For `zigzag` and `isotropic`, PySLM supplies slicing and
contours while the project bead-aware planner supplies the continuous infill.
`grid`, `triangles`, `gyroid`, and `concentric` remain available in the Prusa
kernel and are rejected explicitly by the PySLM kernel.

The shared infill-pattern selector remains visible for both kernels; the UI
disables patterns unsupported by PySLM. The PySLM hatcher strategy selects its
native scan organization (`Hatcher`, `StripeHatcher`, or an island strategy).
Stripe/island width, overlap, and offset use scale-aware defaults; for the
default resin process (0.5 mm layer height and 2 mm line width), the defaults
are 10 mm, 0.1 mm, and 0.5 hatch-spacing units. The UI keeps these controls
collapsed and lets the user switch off automatic values before editing them.

PySLM-native controls are available through `--pyslm-*` CLI options and the UI:
Hatcher/StripeHatcher/IslandHatcher/BasicIslandHatcher, hatch angle and layer
angle increment, hatch distance, contour and spot offsets, volume offset,
contour counts, scan ordering, stripe/island dimensions, polygon repair, and
boundary simplification (`absolute` or boundary-scaled `bound`). Project-owned
`zigzag`/`isotropic` reject contour-geometry overrides so their generated
infill cannot drift inside the required bead-aware contour clearance. Native
patterns may use those overrides on middle layers, while the fixed full-density
top and bottom caps reset contour and hatch-distance overrides to the safe
print schedule. Keep the standalone Prusa kernel as the release baseline until
output parity is proven with fixtures.

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

At 100% density, single-axis legacy patterns (including the four directions
used by `isotropic` and the zigzag raft) use bead-aware solid-fill phasing.
Each disconnected printable island is centered at the configured pitch. Long
boundaries parallel to the hatch may anchor the phase only when every band fits
inside a small, symmetric tolerance around that pitch. For a 2 mm line with
10% overlap, the target is 1.8 mm and the permitted local range is 1.7..1.9 mm;
this bounds both visible gaps and local overlap instead of silently expanding
spacing to a full 2 mm or squeezing lines together without limit. Corridors
narrower than one pitch receive one centered stroke instead of duplicated
boundary strokes.

Coverage is evaluated with the physical round 2 mm bead, not centerlines alone.
Short wall-seam doglegs and free-end tails are folded into an existing zigzag,
and residual narrow-neck pockets may replace a short original interval with a
triangle visit. These corrections keep the path count unchanged, reject
retrace/self-intersection, remain inside the physical part, and keep at least
the bounded clearance from the last perimeter. They are also limited by a
small added-length budget. Solid-fill turns are rounded before correction and
again after wall-seam/residual detours are inserted. The second pass removes
sub-0.01 mm numerical fragments, fits the largest fillet covered by the same
physical centerline-safe region, and samples it at no more than 10 degrees of
heading change per segment. The smoothing factor is interpreted as a physical
centerline radius rather than a tangent-cut length. Acute wall-seam hairpins
use analytical constant-radius/C1 returns; a return that cannot be rounded
inside the safe region is omitted instead of exporting a hidden sharp hook.
The ordinary initial radius remains coverage-limited, residual correction aims
below 40% of the physical bead width, and every added route remains subject to
novel-area/dose guards so a 2 mm bead is neither treated as a zero-width line
nor stacked onto an already printed stroke.

For constant-section models, resin planning results are cached by the effective
fill direction and copied into repeated layers. The browser preview serializes
the role-aware path list once instead of repeating the same coordinates in
legacy contour/infill aliases; neither optimization changes the NPZ path
contract.

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

Legacy triangle infill enables endpoint-based path optimization by default. It
first reverses or reorders open triangle paths, then merges consecutive paths
whose endpoints already coincide, and finally applies the existing smoothing
optimization. Post-planning merges are limited to the configured numerical
geometry tolerance; material-bearing joins remain subject to the bead-aware
safe-connector checks. Disable it with
`--no-triangle-path-optimization` or the UI checkbox when the original path
order is required. This option is ignored by the PySLM kernel.
Legacy zigzag infill, including forced part cap layers and explicit raft
zigzag layers, uses the same ordering, reversal, endpoint merge, and final
smoothing cleanup by default. It can be disabled with
`--no-zigzag-path-optimization` or its UI checkbox.
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
| Resin path kernel | slicing kernel (`Prusa` or `PySLM`), line width, perimeter count, infill pattern, density, overlap, triangle/zigzag path optimization, smoothing, PySLM native settings |
| Raft | fixed two-layer raft with editable outward offsets; layer height, density, gap, and zigzag angles follow the fixed print schedule |
| Curved Z | flat/sinusoidal mode, amplitude, period |

Generate the documented two-layer resin/fiber template:

```powershell
python -m kuka_slicer make-template data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz
```

Run tests:

```powershell
python -m pytest
```
