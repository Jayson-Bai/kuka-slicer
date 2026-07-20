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

Resin infill uses a `10%` CLI/API run-overlap default, while the web UI defaults
the run overlap to `0%`. `--line-width` remains the nominal
process width written to NPZ metadata; it does not have to equal the bead width
measured after the nozzle presses the resin flat. The separate
`--planning-line-width` value is used only by the Prusa resin planner for path
spacing, overlap, and deposited-width safety checks. For example, a measured
flattened width of `2.2 mm` and `10%` overlap gives a requested centerline pitch
of `2.2 * (1 - 0.10) = 1.98 mm`, so the intended physical contact width is
`0.22 mm` between adjacent infill runs. The contour-to-infill seam has its own
independent overlap control, defaulting to `2%` based on the measured planning
width. Set `--infill-overlap 0` to remove intentional overlap between infill
runs without changing `--contour-infill-overlap`. Explicit
measured-width planning adds only a conservative
numerical safety margin (`16 * geometry_tolerance`; `0.008 mm` with the normal
UI tolerance), so this example is generated at `1.988 mm`. A final guard keeps
non-local infill runs, independent endcaps, and opposing sides of closed rings
on the conservative side of the semantic `1.98 mm` limit after smoothing; the
innermost-perimeter seam is checked against its separately configured contract.
Local continuous turns and point-only continuation splits are topology, not a
second parallel hatch, and are excluded from the percentage-overlap contract.

The web UI defaults the measured flattened/planning width to `2.3 mm`. The CLI
keeps backward compatibility: when
`--planning-line-width` is omitted, it falls back to the nominal
`--line-width`. Changing the planning width changes only centerline geometry
and its physical-width validation. This path-only NPZ format has no extrusion
multiplier, volumetric-flow, or per-segment flow field, and the nominal
`slicing.line_width` metadata remains unchanged. Existing perimeter centerlines
also continue to use the nominal line width because their printed result was
already validated; metadata records the infill-run spacing and the independently
configured contour-to-infill seam spacing separately.

Useful options:

```powershell
python -m kuka_slicer slice input.stl output.npz `
  --layer-height 0.5 `
  --line-width 2.0 `
  --planning-line-width 2.2 `
  --build-axis y `
  --infill-pattern zigzag_horizontal `
  --infill-density 100 `
  --infill-overlap 10 `
  --contour-infill-overlap 2 `
  --perimeter-count 2 `
  --smoothing-angle 120 `
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
The four fixed-direction Zigzag choices use the project bead-aware planner for
their continuous infill. Triangle and concentric fill remain available in the
Prusa kernel; the UI disables them when PySLM is selected because PySLM does
not provide equivalent project-owned implementations.

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
  --infill-pattern zigzag_horizontal `
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
| `zigzag_horizontal` | Horizontal scanlines joined by the existing boundary-safe one-stroke Zigzag planner |
| `zigzag_vertical` | Vertical scanlines joined by the existing boundary-safe one-stroke Zigzag planner |
| `zigzag_plus45` | Fixed `+45°` scanlines joined by the existing one-stroke Zigzag planner |
| `zigzag_minus45` | Fixed `-45°` scanlines joined by the existing one-stroke Zigzag planner |
| `triangles` | Three-axis noded lattice with the existing triangle path optimization |
| `concentric` | Fast full-fill offsets at one measured bead width, plus only residual strokes needed above 0.5 mm |


The four Zigzag choices use the existing one-stroke ordering, boundary-following
connectors, endpoint merging, and smoothing logic. Triangle fill keeps its
existing noding, optimization, and safe-connector logic. Concentric fill keeps
each offset ring independent and never creates a material-bearing ring-to-ring
seam. Local residual strokes are also standalone, so this mode is not
constrained to a one-stroke path.

`--infill-density` is a resin fill percentage from `0` to `100`. It controls
the generated path spacing together with the resin planning width and
`--infill-overlap`. For one-axis patterns, the density-adjusted spacing is
`centerline_pitch / density_fraction`. Triangle density is a shared
three-direction material-length budget, with spacing
`centerline_pitch * 3 / density_fraction`.

Selecting `concentric` with any positive density activates a dedicated fast
full-fill mode. Its direct offset pitch is exactly the measured flattened bead
width, independent of the run-density and run-overlap controls, which gives
nominally zero overlap between ordinary neighboring rings. Coverage is then
remeasured with the physical bead footprint. Standalone centerline supplements
are added only for residual components wider than `0.5 mm`; smaller remaining
gaps are deliberately ignored. Degenerate offset rings and supplemental paths
shorter than `0.5 mm` are also filtered from the exported path list so collapse
points do not appear as isolated resin dots.

At 100% density, the four single-axis Zigzag patterns and the Zigzag raft use
bead-aware solid-fill phasing.
Each disconnected printable island is centered at the configured pitch. Long
boundaries parallel to the hatch may anchor the phase only when doing so does
not reduce a neighboring centerline distance below that requested pitch. For a
`2.2 mm` measured width with `10%` overlap, the minimum requested pitch is
`1.98 mm`; geometry fitting may conservatively leave a slightly wider interval
instead of squeezing lines closer and increasing the physical overlap.
Corridors narrower than one pitch receive one centered stroke instead of
duplicated boundary strokes.

When a measured planning width is supplied (the web UI always supplies it),
the Prusa Zigzag solid-fill planner also disables wall-seam and residual gap
detours that cannot prove the same maximum-overlap bound. This may leave a small
local underfill beside difficult concave geometry; it is the conservative
fallback chosen to avoid a raised ridge or material pile-up. The dedicated
concentric full-fill mode does not use that strict maximum-overlap validator.

Concentric fill keeps its ordinary offset rings independent and closed. A
topology-collapse core may receive a short standalone residual stroke, but
rings are never joined through a material-bearing seam.

Coverage is evaluated with the configured planning bead width, not centerlines
or the nominal NPZ line width alone.
Short wall-seam doglegs and free-end tails are folded into an existing zigzag,
and residual narrow-neck pockets may replace a short original interval with a
triangle visit. These corrections keep the path count unchanged, reject
retrace/self-intersection, remain inside the physical part, and keep at least
the bounded clearance from the last perimeter. They are also limited by a
small added-length budget. Solid-fill turns are rounded before correction and
again after wall-seam/residual detours are inserted. In measured-width strict
mode, the second pass deliberately preserves micro arc samples: removing them
before another fillet fit can turn a straight hatch into an under-spaced long
diagonal chord. It fits the largest fillet covered by the same physical
centerline-safe region and samples it at no more than 10 degrees of heading
change per segment. A final indexed postcondition verifies fill-to-wall distance
and materially overlapping parallel returns; an unsafe smoothing result falls
back to the proven baseline with only the minimum required path splits. The
smoothing factor is interpreted as a physical centerline radius rather than a
tangent-cut length. Acute wall-seam hairpins
use analytical constant-radius/C1 returns; a return that cannot be rounded
inside the safe region is omitted instead of exporting a hidden sharp hook.
The ordinary initial radius remains coverage-limited, residual correction aims
below 40% of the physical bead width, and every added route remains subject to
novel-area/dose guards so the measured bead is neither treated as a zero-width
line nor stacked onto an already printed stroke.

For constant-section models, resin planning results are cached by the effective
fill direction and copied into repeated layers. The browser preview serializes
the role-aware path list once instead of repeating the same coordinates in
legacy contour/infill aliases; neither optimization changes the NPZ path
contract.

Triangle fill is noded at crossings and uses a graph trail cover that prints
every real lattice edge once. Virtual edges used to construct the Euler walk
are never printed, which minimizes starts without retracing and piling material
at an existing edge. Concentric full-fill uses measured-width direct offsets
and independent residual strokes instead of density-spaced joined rings.

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
Only path centerlines are exported. The measured planning width is recorded as
separate slicing metadata for traceability, but it does not add or alter any
flow column. PrusaSlicer behaviors that depend on extrusion amount, extrusion
multiplier, variable bead width, volumetric flow, or support/tree-specific
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
| Resin path kernel | slicing kernel (`Prusa` or `PySLM`), nominal line width, Prusa measured flattened/planning width, perimeter count, infill pattern, density, independent infill-run/contour-seam overlaps, triangle/zigzag path optimization, smoothing, PySLM native settings |
| Raft | fixed two-layer raft with one outward-offset input per layer; layer height, density, gap, and zigzag angles follow the fixed print schedule |
| Curved Z | flat/sinusoidal mode, amplitude, period |

Generate the documented two-layer resin/fiber template:

```powershell
python -m kuka_slicer make-template data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz
```

Run tests:

```powershell
python -m pytest
```
