# Parameter Model

## Principle

The source NPZ provides geometry only. Extrusion is derived from material process parameters during conversion.

## Current Defaults

- Resin line width: fixed `2.0 mm` and not user configurable
- Resin filament diameter: fixed `1.75 mm` for volume-to-filament-length conversion
- Resin layer height: `0.5 mm`
- Resin extrusion scale: `1.0`
- First-layer resin print speed: `10 mm/s`
- Later-layer resin print speed: `10 mm/s`
- Resin temperature: `250 C`
- Resin prime: `18 mm @ 15 mm/s`
- Resin retract: `15 mm @ 30 mm/s`
- Fiber layer height: `0.1 mm`
- Fiber extrusion scale: `1.0`
- First-layer fiber print speed: `10 mm/s`
- Later-layer fiber print speed: `10 mm/s`
- Fiber start acceleration time: `2.0 s`
- Fiber temperature: `250 C`
- Fiber prime: `12 mm @ 5 mm/s`
- Fiber retract: `10 mm @ 5 mm/s`
- Resin and fiber fans: always on by default
- First-layer shared travel speed: `10 mm/s`
- Later-layer travel speed: `10 mm/s`
- Global prime settle: `0.5 s`
- Default ABC: `0, 0, 0`

## Z Handling

The source NPZ must contain Z in every path point. The preprocessor treats source Z as trajectory geometry and never replaces it with UI layer-height values. This keeps curved-surface slicing intact.

UI layer-height fields are process references:

- Resin layer height participates in resin extrusion calculation.
- Fiber layer height is retained as a fiber process parameter, but it does not change trajectory Z or fiber `E` in the current speed-matched model.

Shared head offsets and tool-switch compensation are not duplicated here; they remain in `npz_exporter` through `tool_offset` and `resin_z_print_compensation_mm`.

## Resin

Resin equivalent extrusion per millimeter is derived as:

```text
resin_e_per_mm = 2.0 * layer_height_mm * extrusion_scale / (pi * (1.75 / 2)^2)
```

This matches the original G-code test-line allocation: deposited resin volume per path millimeter is converted to 1.75 mm filament length by dividing by the filament cross-section area. Resin line width and filament diameter are fixed in code, so the user-facing controls remain `layer_height_mm` and `extrusion_scale`.

## Fiber

Fiber does not expose diameter, printed line width, or a direct `E/mm` field. The active model forces fiber feed to match TCP movement speed by default:

```text
fiber_e_per_mm = fiber_extrusion_scale
```

With the default `fiber_extrusion_scale = 1.0`, a 10 mm path produces 10 mm of fiber feed. Fiber layer height remains in the process parameters as a process reference, but it does not change source Z or `E` in the current speed-matched fiber model.

External NPZ fiber paths can override only the start acceleration time used by the seven-order time parameterization:

```text
fiber_start_accel_s = 2.0
```

This parameter is attached to fiber `GlobalCurveCommand` objects as curve-level metadata. The global seven-order default remains unchanged for resin, travel, cut lift, and all curves that do not explicitly carry this metadata.

## Shared Parameters

- `resin.first_layer_feed_mm_s` and `fiber.first_layer_feed_mm_s`: print speeds for the independently detected first material-bearing layer. The generated primeline uses the first-layer resin speed.
- `first_layer_travel_feed_mm_s`: shared travel speed when the destination belongs to a material-specific first layer.
- `travel_feed_mm_s`: feed speed when the destination belongs to a later material layer; it remains the exporter default feed for cut lift and tool-offset safety moves. External-NPZ conversion requires both travel values to be finite and greater than zero.
- `prime_settle_s`: global stationary wait after every non-zero prime; default `0.5 s`, and `0` disables only the settle.
- `default_a/default_b/default_c`: pose values appended to Nx3 source paths.
- `dt`: sample period forwarded to `npz_exporter`.
- `cut_lift_mm`: Z lift distance after a fiber `CUT`; default `20.0`.
- `cut_wait_s`: total wait time measured from the exported `cut` event trigger; default `15.0`.

## Prime and Retract

Prime/retract defaults come from the existing test-mode UI values. Resin keeps its existing per-path prime/retract behavior.

Fiber uses the UI values only at fiber-layer boundaries:

- Before the whole job's first fiber: reset, UI retract, reset, UI prime, optional settle, reset, print.
- Before the first fiber of every later fiber-bearing layer: reset, UI prime, optional settle, reset, print.
- Middle fibers in the same layer: no UI prime or UI retract.
- After the last fiber's CUT in every fiber-bearing layer: reset, UI retract, then the existing path reset.

A one-path fiber layer is both first and last. Layers without fiber emit no fiber UI action. A zero prime length emits neither prime nor settle.

`prime_settle_s` is persisted in `print_params.json` and exposed through the CLI, standalone UI, and formal-print UI. Legacy JSON without the field uses `0.5 s`; `0` disables only the settle, while negative, `NaN`, and infinite values are rejected. With `dt=0.004 s`, the default `0.5 s` settle becomes exactly 125 stationary rows with unchanged XYZ and E.

Every printable path, including the generated primeline and final path, ends at an explicit E boundary:

- Resin: `PRINT -> retract -> external_npz_path_reset -> external_npz_reset_anchor -> optional travel(E=0)`.
- Non-final fiber in a layer: `PRINT -> pre-CUT reset -> CUT -> lift/feed -> 3 s hold -> reset -> fixed retract -> 3 s hold -> reset -> remaining high hold -> external_npz_path_reset -> external_npz_reset_anchor -> optional travel(E=0)`.
- Layer-final fiber: the same isolated CUT sequence, then `reset -> UI retract -> external_npz_path_reset -> external_npz_reset_anchor -> optional travel(E=0)`.

For external-NPZ fiber CUT, the blocking pre-CUT reset establishes E=0 before the nonblocking `cut` event. `cut_lift_mm=L` then produces two reset-isolated absolute-E intervals: lift/feed `0→+L`, followed by fixed retract `0→-L`. Each interval holds its terminal E for 3 seconds before reset, so neither phase can be truncated by the next reset or inherit path/UI E.

The two 3-second holds and both motions consume the existing `cut_wait_s` budget measured from the `cut` event; XYZ stays at the lifted pose until that total window ends. If the configured window is shorter than the complete safety sequence, completion takes precedence. CUT does not consume `fiber_prime_length_mm=P` or `fiber_retract_length_mm=R`; UI prime remains `0→+P`, UI retract remains `0→-R`, and their configured speeds remain independent.

The existing tool-change `G92 E0` / `ResetECommand` remains unchanged and is additive to the per-path and phase resets. The reset event is exported at the old E value; only the exact internal `external_npz_reset_anchor` marker makes the following one-`dt` row start at `E=0`. Converter-side `current_e` is then zero, travel remains at `E=0`, and ordinary GCode behavior is unchanged.

## Persistent Parameter JSON

The UI has a `保存打印参数json文件` button. It writes the current print parameters under the project data directory, matching the existing test/formal-mode convention:

```text
data/external_npz_preprocessor/print_params.json
```

The directory is created automatically before saving. When the UI starts, it automatically reads this file if it exists and applies the latest saved print parameters. Resin line width is not stored as a parameter; legacy JSON files containing `bead_width_mm` are accepted but that value is ignored. Legacy JSON without first-layer speed fields inherits the saved resin, fiber, and travel speeds so existing output timing remains unchanged.

## Shared Head Offsets

External NPZ conversion uses the same head-offset data source as test mode and formal printing:

```text
data/head_calibration_offsets/head_offsets.json
```

The converter reads this through `path_processing_core.head_calibration.load_head_calibration()`. During export it mirrors the formal-print path:

```text
tool_offset = (fiber_x_print_compensation_mm,
               fiber_y_print_compensation_mm,
               fiber_z_print_compensation_mm)
resin_z_print_compensation_mm = resin.z_print_compensation_mm
```

These values are passed directly to `path_processing_core.npz_exporter.export_npz()`, so tool switching, resin-Z compensation, cut lift/wait expansion, and short-segment polyline sampling stay centralized in the shared exporter logic. The exporter starts from resin tool `2`; before a tool-change event with non-zero head offset, it first lifts `20 mm`, then performs the XYZ offset travel, and only then emits the tool-change event. For an external-NPZ fiber `CUT`, the exporter first establishes the E=0 boundary, emits the nonblocking `cut` event, immediately lifts Z by `cut_lift_mm` while feeding the same absolute length, and runs the isolated high-pose retract sequence described above before any downstream travel.
