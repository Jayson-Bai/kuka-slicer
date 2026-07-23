# GCode pipeline

## Active entry points

The package exposes:

- `gcode_planner_npz = gcode_planner.cli:main`, the offline exporter
  (`src/my_project/gcode_planner/setup.py:33`).
- `gcode_parser = gcode_planner.gcode_parser:main`, the ROS node wrapper
  (`src/my_project/gcode_planner/setup.py:32`).

The offline CLI chooses an explicit GCode file or the first file in the input
directory, computes a default output path, and creates the output directory
(`src/my_project/gcode_planner/gcode_planner/cli.py:74`,
`src/my_project/gcode_planner/gcode_planner/cli.py:82`,
`src/my_project/gcode_planner/gcode_planner/cli.py:89`).

## End-to-end flow

```text
GCode text
  -> load_gcode_lines
  -> parse_gcode_lines / MachineState
  -> insert_resin_primeline
  -> path_processing_core.export_npz
       -> fit eligible move runs
       -> sample at dt
       -> apply head/tool and resin-Z compensation
       -> emit trajectory and event rows
       -> write NPZ part(s), offset JSON, timing JSON, optional manifest/previews
```

The CLI performs exactly this sequence at
`src/my_project/gcode_planner/gcode_planner/cli.py:91` through
`src/my_project/gcode_planner/gcode_planner/cli.py:117`.

## Parsing behavior

`MachineState` tracks XYZABC, E, feed rate, active tool, absolute/relative
coordinate and extrusion modes, layer, and subtype
(`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:22`).

The parser has handlers for:

- `G90`/`G91` coordinate mode and `M82`/`M83` extrusion mode
  (`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:133`).
- `G92` state reset, represented as `ResetECommand` when E is supplied
  (`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:164`).
- `G0`/`G1` movement (`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:140`).
- Tool changes and generic M commands
  (`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:302`,
  `src/my_project/gcode_planner/gcode_planner/gcode_parser.py:325`).
- Layer and print-type comments, including layer increment and explicit
  `LAYER:` parsing (`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:349`).

Travel is selected for `G0` or a movement without positive extrusion; print
uses the current subtype
(`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:274`).

## Primeline insertion

Before export, `insert_resin_primeline` finds the first print command and inserts
the resin primeline plus configured pre/post extrusion waits
(`src/my_project/gcode_planner/gcode_planner/primeline.py:21`,
`src/my_project/gcode_planner/gcode_planner/primeline.py:109`).
Subsequent E values are shifted by the added extrusion until an explicit G92
reset (`src/my_project/gcode_planner/gcode_planner/primeline.py:27`).

## Fitting and sampling

The active exporter creates a `GlobalSplinePlanner`
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:136`).
Eligible runs are fitted through `fit_global_curve`
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1082`);
the planner returns a `GlobalCurveCommand` with original moves retained
(`src/my_project/path_processing_core/path_processing_core/bspline_approximation.py:310`).

`sample_global_curve_iter` supports:

- explicit `POLYLINE` sampling
  (`src/my_project/path_processing_core/path_processing_core/polynomial_interpolator.py:394`);
- original-move/linear fallback paths;
- generic B-spline evaluation;
- a time profile quantized to `dt`
  (`src/my_project/path_processing_core/path_processing_core/polynomial_interpolator.py:364`,
  `src/my_project/path_processing_core/path_processing_core/polynomial_interpolator.py:422`).

## Event and timing behavior

The exporter maps:

- `M104`/`M109` to resin/fiber heat;
- `M106`/`M107` to fan on/off;
- synthetic `CUT` to cut.

The mapping is implemented at
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:2056`
through `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:2074`.
Unknown M commands are ignored by the NPZ event conversion.

Event rows hold the most recent trajectory pose, set `event_flag=1`, use
`move_type=EVENT`, and set `trigger_seq` to the event row sequence
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1517`,
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1523`,
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1528`).

The RSI timing accumulator advances only trajectory rows; event rows retain the
current trajectory time
(`src/my_project/path_processing_core/path_processing_core/rsi_timing.py:1`,
`src/my_project/path_processing_core/path_processing_core/rsi_timing.py:23`).

## Output topology

The default flat export writes `_partNNNN.npz` chunks, then renames a single
part to the requested `.npz` path
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:265`,
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:290`).
With `--split-by-layer-type`, writers are created per layer/subtype/occurrence
and a manifest records `base_path` and sequence bounds
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:312`,
`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:326`).

Every export attempts to write:

- `<base>.offset.json`
  (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1809`);
- a timing sidecar
  (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1818`);
- a manifest only for split output
  (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1805`).

## Representative real export

On 2026-07-23 the pipeline was executed under ROS 2 Humble in WSL:

```bash
python3 -m gcode_planner.cli \
  --gcode data/input_gcode/split_test0210/test0210_layer_0001.gcode \
  --out /tmp/offline-planner-gcode-audit-20260723-1/gcode-system.npz
```

Observed result:

| Measure | Value |
|---|---:|
| Exit status | 0 |
| Export rows | 247,616 |
| Parts | 1 |
| Sequence | 0–247,615 |
| Field count | 25 |
| `total_layers` unique value | `[2]` |
| Final `planned_time_s` | 990.4199829101562 |
| Planned time monotonic | yes |
| Export stage time reported by CLI | 2.996 s |

The input is a tracked representative sample
(`handoff/SOURCE_TREE.tsv:16`). The output was written to `/tmp`, not to the
repository.

## Pipeline conclusions

- The representative GCode path is operational.
- GCode currently uses the shared spline planner for eligible movement runs.
- The CLI is not standalone plain Python because importing its parser also
  imports `rclpy` (`src/my_project/gcode_planner/gcode_planner/gcode_parser.py:8`).
- Event rows and trajectory rows share one sequence domain, while event rows do
  not advance planned trajectory time.
- No pipeline or schema change was made during this audit.
