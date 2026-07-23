# External NPZ pipeline

## Active entry points

The package exposes a CLI and a Qt UI
(`src/my_project/external_npz_preprocessor/setup.py:21`). Both converge on
`convert_external_npz`:

```text
source NPZ
  -> load_source_npz
  -> source_job_to_parsed_commands
  -> load shared head calibration
  -> path_processing_core.export_npz
  -> system NPZ + offset/timing metadata
```

The export runner implements this sequence directly
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/export_runner.py:71`
through
`src/my_project/external_npz_preprocessor/external_npz_preprocessor/export_runner.py:93`).

## Source contract

Layer/material keys must match `layer_NNNN_R` or `layer_NNNN_F`
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:14`).
Each value can be:

- a numeric three-dimensional array of paths with all-NaN padding rows; or
- a legacy object array of per-path arrays.

Each path must contain at least two rows and either XYZ or XYZABC columns.
XYZ paths receive caller-supplied default ABC values; XYZABC is retained
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:103`,
`src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:121`).
Partial-NaN rows are rejected
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:113`).

An optional scalar JSON `meta` value is parsed as an object
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:73`).
The loader sorts keys and layers, then preserves each array's path order
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:53`,
`src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:66`).

The tracked canonical template is indexed at `handoff/SOURCE_TREE.tsv:2`. Its
observed metadata is `format=external_layer_paths_v1`, unit `mm`, and point
columns `xyz`.

## Conversion semantics

`source_job_to_parsed_commands`:

1. Validates travel feed and initializes command/tool/E state
   (`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:39`).
2. Adds startup head events, determines source XY minimum and first material
   layers (`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:50`).
3. Processes resin paths before fiber paths in each layer
   (`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:71`).
4. Inserts one generated resin primeline before the first source path
   (`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:74`).
5. Applies source-to-start offsets without replacing source Z
   (`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:173`;
   the source-Z rule is stated at
   `src/my_project/external_npz_preprocessor/external_npz_preprocessor/source_npz.py:43`).
6. Emits tool changes, E resets, travel, prime/retract waits, print curves, and
   a synthetic cut after each fiber path
   (`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:122`,
   `src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:261`,
   `src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:281`).

The shared exporter enables extrusion waits and external absolute-E cut
behavior for this path
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/export_runner.py:87`,
`src/my_project/external_npz_preprocessor/external_npz_preprocessor/export_runner.py:92`).

## Current curve behavior: polyline, not fitted spline

The main converter calls `_validated_spline_or_polyline`
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:270`).
Despite the function name, its current live behavior is:

1. prepare smoothed positions;
2. sample the candidate polyline and measure maximum turn angle;
3. optionally apply RDP simplification plus corner fillets;
4. return a `GlobalCurveCommand` whose `cmd` is `POLYLINE`.

The decision path and returns are at
`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:585`
through
`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:618`.
`_make_polyline_curve` is the actual constructor
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:718`).

The file also defines `_fit_validated_spline`, which invokes
`GlobalSplinePlanner.fit_global_curve`
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:685`).
**Inference:** repository-wide reference search found no production call to
this function. Tests explicitly guard the polyline fast path around
`src/my_project/external_npz_preprocessor/test/test_converter.py:1095`.
It is therefore implemented but not integrated, and must not be described as
the current external-NPZ algorithm.

## Calibration and output

The runner loads the shared head-calibration model and passes fiber XYZ
compensation plus resin-Z compensation into the exporter
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/export_runner.py:49`).
The calibration model stores independent fiber XYZ and resin-Z values
(`src/my_project/path_processing_core/path_processing_core/head_calibration.py:29`)
and derives relative head offsets
(`src/my_project/path_processing_core/path_processing_core/head_calibration.py:118`).

The system-NPZ schema and consumer compatibility are documented in
`NPZ_OUTPUT_CONTRACT.md`.

## Golden reproduction

On 2026-07-23, Windows Python 3.13.1 / NumPy 2.2.0 ran:

```powershell
python -m external_npz_preprocessor.cli `
  --source data/external_npz_preprocessor/source_npz_templates/two_layer_rf_template.npz `
  --out <system-temp>\external-template-system.npz
```

Observed result:

| Measure | Value |
|---|---:|
| Exit status | 0 |
| Rows | 43,231 |
| Parts | 1 |
| Sequence | 0–43,230 |
| `total_layers` unique value | `[2]` |
| Final `planned_time_s` | 172.77200317382812 |
| Planned time monotonic | yes |
| All 25 arrays equal to frozen golden | yes |
| Offset JSON semantic equality | yes |
| Timing JSON semantic equality | yes |

The generated NPZ's ZIP-container bytes are not used as the cross-platform
functional assertion: NumPy archive metadata/compression can vary. The
field order, dtypes, shapes, and every array value were equal, and both JSON
objects were semantically equal. The frozen golden itself still matched all
three recorded SHA256 digests (`handoff/GOLDEN_SHA256SUMS:1`).

## Pipeline conclusions

- The external source contract is validated and reproducible.
- Source Z is authoritative trajectory geometry; process layer height does not
  replace it.
- The current path uses smoothed polylines, while dormant spline-validation
  code remains in the module.
- Both CLI and upper-computer UI call the same export runner
  (`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5122`).
- No conversion, smoothing, calibration, or schema logic was changed in phase
  one.
