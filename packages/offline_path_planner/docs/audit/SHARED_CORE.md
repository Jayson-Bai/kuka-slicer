# Shared core audit

## Current boundary

`path_processing_core` is the common implementation used by both the GCode and
external-NPZ producers. It declares only NumPy as a runtime dependency
(`src/my_project/path_processing_core/setup.py:13`) and contains no ROS import.
Its package description names the intended scope: shared command types,
calibration, sampling, and system-NPZ export
(`src/my_project/path_processing_core/setup.py:17`).

## Component map

| Component | Responsibility | Current consumers |
|---|---|---|
| `types.py` | Canonical intermediate command model | GCode parser/primeline/test generator, external converter, exporter |
| `bspline/` | Low-level B-spline basis, curves, surfaces, and parameter selection | Global spline planner |
| `bspline_approximation.py` | Groups moves and fits `GlobalCurveCommand` | GCode exporter; external module imports it but current external main path does not invoke fitting |
| `polynomial_interpolator.py` | Time-profiled sampling of polylines, fallbacks, and B-splines | Shared exporter and external curvature probes |
| `head_calibration.py` | Shared resin/fiber offset JSON and relative-offset calculation | Both offline producers and read-only upper-computer UI |
| `rsi_timing.py` | Per-row planned time and segment summary | Shared exporter |
| `npz_exporter.py` | Tool/event transformation, sampling, output chunking, vocabularies, sidecars | Both offline producers |

The inventory is anchored by `handoff/SOURCE_TREE.tsv:105` through
`handoff/SOURCE_TREE.tsv:117`.

## Canonical command model

The model contains:

- `Position` (`src/my_project/path_processing_core/path_processing_core/types.py:6`);
- `MoveCommand` (`src/my_project/path_processing_core/path_processing_core/types.py:16`);
- `ExtrudeWait`, `ResetECommand`, `ToolChangeCommand`, and `MCommand`
  (`src/my_project/path_processing_core/path_processing_core/types.py:34`,
  `src/my_project/path_processing_core/path_processing_core/types.py:46`,
  `src/my_project/path_processing_core/path_processing_core/types.py:57`,
  `src/my_project/path_processing_core/path_processing_core/types.py:67`);
- `CurveCommand` and `GlobalCurveCommand`
  (`src/my_project/path_processing_core/path_processing_core/types.py:79`,
  `src/my_project/path_processing_core/path_processing_core/types.py:98`).

`GlobalCurveCommand` retains `original_moves`
(`src/my_project/path_processing_core/path_processing_core/types.py:108`), which
allows interpolation/export fallback without discarding the source movement
sequence. The complete parsed-command union is defined at
`src/my_project/path_processing_core/path_processing_core/types.py:111`.

This model is already the correct producer-neutral boundary. Input-specific
objects such as `SourceJob` and `MaterialPath` remain in the external package,
while `MachineState` remains in the GCode package.

## Fitting and sampling

`GlobalSplinePlanner.fit_global_curve` builds a fitted curve and returns the
shared command type
(`src/my_project/path_processing_core/path_processing_core/bspline_approximation.py:180`,
`src/my_project/path_processing_core/path_processing_core/bspline_approximation.py:312`).
The GCode exporter actively invokes it
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1082`).

`sample_global_curve_iter` is the shared sampler
(`src/my_project/path_processing_core/path_processing_core/polynomial_interpolator.py:364`).
It handles `POLYLINE` explicitly
(`src/my_project/path_processing_core/path_processing_core/polynomial_interpolator.py:394`)
and also handles fallback/B-spline paths. Sampling duration is rounded to a
whole number of `dt` steps
(`src/my_project/path_processing_core/path_processing_core/polynomial_interpolator.py:422`).

The external converter uses this sampler to measure the curvature of its
candidate polyline
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:640`).
Its dormant spline helper also uses the core planner
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:690`).

## Calibration ownership

The shared calibration module:

- finds a workspace data root independent of the process working directory
  (`src/my_project/path_processing_core/path_processing_core/head_calibration.py:10`);
- provides zero-valued defaults
  (`src/my_project/path_processing_core/path_processing_core/head_calibration.py:29`);
- tolerates missing/invalid files by returning defaults
  (`src/my_project/path_processing_core/path_processing_core/head_calibration.py:55`);
- calculates the fiber relative Z offset as resin-Z compensation plus
  fiber-Z offset
  (`src/my_project/path_processing_core/path_processing_core/head_calibration.py:129`).

Because the same values affect both export and upper-computer validation, this
module is correctly shared today. A future standalone package must make the
calibration path an explicit job/config input rather than relying only on
workspace discovery.

## Timing ownership

`RsiTimingAccumulator` defines the producer-side meaning of planned time:
trajectory rows advance in `dt` increments after the first row, while events
do not advance motion time
(`src/my_project/path_processing_core/path_processing_core/rsi_timing.py:23`,
`src/my_project/path_processing_core/path_processing_core/rsi_timing.py:32`).
Its sidecar has format `rsi_print_timing`, version 1, sample period, total time,
row counts, and segments
(`src/my_project/path_processing_core/path_processing_core/rsi_timing.py:74`).

The timing format is versioned, but the enclosing NPZ schema is not.

## Export ownership

The active exporter owns:

- movement/event vocabulary IDs
  (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:151`);
- fixed dtypes and field order
  (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:230`);
- layer/preview/path metadata;
- tool offsets, resin-Z compensation, cut sequencing, and extrusion waits;
- part naming, split manifest, offset sidecar, and timing sidecar
  (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:290`,
  `src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1806`).

This is more than a serialization module: it also contains planning policy and
machine-event transformation. That coupling is operationally valid but should
be decomposed only after contract tests exist.

## Compatibility and duplicate-code findings

The GCode package keeps compatibility modules that re-export shared-core
modules. For example:

- `gcode_planner.npz_exporter`
  (`src/my_project/gcode_planner/gcode_planner/npz_exporter.py:1`);
- `gcode_planner.types`
  (`src/my_project/gcode_planner/gcode_planner/types.py:1`);
- `gcode_planner.bspline_approximation`
  (`src/my_project/gcode_planner/gcode_planner/bspline_approximation.py:1`);
- the `gcode_planner.bspline` namespace
  (`src/my_project/gcode_planner/gcode_planner/bspline/__init__.py:1`).

These are forwarding layers, not divergent implementations. Keep them through
at least one deprecation cycle if phase two changes import paths.

Within the shared exporter, `_npz_exporter` is a second private writer
(`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1946`).
**Inference:** repository-wide reference search found no caller. The active
nested `_Writer` supports all current fields and sidecars. The private function
is a cleanup candidate only after owner approval; it was not removed.

Within the external converter, `_fit_validated_spline` is implemented but not
called by the active conversion function
(`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:685`).
It must not be merged into the shared core or deleted until the intended
algorithm is clarified.

## Shared-core conclusion

Keep the canonical command types, fitting, sampling, calibration, timing, and
current exporter together for the baseline. In phase two, first add a versioned
producer/consumer contract test, then consider separating transformation policy
from NPZ serialization. No shared-core source was changed during this audit.
