# Source inventory

## Audit scope and identity

This inventory was produced on branch `audit/offline-planner-baseline` from
handoff commit `444c420d26b5cf8071b681006f015f18ac5dd60f`. The checked-out tag is
`offline-planner-handoff-v1`; the handoff document identifies the upstream
source commit as `eb9091bcf405eaca2dddb07f2998bb3f25c12601`
(`EXTERNAL_CODEX_HANDOFF.md:19`, `EXTERNAL_CODEX_HANDOFF.md:83`).

The repository has 134 tracked files:

- 124 source-handoff files enumerated one by one in
  `handoff/SOURCE_TREE.tsv:1` through `handoff/SOURCE_TREE.tsv:124`.
- 10 handoff-control files: `.gitignore`, `EXTERNAL_CODEX_HANDOFF.md`, the five
  text indexes under `handoff/`, and the three frozen files under
  `handoff/golden/`.

Read-only verification on 2026-07-23 produced:

- `SOURCE_SHA256SUMS`: 124/124 matched. The source list is the immutable
  checksum authority (`handoff/SOURCE_SHA256SUMS:1`).
- `GOLDEN_SHA256SUMS`: 3/3 matched
  (`handoff/GOLDEN_SHA256SUMS:1` through
  `handoff/GOLDEN_SHA256SUMS:3`).
- No existing source, sample, test, handoff file, or golden artifact was
  modified.

## Complete coverage matrix

Every tracked file is covered by exactly one row below. The source-tree line
range is also the per-file audit index: each line contains mode, Git object ID,
and path.

| Files | Count | Role | Audit disposition |
|---|---:|---|---|
| `handoff/SOURCE_TREE.tsv:1`–`handoff/SOURCE_TREE.tsv:2` | 2 | External-NPZ parameters and canonical source template | Active configuration/test input; read and structurally inspected |
| `handoff/SOURCE_TREE.tsv:3`–`handoff/SOURCE_TREE.tsv:19` | 17 | Eleven GCode inputs and six external source NPZ inputs | Representative/fixture data; every GCode was parsed for command and layer-marker inventory, every NPZ was opened and its arrays inspected |
| `handoff/SOURCE_TREE.tsv:20`–`handoff/SOURCE_TREE.tsv:30` | 11 | Historical plans and design specifications | Design history, not executable truth; all documents read, code wins on disagreement |
| `handoff/SOURCE_TREE.tsv:31`–`handoff/SOURCE_TREE.tsv:35` | 5 | Estimation, plotting, and GCode-splitting utilities | Developer tools; active but outside installed packages |
| `handoff/SOURCE_TREE.tsv:36`–`handoff/SOURCE_TREE.tsv:39` | 4 | C++ NPZ loader and queue manager excerpts | Read-only downstream contract consumer; not a buildable package in this handoff |
| `handoff/SOURCE_TREE.tsv:40`–`handoff/SOURCE_TREE.tsv:45` | 6 | External-NPZ package documentation | Current intent/reference; checked against implementation |
| `handoff/SOURCE_TREE.tsv:46`–`handoff/SOURCE_TREE.tsv:55` | 10 | External-NPZ Python implementation | Active producer path: CLI/UI, source loader, converter, parameters, export runner |
| `handoff/SOURCE_TREE.tsv:56`–`handoff/SOURCE_TREE.tsv:59` | 4 | External-NPZ ROS/Python package metadata | Active packaging metadata |
| `handoff/SOURCE_TREE.tsv:60`–`handoff/SOURCE_TREE.tsv:67` | 8 | External-NPZ tests | Active functional contract tests |
| `handoff/SOURCE_TREE.tsv:68`–`handoff/SOURCE_TREE.tsv:68` | 1 | GCode package README | User/developer reference |
| `handoff/SOURCE_TREE.tsv:69`–`handoff/SOURCE_TREE.tsv:85` | 17 | GCode planner implementation and compatibility wrappers | Active parser/CLI/preview/test-generator plus intentional shared-core re-export layer |
| `handoff/SOURCE_TREE.tsv:86`–`handoff/SOURCE_TREE.tsv:89` | 4 | GCode ROS/Python package metadata | Active packaging metadata |
| `handoff/SOURCE_TREE.tsv:90`–`handoff/SOURCE_TREE.tsv:99` | 10 | GCode and repository-wide tests | Seven functional tests plus three ament static-test launchers |
| `handoff/SOURCE_TREE.tsv:100`–`handoff/SOURCE_TREE.tsv:101` | 2 | ROS message excerpts | Read-only downstream NPZ/event contract |
| `handoff/SOURCE_TREE.tsv:102`–`handoff/SOURCE_TREE.tsv:102` | 1 | Upper-computer RQt panel | Read-only integration reference; combines offline export and realtime launch responsibilities |
| `handoff/SOURCE_TREE.tsv:103`–`handoff/SOURCE_TREE.tsv:103` | 1 | Shared-core README | Package boundary reference |
| `handoff/SOURCE_TREE.tsv:104`–`handoff/SOURCE_TREE.tsv:120` | 17 | Shared core implementation, B-spline library, and package metadata | Active shared command model, fitting, sampling, calibration, timing, and NPZ export |
| `handoff/SOURCE_TREE.tsv:121`–`handoff/SOURCE_TREE.tsv:123` | 3 | Shared-core tests | Active compatibility and timing tests |
| `handoff/SOURCE_TREE.tsv:124`–`handoff/SOURCE_TREE.tsv:124` | 1 | Plot utility test | Active script-level test |
| `.gitignore` | 1 | Generated-file exclusions | Handoff control; leaves build/install/log/caches out of commits (`.gitignore:1`) |
| `EXTERNAL_CODEX_HANDOFF.md` | 1 | Governing handoff instructions | Immutable phase gate; audit-only changes are required (`EXTERNAL_CODEX_HANDOFF.md:182`) |
| `handoff/HANDOFF_PATHS.txt` | 1 | Source inclusion list | Immutable handoff control; offline packages begin at `handoff/HANDOFF_PATHS.txt:1` |
| `handoff/SOURCE_ARCHIVE_SHA256.txt` | 1 | Source archive digest | Immutable handoff control (`handoff/SOURCE_ARCHIVE_SHA256.txt:1`) |
| `handoff/SOURCE_SHA256SUMS` | 1 | Per-source-file digests | Immutable integrity authority |
| `handoff/SOURCE_TREE.tsv` | 1 | Per-source-file Git tree index | Immutable coverage authority |
| `handoff/GOLDEN_SHA256SUMS` | 1 | Golden artifact digests | Immutable golden authority |
| `handoff/golden/external-template-system.npz` | 1 | Frozen system-NPZ golden | Immutable binary contract baseline |
| `handoff/golden/external-template-system.offset.json` | 1 | Frozen offset sidecar | Immutable metadata baseline |
| `handoff/golden/external-template-system.timing.json` | 1 | Frozen timing sidecar | Immutable timing baseline |

## Executable component inventory

| Component | Current entry/use | Assessment |
|---|---|---|
| GCode CLI | `gcode_planner_npz` maps to `gcode_planner.cli:main` (`src/my_project/gcode_planner/setup.py:30`) | Active offline producer |
| GCode ROS node | `gcode_parser` maps to `gcode_planner.gcode_parser:main` (`src/my_project/gcode_planner/setup.py:32`) | Active ROS wrapper around the same parser |
| External NPZ CLI | `external_npz_preprocessor_cli` (`src/my_project/external_npz_preprocessor/setup.py:21`) | Active offline producer |
| External NPZ UI | `external_npz_preprocessor_ui` (`src/my_project/external_npz_preprocessor/setup.py:24`) | Active optional Qt frontend, but its Qt dependency is undeclared |
| Shared core | `path_processing_core` package (`src/my_project/path_processing_core/setup.py:3`) | Active shared implementation used by both producers |
| Python preview | `gcode_planner.path_preview` | Active system-NPZ consumer; reads preview and path metadata (`src/my_project/gcode_planner/gcode_planner/path_preview.py:339`) |
| C++ loader/queue | `control_center` excerpts | Read-only downstream consumer; reads trajectory/event fields (`src/my_project/control_center/src/npz_loader.cpp:394`, `src/my_project/control_center/src/queue_manager.cpp:39`) |
| Upper-computer panel | `my_project_ui/ui_panel.py` | Read-only migration source; directly calls both offline producers (`src/my_project/my_project_ui/my_project_ui/ui_panel.py:5120`, `src/my_project/my_project_ui/my_project_ui/ui_panel.py:5136`) |

## Classification findings

### Active code

The active production chain is:

1. GCode parser/primeline or external-NPZ loader/converter.
2. Shared command types.
3. Shared fitting/sampling/export logic.
4. System NPZ plus timing/offset metadata.

The entry-point declarations and direct imports support this classification
(`src/my_project/gcode_planner/gcode_planner/cli.py:10`,
`src/my_project/external_npz_preprocessor/external_npz_preprocessor/export_runner.py:7`).

### Compatibility code

The modules under `gcode_planner/bspline/` and the top-level
`gcode_planner/{types,head_calibration,npz_exporter,polynomial_interpolator,bspline_approximation}.py`
re-export `path_processing_core`. For example,
`src/my_project/gcode_planner/gcode_planner/npz_exporter.py:3` aliases the
shared module rather than duplicating implementation. The compatibility test
explicitly guards this cleanup boundary
(`src/my_project/gcode_planner/test/test_core_import_cleanup.py:1`).
They are not safe deletion candidates until downstream imports are surveyed.

### Implemented but not on the current main path

- External conversion defines `_fit_validated_spline` and constructs a
  `GlobalSplinePlanner` (`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:685`),
  but `_validated_spline_or_polyline` returns a `POLYLINE` in all live branches
  (`src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:566`,
  `src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:607`,
  `src/my_project/external_npz_preprocessor/external_npz_preprocessor/converter.py:618`).
  **Inference:** repository-wide reference search found no production caller of
  `_fit_validated_spline`.
- `path_processing_core.npz_exporter` contains a second private writer named
  `_npz_exporter` (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:1946`).
  **Inference:** repository-wide reference search found only its definition;
  the active streaming `_Writer` persists all current fields
  (`src/my_project/path_processing_core/path_processing_core/npz_exporter.py:230`).

### Reference and historical material

The C++ loader, queue manager, message definitions, and RQt panel were included
as downstream-consumer evidence by the handoff list
(`handoff/HANDOFF_PATHS.txt:23` through `handoff/HANDOFF_PATHS.txt:29`).
They are not authorized refactor targets in phase one.

The `docs/superpowers/` files record prior proposals and implementation plans.
They are useful intent evidence but are not assumed to describe the current
tree when code differs.

## Phase-one conclusion

The source package is internally coherent enough to reproduce both export
pipelines. No source deletion, move, formatting, dependency change, or schema
change is justified during this audit. Ambiguities and proposed ownership
changes are documented separately in `MISSING_OR_AMBIGUOUS.md` and
`SEPARATION_PROPOSAL.md`.
